"""Independent selected-model audit of a fixed exhaustive H1 control search.

The exhaustive grid is never repeated here. Actual research data remain on
research-host. Selected models are checked with native PanelOLS and an independent
rowwise two-way clustered sandwich, including the native intercept convention.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


CURRENT = 'net_sentiment'
LAG = 'net_sentiment_lag1'
OUTCOME = 'quantity_change_pct'
CONTROLS = ['opening_weight', 'closing_log_NAV', 'closing_HHI',
    'log_news_count_interval', 'log_interval_days', 'bond_indicator',
    'Imoex', 'imoex_ret', 'Ruonia', 'USD_RUB', 'RVI', 'Flow_billion_RUB',
    'unique_funds', 'dummy_mar2014', 'dummy_feb2022', 'dummy_sep2018']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def row_hash(frame):
    return hashlib.sha256('\n'.join(frame.row_id.astype(str)).encode()).hexdigest()


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (np.integer, np.bool_)): return value.item()
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value, (Path, pd.Timestamp)): return str(value)
    return value


def selected_terms(frame, required, controls):
    """Declared-order numerical rank check matching the established FE design."""
    names = list(required) + list(controls)
    values = frame[names].astype(float)
    within = values-values.groupby(frame.fund_id.astype(str), sort=False).transform('mean')
    tolerance = 16*np.finfo(float).eps*max(len(frame), len(names))
    keep, basis = [], []
    for term in names:
        column = within[term].to_numpy(float)
        norm = np.linalg.norm(column)
        absorbed = frame.groupby('fund_id', observed=True)[term].nunique().max() <= 1
        if absorbed or norm == 0:
            if term in required: raise ValueError('Required regressor is fund-absorbed: '+term)
            continue
        vector = column/norm
        if basis:
            q = np.column_stack(basis)
            residual = vector-q@np.linalg.lstsq(q, vector, rcond=None)[0]
        else:
            residual = vector
        if np.linalg.norm(residual) <= tolerance:
            if term in required: raise ValueError('Required regressor is collinear: '+term)
            continue
        keep.append(term)
        basis.append(vector)
    return keep


def independent_rowwise(frame, terms):
    """OLS/CGM from actual rows, not the search runner's sufficient statistics.

    A centered intercept basis is used before transforming the constant back
    to the PanelOLS normalization: const=ybar-xbar@beta. This also gives the
    full native covariance, rather than approximating const variance from the
    slope covariance alone.
    """
    fund = frame.fund_id.astype(str)
    company = frame.company_id.astype(str)
    x = frame[terms].astype(float)
    y = frame[OUTCOME].astype(float)
    xbar = x.mean().to_numpy(float)
    ybar = float(y.mean())
    xw = (x-x.groupby(fund, sort=False).transform('mean')).to_numpy(float)
    yw = (y-y.groupby(fund, sort=False).transform('mean')).to_numpy(float)
    scales = np.linalg.norm(xw, axis=0)
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError('A retained regressor lacks finite within variation')
    z = xw/scales
    beta, _, rank, singular = np.linalg.lstsq(z, yw, rcond=None)
    if rank != len(terms): raise ValueError('Retained within design is rank deficient')
    residual = yw-z@beta
    n = len(frame)
    # U=[1/sqrt(N), Xw_scaled]. Unlike a rowwise refit of a demeaned
    # intercept, the constant is retained and its cluster score is explicit.
    u = np.column_stack([np.full(n, 1/np.sqrt(n)), z])
    scores = u*residual[:, None]
    bread = np.linalg.inv(u.T@u)

    def meat(keys):
        codes, labels = pd.factorize(keys, sort=False)
        groups = len(labels)
        if groups < 2: raise ValueError('Fewer than two clusters')
        grouped = np.zeros((groups, scores.shape[1]))
        np.add.at(grouped, codes, scores)
        return grouped.T@grouped*(groups/(groups-1))*((n-1)/n), groups

    fund_meat, funds = meat(fund)
    company_meat, companies = meat(company)
    intersection_meat, pairs = meat(pd.MultiIndex.from_arrays([fund, company]))
    df = n-funds-len(terms)
    if df <= 0: raise ValueError('Nonpositive residual df')
    covariance = bread@(fund_meat+company_meat-intersection_meat)@bread*n/df
    transform = np.zeros((len(terms)+1, len(terms)+1))
    transform[0, 0] = 1/np.sqrt(n)
    transform[0, 1:] = -xbar/scales
    transform[np.arange(1, len(terms)+1), np.arange(1, len(terms)+1)] = 1/scales
    covariance = transform@covariance@transform.T
    covariance = (covariance+covariance.T)/2
    slopes = beta/scales
    coefficients = np.r_[ybar-xbar@slopes, slopes]
    tss, ssr = float(yw@yw), float(residual@residual)
    return dict(terms=['const']+list(terms), beta=coefficients,
        covariance=covariance, nobs=n, funds=funds, companies=companies,
        pair_clusters=pairs, df_resid=df, within_R2=1-ssr/tss,
        condition_number=float(singular.max()/singular.min()))


def native_fit(frame, terms):
    from linearmodels.panel import PanelOLS
    index = pd.MultiIndex.from_arrays([frame.fund_id.astype(str), pd.to_datetime(frame.end_date)], names=['fund_id', 'report_date'])
    x = frame[terms].astype(float).copy()
    x.index = index
    x.insert(0, 'const', 1.)
    y = pd.Series(frame[OUTCOME].to_numpy(float), index=index, name=OUTCOME)
    clusters = pd.DataFrame(dict(
        fund_cluster=pd.Categorical(frame.fund_id.astype(str)).codes,
        company_cluster=pd.Categorical(frame.company_id.astype(str)).codes), index=index)
    return PanelOLS(y, x, entity_effects=True, time_effects=False,
        other_effects=None, singletons=True, drop_absorbed=False,
        check_rank=True).fit(cov_type='clustered', clusters=clusters,
        cluster_entity=False, cluster_time=False, group_debias=True,
        debiased=True, auto_df=False, count_effects=True)


def audit(args):
    if os.name != 'nt': raise RuntimeError('Research-data QA runs on research-host only')
    from scipy import stats
    planpath = Path(args.plan)
    plan = json.loads(planpath.read_text(encoding='utf-8'))
    root = Path(args.root or plan['output_root'])
    output = Path(args.output or root/'qa_independent.json')
    if output.exists(): raise FileExistsError('Refusing to overwrite an existing QA report')
    if 'databaseresources' in [part.lower() for part in output.parts]: raise ValueError('Protected output path')
    summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    manifest = json.loads((root/'input_manifest.json').read_text(encoding='utf-8'))
    table = pd.read_csv(root/'all_models.csv', low_memory=False)
    shortlist = pd.read_csv(root/'shortlist.csv', low_memory=False)
    cached_native = pd.read_csv(root/'native_coefficients.csv', low_memory=False)
    checks, warnings, replicas = [], [], {}

    def check(name, passed, **evidence):
        checks.append(dict(name=name, status='PASS' if bool(passed) else 'FAIL', **clean(evidence)))

    def same(left, right, atol=1e-8, rtol=None):
        return bool(np.allclose(np.asarray(left, float), np.asarray(right, float),
            rtol=args.rtol if rtol is None else rtol, atol=atol, equal_nan=True))

    def covariance_same(left, right):
        left, right = np.asarray(left, float), np.asarray(right, float)
        scales = np.sqrt(np.maximum(np.abs(np.diag(left)), np.abs(np.diag(right))))
        denominator = np.outer(np.maximum(scales, 1e-150), np.maximum(scales, 1e-150))
        return same(left/denominator, right/denominator, 1e-7)

    inputpath = Path(plan['input_panel'])
    before_source = sha(inputpath)
    before_used = sha(root/'used_sample.parquet')
    original = pd.read_parquet(inputpath).sort_values(['fund_id', 'end_date', 'security_id', 'interval_id'], kind='stable').reset_index(drop=True)
    frame = pd.read_parquet(root/'used_sample.parquet')
    check('frozen_plan_and_manifest', summary['plan'] == plan and manifest == summary['manifest']
        and manifest['plan_sha256'] == sha(planpath) and manifest['sha256'] == before_source)
    check('literal_used_source_values_and_all_columns', frame.equals(original),
        mismatched_columns=[c for c in original if c not in frame or not original[c].equals(frame[c])])
    n, funds, companies = len(frame), frame.fund_id.nunique(), frame.company_id.nunique()
    check('unchanged_common_equity_rows_and_keys', (n, funds, companies) == (60903, 91, 136)
        and (n, funds, companies) == (plan['expected_rows'], plan['expected_funds'], plan['expected_companies'])
        and row_hash(frame) == plan['expected_row_hash'] == manifest['row_hash']
        and not frame.row_id.duplicated().any() and frame.fund_focus.eq('equity').all())
    check('declared_same_design_covariance_and_no_adjustments', plan['controls'] == CONTROLS
        and plan['targets'] == [CURRENT, LAG] and plan['outcome'] == OUTCOME
        and plan['fixed_effects'] == ['fund_id'] and plan['clusters'] == ['fund_id', 'company_id']
        and plan['time_FE'] is False and plan['cluster_time'] is False
        and plan['covariance_options'] == dict(debiased=True, auto_df=False, count_effects=True, group_debias=True, PSD_repair=False)
        and not any(any(word in c.lower() for word in ['holm', 'adjusted', 'corrected']) for c in table.columns))
    check('complete_outcome_and_full_potential_design', np.isfinite(frame[[OUTCOME, CURRENT, LAG]+CONTROLS].astype(float)).all().all())
    all_terms = selected_terms(frame, [CURRENT, LAG], CONTROLS)
    check('full_rank_declared_design_all_subsets_identified', all_terms == [CURRENT, LAG]+CONTROLS)
    counts = {c: frame[c].nunique() for c in ['fund_id', 'company_id']}
    counts['intersection'] = frame[['fund_id', 'company_id']].drop_duplicates().shape[0]
    check('real_intersection_cluster_count', summary['cluster_counts'] == dict(fund=counts['fund_id'], company=counts['company_id'], intersection=counts['intersection']))
    check('exactly_131072_unique_model_keys', len(table) == 131072
        and set(table.variant) == {'CURRENT_ONLY', 'CURRENT_PLUS_LAG1'}
        and not table.duplicated(['variant', 'control_mask']).any())
    check('shortlist_native_model_keys_unique', not shortlist.duplicated(['variant', 'control_mask']).any()
        and not cached_native.duplicated(['model', 'term']).any())
    summaries = {record['variant']: record for record in summary['counts']}
    selection = {}
    for variant in ['CURRENT_ONLY', 'CURRENT_PLUS_LAG1']:
        sub = table.loc[table.variant.eq(variant)].sort_values('control_mask').reset_index(drop=True)
        distributed = variant == 'CURRENT_PLUS_LAG1'
        ntargets = 2 if distributed else 1
        masks = sub.control_mask.to_numpy(int)
        expected_controls = ['|'.join(c for j, c in enumerate(CONTROLS) if mask >> j & 1) for mask in masks]
        expected_excluded = ['|'.join(c for j, c in enumerate(CONTROLS) if not (mask >> j & 1)) for mask in masks]
        check('complete_mask_enumeration:'+variant, np.array_equal(masks, np.arange(65536)))
        check('exact_control_bits_and_complement:'+variant, sub.controls_included.fillna('').tolist() == expected_controls
            and sub.controls_excluded.fillna('').tolist() == expected_excluded
            and np.array_equal(sub.n_controls, np.array([int(mask).bit_count() for mask in masks])))
        expected_df = n-funds-sub.n_controls-ntargets
        check('common_N_clusters_and_native_df:'+variant, sub.nobs.eq(n).all() and sub.funds.eq(funds).all()
            and sub.companies.eq(companies).all() and sub.df_resid.eq(expected_df).all())
        check('finite_within_R2_bounds:'+variant, np.isfinite(sub.rsquared_within).all()
            and sub.rsquared_within.between(-1e-10, 1+1e-10).all())
        critical = stats.t.ppf(.975, expected_df)
        for target in ['current']+(['lag', 'sum'] if distributed else []):
            variance = sub[target+'_variance'].to_numpy(float)
            beta = sub[target+'_beta'].to_numpy(float)
            valid = np.isfinite(variance) & (variance > 0) & np.isfinite(beta)
            se = np.full(len(sub), np.nan)
            se[valid] = np.sqrt(variance[valid])
            p = 2*stats.t.sf(np.abs(beta/se), expected_df)
            check('unrepaired_variance_native_t_and_CI:'+variant+':'+target,
                same(sub[target+'_se'], se) and same(sub[target+'_p'], p, 1e-10)
                and same(sub[target+'_ci_low'], beta-critical*se)
                and same(sub[target+'_ci_high'], beta+critical*se))
            if target != 'sum':
                expected_status = np.where(valid, 'FINITE', 'UNDEFINED_VARIANCE')
                check('explicit_undefined_inference_status:'+variant+':'+target, np.array_equal(sub[target+'_status'], expected_status))
            record = summaries[variant][target]
            pos = beta > 0
            sig = p < .05
            expected_counts = dict(finite=int(valid.sum()), undefined=int((~valid).sum()), positive=int(pos.sum()),
                p_lt_05=int(sig.sum()), positive_p_lt_05=int((pos & sig).sum()), negative_p_lt_05=int((~pos & sig).sum()))
            check('all_grid_significance_counts:'+variant+':'+target, all(record[key] == value for key, value in expected_counts.items())
                and same(record['min_p'], np.nanmin(p), 1e-10) and same(record['median_p'], np.nanmedian(p), 1e-10), counts=expected_counts)
            if (~valid).any(): warnings.append(dict(variant=variant, target=target, kind='UNDEFINED_VARIANCE_MODELS', count=int((~valid).sum())))
        if distributed:
            cross = sub.current_lag_covariance.to_numpy(float)
            vc, vl = sub.current_variance.to_numpy(float), sub.lag_variance.to_numpy(float)
            bc, bl = sub.current_beta.to_numpy(float), sub.lag_beta.to_numpy(float)
            determinant = vc*vl-cross*cross
            valid = (vc > 0) & (vl > 0) & (determinant > 0)
            stat = np.full(len(sub), np.nan)
            stat[valid] = (bc[valid]**2*vl[valid]-2*bc[valid]*bl[valid]*cross[valid]+bl[valid]**2*vc[valid])/determinant[valid]
            check('sum_uses_full_cross_covariance:'+variant, same(sub.sum_beta, bc+bl)
                and same(sub.sum_variance, vc+vl+2*cross))
            check('joint_chi_square_requires_positive_target_covariance:'+variant, same(sub.joint_stat, stat)
                and same(sub.joint_p, stats.chi2.sf(stat, 2), 1e-10))
        else:
            check('current_only_has_no_lag_or_sum_inference', sub[['lag_beta', 'lag_p', 'sum_beta', 'sum_p', 'joint_p']].isna().all().all())
        check('summary_model_sample_counts:'+variant, summaries[variant]['models'] == 65536
            and (summaries[variant]['N'], summaries[variant]['funds'], summaries[variant]['companies']) == (n, funds, companies))
        chosen = {0: ['NO_CONTROLS'], 65535: ['ALL_CONTROLS']}
        for target in ['current']+(['lag', 'sum'] if distributed else []):
            valid = sub.loc[np.isfinite(sub[target+'_p'])]
            for positive in [False, True]:
                candidates = valid.loc[valid[target+'_beta'].gt(0)] if positive else valid
                if not candidates.empty:
                    mask = int(candidates.sort_values([target+'_p', 'control_mask']).iloc[0].control_mask)
                    chosen.setdefault(mask, []).append('MIN_P_'+target.upper()+('_POSITIVE' if positive else '_ANY_SIGN'))
        rng = np.random.default_rng(20261103)
        fixed_random = [21845, 43690]+[int(x) for x in rng.choice(np.arange(1,65535), 3, replace=False)]
        for mask in fixed_random: chosen.setdefault(mask, []).append('FIXED_RANDOM_QA_MASK')
        for record in shortlist.loc[shortlist.variant.eq(variant)].to_dict('records'):
            chosen.setdefault(int(record['control_mask']), []).append('RUNNER_SHORTLIST')
        selection[variant] = chosen

    prior_root = inputpath.parent.parent
    prior_summary = json.loads((prior_root/'summary.json').read_text(encoding='utf-8'))
    prior_coef = pd.read_csv(prior_root/'coefficients.csv')
    for variant, old_suffix in [('CURRENT_ONLY', 'CURRENT_COMMON'), ('CURRENT_PLUS_LAG1', 'CURRENT_PLUS_LAG1')]:
        oldname = 'FOCUS_EQUITY_ROWS_60_INTERVALS_24__'+old_suffix
        old = prior_coef.loc[prior_coef.model.eq(oldname)].set_index('term')
        fast = table.loc[table.variant.eq(variant) & table.control_mask.eq(65535)].iloc[0]
        oldrecord = next(record for record in prior_summary['models'] if record['model'] == oldname)
        check('reproduce_prior_full_control_baseline:'+variant, oldrecord['source_row_hash'] == row_hash(frame)
            and oldrecord['nobs'] == n and same(fast.current_beta, old.loc[CURRENT, 'beta'])
            and same(fast.current_se, old.loc[CURRENT, 'se_native']) and same(fast.current_p, old.loc[CURRENT, 'p_native'], 1e-10)
            and same(fast.rsquared_within, oldrecord['rsquared_within'], 1e-10))

    replica_root = root/'qa_selected_replicas'
    replica_root.mkdir(exist_ok=True)
    native_rows = []
    for variant, chosen in selection.items():
        required = [CURRENT, LAG] if variant == 'CURRENT_PLUS_LAG1' else [CURRENT]
        for mask, reasons in sorted(chosen.items()):
            model = variant+f'__{mask:05d}'
            controls = [c for j, c in enumerate(CONTROLS) if mask >> j & 1]
            terms = selected_terms(frame, required, controls)
            fast = table.loc[table.variant.eq(variant) & table.control_mask.eq(mask)].iloc[0]
            result = native_fit(frame, terms)
            independent = independent_rowwise(frame, terms)
            check('selected_full_native_beta_and_covariance:'+model,
                same(independent['beta'], result.params.to_numpy(), 1e-8)
                and covariance_same(independent['covariance'], result.cov.to_numpy())
                and independent['df_resid'] == result.df_resid == fast.df_resid
                and same(independent['within_R2'], result.rsquared_within, 1e-10)
                and same(independent['within_R2'], fast.rsquared_within, 1e-10))
            ci = result.conf_int()
            cached = cached_native.loc[cached_native.model.eq(model)].set_index('term')
            if not cached.empty:
                check('runner_native_all_coefficients_SE_CI:'+model, set(cached.index) == set(result.params.index)
                    and same(cached.reindex(result.params.index).beta, result.params)
                    and same(cached.reindex(result.params.index).se, result.std_errors)
                    and same(cached.reindex(result.params.index).p, result.pvalues, 1e-10)
                    and same(cached.reindex(result.params.index).ci_low, ci.iloc[:,0])
                    and same(cached.reindex(result.params.index).ci_high, ci.iloc[:,1]))
            for target, prefix in zip(required, ['current', 'lag']):
                check('fast_target_matches_independent_and_native:'+model+':'+target,
                    same(fast[prefix+'_beta'], result.params[target])
                    and same(fast[prefix+'_se'], result.std_errors[target])
                    and same(fast[prefix+'_variance'], result.cov.loc[target,target])
                    and same(fast[prefix+'_p'], result.pvalues[target], 1e-10)
                    and same(fast[prefix+'_ci_low'], ci.loc[target].iloc[0])
                    and same(fast[prefix+'_ci_high'], ci.loc[target].iloc[1]))
            if len(required) == 2:
                covariance = result.cov.loc[required,required].to_numpy(float)
                beta = result.params.loc[required].to_numpy(float)
                check('selected_full_target_covariance_and_sum:'+model,
                    same(fast.current_lag_covariance, covariance[0,1])
                    and same(fast.sum_variance, covariance.sum()) and same(fast.sum_beta, beta.sum()))
            (replica_root/(model+'.summary.txt')).write_text(str(result.summary), encoding='utf-8')
            for term in result.params.index:
                j = independent['terms'].index(term)
                native_rows.append(dict(model=model, term=term, beta_native=result.params[term],
                    se_native=result.std_errors[term], p_native=result.pvalues[term],
                    beta_independent=independent['beta'][j], variance_independent=independent['covariance'][j,j]))
            replicas[model] = dict(selection_reasons=reasons, nobs=int(result.nobs), funds=funds, companies=companies,
                native_df_resid=int(result.df_resid), retained_terms=terms, required_targets=required,
                native_within_R2=float(result.rsquared_within), condition_number=independent['condition_number'],
                native_target_coefficients={term:dict(beta=float(result.params[term]), se=float(result.std_errors[term]), p=float(result.pvalues[term])) for term in required})
            print(json.dumps(dict(status='QA_REPLICA_COMPLETE', model=model)), flush=True)
    pd.DataFrame(native_rows).to_csv(replica_root/'all_coefficients.csv', index=False, encoding='utf-8-sig')
    check('input_and_used_samples_preserved', sha(inputpath) == before_source and sha(root/'used_sample.parquet') == before_used)
    failed = [record for record in checks if record['status'] == 'FAIL']
    result = dict(status='FAIL' if failed else ('PASS_WITH_WARNINGS' if warnings else 'PASS'),
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), plan_sha256=sha(planpath),
        source_sha256=before_source, used_sample_sha256=before_used, source_row_hash=row_hash(frame),
        exhaustive_grid_models=131072, exhaustive_grid_repeated=False,
        selected_numerical_replica_count=len(replicas), selected_numerical_replicas=replicas,
        checks=checks, pass_count=len(checks)-len(failed), fail_count=len(failed), warnings=warnings,
        interpretation='Exploratory exhaustive specification search; unadjusted p-values and selected minima are not confirmatory evidence after search.',
        numerical_scope='All grid masks/metadata/native arithmetic/counts checked; selected models only refitted in native PanelOLS and independently via rowwise CGM, including full intercept/slopes covariance.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean(result), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], pass_count=result['pass_count'], fail_count=result['fail_count'], numerical_replicas=len(replicas))))


def fixture():
    rng = np.random.default_rng(5571)
    n, funds = 1800, 30
    frame = pd.DataFrame(dict(fund_id=np.repeat(np.arange(funds), n//funds),
        company_id=np.tile(np.arange(15), n//15), end_date=pd.date_range('2020-01-01', periods=n),
        net_sentiment=rng.normal(size=n), net_sentiment_lag1=rng.normal(size=n),
        control=rng.normal(size=n)))
    frame[OUTCOME] = 2+.4*frame[CURRENT]-.2*frame[LAG]+.5*frame.control+frame.fund_id*.05+rng.normal(size=n)
    terms = [CURRENT, LAG, 'control']
    independent = independent_rowwise(frame, terms)
    native = native_fit(frame, terms)
    assert np.allclose(independent['beta'], native.params, rtol=1e-10, atol=1e-10)
    assert np.allclose(independent['covariance'], native.cov, rtol=1e-9, atol=1e-10)
    assert independent['df_resid'] == native.df_resid == n-funds-len(terms)
    assert np.isclose(independent['within_R2'], native.rsquared_within, rtol=1e-10, atol=1e-10)
    print(json.dumps(dict(status='PASS', scope='synthetic_only', full_native_intercept_and_slope_covariance=True, checks=4)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan')
    parser.add_argument('--root')
    parser.add_argument('--output')
    parser.add_argument('--rtol', type=float, default=2e-6)
    parser.add_argument('--fixture-only', action='store_true')
    args = parser.parse_args()
    if args.fixture_only: fixture()
    else: audit(args)
