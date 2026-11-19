"""Independent read-only QA of conditional entries, fixed P99 and H1 fits.

Research data are read on research-host only. No PanelOLS or regression runner is
imported. The only research write is the separately named QA JSON report.
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


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def row_hash(frame):
    return hashlib.sha256('\n'.join(frame.row_id.astype(str)).encode()).hexdigest()


def truth(values):
    return values.astype(str).str.lower().str.strip().isin(['true', '1', 'yes'])


def finite(frame, columns):
    return np.isfinite(frame[columns].apply(pd.to_numeric, errors='coerce').to_numpy(float)).all(axis=1)


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (np.integer, np.bool_)): return value.item()
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value, (Path, pd.Timestamp)): return str(value)
    return value


def within_ols(frame, terms):
    """Fund-demeaned OLS and independent two-way CGM sandwich."""
    groups = frame.fund_id.astype(str)
    x = frame[terms].astype(float)
    y = frame.quantity_change_pct.astype(float)
    xw = (x-x.groupby(groups, sort=False).transform('mean')).to_numpy(float)
    yw = (y-y.groupby(groups, sort=False).transform('mean')).to_numpy(float)
    scale = np.linalg.norm(xw, axis=0)
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Retained term lacks finite within variation')
    z = xw/scale
    beta_z, _, rank, singular = np.linalg.lstsq(z, yw, rcond=None)
    if rank != len(terms): raise ValueError('Within design is rank deficient')
    residual = yw-z@beta_z
    bread = np.linalg.inv(z.T@z)
    scores = z*residual[:, None]

    def meat(keys):
        codes, labels = pd.factorize(keys, sort=False)
        n, g = len(codes), len(labels)
        if g < 2: raise ValueError('Fewer than two clusters')
        sums = np.zeros((g, scores.shape[1]))
        np.add.at(sums, codes, scores)
        return sums.T@sums * (g/(g-1)) * ((n-1)/n), g

    fund_meat, funds = meat(groups)
    companies = frame.company_id.astype(str)
    company_meat, company_count = meat(companies)
    joint_meat, joint_count = meat(pd.MultiIndex.from_arrays([groups, companies]))
    n, k = z.shape
    df = n-funds-k
    if df <= 0: raise ValueError('Nonpositive residual degrees of freedom')
    covariance = bread@(fund_meat+company_meat-joint_meat)@bread * n/df
    covariance = covariance/scale[:, None]/scale[None, :]
    ssr, tss = float(residual@residual), float(yw@yw)
    return dict(beta=beta_z/scale, covariance=(covariance+covariance.T)/2,
        nobs=n, funds=funds, companies=company_count, joint_clusters=joint_count,
        df_resid=df, rank=rank, condition_number=float(singular.max()/singular.min()),
        within_SSR=ssr, within_TSS=tss, within_R2=1-ssr/tss if tss else np.nan)


def audit(args):
    if os.name != 'nt': raise RuntimeError('Actual research-data QA must run on research-host')
    from scipy import stats
    import pyarrow.parquet as pq

    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    root = Path(args.root or plan['output_root'])
    output = Path(args.output or root/'qa_independent.json')
    if output.exists(): raise FileExistsError('QA reports cannot overwrite an existing version')
    if 'databaseresources' in [p.lower() for p in output.parts]:
        raise ValueError('QA output is within protected inputs')
    summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    threshold = json.loads((root/'threshold.json').read_text(encoding='utf-8'))
    coefficients = pd.read_csv(root/'coefficients.csv')
    checks, warnings, details = [], [], {}

    def check(name, passed, **evidence):
        checks.append(dict(name=name, status='PASS' if bool(passed) else 'FAIL', **clean(evidence)))

    def close(left, right, atol=1e-7):
        return bool(np.isclose(float(left), float(right), rtol=args.rtol, atol=atol))

    check('frozen_plan_sha256', sha(plan_path) == summary['plan_sha256'])
    check('frozen_specification_preserved', summary.get('specification') == plan)
    check('fund_FE_no_time_FE_no_time_clusters', plan['fund_FE'] is True and plan['time_FE'] is False
        and plan['cluster_time'] is False and plan['clusters'] == ['fund_id', 'company_id'])
    for key, expected in plan['expected_source_sha256'].items():
        actual = sha(plan[key])
        check('unchanged_frozen_source:'+key, actual == expected, sha256=actual)

    base_controls, macro_controls = plan['baseline_controls'], plan['macro_controls']
    terms_all = ['net_sentiment']+base_controls+macro_controls
    keys = ['row_id','fund_id','security_id','interval_id','company_id','start_date','end_date',
        'quantity_open','quantity_close','quantity_change_pct','net_sentiment','n_relevant',
        'eligible_primary','target_security_candidate','issuer_identity_conflict','asset_class_conflict',
        'unknown_opening_absence','unknown_closing_absence','opening_position_observed',
        'opening_full_portfolio','known_opening_absence_zero','known_closing_absence_zero']+base_controls
    available = pq.read_schema(plan['all_candidate_panel']).names
    missing = sorted(set(keys)-set(available))
    if missing: raise ValueError('Frozen candidate lacks QA fields: '+repr(missing))
    candidate = pd.read_parquet(plan['all_candidate_panel'], columns=list(dict.fromkeys(keys)))
    old = pd.read_parquet(plan['old_primary_sample'])
    classification = pd.read_parquet(plan['classification_source'], columns=['interval_id','fund_id','index_status','exclude_index_fund','fund_focus'])
    macro = pd.read_parquet(plan['macro_controls_source'], columns=['interval_id']+macro_controls)
    check('candidate_unique_row_ids', candidate.row_id.notna().all() and not candidate.row_id.duplicated().any())
    check('classification_unique_keys', not classification.duplicated(['interval_id','fund_id']).any())
    check('macro_unique_intervals', not macro.interval_id.duplicated().any())
    d = candidate.merge(classification, on=['interval_id','fund_id'], how='left', validate='many_to_one', sort=False)
    d = d.merge(macro, on='interval_id', how='left', validate='many_to_one', sort=False)
    check('candidate_join_preserves_rows_and_order', d.row_id.equals(candidate.row_id))
    q0, q1 = d.quantity_open, d.quantity_close
    nonindex = d.index_status.notna() & d.exclude_index_fund.notna() & d.index_status.ne('index') & ~truth(d.exclude_index_fund)
    known = ~truth(d.unknown_opening_absence) & ~truth(d.unknown_closing_absence)
    identity = truth(d.target_security_candidate) & d.company_id.notna() & ~truth(d.issuer_identity_conflict) & ~truth(d.asset_class_conflict)
    observed_news = truth(d.eligible_primary) & np.isfinite(pd.to_numeric(d.net_sentiment, errors='coerce'))
    common = nonindex & known & identity & observed_news & finite(d, terms_all)
    positive = common & finite(d, ['quantity_open','quantity_close','quantity_change_pct']) & q0.gt(0) & q1.ge(0)
    certified_zero = truth(d.opening_position_observed) | (truth(d.known_opening_absence_zero) & truth(d.opening_full_portfolio))
    raw_entry = finite(d, ['quantity_open','quantity_close']) & q0.eq(0) & q1.gt(0) & certified_zero
    entry = common & raw_entry
    augmented = positive | entry
    continuing = positive & q1.gt(0)
    check('old_sample_rows', len(old) == int(plan['expected_old_primary_rows']))
    check('independent_positive_base_universe_exact_old_ids', set(d.loc[positive,'row_id']) == set(old.row_id))
    old_continuing = old.loc[old.quantity_open.gt(0) & old.quantity_close.gt(0)]
    cutoff = float(old_continuing.quantity_change_pct.quantile(plan['upper_quantile'], interpolation=plan['quantile_interpolation']))
    check('P99_value_independent', close(cutoff, threshold['cutoff_percent'], 1e-10), calculated=cutoff, saved=threshold['cutoff_percent'])
    check('P99_reference_rows', len(old_continuing) == threshold['cutoff_population_rows'])
    check('P99_reference_row_hash', row_hash(old_continuing) == threshold['cutoff_population_row_hash'])
    check('P99_algorithm_fixed', threshold['upper_quantile'] == plan['upper_quantile'] and threshold['quantile_interpolation'] == plan['quantile_interpolation'] and threshold['threshold_recomputed_per_model_or_group'] is False)
    trim = continuing & d.quantity_change_pct.gt(cutoff)
    retained = augmented & ~trim
    exit_mask = positive & q1.eq(0)
    check('trim_only_positive_continuing_positions', not (trim & (entry | exit_mask)).any())
    check('all_eligible_conditional_entries_retained', (retained | ~entry).all(), entries=int(entry.sum()))
    check('all_eligible_complete_exits_retained', (retained | ~exit_mask).all(), exits=int(exit_mask.sum()))
    check('cutoff_equality_retained', not (trim & d.quantity_change_pct.eq(cutoff)).any())
    start, end = pd.to_datetime(d.start_date), pd.to_datetime(d.end_date)
    days = (end-start).dt.days
    month_difference = (end.dt.year-start.dt.year)*12+end.dt.month-start.dt.month
    monthly = month_difference.eq(1) & days.between(20,40,inclusive='both')
    decisions = pd.read_parquet(root/'row_decisions.parquet')
    check('decision_universe_ids_exact', set(decisions.row_id) == set(candidate.row_id) and not decisions.row_id.duplicated().any())
    indexed = decisions.set_index('row_id').reindex(d.row_id)
    for field, mask in [('positive_base_eligible',positive),('entry_zero_to_positive',raw_entry),('augmented_eligible',augmented),('trimmed_upper_p99',trim),('retained_primary',retained),('monthly_interval',monthly)]:
        check('independent_decision_mask:'+field, np.array_equal(truth(indexed[field]).to_numpy(), np.asarray(mask,bool)))
    original_y = pd.to_numeric(indexed.original_quantity_change_pct,errors='coerce').to_numpy(float)
    check('decision_original_Y_preserved', np.allclose(original_y,d.quantity_change_pct.to_numpy(float),rtol=0,atol=0,equal_nan=True))
    for field in ['quantity_open','quantity_close']+terms_all:
        check('decision_source_values_preserved:'+field, np.allclose(pd.to_numeric(indexed[field],errors='coerce'),pd.to_numeric(d[field],errors='coerce'),rtol=0,atol=0,equal_nan=True))

    expected_ids = {
        'REFERENCE_OLD': set(old.row_id),
        'OLD_BASE_AFTER_P99': set(d.loc[positive & ~trim,'row_id']),
        'AUGMENTED_UNTRIMMED': set(d.loc[augmented,'row_id']),
        'AUGMENTED_P99': set(d.loc[retained,'row_id']),
        'MONTHLY_AUGMENTED_P99': set(d.loc[retained & monthly,'row_id']),
    }
    for focus in plan['fund_focus_groups']:
        expected_ids['FOCUS_'+focus.upper()+'_P99'] = set(d.loc[retained & d.fund_focus.eq(focus),'row_id'])
    model_records = {record['model']:record for record in summary['models']}
    check('planned_model_names', set(model_records) == set(plan['planned_models']))
    check('unique_coefficient_model_term_keys', not coefficients.duplicated(['model','term']).any())
    approved = set(pd.read_parquet(plan['news_company_source'],columns=['company_id']).company_id.astype(str))
    datasets, inference_status = {}, {}
    source_indexed = d.set_index('row_id')
    for name, record in model_records.items():
        if record.get('status') != 'ESTIMATED':
            check('not_estimated_model_prespecified_N:'+name, record.get('nobs') == len(expected_ids[name]))
            inference_status[name] = dict(status=record.get('status'),net_inference='NOT_ESTIMATED')
            continue
        info = summary['samples'][name]
        sample = pd.read_parquet(info['parquet'])
        check('sample_exact_prespecified_ids:'+name, set(sample.row_id) == expected_ids[name] and not sample.row_id.duplicated().any())
        check('sample_hash_rows:'+name, row_hash(sample) == info['row_hash'] and len(sample) == info['rows'])
        check('sample_model_counts:'+name, len(sample) == record['nobs'] and sample.fund_id.nunique() == record['funds'] and sample.company_id.nunique() == record['companies'])
        check('no_index_funds:'+name, sample.index_status.notna().all() and sample.index_status.ne('index').all() and not truth(sample.exclude_index_fund).any())
        check('observed_news_only_no_NA_to_zero:'+name, finite(sample,['net_sentiment']).all() and sample.n_relevant.gt(0).all() and truth(sample.eligible_primary).all() and sample.company_id.astype(str).isin(approved).all())
        source_rows = source_indexed.reindex(sample.row_id)
        for field in ['quantity_open','quantity_close']+terms_all:
            check('sample_source_value:'+name+':'+field, np.allclose(pd.to_numeric(sample[field],errors='coerce'),pd.to_numeric(source_rows[field],errors='coerce'),rtol=0,atol=0,equal_nan=True))
        is_entry = sample.quantity_open.eq(0) & sample.quantity_close.gt(0)
        regular = sample.quantity_open.gt(0) & sample.quantity_close.ge(0)
        expected_y = np.full(len(sample), np.nan)
        expected_y[np.asarray(regular)] = 100*(sample.loc[regular,'quantity_close']/sample.loc[regular,'quantity_open']-1)
        expected_y[np.asarray(is_entry)] = float(plan['entry_value_percent'])
        check('sample_Y_exact_definition:'+name, np.allclose(sample.quantity_change_pct,expected_y,rtol=1e-9,atol=1e-7))
        check('sample_no_zero_zero_or_unknown_absence:'+name, (regular|is_entry).all() and not truth(sample.unknown_opening_absence).any() and not truth(sample.unknown_closing_absence).any())
        check('sample_entry_provenance:'+name, (~is_entry | truth(sample.opening_position_observed) | (truth(sample.known_opening_absence_zero)&truth(sample.opening_full_portfolio))).all())
        check('sample_original_Y_preserved:'+name,np.allclose(sample.original_quantity_change_pct,source_rows.quantity_change_pct,rtol=0,atol=0,equal_nan=True))
        native = coefficients.loc[coefficients.model.eq(name)].set_index('term')
        terms = list(record['retained_terms'])
        check('native_fund_FE_only:'+name, record.get('time_effects') is False and record.get('included_effects') == ['Entity'])
        covariance_options=record.get('covariance_options',{})
        check('native_fund_company_cluster_options:'+name,covariance_options.get('clusters')==['fund_id','company_id'] and covariance_options.get('cluster_time') is False and all(covariance_options.get(k)==v for k,v in plan['covariance_options'].items()))
        check('no_entry_flag_regressor:'+name, 'entry_zero_to_positive' not in terms and set(terms).issubset(terms_all))
        check('native_coefficient_terms:'+name, set(native.index) == set(['const']+terms))
        target = native.loc['net_sentiment']
        fields=['beta','se_native','t_native','p_native','ci95_native_low','ci95_native_high']
        target_finite=np.isfinite(target[fields].to_numpy(float)).all() and target.se_native > 0
        inference_status[name]=dict(net_inference='FINITE' if target_finite else 'UNDEFINED',
            target=dict(target[fields]),negative_covariance_diagonal_terms=record.get('negative_covariance_diagonal_terms',[]))
        if target_finite:
            check('native_net_bounds:'+name,0 <= target.p_native <= 1 and target.ci95_native_low <= target.beta <= target.ci95_native_high)
        else: warnings.append(dict(model=name,kind='UNDEFINED_TARGET_INFERENCE'))
        non_target_undefined = native.index[~np.isfinite(native[fields].to_numpy(float)).all(axis=1)].tolist()
        if non_target_undefined: warnings.append(dict(model=name,kind='UNDEFINED_COEFFICIENT_INFERENCE',terms=non_target_undefined))
        joint = record.get('native_robust_F',{})
        if joint and (joint.get('stat') is None or not np.isfinite(joint['stat']) or joint['stat'] < 0):
            warnings.append(dict(model=name,kind='UNDEFINED_ROBUST_JOINT_F',native=joint))
        if name in args.models: datasets[name]=sample

    # The planned three focus comparisons form one fixed Holm family.
    holm_column='p_native_holm_focus_family'
    if holm_column in coefficients:
        focus_names=['FOCUS_'+group.upper()+'_P99' for group in plan['holm_focus_groups']]
        focus_rows=coefficients.loc[coefficients.model.isin(focus_names)&coefficients.term.eq('net_sentiment')].set_index('model').reindex(focus_names)
        family_valid=all(inference_status.get(name,{}).get('net_inference')=='FINITE' for name in focus_names)
        if family_valid:
            p=focus_rows.p_native.to_numpy(float)
            order=np.argsort(p,kind='stable')
            adjusted=np.empty(len(p))
            adjusted[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
            check('focus_Holm_fixed_three_model_family',np.allclose(focus_rows[holm_column],adjusted,rtol=args.rtol,atol=1e-10))
        else:
            warnings.append(dict(kind='HOLM_FAMILY_CONTAINS_UNDEFINED_OR_UNESTIMATED_TARGET',models=focus_names))

    for name in args.models:
        if name not in datasets:
            check('independent_model_available:'+name,False); continue
        sample,record=datasets[name],model_records[name]
        terms=record['retained_terms']
        result=within_ols(sample,terms)
        j=terms.index('net_sentiment')
        variance=float(result['covariance'][j,j])
        se=float(np.sqrt(variance)) if variance > 0 else np.nan
        beta=float(result['beta'][j])
        target=coefficients.loc[coefficients.model.eq(name)&coefficients.term.eq('net_sentiment')].iloc[0]
        check('independent_df:'+name,result['df_resid']==record['native_df_resid'])
        check('independent_net_beta:'+name,close(beta,target.beta,1e-5),independent=beta,native=target.beta)
        check('independent_within_R2:'+name,close(result['within_R2'],record['rsquared_within'],1e-10),independent=result['within_R2'],native=record['rsquared_within'])
        if np.isfinite(se):
            t=beta/se
            p=2*stats.t.sf(abs(t),result['df_resid'])
            critical=stats.t.ppf(.975,result['df_resid'])
            low,high=beta-critical*se,beta+critical*se
            check('independent_net_cluster_SE:'+name,close(se,target.se_native),independent=se,native=target.se_native)
            check('independent_net_native_t_p_CI:'+name,close(t,target.t_native) and close(p,target.p_native,1e-10) and close(low,target.ci95_native_low,1e-5) and close(high,target.ci95_native_high,1e-5))
        else:
            check('undefined_covariance_consistent:'+name,not np.isfinite(target.se_native))
            warnings.append(dict(model=name,kind='INDEPENDENT_UNDEFINED_TARGET_VARIANCE',variance=variance))
        details[name]={k:v for k,v in result.items() if k not in ['beta','covariance']}
        details[name].update(net_beta=beta,net_SE=se,net_variance=variance)

    for key,expected in plan['expected_source_sha256'].items():
        check('source_hash_preserved_after_QA:'+key,sha(plan[key])==expected)
    details['independent_masks']=dict(old_positive=int(positive.sum()),certified_entries_before_eligibility=int(raw_entry.sum()),eligible_new_entries=int(entry.sum()),augmented=int(augmented.sum()),trimmed_continuing=int(trim.sum()),retained=int(retained.sum()),monthly_retained=int((retained&monthly).sum()),cutoff=cutoff)
    failures=sum(check['status']=='FAIL' for check in checks)
    result=dict(status='FAIL' if failures else ('PASS_WITH_WARNINGS' if warnings else 'PASS'),
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),plan_sha256=sha(plan_path),
        independent_replication_models=args.models,mechanical_checks_only_models=[name for name in model_records if name not in args.models],
        pass_count=len(checks)-failures,fail_count=failures,checks=checks,warnings=warnings,
        native_inference_status=inference_status,details=details,
        limitations=['Conditional +100 for entries is not a literal percentage from zero.',
            'Outcome-based upper-tail removal is exploratory and does not establish that removed rows are errors.',
            'Only the named two main models have independent numerical replication; other outputs receive mechanical checks.',
            'The prepared approved company scope is checked, not reconstructed from the original title registry.'])
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],pass_count=result['pass_count'],fail_count=failures)))


def fixture():
    rng=np.random.default_rng(18)
    n=1500
    groups=np.repeat(np.arange(30),50)
    companies=np.tile(np.arange(15),100)
    x=rng.normal(size=n)
    control=rng.normal(size=n)
    frame=pd.DataFrame(dict(fund_id=groups,company_id=companies,net_sentiment=x,control=control,
        quantity_change_pct=3*x-.6*control+groups*.3+rng.normal(scale=.1,size=n)))
    result=within_ols(frame,['net_sentiment','control'])
    assert np.allclose(result['beta'],[3,-.6],atol=.015)
    assert result['df_resid']==1500-30-2
    assert np.isfinite(result['covariance']).all()
    assert result['within_R2']>.99
    print(json.dumps(dict(status='PASS',scope='synthetic_only',checks=4)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan')
    parser.add_argument('--root')
    parser.add_argument('--output')
    parser.add_argument('--models',nargs='+',default=['AUGMENTED_P99','OLD_BASE_AFTER_P99'])
    parser.add_argument('--rtol',type=float,default=1e-5)
    parser.add_argument('--fixture-only',action='store_true')
    args=parser.parse_args()
    fixture() if args.fixture_only else audit(args)
