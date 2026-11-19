"""Independent, bounded QA of sealed four-fund provider-convention H3 output.

Reads the research-approved snapshot, factors and frozen NTS. Reconstructs all
past/future alpha legs and exact model samples, then fits full-dummy OLS and CR1.
No production estimator or alpha implementation is imported.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from pathlib import Path
import datetime as dt
import hashlib
import json
import sys
import time
import traceback
import numpy as np
import pandas as pd
from scipy import linalg, stats

BASE = Path('./analysis_outputs/h3_alpha_20261104')
RUN = BASE / 'outputs/provider_TR4_b00ab703609fb58f'
OUT = BASE / 'qa/external_independent_run_d0440fad412ee5e9_20261104/provider_TR4_b00ab703609fb58f'
OUT.mkdir(parents=True, exist_ok=True)
EXPECTED_MANIFEST = 'af25eb9dc57288ef54d2b0839477f0ced6b9539708bd3a88a8cbd2257b8e98c3'
EXPECTED_SNAPSHOT = '5cd9a9ccf60f9e4eafd7b41bddfc140cd7a2e91377b3e45e87dc6ee0028a6983'
START = time.monotonic()
CHECKS = []
DETAILS = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def clean(v):
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (Path, pd.Timestamp, pd.Period, dt.date)):
        return str(v)
    return v


def save(name, value):
    path = OUT / name
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def stage(name, **kw):
    save('checkpoint.json', dict(stage=name, pid=os.getpid(), elapsed_seconds=time.monotonic()-START, **kw))
    print(name, kw, flush=True)


def ck(name, condition, **kw):
    CHECKS.append(dict(check=name, status='PASS' if bool(condition) else 'FAIL', **kw))


def eq(name, actual, expected, rtol=1e-8, atol=1e-11):
    a = np.asarray(actual, dtype=float)
    b = np.asarray(expected, dtype=float)
    same = a.shape == b.shape and np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    delta = np.abs(a-b) if a.shape == b.shape else np.array([])
    ck(name, same, max_absolute_difference=float(np.nanmax(delta)) if np.isfinite(delta).any() else 0., rtol=rtol, atol=atol)


def digest(sample):
    lines = sample.fund_id.astype(str) + '\t' + sample.interval_id.astype(str)
    return hashlib.sha256('\n'.join(lines).encode()).hexdigest()


def verify_manifest(m, label):
    for group in ['inputs', 'files']:
        for key, item in m[group].items():
            p = Path(item['path'])
            ck(label + ':' + group + ':' + key, p.is_file() and p.stat().st_size == item['bytes'] and sha(p) == item['sha256'])


def main():
    stage('PINNED_SNAPSHOT_AND_SOURCE_HASHES')
    m = json.loads((RUN/'manifest.json').read_text(encoding='utf-8'))
    ck('exact_sealed_four_fund_run', sha(RUN/'manifest.json') == EXPECTED_MANIFEST)
    verify_manifest(m, 'before')
    snapshot = Path(m['inputs']['snapshot']['path'])
    ck('exact_research_approved_228_snapshot', sha(snapshot) == EXPECTED_SNAPSHOT)
    sd = json.loads(snapshot.read_text(encoding='utf-8'))
    source = pd.DataFrame(sd['rows']).sort_values(['fund_id', 'month']).reset_index(drop=True)
    r = pd.read_parquet(RUN/'approved_monthly_provider_returns.parquet').sort_values(['fund_id', 'month']).reset_index(drop=True)
    ck('228_rows_four_funds_no_snapshot_errors', len(source) == len(r) == 228 and r.fund_id.nunique() == 4 and not sd['errors'])
    ck('exact_keys_and_registrations', r[['fund_id','month','registration_number']].astype(str).equals(source[['fund_id','month','registration_number']].astype(str)))
    ck('unique_months_no_pre2022_backfill', not r.duplicated(['fund_id','month']).any() and r.month.ge('2022-01').all())
    expected_months = pd.period_range('2022-01','2026-09',freq='M').astype(str).tolist()
    ck('57_months_per_fund', all(g.month.tolist() == expected_months for _, g in r.groupby('fund_id')))
    eq('printed_return_preserved', r.fund_return, source.return_decimal, rtol=0, atol=0)
    for field in ['source_path','source_url','source_sha256','return_decimal']:
        ck('snapshot_field_preserved:'+field, r[field].equals(source[field]))
    for saved, raw in [('start_valuation_date','actual_start_valuation_date'),('end_valuation_date','actual_end_valuation_date')]:
        ck('observed_endpoint_preserved:'+saved, np.array_equal(pd.to_datetime(r[saved]).to_numpy(), pd.to_datetime(source[raw]).to_numpy()))
    for z in source.itertuples(index=False):
        ck('source_PDF_SHA:'+str(z.fund_id)+':'+z.month, sha(z.source_path) == z.source_sha256)
    old = json.loads((snapshot.parent/'cached_monthly_documentary_snapshot_20261104.json').read_text(encoding='utf-8'))
    merged = pd.DataFrame(old['rows']).merge(source,on=['fund_id','month'],suffixes=('_old','_new'),validate='one_to_one')
    ck('old55_preserved_without_reauditing_one_fund_run', len(merged) == len(old['rows']) == 55 and merged.source_sha256_old.eq(merged.source_sha256_new).all() and merged.return_decimal_old.eq(merged.return_decimal_new).all())
    ck('honest_source_convention_and_no_second_adjustment', r.return_convention.str.contains('accrued',case=False).all() and r.return_convention.str.contains('not audited payment-date reinvestment').all() and r.raw_NAV_history_promoted_TR.eq(False).all() and r.additional_payout_or_split_adjustment_applied.eq(False).all() and r.cash_event_history_complete.eq(False).all())
    cw = pd.read_parquet(m['inputs']['crosswalk']['path']).set_index('fund_id')
    ck('identities_registrations_currency_and_source_gate', r.identity_verified.eq(True).all() and r.identity_unambiguous.eq(True).all() and r.currency_verified_RUB.eq(True).all() and r.return_verified.eq(True).all() and all(str(cw.loc[z.fund_id,'registration_number']) == str(z.registration_number) for z in r.itertuples(index=False)))
    ck('source_endpoints_in_observed_identity_span', (pd.to_datetime(r.start_valuation_date) >= pd.to_datetime(r.identity_observed_from)).all() and (pd.to_datetime(r.end_valuation_date) <= pd.to_datetime(r.identity_observed_to)).all())
    f = pd.read_parquet(m['inputs']['factors']['path'])
    origin = pd.read_parquet(m['inputs']['origins']['path'])
    origin = origin.loc[origin.fund_id.astype(str).isin(r.fund_id.astype(str))].copy()
    ck('unique_source_origins', not origin.duplicated(['fund_id','interval_id']).any())
    origin_identity = origin.set_index('interval_id').identity_origin_verified
    broad = pd.read_parquet(m['inputs']['broad_NTS']['path'])
    broad = broad.loc[broad.fund_id.astype(str).isin(r.fund_id.astype(str))].copy()
    oi = origin.set_index('interval_id')
    for col in origin.columns:
        if col not in broad or col in ['identity_origin_verified','universe_eligible','row_id']:
            broad[col] = broad.interval_id.map(oi[col])
    frames = {}
    replica_alphas = {}
    alpha_stats = []
    examples = []
    stage('ALL_SOURCE_BASED_PAST_FITS_AND_FUTURE_ALPHA')
    for label, lookback, minimum, credit in [('lookback60_min36',60,36,False),('lookback36_min24',36,24,False),('lookback60_min36_credit',60,36,True)]:
        a = pd.read_parquet(RUN/(label+'_alpha.parquet'))
        training = pd.read_parquet(RUN/(label+'_training.parquet'))
        ck(label+':unique_alpha_and_training', not a.duplicated(['fund_id','interval_id','horizon']).any() and not training.duplicated(['training_id','month']).any())
        ck(label+':no_pre2022_training', training.empty or training.month.ge('2022-01').all())
        ck(label+':exact_all_origin_horizon_keys', len(a) == len(origin)*3 and set(map(tuple,a[['fund_id','interval_id']].to_numpy())) == set(map(tuple,origin[['fund_id','interval_id']].to_numpy())))
        d = r.merge(f,on='month',validate='many_to_one')
        d['period'] = pd.PeriodIndex(d.month,freq='M')
        fields = ['equity_return','bond_return'] + (['credit_factor'] if credit else [])
        d['factor_ok'] = d['corporate_sensitivity_complete' if credit else 'factors_verified'].eq(True) & np.isfinite(d[['risk_free']+fields]).all(axis=1)
        d['ok'] = d.return_verified.eq(True) & np.isfinite(d.fund_return) & d.factor_ok
        groups = {str(k): g.sort_values('period').set_index('period',drop=False) for k,g in d.groupby('fund_id')}
        trainmap = {k:g.sort_values('month') for k,g in training.groupby('training_id')}
        computed = {}
        fitcount = 0
        outcomes = 0
        maxerr = 0.
        for (fund, interval), rows in a.groupby(['fund_id','interval_id'],sort=False):
            row = rows.iloc[0]
            g = groups[str(fund)]
            date = pd.Timestamp(row.origin_date)
            month = date.to_period('M')
            last = month if month.end_time.normalize() <= date else month-1
            first = last-lookback+1
            t = g.loc[g.period.between(first,last) & g.ok & pd.to_datetime(g.return_period_end).le(date) & pd.to_datetime(g.end_valuation_date).le(date) & pd.to_datetime(g.start_valuation_date).le(date)]
            recent = int(t.period.ge(last-11).sum())
            ck(label+':past_bounds:'+interval, row.n_past == len(t) and row.n_past_recent == recent and row.training_first_allowed == str(first) and row.training_last_allowed == str(last))
            b = None
            condition = np.nan
            if len(t) >= minimum and recent >= 9 and origin_identity.loc[interval] == True:
                xx = np.column_stack([t.equity_return-t.risk_free,t.bond_return-t.risk_free]+([t.credit_factor] if credit else []))
                X = np.column_stack([np.ones(len(t)),xx])
                y = (t.fund_return-t.risk_free).to_numpy(float)
                condition = float(np.linalg.cond(np.column_stack([np.ones(len(t)),(xx-xx.mean(0))/xx.std(0,ddof=1)])))
                if np.linalg.matrix_rank(X) == X.shape[1] and condition <= 10000:
                    b = linalg.lstsq(X,y,lapack_driver='gelsy')[0]
                    fitcount += 1
                    rec = trainmap.get(row.training_id)
                    ck(label+':past_members:'+interval, rec is not None and rec.month.tolist() == t.month.tolist())
                    if rec is not None:
                        eq(label+':past_design:'+interval, rec[['fund_excess']+['x_'+x for x in fields]], np.column_stack([y,xx]),rtol=1e-12,atol=1e-13)
                    eq(label+':past_betas:'+interval, [row.past_alpha]+[row['beta_'+x] for x in fields],b)
                    eq(label+':past_condition:'+interval,[row.training_condition_number],[condition])
            ck(label+':past_fit_or_NA:'+interval, (b is not None) == pd.notna(row.past_alpha))
            for z in rows.itertuples(index=False):
                months = pd.period_range(month+1,month+int(z.horizon),freq='M')
                legs = g.reindex(months)
                full = legs.ok.eq(True).all()
                anchor = legs.start_valuation_date.iloc[0]
                anchorok = pd.notna(anchor) and pd.Timestamp(anchor) >= date
                value = np.nan
                if b is not None and full and anchorok:
                    xx = np.column_stack([legs.equity_return-legs.risk_free,legs.bond_return-legs.risk_free]+([legs.credit_factor] if credit else []))
                    abnormal = (legs.fund_return-legs.risk_free).to_numpy(float)-xx@b[1:]
                    value = float(abnormal.mean())
                    outcomes += 1
                    maxerr = max(maxerr,abs(value-z.alpha))
                    if len([e for e in examples if e['profile']==label]) < 3 and z.horizon == 3:
                        examples.append(dict(profile=label,fund_id=str(fund),interval_id=interval,origin_date=date,training_months=t.month.tolist(),training_last_allowed=str(last),betas=b.tolist(),future_months=legs.month.tolist(),future_fund_return=legs.fund_return.tolist(),future_rf=legs.risk_free.tolist(),future_factor_design=xx.tolist(),future_abnormal=abnormal.tolist(),alpha_independent=value,alpha_saved=z.alpha))
                computed[(str(fund),interval,int(z.horizon))] = value
                eq(label+':future_alpha_or_NA:'+interval+':h'+str(z.horizon),[z.alpha],[value],rtol=1e-9,atol=1e-11)
                ck(label+':future_status:'+interval+':h'+str(z.horizon), (z.status=='eligible') == np.isfinite(value))
        ck(label+':all_numeric_alpha_replicated', outcomes == int(a.alpha.notna().sum()))
        frames[label] = a
        replica_alphas[label] = computed
        alpha_stats.append(dict(profile=label,origins=a.interval_id.nunique(),past_fits=fitcount,numeric_alpha=outcomes,max_absolute_difference=maxerr))
    stage('ALL90_EXACT_SAMPLES_AND_NINE_FULL_DUMMY_CR1_FITS')
    models = json.loads((RUN/'all_models.json').read_text(encoding='utf-8'))
    ck('all90_unique_fixed_labels', len(models) == 90 and len(set(z['model_id'] for z in models)) == 90)
    replicas = []
    all_samples = []
    singular = 0
    for model in models:
        mid = model['model_id']
        sensitivity = model['sensitivity']
        label = 'lookback36_min24' if sensitivity == 'HISTORY36_MIN24' else 'lookback60_min36_credit' if sensitivity == 'CORPORATE_BOND_SPREAD' else 'lookback60_min36'
        a = frames[label].copy()
        if sensitivity == 'CORPORATE_SAME_SAMPLE_TWO_FACTOR':
            eligible = set(map(tuple,frames['lookback60_min36_credit'].loc[lambda d:d.alpha.notna(),['fund_id','interval_id','horizon']].to_numpy()))
            a.loc[[tuple(x) not in eligible for x in a[['fund_id','interval_id','horizon']].to_numpy()],'alpha'] = np.nan
        o = broad if sensitivity == 'BROAD_OBSERVED_NTS' else origin
        pool = a.merge(o.drop(columns=['origin_month'],errors='ignore'),on=['fund_id','interval_id'],validate='many_to_one',suffixes=('','_origin'))
        mask = pool.horizon.eq(model['horizon']) & pool.universe_eligible.eq(True) & pool.nts_pair_eligible.eq(True) & np.isfinite(pool[['alpha','past_alpha','NTS_Q_current','NTS_Q_lagnews']]).all(axis=1)
        if model['group'] != 'pooled':
            mask &= pool.fund_focus.eq(model['group'])
        if sensitivity == 'NONOVERLAPPING_PHASES':
            periods = pd.PeriodIndex(pool.origin_month,freq='M')
            mask &= (periods.year*12+periods.month-1)%model['horizon'] == model['phase']
        expected = pool.loc[mask].copy()
        sizes = expected.groupby('fund_id').interval_id.transform('size')
        removed = int(sizes.lt(2).sum())
        expected = expected.loc[sizes.ge(2)].sort_values(['fund_id','end_date','interval_id']).reset_index(drop=True)
        if model.get('reuse_of_primary_model_id'):
            path = RUN/'regression/PRIMARY'/model['reuse_of_primary_model_id']
        elif sensitivity == 'NONOVERLAPPING_PHASES':
            path = RUN/('regression/NONOVERLAPPING_PHASE'+str(model['phase']))/mid
        else:
            path = RUN/'regression'/sensitivity/mid
        sample = pd.read_parquet(path/'used_sample.parquet').sort_values(['fund_id','end_date','interval_id']).reset_index(drop=True)
        ck('exact_unique_model_rows:'+mid, not sample.duplicated(['fund_id','interval_id']).any() and sample[['fund_id','interval_id']].equals(expected[['fund_id','interval_id']]))
        ck('sample_hash_N_G_singletons:'+mid, digest(sample) == model['sample_hash'] and len(sample) == model['nobs'] == len(expected) and sample.fund_id.nunique() == model['funds'] and removed == model['singleton_rows_removed'])
        eq('exact_model_inputs:'+mid,sample[['alpha','past_alpha','NTS_Q_current','NTS_Q_lagnews']],expected[['alpha','past_alpha','NTS_Q_current','NTS_Q_lagnews']],rtol=0,atol=0)
        if len(sample):
            eq('model_alpha_from_independent_source_legs:'+mid,sample.alpha,[replica_alphas[label][(str(x.fund_id),x.interval_id,int(x.horizon))] for x in sample.itertuples(index=False)],rtol=1e-9,atol=1e-11)
        all_samples.append(dict(model_id=mid,n=len(sample),funds=model['funds'],sample_hash=model['sample_hash'],status=model['status']))
        n = len(sample)
        G = sample.fund_id.nunique()
        ck('frozen_FE_cluster_reference_and_fragility:'+mid, model['fund_FE'] and not model['time_FE'] and model['clusters']==['fund_id'] and model['reference_df']==G-1 and model['weak_cluster_inference'] and model['significance_stars_suppressed'] and not model['multiple_testing_adjustments'])
        if model['status'] != 'ESTIMATED':
            ck('NA_justified_and_no_inference:'+mid, (n < 30 or G < 2) and model['reason']=='fewer_than_30_rows_or_two_funds_after_singletons' and not model.get('coefficients') and not model.get('target_tests'))
            continue
        ck('estimated_gate:'+mid,n>=30 and 2<=G<=4)
        terms = ['NTS_Q_current']+(['NTS_Q_lagnews'] if model['distributed'] else [])+['past_alpha']
        codes, fundnames = pd.factorize(sample.fund_id,sort=True)
        sizes = np.bincount(codes)
        X = sample[terms].to_numpy(float)
        Y = sample.alpha.to_numpy(float)
        within = X-np.vstack([X[codes==j].mean(0) for j in range(G)])[codes]
        absorbed = np.max(abs(within),axis=0) <= 1e-12*np.maximum(1,np.max(abs(X),axis=0))
        ck('absorbed_columns:'+mid,[term for term,drop in zip(terms,absorbed) if drop] == model['absorbed_columns'])
        terms = [term for term,drop in zip(terms,absorbed) if not drop]
        X = sample[terms].to_numpy(float)
        k = len(terms)
        D = np.column_stack([np.eye(G)[codes],X])
        b = linalg.lstsq(D,Y,lapack_driver='gelsy')[0]
        residual = Y-D@b
        bread = linalg.inv(D.T@D)
        scores = np.vstack([D[codes==j].T@residual[codes==j] for j in range(G)])
        correction = G/(G-1)*(n-1)/(n-G-k)
        covD = correction*bread@scores.T@scores@bread
        T = np.zeros((k+1,G+k))
        T[0,:G] = sizes/n
        T[1:,G:] = np.eye(k)
        beta = T@b
        cov = T@covD@T.T
        names = ['const']+terms
        reported = pd.DataFrame(model['coefficients']).set_index('term').loc[names]
        native_cov = pd.read_csv(path/'full_covariance.csv',index_col=0,encoding='utf-8-sig').loc[names,names]
        eq('full_dummy_OLS_coefficients:'+mid,beta,reported.beta,rtol=1e-7,atol=1e-10)
        eq('full_dummy_CR1_covariance:'+mid,cov,native_cov,rtol=1e-6,atol=1e-11)
        ck('full_rank_df:'+mid,np.linalg.matrix_rank(D)==G+k==model['K_full'] and model['residual_df']==n-G-k)
        eq('CR1_correction:'+mid,[correction],[model['CR1_correction']])
        se = np.sqrt(np.diag(cov))
        tt = beta/se
        pp = 2*stats.t.sf(abs(tt),G-1)
        q = stats.t.ppf(.975,G-1)
        for col, value in [('SE',se),('t',tt),('p',pp),('CI_low',beta-q*se),('CI_high',beta+q*se)]:
            eq('independent_tGminus1_'+col+':'+mid,value,reported[col],rtol=1e-6,atol=1e-9)
        WY = Y-np.array([Y[codes==j].mean() for j in range(G)])[codes]
        r2 = 1-residual@residual/(WY@WY)
        eq('within_R2:'+mid,[r2],[model['within_R2']])
        tests = {z['test']:z for z in model['target_tests']}
        for j,term in enumerate(terms[:2],start=1):
            z = tests[term+'=0']
            eq('individual_target_test:'+mid+':'+term,[z[x] for x in ['beta','SE','t','p','CI_low','CI_high']],[beta[j],se[j],tt[j],pp[j],beta[j]-q*se[j],beta[j]+q*se[j]],rtol=1e-6,atol=1e-9)
        block = cov[1:3,1:3]
        target_beta = beta[1:3]
        sum_beta = target_beta.sum()
        sum_se = np.sqrt(block.sum())
        sum_t = sum_beta/sum_se
        z = tests['current+lagnews=0']
        eq('sum_target_test:'+mid,[z[x] for x in ['beta','SE','t','p','CI_low','CI_high']],[sum_beta,sum_se,sum_t,2*stats.t.sf(abs(sum_t),G-1),sum_beta-q*sum_se,sum_beta+q*sum_se],rtol=1e-6,atol=1e-9)
        joint = tests['joint_current_lagnews=0']
        identifiable = G>=3 and np.linalg.matrix_rank(block)==2 and np.linalg.eigvalsh(block).min()>0
        if identifiable:
            F = float(target_beta@linalg.solve(block,target_beta)/2)
            ck('joint_test_defined_df:'+mid,joint['status']=='defined' and joint['df_num']==2 and joint['df_denom']==G-1)
            eq('joint_F_p:'+mid,[joint['F'],joint['p']],[F,stats.f.sf(F,2,G-1)],rtol=1e-6,atol=1e-9)
        else:
            singular += 1
            ck('singular_joint_honest_NA:'+mid,joint['status']=='undefined_target_covariance_not_positive_definite' and not np.isfinite([joint['F'],joint['p']]).any(),funds=int(G),covariance_rank=int(np.linalg.matrix_rank(block)))
        replicas.append(dict(model_id=mid,n=n,funds=int(G),rank=G+k,residual_df=n-G-k,reference_df=G-1,CR1_correction=correction,sample_hash=model['sample_hash'],max_coef_difference=float(np.max(abs(beta-reported.beta.to_numpy()))),max_cov_difference=float(np.max(abs(cov-native_cov.to_numpy()))),independent_full_dummy_fit=True,joint_status=joint['status']))
    ck('all9_estimated_models_independently_replicated',len(replicas)==9)
    ck('all3_two_fund_joint_tests_NA',singular==3)
    ck('primary_pooled_16_rows_and_NA',all(z['nobs']==16 and z['status']=='NOT_ESTIMABLE' for z in models if z['sensitivity']=='PRIMARY' and z['group']=='pooled'))
    stage('FINAL_UNCHANGED_HASHES')
    ck('source_manifest_unchanged',sha(RUN/'manifest.json')==EXPECTED_MANIFEST)
    verify_manifest(m,'after')
    DETAILS.update(alpha_reconstruction=alpha_stats,representative_source_legs=examples,regression_replicas=replicas,all90_samples=all_samples,source_funds=sorted(r.fund_id.unique()),source_months=228,source_first_month=r.month.min(),source_last_month=r.month.max(),singular_joint_tests_NA=singular,manifest_sha256=EXPECTED_MANIFEST,snapshot_sha256=EXPECTED_SNAPSHOT)


try:
    main()
    report = dict(status='PASS' if all(c['status']=='PASS' for c in CHECKS) else 'FAIL',audit_type='BOUNDED_EXTERNAL_INDEPENDENT_PROVIDER_TR4_NUMERICAL',run=str(RUN),audit_code_sha256=sha(__file__),production_estimator_imports=False,executor_checks_counted_as_external=False,checks_pass=sum(c['status']=='PASS' for c in CHECKS),checks_fail=sum(c['status']=='FAIL' for c in CHECKS),checks=CHECKS,**DETAILS,limitations=['Relies on the research-approved immutable documentary snapshot; PDF hashes and transfer are verified, documentary interpretation is not independently recertified.','All four funds belong to one manager; few clusters make inference fragile, and stars remain suppressed.','Published accrued-income/split-normalized convention is not audited payment-date cash reinvestment; no pre-2022 NAV promotion, complete cash ledger or extra payout/split adjustment is introduced.','Frozen factor and NTS source versions are reused; v3 external QA remains a separate completed audit.'],completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),runtime_seconds=time.monotonic()-START)
    save('independent_qa.json',report)
    save('summary.json',{k:v for k,v in report.items() if k!='checks'})
    stage('COMPLETED',status=report['status'],checks_pass=report['checks_pass'],checks_fail=report['checks_fail'])
    if report['checks_fail']:
        print('FAILURES',json.dumps(clean([x for x in CHECKS if x['status']=='FAIL']),ensure_ascii=False))
        sys.exit(1)
except Exception:
    save('execution_failure.json',dict(error=traceback.format_exc(),checks=CHECKS,**DETAILS))
    stage('EXECUTION_FAILED')
    raise
