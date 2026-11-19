"""Independent read-only QA for the versioned H1 macro rerun.

Run actual-data QA on research-host. This module never imports the regression runner or
PanelOLS, changes research outputs, or opens a database. --self-test is synthetic.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def row_hash(frame):
    return hashlib.sha256('\n'.join(frame.row_id.astype(str)).encode()).hexdigest()


def truth(series):
    return series.astype(str).str.strip().str.lower().isin(['true', '1', 'yes'])


def clean(value):
    if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (np.integer,np.bool_)): return value.item()
    if isinstance(value, (float,np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value, (Path,pd.Timestamp,dt.datetime)): return str(value)
    return value


def cluster_meat(scores, codes):
    codes = np.asarray(codes, dtype=int)
    g, n = int(codes.max()) + 1, len(codes)
    if g < 2: raise ValueError('Need at least two clusters')
    totals = np.zeros((g, scores.shape[1]), dtype=float)
    np.add.at(totals, codes, scores)
    return (totals.T @ totals) * (g / (g-1)) * ((n-1) / n), g


def independent_within(frame, terms):
    """Fund-demeaned OLS and Cameron-Gelbach-Miller two-way sandwich."""
    x = frame[terms].astype(float)
    y = frame.quantity_change_pct.astype(float)
    groups = frame.fund_id.astype(str)
    xw = (x-x.groupby(groups,sort=False).transform('mean')).to_numpy(float)
    yw = (y-y.groupby(groups,sort=False).transform('mean')).to_numpy(float)
    scale = np.sqrt(np.square(xw).sum(axis=0))
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Retained within regressor has no finite variation')
    z = xw / scale
    scaled_beta, _, rank, singular = np.linalg.lstsq(z,yw,rcond=None)
    if rank != len(terms): raise ValueError('Independent within design is rank deficient')
    residual = yw-z@scaled_beta
    bread = np.linalg.inv(z.T@z)
    score = z*residual[:,None]
    fc,_ = pd.factorize(groups,sort=False)
    cc,_ = pd.factorize(frame.company_id.astype(str),sort=False)
    jc,_ = pd.factorize(pd.MultiIndex.from_arrays([groups,frame.company_id.astype(str)]),sort=False)
    fm,nf = cluster_meat(score,fc)
    cm,nc = cluster_meat(score,cc)
    jm,nj = cluster_meat(score,jc)
    n,k = z.shape
    df = n-nf-k
    if df <= 0: raise ValueError('Nonpositive residual degrees of freedom')
    cov_scaled = bread@(fm+cm-jm)@bread*(n/df)
    covariance = cov_scaled/scale[:,None]/scale[None,:]
    beta = scaled_beta/scale
    return dict(beta=beta,covariance=(covariance+covariance.T)/2,
        df_resid=df,nobs=n,funds=nf,companies=nc,joint_clusters=nj,
        design_rank=rank,scaled_condition_number=float(singular.max()/singular.min()),
        residual_SS=float(residual@residual))


def audit(args):
    from scipy import stats
    import pyarrow.parquet as pq
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    root = Path(args.root or plan['output_root'])
    summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    coefs = pd.read_csv(root/'coefficients.csv')
    manifest = json.loads((root/'source_manifest.json').read_text(encoding='utf-8'))
    checks, details, warnings = [], {}, []
    inference_status = {}

    def check(name, passed, **evidence):
        checks.append(dict(name=name,status='PASS' if bool(passed) else 'FAIL',**clean(evidence)))

    def close(a,b,atol=1e-7):
        return bool(np.isclose(float(a),float(b),rtol=args.rtol,atol=atol,equal_nan=False))

    check('frozen_plan_hash',sha(plan_path)==summary['plan_sha256'])
    check('no_time_clustering_or_time_FE',plan.get('cluster_time') is False and plan.get('time_FE') is False
        and plan.get('clusters')==['fund_id','company_id'] and summary.get('no_time_FE') is True)
    check('same_declared_specification',summary.get('specification')==plan)
    source_records = {k:v for k,v in manifest.items() if isinstance(v,dict) and 'path' in v and 'sha256' in v}
    for k,v in manifest.get('macro_raw_sources',{}).items():
        source_records['macro_'+k] = v
    for key,rec in source_records.items():
        p = Path(rec['path'])
        actual = sha(p)
        check('unchanged_source_hash:'+key,actual==rec['sha256'],path=str(p),sha256=actual)
        expected = plan.get('expected_source_sha256',{}).get(key)
        if expected: check('frozen_expected_source:'+key,actual==expected)
    classification = pd.read_parquet(root/'fund_interval_classification.parquet',
        columns=['fund_id','index_status','exclude_index_fund'])
    ever_index = set(classification.loc[classification.index_status.eq('index') |
        truth(classification.exclude_index_fund),'fund_id'].astype(str))
    approved = set(pd.read_parquet(plan['news_company_source'],columns=['company_id']).company_id.dropna().astype(str))
    details['approved_scope'] = dict(companies=len(approved),
        scope='Checked against frozen prepared title-name scope; canonical-minus-GAZ construction is not rebuilt by this QA.')
    model_records = {m['model']:m for m in summary['models']}
    check('unique_coefficient_keys',not coefs.duplicated(['model','term']).any())
    for name,record in model_records.items():
        if record.get('status')!='ESTIMATED':
            inference_status[name] = dict(status=record.get('status'),usable_for_H1=False)
            continue
        found=coefs.loc[coefs.model.eq(name)&coefs.term.eq('net_sentiment')]
        check('native_net_coefficient_present:'+name,len(found)==1)
        if len(found)!=1:continue
        row=found.iloc[0]
        fields=['beta','se_native','t_native','p_native','ci95_native_low','ci95_native_high']
        valid=np.isfinite(row[fields].to_numpy(float)).all()
        undefined_other=[]
        for _,coefficient in coefs.loc[coefs.model.eq(name)&coefs.term.ne('net_sentiment')].iterrows():
            values=coefficient[fields].to_numpy(float)
            if not np.isfinite(values).all() or coefficient.se_native<=0:
                undefined_other.append(str(coefficient.term))
        inference_status[name]=dict(net_inference='FINITE' if valid else 'UNDEFINED',
            usable_for_H1=bool(valid and row.se_native>0 and 0<=row.p_native<=1),
            non_target_undefined_terms=undefined_other,
            negative_covariance_diagonal_terms=record.get('negative_covariance_diagonal_terms',[]))
        if undefined_other:
            warnings.append(dict(model=name,status='UNDEFINED_NON_TARGET_INFERENCE',terms=undefined_other))
        if not valid:
            warnings.append(dict(model=name,status='UNDEFINED_NATIVE_INFERENCE',
                target='net_sentiment',usable_for_H1=False,
                adjusted_pvalue_usable=False,
                adjusted_pvalue_reported=row.get('p_native_holm_focus_family'),
                negative_diagonal_terms=record.get('negative_covariance_diagonal_terms',[])))
        else:
            check('native_net_inference_bounds:'+name,row.se_native>0 and 0<=row.p_native<=1
                and row.ci95_native_low<=row.beta<=row.ci95_native_high)
    datasets = {}
    for name,info in summary['samples'].items():
        columns = pq.read_schema(info['parquet']).names
        keys = [c for c in ['row_id','fund_id','company_id','net_sentiment','n_relevant','eligible_primary',
            'index_status','exclude_index_fund','fund_focus','start_date','end_date'] if c in columns]
        d = pd.read_parquet(info['parquet'],columns=None if name in args.models else keys)
        record = model_records[name]
        check('sample_rows_hash:'+name,len(d)==info['rows'] and row_hash(d)==info['row_hash'])
        check('model_counts:'+name,len(d)==record['nobs'] and d.fund_id.nunique()==record['funds']
            and d.company_id.nunique()==record['companies'])
        check('unique_rows:'+name,not d.row_id.duplicated().any())
        check('observed_news_only:'+name,np.isfinite(d.net_sentiment).all() and
            d.n_relevant.gt(0).all() and truth(d.eligible_primary).all())
        check('used_issuer_within_approved_news_scope:'+name,d.company_id.astype(str).isin(approved).all())
        if name!='ORIGINAL_MICRO':
            check('no_ever_index_fund:'+name,not (set(d.fund_id.astype(str))&ever_index)
                and not truth(d.exclude_index_fund).any() and not d.index_status.eq('index').any())
            if plan['universe_options']['index_unknown_policy']=='exclude':
                check('strict_nonindex_statuses:'+name,d.index_status.isin(plan['strict_nonindex_statuses']).all())
            details.setdefault('retained_status_counts',{})[name] = d.index_status.value_counts(dropna=False).to_dict()
        if name in args.models: datasets[name]=d

    left,right = ('BASELINE_COMMON','NET_MACRO_COMMON')
    check('common_models_both_estimated',left in datasets and right in datasets)
    if left in datasets and right in datasets:
        check('identical_common_row_order',datasets[left].row_id.equals(datasets[right].row_id))
    for name in args.models:
        if name not in datasets:
            check('independent_model_available:'+name,False);continue
        d,record = datasets[name],model_records[name]
        before_checks=len(checks)
        native = coefs.loc[coefs.model.eq(name)].set_index('term')
        terms = list(record['retained_terms'])
        check('coefficient_terms:'+name,set(native.index)==set(['const']+terms))
        result = independent_within(d,terms)
        check('independent_df:'+name,result['df_resid']==record['native_df_resid'],
            calculated=result['df_resid'],native=record['native_df_resid'])
        beta = dict(zip(terms,result['beta']))
        j = terms.index('net_sentiment')
        variance = float(result['covariance'][j,j])
        se = np.sqrt(variance) if variance>=0 else np.nan
        nc = native.loc['net_sentiment']
        check('independent_net_slope:'+name,close(beta['net_sentiment'],nc.beta,1e-5),
            independent=beta['net_sentiment'],native=float(nc.beta))
        if np.isfinite(se) and se>0:
            check('independent_net_cluster_SE:'+name,close(se,nc.se_native,1e-7),
                independent=float(se),native=float(nc.se_native))
            t = beta['net_sentiment']/se
            p = 2*stats.t.sf(abs(t),result['df_resid'])
            critical = stats.t.ppf(.975,result['df_resid'])
            low,high = beta['net_sentiment']-critical*se,beta['net_sentiment']+critical*se
            check('native_t_p_CI:'+name,close(t,nc.t_native) and close(p,nc.p_native,1e-10)
                and close(low,nc.ci95_native_low,1e-5) and close(high,nc.ci95_native_high,1e-5))
        else:
            check('consistent_undefined_cluster_inference:'+name,not np.isfinite(nc.se_native))
            warnings.append(dict(model=name,status='UNDEFINED_NATIVE_INFERENCE',independent_net_variance=variance))
        details[name] = {k:v for k,v in result.items() if k not in ['beta','covariance']}
        details[name].update(beta=beta,net_SE=se,net_variance=variance)
        numerical_checks=checks[before_checks:]
        details[name]['numerical_replication_status']='PASS' if all(c['status']=='PASS' for c in numerical_checks) else 'FAIL'
        details[name]['native_net_inference_status']=inference_status.get(name,{})

    # Independent dated-source and macro-value checks use unique intervals only.
    main = datasets.get('NET_MACRO_COMMON')
    if main is not None:
        md = json.loads((root/'macro_metadata.json').read_text(encoding='utf-8'))
        intervals = main.drop_duplicates('interval_id').copy()
        limit = plan['macro_builder_options']['max_age_days']
        for term,datecol,valcol,fmt in [('Imoex','TRADEDATE','CLOSE','%Y-%m-%d'),
            ('Ruonia','DT','ruo','%Y-%m-%d'),('USD_RUB','Date','VunitRate','%d.%m.%Y'),
            ('RVI','TRADEDATE','CLOSE','%Y-%m-%d')]:
            src = pd.read_csv(md['source_files'][term]['path'],dtype=str)
            dates = pd.to_datetime(src[datecol],format=fmt).to_numpy(dtype='datetime64[ns]')
            values = pd.to_numeric(src[valcol].str.replace(' ','',regex=False).str.replace('\u00a0','',regex=False).str.replace(',','.',regex=False)).to_numpy(float)
            order = np.argsort(dates);dates,values=dates[order],values[order]
            for stem,endpoint in [(term,'end_date')]+([('Imoex_open','start_date')] if term=='Imoex' else []):
                target = pd.to_datetime(intervals[endpoint]).to_numpy(dtype='datetime64[ns]')
                loc = np.searchsorted(dates,target,side='right')-1
                valid=loc>=0
                selected=pd.to_datetime(intervals[stem+'_source_date'])
                age=(pd.to_datetime(intervals[endpoint])-selected).dt.days
                expected_values=np.full(len(target),np.nan);expected_dates=np.full(len(target),np.datetime64('NaT'),dtype='datetime64[ns]')
                expected_values[valid]=values[loc[valid]];expected_dates[valid]=dates[loc[valid]]
                check('macro_endpoint_date_and_age:'+stem,valid.all() and age.between(0,limit).all()
                    and np.array_equal(selected.to_numpy(dtype='datetime64[ns]'),expected_dates)
                    and np.array_equal(age.to_numpy(),intervals[stem+'_source_age_days'].to_numpy()))
                check('macro_endpoint_value:'+stem,np.allclose(intervals[stem].to_numpy(float),expected_values,rtol=1e-12,atol=1e-10))
        check('macro_exact_interval_return',np.allclose(intervals.imoex_ret,
            intervals.Imoex/intervals.Imoex_open-1,rtol=1e-12,atol=1e-12))
        full=pd.read_parquet(plan['interval_source'],columns=['fund_id','end_date'])
        counts=full.groupby('end_date').fund_id.nunique()
        expected=pd.to_datetime(intervals.end_date).map(counts)
        check('macro_unique_funds_full_universe',np.array_equal(intervals.unique_funds.to_numpy(),expected.to_numpy()))
        for term,cutoff in [('dummy_mar2014','2014-03-01'),('dummy_feb2022','2022-02-01'),('dummy_sep2018','2018-09-01')]:
            check('macro_dummy:'+term,np.array_equal(intervals[term].to_numpy(),pd.to_datetime(intervals.end_date).ge(cutoff).astype(int).to_numpy()))
        raw=pd.read_csv(md['source_files']['Flow']['path'],dtype=str)
        raw=raw.loc[raw.data_key_source.eq('inflow')&raw.series_id_source.eq('4')].copy()
        months=pd.to_datetime(pd.to_numeric(raw.timestamp_ms_source),unit='ms',utc=True).dt.tz_convert('Europe/Moscow').dt.tz_localize(None).dt.to_period('M')-1
        values=pd.to_numeric(raw.value_source.str.replace(',','.',regex=False)).to_numpy(float)/1e9
        lookup=dict(zip(months.astype(str),values))
        end=pd.to_datetime(intervals.end_date)
        expected_month=[str(v.to_period('M') if v.is_month_end else v.to_period('M')-1) for v in end]
        expected_value=np.array([lookup.get(m,np.nan) for m in expected_month])
        check('macro_Flow_latest_completed_month',intervals.Flow_latest_completed_economic_month.astype(str).tolist()==expected_month)
        check('macro_Flow_raw_unit_and_value',np.allclose(intervals.Flow_billion_RUB,expected_value,rtol=1e-12,atol=1e-8))
        prior=pd.read_parquet(plan['input_panel'],columns=['row_id','net_sentiment']).set_index('row_id').net_sentiment
        check('net_score_unchanged_from_frozen_input',np.array_equal(main.net_sentiment.to_numpy(),main.row_id.map(prior).to_numpy()))

    exception=json.loads((root/'exception_summary.json').read_text(encoding='utf-8'))
    declared=Counter({r['reason']:int(r['rows']) for r in exception['disjoint_reason_counts']})
    check('exception_disjoint_total',sum(declared.values())==exception['rows']
        and exception['rows']==pq.ParquetFile(plan['all_candidate_panel']).metadata.num_rows)
    observed=Counter();bad=0;bad_index=0;total=0
    # Explicit text dtypes avoid pandas C-parser mixed-type chunk concatenation
    # failures. Parse the numeric/boolean columns below with the same NA policy.
    for block in pd.read_csv(root/'exception_rows.csv.gz',chunksize=100000,
            usecols=['sentiment_exception_reason','net_sentiment','eligible_primary','index_status','new_universe_eligible'],
            dtype=str,low_memory=False,keep_default_na=False):
        observed.update(block.sentiment_exception_reason.value_counts().to_dict());total+=len(block)
        has_score=np.isfinite(pd.to_numeric(block.net_sentiment,errors='coerce'))
        is_observed=block.sentiment_exception_reason.isin(['observed_nonzero_sentiment','observed_zero_sentiment'])
        bad+=int((has_score.ne(is_observed)|(is_observed&~truth(block.eligible_primary))).sum())
        bad_index+=int((block.index_status.eq('index')&truth(block.new_universe_eligible)).sum())
    check('exception_row_export_counts',total==exception['rows'] and observed==declared)
    check('exception_missing_scores_not_zero_filled',bad==0,bad_rows=bad)
    check('exception_index_exclusion',bad_index==0,bad_rows=bad_index)
    failures=[c for c in checks if c['status']=='FAIL']
    return dict(status='FAIL' if failures else 'PASS_WITH_WARNINGS' if warnings else 'PASS',
        checked_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),checks=checks,
        pass_count=len(checks)-len(failures),fail_count=len(failures),warnings=warnings,
        inference_status_by_model=inference_status,
        independent_estimator='Fund-demeaned standardized-column OLS; independent two-way fund/company/joint sandwich, no PanelOLS import',
        tolerance_relative=args.rtol,models_checked=args.models,details=details,
        research_outputs_changed=False,protected_inputs_changed=False)


def self_test():
    rng=np.random.default_rng(191)
    n=600
    fund=np.repeat(np.arange(20),30)
    x=rng.normal(size=(n,2));noise=rng.normal(scale=.1,size=n)
    y=1.8*x[:,0]-.7*x[:,1]+fund*.2+noise
    d=pd.DataFrame(dict(fund_id=fund,company_id=np.tile(np.arange(15),40),quantity_change_pct=y,a=x[:,0],b=x[:,1]))
    r=independent_within(d,['a','b'])
    dummy=np.eye(20)[fund]
    direct=np.linalg.lstsq(np.column_stack([x,dummy]),y,rcond=None)[0][:2]
    assert np.allclose(r['beta'],direct,rtol=1e-10,atol=1e-10)
    assert r['df_resid']==578 and r['funds']==20 and r['companies']==15
    assert np.isfinite(r['covariance']).all() and np.all(np.diag(r['covariance'])>0)
    assert np.allclose(r['beta'],[1.8,-.7],atol=.025)
    return dict(status='PASS',scope='synthetic_only',checks=4)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan');p.add_argument('--root');p.add_argument('--output')
    p.add_argument('--models',nargs='+',default=['BASELINE_COMMON','NET_MACRO_COMMON'])
    p.add_argument('--rtol',type=float,default=1e-5)
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:
        print(json.dumps(self_test()));return 0
    if not args.plan or not args.output:p.error('--plan and --output are required')
    output=Path(args.output)
    if output.exists():raise FileExistsError('Independent QA report must not overwrite an existing file')
    try:report=audit(args)
    except Exception as error:
        report=dict(status='FAIL',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
            research_outputs_changed=False,protected_inputs_changed=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(clean(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],output=str(output),pass_count=report.get('pass_count'),fail_count=report.get('fail_count'))))
    return 0 if report['status'] in ['PASS','PASS_WITH_WARNINGS'] else 2


if __name__=='__main__':raise SystemExit(main())
