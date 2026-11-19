"""Independent saved-sample and within-OLS QA of the finite H1 support suite.

Only the separately named QA JSON is written. No estimator/runner imports,
data changes, threshold searches, model selection, or additional H1 models.
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


REPLICAS=['REFERENCE_AUGMENTED_P99','MIN_FUND_ROWS_30','MIN_FUND_INTERVALS_24']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def row_hash(frame):
    return hashlib.sha256('\n'.join(frame.row_id.astype(str)).encode()).hexdigest()


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,(np.integer,np.bool_)):return value.item()
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    if isinstance(value,(Path,pd.Timestamp)):return str(value)
    return value


def within_ols(frame,terms):
    groups=frame.fund_id.astype(str)
    x=frame[terms].astype(float)
    y=frame.quantity_change_pct.astype(float)
    xw=(x-x.groupby(groups,sort=False).transform('mean')).to_numpy(float)
    yw=(y-y.groupby(groups,sort=False).transform('mean')).to_numpy(float)
    scale=np.linalg.norm(xw,axis=0)
    if (scale<=0).any() or not np.isfinite(scale).all():raise ValueError('Invalid within scale')
    z=xw/scale
    beta,_,rank,singular=np.linalg.lstsq(z,yw,rcond=None)
    if rank!=len(terms):raise ValueError('Rank-deficient within design')
    residual=yw-z@beta
    scores=z*residual[:,None]
    bread=np.linalg.inv(z.T@z)

    def meat(keys):
        codes,labels=pd.factorize(keys,sort=False)
        g,n=len(labels),len(codes)
        if g<2:raise ValueError('Insufficient clusters')
        totals=np.zeros((g,scores.shape[1]))
        np.add.at(totals,codes,scores)
        return totals.T@totals*(g/(g-1))*((n-1)/n),g

    fm,nf=meat(groups)
    companies=frame.company_id.astype(str)
    cm,nc=meat(companies)
    jm,nj=meat(pd.MultiIndex.from_arrays([groups,companies]))
    n,k=z.shape
    df=n-nf-k
    cov=bread@(fm+cm-jm)@bread*(n/df)
    cov=cov/scale[:,None]/scale[None,:]
    ssr,tss=float(residual@residual),float(yw@yw)
    return dict(beta=beta/scale,covariance=(cov+cov.T)/2,nobs=n,funds=nf,
        companies=nc,joint_clusters=nj,df_resid=df,rank=rank,
        condition_number=float(singular.max()/singular.min()),within_SSR=ssr,
        within_TSS=tss,within_R2=1-ssr/tss if tss else np.nan)


def equal_values(left,right):
    try:
        pd.testing.assert_frame_equal(left.reset_index(drop=True),right.reset_index(drop=True),check_exact=True,check_dtype=False)
        return True
    except AssertionError:
        return False


def audit(args):
    if os.name!='nt':raise RuntimeError('Actual-data QA must run on research-host')
    from scipy import stats
    planpath=Path(args.plan)
    plan=json.loads(planpath.read_text(encoding='utf-8'))
    root=Path(args.root or plan['output_root'])
    output=Path(args.output or root/'qa_independent.json')
    if output.exists():raise FileExistsError('Refusing to overwrite an existing QA report')
    if 'databaseresources' in [p.lower() for p in output.parts]:raise ValueError('Protected output')
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    coefficients=pd.read_csv(root/'coefficients.csv')
    checks,warnings,details=[],[],{}

    def check(name,passed,**evidence):
        checks.append(dict(name=name,status='PASS' if bool(passed) else 'FAIL',**clean(evidence)))

    def close(a,b,atol=1e-7):
        return bool(np.isclose(float(a),float(b),rtol=args.rtol,atol=atol))

    check('frozen_plan_hash',sha(planpath)==summary['plan_sha256'])
    check('frozen_specification',summary.get('specification')==plan)
    check('predeclared_independent_replication_models',args.models==plan['independent_qa_models'])
    check('fund_FE_no_time_clustering',plan['fund_FE'] is True and plan['time_FE'] is False and plan['cluster_time'] is False and plan['clusters']==['fund_id','company_id'])
    source_hashes={}
    for key,expected in plan['expected_source_sha256'].items():
        actual=sha(plan[key]);source_hashes[key]=actual
        check('frozen_source:'+key,actual==expected,sha256=actual)
    source=pd.read_parquet(plan['input_panel']).sort_values(['fund_id','end_date','security_id','interval_id'],kind='stable').reset_index(drop=True)
    check('source_unique_rows',source.row_id.notna().all() and not source.row_id.duplicated().any())
    check('source_expected_N',len(source)==plan.get('expected_primary_rows',221899))
    support=source.groupby('fund_id',observed=True,sort=True).agg(fund_rows=('row_id','size'),fund_intervals=('interval_id','nunique'),fund_end_dates=('end_date','nunique'),fund_securities=('security_id','nunique'),fund_companies=('company_id','nunique'))
    saved_support=pd.read_csv(root/'fund_support.csv',dtype={'fund_id':str}).set_index('fund_id')
    lookup=support.copy();lookup.index=lookup.index.astype(str)
    check('support_all_funds_exact',set(saved_support.index)==set(lookup.index) and not saved_support.index.duplicated().any())
    for column in support.columns:
        check('support_counted_once:'+column,np.array_equal(saved_support.reindex(lookup.index)[column],lookup[column]))
    # Counts are computed on the frozen full panel, independently per restriction.
    masks={'REFERENCE_AUGMENTED_P99':pd.Series(True,index=source.index)}
    for name,column,threshold in [('MIN_FUND_ROWS_30','fund_rows',30),('MIN_FUND_ROWS_60','fund_rows',60),('MIN_FUND_INTERVALS_12','fund_intervals',12),('MIN_FUND_INTERVALS_24','fund_intervals',24),('MIN_FUND_INTERVALS_36','fund_intervals',36)]:
        masks[name]=source.fund_id.isin(support.index[support[column].ge(threshold)])
    masks['NEWS_MIN_3']=pd.to_numeric(source.n_relevant,errors='coerce').ge(3)
    masks['NEWS_MIN_5']=pd.to_numeric(source.n_relevant,errors='coerce').ge(5)
    gross=source[['opening_weight','closing_weight','closing_HHI']].apply(pd.to_numeric,errors='coerce')
    masks['BOUNDED_GROSS_CONTROLS']=pd.Series(np.isfinite(gross).all(axis=1),index=source.index)&gross.ge(0).all(axis=1)&gross.le(1.001).all(axis=1)
    check('all_nine_planned_models',set(plan['planned_models'])==set(masks) and len(plan['planned_models'])==9)
    records={m['model']:m for m in summary['models']}
    check('all_nine_model_records',set(records)==set(masks))
    check('unique_native_coefficient_keys',not coefficients.duplicated(['model','term']).any())
    removed=pd.read_csv(root/'removed_funds.csv',dtype={'fund_id':str})
    singleton_funds=set(support.index[support.fund_rows.eq(1)])
    terms_all=['net_sentiment']+plan['baseline_controls']+plan['macro_controls']
    within=source[terms_all+['quantity_change_pct']].astype(float)
    within=within-within.groupby(source.fund_id,sort=False).transform('mean')
    singleton_values=within.loc[source.fund_id.isin(singleton_funds)].to_numpy(float)
    check('single_row_funds_zero_within_X_and_Y',not singleton_values.size or np.max(np.abs(singleton_values))==0, funds=len(singleton_funds),rows=len(singleton_values))
    data,inference={},{}
    for name,mask in masks.items():
        expected=source.loc[mask]
        info=summary['samples'][name]
        sample=pd.read_parquet(info['parquet'])
        record=records[name]
        check('sample_exact_mask_ids_and_order:'+name,sample.row_id.reset_index(drop=True).equals(expected.row_id.reset_index(drop=True)))
        check('sample_all_original_values_unchanged:'+name,set(sample.columns)==set(source.columns) and equal_values(sample[list(source.columns)],expected))
        check('sample_hash_N:'+name,row_hash(sample)==info['row_hash'] and len(sample)==info['rows'])
        check('model_N_funds_companies:'+name,record['nobs']==len(sample) and record['funds']==sample.fund_id.nunique() and record['companies']==sample.company_id.nunique())
        missing_funds=set(source.fund_id.astype(str))-set(sample.fund_id.astype(str))
        saved_missing=removed.loc[removed.model.eq(name),'fund_id']
        check('removed_funds_accounting:'+name,set(saved_missing)==missing_funds and not saved_missing.duplicated().any())
        if record['status']!='ESTIMATED':
            warnings.append(dict(model=name,kind='NOT_ESTIMATED',reason=record.get('reason')))
            continue
        check('same_fund_FE_only:'+name,record.get('included_effects')==['Entity'] and record.get('time_effects') is False)
        options=record['covariance_options']
        check('same_two_way_cluster_options:'+name,options['clusters']==['fund_id','company_id'] and options.get('cluster_time') is False and all(options.get(k)==v for k,v in plan['covariance_options'].items()))
        check('native_unique_interval_counts:'+name,record['distinct_fund_interval_pairs']==len(sample[['fund_id','interval_id']].drop_duplicates()) and record['distinct_fund_end_date_pairs']==len(sample[['fund_id','end_date']].drop_duplicates()))
        native=coefficients.loc[coefficients.model.eq(name)].set_index('term')
        terms=record['retained_terms']
        check('unchanged_declared_terms:'+name,set(terms).issubset(terms_all) and set(native.index)==set(['const']+terms))
        target=native.loc['net_sentiment']
        fields=['beta','se_native','t_native','p_native','ci95_native_low','ci95_native_high']
        valid=np.isfinite(target[fields].to_numpy(float)).all() and target.se_native>0
        inference[name]=dict(net_inference='FINITE' if valid else 'UNDEFINED',target=dict(target[fields]))
        if valid:
            t=target.beta/target.se_native
            df=min(sample.fund_id.nunique(),sample.company_id.nunique())-1
            check('native_net_bounds:'+name,0<=target.p_native<=1 and target.ci95_native_low<=target.beta<=target.ci95_native_high)
            check('conservative_positive_H1_p:'+name,record['conservative_cluster_df']==df and close(stats.t.sf(t,df),target.p_H1_positive_one_sided,1e-10),df=df)
        else:warnings.append(dict(model=name,kind='UNDEFINED_TARGET_INFERENCE'))
        invalid=native.index[~np.isfinite(native[fields].to_numpy(float)).all(axis=1)].tolist()
        if invalid:warnings.append(dict(model=name,kind='UNDEFINED_COEFFICIENT_INFERENCE',terms=invalid))
        joint=record.get('native_robust_F',{})
        if joint and (joint.get('stat') is None or not np.isfinite(joint['stat']) or joint['stat']<0):warnings.append(dict(model=name,kind='UNDEFINED_ROBUST_JOINT_F',native=joint))
        if name in args.models:data[name]=sample

    # Both multiplicity corrections use all eight new restrictions; no reference.
    family=[name for name in plan['planned_models'] if name!='REFERENCE_AUGMENTED_P99']
    for field,outfield in [('p_native','p_native_holm_support_family'),('p_H1_positive_one_sided','p_H1_positive_one_sided_holm_support_family')]:
        rows=coefficients.loc[coefficients.model.isin(family)&coefficients.term.eq('net_sentiment')].set_index('model').reindex(family)
        usable=np.array([inference.get(name,{}).get('net_inference')=='FINITE' for name in family])
        p=rows[field].to_numpy(float)
        usable &= np.isfinite(p)&(p>=0)&(p<=1)
        effective=np.where(usable,p,1.)
        order=np.argsort(effective,kind='stable')
        adjusted=np.empty(len(family))
        adjusted[order]=np.minimum(1,np.maximum.accumulate(effective[order]*(len(family)-np.arange(len(family)))))
        check('Holm_all_eight_new_tests:'+field,np.allclose(rows.loc[usable,outfield],adjusted[usable],rtol=args.rtol,atol=1e-10))
        if (~usable).any():warnings.append(dict(kind='HOLM_UNDEFINED_TARGETS',field=field,models=list(np.array(family)[~usable])))

    for name in args.models:
        if name not in data:check('independent_model_available:'+name,False);continue
        sample,record=data[name],records[name]
        result=within_ols(sample,record['retained_terms'])
        j=record['retained_terms'].index('net_sentiment')
        beta=float(result['beta'][j]);variance=float(result['covariance'][j,j])
        se=np.sqrt(variance) if variance>0 else np.nan
        target=coefficients.loc[coefficients.model.eq(name)&coefficients.term.eq('net_sentiment')].iloc[0]
        check('independent_df:'+name,result['df_resid']==record['native_df_resid'])
        check('independent_net_beta:'+name,close(beta,target.beta,1e-8),independent=beta,native=target.beta)
        check('independent_within_R2:'+name,close(result['within_R2'],record['rsquared_within'],1e-10),independent=result['within_R2'],native=record['rsquared_within'])
        if np.isfinite(se):
            t=beta/se;p=2*stats.t.sf(abs(t),result['df_resid']);critical=stats.t.ppf(.975,result['df_resid'])
            check('independent_net_cluster_SE:'+name,close(se,target.se_native),independent=se,native=target.se_native)
            check('independent_net_t_p_CI:'+name,close(t,target.t_native) and close(p,target.p_native,1e-10) and close(beta-critical*se,target.ci95_native_low) and close(beta+critical*se,target.ci95_native_high))
        else:
            check('independent_undefined_target_consistent:'+name,not np.isfinite(target.se_native))
            warnings.append(dict(model=name,kind='INDEPENDENT_UNDEFINED_TARGET_VARIANCE',variance=variance))
        details[name]={k:v for k,v in result.items() if k not in ['beta','covariance']}
        details[name].update(net_beta=beta,net_SE=se,net_variance=variance)

    # FWL checks concern only design variation and never select an outcome model.
    fwl=json.loads((root/'FWL_information.json').read_text(encoding='utf-8'))
    if fwl.get('status')=='COMPUTED':
        control_terms=[term for term in records['REFERENCE_AUGMENTED_P99']['retained_terms'] if term!='net_sentiment']
        net=within.net_sentiment.to_numpy(float)
        x=within[control_terms].to_numpy(float)
        scale=np.linalg.norm(x,axis=0)
        z=x/scale
        projected=np.linalg.lstsq(z,net,rcond=None)[0]
        residual=net-z@projected
        total=float(residual@residual)
        check('FWL_total_residual_Net_SS',close(total,fwl['total_residual_net_SS']),independent=total,native=fwl['total_residual_net_SS'])
        check('FWL_within_Net_SS',close(float(net@net),fwl['within_net_SS']))
        check('FWL_max_row_information_share',close(float(np.max(residual**2)/total),fwl['max_row_information_share'],1e-12))
        for dimension,filename in fwl['table_files'].items():
            table=pd.read_csv(root/filename)
            check('FWL_table_totals:'+dimension,table.rows.sum()==len(source) and close(table.information_sum.sum(),total) and close(table.information_share.sum(),1.,1e-10))
    else:warnings.append(dict(kind='FWL_INFORMATION_NOT_COMPUTED',reason=fwl.get('reason')))
    for key,expected in source_hashes.items():check('source_preserved_after_QA:'+key,sha(plan[key])==expected)
    details['support_masks']={name:dict(rows=int(mask.sum()),funds=int(source.loc[mask,'fund_id'].nunique()),companies=int(source.loc[mask,'company_id'].nunique())) for name,mask in masks.items()}
    details['single_row_funds']=len(singleton_funds)
    failed=sum(c['status']=='FAIL' for c in checks)
    result=dict(status='FAIL' if failed else ('PASS_WITH_WARNINGS' if warnings else 'PASS'),created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),plan_sha256=sha(planpath),pass_count=len(checks)-failed,fail_count=failed,checks=checks,warnings=warnings,native_inference_status=inference,details=details,independent_replication_models=args.models,mechanical_checks_only_models=[name for name in masks if name not in args.models],limitations=['Finite exploratory restrictions after earlier results, not a guarantee of significance or causal identification.','Support is counted in the already selected covered-news macro-complete panel; intervals are not necessarily months.','One-row funds have zero within information; one-interval funds with several securities may contribute within-fund variation.','Bounded gross controls are a declared subset, not proof excluded values are data errors.','Conditional entries, outcome and the old common P99 are inherited unchanged.'])
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],pass_count=result['pass_count'],fail_count=failed)))


def fixture():
    rng=np.random.default_rng(412)
    n=1200
    fund=np.repeat(np.arange(24),50)
    net=rng.normal(size=n);control=rng.normal(size=n)
    frame=pd.DataFrame(dict(fund_id=fund,company_id=np.tile(np.arange(12),100),net_sentiment=net,control=control,quantity_change_pct=1.3*net-.2*control+fund*.7+rng.normal(scale=.05,size=n)))
    result=within_ols(frame,['net_sentiment','control'])
    assert np.allclose(result['beta'],[1.3,-.2],atol=.01)
    assert result['df_resid']==1200-24-2
    assert result['within_R2']>.99
    assert np.isfinite(result['covariance']).all()
    print(json.dumps(dict(status='PASS',scope='synthetic_only',checks=4)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan')
    parser.add_argument('--root')
    parser.add_argument('--output')
    parser.add_argument('--models',nargs='+',default=REPLICAS)
    parser.add_argument('--rtol',type=float,default=1e-5)
    parser.add_argument('--fixture-only',action='store_true')
    args=parser.parse_args()
    fixture() if args.fixture_only else audit(args)
