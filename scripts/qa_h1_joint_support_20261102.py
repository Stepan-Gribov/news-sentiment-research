"""Independent QA of the fixed six AND-support restrictions and exact caches.

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


REPLICAS=['JOINT_ROWS_30_INTERVALS_24','JOINT_ROWS_60_INTERVALS_24']


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


def holm(values):
    names=list(values)
    raw=np.array([values[name] if values[name] is not None else np.nan for name in names],float)
    valid=np.isfinite(raw)&(raw>=0)&(raw<=1)
    effective=np.where(valid,raw,1.)
    order=np.argsort(effective,kind='stable')
    adjusted=np.empty(len(names))
    adjusted[order]=np.minimum(1,np.maximum.accumulate(effective[order]*(len(names)-np.arange(len(names)))))
    return {name:float(adjusted[i]) if valid[i] else None for i,name in enumerate(names)}


def audit(args):
    if os.name!='nt':raise RuntimeError('Actual-data QA must execute on research-host')
    from scipy import stats
    planpath=Path(args.plan)
    plan=json.loads(planpath.read_text(encoding='utf-8'))
    root=Path(args.root or plan['output_root'])
    output=Path(args.output or root/'qa_independent.json')
    if output.exists():raise FileExistsError('Refusing to overwrite an existing QA version')
    if 'databaseresources' in [p.lower() for p in output.parts]:raise ValueError('Protected output')
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    coef=pd.read_csv(root/'coefficients.csv')
    prior=json.loads(Path(plan['prior_support_summary']).read_text(encoding='utf-8'))
    priorcoef=pd.read_csv(plan['prior_support_coefficients'])
    manifest=json.loads((root/'source_manifest.json').read_text(encoding='utf-8'))
    checks,warnings,details=[],[],{}

    def check(name,passed,**evidence):
        checks.append(dict(name=name,status='PASS' if bool(passed) else 'FAIL',**clean(evidence)))

    def close(a,b,atol=1e-7):
        return bool(np.isclose(float(a),float(b),rtol=args.rtol,atol=atol))

    check('frozen_plan_sha256',sha(planpath)==summary['plan_sha256'])
    check('same_frozen_specification',summary.get('specification')==plan)
    check('predeclared_independent_models',args.models==plan['independent_qa_models'])
    check('same_fund_FE_and_cluster_dimensions',plan['fund_FE'] is True and plan['time_FE'] is False and plan['cluster_time'] is False and plan['clusters']==['fund_id','company_id'])
    before={}
    for key,expected in plan['expected_source_sha256'].items():
        before[key]=sha(plan[key]);check('frozen_source:'+key,before[key]==expected,sha256=before[key])
    cache_before={}
    for path,metadata in manifest.get('used_cache_files',{}).items():
        cache_before[path]=sha(path)
        check('prior_cache_file_hash:'+path,cache_before[path]==metadata['sha256'])
    source=pd.read_parquet(plan['input_panel']).sort_values(['fund_id','end_date','security_id','interval_id'],kind='stable').reset_index(drop=True)
    check('frozen_source_N_unique_ids',len(source)==plan['expected_primary_rows'] and source.row_id.notna().all() and not source.row_id.duplicated().any())
    support=source.groupby('fund_id',sort=True,observed=True).agg(fund_rows=('row_id','size'),fund_intervals=('interval_id','nunique'),fund_end_dates=('end_date','nunique'))
    saved_support=pd.read_csv(root/'fund_support.csv',dtype={'fund_id':str}).set_index('fund_id')
    canonical=support.copy();canonical.index=canonical.index.astype(str)
    check('all_support_funds_exact',set(canonical.index)==set(saved_support.index) and not saved_support.index.duplicated().any())
    for column in canonical:
        check('support_once_on_full_source:'+column,np.array_equal(canonical[column],saved_support.reindex(canonical.index)[column]))
    masks={'REFERENCE_AUGMENTED_P99':pd.Series(True,index=source.index)}
    for rows in plan['fund_rows_thresholds']:
        for intervals in plan['fund_intervals_thresholds']:
            name=f'JOINT_ROWS_{rows}_INTERVALS_{intervals}'
            funds=support.index[support.fund_rows.ge(rows)&support.fund_intervals.ge(intervals)]
            masks[name]=source.fund_id.isin(funds)
            expected=plan['prefit_sample_counts'][name]
            check('prefit_frozen_counts:'+name,int(masks[name].sum())==expected['rows'] and len(funds)==expected['funds'])
    records={r['model']:r for r in summary['models']}
    previous_records={r['model']:r for r in prior['models']}
    check('all_six_AND_models_plus_reference',set(masks)==set(plan['planned_models'])==set(records))
    check('unique_coefficient_keys',not coef.duplicated(['model','term']).any())
    terms_all=['net_sentiment']+plan['baseline_controls']+plan['macro_controls']
    native_fields=['beta','se_native','t_native','p_native','ci95_native_low','ci95_native_high']
    joint=[name for name in masks if name!='REFERENCE_AUGMENTED_P99']
    priornames=prior['specification']['planned_models'][1:]
    check('exact_previous_eight_tests',len(priornames)==8)
    sourcehash=plan['expected_source_sha256']['input_panel']
    check('cache_same_source_helpers_and_controls',all(prior['specification']['expected_source_sha256'][k]==plan['expected_source_sha256'][k] for k in ['input_panel','base_helper','macro_runner_helper']) and prior['specification']['baseline_controls']+prior['specification']['macro_controls']==plan['baseline_controls']+plan['macro_controls'])
    datasets,inference={},{}
    inferred_origins={};seen={};new_unique=0
    for name,mask in masks.items():
        expected=source.loc[mask]
        info=summary['samples'][name]
        sample=pd.read_parquet(info['parquet'])
        record=records[name]
        key=row_hash(expected)
        check('exact_AND_sample_ids_order:'+name,sample.row_id.reset_index(drop=True).equals(expected.row_id.reset_index(drop=True)))
        check('all_original_sample_values_preserved:'+name,set(source.columns)==set(sample.columns) and equal_values(sample[list(source.columns)],expected))
        check('sample_metadata_hash_N:'+name,info['rows']==len(sample) and info['row_hash']==key==row_hash(sample))
        check('model_N_funds_companies:'+name,record['nobs']==len(sample) and record['funds']==sample.fund_id.nunique() and record['companies']==sample.company_id.nunique())
        check('native_source_row_hash:'+name,record.get('source_row_hash')==key)
        prior_match=next((oldname for oldname,oldinfo in prior['samples'].items() if oldinfo['row_hash']==key and oldinfo['rows']==len(sample)),None)
        if key in seen:
            origin='CURRENT_EXACT_SAMPLE_ALIAS'
            check('alias_origin:'+name,record.get('result_origin')==origin and record.get('exact_sample_alias_of')==seen[key])
            original=coef.loc[coef.model.eq(seen[key])].set_index('term')
            current=coef.loc[coef.model.eq(name)].set_index('term').reindex(original.index)
            check('current_alias_native_coefficients_exact:'+name,np.allclose(current[native_fields],original[native_fields],rtol=1e-12,atol=1e-10,equal_nan=True))
        elif prior_match is not None:
            origin='VALIDATED_PRIOR_EXACT_SAMPLE_CACHE'
            check('prior_cache_origin:'+name,record.get('result_origin')==origin and record.get('prior_cached_model')==prior_match)
            old_sample=pd.read_parquet(prior['samples'][prior_match]['parquet'])
            check('prior_cached_sample_exact_original_values:'+name,set(old_sample.columns)==set(source.columns) and equal_values(old_sample[list(source.columns)],expected))
            original=priorcoef.loc[priorcoef.model.eq(prior_match)].set_index('term')
            current=coef.loc[coef.model.eq(name)].set_index('term').reindex(original.index)
            check('prior_cache_native_coefficients_exact:'+name,set(coef.loc[coef.model.eq(name),'term'])==set(original.index) and np.allclose(current[native_fields],original[native_fields],rtol=1e-12,atol=1e-10,equal_nan=True))
            oldrecord=previous_records[prior_match]
            check('prior_cache_estimator_unchanged:'+name,record['retained_terms']==oldrecord['retained_terms'] and record['covariance_options']==oldrecord['covariance_options'] and record['native_df_resid']==oldrecord['native_df_resid'])
        else:
            origin='NEW_UNIQUE_SAMPLE_FIT';new_unique+=1
            check('new_unique_fit_origin:'+name,record.get('result_origin')==origin)
        seen.setdefault(key,name);inferred_origins[name]=dict(row_hash=key,origin=origin,prior_match=prior_match)
        if record.get('status')!='ESTIMATED':
            warnings.append(dict(model=name,kind='NOT_ESTIMATED',reason=record.get('reason')));continue
        options=record['covariance_options']
        check('native_fund_FE_only:'+name,record.get('included_effects')==['Entity'] and record.get('time_effects') is False and record.get('other_effects') is False and record.get('weights') is None)
        check('native_unchanged_cluster_options:'+name,all(options.get(k)==v for k,v in dict(cov_type='clustered',clusters=['fund_id','company_id'],debiased=True,auto_df=False,count_effects=True,group_debias=True,cluster_time=False).items()))
        native=coef.loc[coef.model.eq(name)].set_index('term')
        terms=record['retained_terms']
        check('native_terms_and_df:'+name,set(native.index)==set(['const']+terms) and set(terms).issubset(terms_all) and record['native_df_resid']==len(sample)-sample.fund_id.nunique()-len(terms))
        df=record['native_df_resid']
        finite=np.isfinite(native[native_fields].to_numpy(float)).all(axis=1)&native.se_native.gt(0).to_numpy()
        valid=native.loc[finite]
        t=valid.beta/valid.se_native
        critical=stats.t.ppf(.975,df)
        check('all_defined_native_t_p_CI:'+name,np.allclose(t,valid.t_native,rtol=args.rtol,atol=1e-8) and np.allclose(2*stats.t.sf(np.abs(t),df),valid.p_native,rtol=args.rtol,atol=1e-10) and np.allclose(valid.beta-critical*valid.se_native,valid.ci95_native_low,rtol=args.rtol,atol=1e-7) and np.allclose(valid.beta+critical*valid.se_native,valid.ci95_native_high,rtol=args.rtol,atol=1e-7))
        invalid=native.index[~finite].tolist()
        if invalid:warnings.append(dict(model=name,kind='UNDEFINED_COEFFICIENT_INFERENCE',terms=invalid))
        inference[name]=dict(net_inference='FINITE' if 'net_sentiment' in valid.index else 'UNDEFINED',negative_covariance_diagonal_terms=record.get('negative_covariance_diagonal_terms',[]))
        jointF=record.get('native_robust_F',{})
        if jointF and (jointF.get('stat') is None or not np.isfinite(jointF['stat']) or jointF['stat']<0):warnings.append(dict(model=name,kind='UNDEFINED_ROBUST_JOINT_F',native=jointF))
        if name in args.models:datasets[name]=sample
    check('three_new_unique_masks_only',new_unique==plan['prefit_new_unique_masks'],actual=new_unique)
    check('current_six_five_unique_masks',len({inferred_origins[name]['row_hash'] for name in joint})==5)
    check('reported_new_unique_fit_count',summary['new_unique_sample_fits']==new_unique)
    check('reported_unique_samples',summary['unique_samples']==len(seen))
    check('reported_prior_cache_reuses',summary['validated_prior_cache_reuses']==sum(record['origin']=='VALIDATED_PRIOR_EXACT_SAMPLE_CACHE' for record in inferred_origins.values()))
    for name,priorname in plan['prefit_aliases'].items():check('prefit_alias:'+name,inferred_origins[name]['prior_match']==priorname)

    def target_p(table,name):
        row=table.loc[table.model.eq(name)&table.term.eq('net_sentiment')]
        return float(row.p_native.iloc[0]) if len(row)==1 and np.isfinite(row.p_native.iloc[0]) else None

    current={name:target_p(coef,name) for name in joint}
    previous={'PREVIOUS::'+name:target_p(priorcoef,name) for name in priornames}
    within6=holm(current);append14=holm(dict(list(previous.items())+list(current.items())))
    for name in joint:
        target=coef.loc[coef.model.eq(name)&coef.term.eq('net_sentiment')].iloc[0]
        if current[name] is None:continue
        check('Holm_within_six:'+name,close(target.p_native_holm_joint6,within6[name],1e-10))
        check('Holm_conservative_fourteen:'+name,close(target.p_native_holm_append14,append14[name],1e-10))
    by_hash={}
    model_to_hash={}
    for tagged,p in list(previous.items())+list(current.items()):
        old=tagged.startswith('PREVIOUS::')
        name=tagged.split('::',1)[1] if old else tagged
        key=(prior if old else summary)['samples'][name]['row_hash']
        model_to_hash[tagged]=key
        if key in by_hash:
            priorp=by_hash[key]['p']
            check('duplicate_mask_identical_native_p:'+tagged,(p is None and priorp is None) or (p is not None and priorp is not None and close(p,priorp,1e-12)))
            by_hash[key]['aliases'].append(tagged)
        else:by_hash[key]=dict(representative=tagged,p=p,aliases=[tagged])
    unique_adjusted=holm({key:record['p'] for key,record in by_hash.items()})
    check('primary_eleven_unique_tests',len(by_hash)==plan['multiple_testing']['expected_unique_pooled_tests'],actual=len(by_hash))
    metadata=summary['multiplicity']['primary_pooled_unique']
    check('unique_family_metadata_N',metadata['family_size']==len(by_hash))
    saved_members={record['row_hash']:record for record in metadata['members']}
    check('unique_family_exact_hash_members',set(saved_members)==set(by_hash))
    for key,record in by_hash.items():
        check('unique_family_aliases:'+record['representative'],set(saved_members[key]['aliases'])==set(record['aliases']))
        if unique_adjusted[key] is not None:check('unique_family_Holm_value:'+record['representative'],close(saved_members[key]['p_native_holm'],unique_adjusted[key],1e-10))
    for name in joint:
        target=coef.loc[coef.model.eq(name)&coef.term.eq('net_sentiment')].iloc[0]
        if current[name] is not None:check('primary_unique11_adjusted_target:'+name,close(target.p_native_holm_unique_pooled,unique_adjusted[model_to_hash[name]],1e-10) and target.unique_family_size==len(by_hash))

    for name in args.models:
        if name not in datasets:check('independent_model_available:'+name,False);continue
        sample,record=datasets[name],records[name]
        result=within_ols(sample,record['retained_terms'])
        j=record['retained_terms'].index('net_sentiment')
        beta=float(result['beta'][j]);variance=float(result['covariance'][j,j])
        se=float(np.sqrt(variance)) if variance>0 else np.nan
        target=coef.loc[coef.model.eq(name)&coef.term.eq('net_sentiment')].iloc[0]
        check('independent_df:'+name,result['df_resid']==record['native_df_resid'])
        check('independent_beta:'+name,close(beta,target.beta,1e-8),independent=beta,native=target.beta)
        check('independent_within_R2:'+name,close(result['within_R2'],record['rsquared_within'],1e-10),independent=result['within_R2'],native=record['rsquared_within'])
        if np.isfinite(se):
            t=beta/se;p=2*stats.t.sf(abs(t),result['df_resid']);critical=stats.t.ppf(.975,result['df_resid'])
            check('independent_fund_company_SE:'+name,close(se,target.se_native),independent=se,native=target.se_native)
            check('independent_native_t_p_CI:'+name,close(t,target.t_native) and close(p,target.p_native,1e-10) and close(beta-critical*se,target.ci95_native_low) and close(beta+critical*se,target.ci95_native_high))
        else:
            check('undefined_target_covariance_consistent:'+name,not np.isfinite(target.se_native))
            warnings.append(dict(model=name,kind='INDEPENDENT_UNDEFINED_TARGET_VARIANCE',variance=variance))
        details[name]={key:value for key,value in result.items() if key not in ['beta','covariance']}
        details[name].update(net_beta=beta,net_SE=se,net_variance=variance)
    for key,value in before.items():check('sources_preserved_after_QA:'+key,sha(plan[key])==value)
    for path,value in cache_before.items():check('prior_cache_preserved_after_QA:'+path,sha(path)==value)
    details.update(cache_provenance_inferred=inferred_origins,unique_new_fits=new_unique,unique_pooled_tests=len(by_hash),independent_unique_family=by_hash)
    failed=sum(c['status']=='FAIL' for c in checks)
    result=dict(status='FAIL' if failed else ('PASS_WITH_WARNINGS' if warnings else 'PASS'),created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),plan_sha256=sha(planpath),pass_count=len(checks)-failed,fail_count=failed,checks=checks,warnings=warnings,native_inference_status=inference,details=details,independent_replication_models=args.models,mechanical_checks_only_models=[name for name in masks if name not in args.models],limitations=['Finite exploratory AND restrictions, not a guaranteed significance search.','Exact cached aliases are repeated estimates of the same sample, not independent supporting evidence.','Primary Holm covers 11 unique samples in the declared previous-eight/current-six suite, not all historical H1 experimentation.','The source outcome, conditional entry codes, prior P99 and support-count universe remain unchanged.'])
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],pass_count=result['pass_count'],fail_count=failed)))


def fixture():
    rng=np.random.default_rng(210)
    n=1200;fund=np.repeat(np.arange(24),50)
    net=rng.normal(size=n);control=rng.normal(size=n)
    frame=pd.DataFrame(dict(fund_id=fund,company_id=np.tile(np.arange(12),100),net_sentiment=net,control=control,quantity_change_pct=1.8*net-.3*control+fund*.4+rng.normal(scale=.05,size=n)))
    result=within_ols(frame,['net_sentiment','control'])
    assert np.allclose(result['beta'],[1.8,-.3],atol=.01)
    assert result['df_resid']==1174
    assert np.isfinite(result['covariance']).all()
    assert np.allclose(list(holm({'a':.01,'b':.04,'c':.2}).values()),[.03,.08,.2])
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


