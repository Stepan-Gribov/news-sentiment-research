"""Independent QA of frozen fund-focus panels, group support and shared signals.

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


REPLICAS=['FOCUS_BOND_BASELINE','FOCUS_BOND_ROWS_60_INTERVALS_24']


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
    planpath=Path(args.plan);plan=json.loads(planpath.read_text(encoding='utf-8'))
    root=Path(args.root or plan['output_root']);output=Path(args.output or root/'qa_independent.json')
    if output.exists():raise FileExistsError('Refusing to overwrite an existing QA version')
    if 'databaseresources' in [p.lower()for p in output.parts]:raise ValueError('Protected output')
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    coef=pd.read_csv(root/'coefficients.csv')
    manifest=json.loads((root/'source_manifest.json').read_text(encoding='utf-8'))
    checks,warnings,details=[],[],{}

    def check(name,passed,**evidence):
        checks.append(dict(name=name,status='PASS'if bool(passed)else'FAIL',**clean(evidence)))

    def close(a,b,atol=1e-7):
        return bool(np.isclose(float(a),float(b),rtol=args.rtol,atol=atol))

    check('frozen_plan_hash',sha(planpath)==summary['plan_sha256'])
    check('same_frozen_specification',summary.get('specification')==plan)
    check('predeclared_independent_models',args.models==plan['independent_qa_models'])
    check('fund_FE_no_time_clustering',plan['fund_FE']is True and plan['time_FE']is False and plan['cluster_time']is False and plan['clusters']==['fund_id','company_id'])
    before={}
    for key,expected in plan['expected_source_sha256'].items():
        before[key]=sha(plan[key]);check('frozen_source:'+key,before[key]==expected,sha256=before[key])
    cache_before={}
    for path,info in manifest.get('used_cache_files',{}).items():
        cache_before[path]=sha(path);check('cache_file_hash:'+path,cache_before[path]==info['sha256'])
    source=pd.read_parquet(plan['input_panel']).sort_values(['fund_id','end_date','security_id','interval_id'],kind='stable').reset_index(drop=True)
    check('source_N_unique_rows',len(source)==plan['expected_primary_rows'] and source.row_id.notna().all()and not source.row_id.duplicated().any())
    terms_all=['net_sentiment']+plan['baseline_controls']+plan['macro_controls']
    support=source.groupby('fund_id',sort=True,observed=True).agg(fund_rows=('row_id','size'),fund_intervals=('interval_id','nunique'),fund_end_dates=('end_date','nunique'))
    group_support=source.groupby(['fund_focus','fund_id'],sort=True,dropna=False,observed=True).agg(fund_rows=('row_id','size'),fund_intervals=('interval_id','nunique'),fund_end_dates=('end_date','nunique'))
    saved=pd.read_csv(root/'fund_support.csv',dtype={'fund_id':str}).set_index('fund_id')
    expected=support.copy();expected.index=expected.index.astype(str)
    check('full_support_funds',set(saved.index)==set(expected.index)and not saved.index.duplicated().any())
    for field in expected:check('full_support_descriptive:'+field,np.array_equal(saved.reindex(expected.index)[field],expected[field]))
    saved_group=pd.read_csv(root/'focus_fund_support.csv',dtype={'fund_id':str}).set_index(['fund_focus','fund_id'])
    expected=group_support.reset_index();expected.fund_id=expected.fund_id.astype(str);expected=expected.set_index(['fund_focus','fund_id'])
    check('unfiltered_group_support_keys',set(saved_group.index)==set(expected.index)and not saved_group.index.duplicated().any())
    for field in expected:check('unfiltered_group_support_once:'+field,np.array_equal(saved_group.reindex(expected.index)[field],expected[field]))
    check('support_scope_declared_group_specific',summary.get('support_counted_once_on_unfiltered_focus_baseline')is True and summary.get('full_source_support_descriptive_only')is True and summary.get('iterative_support_recounting')is False)
    masks={'REFERENCE_AUGMENTED_P99':pd.Series(True,index=source.index)}
    rules={'REFERENCE_AUGMENTED_P99':(None,None,None)}
    for group in plan['focus_groups']:
        belongs=source.fund_focus.eq(group)
        name='FOCUS_'+group.upper()+'_BASELINE';masks[name]=belongs;rules[name]=(group,None,None)
        support_group=group_support.loc[group]
        for minimum_rows in plan['fund_rows_thresholds']:
            for minimum_intervals in plan['fund_intervals_thresholds']:
                name=f'FOCUS_{group.upper()}_ROWS_{minimum_rows}_INTERVALS_{minimum_intervals}'
                funds=support_group.index[support_group.fund_rows.ge(minimum_rows)&support_group.fund_intervals.ge(minimum_intervals)]
                masks[name]=belongs&source.fund_id.isin(funds);rules[name]=(group,minimum_rows,minimum_intervals)
    records={r['model']:r for r in summary['models']}
    check('all_thirty_six_labels',len(masks)==36 and set(masks)==set(plan['planned_models'])==set(records))
    check('unique_native_coefficient_keys',not coef.duplicated(['model','term']).any())
    # Score cardinality: one company-window score may appear in many asset rows.
    signal_json=json.loads((root/'news_signal_level_audit.json').read_text(encoding='utf-8'))
    check('every_row_has_company_window_key',source.company_period_id.notna().all())
    signal_groups=source.groupby('company_period_id',sort=False,observed=True)
    uniqueness=signal_groups[['company_id','start_date','end_date','net_sentiment','n_relevant']].nunique(dropna=False)
    check('company_period_one_company_window_score_count',uniqueness.eq(1).all().all())
    reverse=source.groupby(['company_id','start_date','end_date'],dropna=False,observed=True).company_period_id.nunique()
    check('reverse_company_window_single_key',reverse.eq(1).all())
    check('saved_signal_unique_checks',all(signal_json['company_period_unique_checks'].values())and signal_json['company_period_unique_checks'].get('reverse_company_window_key')is True)
    check('focus_fixed_within_opening_fund_interval',source.groupby(['fund_id','interval_id'],observed=True).fund_focus.nunique().le(1).all())
    check('signal_finite_observed_relevant_only',np.isfinite(source.net_sentiment).all()and source.n_relevant.gt(0).all()and source.eligible_primary.eq(True).all())
    signal=signal_groups.agg(rows=('row_id','size'),funds=('fund_id','nunique'),securities=('security_id','nunique'))
    issuer_positions=source.groupby(['fund_id','company_id','interval_id'],observed=True).security_id.nunique()
    signal_counts=dict(regression_rows=len(source),unique_company_period_signals=len(signal),unique_companies=int(source.company_id.nunique()),unique_securities=int(source.security_id.nunique()),company_periods_repeated_across_funds=int(signal.funds.gt(1).sum()),company_periods_repeated_across_securities=int(signal.securities.gt(1).sum()),fund_company_intervals_with_multiple_securities=int(issuer_positions.gt(1).sum()),max_securities_in_fund_company_interval=int(issuer_positions.max()),funds_with_multiple_focus_labels_over_time=int(source.groupby('fund_id',observed=True).fund_focus.nunique().gt(1).sum()))
    for key,value in signal_counts.items():check('saved_signal_count:'+key,signal_json[key]==value)
    saved_signal=pd.read_csv(root/'company_period_signal_counts.csv',dtype={'company_period_id':str}).set_index('company_period_id')
    check('signal_count_table_keys',set(saved_signal.index)==set(signal.index.astype(str)))
    for field in ['rows','funds','securities']:check('signal_count_table:'+field,np.array_equal(saved_signal.reindex(signal.index.astype(str))[field],signal[field]))
    count_fields=['n_positive','n_negative','n_neutral']
    if set(count_fields).issubset(source):
        counts=source[count_fields].apply(pd.to_numeric,errors='coerce')
        check('stored_relevant_count_partition',np.isfinite(counts).all().all()and counts.ge(0).all().all()and np.allclose(counts.sum(axis=1),source.n_relevant,rtol=0,atol=0))
        computed=(counts.n_positive-counts.n_negative)/source.n_relevant
        check('stored_raw_net_formula',np.allclose(computed,source.net_sentiment,rtol=1e-10,atol=1e-12))
        check('saved_net_formula_audit',signal_json['net_formula_audit']['status']=='PASS')
    else:
        details['net_formula_from_counts']='COUNTS_NOT_IN_SOURCE; checked shared score and frozen source, not reconstructed original article labels'
        check('saved_missing_count_disclosure',signal_json['net_formula_audit']['status']=='COUNTS_NOT_IN_SOURCE')
    example=pd.read_csv(root/'issuer_period_example.csv',dtype={'row_id':str,'company_period_id':str})
    example_source=source.loc[source.company_period_id.astype(str).eq(signal_json['example_company_period_id'])]
    check('issuer_period_example_all_source_rows',len(example)==signal_json['example_rows']==len(example_source)and set(example.row_id)==set(example_source.row_id))
    for field in ['net_sentiment','n_relevant','quantity_open','quantity_close','quantity_change_pct']:
        check('issuer_period_example_values:'+field,np.allclose(example.set_index('row_id').reindex(example_source.row_id)[field],example_source[field],rtol=1e-12,atol=1e-8,equal_nan=True))
    coverage=pd.read_csv(root/'type_coverage.csv').set_index('fund_focus')
    for group,part in source.groupby('fund_focus',observed=True,dropna=False):
        row=coverage.loc[group]
        check('type_coverage:'+str(group),row.rows==len(part)and row.funds==part.fund_id.nunique()and row.companies==part.company_id.nunique()and row.securities==part.security_id.nunique()and row.company_period_signals==part.company_period_id.nunique()and row.fund_intervals==len(part[['fund_id','interval_id']].drop_duplicates()))
    named=source.fund_focus.isin(plan['focus_groups'])
    details['coverage']=dict(named_focus_rows=int(named.sum()),other_or_unclassified_rows=int((~named).sum()),group_row_counts=source.fund_focus.value_counts(dropna=False).to_dict(),fund_counts_not_additive_across_focus=True)
    # Cached baseline fits remain independently auditable against current Y/X.
    caches=[]
    for family in ['entries','joint']:
        old=json.loads(Path(plan[family+'_summary']).read_text(encoding='utf-8'))
        oldcoef=pd.read_csv(plan[family+'_coefficients'])
        caches.append((family,old,oldcoef))
    datasets,inference,origins,seen={}, {}, {}, {}
    native_fields=['beta','se_native','t_native','p_native','ci95_native_low','ci95_native_high']
    new_estimated=0;uncached_unestimated=0;prior_reuses=0
    for name,mask in masks.items():
        expected=source.loc[mask];meta=summary['samples'][name];record=records[name]
        sample=pd.read_parquet(meta['parquet']);key=row_hash(expected)
        group,minimum_rows,minimum_intervals=rules[name]
        check('exact_group_AND_mask_ids_order:'+name,sample.row_id.reset_index(drop=True).equals(expected.row_id.reset_index(drop=True)))
        check('unchanged_all_source_columns:'+name,set(sample.columns)==set(source.columns)and equal_values(sample[list(source.columns)],expected))
        check('sample_hash_N:'+name,meta['rows']==len(sample)and meta['row_hash']==key==row_hash(sample))
        check('model_N_funds_companies:'+name,record['nobs']==len(sample)and record['funds']==sample.fund_id.nunique()and record['companies']==sample.company_id.nunique())
        if minimum_rows is not None and len(sample):
            actual=sample.groupby('fund_id',observed=True).agg(rows=('row_id','size'),intervals=('interval_id','nunique'))
            check('actual_group_minimums:'+name,actual.rows.ge(minimum_rows).all()and actual.intervals.ge(minimum_intervals).all())
        current=coef.loc[coef.model.eq(name)].set_index('term')
        matched=None
        for family,old,oldcoef in caches:
            for oldrecord in old['models']:
                oldname=oldrecord['model'];oldmeta=old.get('samples',{}).get(oldname,{})
                if oldrecord.get('status')=='ESTIMATED'and oldmeta.get('row_hash')==key and oldmeta.get('rows')==len(sample):
                    matched=(family,oldname,oldrecord,oldmeta,oldcoef);break
            if matched:break
        if key in seen:
            origin='CURRENT_EXACT_SAMPLE_ALIAS'
            check('exact_current_alias_origin:'+name,record['result_origin']==origin and record.get('exact_sample_alias_of')==seen[key])
            original=coef.loc[coef.model.eq(seen[key])].set_index('term')
            check('current_alias_native_values:'+name,set(current.index)==set(original.index)and np.allclose(current.reindex(original.index)[native_fields],original[native_fields],rtol=1e-12,atol=1e-10,equal_nan=True))
        elif matched:
            origin='VALIDATED_PRIOR_EXACT_SAMPLE_CACHE';prior_reuses+=1
            family,oldname,oldrecord,oldmeta,oldcoef=matched
            proof=record.get('cache_provenance')or{}
            check('prior_cache_provenance:'+name,record['result_origin']==origin and proof.get('cache_family')==family and proof.get('source_model')==oldname and proof.get('row_hash')==key)
            oldsample=pd.read_parquet(oldmeta['parquet'])
            mandatory=['row_id','quantity_change_pct','quantity_open','quantity_close']+terms_all
            check('prior_cache_exact_regression_input_values:'+name,set(mandatory).issubset(oldsample)and equal_values(oldsample[mandatory],expected[mandatory]))
            original=oldcoef.loc[oldcoef.model.eq(oldname)].set_index('term')
            check('prior_cache_native_coefficients:'+name,set(current.index)==set(original.index)and np.allclose(current.reindex(original.index)[native_fields],original[native_fields],rtol=1e-12,atol=1e-10,equal_nan=True))
            check('prior_cache_native_estimator:'+name,record['retained_terms']==oldrecord['retained_terms']and record['covariance_options']==oldrecord['covariance_options']and record['native_df_resid']==oldrecord['native_df_resid'])
        elif record['status']=='ESTIMATED':
            origin='NEW_UNIQUE_SAMPLE_FIT';new_estimated+=1
            check('new_unique_estimate_origin:'+name,record['result_origin']==origin)
        else:
            origin='NOT_ESTIMABLE';uncached_unestimated+=1
            check('not_estimable_origin:'+name,record['result_origin']==origin)
        seen.setdefault(key,name);origins[name]=dict(row_hash=key,origin=origin,status=record['status'])
        if record['status']!='ESTIMATED':
            check('not_estimated_has_no_coefficients:'+name,len(current)==0)
            inference[name]=dict(net_inference='NOT_ESTIMATED',reason=record.get('reason'))
            warnings.append(dict(model=name,kind='NOT_ESTIMATED',reason=record.get('reason')));continue
        if min(record['funds'],record['companies'])<30:
            warnings.append(dict(model=name,kind='LOW_CLUSTER_COUNT_DIAGNOSTIC',funds=record['funds'],companies=record['companies'],note='Descriptive flag below 30 clusters, not an automatic invalidity or filtering rule. Native inference retained; small-sample caution applies.'))
        check('native_source_hash:'+name,record.get('source_row_hash')==key)
        check('same_fund_FE_only:'+name,record.get('included_effects')==['Entity']and record.get('time_effects')is False and record.get('other_effects')is False and record.get('weights')is None)
        options=record['covariance_options']
        check('same_two_way_covariance_options:'+name,all(options.get(k)==v for k,v in dict(cov_type='clustered',clusters=['fund_id','company_id'],debiased=True,auto_df=False,count_effects=True,group_debias=True,cluster_time=False).items()))
        terms=record['retained_terms'];df=record['native_df_resid']
        check('native_terms_and_df:'+name,set(current.index)==set(['const']+terms)and set(terms).issubset(terms_all)and df==len(sample)-sample.fund_id.nunique()-len(terms))
        valid=np.isfinite(current[native_fields].to_numpy(float)).all(axis=1)&current.se_native.gt(0).to_numpy()
        defined=current.loc[valid];t=defined.beta/defined.se_native;critical=stats.t.ppf(.975,df)
        check('all_defined_native_t_p_CI:'+name,np.allclose(t,defined.t_native,rtol=args.rtol,atol=1e-7)and np.allclose(2*stats.t.sf(np.abs(t),df),defined.p_native,rtol=args.rtol,atol=1e-10)and np.allclose(defined.beta-critical*defined.se_native,defined.ci95_native_low,rtol=args.rtol,atol=1e-7)and np.allclose(defined.beta+critical*defined.se_native,defined.ci95_native_high,rtol=args.rtol,atol=1e-7))
        invalid=current.index[~valid].tolist()
        if invalid:warnings.append(dict(model=name,kind='UNDEFINED_COEFFICIENT_INFERENCE',terms=invalid))
        inference[name]=dict(net_inference='FINITE'if'net_sentiment'in defined.index else'UNDEFINED',negative_covariance_diagonal_terms=record.get('negative_covariance_diagonal_terms',[]))
        check('declared_target_inference_status:'+name,record.get('net_inference_status')==inference[name]['net_inference'])
        if 'net_sentiment'in defined.index:
            target=current.loc['net_sentiment'];cluster_df=min(record['funds'],record['companies'])-1
            check('existing_conservative_cluster_t_p:'+name,record['conservative_cluster_df']==cluster_df and close(2*stats.t.sf(abs(target.beta/target.se_native),cluster_df),target.p_cluster_t_two_sided,1e-10))
        joint=record.get('native_robust_F',{})
        if joint and(joint.get('stat')is None or not np.isfinite(joint['stat'])or joint['stat']<0):warnings.append(dict(model=name,kind='UNDEFINED_ROBUST_JOINT_F',native=joint))
        if name in args.models:datasets[name]=sample
    check('unique_mask_count',summary['unique_samples']==len(seen))
    check('prior_cache_reuse_count',summary['validated_prior_cache_reuses']==prior_reuses)
    check('new_fit_attempt_count_bounds',new_estimated<=summary['new_unique_sample_fits']<=new_estimated+uncached_unestimated)
    # Retain every distinct planned mask, including an empty/unestimated slot.
    family={}
    for name in masks:
        if name=='REFERENCE_AUGMENTED_P99':continue
        target=coef.loc[coef.model.eq(name)&coef.term.eq('net_sentiment')]
        p=float(target.p_native.iloc[0])if len(target)==1 and np.isfinite(target.p_native.iloc[0])and np.isfinite(target.se_native.iloc[0])and target.se_native.iloc[0]>0 else None
        key=origins[name]['row_hash']
        if key in family:
            oldp=family[key]['p']
            check('identical_focus_mask_same_p:'+name,(p is None and oldp is None)or(p is not None and oldp is not None and close(p,oldp,1e-12)))
            family[key]['aliases'].append(name)
        else:family[key]=dict(representative=name,p=p,aliases=[name])
    adjusted=holm({key:record['p']for key,record in family.items()})
    metadata=summary['multiplicity'];members={r['row_hash']:r for r in metadata['members']}
    check('Holm_unique_current_family_size',metadata['family_size']==len(family)and set(members)==set(family))
    check('Holm_reserved_undefined_slots',metadata.get('empty_or_unestimable_unique_masks_keep_p1_slots')is True)
    for key,record in family.items():
        check('Holm_all_aliases:'+record['representative'],set(members[key]['aliases'])==set(record['aliases']))
        if adjusted[key]is None:check('undefined_adjusted_p_not_valid_one:'+record['representative'],members[key]['p_native_holm']is None)
        else:check('Holm_unique_mask_value:'+record['representative'],close(members[key]['p_native_holm'],adjusted[key],1e-10))
        for name in record['aliases']:
            target=coef.loc[coef.model.eq(name)&coef.term.eq('net_sentiment')]
            if len(target)==1 and adjusted[key]is not None:check('Holm_displayed_target:'+name,close(target.p_native_holm_focus_unique.iloc[0],adjusted[key],1e-10)and target.unique_focus_family_size.iloc[0]==len(family))
    for name in args.models:
        if name not in datasets:
            details[name]=dict(numerical_replication_status='NOT_AVAILABLE',reason=records[name].get('reason'));warnings.append(dict(model=name,kind='PRESPECIFIED_NUMERICAL_REPLICA_NOT_ESTIMATED'));continue
        sample,record=datasets[name],records[name];result=within_ols(sample,record['retained_terms'])
        j=record['retained_terms'].index('net_sentiment');beta=float(result['beta'][j]);variance=float(result['covariance'][j,j]);se=float(np.sqrt(variance))if variance>0 else np.nan
        target=coef.loc[coef.model.eq(name)&coef.term.eq('net_sentiment')].iloc[0]
        start=len(checks)
        check('independent_df:'+name,result['df_resid']==record['native_df_resid'])
        check('independent_beta:'+name,close(beta,target.beta,1e-8),independent=beta,native=target.beta)
        check('independent_within_R2:'+name,close(result['within_R2'],record['rsquared_within'],1e-10),independent=result['within_R2'],native=record['rsquared_within'])
        if np.isfinite(se):
            t=beta/se;p=2*stats.t.sf(abs(t),result['df_resid']);critical=stats.t.ppf(.975,result['df_resid'])
            check('independent_fund_company_SE:'+name,close(se,target.se_native),independent=se,native=target.se_native)
            check('independent_native_t_p_CI:'+name,close(t,target.t_native)and close(p,target.p_native,1e-10)and close(beta-critical*se,target.ci95_native_low)and close(beta+critical*se,target.ci95_native_high))
        else:
            check('independent_undefined_variance_consistent:'+name,not np.isfinite(target.se_native));warnings.append(dict(model=name,kind='INDEPENDENT_UNDEFINED_TARGET_VARIANCE',variance=variance))
        details[name]={key:value for key,value in result.items()if key not in['beta','covariance']}
        details[name].update(net_beta=beta,net_SE=se,net_variance=variance,numerical_replication_status='PASS'if all(r['status']=='PASS'for r in checks[start:])else'FAIL')
    for key,value in before.items():check('frozen_sources_preserved_after_QA:'+key,sha(plan[key])==value)
    for path,value in cache_before.items():check('cached_files_preserved_after_QA:'+path,sha(path)==value)
    details.update(shared_signal=signal_counts,origins=origins,current_unique_focus_masks=len(family),new_unique_estimated_samples=new_estimated,unestimated_uncached_unique_samples=uncached_unestimated)
    failed=sum(r['status']=='FAIL'for r in checks)
    result=dict(status='FAIL'if failed else('PASS_WITH_WARNINGS'if warnings else'PASS'),created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),plan_sha256=sha(planpath),pass_count=len(checks)-failed,fail_count=failed,checks=checks,warnings=warnings,native_inference_status=inference,details=details,independent_replication_models=args.models,mechanical_checks_only_models=[name for name in masks if name not in args.models],limitations=['Fund focus is an existing opening-interval classification and may change over time; fund totals across types are not additive.','Y is fund-security-window, while Net is company-window reused across assets/funds; company-period counts are not unique article counts.','Only the two predeclared bond models are numerically replicated; other outputs have exact-sample/cache/mechanical QA.','Current unique-mask Holm covers this frozen exploratory focus suite, not every historical H1 test.','Original article-level classifications are not independently rebuilt by this audit.'])
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],pass_count=result['pass_count'],fail_count=failed)))


def fixture():
    rng=np.random.default_rng(506);n=1200;fund=np.repeat(np.arange(24),50)
    net=rng.normal(size=n);control=rng.normal(size=n)
    frame=pd.DataFrame(dict(fund_id=fund,company_id=np.tile(np.arange(12),100),net_sentiment=net,control=control,quantity_change_pct=.8*net-.2*control+fund*.4+rng.normal(scale=.05,size=n)))
    result=within_ols(frame,['net_sentiment','control'])
    assert np.allclose(result['beta'],[.8,-.2],atol=.01)
    assert result['df_resid']==1174
    assert np.isfinite(result['covariance']).all()
    p=holm({'a':.02,'b':None,'c':.1});assert np.isclose(p['a'],.06)and p['b']is None and np.isclose(p['c'],.2)
    print(json.dumps(dict(status='PASS',scope='synthetic_only',checks=4)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan');parser.add_argument('--root');parser.add_argument('--output')
    parser.add_argument('--models',nargs='+',default=REPLICAS);parser.add_argument('--rtol',type=float,default=1e-5)
    parser.add_argument('--fixture-only',action='store_true');args=parser.parse_args()
    fixture()if args.fixture_only else audit(args)


