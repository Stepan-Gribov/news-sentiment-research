"""Independent QA of signed-sentiment quantity panels and absolute-response tests.

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


REPLICAS=[f"{cut}__FOCUS_{group}_ROWS_60_INTERVALS_24__CURRENT_PLUS_LAG1" for cut in ["P99","P95"] for group in ["EQUITY","BOND","MIXED"]]


def equal_values(left,right):
    try:
        pd.testing.assert_frame_equal(left.reset_index(drop=True),right.reset_index(drop=True),check_exact=True,check_dtype=False)
        return True
    except AssertionError:return False


def independent_contrast(beta,covariance,vector,df):
    from scipy import stats
    beta=np.asarray(beta,float);covariance=np.asarray(covariance,float);vector=np.asarray(vector,float)
    estimate=float(vector@beta);variance=float(vector@covariance@vector)
    result=dict(status='UNDEFINED_COVARIANCE',beta=estimate,variance=variance,se_native=None,
        t_native=None,p_native=None,ci95_native_low=None,ci95_native_high=None,native_df_resid=int(df),contrast=vector.tolist())
    if np.isfinite(estimate)and np.isfinite(variance)and variance>0 and df>0:
        se=float(np.sqrt(variance));t=estimate/se;critical=stats.t.ppf(.975,df)
        result.update(status='ESTIMATED',se_native=se,t_native=t,p_native=float(2*stats.t.sf(abs(t),df)),
            ci95_native_low=float(estimate-critical*se),ci95_native_high=float(estimate+critical*se))
    return result


def independent_pair(beta,covariance,df):
    beta=np.asarray(beta,float);covariance=np.asarray(covariance,float)
    difference=independent_contrast(beta,covariance,[1,-1],df)
    total=independent_contrast(beta,covariance,[1,1],df)
    valid=difference['status']=='ESTIMATED'and total['status']=='ESTIMATED'
    return dict(status='ESTIMATED'if valid else'UNDEFINED_COVARIANCE',beta_positive=float(beta[0]),
        beta_negative=float(beta[1]),expected_directional_signs=bool(beta[0]>=0 and beta[1]<=0),
        negative_absolute_minus_positive_absolute=float(abs(beta[1])-abs(beta[0])),
        negative_absolute_stronger=bool(abs(beta[1])>abs(beta[0])),difference_zero=difference,sum_zero=total,
        positive_coefficient=independent_contrast(beta,covariance,[1,0],df),
        negative_coefficient=independent_contrast(beta,covariance,[0,1],df),
        p_abs=max(difference['p_native'],total['p_native'])if valid else None)


def audit(args):
    if os.name!='nt':raise RuntimeError('Actual-data QA must run on research-host')
    from scipy import stats
    planpath=Path(args.plan);plan=json.loads(planpath.read_text(encoding='utf-8'))
    root=Path(args.root or plan['output_root']);output=Path(args.output or root/'qa_independent.json')
    if output.exists():raise FileExistsError('Refusing to overwrite an existing QA version')
    if 'databaseresources'in[p.lower()for p in output.parts]:raise ValueError('Protected output')
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'));coef=pd.read_csv(root/'coefficients.csv')
    prior=json.loads(Path(plan['focus_summary']).read_text(encoding='utf-8'));checks,warnings,details=[],[],{}
    def check(name,passed,**evidence):checks.append(dict(name=name,status='PASS'if bool(passed)else'FAIL',**clean(evidence)))
    def close(a,b,atol=1e-7):return bool(np.isclose(float(a),float(b),rtol=args.rtol,atol=atol))
    check('frozen_plan_hash',sha(planpath)==summary['plan_sha256'])
    check('same_frozen_specification',summary.get('specification')==plan)
    check('predeclared_independent_models',args.models==plan['independent_qa_models'])
    check('native_scalar_inference_only',plan['multiple_testing_adjustments']is False and set(coef.columns)=={'model','outcome','term','beta','se_native','t_native','p_native','ci95_native_low','ci95_native_high'})
    check('same_fund_FE_covariance_dimensions',plan['fund_FE']is True and plan['time_FE']is False and plan['cluster_time']is False and plan['clusters']==['fund_id','company_id'])
    before={}
    for key,expected in plan['expected_source_sha256'].items():
        before[key]=sha(plan[key]);check('frozen_source:'+key,before[key]==expected,sha256=before[key])
    source=pd.read_parquet(plan['input_panel']).sort_values(['fund_id','end_date','security_id','interval_id'],kind='stable').reset_index(drop=True)
    mapping=pd.read_parquet(plan['lag_map']);reference=pd.read_parquet(plan['cutoff_reference_sample'])
    check('source_N_unique_rows',len(source)==plan['expected_primary_rows']and source.row_id.notna().all()and not source.row_id.duplicated().any())
    check('lagmap_same_unique_source_ids',set(mapping.row_id)==set(source.row_id)and not mapping.row_id.duplicated().any())
    aligned=mapping.set_index('row_id').reindex(source.row_id).reset_index(drop=True)
    check('lagmap_company_current_interval_binding',np.array_equal(source.company_id.astype(str),aligned.company_id.astype(str))and np.array_equal(source.fund_id.astype(str),aligned.fund_id.astype(str))and np.array_equal(source.interval_id.astype(str),aligned.current_interval_id.astype(str)))
    frame=source.copy()
    for field in aligned:
        if field.startswith('lag_')or field=='net_sentiment_lag1':frame[field]=aligned[field].to_numpy()
    lag_eligible=truth(aligned.lag_eligible_primary)&np.isfinite(pd.to_numeric(aligned.net_sentiment_lag1,errors='coerce'))
    check('no_numeric_ineligible_lag',not(np.isfinite(pd.to_numeric(aligned.net_sentiment_lag1,errors='coerce'))&~truth(aligned.lag_eligible_primary)).any())
    frame['complete_current_lag1']=np.asarray(lag_eligible,bool)
    parts=plan['required_targets_by_variant']['CURRENT_PLUS_LAG1']
    check('two_and_four_required_targets',len(parts)==4 and plan['required_targets_by_variant']['CURRENT_FULL']==parts[:2]and plan['required_targets_by_variant']['CURRENT_COMMON']==parts[:2])
    for score,positive,negative in [('net_sentiment',parts[0],parts[1]),('net_sentiment_lag1',parts[2],parts[3])]:
        values=pd.to_numeric(frame[score],errors='coerce');frame[positive]=values.clip(lower=0);frame[negative]=(-values).clip(lower=0)
        valid=np.isfinite(values)
        check('signed_parts_reconstruct:'+score,np.array_equal((frame.loc[valid,positive]-frame.loc[valid,negative]).to_numpy(),values.loc[valid].to_numpy()))
        check('parts_disjoint_nonnegative:'+score,frame.loc[valid,[positive,negative]].ge(0).all().all()and frame.loc[valid,positive].mul(frame.loc[valid,negative]).eq(0).all())
        check('missing_parts_not_zero:'+score,frame.loc[~valid,[positive,negative]].isna().all().all())
    population=reference.loc[reference.quantity_open.gt(0)&reference.quantity_close.gt(0)]
    check('original_quantile_reference_universe',len(reference)==plan['cutoff_reference_rows']and len(population)==plan['cutoff_calibration_continuing_rows'])
    continuing=source.quantity_open.gt(0)&source.quantity_close.gt(0);entries=source.quantity_open.eq(0)&source.quantity_close.gt(0);exits=source.quantity_open.gt(0)&source.quantity_close.eq(0)
    check('protected_entry_exit_coding',source.loc[entries,'quantity_change_pct'].eq(100).all()and source.loc[exits,'quantity_change_pct'].eq(-100).all())
    saved_thresholds={cut['tag']:cut for cut in summary['thresholds']};cut_masks={};last_mask=None
    for cut in plan['cutoffs']:
        tag=cut['tag'];threshold=float(population.quantity_change_pct.quantile(cut['quantile'],interpolation='linear'))
        mask=~continuing|source.quantity_change_pct.le(threshold);cut_masks[tag]=mask
        check('fixed_calibrated_quantile:'+tag,close(threshold,cut['cutoff_percent'],1e-10)and close(threshold,saved_thresholds[tag]['cutoff_percent'],1e-10),independent=threshold)
        check('threshold_calibration_hash:'+tag,row_hash(population)==saved_thresholds[tag]['reference_population_row_hash'])
        check('threshold_row_counts:'+tag,int(mask.sum())==saved_thresholds[tag]['kept_from_P99_primary']and int((~mask).sum())==saved_thresholds[tag]['removed_from_P99_primary'])
        check('all_entries_exits_and_ties_retained:'+tag,mask.loc[entries|exits].all()and mask.loc[source.quantity_change_pct.eq(threshold)].all())
        if last_mask is not None:check('nested_stricter_mask:'+tag,not(mask&~last_mask).any())
        last_mask=mask
    support=source.groupby(['fund_focus','fund_id'],observed=True).agg(fund_rows=('row_id','size'),fund_intervals=('interval_id','nunique'),fund_end_dates=('end_date','nunique'))
    saved_support=pd.read_csv(plan['focus_support'],dtype={'fund_id':str}).set_index(['fund_focus','fund_id'])
    for field in support:check('old_support_fixed_before_new_filters:'+field,np.array_equal(support[field],saved_support.reindex(support.index)[field]))
    old_masks={'REFERENCE_AUGMENTED_P99':pd.Series(True,index=source.index)}
    for group in plan['focus_groups']:
        belongs=source.fund_focus.eq(group);old_masks['FOCUS_'+group.upper()+'_BASELINE']=belongs;sg=support.loc[group]
        for rows in [30,60]:
            for periods in [12,24,36]:old_masks[f'FOCUS_{group.upper()}_ROWS_{rows}_INTERVALS_{periods}']=belongs&source.fund_id.isin(sg.index[sg.fund_rows.ge(rows)&sg.fund_intervals.ge(periods)])
    for name,mask in old_masks.items():check('old_base_mask_unchanged:'+name,row_hash(source.loc[mask])==prior['samples'][name]['row_hash']and int(mask.sum())==prior['samples'][name]['rows'])
    records={r['model']:r for r in summary['models']};controls=plan['baseline_controls']+plan['macro_controls']
    expected_models={cut+'__'+name+'__'+variant for cut in cut_masks for name in old_masks for variant in plan['variants']}
    check('all_four_hundred_thirty_two_labels',len(expected_models)==432 and set(records)==set(plan['planned_models'])==expected_models)
    check('unique_native_coefficient_keys',not coef.duplicated(['model','term']).any())
    native_fields=['beta','se_native','t_native','p_native','ci95_native_low','ci95_native_high']
    verified_samples={};seen_designs={};datasets={};inference={}
    def compare_contrast(name,saved,computed,prefix):
        check(prefix+'status:'+name,saved['status']==computed['status'])
        check(prefix+'vector_and_df:'+name,saved['contrast']==computed['contrast']and saved['native_df_resid']==computed['native_df_resid'])
        for field in ['beta','variance']:
            left,right=saved[field],computed[field]
            check(prefix+field+':'+name,(left is None and not np.isfinite(right))or(left is not None and close(left,right)))
        if computed['status']=='ESTIMATED':
            check(prefix+'native_t_p_CI:'+name,all(close(saved[field],computed[field],1e-10 if field=='p_native'else 1e-7)for field in ['se_native','t_native','p_native','ci95_native_low','ci95_native_high']))
        else:check(prefix+'undefined_not_replaced_with_p_one:'+name,saved['p_native']is None and saved['se_native']is None)
    def check_tests(name,beta,covariance,df,saved,prefix):
        beta=np.asarray(beta,float);covariance=np.asarray(covariance,float);distributed=len(beta)==4
        matrices={'current':np.array([[1,0,0,0],[0,1,0,0]])}if distributed else{'current':np.eye(2)}
        if distributed:matrices.update(lag1=np.array([[0,0,1,0],[0,0,0,1]]),total_current_plus_lag1=np.array([[1,0,1,0],[0,1,0,1]]))
        check(prefix+'exposure_names:'+name,set(saved['tests'])==set(matrices))
        check(prefix+'full_target_df_and_repair:'+name,saved['native_df_resid']==df and saved['covariance_repair']is False)
        for scope,matrix in matrices.items():
            b=matrix@beta;v=matrix@covariance@matrix.T;found=saved['tests'][scope];computed=independent_pair(b,v,df);label=name+':'+scope
            check(prefix+'pair_transformation_and_covariance:'+label,np.array_equal(found['transformation_from_full_target_block'],matrix)and np.allclose(np.asarray(found['target_covariance'],float),v,rtol=args.rtol,atol=1e-8,equal_nan=True))
            check(prefix+'signed_slopes_and_magnitudes:'+label,close(found['beta_positive'],b[0])and close(found['beta_negative'],b[1])and close(found['negative_absolute_minus_positive_absolute'],abs(b[1])-abs(b[0]))and found['negative_absolute_stronger']==bool(abs(b[1])>abs(b[0]))and found['expected_directional_signs']==bool(b[0]>=0 and b[1]<=0))
            for component in ['difference_zero','sum_zero','positive_coefficient','negative_coefficient']:compare_contrast(label+':'+component,found[component],computed[component],prefix)
            if computed['p_abs']is None:check(prefix+'undefined_IUT_not_p_one:'+label,found['p_abs']is None and found['status']!='ESTIMATED')
            else:check(prefix+'absolute_equality_IUT:'+label,found['status']=='ESTIMATED'and close(found['p_abs'],computed['p_abs'],1e-10)and close(found['p_abs'],max(found['difference_zero']['p_native'],found['sum_zero']['p_native']),1e-10))
    for cut_tag,cut_mask in cut_masks.items():
        for base_name,base_mask in old_masks.items():
            full_mask=base_mask&cut_mask;common_mask=full_mask&lag_eligible;pair=[]
            for variant in plan['variants']:
                name=cut_tag+'__'+base_name+'__'+variant;distributed=variant=='CURRENT_PLUS_LAG1'
                mask=full_mask if variant=='CURRENT_FULL'else common_mask;expected=frame.loc[mask];record=records[name];info=summary['samples'][name]
                expected_hash=row_hash(expected);required=plan['required_targets_by_variant'][variant]
                check('exact_fixed_mask_hash_N:'+name,info['row_hash']==expected_hash and info['rows']==len(expected))
                check('model_N_funds_companies:'+name,record['nobs']==len(expected)and record['funds']==expected.fund_id.nunique()and record['companies']==expected.company_id.nunique())
                check('support_not_requalified:'+name,record['support_qualification_fixed_before_cutoff_and_lag_drop']is True and record['old_rows']==int(base_mask.sum())and record['trimmed_full_rows']==int(full_mask.sum())and record['common_rows']==int(common_mask.sum()))
                if expected_hash not in verified_samples:
                    sample=pd.read_parquet(info['parquet'])
                    check('actual_sample_ids_order:'+name,sample.row_id.reset_index(drop=True).equals(expected.row_id.reset_index(drop=True)))
                    check('original_Y_X_Q_unchanged:'+name,equal_values(sample[list(source.columns)],expected[list(source.columns)]))
                    added=[c for c in frame if c.startswith('lag_')or c in parts+['net_sentiment_lag1','complete_current_lag1']]
                    check('lag_and_signed_parts_literal:'+name,equal_values(sample[added],expected[added]))
                    verified_samples[expected_hash]=info['parquet']
                else:check('reused_sample_file_same_path:'+name,info['parquet']==verified_samples[expected_hash])
                design='CURRENT_LAG_PARTS'if distributed else'CURRENT_PARTS';key=(design,expected_hash)
                current=coef.loc[coef.model.eq(name)].set_index('term')
                if key in seen_designs:
                    check('same_design_alias:'+name,record['exact_sample_alias_of']==seen_designs[key])
                    original=coef.loc[coef.model.eq(seen_designs[key])].set_index('term')
                    check('alias_native_values:'+name,set(current.index)==set(original.index)and np.allclose(current.reindex(original.index)[native_fields],original[native_fields],rtol=1e-12,atol=1e-10,equal_nan=True))
                else:seen_designs[key]=name
                if variant!='CURRENT_FULL':pair.append(info['row_hash'])
                if record['status']!='ESTIMATED':
                    check('not_estimated_no_native_coefficients:'+name,len(current)==0);warnings.append(dict(model=name,kind='NOT_ESTIMATED',reason=record.get('reason')));inference[name]=dict(status='NOT_ESTIMATED');continue
                terms=record['retained_terms'];df=record['native_df_resid'];options=record['covariance_options']
                check('all_parts_required_raw_net_excluded:'+name,set(required).issubset(terms)and set(terms).issubset(required+controls)and 'net_sentiment'not in terms and 'net_sentiment_lag1'not in terms)
                check('native_terms_df:'+name,set(current.index)==set(['const']+terms)and df==len(expected)-expected.fund_id.nunique()-len(terms))
                check('same_FE_covariance:'+name,record['included_effects']==['Entity']and record['time_effects']is False and all(options.get(k)==v for k,v in dict(cov_type='clustered',clusters=['fund_id','company_id'],debiased=True,auto_df=False,count_effects=True,group_debias=True,cluster_time=False).items()))
                finite=np.isfinite(current[native_fields].to_numpy(float)).all(axis=1)&current.se_native.gt(0).to_numpy();valid=current.loc[finite];t=valid.beta/valid.se_native;critical=stats.t.ppf(.975,df)
                check('defined_native_t_p_CI:'+name,np.allclose(t,valid.t_native,rtol=args.rtol,atol=1e-7)and np.allclose(2*stats.t.sf(abs(t),df),valid.p_native,rtol=args.rtol,atol=1e-10)and np.allclose(valid.beta-critical*valid.se_native,valid.ci95_native_low,rtol=args.rtol,atol=1e-7)and np.allclose(valid.beta+critical*valid.se_native,valid.ci95_native_high,rtol=args.rtol,atol=1e-7))
                if(~finite).any():warnings.append(dict(model=name,kind='UNDEFINED_COEFFICIENT_INFERENCE',terms=current.index[~finite].tolist()))
                if min(record['funds'],record['companies'])<30:warnings.append(dict(model=name,kind='LOW_CLUSTER_COUNT',funds=record['funds'],companies=record['companies']))
                overall=record.get('overall_robust_F_inference',{})
                if overall.get('status')=='UNDEFINED_COVARIANCE':warnings.append(dict(model=name,kind='UNDEFINED_OVERALL_ROBUST_F',native=overall))
                inference[name]={term:'FINITE'if term in valid.index else'UNDEFINED'for term in required}
                saved=summary['h2_tests'][name];block=np.asarray(saved['full_target_covariance'],float);betas=current.loc[required,'beta'].to_numpy(float)
                check('declared_test_targets:'+name,saved['target_terms']==required)
                check('raw_target_cov_diagonal:'+name,all(not np.isfinite(current.loc[term,'se_native'])or close(current.loc[term,'se_native']**2,block[j,j])for j,term in enumerate(required)))
                check_tests(name,betas,block,df,saved,'mechanical_')
                if any(test['status']!='ESTIMATED'for test in saved['tests'].values()):warnings.append(dict(model=name,kind='UNDEFINED_ABSOLUTE_RESPONSE_TEST'))
                if name in args.models:datasets[name]=(info['parquet'],record)
            check('current_common_and_distributed_same_rows:'+cut_tag+':'+base_name,len(pair)==2 and pair[0]==pair[1])
    check('unique_sample_file_count',summary['unique_sample_files']==len(verified_samples))
    check('unique_sample_design_count',summary['unique_sample_designs']==len(seen_designs))
    for name in args.models:
        if name not in datasets:details[name]=dict(numerical_replication_status='NOT_AVAILABLE');warnings.append(dict(model=name,kind='NUMERICAL_REPLICA_NOT_ESTIMATED'));continue
        path,record=datasets[name];sample=pd.read_parquet(path);terms=record['retained_terms'];required=plan['required_targets_by_variant']['CURRENT_PLUS_LAG1']
        result=within_ols(sample,terms);native=coef.loc[coef.model.eq(name)].set_index('term');begin=len(checks)
        check('independent_df:'+name,result['df_resid']==record['native_df_resid'])
        check('independent_within_R2:'+name,close(result['within_R2'],record['rsquared_within'],1e-10))
        for term in required:
            j=terms.index(term);beta=float(result['beta'][j]);variance=float(result['covariance'][j,j]);se=float(np.sqrt(variance))if variance>0 else np.nan;target=native.loc[term]
            check('independent_beta:'+name+':'+term,close(beta,target.beta,1e-8),independent=beta,native=target.beta)
            if np.isfinite(se):
                t=beta/se;p=2*stats.t.sf(abs(t),result['df_resid']);critical=stats.t.ppf(.975,result['df_resid'])
                check('independent_CGM_SE:'+name+':'+term,close(se,target.se_native),independent=se,native=target.se_native)
                check('independent_native_t_p_CI:'+name+':'+term,close(t,target.t_native)and close(p,target.p_native,1e-10)and close(beta-critical*se,target.ci95_native_low)and close(beta+critical*se,target.ci95_native_high))
            else:check('independent_undefined_part_variance:'+name+':'+term,not np.isfinite(target.se_native)or target.se_native<=0)
        indices=[terms.index(term)for term in required];betas=result['beta'][indices];block=result['covariance'][np.ix_(indices,indices)];saved=summary['h2_tests'][name]
        check('independent_full_four_target_covariance:'+name,np.allclose(block,np.asarray(saved['full_target_covariance'],float),rtol=args.rtol,atol=1e-8,equal_nan=True))
        check_tests(name,betas,block,result['df_resid'],saved,'independent_')
        details[name]={key:value for key,value in result.items()if key not in['beta','covariance']};details[name].update(coefficients=dict(zip(terms,result['beta'])),numerical_replication_status='PASS'if all(row['status']=='PASS'for row in checks[begin:])else'FAIL')
    for key,value in before.items():check('frozen_source_preserved_after_QA:'+key,sha(plan[key])==value)
    details['audit_scope']=dict(all_model_labels=432,unique_sample_data_files_verified=len(verified_samples),independent_numeric_models=args.models,calibration_continuing_rows=len(population),protected_entries=int(entries.sum()),protected_exits=int(exits.sum()),common_lag_rows_before_new_cuts=int(lag_eligible.sum()))
    failed=sum(row['status']=='FAIL'for row in checks)
    result=dict(status='FAIL'if failed else('PASS_WITH_WARNINGS'if warnings else'PASS'),created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),plan_sha256=sha(planpath),pass_count=len(checks)-failed,fail_count=failed,checks=checks,warnings=warnings,native_inference_status=inference,details=details,independent_replication_models=args.models,mechanical_checks_only_models=[name for name in records if name not in args.models],limitations=['This is the quantity-outcome H2 counterpart with conditional entries, not reconstructed full passive-weight active trading.','Absolute equality is a union null; both difference and sum restrictions must reject, with no assumption inferred from observed coefficient signs.','Total response means equality of magnitudes of current-plus-lag coefficient sums, using all cross-covariances.','The four outcome restrictions are exploratory subsets; stricter positive-tail removal is not proof of source errors.','Only six prespecified distributed models have independent numerical replication; all other labels receive fixed-mask, native-arithmetic and contrast checks.'])
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(dict(status=result['status'],pass_count=result['pass_count'],fail_count=failed)))


def fixture():
    covariance=np.diag([.01,.01])
    assert independent_pair([1,1],covariance,100)['p_abs']==1
    assert independent_pair([1,-1],covariance,100)['p_abs']==1
    assert independent_pair([0,0],covariance,100)['p_abs']==1
    unequal=independent_pair([1,-2],covariance,100);assert unequal['p_abs']<.05 and unequal['negative_absolute_stronger']
    invalid=independent_pair([1,-2],[[1,2],[2,1]],100);assert invalid['p_abs']is None
    beta=np.array([1.,-2.,-1.,2.]);matrix=np.array([[1,0,1,0],[0,1,0,1]])
    assert independent_pair(matrix@beta,matrix@np.eye(4)@matrix.T,100)['p_abs']==1
    rng=np.random.default_rng(132);n=2400;fund=np.repeat(np.arange(24),100);raw=rng.uniform(-1,1,n);lag=rng.uniform(-1,1,n)
    x=np.column_stack([np.maximum(raw,0),np.maximum(-raw,0),np.maximum(lag,0),np.maximum(-lag,0)])
    frame=pd.DataFrame(dict(fund_id=fund,company_id=np.tile(np.arange(12),200),p0=x[:,0],n0=x[:,1],p1=x[:,2],n1=x[:,3],quantity_change_pct=x@np.array([1.,-2.,.5,-.8])+fund*.3+rng.normal(scale=.05,size=n)))
    fit=within_ols(frame,['p0','n0','p1','n1']);assert np.allclose(fit['beta'],[1,-2,.5,-.8],atol=.025)
    assert fit['df_resid']==2372
    print(json.dumps(dict(status='PASS',scope='synthetic_only',checks=6)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan');parser.add_argument('--root');parser.add_argument('--output')
    parser.add_argument('--models',nargs='+',default=REPLICAS);parser.add_argument('--rtol',type=float,default=1e-5);parser.add_argument('--fixture-only',action='store_true');args=parser.parse_args()
    fixture()if args.fixture_only else audit(args)

