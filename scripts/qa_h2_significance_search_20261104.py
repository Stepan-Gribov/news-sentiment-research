"""Independent sample/grid and selected-rowwise QA of the fixed H2 search.

This module never repeats the search, calls its solver, changes research inputs,
or estimates additional specifications. Its only regressions are independent
rowwise replicas of already selected, saved native models.
"""
from __future__ import annotations

import os
for _option in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:
    os.environ.setdefault(_option,'1')

import argparse
import datetime as dt
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

PARTS=['net_positive','net_negative','net_positive_lag1','net_negative_lag1']
SIX=['closing_HHI','bond_indicator','imoex_ret','RVI','dummy_mar2014','dummy_sep2018']
MATRICES={
    'CURRENT_ONLY':{'current':np.eye(2)},
    'CURRENT_PLUS_LAG1':{
        'current':np.array([[1.,0.,0.,0.],[0.,1.,0.,0.]]),
        'lag1':np.array([[0.,0.,1.,0.],[0.,0.,0.,1.]]),
        'total_current_plus_lag1':np.array([[1.,0.,1.,0.],[0.,1.,0.,1.]])}}


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def catalog_map(payload,container,key):
    if isinstance(payload,dict) and container in payload:payload=payload[container]
    if isinstance(payload,list):return {str(record[key]):record for record in payload}
    if isinstance(payload,dict):return payload
    raise TypeError('Unrecognized catalog schema: '+container)


def truth(values):
    return values.astype(str).str.strip().str.lower().isin(['true','1','yes'])


def independent_inputs(plan,root,helper,check,same):
    """Reconstruct masks directly; no import of the search/preflight helpers."""
    source=plan['input_panel'];original=pd.read_parquet(source).sort_values(
        ['fund_id','end_date','security_id','interval_id'],kind='stable').reset_index(drop=True)
    canonical_path=root/'data'/'canonical_signed_panel.parquet'
    canonical=pd.read_parquet(canonical_path)
    check('canonical_panel_preserves_every_original_column',len(original)==221899 and not original.row_id.duplicated().any()
        and set(original.columns).issubset(canonical.columns) and original.equals(canonical[original.columns]),
        mismatched_columns=[column for column in original if column not in canonical or not original[column].equals(canonical[column])])
    lag=pd.read_parquet(plan['lag_map'])
    check('exact_lag_row_key_universe',not lag.row_id.duplicated().any() and set(lag.row_id)==set(original.row_id))
    lag=lag.set_index('row_id').reindex(original.row_id)
    lag_values=pd.to_numeric(lag.net_sentiment_lag1,errors='coerce').to_numpy(float)
    lag_eligible=truth(lag.lag_eligible_primary).to_numpy()
    common=lag_eligible & np.isfinite(lag_values)
    check('independent_current_lag_common_gate',np.array_equal(truth(canonical.complete_current_lag1),common)
        and np.array_equal(canonical.net_sentiment_lag1.to_numpy(float),lag_values,equal_nan=True)
        and int(common.sum())==194041)
    for source,positive,negative in [('net_sentiment',PARTS[0],PARTS[1]),('net_sentiment_lag1',PARTS[2],PARTS[3])]:
        values=canonical[source].to_numpy(float)
        check('exact_sign_parts_and_missingness:'+source,
            np.array_equal(canonical[positive].to_numpy(float),np.maximum(values,0),equal_nan=True)
            and np.array_equal(canonical[negative].to_numpy(float),np.maximum(-values,0),equal_nan=True))
    keep=list(dict.fromkeys(['row_id','fund_id','company_id','security_id','interval_id','fund_focus','start_date','end_date',
        'quantity_open','quantity_close','quantity_change_pct','net_sentiment','net_sentiment_lag1']+PARTS+plan['all16_controls']))
    frame=canonical[keep].copy()
    q0=pd.to_numeric(frame.quantity_open,errors='coerce').to_numpy(float)
    q1=pd.to_numeric(frame.quantity_close,errors='coerce').to_numpy(float)
    raw=frame.quantity_change_pct.to_numpy(float)
    continuing=(q0>0)&(q1>0);entry=(q0==0)&(q1>0);exit=(q0>0)&(q1==0)
    check('accepted_primary_quantity_states',np.all(continuing|entry|exit)
        and np.all(raw[entry]==100) and np.all(raw[exit]==-100))
    reference=pd.read_parquet(plan['cutoff_reference_sample'],columns=['quantity_open','quantity_close','quantity_change_pct'])
    population=reference.loc[reference.quantity_open.gt(0)&reference.quantity_close.gt(0),'quantity_change_pct']
    check('frozen_cutoff_calibration_population',len(population)==204152)
    cuts={}
    for cut in plan['cutoffs']:
        cutoff=float(population.quantile(cut['quantile'],interpolation='linear'))
        check('fixed_cutoff:'+cut['tag'],same(cutoff,cut['cutoff_percent'],1e-10))
        cuts[cut['tag']]=(~continuing)|(raw<=cut['cutoff_percent'])
    saved_masks=pd.read_parquet(root/'data'/'sample_masks.parquet')
    check('membership_mask_row_order',np.array_equal(saved_masks.row_id,frame.row_id))
    masks={};mask_metadata={}
    for group in plan['groups']:
        group_mask=np.ones(len(frame),dtype=bool) if group=='pooled' else frame.fund_focus.eq(group).to_numpy()
        history=frame.loc[group_mask].groupby(frame.loc[group_mask,'fund_id'].astype(str),sort=False).agg(
            rows=('row_id','size'),intervals=('interval_id','nunique'))
        for support in plan['support_rules']:
            qualified=history.index if support['rows'] is None else history.index[
                (history.rows>=support['rows'])&(history.intervals>=support['intervals'])]
            history_mask=group_mask & frame.fund_id.astype(str).isin(qualified).to_numpy()
            for cut in plan['cutoffs']:
                sample_id='B__'+group+'__'+support['tag']+'__'+cut['tag']
                masks[sample_id]=history_mask & cuts[cut['tag']] & common
                mask_metadata[sample_id]=dict(family='B',group=group,support=support['tag'],cutoff=cut['tag'],
                    pre_lag_pre_cut_rows=int(history_mask.sum()),excluded_by_cut=int((history_mask & ~cuts[cut['tag']]).sum()),
                    excluded_lag_after_cut=int((history_mask & cuts[cut['tag']] & ~common).sum()))
    for definition in plan['exact_samples']:
        group=definition['group'];exact=pd.read_parquet(definition['input_panel']).sort_values(
            ['fund_id','end_date','security_id','interval_id'],kind='stable').reset_index(drop=True)
        check('exact_A_source_sha_and_row_hash:'+group,helper.sha(definition['input_panel'])==definition['expected_source_sha256']
            and helper.row_hash(exact)==definition['expected_ordered_row_hash'] and len(exact)==definition['expected_rows'])
        mask=frame.row_id.isin(exact.row_id).to_numpy()
        check('exact_A_membership_and_common_rows:'+group,int(mask.sum())==len(exact)
            and np.array_equal(frame.loc[mask,'row_id'],exact.row_id) and np.all(common[mask]))
        masks['A__'+group]=mask
        mask_metadata['A__'+group]=dict(family='A',group=group,support='NONE' if group=='pooled' else 'R60_I24',cutoff='P99')
        masks['C_CONTINUING__'+group]=mask & continuing
        mask_metadata['C_CONTINUING__'+group]=dict(family='C',group=group,support='exact_A_continuing',cutoff='P99')
    check('exactly_72_declared_membership_masks',len(masks)==72 and set(saved_masks)=={'row_id'}|set(masks))
    samples=catalog_map(json.loads((root/'sample_catalog.json').read_text(encoding='utf-8')),'samples','sample_id')
    check('all_sample_catalog_keys',set(samples)==set(masks))
    for sample_id,mask in masks.items():
        check('independent_sample_membership:'+sample_id,np.array_equal(saved_masks[sample_id].to_numpy(bool),mask))
        part=frame.loc[mask];metadata=samples[sample_id]
        rows=len(part);funds=int(part.fund_id.nunique());companies=int(part.company_id.nunique());rh=helper.row_hash(part)
        mask_metadata[sample_id].update(rows=rows,funds=funds,companies=companies,ordered_row_hash=rh)
        check('independent_sample_counts_and_hash:'+sample_id,
            (metadata['rows'],metadata['funds'],metadata['companies'],metadata['ordered_row_hash'])==(rows,funds,companies,rh))
    alt=pd.read_parquet(root/'data'/'alternative_outcomes.parquet')
    check('alternative_outcome_row_order_and_only_new_columns',np.array_equal(alt.row_id,frame.row_id)
        and set(alt)=={'row_id','h2_y_continuing_percent','h2_y_continuing_log','h2_y_symmetric'})
    outcomes={'quantity_change_pct':raw,
        'h2_y_continuing_percent':np.where(continuing,raw,np.nan),
        'h2_y_continuing_log':np.full(len(frame),np.nan),
        'h2_y_symmetric':200*(q1-q0)/(q1+q0)}
    outcomes['h2_y_continuing_log'][continuing]=100*(np.log(q1[continuing])-np.log(q0[continuing]))
    for column,expected in outcomes.items():
        if column=='quantity_change_pct':continue
        check('independent_alternative_outcome_formula:'+column,same(alt[column],expected,1e-10))
    boundary_tolerance=2*np.spacing(200.)
    check('symmetric_entries_exits_and_continuing_log_missingness',
        np.allclose(outcomes['h2_y_symmetric'][entry],200,rtol=0,atol=boundary_tolerance)
        and np.allclose(outcomes['h2_y_symmetric'][exit],-200,rtol=0,atol=boundary_tolerance)
        and np.isnan(outcomes['h2_y_continuing_log'][~continuing]).all())
    return frame,masks,mask_metadata,outcomes


def audit(args):
    if os.name!='nt':raise RuntimeError('Research-data QA runs only on research-host')
    from scipy import stats
    helper=load_module(args.rowwise_helper,'independent_rowwise_utilities')
    h2qa=load_module(args.h2_qa_helper,'independent_h2_contrasts')
    planpath=Path(args.plan);plan=json.loads(planpath.read_text(encoding='utf-8'))
    root=Path(args.root or plan['output_root']);output=Path(args.output or root/'qa_independent.json')
    if output.exists():raise FileExistsError('Refusing to overwrite an existing QA report')
    if 'databaseresources'in [p.lower()for p in output.parts]:raise ValueError('Protected output')
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    manifest=json.loads((root/'source_manifest.json').read_text(encoding='utf-8'))
    catalog=catalog_map(json.loads((root/'model_catalog.json').read_text(encoding='utf-8')),'models','model_id')
    table=pd.read_csv(root/'all_models.csv',low_memory=False).set_index('model_id',drop=False)
    long=pd.read_csv(root/'all_h2_tests.csv',low_memory=False)
    short=pd.read_csv(root/'shortlist.csv',low_memory=False).set_index('model_id',drop=False)
    table.index.name=None;short.index.name=None
    targets=json.loads((root/'all_target_covariance.json').read_text(encoding='utf-8'))
    native=json.loads((root/'native_results.json').read_text(encoding='utf-8'))
    checks,warnings,numerical,source_before=[],[],{},{}
    source_paths={key:plan[key]for key in plan['expected_source_sha256']}
    source_paths.update({'exact_'+definition['group']:definition['input_panel']for definition in plan['exact_samples']})
    source_paths.update(h2_helper=plan['h2_helper'],preflight_helper=plan['preflight_helper_path'])
    def check(name,passed,**evidence):checks.append(dict(name=name,status='PASS'if bool(passed)else'FAIL',**helper.clean(evidence)))
    def same(a,b,atol=1e-8):return bool(np.allclose(np.asarray(a,float),np.asarray(b,float),rtol=args.rtol,atol=atol,equal_nan=True))
    def covsame(a,b):
        a,b=np.asarray(a,float),np.asarray(b,float)
        scale=np.sqrt(np.maximum(np.abs(np.diag(a)),np.abs(np.diag(b))))
        denominator=np.outer(np.maximum(scale,1e-150),np.maximum(scale,1e-150))
        return same(a/denominator,b/denominator,1e-7)
    def boolean(value):return str(value).strip().lower()in ['true','1']
    check('fixed_plan_and_complete_finite_search',summary['plan']==plan and manifest['plan_sha256']==helper.sha(planpath)
        and len(table)==len(catalog)==len(targets)==660 and set(table.index)==set(catalog)==set(targets)
        and table.index.is_unique and summary['model_labels']==660 and summary['unique_fast_fits']==652)
    check('unchanged_covariance_H2_and_no_adjustment_policy',plan['targets']==PARTS and plan['six_controls']==SIX
        and plan['fixed_effects']==['fund_id'] and plan['clusters']==['fund_id','company_id']
        and plan['time_FE']is False and plan['cluster_time']is False and plan['covariance_repair']is False
        and plan['multiple_testing_adjustments']is False and summary['multiple_testing_adjustments']is False
        and not any('holm'in c.lower()or'adjusted_p'in c.lower()for c in table.columns)
        and plan['covariance_options']==dict(cov_type='clustered',cluster_entity=False,cluster_time=False,
            group_debias=True,debiased=True,auto_df=False,count_effects=True))
    for key,expected in plan['expected_source_sha256'].items():
        source_before[key]=helper.sha(plan[key]);check('frozen_source:'+key,source_before[key]==expected)
    for definition in plan['exact_samples']:
        key='exact_'+definition['group'];source_before[key]=helper.sha(definition['input_panel'])
        check('frozen_source:'+key,source_before[key]==definition['expected_source_sha256'])
    source_before['h2_helper']=helper.sha(plan['h2_helper'])
    check('original_H2_helper',source_before['h2_helper']==plan['h2_helper_sha256'])
    source_before['preflight_helper']=helper.sha(plan['preflight_helper_path'])
    check('frozen_preflight_helper_and_prefit_count_audit',source_before['preflight_helper']==plan['preflight_helper_sha256']
        and helper.sha(plan['prefit_mask_audit_path'])==plan['prefit_mask_audit_sha256'])
    generated_before={p:helper.sha(root/'data'/p)for p in ['canonical_signed_panel.parquet','sample_masks.parquet','alternative_outcomes.parquet']}
    frame,masks,metadata,outcomes=independent_inputs(plan,root,helper,check,same)
    expected={}
    def add(model_id,family,group,variant,sample_id,controls,mask,outcome,tag):
        expected[model_id]=dict(model_id=model_id,family=family,group=group,variant=variant,sample_id=sample_id,
            controls=list(controls),control_mask=mask,outcome_column=outcome,outcome_tag=tag)
    for group in plan['groups']:
        for variant in plan['variants']:
            for mask in list(range(64))+['ALL16']:
                controls=plan['all16_controls']if mask=='ALL16'else[c for j,c in enumerate(SIX)if mask>>j&1]
                label='ALL16'if mask=='ALL16'else f'SIXMASK_{mask:02d}'
                add(f'A__{group}__{variant}__{label}','A',group,variant,'A__'+group,controls,mask,'quantity_change_pct','RAW_PERCENT')
    for sample_id,meta in metadata.items():
        if meta['family']!='B':continue
        for variant in plan['variants']:
            add(sample_id+'__'+variant,'B',meta['group'],variant,sample_id,SIX,63,'quantity_change_pct','RAW_PERCENT')
    for group in plan['groups']:
        for alternative in plan['family_C']['alternatives']:
            tag=alternative['tag'];sample=('A__'if tag=='SYMMETRIC_ALL'else'C_CONTINUING__')+group
            add('C__'+group+'__'+tag,'C',group,'CURRENT_PLUS_LAG1',sample,SIX,63,alternative['dependent_column'],tag)
    check('all_660_exact_model_ids_and_families',set(expected)==set(catalog)==set(table.index)
        and table.family.value_counts().to_dict()=={'A':520,'B':128,'C':12})
    keys={};scope_count=0;current_scope_failures=[]
    def key(spec):return(metadata[spec['sample_id']]['ordered_row_hash'],spec['outcome_column'],spec['variant'],tuple(spec['controls']))
    for model_id,spec in expected.items():
        recorded=catalog[model_id];row=table.loc[model_id];meta=metadata[spec['sample_id']]
        fields=['model_id','family','group','variant','sample_id','outcome_column','outcome_tag']
        controls_match=(str(row.controls_included)=='|'.join(spec['controls']))if spec['controls']else pd.isna(row.controls_included)
        check('declared_model_and_sample_metadata:'+model_id,
            all(recorded[field]==spec[field]and row[field]==spec[field]for field in fields)
            and recorded['controls']==spec['controls'] and str(recorded['control_mask'])==str(spec['control_mask'])==str(row.control_mask)
            and controls_match
        )
        check('fixed_model_N_clusters_control_count_and_row_hash:'+model_id,
            (int(row.nobs),int(row.funds),int(row.companies),row.row_hash)==(meta['rows'],meta['funds'],meta['companies'],meta['ordered_row_hash'])
            and int(row.n_controls)==len(spec['controls']))
        fitkey=key(spec);keys.setdefault(fitkey,[]).append(model_id)
        raw=targets[model_id];nt=4 if spec['variant']=='CURRENT_PLUS_LAG1'else 2
        if row.status!='ESTIMATED':
            check('unestimable_result_not_dropped:'+model_id,raw['status']!='ESTIMATED' and not pd.isna(row.reason))
            warnings.append(dict(model_id=model_id,kind='NOT_ESTIMABLE',reason=str(row.reason)));continue
        df=meta['rows']-meta['funds']-nt-len(spec['controls'])
        b=np.asarray(raw['target_beta'],float);v=np.asarray(raw['target_covariance'],float)
        check('retained_target_dimension_native_df_and_covariance:'+model_id,
            raw['status']=='ESTIMATED'and raw['target_terms']==PARTS[:nt]and b.shape==(nt,)and v.shape==(nt,nt)
            and df==int(row.df_resid)==raw['df_resid']and np.isfinite(b).all()and np.isfinite(v).all()
            and same(v,v.T)and np.isfinite(row.rsquared_within)and -1e-9<=row.rsquared_within<=1+1e-9)
        for scope,matrix in MATRICES[spec['variant']].items():
            scope_count+=1;pair=matrix@b;paircov=matrix@v@matrix.T
            diff=h2qa.contrast(pair,paircov,[1.,-1.],df);summed=h2qa.contrast(pair,paircov,[1.,1.],df)
            positive=h2qa.contrast(pair,paircov,[1.,0.],df);negative=h2qa.contrast(pair,paircov,[0.,1.],df)
            valid=diff['status']=='ESTIMATED'and summed['status']=='ESTIMATED'
            p_abs=max(diff['p'],summed['p'])if valid else None
            expected_status='ESTIMATED'if valid else'UNDEFINED_COVARIANCE'
            check('complete_H2_IUT_and_signed_labels:'+model_id+':'+scope,
                same(row[scope+'_beta_positive'],pair[0])and same(row[scope+'_beta_negative'],pair[1])
                and same(row[scope+'_p_difference'],diff['p'],1e-10)and same(row[scope+'_p_sum'],summed['p'],1e-10)
                and same(row[scope+'_p_abs'],p_abs,1e-10)and row[scope+'_status']==expected_status
                and boolean(row[scope+'_expected_directional_signs'])==bool(pair[0]>=0 and pair[1]<=0)
                and same(row[scope+'_negative_absolute_minus_positive_absolute'],abs(pair[1])-abs(pair[0])))
            for sign,result in [('positive',positive),('negative',negative)]:
                check('native_coefficient_inference:'+model_id+':'+scope+':'+sign,
                    all(same(row[scope+'_'+sign+'_'+suffix],result[field],1e-10 if suffix=='p'else 1e-8)
                        for suffix,field in [('se','se'),('p','p'),('ci_low','low'),('ci_high','high')]))
            if not valid:warnings.append(dict(model_id=model_id,scope=scope,kind='UNDEFINED_H2_VARIANCE'))
        if spec['variant']=='CURRENT_ONLY':
            check('current_only_no_lag_or_total_test:'+model_id,pd.isna(row.get('lag1_p_abs'))and pd.isna(row.get('total_current_plus_lag1_p_abs')))
    check('exact_652_unique_fits_and_eight_aliases',len(keys)==652 and sum(len(value)-1 for value in keys.values())==8
        and table.exact_alias_of.notna().sum()==8)
    for fitkey,labels in keys.items():
        if len(labels)<2:continue
        reference=targets[labels[0]]
        check('identical_alias_target_results:'+labels[0],all(same(targets[label]['target_beta'],reference['target_beta'])
            and same(targets[label]['target_covariance'],reference['target_covariance'])for label in labels[1:]))
    expected_long={(mid,scope)for mid,spec in expected.items()for scope in MATRICES[spec['variant']]}
    check('all_1332_long_tests_preserved',len(long)==1332 and not long.duplicated(['model_id','scope']).any()
        and set(zip(long.model_id,long.scope))==expected_long)
    for record in long.to_dict('records'):
        mid=record['model_id'];scope=record['scope'];row=table.loc[mid]
        check('long_test_matches_grid:'+mid+':'+scope,all(same(record[column],row[scope+'_'+column],1e-10)
            for column in ['p_abs','p_difference','p_sum','beta_positive','beta_negative']))
    selection={}
    def select(mid,reason):selection.setdefault(mid,[]).append(reason)
    for mid,spec in expected.items():
        if spec['family']=='A'and spec['control_mask']in [0,63,'ALL16']:select(mid,'FIXED_A_BASELINE_'+str(spec['control_mask']))
        if spec['family']=='C':select(mid,'ALL_ALTERNATIVE_OUTCOMES')
    count_records=[]
    for (family,group,variant),sub in table.groupby(['family','group','variant'],sort=False):
        for scope in MATRICES[variant]:
            pcol=scope+'_p_abs';valid=sub.loc[np.isfinite(pd.to_numeric(sub[pcol],errors='coerce'))]
            best=valid.sort_values([pcol,'model_id'],kind='stable').iloc[0]if len(valid)else None
            record=dict(family=family,group=group,variant=variant,scope=scope,labels=len(sub),finite=len(valid),
                p_lt_05=int((valid[pcol]<.05).sum()),minimum_p=None if best is None else float(best[pcol]),
                maximum_p=None if len(valid)==0 else float(valid[pcol].max()),best_model_id=None if best is None else best.model_id)
            count_records.append(record)
            stored=next(c for c in summary['family_counts']if all(c[k]==record[k]for k in ['family','group','variant','scope']))
            check('full_grid_min_max_and_significance_counts:'+family+':'+group+':'+variant+':'+scope,
                all(stored[k]==record[k]for k in ['labels','finite','p_lt_05'])and same(stored['minimum_p'],record['minimum_p'],1e-10))
            if family in ['A','B']and best is not None:select(best.model_id,f'MIN_P_ABS_{family}_{group}_{variant}_{scope}')
            if best is not None:
                storedmin=next(c for c in summary['minima']if all(c[k]==record[k]for k in ['family','group','variant','scope']))
                check('minimum_tie_break_and_signs:'+family+':'+group+':'+variant+':'+scope,
                    storedmin['model_id']==best.model_id and same(storedmin['minimum_p'],best[pcol],1e-10)
                    and same(storedmin['beta_positive'],best[scope+'_beta_positive'])
                    and same(storedmin['beta_negative'],best[scope+'_beta_negative']))
    check('exact_shortlist_all_baselines_minima_and_twelve_C',set(selection)==set(short.index)
        and len(short)==summary['shortlist_labels']<=68 and sum(expected[mid]['family']=='C'for mid in selection)==12)
    for mid,reasons in selection.items():
        check('exact_shortlist_reasons:'+mid,set(str(short.loc[mid,'selection_reasons']).split('|'))==set(reasons))
        native_id=short.loc[mid,'native_model_id']
        if pd.isna(native_id):continue
        check('native_alias_provenance:'+mid,native_id in native and key(expected[native_id])==key(expected[mid]))
    check('native_unique_models_all_shortlisted',set(native)==set(short.native_model_id.dropna())
        and set(native)==set(summary['native_models'])and len(native)==summary['native_unique_fits'])
    shared=json.loads((root/'shared_hhi_native.json').read_text(encoding='utf-8'))
    if 'models'in shared:shared=shared['models']
    elif 'results'in shared:shared=shared['results']
    if isinstance(shared,list):shared={record['model_id']:record for record in shared}
    mandatory_shared={'A__'+group+'__CURRENT_ONLY__SIXMASK_01'for group in plan['groups']}
    check('four_post_grid_common_HHI_native_records',set(shared)==mandatory_shared)
    all_native=dict(native)
    for mid,record in shared.items():
        check('post_grid_shared_HHI_exact_same_declared_model:'+mid,
            record['model_id']==mid and record['sample_id']==expected[mid]['sample_id']
            and record['controls']==['closing_HHI']and record['variant']=='CURRENT_ONLY')
        if mid in all_native:
            check('shared_HHI_reused_native_matches_original:'+mid,
                same(record['full_covariance']['matrix'],all_native[mid]['full_covariance']['matrix'])
                and same([r['beta']for r in record['coefficients']],[r['beta']for r in all_native[mid]['coefficients']]))
        else:all_native[mid]=record
    for index,(mid,native_result)in enumerate(all_native.items()):
        spec=expected[mid];mask=masks[spec['sample_id']];nt=4 if spec['variant']=='CURRENT_PLUS_LAG1'else 2
        terms=PARTS[:nt]+spec['controls'];d=frame.loc[mask,['fund_id','company_id']+terms].copy()
        d['quantity_change_pct']=outcomes[spec['outcome_column']][mask]
        check('selected_rank_and_outcome_complete:'+mid,helper.selected_terms(d,PARTS[:nt],spec['controls'])==terms
            and np.isfinite(d[terms+['quantity_change_pct']].astype(float)).all().all())
        result=helper.independent_rowwise(d,terms);coef=pd.DataFrame(native_result['coefficients']).set_index('term')
        order=result['terms'];full=np.asarray(native_result['full_covariance']['matrix'],float)
        check('independent_native_full_beta_covariance_and_R2:'+mid,
            native_result['terms']==native_result['full_covariance']['terms']==order
            and same(result['beta'],coef.reindex(order).beta)and covsame(result['covariance'],full)
            and result['df_resid']==native_result['df_resid']==int(table.loc[mid,'df_resid'])
            and same(result['within_R2'],native_result['rsquared_within'],1e-10)
            and same(result['within_R2'],table.loc[mid,'rsquared_within'],1e-10))
        variance=np.diag(result['covariance']);se=np.sqrt(np.where(variance>0,variance,np.nan));df=result['df_resid']
        p=2*stats.t.sf(abs(result['beta']/se),df);critical=stats.t.ppf(.975,df)
        check('independent_full_native_SE_p_CI:'+mid,same(coef.reindex(order).se_native,se)
            and same(coef.reindex(order).p_native,p,1e-10)
            and same(coef.reindex(order).ci95_native_low,result['beta']-critical*se)
            and same(coef.reindex(order).ci95_native_high,result['beta']+critical*se))
        b=result['beta'][1:1+nt];v=result['covariance'][1:1+nt,1:1+nt]
        check('independent_fast_target_beta_and_covariance:'+mid,same(targets[mid]['target_beta'],b)
            and covsame(targets[mid]['target_covariance'],v))
        h2=native_result['h2_tests'];scope_results={}
        for scope,matrix in MATRICES[spec['variant']].items():
            pair=matrix@b;paircov=matrix@v@matrix.T;test=h2['tests'][scope]
            diff=h2qa.contrast(pair,paircov,[1.,-1.],df);summed=h2qa.contrast(pair,paircov,[1.,1.],df)
            valid=diff['status']=='ESTIMATED'and summed['status']=='ESTIMATED'
            p_abs=max(diff['p'],summed['p'])if valid else None
            check('independent_selected_H2_fullcov_and_IUT:'+mid+':'+scope,
                same(test['target_covariance'],paircov)and same([test['beta_positive'],test['beta_negative']],pair)
                and same(test['difference_zero']['variance'],diff['variance'])and same(test['sum_zero']['variance'],summed['variance'])
                and same(test['difference_zero']['p_native'],diff['p'],1e-10)and same(test['sum_zero']['p_native'],summed['p'],1e-10)
                and same(test['p_abs'],p_abs,1e-10)and same(table.loc[mid,scope+'_p_abs'],p_abs,1e-10))
            scope_results[scope]=dict(beta_positive=pair[0],beta_negative=pair[1],p_difference=diff['p'],p_sum=summed['p'],p_abs=p_abs,
                expected_directional_signs=bool(pair[0]>=0 and pair[1]<=0),SE_difference=diff['se'],SE_sum=summed['se'])
        numerical[mid]=dict(family=spec['family'],group=spec['group'],variant=spec['variant'],outcome_column=spec['outcome_column'],
            nobs=result['nobs'],funds=result['funds'],companies=result['companies'],df_resid=df,within_R2=result['within_R2'],
            source_row_hash=metadata[spec['sample_id']]['ordered_row_hash'],controls=spec['controls'],H2=scope_results)
        if not np.isfinite(se).all():warnings.append(dict(model_id=mid,kind='UNDEFINED_NATIVE_COEFFICIENT_INFERENCE',terms=[order[j]for j in np.where(~np.isfinite(se))[0]]))
        print(json.dumps(dict(state='INDEPENDENT_SHORTLIST_MODEL',completed=index+1,total=len(all_native),model_id=mid)),flush=True)
    for key0,old in source_before.items():
        path=source_paths[key0]
        check('source_preserved_after_QA:'+key0,helper.sha(path)==old)
    for filename,old in generated_before.items():check('derived_inputs_preserved_after_QA:'+filename,helper.sha(root/'data'/filename)==old)
    failed=sum(c['status']=='FAIL'for c in checks)
    result=dict(status='FAIL'if failed else('PASS_WITH_WARNINGS'if warnings else'PASS'),created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        plan_sha256=helper.sha(planpath),checks=checks,pass_count=len(checks)-failed,fail_count=failed,warnings=warnings,
        all_grid_model_count=660,unique_fit_count=652,numerical_replica_count=len(numerical),numerical_replicas=numerical,
        family_counts_and_minima=count_records,all_twelve_C_independently_replicated=sum(r['family']=='C'for r in numerical.values())==12,
        post_grid_shared_HHI_models=sorted(mandatory_shared),all_shared_HHI_independently_replicated=mandatory_shared.issubset(numerical),
        duplicate_grid_or_native_refits_run=False,
        numerical_scope='Independent sample/lag/support/cut/outcome reconstruction; all660/1332 mechanical inference/selection checks; all saved native shortlist unique models independently replicated using rowwise within OLS and fund/company/intersection CGM.',
        interpretation='A/B retain the raw-percent outcome with explicit sample/design changes. C changes outcome scale and/or continuing composition and is reported separately. Current primary, lag/total separate, all selected minima exploratory and unadjusted.')
    output.write_text(json.dumps(helper.clean(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],pass_count=result['pass_count'],fail_count=failed,numerical_replicas=len(numerical))))


def boundary_recheck(args):
    """Correct one floating-boundary check; reuse the immutable61 replicas."""
    if os.name!='nt':raise RuntimeError('Research-data QA runs only on research-host')
    helper=load_module(args.rowwise_helper,'independent_rowwise_utilities')
    planpath=Path(args.plan);plan=json.loads(planpath.read_text(encoding='utf-8'))
    root=Path(args.root or plan['output_root']);oldpath=root/'qa_independent.json'
    output=Path(args.output or root/'qa_independent_v2.json')
    if output.exists():raise FileExistsError('Refusing to overwrite v2 QA')
    old=json.loads(oldpath.read_text(encoding='utf-8'));old_sha=helper.sha(oldpath)
    failed=[c['name']for c in old['checks']if c['status']=='FAIL']
    assert failed==['symmetric_entries_exits_and_continuing_log_missingness'],failed
    assert old['numerical_replica_count']==61 and old['all_shared_HHI_independently_replicated']
    assert old['all_twelve_C_independently_replicated'] and old['plan_sha256']==helper.sha(planpath)
    manifest=json.loads((root/'source_manifest.json').read_text(encoding='utf-8'))
    for key,expected in plan['expected_source_sha256'].items():assert helper.sha(plan[key])==expected,key
    for definition in plan['exact_samples']:
        assert helper.sha(definition['input_panel'])==definition['expected_source_sha256'],definition['group']
    for filename,metadata in manifest['generated_data'].items():
        assert helper.sha(root/'data'/filename)==metadata['sha256'],filename
    output_manifest=json.loads((root/'output_manifest.json').read_text(encoding='utf-8'))
    for filename in ['native_results.json','all_target_covariance.json','all_models.csv','shortlist.csv','model_catalog.json']:
        assert helper.sha(root/filename)==output_manifest[filename]['sha256'],filename
    q=pd.read_parquet(root/'data'/'canonical_signed_panel.parquet',columns=['row_id','quantity_open','quantity_close'])
    alt=pd.read_parquet(root/'data'/'alternative_outcomes.parquet')
    assert np.array_equal(q.row_id,alt.row_id)
    q0=q.quantity_open.to_numpy(float);q1=q.quantity_close.to_numpy(float)
    entry=(q0==0)&(q1>0);exit=(q0>0)&(q1==0);continuing=(q0>0)&(q1>0)
    expected=200*(q1-q0)/(q1+q0);actual=alt.h2_y_symmetric.to_numpy(float)
    tolerance=2*np.spacing(200.)
    evidence=dict(tolerance_absolute=tolerance,tolerance_ULP_at_200=2,entry_rows=int(entry.sum()),exit_rows=int(exit.sum()),
        numpy_entry_nonexact_count=int(np.sum(expected[entry]!=200)),numpy_exit_nonexact_count=int(np.sum(expected[exit]!=-200)),
        numpy_max_entry_error=float(np.max(abs(expected[entry]-200))),numpy_max_exit_error=float(np.max(abs(expected[exit]+200))),
        actual_max_entry_error=float(np.max(abs(actual[entry]-200))),actual_max_exit_error=float(np.max(abs(actual[exit]+200))),
        actual_log_missing_outside_continuing=bool(alt.loc[~continuing,'h2_y_continuing_log'].isna().all()))
    assert np.allclose(actual,expected,rtol=1e-12,atol=1e-10)
    assert np.allclose(actual[entry],200,rtol=0,atol=tolerance)
    assert np.allclose(actual[exit],-200,rtol=0,atol=tolerance)
    assert np.allclose(expected[entry],200,rtol=0,atol=tolerance)
    assert np.allclose(expected[exit],-200,rtol=0,atol=tolerance)
    assert evidence['actual_log_missing_outside_continuing']
    for record in old['checks']:
        if record['name']==failed[0]:record.update(status='PASS',boundary_recheck=evidence)
    old.update(status='PASS_WITH_WARNINGS'if old['warnings']else'PASS',pass_count=len(old['checks']),fail_count=0,
        qa_version=2,original_failed_QA_preserved=True,original_failed_QA_sha256=old_sha,
        original_failed_QA_path=str(oldpath),v2_repair_scope='One exact-equality boundary check corrected to2ULP absolute tolerance at200; unchanged formula/data and all previously passed checks retained.',
        v2_created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),v2_boundary_evidence=evidence,
        frozen_source_derived_and_native_result_bytes_revalidated=True,replicas_recomputed_in_v2=False)
    output.write_text(json.dumps(helper.clean(old),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    assert helper.sha(oldpath)==old_sha
    print(json.dumps(dict(status=old['status'],pass_count=old['pass_count'],fail_count=0,numerical_replicas=old['numerical_replica_count'],replicas_recomputed=False,boundary_evidence=evidence)))


def fixture():
    # This fixture contains no research data or model estimation.
    q0=np.array([0.,4.,4.,4.]);q1=np.array([4.,0.,8.,2.])
    symmetric=200*(q1-q0)/(q1+q0)
    assert np.array_equal(symmetric[:2],[200.,-200.])
    continuing=(q0>0)&(q1>0)
    log=np.full(4,np.nan);log[continuing]=100*(np.log(q1[continuing])-np.log(q0[continuing]))
    assert np.isnan(log[:2]).all() and np.isclose(log[2],-log[3])
    print(json.dumps(dict(status='PASS',scope='synthetic_only',checks=2)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--fixture-only',action='store_true')
    parser.add_argument('--plan');parser.add_argument('--root');parser.add_argument('--output')
    parser.add_argument('--rowwise-helper');parser.add_argument('--h2-qa-helper');parser.add_argument('--rtol',type=float,default=2e-6)
    parser.add_argument('--boundary-recheck-only',action='store_true')
    args=parser.parse_args()
    if args.fixture_only:fixture()
    elif args.boundary_recheck_only:boundary_recheck(args)
    else:audit(args)
