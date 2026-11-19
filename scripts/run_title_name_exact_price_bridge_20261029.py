"""One finite exact-window price bridge; all actual data and fits stay on research-host."""
import argparse
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(name,'2')
BASE=Path('./analysis_outputs/title_name_regressions_20261026_w2_v2')
PREVIOUS=Path('./analysis_outputs/title_name_hypothesis_research_20261029')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'code'));sys.path.insert(0,str(HERE))
import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from hypothesis_methods_20261029 import clean,save
from hypothesis_extended_20261029 import target_panel,quote_pairs,cash_labels,price_adjustment
from run_title_name_cluster_sensitivity_20261029 import nested_rank,cluster_covariance
from title_name_regressions_core import IntegrityError,NotEstimable,sha256_file,freeze_json,unique_keys
from title_name_regressions_estimation import (fit_panel_fe,prune_singletons,within_transform,
    h1_directional_test,h2_absolute_reaction_test)

CONTROLS=['opening_weight','lag_log_news_count','opening_log_NAV','opening_HHI','log_interval_days']
TARGETS=['S_positive','S_negative']
KEYS=['fund_id','security_id','start_date','end_date']
GROUPS=[('full_portfolio_AT',[('E1_AT_PRICE_EXACT_FULL','AT_PRICE'),('E1B_DW_AT_MATCHED','DW')],False),
        ('relative_position_growth',[('E2_LOG_VALUE_GROWTH_EXACT_MATCHED','log_value_growth'),
          ('E3_PRICE_ADJUSTED_VALUE_GROWTH_EXACT_MATCHED','price_adjusted_value_growth'),
          ('E4_LOG_PRICE_GROWTH_DIAGNOSTIC','log_price_growth')],True)]
BRIDGE_ROOT=Path('./analysis_outputs/title_name_exact_price_bridge_20261029')
REPAIR_ROOT=BRIDGE_ROOT/'v2_deterministic_absorption'


def announce(root,stage,**details):
    state=dict(stage=stage,pid=os.getpid(),updated_at_utc=datetime.now(timezone.utc).isoformat(),**details)
    save(root/'checkpoint.json',state);print(json.dumps(clean(state)),flush=True)


@contextmanager
def writer(root):
    with (root/'.bridge_writer.lock').open('a+b') as handle:
        handle.seek(0,2)
        if handle.tell()==0:handle.write(b'0');handle.flush()
        handle.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            handle.seek(0)
            if os.name=='nt':msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(handle,fcntl.LOCK_UN)


def frame_hash(frame,columns):
    payload=pd.util.hash_pandas_object(frame[columns],index=False).values.tobytes()
    return hashlib.sha256(payload).hexdigest()


def common_sample(frame,outcomes,relative):
    d=frame.copy().reset_index(drop=True)
    d['S_positive']=d.S.clip(lower=0);d['S_negative']=(-d.S).clip(lower=0)
    numbers=outcomes+TARGETS+CONTROLS
    required=list(dict.fromkeys(numbers+KEYS+['fund_security_id','interval_id','month','company_id']))
    complete=d[required].notna().all(axis=1)&np.isfinite(d[numbers]).all(axis=1)
    d=d.loc[complete].copy()
    complete_n=len(d)
    d,removed,rounds=prune_singletons(d,['fund_security_id','interval_id' if relative else 'month'])
    d=d.sort_values(KEYS).reset_index(drop=True)
    unique_keys(d,KEYS,'locked final common sample')
    return d,dict(input_rows=len(frame),complete_rows=complete_n,missing_rows=len(frame)-complete_n,
                  singleton_rows_removed=removed,singleton_rounds=rounds,final_rows=len(d),
                  sample_hash=frame_hash(d,KEYS),sealed_S_hash=frame_hash(d,KEYS+['S']))


def duration(frame):
    if not len(frame):return dict(rows=0)
    days=(frame.end_date-frame.start_date).dt.days
    return dict(rows=len(frame),intervals=int(frame.interval_id.nunique()),
                quantiles=days.quantile([0,.25,.5,.75,.9,.99,1]).to_dict(),
                bins={label:int(mask.sum()) for label,mask in [
                    ('<=35',days.le(35)),('36-93',days.between(36,93)),
                    ('94-366',days.between(94,366)),('>366',days.gt(366))]})


def structural_absorption(frame,effects,candidates=()):
    """Exact groupwise constancy, not a relaxed numerical rank tolerance."""
    removed={}
    for control in candidates:
        maxima={effect:int(frame.groupby(effect,observed=True)[control].nunique(dropna=False).max()) for effect in effects}
        if maxima.get('interval_id')!=1:raise IntegrityError('The fixed structural absorption proof failed: '+control)
        removed[control]=dict(reason='EXACTLY_CONSTANT_WITHIN_ABSORBED_FE',max_unique_by_FE=maxima)
    return [name for name in CONTROLS if name not in removed],removed


def reuse_prepared(root):
    """Preserve the original preparation identity; reuse its unchanged artifacts."""
    old=json.loads((BRIDGE_ROOT/'summary.json').read_text())
    if sha256_file(BRIDGE_ROOT/'summary.json')!='20142e0968ef404d4d85b1fce0552d8a8358c1b647f33ae87fb9a11b2d1fd436':
        raise IntegrityError('Historical first-attempt summary changed')
    names=['full_AT_common_final_sample.parquet','relative_common_final_sample.parquet',
           'accepted_price_pairs.parquet','price_coverage.csv','sample_flow.csv','cash_assumptions_evidence.json',
           'quote_source_hashes.json','feasibility.json']
    hashes={name:sha256_file(BRIDGE_ROOT/name) for name in names}
    for name in names:
        shutil.copyfile(BRIDGE_ROOT/name,root/name)
        if sha256_file(root/name)!=hashes[name]:raise IntegrityError('Prepared copy changed: '+name)
    pair=pd.read_parquet(root/'full_AT_common_final_sample.parquet')
    logs=pd.read_parquet(root/'relative_common_final_sample.parquet')
    feasibility=old['feasibility']
    for frame,key in [(pair,'full_AT_pair'),(logs,'relative_triple')]:
        info=feasibility[key]
        if len(frame)!=info['final_rows'] or frame_hash(frame,KEYS)!=info['sample_hash'] or frame_hash(frame,KEYS+['S'])!=info['sealed_S_hash']:
            raise IntegrityError('Frozen common rows/S changed')
    provenance=dict(status='REUSED_UNCHANGED_PREPARATION_NOT_RESTAMPED',
        original_summary_sha256=sha256_file(BRIDGE_ROOT/'summary.json'),original_run_manifest_sha256=sha256_file(BRIDGE_ROOT/'run_manifest.json'),
        original_preparation_code_hashes=old['manifest']['code_hashes'],artifact_hashes=hashes,
        source_root=str(BRIDGE_ROOT),change='Exact structural nuisance absorption in estimation only; no SQL or new price/S preparation')
    freeze_json(root/'reused_preparation_provenance.json',provenance)
    return pair,logs,feasibility,json.loads((root/'quote_source_hashes.json').read_text())


def prepare(root,plan):
    announce(root,'PREPARING_SEALED_EXACT_WINDOWS')
    source=pd.read_parquet(PREVIOUS/'exact_actual_window_panel.parquet')
    unique_keys(source,KEYS,'sealed exact-window source')
    inventory=pd.read_parquet(BASE/'holdings_inventory.parquet')
    intervals=pd.read_parquet(BASE/'all_true_intervals.parquet')
    facts=pd.read_parquet(BASE/'scheduled_corporate_events.parquet')
    ids=set(source.interval_id)
    selected=intervals.loc[intervals.interval_id.isin(ids)].copy()
    if set(selected.interval_id)!=ids:raise IntegrityError('Missing actual source interval')
    if not (selected.opening_full_portfolio&selected.closing_full_portfolio&~selected.opening_passive_explicit&~selected.closing_passive_explicit).all():
        raise IntegrityError('Source interval quality scope changed')
    targets,whole,audit=target_panel(inventory,selected,facts)
    key=lambda f:pd.MultiIndex.from_frame(f[['interval_id','security_id']])
    if set(key(targets))!=set(key(source)):raise IntegrityError('Target scope differs from sealed exact-window source')
    check=source[['interval_id','security_id','DW']].merge(targets[['interval_id','security_id','DW']],
        on=['interval_id','security_id'],validate='one_to_one',suffixes=('_sealed','_rebuilt'))
    if not np.allclose(check.DW_sealed,check.DW_rebuilt,atol=1e-12,rtol=1e-12):raise IntegrityError('Exact source weights changed')
    announce(root,'EXTRACTING_EXISTING_EXACT_PRICE_PAIRS',source_rows=len(source),intervals=len(selected),whole_portfolio_rows=len(whole))
    connection=duckdb.connect(plan['inputs']['catalog'],read_only=True,config={'threads':2,'memory_limit':'2GB'})
    try:pairs,quote_audit=quote_pairs(whole,connection);allowed_cash,cash_names=cash_labels(connection)
    finally:connection.close()
    pairs.to_parquet(root/'accepted_price_pairs.parquet',index=False,compression='zstd')
    quote_files=quote_audit.pop('quote_source_files',[])
    quote_sources={path:sha256_file(path) for path in quote_files}
    freeze_json(root/'quote_source_hashes.json',quote_sources)
    whole_priced,coverage=price_adjustment(whole,pairs,allowed_cash)
    coverage.to_csv(root/'price_coverage.csv',index=False)
    missing=whole_priced.loc[whole_priced.weight_open.gt(1e-12)&whole_priced.price_return.isna(),
        ['interval_id','security_id','asset_type','weight_open']].copy()
    missing['source_label']=missing.security_id.map(cash_names)
    missing.to_parquet(root/'uncovered_opening_positions.parquet',index=False,compression='zstd')
    save(root/'cash_assumptions_evidence.json',dict(allowed_RUB_zero_keys=sorted(allowed_cash),source_labels=cash_names))
    full=source.merge(whole_priced[['interval_id','security_id','AT_PRICE','eligible_full_price_AT']],
        on=['interval_id','security_id'],how='left',validate='one_to_one')
    full=full.loc[full.eligible_full_price_AT.eq(True)].copy()
    matched=source.merge(pairs,on=['security_id','start_date','end_date'],how='inner',validate='many_to_one')
    values=inventory[['report_id','security_id','amount_thousand_rub']]
    unique_keys(values,['report_id','security_id'],'inventory amounts')
    for boundary,label in [('opening_report_id','reported_value_open'),('closing_report_id','reported_value_close')]:
        matched=matched.merge(values.rename(columns={'report_id':boundary,'amount_thousand_rub':label}),
            on=[boundary,'security_id'],how='left',validate='many_to_one')
    continuing=matched.reported_value_open.gt(0)&matched.reported_value_close.gt(0)
    matched=matched.loc[continuing].copy()
    matched['log_value_growth']=np.log(matched.reported_value_close/matched.reported_value_open)
    matched['log_price_growth']=np.log(matched.price_close/matched.price_open)
    matched['price_adjusted_value_growth']=matched.log_value_growth-matched.log_price_growth
    pair,d1=common_sample(full,['AT_PRICE','DW'],False)
    logs,d2=common_sample(matched,['log_value_growth','price_adjusted_value_growth','log_price_growth'],True)
    pair.to_parquet(root/'full_AT_common_final_sample.parquet',index=False,compression='zstd')
    logs.to_parquet(root/'relative_common_final_sample.parquet',index=False,compression='zstd')
    unique_keys(pairs,['security_id','start_date','end_date'],'accepted quotes')
    if not (pairs.quote_open_date.le(pairs.start_date).all() and pairs.quote_close_date.le(pairs.end_date).all()
            and pairs.quote_open_age_days.between(0,7).all() and pairs.quote_close_age_days.between(0,7).all()):
        raise IntegrityError('Future or stale quote accepted')
    for frame in [pair,logs]:
        bound=frame[KEYS+['S']].merge(source[KEYS+['S']],on=KEYS,validate='one_to_one',suffixes=('_final','_sealed'))
        if not np.array_equal(bound.S_final.to_numpy(),bound.S_sealed.to_numpy()):raise IntegrityError('Sealed S changed')
    identity_error=float((logs.log_value_growth-logs.price_adjusted_value_growth-logs.log_price_growth).abs().max()) if len(logs) else 0.0
    if identity_error>1e-12:raise IntegrityError('Outcome price decomposition failed')
    rows=[dict(stage='sealed_exact_target_rows',n=len(source)),dict(stage='sealed_S_defined',n=int(source.S.notna().sum())),
        dict(stage='exact_source_intervals',n=len(selected)),dict(stage='whole_union_inventory_rows',n=len(whole)),
        dict(stage='accepted_instrument_interval_price_pairs',n=len(pairs)),dict(stage='full_price_portfolio_intervals',n=int(coverage.eligible_full_price_AT.sum())),
        dict(stage='full_AT_target_rows_before_common_complete_case',n=len(full)),dict(stage='continuing_priced_rows_before_common_complete_case',n=len(matched)),
        dict(stage='full_AT_final_common_rows',n=len(pair)),dict(stage='relative_final_common_rows',n=len(logs))]
    pd.DataFrame(rows).to_csv(root/'sample_flow.csv',index=False)
    diag=dict(source_rows=len(source),source_intervals=len(selected),whole_target_audit=audit,
        quote_audit=quote_audit,quote_source_files_hashed=len(quote_sources),whole_portfolio_intervals=len(coverage),
        fully_price_covered_intervals=int(coverage.eligible_full_price_AT.sum()),
        coverage_fraction_quantiles=coverage.coverage_fraction.quantile([0,.25,.5,.75,1]).to_dict(),
        full_AT_pair=d1,relative_triple=d2,outcome_decomposition_max_abs_error=identity_error,
        source_duration=duration(source),full_AT_final_duration=duration(pair),relative_final_duration=duration(logs),sample_flow=rows,
        full_AT_entries_retained=int(pair.full_entry.sum()) if len(pair) else 0,
        full_AT_exits_retained=int(pair.full_exit.sum()) if len(pair) else 0,
        log_estimand_excludes_all_entries_and_exits=True)
    save(root/'feasibility.json',diag)
    return pair,logs,diag,quote_sources


def holm(tests,key):
    order=sorted(range(len(tests)),key=lambda i:1 if tests[i]['p'] is None else tests[i]['p'])
    last=0.0
    for rank,index in enumerate(order):
        value=tests[index]['p'];last=max(last,min(1.,(len(tests)-rank)*(1 if value is None else value)))
        tests[index][key]=None if value is None else last


def empty_cells(specs,family,frame,sample_info,reason):
    cells=[]
    for identifier,outcome in specs:
        diagnostic=identifier.startswith('E4_')
        for clusters in [['fund_id','company_id'],['fund_id','company_id','month']]:
            for convention in ['full_FE_rank','nested_FE_adjusted']:
                cell=identifier+'__'+'_'.join(clusters)+'__'+convention
                tests=[dict(cell=cell,model=identifier,family='price_component_diagnostic' if diagnostic else family,test=t,p=None,
                    reason=reason) for t in TARGETS+([] if diagnostic else ['H1_directional_IUT','H2_absolute_union_null'])]
                cells.append(dict(cell=cell,model=identifier,outcome=outcome,status='NOT_ESTIMABLE',reason=reason,
                    family='price_component_diagnostic' if diagnostic else family,clusters=clusters,convention=convention,
                    N=len(frame),sample_hash=sample_info['sample_hash'],sealed_S_hash=sample_info['sealed_S_hash'],
                    coefficients=[],tests=tests))
    return cells,dict(status='NOT_ESTIMABLE',reason=reason,common_sample=sample_info)


def estimate_group(root,frame,specs,family,relative,sample_info):
    if not len(frame):return empty_cells(specs,family,frame,sample_info,'No rows after joint finite-mask and identical singleton gates')
    time_effect='interval_id' if relative else 'month'
    fixed_absorbed=['opening_log_NAV','opening_HHI','log_interval_days'] if relative else []
    kept_controls,structural_drops=structural_absorption(frame,['fund_security_id',time_effect],fixed_absorbed)
    try:
        template=fit_panel_fe(frame,specs[0][1],TARGETS+kept_controls,time=time_effect,allow_absorbed=kept_controls,
                              allow_collinear=kept_controls,drop_singletons=False)
    except NotEstimable as error:return empty_cells(specs,family,frame,sample_info,str(error))
    if len(template.sample)!=len(frame) or not template.sample.index.equals(frame.index):raise IntegrityError('Final sample changed inside fit')
    names=template.names
    all_outcomes=[outcome for identifier,outcome in specs]
    # One common projection/operator for all outcomes guarantees the linear decomposition.
    transformed,iterations=within_transform(frame[all_outcomes+names].to_numpy(float),[frame.fund_security_id,frame[time_effect]])
    ys=transformed[:,:len(specs)];x=transformed[:,len(specs):]
    betas=np.linalg.lstsq(x,ys,rcond=None)[0]
    reference_error=float(np.max(np.abs(betas[:,0]-template.beta)))
    if reference_error>1e-8:raise IntegrityError('Shared projection differs from verified estimator')
    xhash=hashlib.sha256(np.asarray(x,dtype='<f8').tobytes()).hexdigest()
    cells=[];allbeta={};cov_ref_errors=[]
    for index,(identifier,outcome) in enumerate(specs):
        announce(root,'ESTIMATING_LOCKED_BRIDGE_MODEL',model=identifier,N=len(frame),family=family)
        beta=betas[:,index];residual=ys[:,index]-x@beta;allbeta[identifier]=dict(zip(names,beta))
        mass=float(np.dot(ys[:,index],ys[:,index]));r2=1-float(np.dot(residual,residual))/mass if mass>0 else None
        diagnostic=identifier.startswith('E4_');thisfamily='price_component_diagnostic' if diagnostic else family
        cache={}
        for clusters in [['fund_id','company_id'],['fund_id','company_id','month']]:
            rank,nesting=nested_rank(frame,['fund_security_id',time_effect],clusters)
            for convention in ['full_FE_rank','nested_FE_adjusted']:
                a=template.diagnostics['fe_rank'] if convention=='full_FE_rank' else rank
                covariance,counts,q,denominator=cluster_covariance(frame,x,residual,clusters,a,cache)
                df=min(counts[c] for c in clusters)-1
                diag=dict(template.diagnostics,within_r2=r2,inference_df=df,cluster_dimensions=clusters,
                    covariance_min_eigenvalue=float(np.linalg.eigvalsh(covariance).min()),
                    negative_coefficient_variances=int((np.diag(covariance)<0).sum()))
                current=replace(template,beta=beta,covariance=covariance,sample=frame,residuals=residual,diagnostics=diag)
                if index==0 and clusters==['fund_id','company_id'] and convention=='full_FE_rank':
                    difference=float(np.max(np.abs(covariance-template.covariance)));cov_ref_errors.append(difference)
                    if difference>1e-8:raise IntegrityError('Shared operator CGM covariance differs from verified estimator')
                cell=identifier+'__'+'_'.join(clusters)+'__'+convention
                coeffs=[];tests=[]
                for row in current.table():
                    row=clean(row);scale=100 if relative else 10000
                    row.update(cell=cell,model=identifier,family=thisfamily,outcome=outcome,
                        display_units='log percentage points' if relative else 'weight basis points',
                        effect_display=row['beta']*scale,se_display=None if row['se'] is None else row['se']*scale,
                        ci95_low_display=None if row['ci95_low'] is None else row['ci95_low']*scale,
                        ci95_high_display=None if row['ci95_high'] is None else row['ci95_high']*scale)
                    if relative:
                        row['exact_percent_change']=float(100*np.expm1(row['beta']))
                        row['exact_percent_CI']=[None if row[c] is None else float(100*np.expm1(row[c])) for c in ['ci95_low','ci95_high']]
                    coeffs.append(row)
                    if row['term'] in TARGETS:tests.append(dict(cell=cell,model=identifier,family=thisfamily,test=row['term'],p=row['p']))
                h1=h1_directional_test(current) if not diagnostic else None
                h2=h2_absolute_reaction_test(current) if not diagnostic else None
                if not diagnostic:
                    tests += [dict(cell=cell,model=identifier,family=thisfamily,test='H1_directional_IUT',p=h1.get('intersection_union_p')),
                              dict(cell=cell,model=identifier,family=thisfamily,test='H2_absolute_union_null',p=h2.get('p_abs'))]
                    holm(tests,'p_holm_four_tests_in_cell')
                cells.append(dict(cell=cell,model=identifier,outcome=outcome,family=thisfamily,status='ESTIMATED_POSTHOC_PRICE_BRIDGE',
                    N=len(frame),funds=int(frame.fund_id.nunique()),companies=int(frame.company_id.nunique()),
                    securities=int(frame.security_id.nunique()),months=int(frame.month.nunique()),intervals=int(frame.interval_id.nunique()),
                    period_start=str(frame.month.min()),period_end=str(frame.month.max()),FE=['fund_security_id',time_effect],
                    regressors=names,absorbed_controls=list(structural_drops)+template.diagnostics['absorbed'],
                    structural_absorption=structural_drops,collinear_controls=template.diagnostics['collinear_nuisance_dropped'],
                    sample_hash=sample_info['sample_hash'],sealed_S_hash=sample_info['sealed_S_hash'],common_X_hash=xhash,
                    clusters=clusters,cluster_counts=counts,inference_df=df,full_FE_rank=template.diagnostics['fe_rank'],
                    correction_FE_rank=a,small_sample_q=q,correction_denominator=denominator,nesting=nesting,
                    covariance_min_eigenvalue=diag['covariance_min_eigenvalue'],negative_coefficient_variances=diag['negative_coefficient_variances'],
                    within_R2=r2,H1=h1,H2=h2,coefficients=coeffs,tests=tests))
    decomp=None
    if relative:
        difference=betas[:,0]-betas[:,1]-betas[:,2]
        decomp=dict(status='PASS' if np.max(np.abs(difference))<=1e-10 else 'FAIL',
            maximum_absolute_coefficient_error=float(np.max(np.abs(difference))),
            by_regressor={name:dict(raw=float(betas[i,0]),adjusted=float(betas[i,1]),price=float(betas[i,2]),difference=float(difference[i])) for i,name in enumerate(names)},
            rule='beta(log value)=beta(price-adjusted value)+beta(log price); SE, p and R2 do not add')
        if decomp['status']!='PASS':raise IntegrityError('Coefficient decomposition failed')
    return cells,dict(status='ESTIMATED',common_sample=sample_info,common_X_hash=xhash,regressors=names,
        structural_absorption=structural_drops,
        FE=['fund_security_id',time_effect],projection_iterations=iterations,
        verified_estimator_reference_beta_error=reference_error,reference_covariance_max_errors=cov_ref_errors,
        coefficient_decomposition=decomp)


def tables(root,output):
    pd.DataFrame([row for record in output['records'] for row in record['coefficients']]).to_csv(root/'coefficients.csv',index=False)
    pd.DataFrame(output['tests']).to_csv(root/'tests.csv',index=False)
    lines=['# Exact-window price bridge — 29 October 2026','','Fixed post-hoc mechanism check. No new S fit, LLM run or raw download.','',
        '| Model | Clusters | FE DoF | N | Positive effect | p+ | Negative effect | p− | H1 p | H2 p | Display units |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    fmt=lambda value:'NA' if value is None else f'{value:.6g}'
    for r in output['records']:
        row={v['term']:v for v in r['coefficients']}
        a=row.get('S_positive',{});b=row.get('S_negative',{})
        lines.append('| '+' | '.join([r['model'],'+'.join(r['clusters']),r['convention'],str(r['N']),fmt(a.get('effect_display')),fmt(a.get('p')),
            fmt(b.get('effect_display')),fmt(b.get('p')),fmt((r.get('H1') or {}).get('intersection_union_p')),fmt((r.get('H2') or {}).get('p_abs')),a.get('display_units',r.get('reason','NA'))])+' |')
    lines+=['','All coefficients, CI and per-cell/family/global Holm values are in the accompanying CSV/JSON.',
        'E1 is price-only allocation under accepted cash/payout assumptions, not verified total-return AT. E3 is implied position growth on continuing positions, not actual units. E4 is diagnostic, not alpha.']
    (root/'tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();root=args.output
    if os.name!='nt':raise IntegrityError('Actual bridge execution belongs on research-host')
    if root not in [BRIDGE_ROOT,REPAIR_ROOT]:raise IntegrityError('Unexpected output scope')
    root.mkdir(parents=True,exist_ok=True)
    if (root/'summary.json').exists():raise IntegrityError('Completed bridge is immutable; never rerun')
    start=time.monotonic()
    with writer(root):
        try:
            announce(root,'VERIFYING_FROZEN_PLAN_AND_METHOD_GATE')
            plan=json.loads((root/'config/plan.json').read_text(encoding='utf-8-sig'))
            gate=json.loads((root/'config/method_review.json').read_text(encoding='utf-8-sig'))
            if plan.get('status')!='FIXED_BEFORE_NEW_BRIDGE_OUTCOMES' or gate.get('ready_for_execution') is not True or gate.get('blocking_findings'):
                raise IntegrityError('No final approved method gate; estimation must not proceed')
            if gate['plan']['sha256']!=sha256_file(root/'config/plan.json') or gate['authority']['sha256']!=sha256_file(root/'config/assumptions.json'):
                raise IntegrityError('Gate binds a different plan or assumptions')
            paths=[PREVIOUS/'exact_actual_window_panel.parquet',BASE/'holdings_inventory.parquet',BASE/'all_true_intervals.parquet',
                BASE/'scheduled_corporate_events.parquet',BASE/'preparation.json',
                Path('./databaseresources/market/moex_quotes_2008_2026/mapping.csv'),
                Path('./databaseresources/funds/mac-20261008/fund_holdings_long.csv.gz')]
            sources={str(path):sha256_file(path) for path in paths}
            code_hashes={path.name:sha256_file(path) for path in HERE.glob('*.py')}
            base_code={name:sha256_file(BASE/'code'/name) for name in ['title_name_regressions_core.py','title_name_regressions_estimation.py','title_name_regressions_crosswalk.py','title_name_regressions_data.py']}
            manifest=dict(plan_sha256=sha256_file(root/'config/plan.json'),method_gate_sha256=sha256_file(root/'config/method_review.json'),
                assumptions_sha256=sha256_file(root/'config/assumptions.json'),source_hashes=sources,code_hashes=code_hashes,base_code_hashes=base_code,
                created_at_utc=datetime.now(timezone.utc).isoformat(),sample_policy='Joint finite outcome/control mask, identical singleton pruning, one common FE projection per matched group',
                covariance_policy='Component-specific CGM cluster multipliers; full and nested-FE parameter counts; no PSD clipping')
            freeze_json(root/'run_manifest.json',manifest)
            pair,logs,feasibility,quote_sources=reuse_prepared(root) if root==REPAIR_ROOT else prepare(root,plan)
            announce(root,'PREPARED_FIXED_PRICE_BRIDGE',full_AT_common_rows=len(pair),relative_common_rows=len(logs))
            records=[];group_checks={}
            for family,specs,relative in GROUPS:
                frame=logs if relative else pair
                info=feasibility['relative_triple' if relative else 'full_AT_pair']
                cells,check=estimate_group(root,frame,specs,family,relative,info)
                records+=cells;group_checks[family]=check
                save(root/(family+'_results.json'),dict(records=cells,validation=check))
            tests=[test for record in records for test in record['tests']]
            if len(records)!=20 or len(tests)!=72:raise IntegrityError('Frozen covariance/test-slot list changed')
            for family,size in [('full_portfolio_AT',32),('relative_position_growth',32),('price_component_diagnostic',8)]:
                selected=[test for test in tests if test['family']==family]
                if len(selected)!=size:raise IntegrityError('Frozen family size changed')
                holm(selected,'p_holm_fixed_family')
            holm(tests,'p_holm_all_72_locked_tests')
            for name,digest in sources.items():
                if sha256_file(name)!=digest:raise IntegrityError('An immutable bridge input changed during execution')
            for name,digest in quote_sources.items():
                if sha256_file(name)!=digest:raise IntegrityError('Quote source changed during execution')
            for name,digest in code_hashes.items():
                if sha256_file(HERE/name)!=digest:raise IntegrityError('Frozen code snapshot changed')
            output=dict(status='EXACT_PRICE_BRIDGE_COMPLETED',client_date='2026-10-29',output_root=str(root),
                completed_at_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,
                manifest=manifest,quote_source_manifest_sha256=sha256_file(root/'quote_source_hashes.json'),
                feasibility=feasibility,group_validation=group_checks,records=records,tests=tests,
                estimated_model_count=len({r['model'] for r in records if r['status'].startswith('ESTIMATED')}),
                estimated_covariance_cells=sum(r['status'].startswith('ESTIMATED') for r in records),
                fixed_covariance_cells=20,fixed_test_slots=72,source_contents_reverified=True,
                no_new_S_fit=True,no_new_LLM_run=True,new_external_data=False,
                limitations=['Fixed post-hoc price mechanism check, not untouched confirmatory evidence',
                    'Price-only allocation omits payouts under explicit user approximation; unknown security returns are never zero',
                    'Continuing-position log proxy is not actual quantities or full-portfolio AT; accrual/valuation/split comparability remains imperfect',
                    'E4 supplies a linear coefficient difference, not additive SE/p/R2 or causal mediation',
                    'Actual 28–94 day disclosure windows are not a uniform monthly response',
                    'CGM uncertainty is conditional on sealed LLM labels and generated S; no H3 alpha is estimated'])
            save(root/'summary.json',output);tables(root,output)
            announce(root,'COMPLETED',models=output['estimated_model_count'],estimated_covariance_cells=output['estimated_covariance_cells'],
                fixed_test_slots=72,summary_sha256=sha256_file(root/'summary.json'),elapsed_seconds=output['elapsed_seconds'])
        except Exception as error:
            announce(root,'FAILED',error=repr(error));raise


if __name__=='__main__':main()
