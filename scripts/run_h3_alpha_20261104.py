"""Resumable H3 fund-period alpha/panel execution on research-host only.

First sealed dependency release is pinned. No source inputs or earlier results
are modified. No new LLM, news scraping, outcome trimming or control search.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from h3_alpha_core_20261104 import numeric, truth, unique, rolling_future_alpha, join_fund_alpha, digest_lines
from h3_alpha_io_20261104 import ROOT, sha, utc, save_json, checkpoint, file_record, seal_directory
from fit_h3_alpha_fe_20261104 import fit_fund_fe, CURRENT, LAG, synthetic_tests

SPEC_SHA='565a726fd3858b94b2bfb3dbf43442d8cab4812c79749fd2ea24db806c44e811'
GROUPS=['pooled','equity','bond','mixed']
BRANCHES=['TR_VERIFIED','UNIT_VALUE_KNOWN_ADJUSTMENTS']


def prepare_returns(raw,cross,cash,snapshot):
    unique(raw,['fund_id','month'],'sealed unit returns')
    unique(cross,['fund_id'],'dated fund identity')
    d=raw.copy();d['fund_id']=d.fund_id.astype(str)
    for c in ['start_valuation_date','end_valuation_date','identity_valid_from','identity_valid_to']:
        d[c]=pd.to_datetime(d[c])
    periods=pd.PeriodIndex(d.month,freq='M')
    expected_start=(periods-1).astype(str)
    expected_end=periods.astype(str)
    adjacent=(d.start_valuation_date.dt.to_period('M').astype(str).to_numpy()==expected_start)
    adjacent&=d.end_valuation_date.dt.to_period('M').astype(str).to_numpy()==expected_end
    observed=numeric(d.unit_value)/numeric(d.previous_unit_nav)-1.
    verified=truth(d.return_verified)
    if not np.allclose(observed.loc[verified],numeric(d.loc[verified,'fund_return']),rtol=1e-12,atol=1e-12):
        raise AssertionError('Sealed returns not reconstructed from observed per-unit values')
    identity=(truth(d.identity_verified)&d.start_valuation_date.ge(d.identity_valid_from)
              &d.end_valuation_date.le(d.identity_valid_to))
    positive=numeric(d.unit_value).gt(0)&numeric(d.previous_unit_nav).gt(0)
    rub=d.currency.fillna('').str.upper().isin(['RUB','RUR','SUR'])
    if snapshot.get('known_cash_exclusion_scope')=='source_flagged_record_payment_months':
        affected=set()
        for column in ['record_date','payment_date']:
            for fund,date in cash.loc[cash[column].notna(),['fund_id',column]].itertuples(index=False,name=None):
                affected.add((str(fund),str(pd.Timestamp(date).to_period('M'))))
        reconstructed=pd.Series([(str(f),str(m)) in affected for f,m in zip(d.fund_id,d.month)],index=d.index)
        if 'known_distribution_date_unresolved' not in d:
            raise AssertionError('Action-aware release missing explicit known-event month flag')
        if not reconstructed.equals(truth(d.known_distribution_date_unresolved)):
            raise AssertionError('Sealed known distribution months disagree with raw record/payment evidence')
        if (verified&reconstructed).any():
            raise AssertionError('Sealed action-aware return is verified in an unresolved event month')
        d['known_unresolved_cash_event']=reconstructed
    else:
        known_cash_funds=set(cash.fund_id.astype(str)) if len(cash) else set()
        d['known_unresolved_cash_event']=d.fund_id.isin(known_cash_funds)
    d['known_unresolved_split_event']=False
    for event in snapshot.get('known_fund_splits',[]):
        if event.get('source_already_backadjusted'):
            if event.get('apply_additional_split_adjustment') is not False:
                raise AssertionError('Back-adjusted source cannot receive a duplicate split correction')
            continue
        date=pd.Timestamp(event['event_date'])
        crossing=d.fund_id.eq(str(event['fund_id']))&d.start_valuation_date.lt(date)&d.end_valuation_date.ge(date)
        d.loc[crossing,'known_unresolved_split_event']=True
    eligible=verified&identity&positive&rub&adjacent&np.isfinite(numeric(d.fund_return))
    eligible&=~d.known_unresolved_cash_event&~d.known_unresolved_split_event
    d['diagnostic_eligible']=eligible
    d['TR_eligible']=eligible&truth(d.total_return_verified)&truth(d.distribution_verified)&truth(d.split_verified)
    cash_reason=('known_distribution_record_payment_month_unresolved' if snapshot.get('known_cash_exclusion_scope')
        =='source_flagged_record_payment_months' else 'known_cash_events_unlocated_exdate_fund_series_excluded')
    d['adapter_exclusion_reason']=np.select([
        d.known_unresolved_cash_event,d.known_unresolved_split_event,~verified,~identity,~positive,~rub,~adjacent],
        [cash_reason,'known_split_comparability_unresolved','source_monthly_return_unverified',
         'dated_endpoint_identity_unverified','nonpositive_unit_value','unit_currency_not_verified_RUB',
         'monthly_endpoints_not_adjacent'],default='eligible_observed_unit_value')
    d['return_period_end']=periods.end_time.normalize()
    d['raw_return_verified']=verified
    return d


def dated_mandate_evidence(origins):
    """Read only the exact opening disclosures, not a latest-name back-projection."""
    import duckdb
    con=duckdb.connect('./local/research-data-service/research.duckdb',read_only=True)
    columns={r[0] for r in con.execute('DESCRIBE funds.reports').fetchall()}
    optional=[c for c in ['fund_name_file','fund_name_xls','fund_type_category','fund_strategy','investment_strategy'] if c in columns]
    con.register('h3_origins',origins[['fund_id','opening_report_id']].drop_duplicates())
    wanted=','.join('r."'+c+'"' for c in optional)
    rows=con.execute('SELECT r.fund_id,r.report_id'+(','+wanted if wanted else '')+
        ' FROM funds.reports r JOIN h3_origins o ON r.fund_id=o.fund_id AND r.report_id=o.opening_report_id').fetchdf()
    con.close()
    unique(rows,['fund_id','report_id'],'dated opening mandate evidence')
    rows['fund_id']=rows.fund_id.astype(str)
    text=rows[optional].fillna('').astype(str).agg(' | '.join,axis=1).str.lower() if optional else pd.Series('',index=rows.index)
    # Literal explicit incompatible geographic/commodity mandates only. A vague
    # word like "global" or an ISIN prefix is not used as proof of geography.
    pattern=r'зарубеж|иностранн|(?:\bсша\b)|американск|китайск|китая|азиатск|европейск|японск|индийск|драгоценн|золото|золотых|недвижим|сырьев|товарных'
    rows['explicit_incompatible_mandate']=text.str.contains(pattern,regex=True,na=False)
    rows['mandate_evidence_text']=text
    rows['mandate_geography_otherwise']='unknown_not_inferred_from_ISIN'
    return rows.rename(columns={'report_id':'opening_report_id'})


def make_origins(nts,cross,mandates):
    d=nts.copy();d['fund_id']=d.fund_id.astype(str)
    cross=cross.copy();cross['fund_id']=cross.fund_id.astype(str)
    cols=['fund_id','identity_unambiguous','identity_observed_from','identity_observed_to','registration_number']
    d=d.merge(cross[cols],on='fund_id',how='left',validate='many_to_one')
    for c in ['end_date','identity_observed_from','identity_observed_to']:
        d[c]=pd.to_datetime(d[c])
    d['identity_origin_verified']=truth(d.identity_unambiguous)&d.end_date.ge(d.identity_observed_from)&d.end_date.le(d.identity_observed_to)
    d=d.merge(mandates,on=['fund_id','opening_report_id'],how='left',validate='many_to_one')
    d['explicit_incompatible_mandate']=truth(d.explicit_incompatible_mandate)
    d['universe_eligible']=truth(d.universe_eligible)&~d.explicit_incompatible_mandate
    d['row_id']='H3F:'+d.fund_id.astype(str)+':'+d.interval_id.astype(str)
    return d


def compute_alpha(origins,returns,factors,output,*,branch,history=60,minimum=36,credit=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    definitions={'equity_return':'minus_rf','bond_return':'minus_rf'}
    if credit:definitions['credit_factor']='already_excess_or_spread'
    eligible_col='TR_eligible' if branch=='TR_VERIFIED' else 'diagnostic_eligible'
    r=returns.copy();r['return_verified']=truth(r[eligible_col])
    candidate=truth(origins.universe_eligible)&truth(origins.nts_pair_eligible)
    # Score eligibility is fixed before alpha; uncomputed origins remain in the
    # selection audit, never silently disappear from the available universe.
    work=origins.loc[candidate].copy()
    merged=[];member_paths=[]
    if r.return_verified.sum()==0 or work.empty:
        rows=[]
        for o in work.itertuples(index=False):
            month=pd.Timestamp(o.end_date).to_period('M')
            for h in [1,2,3]:
                rows.append(dict(fund_id=str(o.fund_id),interval_id=o.interval_id,origin_date=o.end_date,
                    origin_month=str(month),horizon=h,alpha=np.nan,past_alpha=np.nan,n_past=0,n_past_recent=0,
                    status='no_verified_total_return_months' if branch=='TR_VERIFIED' else 'no_eligible_origins',
                    future_first_month=str(month+1),future_last_month=str(month+h),future_months_complete=0))
        a=pd.DataFrame(rows,columns=['fund_id','interval_id','origin_date','origin_month','horizon','alpha',
            'past_alpha','n_past','n_past_recent','status','future_first_month','future_last_month','future_months_complete'])
        a.to_parquet(output/'alpha.parquet',index=False,compression='zstd')
        return a
    funds=sorted(work.fund_id.astype(str).unique())
    chunks=[funds[i:i+20] for i in range(0,len(funds),20)]
    for i,fs in enumerate(chunks):
        prefix=output/f'chunk_{i:04d}'
        done=prefix.with_suffix('.done.json')
        if done.exists():
            recorded=json.loads(done.read_text(encoding='utf-8'))
            file_record(prefix.with_suffix('.alpha.parquet'),recorded['alpha_sha256'])
            file_record(prefix.with_suffix('.training.parquet'),recorded['training_sha256'])
            a=pd.read_parquet(prefix.with_suffix('.alpha.parquet'))
        else:
            batch=work.loc[work.fund_id.astype(str).isin(fs),['fund_id','interval_id','end_date','identity_origin_verified']]
            a,m=rolling_future_alpha(batch,r.loc[r.fund_id.astype(str).isin(fs)],factors,
                factor_definitions=definitions,lookback_months=history,minimum_history=minimum,
                minimum_recent=9,recent_months=12,horizons=(1,2,3),condition_max=10000,include_memberships=True)
            a.to_parquet(prefix.with_suffix('.alpha.parquet'),index=False,compression='zstd')
            m.to_parquet(prefix.with_suffix('.training.parquet'),index=False,compression='zstd')
            save_json(done,dict(funds=fs,alpha_sha256=sha(prefix.with_suffix('.alpha.parquet')),
                               training_sha256=sha(prefix.with_suffix('.training.parquet')),completed_utc=utc()))
        merged.append(a);member_paths.append(str(prefix.with_suffix('.training.parquet')))
        checkpoint('ROLLING_ALPHA',branch=branch,lookback=history,minimum=minimum,credit=credit,
            completed_chunks=i+1,total_chunks=len(chunks),completed_funds=min((i+1)*20,len(funds)),
            total_funds=len(funds),eligible_alpha_so_far=sum(int(x.status.eq('eligible').sum()) for x in merged),
            output=str(output))
    a=pd.concat(merged,ignore_index=True)
    unique(a,['fund_id','interval_id','horizon'],'computed alpha')
    a.to_parquet(output/'alpha.parquet',index=False,compression='zstd')
    save_json(output/'training_membership_manifest.json',dict(paths=member_paths))
    # Future legs are explicit; not an extra security-level regression sample.
    legs=[]
    for row in a.loc[a.alpha.notna()].itertuples(index=False):
        for month in pd.period_range(row.future_first_month,row.future_last_month,freq='M'):
            legs.append(dict(fund_id=row.fund_id,interval_id=row.interval_id,horizon=row.horizon,month=str(month),
                past_alpha=row.past_alpha,**{f'beta_{c}':getattr(row,'beta_'+c) for c in definitions}))
    if legs:
        leg=pd.DataFrame(legs).merge(r[['fund_id','month','fund_return','start_valuation_date','end_valuation_date','source_path']],
            on=['fund_id','month'],how='left',validate='many_to_one').merge(factors,on='month',how='left',validate='many_to_one')
        leg['abnormal']=leg.fund_return-leg.risk_free
        for c,mode in definitions.items():
            leg['abnormal']-=leg['beta_'+c]*(leg[c]-leg.risk_free if mode=='minus_rf' else leg[c])
        verification=leg.groupby(['fund_id','interval_id','horizon'],observed=True).abnormal.mean()
        actual=a.loc[a.alpha.notna()].set_index(['fund_id','interval_id','horizon']).alpha
        if not np.allclose(verification.reindex(actual.index),actual,rtol=1e-10,atol=1e-12):
            raise AssertionError('Future monthly legs do not reconstruct reported alpha')
        leg.to_parquet(output/'future_monthly_legs.parquet',index=False,compression='zstd')
    return a


def estimate_grid(origins,alpha,out,branch,sensitivity='PRIMARY',groups=GROUPS,variants=('CURRENT_PLUS_LAGNEWS','CURRENT_COMMON'),phase=None,reference_ids=None,horizons=(1,2,3)):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    # Both sources carry the same closing calendar month. Keep the alpha field
    # once instead of producing origin_month_x/origin_month_y during the join.
    panel=join_fund_alpha(origins.drop(columns=['origin_month'],errors='ignore'),alpha) if len(alpha) else pd.DataFrame()
    if len(panel):
        panel['finite_inputs']=np.isfinite(panel[['alpha','past_alpha',CURRENT,LAG]].apply(numeric).to_numpy(float)).all(axis=1)
        panel['included_before_singletons']=truth(panel.universe_eligible)&truth(panel.nts_pair_eligible)&panel.finite_inputs
        panel.to_parquet(out/'fund_period_selection_panel.parquet',index=False,compression='zstd')
    results=[]
    for group in groups:
        for h in horizons:
            if phase is not None and phase>=h:continue
            mask=pd.Series(False,index=panel.index) if not len(panel) else panel.horizon.eq(h)&panel.included_before_singletons
            if len(panel) and group!='pooled':mask&=panel.fund_focus.eq(group)
            if len(panel) and phase is not None:
                m=pd.PeriodIndex(panel.origin_month,freq='M')
                mask&=((m.year*12+m.month-1)%h==phase)
            sample=panel.loc[mask].copy() if len(panel) else pd.DataFrame(columns=['fund_id','interval_id','end_date','alpha','past_alpha',CURRENT,LAG])
            hashes=[]
            for variant in variants:
                model_id=f'{branch}__{sensitivity}__{group}__h{h}__{variant}'
                if phase is not None:model_id+=f'__phase{phase}'
                info,used=fit_fund_fe(sample,distributed=variant=='CURRENT_PLUS_LAGNEWS',output=out/model_id)
                info.update(model_id=model_id,branch=branch,sensitivity=sensitivity,group=group,horizon=h,variant=variant)
                if phase is not None:info['phase']=phase
                if reference_ids is not None:
                    old=reference_ids.get((group,h),set())
                    ids=set(used.row_id) if len(used) else set()
                    info.update(primary_sample_overlap=len(old&ids),primary_reference_n=len(old))
                if branch=='TR_VERIFIED' and not len(sample):
                    info['data_limitation']='No complete verified total-return alpha on eligible NTS origins; not replaced by unit-value growth.'
                hashes.append(info['sample_hash']);results.append(info)
                save_json(out/model_id/'fit.json',info)
                checkpoint('REGRESSION_CELL',model_id=model_id,status=info['status'],nobs=info['nobs'],funds=info['funds'])
            if len(hashes)>1 and len(set(hashes))!=1:
                raise AssertionError('Current-only and distributed models did not use the exact same rows')
    return results


def compare_versions(current,previous):
    previous=Path(previous)
    old_summary=json.loads((previous/'summary.json').read_text(encoding='utf-8'))
    old_models=pd.read_csv(previous/'models.csv')
    new_models=pd.read_csv(current/'models.csv')
    for frame in [old_models,new_models]:
        frame['phase']=pd.to_numeric(frame.get('phase',pd.Series(-1,index=frame.index)),errors='coerce').fillna(-1).astype(int)
    keys=['branch','sensitivity','group','horizon','variant','phase']
    columns=keys+['model_id','status','nobs','funds','sample_hash']
    comparison=new_models[columns].merge(old_models[columns],on=keys,how='outer',validate='one_to_one',suffixes=('_v3','_v1'))
    comparison['delta_N']=comparison.nobs_v3-comparison.nobs_v1
    comparison['delta_funds']=comparison.funds_v3-comparison.funds_v1
    comparison.to_csv(current/'comparison_v1_v3_models.csv',index=False,encoding='utf-8-sig')
    rows=[];changes=[]
    base=comparison.loc[comparison.sensitivity.eq('PRIMARY')&comparison.variant.eq('CURRENT_PLUS_LAGNEWS')]
    for row in base.itertuples(index=False):
        before=pd.read_parquet(previous/'regression'/row.branch/'PRIMARY'/row.model_id_v1/'used_sample.parquet')
        after=pd.read_parquet(current/'regression'/row.branch/'PRIMARY'/row.model_id_v3/'used_sample.parquet')
        old_ids=set(before.row_id) if len(before) else set();new_ids=set(after.row_id) if len(after) else set()
        record=dict(branch=row.branch,group=row.group,horizon=row.horizon,N_v1=len(before),N_v3=len(after),
            common_rows=len(old_ids&new_ids),added_rows=len(new_ids-old_ids),removed_rows=len(old_ids-new_ids))
        for label,sample in [('v1',before),('v3',after)]:
            record['first_origin_'+label]=str(pd.to_datetime(sample.end_date).min().date()) if len(sample) else None
            record['last_origin_'+label]=str(pd.to_datetime(sample.end_date).max().date()) if len(sample) else None
            record['last_future_month_'+label]=str(sample.future_last_month.max()) if len(sample) else None
        rows.append(record)
        for direction,ids in [('added',new_ids-old_ids),('removed',old_ids-new_ids)]:
            changes.extend(dict(branch=row.branch,group=row.group,horizon=row.horizon,row_id=identifier,change=direction) for identifier in sorted(ids))
    pd.DataFrame(rows).to_csv(current/'comparison_v1_v3_coverage_and_overlap.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(changes,columns=['branch','group','horizon','row_id','change']).to_parquet(
        current/'comparison_v1_v3_changed_sample_keys_PC_ONLY.parquet',index=False,compression='zstd')
    counts=[]
    for label,root in [('v1',previous),('v3',current)]:
        frame=pd.read_csv(root/'fund_return_exclusion_counts.csv')
        counts+=frame.assign(release=label).to_dict('records')
    pd.DataFrame(counts).to_csv(current/'comparison_v1_v3_return_exclusions.csv',index=False,encoding='utf-8-sig')
    return dict(previous_run=str(previous),previous_manifest_sha256=sha(previous/'manifest.json'),
                previous_summary=old_summary,coverage_comparison=rows)


def audit_sensitivity_registry(results,returns,factors,primary,broad):
    families=[('PRIMARY',48,'Both return bases; two baseline variants for four groups and three horizons.'),
        ('HISTORY36_MIN24',12,'TR_VERIFIED only, fixed36-month lookback/min24.'),
        ('BROAD_OBSERVED_NTS',12,'TR_VERIFIED only, one company/positive denominators and no0.5 coverage gate.'),
        ('NONOVERLAPPING_PHASES',24,'TR_VERIFIED only: all h phases for h2/h3; four h1 references reuse primary.'),
        ('CORPORATE_BOND_SPREAD',9,'TR_VERIFIED only; pooled/bond/mixed with verified RUCBTRNS-RGBITR.'),
        ('CORPORATE_SAME_SAMPLE_TWO_FACTOR',9,'Two-factor comparator on exactly the same eligible credit-model rows.')]
    data=pd.DataFrame([{k:v for k,v in result.items() if not isinstance(v,(dict,list))} for result in results])
    rows=[];tr_months=int(returns.TR_eligible.sum())
    for family,expected,meaning in families:
        frame=data.loc[data.sensitivity.eq(family)]
        if len(frame)!=expected:raise AssertionError(f'Missing frozen labels in {family}: {len(frame)} != {expected}')
        count=int(frame.status.eq('ESTIMATED').sum())
        inactive=family!='PRIMARY' and tr_months==0
        rows.append(dict(family=family,expected_labels=expected,actual_labels=len(frame),estimated_labels=count,
            not_estimable_labels=int(frame.status.eq('NOT_ESTIMABLE').sum()),definition=meaning,
            reason_not_estimated='No TR_VERIFIED fund return months; frozen sensitivities are TR-only, not waived or silently applied to diagnostic.' if inactive else 'See every cell status and reason.',
            verified_TR_months=tr_months,verified_primary_factor_months=int(truth(factors.primary_complete).sum()),
            verified_credit_factor_months=int(truth(factors.corporate_sensitivity_complete).sum()),
            primary_NTS_origins=int(primary.nts_pair_eligible.sum()),broad_NTS_origins=int(broad.nts_pair_eligible.sum())))
    if len(data)!=114 or data.model_id.duplicated().any():raise AssertionError('Frozen registry count/identity mismatch')
    return dict(status='COMPLETE_ALL_FROZEN_LABELS_ACCOUNTED_FOR',model_labels=len(data),families=rows,
        no_UNIT_sensitivity_grid='Explicit frozen not_in_default_grid rule, not a missing run.',
        original_AT='NOT_AVAILABLE: no verified whole-portfolio passive-revaluation AT for these intervals.',
        MCFTRR='Collected but excluded from the frozen default grid; not silently substituted for MCFTR.')


def run(snapshot_path=None):
    started=time.monotonic();code=Path(__file__).parent
    spec_path=code/'h3_alpha_spec_20261104.json';file_record(spec_path,SPEC_SHA)
    spec=json.loads(spec_path.read_text(encoding='utf-8'))
    snapshot_path=Path(snapshot_path) if snapshot_path else code/'h3_alpha_handoff_snapshot_20261104.json'
    snapshot=json.loads(snapshot_path.read_text(encoding='utf-8'))
    checkpoint('WAITING_FOR_NTS_SEAL')
    deadline=time.monotonic()+4*3600
    while not (ROOT/'runtime/nts_ready.json').exists():
        nts_task=ROOT/'runtime/execution_checkpoint.json'
        if time.monotonic()>deadline:raise TimeoutError('NTS seal not ready after four hours')
        time.sleep(15)
    ready=json.loads((ROOT/'runtime/nts_ready.json').read_text(encoding='utf-8'))
    nts_root=Path(ready['output']);file_record(nts_root/'manifest.json',ready['manifest_sha256'])
    nts_manifest=json.loads((nts_root/'manifest.json').read_text(encoding='utf-8'))
    for r in nts_manifest['files'].values():file_record(r['path'],r['sha256'])
    fund_root=ROOT/'fund_history'/snapshot['fund_release']
    paths=dict(fund_returns=fund_root/'fund_monthly_returns.parquet',crosswalk=fund_root/'fund_series_crosswalk.parquet',
        factors=ROOT/'factors/factors_monthly.parquet',cash_events=ROOT/'fund_history/action_evidence'/snapshot['known_cash_events'],
        nts_primary=nts_root/'fund_period_nts_primary.parquet',nts_broad=nts_root/'fund_period_nts_broad.parquet')
    expected=dict(fund_returns=snapshot['fund_returns_sha256'],crosswalk=snapshot['fund_crosswalk_sha256'],factors=snapshot['factors_sha256'])
    inputs={k:file_record(v,expected.get(k)) for k,v in paths.items()}
    if snapshot.get('fund_release_seal_sha256'):
        release_seal_path=fund_root/'sealed_outputs.json'
        inputs['fund_release_seal']=file_record(release_seal_path,snapshot['fund_release_seal_sha256'])
        release_seal=json.loads(release_seal_path.read_text(encoding='utf-8'))
        for name,record in release_seal['files'].items():
            file_record(fund_root/name,record['sha256'])
        inputs['fund_normalization_policy']=file_record(fund_root/'normalization_policy.json',
            release_seal['files']['normalization_policy.json']['sha256'])
    inputs.update(spec=file_record(spec_path,SPEC_SHA),handoff=file_record(snapshot_path),nts_manifest=file_record(nts_root/'manifest.json'))
    for name in ['run_h3_alpha_20261104.py','h3_alpha_core_20261104.py','fit_h3_alpha_fe_20261104.py','h3_alpha_io_20261104.py']:
        inputs['code:'+name]=file_record(code/name)
    version=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()[:16]
    out=ROOT/'outputs'/('run_'+version);out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():
        manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        for record in manifest['files'].values():file_record(record['path'],record['sha256'])
        checkpoint('H3_ALREADY_COMPLETE',output=str(out));return
    # Pin executable source copies inside this result version so later code
    # maintenance cannot invalidate the provenance of a completed run.
    code_snapshot=out/'code';code_snapshot.mkdir(exist_ok=True)
    for key,record in list(inputs.items()):
        if not key.startswith('code:'):continue
        destination=code_snapshot/Path(record['path']).name
        if not destination.exists():shutil.copy2(record['path'],destination)
        inputs[key]=file_record(destination,record['sha256'])
    save_json(out/'input_manifest.json',dict(version=version,inputs=inputs,spec_id=spec['spec_id'],frozen_before_coefficients=True))
    checkpoint('PREPARING_SEALED_MONTHLY_INPUTS',output=str(out),input_version=version)
    save_json(out/'synthetic_FE_QA.json',synthetic_tests())
    raw=pd.read_parquet(paths['fund_returns']);cross=pd.read_parquet(paths['crosswalk']);cash=pd.read_parquet(paths['cash_events'])
    returns=prepare_returns(raw,cross,cash,snapshot)
    factors=pd.read_parquet(paths['factors']).rename(columns={'rf_m':'risk_free','MCFTR_return_m':'equity_return',
        'RGBITR_return_m':'bond_return','credit_factor_m':'credit_factor'})
    unique(factors,['month'],'frozen factor month')
    factors['factors_verified']=truth(factors.primary_complete)
    returns.to_parquet(out/'monthly_returns_adapter.parquet',index=False,compression='zstd')
    factors.to_parquet(out/'monthly_factors_adapter.parquet',index=False,compression='zstd')
    primary=pd.read_parquet(paths['nts_primary']);broad=pd.read_parquet(paths['nts_broad'])
    mandates=dated_mandate_evidence(primary)
    mandates.to_parquet(out/'dated_opening_mandate_evidence.parquet',index=False,compression='zstd')
    primary=make_origins(primary,cross,mandates);broad=make_origins(broad,cross,mandates)
    primary.to_parquet(out/'origin_eligibility.parquet',index=False,compression='zstd')
    results=[];alphas={}
    for branch in BRANCHES:
        alpha=compute_alpha(primary,returns,factors,out/'alpha'/branch/'HISTORY60_MIN36',branch=branch)
        alphas[branch]=alpha
        results+=estimate_grid(primary,alpha,out/'regression'/branch/'PRIMARY',branch)
        alpha.groupby(['horizon','status'],dropna=False).size().rename('origins').reset_index().to_csv(
            out/f'alpha_gate_counts_{branch}.csv',index=False,encoding='utf-8-sig')
    # Fixed sensitivities are TR-only. Empty primary return data must still yield
    # a complete registry of explicitly unestimable cells, not a diagnostic grid
    # selected instead to manufacture an apparently primary answer.
    tr_alpha=alphas['TR_VERIFIED']
    reference={}
    for result in results:
        if result['branch']=='TR_VERIFIED' and result['variant']=='CURRENT_PLUS_LAGNEWS':
            sample=pd.read_parquet(out/'regression'/'TR_VERIFIED'/'PRIMARY'/result['model_id']/'used_sample.parquet')
            reference[(result['group'],result['horizon'])]=set(sample.row_id) if len(sample) else set()
    short=compute_alpha(primary,returns,factors,out/'alpha/TR_VERIFIED/HISTORY36_MIN24',branch='TR_VERIFIED',history=36,minimum=24)
    results+=estimate_grid(primary,short,out/'regression/TR_VERIFIED/HISTORY36_MIN24','TR_VERIFIED','HISTORY36_MIN24',
        variants=('CURRENT_PLUS_LAGNEWS',),reference_ids=reference)
    broad_alpha=compute_alpha(broad,returns,factors,out/'alpha/TR_VERIFIED/BROAD_OBSERVED_NTS',branch='TR_VERIFIED')
    results+=estimate_grid(broad,broad_alpha,out/'regression/TR_VERIFIED/BROAD_OBSERVED_NTS','TR_VERIFIED','BROAD_OBSERVED_NTS',
        variants=('CURRENT_PLUS_LAGNEWS',),reference_ids=reference)
    for phase in range(3):
        phase_results=estimate_grid(primary,tr_alpha,out/f'regression/TR_VERIFIED/NONOVERLAPPING_PHASE{phase}',
            'TR_VERIFIED','NONOVERLAPPING_PHASES',variants=('CURRENT_PLUS_LAGNEWS',),phase=phase,reference_ids=reference,horizons=(2,3))
        # A horizon h has exactly h phases; h1 references the primary fit.
        results += phase_results
    results += [dict(r,model_id=r['model_id'].replace('__PRIMARY__','__NONOVERLAPPING_PHASES__'),
        sensitivity='NONOVERLAPPING_PHASES',reuse_of_primary_model_id=r['model_id'],phase=0)
        for r in results.copy() if r['branch']=='TR_VERIFIED' and r['sensitivity']=='PRIMARY'
        and r['variant']=='CURRENT_PLUS_LAGNEWS' and r['horizon']==1]
    credit_factors=factors.copy();credit_factors['factors_verified']=truth(factors.corporate_sensitivity_complete)
    credit=compute_alpha(primary,returns,credit_factors,out/'alpha/TR_VERIFIED/CORPORATE_BOND_SPREAD',branch='TR_VERIFIED',credit=True)
    results+=estimate_grid(primary,credit,out/'regression/TR_VERIFIED/CORPORATE_BOND_SPREAD','TR_VERIFIED','CORPORATE_BOND_SPREAD',
        groups=['pooled','bond','mixed'],variants=('CURRENT_PLUS_LAGNEWS',),reference_ids=reference)
    # Two-factor reference on the identical eligible three-factor origin/h rows.
    eligible_keys=credit.loc[credit.alpha.notna(),['fund_id','interval_id','horizon']].assign(_credit_match=True)
    same=tr_alpha.merge(eligible_keys,on=['fund_id','interval_id','horizon'],how='left',validate='one_to_one')
    same.loc[~truth(same._credit_match),'alpha']=np.nan
    results+=estimate_grid(primary,same,out/'regression/TR_VERIFIED/CORPORATE_SAME_SAMPLE_TWO_FACTOR','TR_VERIFIED','CORPORATE_SAME_SAMPLE_TWO_FACTOR',
        groups=['pooled','bond','mixed'],variants=('CURRENT_PLUS_LAGNEWS',),reference_ids=reference)
    records=[];coef=[];tests=[]
    for result in results:
        records.append({k:v for k,v in result.items() if not isinstance(v,(list,dict))})
        tags={k:result[k] for k in ['model_id','branch','sensitivity','group','horizon','variant','nobs','funds','status']}
        coef += [dict(tags,**c) for c in result.get('coefficients',[])]
        tests += [dict(tags,**t) for t in result.get('target_tests',[])]
    pd.DataFrame(records).to_csv(out/'models.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(coef).to_csv(out/'coefficients.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(tests).to_csv(out/'tests.csv',index=False,encoding='utf-8-sig')
    returns.groupby(['adapter_exclusion_reason'],dropna=False).size().rename('monthly_rows').reset_index().to_csv(
        out/'fund_return_exclusion_counts.csv',index=False,encoding='utf-8-sig')
    if 'return_missing_reason' in returns:
        returns.groupby(['return_missing_reason'],dropna=False).size().rename('monthly_rows').reset_index().to_csv(
            out/'source_return_exclusion_counts.csv',index=False,encoding='utf-8-sig')
    primary.assign(year=pd.to_datetime(primary.end_date).dt.year).groupby(
        ['year','fund_focus','universe_eligible','nts_pair_eligible','identity_origin_verified'],dropna=False).size().rename('origins').reset_index().to_csv(
        out/'origin_selection_flow_by_group_year.csv',index=False,encoding='utf-8-sig')
    diagnostic=alphas['UNIT_VALUE_KNOWN_ADJUSTMENTS']
    sensitivity_audit=audit_sensitivity_registry(results,returns,factors,primary,broad)
    save_json(out/'frozen_sensitivity_audit.json',sensitivity_audit)
    pd.DataFrame(sensitivity_audit['families']).to_csv(out/'frozen_sensitivity_audit.csv',index=False,encoding='utf-8-sig')
    comparison=compare_versions(out,snapshot['compare_with_run']) if snapshot.get('compare_with_run') else None
    if comparison:save_json(out/'comparison_v1_v3.json',comparison)
    summary=dict(status='COMPUTED_PENDING_INDEPENDENT_NUMERICAL_AND_RESEARCH_QA',output_root=str(out),version=version,
        created_utc=utc(),spec_id=spec['spec_id'],spec_sha256=SPEC_SHA,input_manifest=inputs,
        fund_release=snapshot['fund_release'],sensitivity_registry_status=sensitivity_audit['status'],
        previous_run=snapshot.get('compare_with_run'),
        source_funds=int(primary.fund_id.nunique()),selected_origins=len(primary),
        primary_NTS_origins=int(primary.nts_pair_eligible.sum()),
        primary_NTS_origins_in_mandate_universe=int((primary.nts_pair_eligible&primary.universe_eligible).sum()),
        monthly_returns=dict(observed_eligible=int(returns.diagnostic_eligible.sum()),TR_verified=int(returns.TR_eligible.sum()),
            funds_observed_eligible=int(returns.loc[returns.diagnostic_eligible,'fund_id'].nunique())),
        diagnostic_eligible_alpha_by_horizon=diagnostic.groupby('horizon').alpha.count().to_dict(),
        model_labels=len(results),estimated_model_labels=sum(r['status']=='ESTIMATED' for r in results),
        primary_TR_H3_status='NOT_ESTIMABLE' if not tr_alpha.alpha.notna().any() else 'ESTIMATED_PENDING_QA',
        original_AT_branch_status='NOT_AVAILABLE',runtime_seconds=time.monotonic()-started,
        independent_QA_required=True,publication_interpretation_ready=False,
        limitations=['UNIT_VALUE_KNOWN_ADJUSTMENTS is factor-adjusted unit-value growth, not verified total-return alpha.',
            'Unknown distributions, splits, revised valuation vintages, illiquidity and omitted risk exposures remain.',
            'Q-value changes include fund flows and valuation/price approximation, not pure discretionary trading.',
            'One-way fund clustering does not cover dependence across different funds; few-fund inference is fragile.',
            'No multiple-testing adjustment, alpha trimming, control search or new LLM estimation.'])
    save_json(out/'summary.json',summary);save_json(out/'all_model_details.json',results)
    table=pd.DataFrame(records)
    show=table.loc[table.sensitivity.eq('PRIMARY'),['branch','group','horizon','variant','status','nobs','funds']]
    title='H3: первый зафиксированный расчёт — требуется независимый QA'
    page='<!doctype html><meta charset="utf-8"><title>'+title+'</title><style>body{font:17px system-ui;margin:35px;max-width:1400px}td,th{padding:6px 9px;border:1px solid #ddd}table{border-collapse:collapse}pre{white-space:pre-wrap}</style><h1>'+title+'</h1>'
    page+='<p>Проверенная полная доходность: '+summary['primary_TR_H3_status']+'. Диагностическая ветка не подменяет основную H3.</p>'
    page+=show.to_html(index=False,escape=True)
    if coef:page+='<h2>Коэффициенты: t(G−1), без поправки на множественные проверки</h2>'+pd.DataFrame(coef).to_html(index=False,escape=True)
    page+='<h2>Ограничения</h2><ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in summary['limitations'])+'</ul>'
    (out/'review.html').write_text(page,encoding='utf-8')
    for record in inputs.values():file_record(record['path'],record['sha256'])
    seal_directory(out,dict(status=summary['status'],inputs=inputs,summary='summary.json',version=version))
    save_json(ROOT/'runtime/alpha_ready.json',dict(ready=True,output=str(out),manifest_sha256=sha(out/'manifest.json'),summary=summary))
    checkpoint('H3_COMPUTED_QA_PENDING',output=str(out),summary={k:v for k,v in summary.items() if k!='input_manifest'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot')
    args=parser.parse_args();run(args.snapshot)
