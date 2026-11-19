"""Relaxed independent news panel from immutable existing ledgers only."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
from h3_alpha_core_20261104 import numeric,truth,unique
from h3_alpha_io_20261104 import sha,file_record,save_json,seal_directory,utc
from h3_relaxed_news_core_20261104 import classify_news,independent_scores

ROOT=Path('./analysis_outputs/h3_relaxed_news_20261104')
BASE=ROOT.parent;OLD=BASE/'h3_alpha_20261104'
SPEC_SHA='36a64070c5dfcba749d7baa1f15b639f9801a39fcac12142e372a94a2554699f'

def checkpoint(stage,**kw):
    import os
    d=dict(stage=stage,pid=os.getpid(),utc=utc(),**kw);save_json(ROOT/'runtime/checkpoint.json',d);print(json.dumps(d,default=str),flush=True)

def wid(company,start,end):
    if pd.isna(company) or pd.isna(start) or pd.isna(end):return None
    return hashlib.sha256(('actual_report_interval\t'+str(company)+'\t'+pd.Timestamp(start).date().isoformat()+'\t'+pd.Timestamp(end).date().isoformat()).encode()).hexdigest()

def run():
    code=Path(__file__).parent;spec=code/'h3_relaxed_news_spec_20261104.json'
    file_record(spec,SPEC_SHA)
    paths=dict(contributions=OLD/'panels/nts_ee5a8bf9b450137a/quantity_news_contributions.parquet',
        decisions=BASE/'h1_entries_trim_20261102/row_decisions.parquet',threshold=BASE/'h1_entries_trim_20261102/threshold.json',
        origins=OLD/'outputs/run_d0440fad412ee5e9/origin_eligibility.parquet',
        net=BASE/'h1_net_fund_fe_20261030/net_sentiment_company_period.parquet',
        lag=BASE/'h1_sentiment_lag_20261103/data/lag_company_windows.parquet',
        calendar=BASE/'title_name_regressions_20261026_w2_v2/all_true_intervals.parquet',
        events=BASE/'title_name_regressions_20261026_w2_v2/event_representatives.parquet',
        registry=Path('./databaseresources/mapping/title-name-20261008-no-gaz/company_registry.parquet'),
        approved_scope=BASE/'h1_macro_no_time_20261031/inputs/approved_news_company_scope.parquet')
    expected=dict(net='760d8f645e5133d0fa1217387a55296b6c55b8e0c8542aceb29eda050dcddc50',lag='8642a693e281a033964b5c868a2baa4c9335ea85073c541f88e6aa17a19b7fd3',
        calendar='8eb84efe7e01699d46595e7c2c1f5ccadd6dfa3d42c14681d3e5a97d0c966709',events='fb188b547da90fa71e485ac931c0e19fea858e102472c90df31222dcb9c105a4')
    checkpoint('PINNING_RELAXED_PANEL_INPUTS')
    inputs={k:file_record(p,expected.get(k)) for k,p in paths.items()}
    inputs.update(spec=file_record(spec,SPEC_SHA),builder=file_record(__file__),core=file_record(code/'h3_relaxed_news_core_20261104.py'))
    version=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()[:16]
    out=ROOT/'panels'/version;out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise RuntimeError('Panel already sealed; do not duplicate')
    save_json(out/'input_manifest.json',dict(inputs=inputs,version=version))
    raw_origins=pd.read_parquet(paths['origins'])
    cols=['fund_id','interval_id','opening_report_id','closing_report_id','start_date','end_date','origin_month',
        'fund_focus','fund_focus_source','fund_focus_evidence','index_status','exclude_index_fund','universe_eligible',
        'identity_origin_verified','row_id','explicit_incompatible_mandate']
    origins=raw_origins[cols].copy()
    origins['fund_id']=origins.fund_id.astype(str)
    u=pd.read_parquet(paths['contributions']);u['fund_id']=u.fund_id.astype(str);u['position_id']=u.row_id.astype(str)
    for c in ['start_date','end_date']:u[c]=pd.to_datetime(u[c])
    decision_cols=['fund_id','interval_id','security_id','row_id','exclusion_reason_augmented','quantity_open','quantity_close',
                   'quantity_change_pct','trimmed_upper_p99','retained_primary']
    decisions=pd.read_parquet(paths['decisions'],columns=decision_cols).rename(columns={'row_id':'candidate_row_id'})
    decisions['fund_id']=decisions.fund_id.astype(str)
    unique(decisions,['fund_id','interval_id','security_id'],'all original quantity decisions')
    news_only={'sentiment_not_primary','missing_net_sentiment','missing_control_log_news_count_interval'}
    decisions['inherited_nonnews_reason']=decisions.exclusion_reason_augmented.fillna('').map(
        lambda s:';'.join(x for x in str(s).split(';') if x and x not in news_only))
    decisions['removed_legacy_news_reasons']=decisions.exclusion_reason_augmented.fillna('').map(
        lambda s:';'.join(x for x in str(s).split(';') if x in news_only))
    threshold=float(json.loads(paths['threshold'].read_text(encoding='utf-8'))['cutoff_percent'])
    if not np.isclose(threshold,322.7771642245575,rtol=0,atol=1e-10):raise AssertionError('Fixed inherited P99 differs')
    continuing=numeric(decisions.quantity_open).gt(0)&numeric(decisions.quantity_close).gt(0)
    decisions['inherited_fixed_P99_excluded']=truth(decisions.trimmed_upper_p99)|(continuing&numeric(decisions.quantity_change_pct).gt(threshold))
    decisions['quantity_in_scope']=decisions.inherited_nonnews_reason.eq('')&~decisions.inherited_fixed_P99_excluded
    u=u.merge(decisions,on=['fund_id','interval_id','security_id'],how='left',validate='one_to_one')
    if u.candidate_row_id.isna().any():raise AssertionError('Union-held positions lack original quantity decisions')
    if not truth(u.loc[truth(u.inherited_target),'quantity_in_scope']).all():raise AssertionError('An inherited non-news eligible row was newly rejected')
    u['quantity_in_scope']=truth(u.quantity_in_scope)
    u['quantity_action_eligible']=np.isfinite(numeric(u.B_Q))&~truth(u.scheduled_redemption_in_interval)
    u['restored_news_only']=u.quantity_in_scope&~truth(u.inherited_target)&u.removed_legacy_news_reasons.ne('')
    # These inherited paired flags are audit-only, never new eligibility gates.
    u=u.rename(columns={c:'legacy_'+c for c in ['eligible_for_score','NTS_common_E','eligible_primary','lag_eligible_primary']})
    scope=pd.read_parquet(paths['approved_scope']).company_id.dropna().astype(str)
    registry=pd.read_parquet(paths['registry'],columns=['company_id']).company_id.dropna().astype(str)
    if set(scope)!=set(registry):raise AssertionError('Frozen approved company registry changed')
    u['mapping_in_scope']=u.company_id.astype(str).isin(set(scope))&u.company_id.notna()
    calendar=pd.read_parquet(paths['calendar']);calendar['fund_id']=calendar.fund_id.astype(str)
    previous=calendar[['fund_id','closing_report_id','interval_id','start_date','end_date']].rename(columns={
        'closing_report_id':'opening_report_id','interval_id':'previous_interval_id','start_date':'previous_start','end_date':'previous_end'})
    unique(previous,['fund_id','opening_report_id'],'exact predecessor calendar')
    origins=origins.merge(previous,on=['fund_id','opening_report_id'],how='left',validate='one_to_one')
    origins['previous_window_valid']=origins.previous_interval_id.notna()&pd.to_datetime(origins.previous_end).eq(pd.to_datetime(origins.start_date))
    for c in ['previous_start','previous_end','previous_window_valid']:
        u[c]=u.interval_id.map(origins.set_index('interval_id')[c])
    fields=['company_period_id','company_id','period_start','period_end','net_sentiment','n_candidates','n_valid','n_relevant',
        'n_errors_or_unattempted','n_undated_company_candidates','eligible_primary','state','sample_hash']
    net=pd.read_parquet(paths['net']);existing=net.loc[net.period_kind.eq('actual_report_interval'),fields].copy()
    lag=pd.read_parquet(paths['lag'],columns=fields)
    overlap=existing.merge(lag,on='company_period_id',how='inner',suffixes=('_a','_b'))
    for field in ['net_sentiment','n_candidates','n_valid','n_relevant','n_undated_company_candidates']:
        if not np.allclose(numeric(overlap[field+'_a']),numeric(overlap[field+'_b']),rtol=0,atol=1e-12,equal_nan=True):raise AssertionError('Frozen window conflict '+field)
    windows=pd.concat([existing,lag],ignore_index=True).drop_duplicates('company_period_id')
    windows['window_source']='EXISTING_FROZEN_EXACT_WINDOW'
    needed=[]
    active=u.quantity_in_scope&u.quantity_action_eligible&u.mapping_in_scope
    for label,start,end,valid in [('current','start_date','end_date',pd.Series(True,index=u.index)),
        ('lagnews','previous_start','previous_end',truth(u.previous_window_valid))]:
        keys=u.loc[active&valid,['company_id',start,end]].rename(columns={start:'period_start',end:'period_end'}).drop_duplicates()
        keys['company_period_id']=[wid(c,s,e) for c,s,e in zip(keys.company_id,keys.period_start,keys.period_end)]
        needed.append(keys)
        u[label+'_window_id']=[wid(c,s,e) if a and v else None for c,s,e,a,v in zip(u.company_id,u[start],u[end],active,valid)]
        u[label+'_window_valid']=valid&u.mapping_in_scope
    needed=pd.concat(needed,ignore_index=True).drop_duplicates('company_period_id')
    missing=needed.loc[~needed.company_period_id.isin(windows.company_period_id)].copy()
    checkpoint('RELAXED_RECONSTRUCT_MISSING_EXACT_WINDOWS',corporate_positions=len(u),quantity_scope=int(u.quantity_in_scope.sum()),restored_news_only=int(u.restored_news_only.sum()),needed_windows=len(needed),missing_windows=len(missing))
    event_columns=['pair_id','article_id','company_id','event_date','attempted','valid','relevance','sentiment_label','is_representative']
    events=pd.read_parquet(paths['events'],columns=event_columns)
    if not truth(events.is_representative).all():raise AssertionError('Noncanonical event included')
    events['event_date']=pd.to_datetime(events.event_date)
    undated=events.loc[events.event_date.isna()].groupby('company_id',observed=True).size()
    first,last=events.event_date.min(),events.event_date.max()
    if len(missing):
        con=duckdb.connect(':memory:');con.execute("SET threads=2; SET memory_limit='2GB'")
        con.register('w',missing);con.register('e',events)
        members=con.execute('''SELECT w.company_period_id,e.pair_id,e.article_id,e.company_id,e.event_date,e.valid,e.attempted,e.relevance,e.sentiment_label
            FROM w JOIN e ON w.company_id=e.company_id AND e.event_date>w.period_start AND e.event_date<=w.period_end''').fetchdf()
        members.to_parquet(out/'reconstructed_window_members.parquet',index=False,compression='zstd')
        con.register('m',members)
        rebuilt=con.execute('''SELECT w.company_period_id,w.company_id,w.period_start,w.period_end,count(m.pair_id) n_candidates,
            count(m.pair_id) FILTER(WHERE m.valid=true) n_valid,
            count(m.pair_id) FILTER(WHERE m.valid=true AND m.relevance='relevant') n_relevant,
            count(m.pair_id) FILTER(WHERE m.valid=true AND m.relevance='relevant' AND m.sentiment_label='positive') n_positive,
            count(m.pair_id) FILTER(WHERE m.valid=true AND m.relevance='relevant' AND m.sentiment_label='negative') n_negative,
            sha256(string_agg(m.pair_id,chr(10) ORDER BY m.pair_id)) sample_hash
            FROM w LEFT JOIN m USING(company_period_id) GROUP BY w.company_period_id,w.company_id,w.period_start,w.period_end''').fetchdf();con.close()
        rebuilt['n_errors_or_unattempted']=rebuilt.n_candidates-rebuilt.n_valid
        rebuilt['n_undated_company_candidates']=rebuilt.company_id.map(undated).fillna(0)
        rebuilt['eligible_primary']=rebuilt.n_candidates.gt(0)&rebuilt.n_candidates.eq(rebuilt.n_valid)&rebuilt.n_relevant.gt(0)
        rebuilt['net_sentiment']=((rebuilt.n_positive-rebuilt.n_negative)/rebuilt.n_relevant.where(rebuilt.n_relevant.gt(0))).where(rebuilt.eligible_primary)
        rebuilt['state']='RECONSTRUCTED_FROZEN_EVENTS';rebuilt['window_source']='RECONSTRUCTED_MISSING_EXACT_WINDOW'
        windows=pd.concat([windows,rebuilt],ignore_index=True)
    unique(windows,['company_period_id'],'all exact company news windows')
    windows['source_present']=True;windows['mapping_in_scope']=windows.company_id.astype(str).isin(set(scope))
    windows['window_valid']=pd.to_datetime(windows.period_start).ge(first)&pd.to_datetime(windows.period_end).le(last)
    classified=classify_news(windows)
    old_known=truth(windows.eligible_primary)&np.isfinite(numeric(windows.net_sentiment))&windows.mapping_in_scope&windows.window_valid
    if not np.allclose(numeric(classified.loc[old_known,'value']),numeric(windows.loc[old_known,'net_sentiment']),rtol=0,atol=0):raise AssertionError('Previously valid Net changed, including undated audit cases')
    classified.to_parquet(out/'company_window_status_ledger.parquet',index=False,compression='zstd')
    indexed=classified.set_index('company_period_id')
    for label in ['current','lagnews']:
        ids=u[label+'_window_id']
        u[label+'_source_present']=ids.isin(indexed.index)
        for a,b in [('value','news_value'),('news_known','news_known'),('news_status','news_status'),('n_undated_company_candidates','undated_audit_count')]:
            u[label+'_'+b]=ids.map(indexed[a])
        u[label+'_news_known']=truth(u[label+'_news_known'])
        # A real verified report interval remains valid when no source record
        # exists: measured B_Q=0 alone can establish its product. Source-present
        # is separately required for news-based contributions by the kernel.
    checkpoint('RELAXED_AGGREGATING_INDEPENDENT_SCORES',positions=len(u),window_rows=len(classified))
    scores,contributions=independent_scores(u,origins)
    scores.to_parquet(out/'relaxed_fund_period_scores.parquet',index=False,compression='zstd')
    contributions.to_parquet(out/'relaxed_security_window_ledger.parquet',index=False,compression='zstd')
    old=raw_origins[['fund_id','interval_id','NTS_Q_current','NTS_Q_lagnews','nts_pair_eligible']].rename(columns={
        'NTS_Q_current':'old_NTS_current','NTS_Q_lagnews':'old_NTS_lagnews','nts_pair_eligible':'old_pair_eligible'})
    compare=scores.merge(old,on=['fund_id','interval_id'],how='left',validate='one_to_one')
    for label in ['current','lagnews']:
        oldcol='old_NTS_'+label;newcol='NTS_Q_'+label+'_relaxed'
        compare[label+'_same_numeric_ratio']=np.isclose(numeric(compare[oldcol]),numeric(compare[newcol]),rtol=1e-10,atol=1e-12,equal_nan=False)
    compare.to_parquet(out/'old_new_score_comparison.parquet',index=False,compression='zstd')
    counts=dict(origins=len(scores),positions=len(contributions),quantity_scope=int(u.quantity_in_scope.sum()),
        restored_news_only_positions=int(u.restored_news_only.sum()),needed_windows=len(needed),reconstructed_windows=len(missing),
        current_eligible=int(scores.current_eligible.sum()),lagnews_eligible=int(scores.lagnews_eligible.sum()),both_eligible=int((scores.current_eligible&scores.lagnews_eligible).sum()),
        current_known_zero=int(scores.NTS_Q_current_relaxed.eq(0).sum()),lagnews_known_zero=int(scores.NTS_Q_lagnews_relaxed.eq(0).sum()),
        old_current_unchanged=int(compare.current_same_numeric_ratio.sum()),old_lagnews_unchanged=int(compare.lagnews_same_numeric_ratio.sum()))
    for label in ['current','lagnews']:
        scores.groupby(['fund_focus',label+'_status'],dropna=False).size().rename('origins').reset_index().to_csv(out/(label+'_score_status_counts.csv'),index=False,encoding='utf-8-sig')
    for rec in inputs.values():file_record(rec['path'],rec['sha256'])
    seal_directory(out,dict(status='RELAXED_PANEL_SEALED',inputs=inputs,counts=counts,spec_sha256=SPEC_SHA))
    save_json(ROOT/'runtime/panel_ready.json',dict(output=str(out),manifest_sha256=sha(out/'manifest.json'),counts=counts))
    checkpoint('RELAXED_PANEL_READY',output=str(out),counts=counts)
    return out

if __name__=='__main__':run()
