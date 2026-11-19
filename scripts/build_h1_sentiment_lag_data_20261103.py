"""Exact preceding-report-interval NetSentiment data, executed only on research-host.

No regression, LLM call, calendar-month substitution, source writes or imputation.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import duckdb
import numpy as np
import pandas as pd

BASE = Path('./analysis_outputs')
OUT = BASE / 'h1_sentiment_lag_20261103/data'
UPSTREAM = BASE / 'title_name_regressions_20261026_w2_v2'
INPUTS = {
    'panel': BASE / 'h1_entries_trim_20261102/samples/AUGMENTED_P99.parquet',
    'intervals': UPSTREAM / 'all_true_intervals.parquet',
    'net': BASE / 'h1_net_fund_fe_20261030/net_sentiment_company_period.parquet',
    'events': UPSTREAM / 'event_representatives.parquet',
    'core': UPSTREAM / 'code/title_name_regressions_core.py',
}
EXPECTED = {
    'panel': '8bd9cfd2eef8668036c8c6c95c9863d8edbc1923d2c4d627d5b1306d03cfd43c',
    'intervals': '8eb84efe7e01699d46595e7c2c1f5ccadd6dfa3d42c14681d3e5a97d0c966709',
    'net': '760d8f645e5133d0fa1217387a55296b6c55b8e0c8542aceb29eda050dcddc50',
    'events': 'fb188b547da90fa71e485ac931c0e19fea858e102472c90df31222dcb9c105a4',
    'core': '82e8ca05e420dafb0b17f154c7a349a2497fb02dc63c4f6aee8bd656fdb014e4',
}
COUNT_FIELDS = ['n_candidates','n_attempted','n_valid','n_relevant','n_positive',
                'n_negative','n_neutral','n_errors_or_unattempted','n_irrelevant',
                'n_error','n_unattempted','n_undated_company_candidates']
PARITY_FIELDS = COUNT_FIELDS + ['net_sentiment','net_sentiment_observed',
    'classification_coverage','state','eligible_primary','strict_complete']
EVENT_FIELDS = ['pair_id','article_id','company_id','source','cluster_id','event_date',
    'event_timestamp','date_inferred','attempted','valid','relevance','sentiment_label',
    'pair_state','text_sha256','is_representative']
START = time.monotonic()


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):value.update(block)
    return value.hexdigest()


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,(pd.Timestamp,dt.date,dt.datetime,Path)):return str(value)
    if value is pd.NA or value is pd.NaT:return None
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def save(name,value):
    target=OUT/name;temporary=target.with_suffix(target.suffix+'.tmp')
    temporary.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(target)


def stage(name,**extra):
    value=dict(stage=name,utc=dt.datetime.now(dt.timezone.utc).isoformat(),elapsed_seconds=round(time.monotonic()-START,3),**extra)
    save('data_checkpoint.json',value);print(json.dumps(clean(value),ensure_ascii=True),flush=True)


def window_id(company,start,end):
    content='\t'.join(['actual_report_interval',str(company),pd.Timestamp(start).date().isoformat(),pd.Timestamp(end).date().isoformat()])
    return hashlib.sha256(content.encode()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'data_ready.json').exists():raise RuntimeError('Completed lag data is immutable')
    stage('VERIFYING_SOURCES')
    source_hashes={key:sha(path) for key,path in INPUTS.items()}
    assert source_hashes==EXPECTED,{'actual':source_hashes,'expected':EXPECTED}
    sys.path.insert(0,str(UPSTREAM/'code'))
    from title_name_regressions_core import aggregate_news_windows
    keys=['row_id','fund_id','security_id','company_id','interval_id','opening_report_id',
          'closing_report_id','start_date','end_date','net_sentiment','company_period_id']
    panel=pd.read_parquet(INPUTS['panel'],columns=keys)
    assert len(panel)==221899 and not panel.row_id.duplicated().any()
    assert panel.company_id.notna().all()
    panel['fund_id']=panel.fund_id.astype(str)
    for col in ['start_date','end_date']:panel[col]=pd.to_datetime(panel[col])
    intervals=pd.read_parquet(INPUTS['intervals'],columns=['interval_id','fund_id','start_date','end_date','opening_report_id','closing_report_id'])
    intervals['fund_id']=intervals.fund_id.astype(str)
    for col in ['start_date','end_date']:intervals[col]=pd.to_datetime(intervals[col])
    assert intervals.end_date.gt(intervals.start_date).all()
    assert not intervals.interval_id.duplicated().any()
    assert not intervals.duplicated(['fund_id','end_date']).any()
    current=panel[['interval_id','fund_id','start_date','end_date','opening_report_id','closing_report_id']].drop_duplicates()
    assert not current.interval_id.duplicated().any()
    validation=current.merge(intervals,on='interval_id',how='left',validate='one_to_one',suffixes=('_current','_calendar'),indicator=True)
    assert validation._merge.eq('both').all()
    for col in ['fund_id','start_date','end_date','opening_report_id','closing_report_id']:
        assert validation[col+'_current'].eq(validation[col+'_calendar']).all(),col
    mapping=panel[['row_id','fund_id','company_id','interval_id','start_date','end_date','opening_report_id']].rename(columns={
        'interval_id':'current_interval_id','start_date':'current_start_date','end_date':'current_end_date',
        'opening_report_id':'current_opening_report_id'})
    previous=intervals.rename(columns={'interval_id':'lag_interval_id','start_date':'lag_start_date',
        'end_date':'lag_end_date','opening_report_id':'lag_opening_report_id','closing_report_id':'lag_closing_report_id'})
    mapping=mapping.merge(previous,left_on=['fund_id','current_start_date'],right_on=['fund_id','lag_end_date'],how='left',validate='many_to_one',sort=False)
    has_previous=mapping.lag_interval_id.notna()
    assert int((~has_previous).sum())==1673
    assert mapping.loc[has_previous,'lag_closing_report_id'].eq(mapping.loc[has_previous,'current_opening_report_id']).all()
    assert mapping.loc[has_previous,'lag_end_date'].eq(mapping.loc[has_previous,'current_start_date']).all()
    assert mapping.loc[has_previous,'lag_start_date'].lt(mapping.loc[has_previous,'lag_end_date']).all()
    assert mapping.current_start_date.lt(mapping.current_end_date).all()
    needed=mapping.loc[has_previous,['company_id','lag_start_date','lag_end_date']].drop_duplicates().rename(columns={
        'lag_start_date':'period_start','lag_end_date':'period_end'})
    needed['company_period_id']=[window_id(c,s,e) for c,s,e in zip(needed.company_id,needed.period_start,needed.period_end)]
    assert not needed.company_period_id.duplicated().any()
    net=pd.read_parquet(INPUTS['net']);actual=net.loc[net.period_kind.eq('actual_report_interval')].copy()
    assert not actual.company_period_id.duplicated().any()
    assert not actual.duplicated(['company_id','period_start','period_end']).any()
    exists=needed.company_period_id.isin(actual.company_period_id)
    missing=needed.loc[~exists].sort_values('company_period_id').reset_index(drop=True)
    assert len(missing)==85
    existing=actual.loc[actual.company_period_id.isin(needed.loc[exists,'company_period_id'])].copy()
    existing_check=existing.merge(needed,on='company_period_id',validate='one_to_one',suffixes=('_source','_needed'))
    for col in ['company_id','period_start','period_end']:
        assert existing_check[col+'_source'].eq(existing_check[col+'_needed']).all(),col
    current_net=panel.merge(actual[['company_period_id','company_id','period_start','period_end','net_sentiment']],
        on='company_period_id',how='left',validate='many_to_one',suffixes=('_current','_source'))
    assert current_net.company_id_current.eq(current_net.company_id_source).all()
    assert current_net.start_date.eq(current_net.period_start).all() and current_net.end_date.eq(current_net.period_end).all()
    assert np.allclose(current_net.net_sentiment_current,current_net.net_sentiment_source,rtol=0,atol=1e-12)
    stage('AGGREGATING_ONLY_MISSING_WINDOWS',needed_windows=len(needed),existing_windows=len(existing),missing_windows=len(missing))
    events=pd.read_parquet(INPUTS['events'],columns=EVENT_FIELDS)
    assert events.is_representative.eq(True).all() and not events.date_inferred.eq(True).any()
    events=events.loc[events.company_id.isin(needed.company_id)].copy()
    events['event_date']=pd.to_datetime(events.event_date)
    derived_date=pd.to_datetime(events.event_timestamp,utc=True).dt.tz_convert('Europe/Moscow').dt.tz_localize(None).dt.normalize()
    assert (derived_date.eq(events.event_date)|(derived_date.isna()&events.event_date.isna())).all()
    undated=events.loc[events.event_date.isna()].groupby('company_id').size()
    def recompute(windows):
        targets=windows[['company_period_id','company_id','period_start','period_end']].rename(columns={
            'company_period_id':'interval_id','period_start':'news_start','period_end':'news_end'})
        result,_=aggregate_news_windows(targets,events)
        result=result.rename(columns={'interval_id':'company_period_id','Tone':'net_sentiment',
            'tone_observed':'net_sentiment_observed','valid_fraction':'classification_coverage',
            'news_status':'state','n_expected':'n_candidates'})
        result['period_start']=result.news_start;result['period_end']=result.news_end
        result['period_kind']='actual_report_interval';result['period_type']='actual_report_interval'
        result['strict_complete']=result.n_candidates.gt(0)&result.n_valid.eq(result.n_candidates)
        result['eligible_primary']=result.strict_complete&result.n_relevant.gt(0)
        result['net_sentiment']=result.net_sentiment.where(result.eligible_primary)
        result['n_irrelevant']=result.n_valid-result.n_relevant
        result['n_error']=result.n_attempted-result.n_valid
        result['n_unattempted']=result.n_candidates-result.n_attempted
        result['n_undated_company_candidates']=result.company_id.map(undated).fillna(0).astype('int64')
        result['source_id']='gemma_title_name_no_gaz_20261008_01'
        result['event_representatives_sha256']=EXPECTED['events']
        return result
    recomputed=recompute(missing)
    parity_expected=existing.sort_values('company_period_id').head(20).copy()
    parity_actual=recompute(parity_expected)
    parity=parity_expected[['company_period_id']+PARITY_FIELDS].merge(
        parity_actual[['company_period_id']+PARITY_FIELDS],on='company_period_id',validate='one_to_one',suffixes=('_expected','_actual'))
    parity_errors=[]
    for field in PARITY_FIELDS:
        a,b=parity[field+'_expected'],parity[field+'_actual']
        same=(a.eq(b)|(a.isna()&b.isna()))
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            same=pd.Series(np.isclose(a.to_numpy(float),b.to_numpy(float),rtol=0,atol=1e-12,equal_nan=True))
        if not same.all():parity_errors.append(field)
    assert len(parity)==20 and not parity_errors,parity_errors
    connection=duckdb.connect(':memory:',config={'threads':2,'memory_limit':'512MB'})
    member_windows=pd.concat([missing,parity_expected[['company_period_id','company_id','period_start','period_end']]],ignore_index=True)
    connection.register('windows',member_windows)
    connection.register('events',events[['pair_id','article_id','company_id','cluster_id','event_date','source',
        'attempted','valid','relevance','sentiment_label','pair_state','text_sha256']])
    members=connection.execute('''SELECT w.company_period_id,w.company_id,w.period_start,w.period_end,
        e.pair_id,e.article_id,e.cluster_id event_id,e.event_date,e.source,e.attempted,e.valid,e.relevance,
        e.sentiment_label,e.pair_state,e.text_sha256,coalesce(e.valid AND e.relevance='relevant',false) included_in_net_denominator
        FROM windows w JOIN events e ON w.company_id=e.company_id AND e.event_date>w.period_start AND e.event_date<=w.period_end
        ORDER BY w.company_period_id,e.pair_id''').fetchdf()
    connection.close()
    assert not members.duplicated(['company_period_id','pair_id']).any()
    assert members.event_date.gt(members.period_start).all() and members.event_date.le(members.period_end).all()
    digests={key:hashlib.sha256('\n'.join(group.pair_id.astype(str)).encode()).hexdigest() for key,group in members.groupby('company_period_id',sort=False)}
    counts=members.groupby('company_period_id').size()
    recomputed['n_members']=recomputed.company_period_id.map(counts).fillna(0).astype('int64')
    recomputed['sample_hash']=recomputed.company_period_id.map(digests)
    assert recomputed.n_members.eq(recomputed.n_candidates).all()
    for row in parity_expected.to_dict('records'):
        digest=digests.get(row['company_period_id'])
        assert (pd.isna(row['sample_hash']) and digest is None) or row['sample_hash']==digest
    stage('BUILDING_ROW_LAG_MAP',overlap_parity_windows=20,parity_errors=0)
    existing['lag_source_kind']='existing_actual_window'
    recomputed['lag_source_kind']='recomputed_missing_actual_window'
    columns=['company_period_id','company_id','period_kind','period_start','period_end','net_sentiment',
        'net_sentiment_observed','classification_coverage','state','strict_complete','eligible_primary',
        *COUNT_FIELDS,'n_members','sample_hash','source_id','event_representatives_sha256','lag_source_kind']
    windows=pd.concat([existing[columns],recomputed[columns]],ignore_index=True).sort_values('company_period_id').reset_index(drop=True)
    windows['source_net_full_sha256']=EXPECTED['net']
    windows['source_aggregate_code_sha256']=EXPECTED['core']
    assert len(windows)==len(needed) and not windows.company_period_id.duplicated().any()
    assert windows.n_positive.add(windows.n_negative).add(windows.n_neutral).eq(windows.n_relevant).all()
    strict=windows.n_candidates.gt(0)&windows.n_valid.eq(windows.n_candidates)&windows.n_relevant.gt(0)
    assert windows.eligible_primary.eq(strict).all()
    assert windows.loc[~strict,'net_sentiment'].isna().all()
    assert np.allclose(windows.loc[strict,'net_sentiment'],
        (windows.loc[strict,'n_positive']-windows.loc[strict,'n_negative'])/windows.loc[strict,'n_relevant'],rtol=0,atol=1e-12)
    renamed=windows.rename(columns={'company_id':'lag_company_id','company_period_id':'lag_company_period_id',
        'period_start':'lag_start_date','period_end':'lag_end_date','net_sentiment':'net_sentiment_lag1',
        'eligible_primary':'lag_eligible_primary','state':'lag_state','classification_coverage':'lag_classification_coverage',
        'sample_hash':'lag_window_sample_hash',**{col:'lag_'+col for col in COUNT_FIELDS}})
    join_cols=['lag_company_id','lag_start_date','lag_end_date','lag_company_period_id','net_sentiment_lag1',
        'lag_eligible_primary','lag_state','lag_source_kind','lag_classification_coverage','lag_window_sample_hash',
        *['lag_'+col for col in COUNT_FIELDS]]
    mapping=mapping.merge(renamed[join_cols],left_on=['company_id','lag_start_date','lag_end_date'],
        right_on=['lag_company_id','lag_start_date','lag_end_date'],how='left',validate='many_to_one',sort=False)
    has_previous=mapping.lag_interval_id.notna()
    assert mapping.loc[has_previous,'lag_company_id'].eq(mapping.loc[has_previous,'company_id']).all()
    assert mapping.loc[has_previous,'lag_company_period_id'].notna().all()
    mapping['lag_eligible_primary']=mapping.lag_eligible_primary.eq(True)
    mapping['lag_reason']=np.where(~has_previous,'no_previous_interval',np.where(mapping.lag_eligible_primary,'eligible',mapping.lag_state))
    mapping['lag_missing_reason']=mapping.lag_reason
    mapping.loc[~has_previous,'lag_source_kind']='no_previous_interval'
    mapping.loc[~has_previous,'lag_state']='no_previous_interval'
    for col in COUNT_FIELDS:mapping['lag_'+col]=mapping['lag_'+col].astype('Int64')
    assert len(mapping)==len(panel) and mapping.row_id.equals(panel.row_id)
    existing_eligible=int((mapping.lag_source_kind.eq('existing_actual_window')&mapping.lag_eligible_primary).sum())
    new_rows=int(mapping.lag_source_kind.eq('recomputed_missing_actual_window').sum())
    assert existing_eligible==193933 and new_rows==142,(existing_eligible,new_rows)
    assert mapping.loc[~mapping.lag_eligible_primary,'net_sentiment_lag1'].isna().all()
    assert np.isfinite(mapping.loc[mapping.lag_eligible_primary,'net_sentiment_lag1']).all()
    missing_members=members.loc[members.company_period_id.isin(missing.company_period_id)].copy()
    missing_members=missing_members.merge(recomputed[['company_period_id','eligible_primary']].rename(columns={
        'eligible_primary':'window_eligible_primary'}),on='company_period_id',validate='many_to_one')
    windows.to_parquet(OUT/'lag_company_windows.parquet',index=False,compression='zstd')
    mapping.to_parquet(OUT/'row_lag_map.parquet',index=False,compression='zstd')
    missing_members.to_parquet(OUT/'missing_window_members.parquet',index=False,compression='zstd')
    save('overlap_parity.json',dict(status='PASS',windows=20,fields=PARITY_FIELDS,
        company_period_ids=parity.company_period_id.tolist(),records=parity.to_dict('records'),member_hash_parity=True))
    final_hashes={key:sha(path) for key,path in INPUTS.items()}
    assert final_hashes==source_hashes
    audit=dict(status='PASS',panel_rows=len(panel),panel_funds=int(panel.fund_id.nunique()),
        panel_companies=int(panel.company_id.nunique()),full_calendar_intervals=len(intervals),
        rows_without_previous_interval=int((~has_previous).sum()),needed_unique_lag_windows=len(needed),
        existing_actual_lag_windows=len(existing),recomputed_missing_windows=len(missing),
        recomputed_window_rows=new_rows,existing_eligible_lag_rows=existing_eligible,
        total_eligible_lag_rows=int(mapping.lag_eligible_primary.sum()),
        newly_eligible_lag_rows=int((mapping.lag_source_kind.eq('recomputed_missing_actual_window')&mapping.lag_eligible_primary).sum()),
        missing_window_candidate_members=len(missing_members),
        lag_reason_counts=mapping.lag_reason.value_counts(dropna=False).to_dict(),
        window_source_counts=windows.lag_source_kind.value_counts().to_dict(),
        eligible_lag_sorted_row_ids_sha256=hashlib.sha256('\n'.join(sorted(mapping.loc[mapping.lag_eligible_primary,'row_id'])).encode()).hexdigest(),
        overlap_parity_windows=20,overlap_numeric_and_member_hash_parity=True,
        predecessor_calendar_identity=True,previous_closing_report_equals_current_opening=True,
        exact_window_order='lag_start < lag_end == current_start < current_end',
        lookup_policy='actual_report_interval only; no calendar fallback, holding gate, shifted surviving rows, zero fill or forward fill',
        event_date_policy='Frozen Moscow publication date; lag window (lag_start,lag_end]',
        source_paths={key:str(path) for key,path in INPUTS.items()},source_sha256=source_hashes,
        source_sha256_unchanged=True,current_score_unchanged=True,
        no_new_LLM=True,no_fits=True,protected_sources_changed=False,
        output_row_counts={'lag_company_windows.parquet':len(windows),'row_lag_map.parquet':len(mapping),
                           'missing_window_members.parquet':len(missing_members)})
    save('data_audit.json',audit)
    files={name:dict(sha256=sha(OUT/name),bytes=(OUT/name).stat().st_size) for name in [
        'lag_company_windows.parquet','row_lag_map.parquet','missing_window_members.parquet','data_audit.json','overlap_parity.json']}
    save('hash_manifest.json',dict(status='READY',files=files,source_sha256=source_hashes,
        builder_script_sha256=sha(__file__),source_paths={key:str(path) for key,path in INPUTS.items()}))
    save('data_ready.json',dict(ready=True,audit_status='PASS',manifest_sha256=sha(OUT/'hash_manifest.json'),
        total_eligible_lag_rows=audit['total_eligible_lag_rows'],no_fits=True))
    stage('READY',**{key:audit[key] for key in ['panel_rows','needed_unique_lag_windows','recomputed_missing_windows','total_eligible_lag_rows','newly_eligible_lag_rows']})


if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():save('data_failure.json',dict(error=traceback.format_exc(),no_fits=True))
        raise
