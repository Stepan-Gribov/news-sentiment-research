"""Finalize action-aware observed-NAV release after the single collection job exits.
No alpha fits. No writes to older releases. Prefer documented official unit values
for the four exactly registration-linked Arsagera API series.
"""
import pathlib,json,hashlib,importlib.util,subprocess,datetime as dt,sys,os,traceback,runpy
import pandas as pd,duckdb,numpy as np
R=pathlib.Path('./analysis_outputs/h3_alpha_20261104/fund_history')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def q(p):return "'"+str(p).replace('\\','/').replace("'","''")+"'"
sys.stdout=(R/'finalization.stdout.log').open('a',encoding='utf-8',buffering=1);sys.stderr=(R/'finalization.stderr.log').open('a',encoding='utf-8',buffering=1)
def failure_hook(kind,value,tb):
 (R/'finalization_failure.json').write_text(json.dumps(dict(error=repr(value),traceback=''.join(traceback.format_exception(kind,value,tb)),active_pid=os.getpid()),ensure_ascii=False,indent=2),encoding='utf-8')
 sys.__excepthook__(kind,value,tb)
sys.excepthook=failure_hook
if not (R/'v3_action_aware/sealed_outputs.json').exists():
 cp=json.loads((R/'checkpoint.json').read_text(encoding='utf-8'));assert cp['stage']=='COMPLETED' or (R/'v2_single_id_extended/sealed_outputs.json').exists(),cp
 calendar=pathlib.Path('./analysis_outputs/h3_alpha_20261104/factors/normalized/MCFTR_daily.parquet');assert sha(calendar)=='6373ddcdfa68964a740b83d4218a000e3d47f60a0b42669f49f9258cc9c2b07f'
 # Saved event evidence is collected before any coefficient inspection by this worker.
 action_summaries=sorted((R/'action_evidence').glob('summary_*.json'));a=json.loads(action_summaries[-1].read_text(encoding='utf-8'))
 assert a['profile_files']==603,'Run the public-action parser after all profiles are collected'
 c=duckdb.connect(config={'threads':2,'memory_limit':'2GB'});c.execute("SET TimeZone='Europe/Moscow'")
 o=pd.read_parquet(R/'official_sources/arsagera_official_unit_nav.parquet');official=pd.DataFrame(dict(timestamp_ms_source=pd.array([pd.NA]*len(o),dtype='Int64'),unit_nav=o.unit_value.astype(float),fund_id=o.fund_id.astype(str),series_name_source=o.series_id,source_kind='official_Arsagera_API',source_url=o.source_url,raw_file=o.source_path,raw_sha256=o.source_sha256,series_index=-1,point_index=o.json_pointer.str.rsplit('/',n=1).str[-1].astype('int64'),request_currency_id=pd.array([pd.NA]*len(o),dtype='Int64'),currency=o.currency,identity_series_name_exact=True,value_date=pd.to_datetime(o.valuation_date),source_json_pointer=o.json_pointer))
 c.register('official',official);src=R/'extended_unit_nav_daily.parquet';assert src.exists()
 c.execute('CREATE TEMP VIEW fresh AS SELECT * FROM read_parquet('+q(src)+') WHERE unit_nav>0 AND isfinite(unit_nav)')
 c.execute('CREATE TEMP VIEW old_cache AS FROM read_parquet('+q(R/'cached_unit_nav_daily.parquet')+')')
 c.execute('CREATE TEMP TABLE same_basis AS SELECT e.fund_id FROM fresh e JOIN old_cache p USING(fund_id,value_date) GROUP BY e.fund_id HAVING max(abs(e.unit_nav-p.unit_nav)/greatest(abs(p.unit_nav),1))<=1e-10')
 c.execute("CREATE TEMP VIEW provider AS SELECT *, '/'||cast(series_index AS VARCHAR)||'/data/'||cast(point_index AS VARCHAR) source_json_pointer FROM (SELECT * FROM fresh UNION ALL BY NAME SELECT p.* FROM old_cache p ANTI JOIN fresh e ON p.fund_id=e.fund_id AND p.value_date=e.value_date WHERE p.fund_id IN (SELECT fund_id FROM same_basis) OR p.fund_id NOT IN (SELECT DISTINCT fund_id FROM fresh))")
 c.execute('CREATE TEMP TABLE merged AS SELECT p.* FROM provider p ANTI JOIN official o ON p.fund_id=o.fund_id AND p.value_date=o.value_date UNION ALL BY NAME SELECT * FROM official')
 merged=R/'v3_source_unit_nav_daily.parquet';c.execute('COPY merged TO '+q(merged)+' (FORMAT PARQUET,COMPRESSION ZSTD)');c.close()
 source=(R/'collect_h3_fund_history_20261104.py').read_text(encoding='utf-8')
 old="calendar_path=Path('./analysis_outputs/h1_macro_no_time_20261031/inputs/raw_external_2026-09-30/moex/indices/IMOEX_daily_history_2012-01-01_2026-08-31.csv')\n cal=pd.read_csv(calendar_path);datecol=next(x for x in cal.columns if x.upper()=='TRADEDATE');days=np.sort(pd.to_datetime(cal[datecol]).dropna().unique().astype('datetime64[ns]'))"
 new="calendar_path=Path('./analysis_outputs/h3_alpha_20261104/factors/normalized/MCFTR_daily.parquet')\n cal=pd.read_parquet(calendar_path);days=np.sort(pd.to_datetime(cal['date']).dropna().unique().astype('datetime64[ns]'))"
 assert old in source;source=source.replace(old,new)
 old="grid['fund_return']=grid.fund_return_observed.where(grid.return_verified)"
 new="action_summaries=sorted((R/'action_evidence').glob('summary_*.json'));act=json.loads(action_summaries[-1].read_text(encoding='utf-8'));known=pd.read_parquet(act['cash_file']) if act['cash_file'] else pd.DataFrame()\n affected=set()\n if not known.empty:\n  for field in ['record_date','payment_date']:\n   for row in known.loc[known[field].notna(),['fund_id',field]].itertuples(index=False,name=None):affected.add((str(row[0]),str(pd.Timestamp(row[1]).to_period('M'))))\n grid['known_distribution_date_unresolved']=[(f,m) in affected for f,m in zip(grid.fund_id,grid.month)]\n grid['return_verified']=grid.return_verified & ~grid.known_distribution_date_unresolved\n grid['known_split_status']=np.where(grid.fund_id.eq('214'),'known20260814_already_rebased_by_source','no_complete_split_history')\n grid['fund_return']=grid.fund_return_observed.where(grid.return_verified)"
 assert old in source;source=source.replace(old,new)
 source=source.replace("'unit_NAV_change_cash_distributions_and_splits_not_fully_verified'","'UNIT_VALUE_KNOWN_ADJUSTMENTS'")
 source=source.replace("default='eligible_observed_NAV_return')","default='eligible_observed_NAV_return')\n grid.loc[grid.known_distribution_date_unresolved,'return_missing_reason']='known_distribution_ex_date_unresolved'\n grid['source_vintage']='collected20261104';grid['publication_timestamp']=pd.NaT;grid['publication_lag_status']='not_verified_for_most_source_points'")
 source=source.replace("grid['source_id']='investfunds_public_chart'","grid['source_id']=np.where(grid.source_kind.eq('official_Arsagera_API'),'arsagera_official_API','investfunds_public_chart_singleID')")
 source=source.replace("grid['distribution_status']='unverified_completeness'","grid['distribution_status']=np.where(grid.known_distribution_date_unresolved,'known_event_ex_date_unresolved','unverified_completeness')")
 source=source.replace("grid['split_status']='unverified_completeness'","grid['split_status']=np.where(grid.fund_id.eq('214'),'known20260814_source_rebased_other_history_unknown','unverified_completeness')")
 normalizer=R/'normalization_v3_20261104.py';normalizer.write_text(source,encoding='utf-8')
 spec=importlib.util.spec_from_file_location('h3_normalization_v3',normalizer);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 u,reports,origins,cross=m.load_context()
 # For historically conflicting registrations retain only the final continuously observed registration span.
 # Never concatenate returns across distinct registration numbers under the same numeric provider ID.
 cross['historical_registration_conflict']=cross.registration_count.gt(1);cross['selected_identity_span']='all_consistent_disclosures'
 for fund,g in reports.groupby('fund_id',sort=False):
  g=g.loc[g.registration_normalized.ne('')].sort_values('report_date');idx=cross.index[cross.fund_id.eq(fund)]
  if not len(g) or not len(idx) or len(g.registration_normalized.unique())<=1:continue
  current=g.registration_normalized.iloc[-1];other=g.loc[g.registration_normalized.ne(current),'report_date'];cutoff=other.max()
  span=g.loc[g.report_date.gt(cutoff)&g.registration_normalized.eq(current)]
  if len(span)>=2:
   cross.loc[idx,'registration_number']=current;cross.loc[idx,'identity_unambiguous']=True;cross.loc[idx,'identity_observed_from']=span.report_date.min();cross.loc[idx,'identity_observed_to']=span.report_date.max();cross.loc[idx,'selected_identity_span']='last_contiguous_exact_registration_span';cross.loc[idx,'identity_source']='Exact dated disclosure registration span after last conflicting observation; earlier registration history excluded'
 profiles=pd.read_parquet(R/'fund_profile_evidence.parquet');extra=dict(zip(profiles.fund_id,profiles.registration_candidates));cross['identity_extended_by_fresh_profile']=cross.apply(lambda x:bool(x.identity_unambiguous and x.registration_number in str(extra.get(x.fund_id,'')).split('|')),axis=1);cross.loc[cross.identity_extended_by_fresh_profile,'identity_observed_to']=pd.Timestamp('2026-11-04')
 D=R/'v3_action_aware';D.mkdir(exist_ok=True)
 policy=dict(frozen_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),alpha_coefficients_inspected=False,provider_vintage_rule='Fresh single-ID preferred; preserve unavailable/closed funds from the earlier verified cache. Fill a fresh-series gap from old cache only when overlap establishes the same unit-value basis; no blind vintage splice',source_preference='Official Arsagera API values by exact code-registration crosswalk supersede same-date secondary values for four funds; otherwise single-ID provider histories',calendar=dict(path=str(calendar),sha256=sha(calendar),range='2007-12-03..2026-11-03'),action_snapshot=a,identity_rule='For conflicting historical registration IDs use only final continuous same-registration disclosure span with at least two dated observations; no stitching across registration changes',known_cash_rule='Unverified ex-date: exclude the calendar months of announced record/payment dates from observed-NAV return eligibility, preserve raw values and known events; no unknown payouts asserted zero',known_split_rule='Fund214 known100-for-1 split already normalized by source, independently checked against official prices, no second multiplication',untouched_large_move='Fund4091 Jan2024 +2146% independently reported by official manager; kept, no magnitude trimming',basis='UNIT_VALUE_KNOWN_ADJUSTMENTS; total_return_verified remains false without complete payout/split evidence',source_files={str(src):sha(src),str(merged):sha(merged),str(R/'official_sources/arsagera_official_unit_nav.parquet'):sha(R/'official_sources/arsagera_official_unit_nav.parquet')})
 (D/'normalization_policy.json').write_text(json.dumps(policy,ensure_ascii=False,indent=2),encoding='utf-8')
 ready=m.release('v3_action_aware',merged,u,reports,origins,cross);m.stage('FINAL_ACTION_AWARE_RELEASE_READY',release='v3_action_aware',eligible36_funds=ready['counts']['eligible36_funds']);print(json.dumps(ready,ensure_ascii=False))
else:
 print('Previously sealed v3 preserved',flush=True)
deadline=dt.datetime(2026,11,4,4,30,tzinfo=dt.timezone.utc)
retry_pass=0
while True:
 retry_pass+=1
 runpy.run_path(str(R/'collect_h3_fund_history_20261104_official_monthly.py'),run_name='h3_official_monthly_data_collection')
 summary=json.loads((R/'official_monthly_returns/summary.json').read_text(encoding='utf-8'))
 if (summary['errors']==0 and summary['funds']==4) or dt.datetime.now(dt.timezone.utc)>=deadline:
  complete=summary['errors']==0 and summary['funds']==4
  (R/'collection_completed.json').write_text(json.dumps(dict(stage='COMPLETED' if complete else 'COMPLETED_WITH_SOURCE_GAPS',active_pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),final_release='v3_action_aware',official_monthly_complete=complete,official_monthly_summary=str(R/'official_monthly_returns/summary.json')),indent=2),encoding='utf-8')
  break
 import time
 retry_at=dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=5)
 for heartbeat in range(10):
  v=dict(stage='SOURCE_BACKOFF_RETRY_PENDING',active_pid=os.getpid(),pass_number=retry_pass,next_retry_utc=retry_at.isoformat(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),heartbeat=heartbeat,completed_reports=summary['reports'],completed_funds=summary['funds'],source_errors=summary['errors'],ready_numeric_release='v3_action_aware',no_access_or_TLS_bypass=True)
  path=R/'official_monthly_returns/checkpoint.json';tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
  time.sleep(30)
