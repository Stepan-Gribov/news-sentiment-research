"""Prepare only two immutable benchmark inputs on research-host; no H3 model fitting.
Reuse the sealed IMOEX CSV and verified RGBITR raw/daily sources. No history
download unless an actual uncovered requirement is established (none expected).
"""
from pathlib import Path
import datetime as dt
import hashlib,json,shutil,sys
import numpy as np,pandas as pd,duckdb,requests

BASE=Path('./analysis_outputs')
ROOT=BASE/'h3_index_excess_20261104/benchmarks/benchmark_release_20261104_v1'
ROOT.mkdir(parents=True,exist_ok=True)
IMOEX=BASE/'h1_macro_no_time_20261031/inputs/raw_external_2026-09-30/moex/indices/IMOEX_daily_history_2012-01-01_2026-08-31.csv'
FACTORS=BASE/'h3_alpha_20261104/factors'
RGB=FACTORS/'normalized/RGBITR_daily.parquet'
SCORES=BASE/'h3_relaxed_news_20261104/panels/0a4d9c7365f0faa1/relaxed_fund_period_scores.parquet'
RETURNS=BASE/'h3_alpha_20261104/outputs/run_d0440fad412ee5e9/monthly_returns_adapter.parquet'
EXPECTED={'IMOEX':'96e83ceb46c314f5f79900ec6361e77afe49ed28c71a4dbd2c9ec530eddafed3','RGBITR':'ac2be7fcd189a9a7780908b8bf968c48ab09dd6fce0273257c4fac53ade56138','scores':'b91618a88d33e3ed0a705ac5e30c7bd34542a4266a8097b99791c7d09a8b2f12','returns':'f079a4a6a7e24ce9561aee94d53809ccfe2d872b4ef2b78419243f4a18cf8c8e'}
checks=[];inputs={};http_log=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def record(p):p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def clean(v):
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
 if isinstance(v,(np.integer,)):return int(v)
 if isinstance(v,(np.bool_,)):return bool(v)
 if isinstance(v,(Path,pd.Timestamp,dt.date)):return str(v)
 return v
def save(name,d):
 p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(clean(d),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def ck(n,v,**kw):checks.append(dict(check=n,status='PASS' if bool(v) else 'FAIL',**kw))
def block(j,key):return pd.DataFrame(j[key]['data'],columns=j[key]['columns'])
if (ROOT/'benchmark_ready.json').exists():raise SystemExit('Already sealed: use existing benchmark_ready.json')
for k,p in [('IMOEX',IMOEX),('RGBITR',RGB),('scores',SCORES),('returns',RETURNS)]:
 inputs[k]=record(p);ck('pinned_'+k,inputs[k]['sha256']==EXPECTED[k])
inputs['builder']=record(__file__)
spec=Path(__file__).with_name('h3_index_excess_spec_20261104.json')
inputs['requested_spec']=record(spec);ck('frozen_index_relative_spec',inputs['requested_spec']['sha256']=='b0d982e5bf57866a2ea6d0fca5e57186a54076d4359f5ebe42ec752224dbd74b')
raw=ROOT/'raw';dailyroot=ROOT/'daily';raw.mkdir(exist_ok=True);dailyroot.mkdir(exist_ok=True)

# Exact reused archive snapshot and original collector endpoint, not a new fetch.
snapshot=raw/IMOEX.name
if not snapshot.exists():shutil.copy2(IMOEX,snapshot)
ck('IMOEX_snapshot_exact',sha(snapshot)==EXPECTED['IMOEX'])
im=pd.read_csv(IMOEX);ck('IMOEX_identity_currency_board_session',im.SECID.eq('IMOEX').all() and im.BOARDID.eq('SNDX').all() and im.CURRENCYID.eq('RUB').all() and pd.to_numeric(im.TRADINGSESSION).eq(3).all())
imeta=raw/'IMOEX_metadata.json'
if not imeta.exists():
 response=requests.get('https://iss.moex.com/iss/securities/IMOEX.json',params={'iss.meta':'off'},timeout=(10,25));response.raise_for_status();imeta.write_bytes(response.content)
 http_log.append(dict(url=response.url,tls_verify=True,downloaded_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),**record(imeta)))
ij=json.loads(imeta.read_text(encoding='utf-8'));idsc=block(ij,'description').set_index('name').value.to_dict()
ck('IMOEX_official_price_RUB_identity',idsc['SECID']=='IMOEX' and idsc['CALCMODE']=='PR' and idsc['CURRENCYID']=='RUB')
imoex_url='https://iss.moex.com/iss/history/engines/stock/markets/index/securities/IMOEX.json?from=2012-01-01&till=2026-08-31&iss.meta=off'
idaily=pd.DataFrame(dict(series_id='IMOEX',date=pd.to_datetime(im.TRADEDATE),index_value=pd.to_numeric(im.CLOSE),source_path=str(IMOEX),source_sha256=EXPECTED['IMOEX'],source_row_number=np.arange(len(im))+2,source_url=imoex_url,source_url_role='Existing sealed CSV from original official paginated collector; not a new download',BOARDID='SNDX',TRADINGSESSION='3',currency='RUB',calcmode='PR'))

# Validate RGBITR daily values against every original official raw page/log SHA.
logs=[json.loads(x) for x in (FACTORS/'raw_requests.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
lookup={str(Path(z['path'])):z for z in logs}
rd=pd.read_parquet(RGB);rmeta=FACTORS/'raw/moex/RGBITR/metadata.json';rj=json.loads(rmeta.read_text(encoding='utf-8'));rdesc=block(rj,'description').set_index('name').value.to_dict()
ck('RGBITR_official_total_return_RUB_identity',rdesc['SECID']=='RGBITR' and rdesc['CALCMODE']=='TR' and rdesc['CURRENCYID']=='RUB')
pages=[];raw_records=[]
for name in sorted(rd.raw_page.unique()):
 p=FACTORS/'raw/moex/RGBITR'/name;z=lookup[str(p)];ck('RGBITR_raw_SHA:'+name,sha(p)==z['sha256'] and z['tls_verify'] is True)
 d=block(json.loads(p.read_text(encoding='utf-8')),'history');d['source_path']=str(p);d['source_sha256']=z['sha256'];d['source_url']=z['url'];d['source_row_number']=np.arange(len(d));pages.append(d);raw_records.append(record(p)|{'url':z['url'],'tls_verify':True})
rr=pd.concat(pages,ignore_index=True);rr['date']=pd.to_datetime(rr.TRADEDATE)
joined=rd.merge(rr,on='date',validate='one_to_one',suffixes=('_normalized','_raw'))
ck('RGBITR_all4746_daily_levels_raw_parity',len(joined)==len(rd)==4746 and np.array_equal(joined.index_value.to_numpy(float),pd.to_numeric(joined.CLOSE).to_numpy(float)))
ck('RGBITR_raw_identity_board_session',rr.SECID.eq('RGBITR').all() and rr.BOARDID.eq('SNDX').all() and pd.to_numeric(rr.TRADINGSESSION).eq(3).all())
rdaily=pd.DataFrame(dict(series_id='RGBITR',date=rr.date,index_value=pd.to_numeric(rr.CLOSE),source_path=rr.source_path,source_sha256=rr.source_sha256,source_row_number=rr.source_row_number,source_url=rr.source_url,source_url_role='Original archived official HTTP request URL',BOARDID='SNDX',TRADINGSESSION='3',currency='RUB',calcmode='TR'))
save('source_inputs.json',dict(inputs=inputs,IMOEX_archive_query=imoex_url,IMOEX_archive_transport='Reused sealed input; new history requests=0',RGBITR_raw_pages=raw_records,RGBITR_metadata=record(rmeta),IMOEX_metadata=record(imeta),metadata_request_log=(json.loads((ROOT/'metadata_request.json').read_text(encoding='utf-8')) if (ROOT/'metadata_request.json').exists() else None),new_requests=http_log))

# Identify the actually needed forecast months and latest possible past controls
# before imposing any benchmark, risk-free or alpha gate.
scores=pd.read_parquet(SCORES);targets=scores.loc[scores.universe_eligible.eq(True)&scores.current_eligible.eq(True)&scores.fund_focus.isin(['equity','bond','mixed'])].copy()
fund=pd.read_parquet(RETURNS);c=duckdb.connect(config={'threads':2,'memory_limit':'1GB'});c.register('o',targets);c.register('r',fund)
control=c.execute('''SELECT o.fund_id,o.interval_id,max(r.month) candidate_past_month FROM o LEFT JOIN r ON o.fund_id=r.fund_id
 AND r.diagnostic_eligible=true AND isfinite(r.fund_return) AND r.fund_return>=-1
 AND last_day(r.return_period_end)<=o.end_date AND r.return_period_end<=o.end_date
 AND r.start_valuation_date<=o.end_date AND r.end_valuation_date<=o.end_date GROUP BY o.fund_id,o.interval_id''').fetchdf()
origin_months=pd.to_datetime(targets.end_date).dt.to_period('M')
needed=set(str(m+h) for m in origin_months for h in [1,2,3])|set(control.candidate_past_month.dropna().astype(str))
save('coverage_requirements.json',dict(origin_count=len(targets),first_origin=str(targets.end_date.min()),last_origin=str(targets.end_date.max()),first_future_month=str(origin_months.min()+1),last_future_month=str(origin_months.max()+3),earliest_latest_available_past_fund_month=control.candidate_past_month.min(),last_latest_available_past_fund_month=control.candidate_past_month.max(),needed_months=sorted(needed),origins_without_any_past_fund_month=int(control.candidate_past_month.isna().sum())))
rows=[];endpoints=[];gaps=[]
calendar=pd.period_range('2012-01','2026-08',freq='M')
for series,d in [('IMOEX',idaily),('RGBITR',rdaily)]:
 d=d.sort_values('date').reset_index(drop=True);d['month']=d.date.dt.to_period('M').astype(str)
 ck(series+':daily_unique_positive_finite',not d.date.duplicated().any() and np.isfinite(d.index_value).all() and d.index_value.gt(0).all())
 d['previous_observed_date']=d.date.shift();d['observed_gap_calendar_days']=d.date.diff().dt.days
 exceptional=d.loc[d.date.between('2022-03-01','2022-03-31')&d.observed_gap_calendar_days.gt(14)]
 ck(series+':halt_gap_unfilled',len(exceptional)==1 and exceptional.previous_observed_date.iloc[0]==pd.Timestamp('2022-02-25'))
 gaps.extend(exceptional[['series_id','previous_observed_date','date','observed_gap_calendar_days']].to_dict('records'))
 d.to_parquet(dailyroot/(series+'_daily.parquet'),index=False,compression='zstd')
 groups={k:g for k,g in d.groupby('month')}
 for month in calendar:
  a=groups.get(str(month-1));b=groups.get(str(month));start=None if a is None else a.iloc[-1];end=None if b is None else b.iloc[-1]
  reason='NO_CURRENT_MONTH_CLOSE' if end is None else 'NO_PREVIOUS_ADJACENT_MONTH_CLOSE' if start is None else 'VERIFIED_ADJACENT_MONTH_CLOSES'
  value=float(end.index_value/start.index_value-1) if reason=='VERIFIED_ADJACENT_MONTH_CLOSES' else np.nan
  halt=month in [pd.Period('2022-02'),pd.Period('2022-03')]
  row=dict(series_id=series,benchmark_id=series,month=str(month),return_basis='price_return' if series=='IMOEX' else 'government_bond_total_return',currency='RUB',benchmark_return=value,return_m=value,return_verified=bool(np.isfinite(value)),benchmark_eligible=bool(np.isfinite(value)),return_status='VERIFIED_OBSERVED_CLOSES_HALT_AFFECTED' if halt and np.isfinite(value) else reason,return_period_end=month.end_time.normalize(),index_start_date=pd.NaT if start is None else start.date,index_end_date=pd.NaT if end is None else end.date,index_start_value=np.nan if start is None else start.index_value,index_end_value=np.nan if end is None else end.index_value,start_valuation_date=pd.NaT if start is None else start.date,end_valuation_date=pd.NaT if end is None else end.date,start_endpoint_calendar_age_days=np.nan if start is None else ((month-1).end_time.normalize()-start.date).days,end_endpoint_calendar_age_days=np.nan if end is None else (month.end_time.normalize()-end.date).days,observations_in_month=0 if b is None else len(b),halt_2022_affected=halt,ordinary_endpoint=not halt,start_source_path=None if start is None else start.source_path,end_source_path=None if end is None else end.source_path,start_source_row=None if start is None else int(start.source_row_number),end_source_row=None if end is None else int(end.source_row_number),start_source_url=None if start is None else start.source_url,end_source_url=None if end is None else end.source_url,start_source_sha256=None if start is None else start.source_sha256,end_source_sha256=None if end is None else end.source_sha256)
  rows.append(row)
  if end is not None:endpoints.append(dict(series_id=series,month=str(month),actual_close_date=end.date,close=end.index_value,source_path=end.source_path,source_sha256=end.source_sha256,source_row_number=int(end.source_row_number),source_url=end.source_url,n_daily_observations=len(b)))
 out=pd.DataFrame([r for r in rows if r['series_id']==series])
 ck(series+':continuous176_calendar_month_rows',out.month.tolist()==calendar.astype(str).tolist())
 need=out.loc[out.month.isin(needed)];ck(series+':all_needed_future_and_past_months_covered',len(need)==len(needed) and need.return_verified.all())
 ck(series+':no_borrowing_across_missing_month',out.loc[~out.return_verified,'benchmark_return'].isna().all())
 ck(series+':actual_endpoints_within_adjacent_months',all(pd.Timestamp(z.index_start_date).to_period('M')==pd.Period(z.month)-1 and pd.Timestamp(z.index_end_date).to_period('M')==pd.Period(z.month) for z in out.loc[out.return_verified].itertuples(index=False)))
result=pd.DataFrame(rows);result.to_parquet(ROOT/'benchmarks_monthly.parquet',index=False,compression='zstd');result.to_csv(ROOT/'benchmarks_monthly.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(endpoints).to_parquet(ROOT/'monthly_closes.parquet',index=False,compression='zstd');pd.DataFrame(gaps).to_csv(ROOT/'observed_2022_halt_gaps.csv',index=False)
halt=result.loc[result.halt_2022_affected,['series_id','month','index_start_date','index_end_date','index_start_value','index_end_value','benchmark_return','return_status']]
save('halt_2022_endpoint_evidence.json',dict(rows=halt.to_dict('records'),observed_gaps=gaps,official_sources=['https://www.moex.com/n45534?nt=206','https://www.moex.com/n45570'],interpretation='Observed last trading closes only. February closes are dated2022-02-25, not2022-02-28. March returns use that observed prior-month endpoint; no daily closes are generated inside the suspension. Halt flags are disclosed, not a new exclusion or a new variant.'))
# Independent monthly arithmetic/calendar QA against the normalized daily close
# table using a separate DuckDB aggregation; no eligibility from RF/MCFTR.
c.register('daily',pd.concat([idaily,rdaily],ignore_index=True));c.register('monthly',result)
comparison=c.execute('''WITH e AS (SELECT series_id,strftime(date,'%Y-%m') AS month_key,arg_max(index_value,date) AS close_level FROM daily GROUP BY ALL)
 SELECT count(*) FROM monthly m LEFT JOIN e b ON b.series_id=m.series_id AND b.month_key=m.month LEFT JOIN e a ON a.series_id=m.series_id AND a.month_key=strftime(strptime(m.month,'%Y-%m')-INTERVAL1 MONTH,'%Y-%m')
 WHERE (m.return_verified AND (a.close_level IS NULL OR b.close_level IS NULL OR abs(m.benchmark_return-(b.close_level/a.close_level-1))>1e-13)) OR (NOT m.return_verified AND a.close_level IS NOT NULL AND b.close_level IS NOT NULL)'''.replace('INTERVAL1','INTERVAL 1')).fetchone()[0]
ck('independent_monthly_endpoint_ratio_arithmetic',comparison==0)
for k,p in [('IMOEX',IMOEX),('RGBITR',RGB),('scores',SCORES),('returns',RETURNS)]:ck('source_unchanged:'+k,sha(p)==EXPECTED[k])
summary=dict(status='READY' if all(x['status']=='PASS' for x in checks) else 'FAIL',checks_pass=sum(x['status']=='PASS' for x in checks),checks_fail=sum(x['status']=='FAIL' for x in checks),checks=checks,monthly_path=str(ROOT/'benchmarks_monthly.parquet'),monthly_sha256=sha(ROOT/'benchmarks_monthly.parquet'),series=dict(IMOEX=dict(rows=176,eligible=int(result.loc[result.series_id.eq('IMOEX'),'return_verified'].sum()),first='2012-01',last='2026-08',basis='price return'),RGBITR=dict(rows=176,eligible=int(result.loc[result.series_id.eq('RGBITR'),'return_verified'].sum()),first='2012-01',last='2026-08',basis='total return')),target_future_coverage=dict(first='2014-01',last='2026-08',all_needed_future_and_latest_past_controls_covered=all(x['status']=='PASS' for x in checks if 'needed' in x['check'])),history_download_requests=0,new_metadata_https_verification=True,no_TLS_disable=True,no_fill0=True,no_forwardfill=True,no_alpha_or_unrelated_factor_gates=True,halt_policy='Return is the observed adjacent-month last-trading-close ratio, with actual dates and explicit halt flags. No fake2022-02-28 endpoint, no zero daily return during halt, no additional halt exclusion.',created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
save('benchmark_audit.json',summary)
if summary['checks_fail']:print(json.dumps(clean(summary),ensure_ascii=False));raise SystemExit(1)
files={str(p.relative_to(ROOT)).replace('\\','/'):record(p) for p in ROOT.rglob('*') if p.is_file() and p.name not in ['manifest.json','benchmark_ready.json']}
save('manifest.json',dict(status='SEALED_VERIFIED_INDEX_BENCHMARKS',inputs=inputs,files=files,return_units='simple monthly decimal returns',series_ids=['IMOEX','RGBITR'],independent_eligibility=True,created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
ready={k:v for k,v in summary.items() if k!='checks'};ready.update(manifest_path=str(ROOT/'manifest.json'),manifest_sha256=sha(ROOT/'manifest.json'),return_column='benchmark_return',eligibility_column='return_verified',month_column='month',series_column='series_id',actual_start_date_column='index_start_date',actual_end_date_column='index_end_date')
save('benchmark_ready.json',ready)
print(json.dumps(clean(ready),ensure_ascii=False))
