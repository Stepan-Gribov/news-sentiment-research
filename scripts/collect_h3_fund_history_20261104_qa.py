"""Independent bounded QA of the first immutable observed unit-NAV release."""
import pathlib,json,hashlib,datetime as dt
import duckdb,pandas as pd,numpy as np
R=pathlib.Path('./analysis_outputs/h3_alpha_20261104/fund_history');D=R/'v1_existing_cache';Q=R/'qa';Q.mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def q(p):return "'"+str(p).replace('\\','/').replace("'","''")+"'"
c=duckdb.connect(config={'threads':2,'memory_limit':'2GB'});c.execute("SET TimeZone='Europe/Moscow'")
c.execute('CREATE TEMP VIEW r AS FROM read_parquet('+q(D/'fund_monthly_returns.parquet')+')');c.execute('CREATE TEMP VIEW e AS FROM read_parquet('+q(D/'history_eligibility_origins.parquet')+')');checks=[]
def check(name,sql):
 n=int(c.execute(sql).fetchone()[0]);checks.append(dict(check=name,violations=n,status='PASS' if n==0 else 'FAIL'));assert n==0,(name,n)
seal=json.loads((D/'sealed_outputs.json').read_text(encoding='utf-8'))
for n,v in seal['files'].items():assert sha(D/n)==v['sha256'],n
checks.append(dict(check='release_sealed_hashes',violations=0,status='PASS'))
check('fund_month_unique','SELECT count(*) FROM (SELECT fund_id,month,count(*) n FROM r GROUP BY ALL HAVING n>1)')
check('scope_603_funds','SELECT abs(count(distinct fund_id)-603) FROM r')
check('strict_formula','SELECT count(*) FROM r WHERE return_verified AND abs(fund_return-(unit_nav/previous_unit_nav-1))>1e-12')
check('strict_finite','SELECT count(*) FROM r WHERE return_verified AND (NOT isfinite(fund_return) OR fund_return< -1)')
check('strict_date_order','SELECT count(*) FROM r WHERE return_verified AND (value_date>month_end OR previous_value_date>=value_date OR previous_value_date>cast(last_day(month_end-INTERVAL 1 MONTH) AS DATE))')
check('strict_stale_gate','SELECT count(*) FROM r WHERE return_verified AND (source_age_trading_days>3 OR source_age_trading_days<0 OR NOT trading_calendar_verified)')
check('strict_identity','SELECT count(*) FROM r WHERE return_verified AND (NOT identity_unambiguous OR value_date<identity_observed_from OR value_date>identity_observed_to OR previous_value_date<identity_observed_from OR previous_value_date>identity_observed_to)')
check('no_total_NAV_substitution',"SELECT count(*) FROM r WHERE return_verified AND (source_id<>'investfunds_public_chart' OR NOT starts_with(series_id,'investfunds:') OR unit_value<>unit_nav)")
check('unknown_actions_not_certified','SELECT count(*) FROM r WHERE total_return_verified OR distribution_verified OR split_verified')
check('origins_unique','SELECT count(*) FROM (SELECT fund_id,origin_date,horizon,count(*) n FROM e GROUP BY ALL HAVING n>1)')
check('training_month_not_future',"SELECT count(*) FROM e WHERE cast(last_day(cast(training_end_month||'-01' AS DATE)) AS DATE)>origin_date")
check('primary_history_gate','SELECT count(*) FROM e WHERE eligible_36in60 AND (n_past60<36 OR n_past_last12<9 OR NOT future_complete)')
check('sensitivity_history_gate','SELECT count(*) FROM e WHERE eligible_24in36 AND (n_past36<24 OR n_past_last12<9 OR NOT future_complete)')
sample=c.execute('SELECT * FROM r WHERE return_verified ORDER BY hash(fund_id,month) LIMIT 30').fetchdf();provenance=[]
for x in sample.itertuples(index=False):
 raw=pathlib.Path(x.raw_file);assert sha(raw)==x.raw_sha256
 v=json.loads(raw.read_text(encoding='utf-8'))[int(x.series_index)]['data'][int(x.point_index)];assert abs(v[1]-x.unit_nav)<1e-10
 day=pd.Timestamp(v[0],unit='ms',tz='UTC').tz_convert('Europe/Moscow').tz_localize(None).normalize();assert day==pd.Timestamp(x.value_date)
 provenance.append(dict(fund_id=x.fund_id,month=x.month,raw_file=str(raw),json_pointer='/'+str(int(x.series_index))+'/data/'+str(int(x.point_index)),unit_nav=x.unit_nav,date=str(day.date()),status='PASS'))
checks.append(dict(check='30_source_point_date_value_hash_replications',violations=0,status='PASS'))
# Currency confirmation uses observed total NAV only for cross-source semantics, never to calculate a fund return.
c.execute("ATTACH './local/research-data-service/research.duckdb' AS source (READ_ONLY)")
c.execute('''CREATE TEMP TABLE nav_checks AS SELECT f.fund_id,cast(f.report_date AS DATE) report_date,try_cast(f.net_asset_value_rub AS DOUBLE) disclosure_NAV_RUB,h.value_source chart_NAV,
 abs(try_cast(f.net_asset_value_rub AS DOUBLE)-h.value_source)/greatest(abs(try_cast(f.net_asset_value_rub AS DOUBLE)),1) relative_difference
 FROM source.funds.reports f JOIN (SELECT fund_id,cast(to_timestamp(timestamp_ms_source/1000) AS DATE) d,value_source FROM read_parquet('''+q(R/'existing_public_chart_history.parquet')+''') WHERE data_key='sca') h ON f.fund_id=cast(h.fund_id AS VARCHAR) AND cast(f.report_date AS DATE)=h.d
 WHERE f.fund_id IN (SELECT DISTINCT fund_id FROM r) AND try_cast(f.net_asset_value_rub AS DOUBLE)>0''')
nav=c.execute('SELECT count(*) n_compared,count(*) FILTER(WHERE relative_difference<=1e-6) n_agree,median(relative_difference) median_relative_difference,quantile_cont(relative_difference,.95) p95_relative_difference FROM nav_checks').fetchdf().to_dict('records')[0]
c.execute('COPY (SELECT * FROM nav_checks WHERE relative_difference>1e-6) TO '+q(Q/'chart_NAV_disclosure_disagreements.parquet')+' (FORMAT PARQUET,COMPRESSION ZSTD)')
# Large observed moves are inspection candidates, not silently trimmed investment performance.
c.execute('COPY (SELECT fund_id,fund_name,registration_number,month,unit_nav,previous_unit_nav,fund_return_observed,value_date,previous_value_date,return_verified,raw_file,raw_sha256 FROM r WHERE isfinite(fund_return_observed) AND (unit_nav/previous_unit_nav<0.1 OR unit_nav/previous_unit_nav>10)) TO '+q(Q/'possible_unit_scale_breaks.parquet')+' (FORMAT PARQUET,COMPRESSION ZSTD)')
moves=c.execute('SELECT fund_id,fund_name,month,fund_return_observed,return_verified FROM r WHERE isfinite(fund_return_observed) AND (unit_nav/previous_unit_nav<0.1 OR unit_nav/previous_unit_nav>10) ORDER BY fund_id,month').fetchdf().to_dict('records')
result=dict(status='PASS',observed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),checks=checks,source_point_replications=provenance,NAV_currency_semantics_check=nav,possible_unit_scale_breaks=moves,release_seal_sha256=sha(D/'sealed_outputs.json'),calendar_source=dict(path='./analysis_outputs/h1_macro_no_time_20261031/inputs/raw_external_2026-09-30/moex/indices/IMOEX_daily_history_2012-01-01_2026-08-31.csv'),basis_limitation='Unknown distributions and split completeness remain unknown; none of this release is asserted TR_VERIFIED')
(Q/'v1_independent_qa.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False,default=str))
