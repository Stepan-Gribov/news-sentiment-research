"""Raw canonical-event NetSentiment; no regressions and no writes to source resources."""
import sys,os,pathlib,json,hashlib,datetime as dt,time,sqlite3,traceback
import duckdb,numpy as np,pandas as pd
R=pathlib.Path('./analysis_outputs/h1_net_fund_fe_20261030')
B=pathlib.Path('./analysis_outputs/title_name_regressions_20261026_w2_v2')
P=pathlib.Path('./analysis_outputs/portfolio_units_20261030')
L=pathlib.Path('./news_data/llm_runs/gemma_title_name_no_gaz_20261008_01')
sys.path.insert(0,str(B/'code'))
from title_name_regressions_core import aggregate_news_windows
START=time.monotonic()
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def save(n,v):
 p=R/n;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def stage(s,**kw):
 v=dict(stage=s,pid=os.getpid(),at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),elapsed_seconds=round(time.monotonic()-START,2),**kw);save('checkpoint.json',v);print(json.dumps(v,default=str),flush=True)
def qs(p):return "'"+str(p).replace('\\','/').replace("'","''")+"'"
def live_digest():
 h=hashlib.sha256();n=0
 with sqlite3.connect(L.joinpath('results.sqlite3').as_uri()+'?mode=ro',uri=True,timeout=30) as db:
  db.execute('PRAGMA query_only=ON');db.execute('BEGIN')
  for row in db.execute('SELECT pair_id,task_index,valid,result_json FROM results ORDER BY task_index'):
   h.update(json.dumps(list(row),ensure_ascii=False,separators=(',',':')).encode()+b'\n');n+=1
  db.rollback()
 return dict(logical_results_sha256=h.hexdigest(),row_count=n,read_only=True,wal_aware=True)
def main():
 R.mkdir(exist_ok=True,parents=True)
 if (R/'sealed_outputs.json').exists():raise RuntimeError('Completed output is immutable')
 with (R/'owner.lock').open('x') as f:f.write(str(os.getpid()))
 save('ready_manifest.json',dict(ready=False,stage='VERIFYING_SOURCE_IDENTITIES'))
 stage('VERIFYING_SOURCE_IDENTITIES')
 prep=json.loads((B/'preparation.json').read_text(encoding='utf-8'));prov=prep['preparation_provenance'];binding=prov['identity']['llm_binding']
 paths={n:B/n for n in ['event_representatives.parquet','news_candidates.parquet','event_cluster_audit.parquet','all_true_intervals.parquet']}
 source_hashes={}
 for n,p in paths.items():
  source_hashes[str(p)]=sha(p);assert source_hashes[str(p)]==prov['derived_output_hashes'][n],n
 for n in ['title_name_regressions_core.py','title_name_regressions_news.py','title_name_regressions_integrity.py']:
  source_hashes[str(B/'code'/n)]=sha(B/'code'/n);assert source_hashes[str(B/'code'/n)]==prov['identity']['preparation_code_hashes'][n]
 for n,key in [('input_manifest.json','input_manifest_sha256'),('run_manifest.json','run_manifest_sha256'),('input.jsonl','input_sha256')]:
  source_hashes[str(L/n)]=sha(L/n);assert source_hashes[str(L/n)]==binding[key],n
 seal=json.loads((P/'sealed_outputs.json').read_text(encoding='utf-8'));source_hashes[str(P/'portfolio_units.parquet')]=sha(P/'portfolio_units.parquet');assert source_hashes[str(P/'portfolio_units.parquet')]==seal['files']['portfolio_units.parquet']['sha256']
 live=live_digest();assert live['logical_results_sha256']==binding['logical_results_sha256'];assert live['row_count']==prep['pair_count']
 assert sha(R/'plan.json')=='2e685b1ea18c37c649ae1845c9c9a02103d09b45ef2840596141884cdc4ed575'
 save('input_manifest.json',dict(plan_sha256=sha(R/'plan.json'),source_hashes=source_hashes,binding=binding,live_sqlite=live,script_sha256=sha(__file__),reuse_canonical_events=True,no_new_llm=True,no_regressions=True,model_weights_content_hash_verified=False))
 stage('BUILDING_FULL_ACTUAL_INTERVAL_UNIVERSE',live_rows=live['row_count'])
 c=duckdb.connect(str(R/'derived.duckdb'),config={'threads':2,'memory_limit':'2GB','temp_directory':str(R/'spill')})
 def run(s):return c.execute(s)
 def scalar(s):return run(s).fetchone()[0]
 def export(table,file=None):run('COPY (SELECT * FROM '+table+') TO '+qs(R/(file or table+'.parquet'))+' (FORMAT PARQUET,COMPRESSION ZSTD)')
 run('CREATE TEMP VIEW ledger AS FROM read_parquet('+qs(P/'portfolio_units.parquet')+')')
 run('CREATE TEMP VIEW intervals AS FROM read_parquet('+qs(paths['all_true_intervals.parquet'])+')')
 run('CREATE TEMP TABLE companies AS SELECT DISTINCT company_id FROM ledger WHERE company_id IS NOT NULL')
 run('CREATE TEMP TABLE report_companies AS SELECT DISTINCT report_id,company_id FROM ledger WHERE company_id IS NOT NULL AND is_reported_position')
 run('''CREATE TEMP TABLE actual_bridge AS SELECT i.interval_id,i.fund_id,i.opening_report_id,i.closing_report_id,cast(i.start_date AS DATE) news_start,cast(i.end_date AS DATE) news_end,r.company_id,
 bool_or(r.report_id=i.opening_report_id) held_at_opening,bool_or(r.report_id=i.closing_report_id) held_at_closing
 FROM (SELECT *,opening_report_id endpoint_report_id FROM intervals UNION ALL SELECT *,closing_report_id endpoint_report_id FROM intervals) i JOIN report_companies r ON r.report_id=i.endpoint_report_id GROUP BY ALL''')
 run("CREATE TEMP TABLE actual_windows AS SELECT DISTINCT 'actual_report_interval' period_type,company_id,news_start,news_end,cast(NULL AS VARCHAR) AS month FROM actual_bridge")
 earliest,latest=run('SELECT min(cast(report_date AS DATE)),max(cast(report_date AS DATE)) FROM ledger').fetchone()
 months=pd.period_range(earliest,latest,freq='M')
 month_frame=pd.DataFrame(dict(month=months.astype(str),news_start=(months-1).to_timestamp(how='end').normalize(),news_end=months.to_timestamp(how='end').normalize()))
 c.register('month_frame',month_frame)
 run("CREATE TEMP TABLE calendar_windows AS SELECT 'calendar_month' period_type,company_id,cast(news_start AS DATE) news_start,cast(news_end AS DATE) news_end,month FROM companies CROSS JOIN month_frame")
 run("CREATE TEMP TABLE windows AS SELECT sha256(period_type||chr(9)||company_id||chr(9)||cast(news_start AS VARCHAR)||chr(9)||cast(news_end AS VARCHAR)) company_period_id,* FROM (FROM actual_windows UNION ALL FROM calendar_windows)")
 run("CREATE TEMP TABLE fund_interval_company_windows AS SELECT b.*,w.company_period_id FROM actual_bridge b JOIN windows w ON w.period_type='actual_report_interval' AND w.company_id=b.company_id AND w.news_start=b.news_start AND w.news_end=b.news_end")
 run('CREATE TEMP TABLE fund_interval_coverage AS SELECT i.interval_id,i.fund_id,i.start_date,i.end_date,count(b.company_id) company_count FROM intervals i LEFT JOIN actual_bridge b USING(interval_id) GROUP BY ALL')
 export('fund_interval_company_windows');export('fund_interval_coverage');export('companies','company_universe.parquet')
 targets=run('SELECT *,company_period_id interval_id FROM windows ORDER BY period_type,company_id,news_start,news_end').fetchdf()
 assert len(targets)==targets.company_period_id.nunique();assert scalar('SELECT count(*) FROM fund_interval_coverage')==42570
 stage('AGGREGATING_RAW_NET',companies=scalar('SELECT count(*) FROM companies'),windows=len(targets),actual_windows=scalar('SELECT count(*) FROM actual_windows'),fund_company_intervals=scalar('SELECT count(*) FROM actual_bridge'))
 events=pd.read_parquet(paths['event_representatives.parquet']);assert events.is_representative.eq(True).all();assert not events.date_inferred.eq(True).any()
 company_set=set(run('SELECT company_id FROM companies').fetchnumpy()['company_id']);events=events.loc[events.company_id.isin(company_set)].copy()
 dated=pd.to_datetime(events.event_timestamp,utc=True).dt.tz_convert('Europe/Moscow').dt.tz_localize(None).dt.normalize()
 assert (dated.eq(pd.to_datetime(events.event_date)) | (dated.isna() & events.event_date.isna())).all()
 net,audit=aggregate_news_windows(targets,events)
 net=net.drop(columns=['interval_id']).rename(columns={'Tone':'net_sentiment','tone_observed':'net_sentiment_observed','valid_fraction':'classification_coverage','news_status':'state','n_expected':'n_candidates'})
 net['period_kind']=net.period_type;net['period_start']=net.news_start;net['period_end']=net.news_end
 net['log_news_count_interval']=np.log1p(net.n_relevant).where(net.n_candidates.gt(0)&net.n_valid.eq(net.n_candidates))
 net['n_irrelevant']=net.n_valid-net.n_relevant
 net['n_error']=net.n_attempted-net.n_valid;net['n_unattempted']=net.n_candidates-net.n_attempted
 net['strict_complete']=net.n_candidates.gt(0)&net.n_valid.eq(net.n_candidates)
 net['eligible_primary']=net.strict_complete&net.n_relevant.gt(0)
 net['eligible_partial95_secondary']=net.n_candidates.gt(0)&net.classification_coverage.ge(.95)&net.n_relevant.gt(0)
 net['net_sentiment_partial95_secondary']=net.net_sentiment_observed.where(net.eligible_partial95_secondary)
 net['n_positive']=net.n_positive.astype('int64');net['n_negative']=net.n_negative.astype('int64');net['n_neutral']=net.n_neutral.astype('int64')
 missing=events.loc[events.event_date.isna()].groupby('company_id').size()
 net['n_undated_company_candidates']=net.company_id.map(missing).fillna(0).astype('int64')
 net['source_id']='gemma_title_name_no_gaz_20261008_01';net['event_representatives_sha256']=source_hashes[str(paths['event_representatives.parquet'])]
 # Source-specific counts and membership hashes are computed from exactly the same canonical rows.
 c.register('net_frame',net);run('CREATE TEMP TABLE preliminary_net AS FROM net_frame');c.unregister('net_frame')
 c.register('event_frame',events[['pair_id','article_id','company_id','cluster_id','event_date','source','attempted','valid','relevance','sentiment_label','pair_state','text_sha256']]);run('CREATE TEMP TABLE canonical_events AS FROM event_frame');c.unregister('event_frame')
 stage('EXPORTING_COMPANY_PERIOD_MEMBERS')
 run('''CREATE TEMP TABLE companyperiod_news_members AS SELECT w.company_period_id,w.period_type,w.company_id,w.news_start,w.news_end,e.article_id,e.pair_id,e.cluster_id event_id,cast(e.event_date AS DATE) event_date,e.source,e.attempted,e.valid,e.relevance,e.sentiment_label,e.pair_state,e.text_sha256,
 (e.valid AND e.relevance='relevant') included_in_net_denominator FROM windows w JOIN canonical_events e ON w.company_id=e.company_id AND e.event_date>w.news_start AND e.event_date<=w.news_end''')
 export('companyperiod_news_members')
 run('''CREATE TEMP TABLE member_digest AS SELECT company_period_id,count(*) n_members,sha256(string_agg(pair_id,chr(10) ORDER BY pair_id)) sample_hash,
 count(*) FILTER(WHERE included_in_net_denominator AND source='interfax') n_relevant_interfax,
 count(*) FILTER(WHERE included_in_net_denominator AND source='rbc') n_relevant_rbc,
 count(*) FILTER(WHERE included_in_net_denominator AND source='prime') n_relevant_prime,
 count(*) FILTER(WHERE included_in_net_denominator AND source='kommersant') n_relevant_kommersant FROM companyperiod_news_members GROUP BY company_period_id''')
 run('CREATE TEMP TABLE net_sentiment_company_period AS SELECT p.*,coalesce(d.n_members,0) n_members,d.sample_hash,coalesce(n_relevant_interfax,0) n_relevant_interfax,coalesce(n_relevant_rbc,0) n_relevant_rbc,coalesce(n_relevant_prime,0) n_relevant_prime,coalesce(n_relevant_kommersant,0) n_relevant_kommersant FROM preliminary_net p LEFT JOIN member_digest d USING(company_period_id)')
 stage('CHECKING_NET_DATASET')
 checks=[]
 def check(name,sql):
  n=int(scalar(sql));checks.append(dict(check=name,violations=n,status='PASS' if n==0 else 'FAIL'));assert n==0,(name,n)
 check('unique_company_period_id','SELECT count(*) FROM (SELECT company_period_id,count(*) n FROM net_sentiment_company_period GROUP BY 1 HAVING n>1)')
 check('unique_company_window_type','SELECT count(*) FROM (SELECT period_type,company_id,news_start,news_end,count(*) n FROM net_sentiment_company_period GROUP BY ALL HAVING n>1)')
 check('counts_partition','SELECT count(*) FROM net_sentiment_company_period WHERE n_positive+n_negative+n_neutral<>n_relevant OR n_relevant+n_irrelevant<>n_valid OR n_valid+n_error+n_unattempted<>n_candidates')
 check('canonical_member_count_matches','SELECT count(*) FROM net_sentiment_company_period WHERE n_candidates<>n_members')
 check('strict_NA_for_absence_incomplete','SELECT count(*) FROM net_sentiment_company_period WHERE net_sentiment IS NOT NULL AND (n_relevant=0 OR n_valid<>n_candidates OR n_candidates=0)')
 check('strict_formula','SELECT count(*) FROM net_sentiment_company_period WHERE net_sentiment IS NOT NULL AND abs(net_sentiment-(n_positive-n_negative)::DOUBLE/n_relevant)>1e-12')
 check('observed_zero_is_real_balance','SELECT count(*) FROM net_sentiment_company_period WHERE net_sentiment=0 AND (n_relevant=0 OR n_positive<>n_negative)')
 check('bounded_net','SELECT count(*) FROM net_sentiment_company_period WHERE (net_sentiment IS NOT NULL AND (NOT isfinite(net_sentiment) OR abs(net_sentiment)>1)) OR (net_sentiment_partial95_secondary IS NOT NULL AND (NOT isfinite(net_sentiment_partial95_secondary) OR abs(net_sentiment_partial95_secondary)>1))')
 check('member_window_boundaries','SELECT count(*) FROM companyperiod_news_members WHERE event_date<=news_start OR event_date>news_end')
 check('unique_window_event','SELECT count(*) FROM (SELECT company_period_id,pair_id,count(*) n FROM companyperiod_news_members GROUP BY ALL HAVING n>1)')
 check('fund_company_bridge_complete','SELECT abs((SELECT count(*) FROM actual_bridge)-(SELECT count(*) FROM fund_interval_company_windows))')
 check('company_period_identical_across_funds','SELECT count(*) FROM (SELECT company_id,news_start,news_end,count(distinct net_sentiment) n FROM net_sentiment_company_period GROUP BY ALL HAVING n>1)')
 check('calendar_covers_all_portfolio_companies','SELECT abs((SELECT count(*) FROM calendar_windows)-(SELECT count(*) FROM companies)*'+str(len(months))+')')
 check('all_actual_intervals_retained','SELECT abs(count(*)-42570) FROM fund_interval_coverage')
 assert live_digest()==live
 checks.append(dict(check='live_SQLite_logical_hash_unchanged',status='PASS',violations=0))
 assert all(sha(path)==value for path,value in source_hashes.items())
 checks.append(dict(check='all_input_hashes_unchanged',status='PASS',violations=0))
 export('net_sentiment_company_period')
 run('COPY (SELECT * FROM net_sentiment_company_period ORDER BY period_type,company_id,news_start,news_end) TO '+qs(R/'net_sentiment_company_period.csv')+" (FORMAT CSV,HEADER,DATEFORMAT '%Y-%m-%d',TIMESTAMPFORMAT '%Y-%m-%d',NULL '')")
 schema=[dict(zip(['column_name','column_type','null','key','default','extra'],row)) for row in run('DESCRIBE net_sentiment_company_period').fetchall()];save('schema.json',dict(table='net_sentiment_company_period',primary_key=['company_period_id'],unique_key=['period_type','company_id','news_start','news_end'],columns=schema))
 sample=run('''SELECT * EXCLUDE(rn) FROM (SELECT *,row_number() OVER(PARTITION BY period_type,state ORDER BY n_relevant DESC,company_id,news_end) rn FROM net_sentiment_company_period) ORDER BY rn,period_type,state,company_id LIMIT 20''').fetchdf()
 save('preview20.json',json.loads(sample.to_json(orient='records',date_format='iso')));sample.to_csv(R/'preview20.csv',index=False,encoding='utf-8-sig')
 run('CREATE TEMP TABLE missing_date_audit AS SELECT company_id,count(*) n_canonical_candidates,count(*) FILTER(WHERE event_date IS NULL) n_missing_event_date FROM canonical_events GROUP BY company_id');export('missing_date_audit')
 counts=dict(companies=scalar('SELECT count(*) FROM companies'),all_actual_fund_intervals=42570,fund_company_intervals=scalar('SELECT count(*) FROM actual_bridge'),fund_intervals_without_company=scalar('SELECT count(*) FROM fund_interval_coverage WHERE company_count=0'),actual_company_windows=scalar('SELECT count(*) FROM actual_windows'),calendar_company_months=scalar('SELECT count(*) FROM calendar_windows'),company_period_rows=len(net),membership_rows=scalar('SELECT count(*) FROM companyperiod_news_members'),primary_defined=scalar('SELECT count(*) FROM net_sentiment_company_period WHERE net_sentiment IS NOT NULL'),partial95_secondary_defined=scalar('SELECT count(*) FROM net_sentiment_company_period WHERE net_sentiment_partial95_secondary IS NOT NULL'),strict_true_zero=scalar('SELECT count(*) FROM net_sentiment_company_period WHERE net_sentiment=0'),state_counts=run('SELECT period_type,state,count(*) n FROM net_sentiment_company_period GROUP BY ALL ORDER BY 1,2').fetchdf().to_dict('records'))
 save('counts.json',counts)
 save('qa.json',dict(passed=True,checks=checks,missing_date_audit=audit))
 save('metadata.json',dict(formula='(Npositive-Nnegative)/(Npositive+Nneutral+Nnegative)',weighting='Equal canonical relevant event representatives, no new collapse',primary='strict_complete only; net NULL for no candidates, no relevant, incomplete classification',secondary='net_sentiment_partial95_secondary only; not primary',event_window='(news_start,news_end]',timezone='Europe/Moscow',company_universe='All non-null company_id in sealed portfolio_units, 709 companies',actual_universe='Opening and closing reported-company union for every all_true_intervals row, no eligibility/EOM/regression gates',calendar_scope=dict(first_month=str(months[0]),last_month=str(months[-1]),full_calendar_months=True,last_month_extends_to_calendar_end=True),observed_report_date_min=str(earliest),observed_report_date_max=str(latest),undated_policy='Cannot assign to intervals; company-level counts and separate audit, never imputed or counted as zero news evidence',candidate_counts='Canonical expected representatives; n_candidates=n_members; includes relevant, irrelevant, invalid/unattempted',sample_hash='SHA256 of canonical pair_ids sorted and separated by LF; NULL when no candidates',members='All canonical candidate representatives, not only denominator members; included_in_net_denominator flag',source_binding=binding,no_regressions=True,sqlite_import_owner='DB assistant; this job never writes live SQLite'))
 save('ready_manifest.json',dict(ready=True,output_root=str(R),table_name='net_sentiment_company_period',primary_signal='net_sentiment',primary_filter='eligible_primary = TRUE',secondary_signal='net_sentiment_partial95_secondary',period_types=['actual_report_interval','calendar_month'],join_actual=['fund_interval_company_windows.company_period_id','net_sentiment_company_period.company_period_id'],data_file=str(R/'net_sentiment_company_period.parquet'),sqlite_import_csv=str(R/'net_sentiment_company_period.csv'),members_file=str(R/'companyperiod_news_members.parquet'),schema_file=str(R/'schema.json'),preview_file=str(R/'preview20.json'),qa_file=str(R/'qa.json'),counts=counts))
 stage('READY_FOR_DB_AND_CODE',**{k:v for k,v in counts.items() if not isinstance(v,list)})
 files=[p for p in R.iterdir() if p.is_file() and p.suffix in ['.json','.csv','.parquet'] and p.name not in ['checkpoint.json','sealed_outputs.json']]
 save('sealed_outputs.json',dict(sealed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files},script_sha256=sha(__file__),source_hashes=source_hashes))
 stage('COMPLETED',ready=True,qa_checks=len(checks),rows=len(net),companies=counts['companies']);c.close()
if __name__=='__main__':
 R.mkdir(exist_ok=True,parents=True)
 sys.stdout=(R/'run.stdout.log').open('a',encoding='utf-8',buffering=1);sys.stderr=(R/'run.stderr.log').open('a',encoding='utf-8',buffering=1)
 try:main()
 except Exception as e:
  R.mkdir(exist_ok=True,parents=True);save('failure.json',dict(error=repr(e),traceback=traceback.format_exc()));stage('FAILED',error=repr(e));raise
