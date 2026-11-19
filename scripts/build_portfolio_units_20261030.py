import pathlib,json,hashlib,datetime as dt,os,time,gzip,math,traceback,sys,collections
import duckdb
import pyarrow as pa

R=pathlib.Path('./analysis_outputs/portfolio_units_20261030')
B=pathlib.Path('./analysis_outputs/title_name_regressions_20261026_w2_v2')
CAT='./local/research-data-service/research.duckdb'
R.mkdir(parents=True,exist_ok=True)
START=time.monotonic()
HEADERS=['Фонд','Дата отчета','Актив','Расчетное количество','Количество в прошлом отчете','Изменение количества','Относительное изменение количества','Доля в портфеле','Доля в прошлом отчете','Изменение доли, п.п.','Относительное изменение доли','Стоимость позиции, руб.','Цена одной бумаги, руб.','Дата котировки','Дата предыдущего отчета','Статус количества','Тип актива','ID фонда','ISIN / ID актива','Статус сверки отчета']
SH=['Фонд','ID фонда','Дата отчета','Позиций в отчете','Позиций с оценкой количества','Бумаг без оценки','Выбывших позиций','Стоимость всех позиций, руб.','Доля стоимости бумаг с ценой','Статус сверки отчета']
def save(name,data):
 p=R/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def stage(name,**data):
 v=dict(stage=name,pid=os.getpid(),at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),elapsed_seconds=round(time.monotonic()-START,2),**data);save('checkpoint.json',v);print(json.dumps(v,ensure_ascii=True,default=str),flush=True)
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def sqlstr(x):return "'"+str(x).replace('\\','/').replace("'","''")+"'"
def clean(v):
 if isinstance(v,(dt.date,dt.datetime)):return v.date().isoformat() if isinstance(v,dt.datetime) else v.isoformat()
 if isinstance(v,float) and not math.isfinite(v):return None
 return v

def main():
 if (R/'excel_manifest.json').exists() and json.loads((R/'excel_manifest.json').read_text(encoding='utf-8')).get('ready'):raise RuntimeError('Completed dataset immutable')
 plan=json.loads((R/'plan.json').read_text(encoding='utf-8'));assert sha(R/'plan.json')=='fd2b016e46d62070b130ff5a7ca554d796d7df862fea6e4f5f4c28e33c03e1d2'
 save('excel_manifest.json',dict(ready=False,stage='STARTING'))
 paths=[B/'holdings_inventory.parquet',B/'all_true_intervals.parquet',B/'scheduled_corporate_events.parquet',pathlib.Path('./databaseresources/funds/mac-20261008/fund_holdings_long.csv.gz'),pathlib.Path('./databaseresources/market/moex_quotes_2008_2026/mapping.csv'),pathlib.Path('./databaseresources/funds/mac-20261008/holdings_security_name_isin_mapping.csv')]
 hashes={str(p):sha(p) for p in paths};save('input_manifest.json',dict(plan_sha256=sha(R/'plan.json'),source_hashes=hashes,script_sha256=sha(__file__),sources_read_only=True))
 stage('READING_ALL_REPORT_POSITIONS')
 (R/'spill').mkdir(exist_ok=True)
 c=duckdb.connect(str(R/'derived.duckdb'),config={'threads':2,'memory_limit':'2GB','temp_directory':str(R/'spill')})
 c.execute('ATTACH '+sqlstr(CAT)+' AS upstream (READ_ONLY)')
 def run(s):return c.execute(s)
 def scalar(s):return run(s).fetchone()[0]
 def export(table,name):run('COPY (SELECT * FROM '+table+') TO '+sqlstr(R/name)+" (FORMAT PARQUET, COMPRESSION ZSTD)")
 run('CREATE TEMP TABLE inv AS SELECT row_number() OVER ()::BIGINT source_row_id,* FROM read_parquet('+sqlstr(paths[0])+')')
 run('CREATE TEMP TABLE intervals AS SELECT * FROM read_parquet('+sqlstr(paths[1])+')')
 assert scalar('SELECT count(*) FROM inv')==1195974
 assert scalar('SELECT count(*) FROM (SELECT report_id,security_id,count(*) n FROM inv GROUP BY ALL HAVING n>1)')==0
 assert scalar('SELECT count(*) FROM (SELECT closing_report_id,count(*) n FROM intervals GROUP BY ALL HAVING n>1)')==0
 run("CREATE TEMP TABLE reports AS SELECT report_id,fund_id,coalesce(nullif(trim(fund_name_file),''),nullif(trim(fund_name_xls),''),fund_id) fund_name,source_path FROM upstream.funds.reports")
 assert scalar('SELECT count(*) FROM reports')==scalar('SELECT count(distinct report_id) FROM reports')
 run('''CREATE TEMP TABLE report_state AS SELECT report_id,bool_and(full_flag) full_report FROM
 (SELECT opening_report_id report_id,opening_full_portfolio full_flag FROM intervals UNION ALL SELECT closing_report_id,closing_full_portfolio FROM intervals) GROUP BY report_id''')
 run("CREATE TEMP TABLE mapping AS SELECT upper(trim(requested_isin)) isin, CASE WHEN count(distinct nullif(trim(primary_boardid),''))=1 THEN max(nullif(trim(primary_boardid),'')) END primary_board, string_agg(distinct coalesce(nullif(shortname,''),nullif(name,'')),' / ' ORDER BY coalesce(nullif(shortname,''),nullif(name,''))) ref_name FROM upstream.refs.moex_mapping WHERE status='resolved' GROUP BY 1")
 name_columns={x[0].lower() for x in run('DESCRIBE upstream.refs.security_name_mapping').fetchall()}
 if {'isin','news_search_name'}<=name_columns:
  run("CREATE TEMP TABLE security_names AS SELECT upper(trim(isin)) isin,string_agg(distinct nullif(trim(news_search_name),''),' / ' ORDER BY nullif(trim(news_search_name),'')) fallback_name FROM upstream.refs.security_name_mapping GROUP BY 1")
 else:run('CREATE TEMP TABLE security_names(isin VARCHAR,fallback_name VARCHAR)')
 # Preserve all reported rows; separately mark inferred complete-report exits.
 cols=[x[0] for x in run('DESCRIBE inv').fetchall()]
 override={'source_row_id':'NULL::BIGINT','report_id':'t.closing_report_id','amount_thousand_rub':'0::DOUBLE','missing_amount':'false','source_lots':'NULL::DOUBLE','report_date':'t.end_date','gross_thousand':'g.gross_thousand','weight':'0::DOUBLE'}
 select=','.join(override.get(k,'p."'+k+'"')+' AS "'+k+'"' for k in cols)
 run('''CREATE TEMP TABLE raw_ledger AS SELECT *,true is_reported_position,false is_exit FROM inv UNION ALL BY NAME SELECT '''+select+''',false is_reported_position,true is_exit FROM intervals t JOIN inv p ON p.report_id=t.opening_report_id LEFT JOIN (SELECT report_id,max(gross_thousand) gross_thousand FROM inv GROUP BY report_id) g ON g.report_id=t.closing_report_id WHERE t.closing_full_portfolio=true AND NOT EXISTS(SELECT 1 FROM inv n WHERE n.report_id=t.closing_report_id AND n.security_id=p.security_id)''')
 run("CREATE TEMP TABLE classified AS SELECT *, CASE WHEN security_id LIKE 'isin:%' THEN upper(substr(security_id,6)) WHEN regexp_full_match(upper(coalesce(isin,'')),'[A-Z]{2}[A-Z0-9]{9}[0-9]') THEN upper(isin) END price_isin, (security_id LIKE 'isin:%' OR asset_class IN ('equity','bond')) AND security_id NOT LIKE 'documented_bucket:%' AND security_id<>'__UNALLOCATED__' is_countable, CASE WHEN NOT missing_amount AND isfinite(amount_thousand_rub) AND amount_thousand_rub>=0 AND isfinite(gross_thousand) AND gross_thousand>0 THEN amount_thousand_rub/gross_thousand END usable_weight, CASE WHEN isfinite(amount_thousand_rub) THEN amount_thousand_rub*1000 END position_value_rub FROM raw_ledger")
 run('CREATE TEMP TABLE requests AS SELECT DISTINCT price_isin isin,cast(report_date AS DATE) report_date FROM classified WHERE is_countable AND price_isin IS NOT NULL AND position_value_rub>0 AND NOT missing_amount AND report_date IS NOT NULL')
 stage('SELECTING_CANONICAL_ENDPOINT_PRICES',requests=scalar('SELECT count(*) FROM requests'))
 run('''CREATE TEMP TABLE qraw AS SELECT upper(trim(q.REQUESTED_ISIN)) isin,q.SECID secid,q.BOARDID boardid,try_cast(q.TRADEDATE AS DATE) quote_date,q.CLOSE raw_close,q.VALUE trading_value,upper(q.CURRENCYID) currency,q.FACEVALUE facevalue,upper(q.FACEUNIT) faceunit,q.SOURCE_MARKET source_market,q.SHORTNAME shortname,q.ACCINT accint,q.source_file,
 CASE WHEN q.SOURCE_MARKET='bonds' AND q.FACEVALUE>0 AND isfinite(q.FACEVALUE) AND upper(q.FACEUNIT) IN ('RUB','RUR','SUR') THEN q.CLOSE*q.FACEVALUE/100 WHEN q.SOURCE_MARKET<>'bonds' THEN q.CLOSE END unit_price_rub
 FROM upstream.market.security_quotes q JOIN (SELECT DISTINCT isin FROM requests) r ON upper(trim(q.REQUESTED_ISIN))=r.isin
 WHERE q.CLOSE>0 AND isfinite(q.CLOSE) AND upper(q.CURRENCYID) IN ('RUB','RUR','SUR') AND q.SECID IS NOT NULL AND q.BOARDID IS NOT NULL AND try_cast(q.TRADEDATE AS DATE) IS NOT NULL''')
 conflicts=scalar('SELECT count(*) FROM (SELECT isin,secid,boardid,quote_date FROM qraw WHERE unit_price_rub>0 AND isfinite(unit_price_rub) GROUP BY ALL HAVING count(distinct unit_price_rub)>1)')
 run('''CREATE TEMP TABLE qclean AS SELECT isin,secid,boardid,quote_date,max(raw_close) raw_close,max(trading_value) trading_value,max(currency) currency,max(facevalue) facevalue,max(faceunit) faceunit,max(source_market) source_market,max(shortname) shortname,max(accint) accint,min(source_file) source_file,max(unit_price_rub) unit_price_rub FROM qraw WHERE unit_price_rub>0 AND isfinite(unit_price_rub) GROUP BY isin,secid,boardid,quote_date HAVING count(distinct unit_price_rub)=1''')
 run('CREATE TEMP TABLE request_boards AS SELECT r.*,b.secid,b.boardid FROM requests r JOIN (SELECT DISTINCT isin,secid,boardid FROM qclean) b USING(isin)')
 run('''CREATE TEMP TABLE candidate_prices AS SELECT r.isin,r.report_date,q.* EXCLUDE(isin,secid,boardid),r.secid,r.boardid FROM request_boards r ASOF LEFT JOIN qclean q ON r.isin=q.isin AND r.secid=q.secid AND r.boardid=q.boardid AND r.report_date>=q.quote_date''')
 run('''CREATE TEMP TABLE endpoint_prices AS SELECT p.*,date_diff('day',quote_date,report_date) quote_age_days,coalesce(p.boardid=m.primary_board,false) primary_board_preferred FROM candidate_prices p LEFT JOIN mapping m USING(isin) WHERE quote_date IS NOT NULL AND date_diff('day',quote_date,report_date) BETWEEN 0 AND 7 QUALIFY row_number() OVER(PARTITION BY p.isin,p.report_date ORDER BY coalesce(p.boardid=m.primary_board,false) DESC,quote_date DESC,trading_value DESC NULLS LAST,p.secid,p.boardid)=1''')
 export('endpoint_prices','endpoint_prices.parquet')
 # Names are labels only, never new identity links.
 stage('ADDING_HISTORICAL_NAMES')
 raw_names=run('SELECT DISTINCT registration_number_security,parent_category,parent_subcategory,item_name FROM upstream.funds.holdings').fetchall();names=collections.defaultdict(set)
 for code,cat,sub,name in raw_names:
  code=str(code or '').strip().upper();cat=str(cat or '');sub=str(sub or '');name=str(name or '')
  if code:names['reg:'+code].add(name);names['isin:'+code].add(name)
  else:names['unidentified:'+hashlib.blake2b((cat+'\t'+sub+'\t'+name).encode(),digest_size=12).hexdigest()].add(name)
  names['documented_bucket:'+hashlib.blake2b((cat+'\t'+sub).encode(),digest_size=12).hexdigest()].add(' / '.join(x for x in [cat,sub] if x))
 for cat,sub in run('SELECT DISTINCT parent_category,parent_subcategory FROM upstream.funds.report_lines').fetchall():
  cat=str(cat or '');sub=str(sub or '');label=' / '.join(x for x in [cat,sub] if x)
  if label:names['documented_bucket:'+hashlib.blake2b((cat+'\t'+sub).encode(),digest_size=12).hexdigest()].add(label)
 names['__UNALLOCATED__'].add('Нерасшифрованный остаток')
 nr=[{'security_id':k,'raw_name':' / '.join(sorted(v))} for k,v in names.items()]
 c.register('name_rows',pa.Table.from_pylist(nr));run('CREATE TEMP TABLE raw_names AS SELECT * FROM name_rows');c.unregister('name_rows');del raw_names,names,nr
 run('''CREATE TEMP TABLE valued AS SELECT a.*,p.unit_price_rub,p.quote_date,p.quote_age_days,p.secid,p.boardid,p.currency,p.facevalue,p.faceunit,p.source_market quote_market,p.raw_close quote_close,p.accint,p.source_file quote_source_file,
 coalesce(nullif(p.shortname,''),nullif(m.ref_name,''),nullif(n.raw_name,''),nullif(sn.fallback_name,''),a.security_id) asset_name,
 CASE WHEN a.is_exit AND a.is_countable THEN 0.0 WHEN a.is_countable AND NOT a.missing_amount AND a.amount_thousand_rub>=0 AND a.position_value_rub IS NOT NULL AND p.unit_price_rub>0 AND isfinite(a.position_value_rub/p.unit_price_rub) THEN a.position_value_rub/p.unit_price_rub END quantity_est,
 CASE WHEN a.is_exit THEN 'Выбыла из портфеля' WHEN a.amount_thousand_rub<0 THEN 'Отрицательный исходник' WHEN a.missing_amount OR a.amount_thousand_rub IS NULL OR NOT isfinite(a.amount_thousand_rub) THEN 'Неполная сумма' WHEN NOT a.is_countable THEN 'Штуки неприменимы' WHEN a.price_isin IS NULL THEN 'Нет ISIN' WHEN p.unit_price_rub IS NULL THEN 'Нет котировки за 7 дней' WHEN p.source_market='bonds' THEN 'Оценка: чистая цена без НКД' ELSE 'Расчетное V/P' END quantity_status
 FROM classified a LEFT JOIN endpoint_prices p ON a.price_isin=p.isin AND cast(a.report_date AS DATE)=p.report_date LEFT JOIN mapping m ON a.price_isin=m.isin LEFT JOIN raw_names n USING(security_id) LEFT JOIN security_names sn ON a.price_isin=sn.isin''')
 stage('COMPARING_PREVIOUS_ACTUAL_REPORTS',endpoint_prices=scalar('SELECT count(*) FROM endpoint_prices'))
 run('''CREATE TEMP TABLE linked AS SELECT v.*,t.opening_report_id previous_report_id,t.start_date previous_report_date,t.opening_full_portfolio,p.report_id previous_position_report_id,
 CASE WHEN p.report_id IS NOT NULL THEN p.quantity_est WHEN t.opening_full_portfolio=true AND v.is_countable THEN 0.0 END previous_quantity_est,
 CASE WHEN p.report_id IS NOT NULL THEN p.usable_weight WHEN t.opening_full_portfolio=true THEN 0.0 END previous_weight,
 s.full_report,r.fund_name,r.source_path
 FROM valued v LEFT JOIN intervals t ON v.report_id=t.closing_report_id LEFT JOIN valued p ON p.report_id=t.opening_report_id AND p.security_id=v.security_id AND p.is_reported_position LEFT JOIN report_state s ON v.report_id=s.report_id LEFT JOIN reports r ON v.report_id=r.report_id''')
 run('''CREATE TEMP TABLE portfolio_units AS SELECT * EXCLUDE(quantity_status),CASE WHEN is_reported_position AND previous_position_report_id IS NULL AND opening_full_portfolio=true AND position_value_rub>0 THEN 'Появилась в раскрытии; '||quantity_status ELSE quantity_status END quantity_status,quantity_est-previous_quantity_est delta_quantity_est,
 CASE WHEN previous_quantity_est>0 AND isfinite(quantity_est/previous_quantity_est-1) THEN quantity_est/previous_quantity_est-1 END relative_quantity_fraction,
 100*(usable_weight-previous_weight) delta_share_pp,CASE WHEN previous_weight>0 AND isfinite(usable_weight/previous_weight-1) THEN usable_weight/previous_weight-1 END relative_weight_fraction,
 CASE WHEN full_report THEN 'Прошел сверку' WHEN full_report=false THEN 'Есть ограничения' ELSE 'Не установлено' END full_report_status,
 CASE WHEN asset_type='equity' THEN 'Акции' WHEN asset_type='bond' THEN 'Облигации' WHEN NOT is_countable THEN 'Денежные средства / прочие активы' ELSE 'Другие инструменты' END asset_type_ru
 FROM linked''')
 export('portfolio_units','portfolio_units.parquet')
 run('''CREATE TEMP TABLE summary_report_periods AS SELECT fund_id,coalesce(fund_name,fund_id) fund_name,cast(report_date AS DATE) report_date,report_id,
 count(*) FILTER(WHERE is_reported_position) reported_positions,count(*) FILTER(WHERE is_reported_position AND is_countable AND quantity_est IS NOT NULL) estimated_positions,
 count(*) FILTER(WHERE is_reported_position AND is_countable AND quantity_est IS NULL) countable_positions_without_estimate,count(*) FILTER(WHERE is_exit) exits,
 sum(position_value_rub) FILTER(WHERE is_reported_position) total_position_value_rub,
 CASE WHEN sum(position_value_rub) FILTER(WHERE is_reported_position AND is_countable)>0 THEN sum(position_value_rub) FILTER(WHERE is_reported_position AND is_countable AND unit_price_rub IS NOT NULL)/sum(position_value_rub) FILTER(WHERE is_reported_position AND is_countable) END countable_value_price_coverage,
 max(full_report_status) full_report_status FROM portfolio_units GROUP BY fund_id,fund_name,report_date,report_id''')
 export('summary_report_periods','summary_report_periods.parquet')
 stage('CHECKING_DATASET')
 qa=[]
 def check(name,s):
  n=int(scalar(s));qa.append(dict(check=name,violations=n,status='PASS' if n==0 else 'FAIL'))
 check('all_original_rows_preserved','SELECT abs((SELECT count(*) FROM portfolio_units WHERE is_reported_position)-(SELECT count(*) FROM inv))')
 check('raw_monetary_values_preserved','SELECT count(*) FROM portfolio_units WHERE is_reported_position AND isfinite(amount_thousand_rub) AND position_value_rub<>amount_thousand_rub*1000')
 check('unique_current_keys','SELECT count(*) FROM (SELECT report_id,security_id,count(*) n FROM portfolio_units GROUP BY ALL HAVING n>1)')
 check('original_key_set_preserved','SELECT count(*) FROM (SELECT report_id,security_id FROM inv EXCEPT SELECT report_id,security_id FROM portfolio_units WHERE is_reported_position)')
 check('no_future_or_stale_quotes',"SELECT count(*) FROM endpoint_prices WHERE quote_date>report_date OR quote_age_days NOT BETWEEN 0 AND 7")
 check('endpoint_key_unique','SELECT count(*) FROM (SELECT isin,report_date,count(*) n FROM endpoint_prices GROUP BY ALL HAVING n>1)')
 check('positive_finite_RUB_prices',"SELECT count(*) FROM endpoint_prices WHERE NOT (unit_price_rub>0 AND isfinite(unit_price_rub)) OR currency NOT IN ('RUB','RUR','SUR') OR (source_market='bonds' AND NOT (facevalue>0 AND isfinite(facevalue) AND faceunit IN ('RUB','RUR','SUR')))")
 check('V_equals_Q_times_P','SELECT count(*) FROM portfolio_units WHERE NOT is_exit AND quantity_est IS NOT NULL AND abs(position_value_rub-quantity_est*unit_price_rub)>1e-7+1e-10*abs(position_value_rub)')
 check('zero_base_relative_is_NA','SELECT count(*) FROM portfolio_units WHERE (previous_quantity_est=0 AND relative_quantity_fraction IS NOT NULL) OR (previous_weight=0 AND relative_weight_fraction IS NOT NULL)')
 check('absence_zero_only_complete','SELECT count(*) FROM portfolio_units WHERE previous_position_report_id IS NULL AND ((previous_quantity_est=0 OR previous_weight=0) AND coalesce(opening_full_portfolio,false)=false)')
 check('first_report_previous_unknown','SELECT count(*) FROM portfolio_units WHERE previous_report_id IS NULL AND (previous_quantity_est IS NOT NULL OR previous_weight IS NOT NULL)')
 check('exits_only_complete_reports','SELECT count(*) FROM portfolio_units WHERE is_exit AND coalesce(full_report,false)=false')
 check('negative_partial_not_estimated','SELECT count(*) FROM portfolio_units WHERE is_reported_position AND (amount_thousand_rub<0 OR missing_amount OR amount_thousand_rub IS NULL) AND (quantity_est IS NOT NULL OR usable_weight IS NOT NULL)')
 check('no_count_for_cash_and_buckets','SELECT count(*) FROM portfolio_units WHERE NOT is_countable AND quantity_est IS NOT NULL')
 check('delta_arithmetic','SELECT count(*) FROM portfolio_units WHERE abs(delta_quantity_est-(quantity_est-previous_quantity_est))>1e-10*greatest(abs(delta_quantity_est),1) OR abs(delta_share_pp-100*(usable_weight-previous_weight))>1e-10')
 for col in ['quantity_est','previous_quantity_est','delta_quantity_est','relative_quantity_fraction','usable_weight','previous_weight','delta_share_pp','relative_weight_fraction','position_value_rub','unit_price_rub']:
  check('finite_'+col,'SELECT count(*) FROM portfolio_units WHERE "'+col+'" IS NOT NULL AND NOT isfinite("'+col+'")')
 check('summary_matches_reported_row_count','SELECT abs((SELECT sum(reported_positions) FROM summary_report_periods)-(SELECT count(*) FROM inv))')
 for p,h in hashes.items():qa.append(dict(check='unchanged_input_'+pathlib.Path(p).name,status='PASS' if sha(p)==h else 'FAIL',violations=0 if sha(p)==h else 1))
 save('data_qa.json',dict(checks=qa,passed=all(x['status']=='PASS' for x in qa)))
 if not all(x['status']=='PASS' for x in qa):raise RuntimeError('Dataset QA violations: '+str([x for x in qa if x['status']!='PASS']))
 stage('EXPORTING_EXCEL_INTERMEDIATES')
 fields="coalesce(fund_name,fund_id),cast(report_date AS DATE),asset_name,quantity_est,previous_quantity_est,delta_quantity_est,relative_quantity_fraction,usable_weight,previous_weight,delta_share_pp,relative_weight_fraction,position_value_rub,unit_price_rub,quote_date,cast(previous_report_date AS DATE),quantity_status,asset_type_ru,cast(fund_id AS VARCHAR),coalesce(price_isin,security_id),full_report_status"
 years=run('SELECT coalesce(year(report_date),0)::INTEGER y,count(*) n FROM portfolio_units GROUP BY y ORDER BY y').fetchall();manifest_years=[];preview=[]
 for year,n in years:
  name=f'portfolio_{year}.jsonl.gz';writer=gzip.open(R/name,'wt',encoding='utf-8',compresslevel=3,newline='\n');cur=c;cur.execute('SELECT '+fields+' FROM portfolio_units WHERE coalesce(year(report_date),0)='+str(year)+' ORDER BY fund_name,report_date,security_id');written=0
  while True:
   batch=cur.fetchmany(10000)
   if not batch:break
   for row in batch:
    values=[clean(v) for v in row];assert len(values)==20;writer.write(json.dumps(values,ensure_ascii=False,allow_nan=False)+'\n');written+=1
    if len(preview)<20:preview.append(values)
  writer.close();assert written==n
  manifest_years.append(dict(year=year,file=str(R/name),row_count=n,sheet_name=str(year) if year else 'Без даты'))
  stage('EXPORTING_EXCEL_INTERMEDIATES',completed_year=year,rows_written=n)
 cur=c;cur.execute('SELECT fund_name,cast(fund_id AS VARCHAR),report_date,reported_positions,estimated_positions,countable_positions_without_estimate,exits,total_position_value_rub,countable_value_price_coverage,full_report_status FROM summary_report_periods ORDER BY fund_name,report_date')
 sn=0
 with gzip.open(R/'summary.jsonl.gz','wt',encoding='utf-8',compresslevel=3,newline='\n') as f:
  while True:
   rows=cur.fetchmany(10000)
   if not rows:break
   for row in rows:f.write(json.dumps([clean(v) for v in row],ensure_ascii=False,allow_nan=False)+'\n');sn+=1
 features=duckdb.connect(str(R/'features.duckdb'))
 for table,file in [('portfolio_units','portfolio_units.parquet'),('summary_report_periods','summary_report_periods.parquet'),('endpoint_prices','endpoint_prices.parquet')]:features.execute('CREATE OR REPLACE VIEW '+table+' AS SELECT * FROM read_parquet('+sqlstr(R/file)+')')
 features.close()
 totals=run('SELECT count(*) AS total_rows,count(*) FILTER(WHERE is_reported_position) reported_rows,count(*) FILTER(WHERE is_exit) exit_rows,count(*) FILTER(WHERE is_reported_position AND quantity_est IS NOT NULL) reported_rows_quantity_estimated,count(distinct fund_id) funds,count(distinct report_id) reports,min(report_date) first_report,max(report_date) last_report FROM portfolio_units').fetchdf().to_dict('records')[0]
 label_stats=run('SELECT count(*) FILTER(WHERE asset_name=security_id) names_equal_identifier,count(*) FILTER(WHERE fund_name IS NULL) missing_fund_name FROM portfolio_units').fetchdf().to_dict('records')[0]
 statuses=run('SELECT quantity_status,count(*) n FROM portfolio_units GROUP BY 1 ORDER BY 2 DESC').fetchall()
 save('preview20.json',dict(headers=HEADERS,rows=preview));save('dataset_summary.json',dict(status='COMPLETE',totals=totals,status_counts=statuses,label_coverage=label_stats,canonical_endpoint_prices=scalar('SELECT count(*) FROM endpoint_prices'),conflicting_quote_groups_rejected=conflicts,QA_passed=True,QA_checks=len(qa),output_root=str(R),no_regressions=True))
 notes=['Расчетные количества V/P, не раскрытые фактические штуки. Количество не округлено до целого.','Облигации: чистая цена CLOSE*FACEVALUE/100; НКД не прибавляется, оценка условна относительно раскрытой стоимости.','Только существующие положительные RUB-котировки не позже фактической даты отчета, максимум 7 календарных дней.','Доли H/I и относительные изменения G/K хранятся как fractions; J уже процентные пункты.','При нулевой предыдущей базе относительное изменение отсутствует. Отсутствие позиции становится нулем только в полном отчете.','Строки выбытий добавлены отдельно и не считаются исходными раскрытыми позициями. Первые отчеты без предыдущих значений.','Не суммировать штуки разных бумаг. Источники: Investfunds https://investfunds.ru/ ; MOEX ISS https://iss.moex.com/iss/']
 save('excel_manifest.json',dict(ready=True,years=manifest_years,summary_file=str(R/'summary.jsonl.gz'),summary_row_count=sn,headers=HEADERS,summary_headers=SH,notes=notes,totals=totals,data_qa_file=str(R/'data_qa.json'),dataset_file=str(R/'portfolio_units.parquet')))
 stage('COMPLETED',totals=totals,excel_manifest_ready=True,qa_checks=len(qa));c.close()

if __name__=='__main__':
 try:main()
 except Exception as e:
  save('failure.json',dict(error=repr(e),traceback=traceback.format_exc()));stage('FAILED',error=repr(e));raise
