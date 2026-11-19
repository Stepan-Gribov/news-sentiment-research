import sys
sys.stdout.reconfigure(encoding='utf-8')
import pathlib,json,hashlib,datetime as dt,os,time,gzip,math,traceback,collections,importlib.util
import duckdb,pyarrow as pa
R=pathlib.Path('./analysis_outputs/portfolio_units_20261030')
spec=importlib.util.spec_from_file_location('runner',R/'build_portfolio_units_20261030.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
summary=json.loads((R/'dataset_summary.json').read_text(encoding='utf-8'))
qa=json.loads((R/'data_qa.json').read_text(encoding='utf-8'))['checks'];assert all(x['status']=='PASS' for x in qa)
manifest=json.loads((R/'excel_manifest.json').read_text(encoding='utf-8'));assert manifest['ready']
m.save('excel_manifest.json',dict(ready=False,stage='IMPROVING_CATEGORY_LABELS'))
m.stage('IMPROVING_CATEGORY_LABELS')
c=duckdb.connect(config={'threads':2,'memory_limit':'2GB','temp_directory':str(R/'spill_labels')})
c.execute("ATTACH './local/research-data-service/research.duckdb' AS upstream (READ_ONLY)")
names=collections.defaultdict(set)
for cat,sub in c.execute('SELECT DISTINCT parent_category,parent_subcategory FROM upstream.funds.report_lines').fetchall():
 cat=str(cat or '');sub=str(sub or '');label=' / '.join(x for x in [cat,sub] if x)
 if label:names['documented_bucket:'+hashlib.blake2b((cat+'\t'+sub).encode(),digest_size=12).hexdigest()].add(label)
c.register('bucket_labels',pa.Table.from_pylist([dict(security_id=k,label=' / '.join(sorted(v))) for k,v in names.items()]))
ps=m.sqlstr(R/'portfolio_units.parquet')
cols=[x[0] for x in c.execute('DESCRIBE SELECT * FROM read_parquet('+ps+')').fetchall() if x[0]!='asset_name']
hash_expr='hash('+','.join('"'+x+'"' for x in cols)+')'
base=c.execute('SELECT count(*),count(*) FILTER(WHERE is_reported_position),bit_xor('+hash_expr+') FROM read_parquet('+ps+')').fetchone()
before=c.execute("SELECT count(*) FROM read_parquet("+ps+") WHERE is_reported_position AND asset_name=security_id AND starts_with(security_id,'documented_bucket:')").fetchone()[0]
c.execute('CREATE TEMP TABLE portfolio_units AS SELECT p.* REPLACE(CASE WHEN p.asset_name=p.security_id THEN coalesce(b.label,p.asset_name) ELSE p.asset_name END AS asset_name) FROM read_parquet('+ps+') p LEFT JOIN bucket_labels b USING(security_id)')
after=c.execute('SELECT count(*),count(*) FILTER(WHERE is_reported_position),bit_xor('+hash_expr+') FROM portfolio_units').fetchone();assert base==after
remaining=c.execute("SELECT count(*) FROM portfolio_units WHERE is_reported_position AND asset_name=security_id AND starts_with(security_id,'documented_bucket:')").fetchone()[0];assert remaining==0,(before,remaining)
c.execute('COPY portfolio_units TO '+m.sqlstr(R/'portfolio_units.labels.tmp.parquet')+' (FORMAT PARQUET, COMPRESSION ZSTD)')
(R/'portfolio_units.labels.tmp.parquet').replace(R/'portfolio_units.parquet')
for table in ['summary_report_periods','endpoint_prices']:c.execute('CREATE TEMP VIEW '+table+' AS FROM read_parquet('+m.sqlstr(R/(table+'.parquet'))+')')
qa.extend([dict(check='label_repair_preserves_all_other_values',status='PASS',violations=0),dict(check='documented_buckets_have_readable_names',status='PASS',violations=remaining)])
m.save('data_qa.json',dict(checks=qa,passed=True,label_repair=dict(reported_bucket_labels_fixed=before-remaining,remaining_bucket_ids=remaining)))
def run(s):return c.execute(s)
def scalar(s):return run(s).fetchone()[0]
HEADERS=m.HEADERS;SH=m.SH;save=m.save;stage=m.stage;clean=m.clean;sqlstr=m.sqlstr
conflicts=summary['conflicting_quote_groups_rejected']
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
