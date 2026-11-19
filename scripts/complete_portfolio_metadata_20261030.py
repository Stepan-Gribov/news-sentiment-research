import sys
sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
import pathlib,json,gzip,importlib.util,duckdb
R=pathlib.Path('./analysis_outputs/portfolio_units_20261030')
spec=importlib.util.spec_from_file_location('runner',R/'build_portfolio_units_20261030.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
c=duckdb.connect(config={'threads':2,'memory_limit':'2GB'})
for table in ['portfolio_units','summary_report_periods','endpoint_prices']:c.execute('CREATE TEMP VIEW '+table+' AS FROM read_parquet('+m.sqlstr(R/(table+'.parquet'))+')')
def run(s):return c.execute(s)
def scalar(s):return run(s).fetchone()[0]
qa=json.loads((R/'data_qa.json').read_text(encoding='utf-8'))['checks'];assert all(x['status']=='PASS' for x in qa)
assert scalar("SELECT count(*) FROM portfolio_units WHERE is_reported_position AND asset_name=security_id AND starts_with(security_id,'documented_bucket:')")==0
prior=json.loads((R/'before_category_labels/excel_manifest.json').read_text(encoding='utf-8'))
manifest_years=prior['years'];sn=scalar('SELECT count(*) FROM summary_report_periods');preview=[]
for y in manifest_years:
 assert scalar('SELECT count(*) FROM portfolio_units WHERE coalesce(year(report_date),0)='+str(y['year']))==y['row_count']
 assert 0<y['row_count']<=1048575
 with gzip.open(y['file'],'rt',encoding='utf-8') as f:
  if len(preview)<20:
   for line in f:
    preview.append(json.loads(line))
    if len(preview)==20:break
conflicts=json.loads((R/'dataset_summary.json').read_text(encoding='utf-8'))['conflicting_quote_groups_rejected']
HEADERS=m.HEADERS;SH=m.SH;save=m.save;stage=m.stage
totals=run('SELECT count(*) AS total_rows,count(*) FILTER(WHERE is_reported_position) reported_rows,count(*) FILTER(WHERE is_exit) exit_rows,count(*) FILTER(WHERE is_reported_position AND quantity_est IS NOT NULL) reported_rows_quantity_estimated,count(distinct fund_id) funds,count(distinct report_id) reports,min(report_date) first_report,max(report_date) last_report FROM portfolio_units').fetchdf().to_dict('records')[0]
label_stats=run('SELECT count(*) FILTER(WHERE asset_name=security_id) names_equal_identifier,count(*) FILTER(WHERE fund_name IS NULL) missing_fund_name FROM portfolio_units').fetchdf().to_dict('records')[0]
statuses=run('SELECT quantity_status,count(*) n FROM portfolio_units GROUP BY 1 ORDER BY 2 DESC').fetchall()
save('preview20.json',dict(headers=HEADERS,rows=preview));save('dataset_summary.json',dict(status='COMPLETE',totals=totals,status_counts=statuses,label_coverage=label_stats,canonical_endpoint_prices=scalar('SELECT count(*) FROM endpoint_prices'),conflicting_quote_groups_rejected=conflicts,QA_passed=True,QA_checks=len(qa),output_root=str(R),no_regressions=True))
notes=['Расчетные количества V/P, не раскрытые фактические штуки. Количество не округлено до целого.','Облигации: чистая цена CLOSE*FACEVALUE/100; НКД не прибавляется, оценка условна относительно раскрытой стоимости.','Только существующие положительные RUB-котировки не позже фактической даты отчета, максимум 7 календарных дней.','Доли H/I и относительные изменения G/K хранятся как fractions; J уже процентные пункты.','При нулевой предыдущей базе относительное изменение отсутствует. Отсутствие позиции становится нулем только в полном отчете.','Строки выбытий добавлены отдельно и не считаются исходными раскрытыми позициями. Первые отчеты без предыдущих значений.','Не суммировать штуки разных бумаг. Источники: Investfunds https://investfunds.ru/ ; MOEX ISS https://iss.moex.com/iss/']
save('excel_manifest.json',dict(ready=True,years=manifest_years,summary_file=str(R/'summary.jsonl.gz'),summary_row_count=sn,headers=HEADERS,summary_headers=SH,notes=notes,totals=totals,data_qa_file=str(R/'data_qa.json'),dataset_file=str(R/'portfolio_units.parquet')))
stage('COMPLETED',totals=totals,excel_manifest_ready=True,qa_checks=len(qa));c.close()
