"""Collect observed unit NAV histories and dated fund identities on research-host.
Protected sources are read only. Versioned releases are immutable; no alpha fits.
"""
from pathlib import Path
import sys,os,json,hashlib,time,datetime as dt,io,re,tarfile,traceback,sqlite3,collections
R=Path('./analysis_outputs/h3_alpha_20261104/fund_history')
CAT='./local/research-data-service/research.duckdb'
PANEL=Path('./analysis_outputs/h1_entries_trim_20261102/samples/AUGMENTED_P99.parquet')
ARCHIVE=Path('./databaseresources/funds/mac-20261008/fund_holdings_raw_20261008.tar.gz')
sys.path.insert(0,str(R/'vendor'))
import duckdb,pandas as pd,numpy as np,pyarrow as pa,pyarrow.parquet as pq,xlrd,requests
from bs4 import BeautifulSoup
START=time.monotonic()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def save(n,v):
 p=R/n;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def stage(s,**kw):
 v=dict(stage=s,active_pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),elapsed_seconds=round(time.monotonic()-START,2),**kw);save('checkpoint.json',v);print(json.dumps(v,default=str),flush=True)
def normreg(v):
 if v is None or pd.isna(v):return ''
 return re.sub(r'[\s№]','',str(v or '')).translate(str.maketrans({'–':'-','—':'-','−':'-'})).upper()
def normname(v):return re.sub(r'[^а-яa-z0-9]','',str(v).lower().replace('ё','е'))
def q(p):return "'"+str(p).replace('\\','/').replace("'","''")+"'"
def source_manifest():
 t=json.loads((R/'input_transfer_manifest.json').read_text(encoding='utf-8'));aliases={'fund_nav_and_unit_value_public_chart_data_verified.parquet':'existing_public_chart_history.parquet','fund_ids_from_holdings.csv':'existing_chart_fund_catalog.csv'}
 for n,v in t['files'].items():
  p=R/aliases.get(n,n);assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],n
 save('source_manifest.json',dict(transfer=t,archive=dict(path=str(ARCHIVE),bytes=ARCHIVE.stat().st_size,sha256=sha(ARCHIVE)),universe=dict(path=str(PANEL),bytes=PANEL.stat().st_size,sha256=sha(PANEL)),script_sha256=sha(__file__),data_key_semantics=dict(pay='Observed investment unit NAV, not fund total NAV and not exchange price',sca='Total fund net assets, not a fund return'),collection_start_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
def load_context():
 c=duckdb.connect(CAT,read_only=True)
 u=c.execute("SELECT fund_id,CASE WHEN count(distinct fund_focus)=1 THEN min(fund_focus) ELSE 'varies_over_time' END fund_focus,string_agg(DISTINCT fund_focus,'|') fund_focus_history FROM read_parquet(?) GROUP BY fund_id ORDER BY fund_id",[str(PANEL)]).fetchdf();assert u.fund_id.nunique()==603 and len(u)==603
 reports=c.execute('SELECT fund_id,report_id,cast(report_date AS DATE) report_date,registration_number,fund_name_file,fund_name_xls,net_asset_value_rub,source_path FROM funds.reports WHERE fund_id IN (SELECT fund_id FROM read_parquet(?))',[str(PANEL)]).fetchdf()
 origins=c.execute('SELECT DISTINCT fund_id,cast(end_date AS DATE) origin_date FROM read_parquet(?) ORDER BY fund_id,origin_date',[str(PANEL)]).fetchdf();c.close()
 reports['fund_id']=reports.fund_id.astype(str);reports['registration_normalized']=reports.registration_number.map(normreg);reports['report_date']=pd.to_datetime(reports.report_date)
 u['fund_id']=u.fund_id.astype(str)
 rows=[]
 for fund,g in reports.groupby('fund_id',sort=False):
  vals=[x for x in g.registration_normalized.unique() if x and x!='NONE'];primary=g.registration_normalized.value_counts().index[0]
  rows.append(dict(fund_id=fund,registration_number=primary,registration_variants='|'.join(sorted(vals)),registration_count=len(vals),identity_unambiguous=len(vals)==1,identity_observed_from=g.report_date.min(),identity_observed_to=g.report_date.max(),fund_name=g.sort_values('report_date').fund_name_file.iloc[-1],identity_source='Original dated Investfunds disclosure, exact source /funds/{fund_id}/ identity',source_url='https://investfunds.ru/funds/'+fund+'/'))
 cross=pd.DataFrame(rows).merge(u,on='fund_id',validate='one_to_one');return u,reports,origins,cross

def cache_daily(u,cross):
 out=R/'cached_unit_nav_daily.parquet'
 if out.exists() and (R/'cache_parse_complete.json').exists():return out
 catalog=pd.read_csv(R/'existing_chart_fund_catalog.csv',encoding='utf-8-sig',dtype={'fund_id':str}).sort_values('fund_id',key=lambda v:v.astype(int));allids=catalog.fund_id.tolist();name_by_id=dict(zip(catalog.fund_id,catalog.fund_name));wanted=set(u.fund_id);rows=[];issues=[];counts=0;writer=None
 try:
  for p in sorted((R/'nav_unit_value_chart_raw').glob('pay_batch_*.json')):
   n=int(re.search(r'batch_(\d+)',p.name).group(1));ids=allids[(n-1)*5:n*5];payload=json.loads(p.read_text(encoding='utf-8'))
   if len(payload)!=len(ids):issues.append(dict(file=p.name,reason='batch_cardinality_mismatch'));continue
   digest=sha(p)
   for j,(fund,series) in enumerate(zip(ids,payload)):
    if fund not in wanted:continue
    if normname(series.get('name'))!=normname(name_by_id[fund]):issues.append(dict(fund_id=fund,file=p.name,reason='batch_series_name_not_exact',returned_name=series.get('name'),expected_name=name_by_id[fund]));continue
    pts=series.get('data') or []
    if not pts:continue
    frame=pd.DataFrame(pts,columns=['timestamp_ms_source','unit_nav']);frame['fund_id']=fund;frame['series_name_source']=series.get('name');frame['source_kind']='Investfunds_public_chart_secondary';frame['source_url']='https://investfunds.ru/funds/1566/?action=chartData&data_key=pay&date_from=01.01.2012&date_to=31.08.2026&currencyId=1&ids[]='+fund
    frame['raw_file']=str(p);frame['raw_sha256']=digest;frame['series_index']=j;frame['point_index']=np.arange(len(frame),dtype='int64');frame['request_currency_id']=1;frame['currency']='RUB';frame['identity_series_name_exact']=True
    frame['value_date']=pd.to_datetime(frame.timestamp_ms_source,unit='ms',utc=True).dt.tz_convert('Europe/Moscow').dt.tz_localize(None).dt.normalize()
    frame['unit_nav']=pd.to_numeric(frame.unit_nav,errors='coerce').astype('float64');frame['timestamp_ms_source']=frame.timestamp_ms_source.astype('int64');frame['series_index']=frame.series_index.astype('int64')
    table=pa.Table.from_pandas(frame,preserve_index=False)
    if writer is None:writer=pq.ParquetWriter(out,table.schema,compression='zstd')
    writer.write_table(table);counts+=len(frame)
  if writer:writer.close();writer=None
 finally:
  if writer:writer.close()
 pd.DataFrame(issues).to_csv(R/'cache_identity_issues.csv',index=False,encoding='utf-8-sig')
 c=duckdb.connect(config={'threads':2,'memory_limit':'1GB'})
 c.execute('CREATE TEMP VIEW a AS FROM read_parquet('+q(out)+')')
 # Parity to the independently stored September collection; never trust batch order alone.
 c.execute('CREATE TEMP VIEW b AS SELECT cast(fund_id AS VARCHAR) fund_id,timestamp_ms_source,value_source unit_nav FROM read_parquet('+q(R/'existing_public_chart_history.parquet')+") WHERE data_key='pay'")
 mismatch=c.execute('SELECT count(*) FROM a LEFT JOIN b USING(fund_id,timestamp_ms_source) WHERE b.fund_id IS NULL OR a.unit_nav<>b.unit_nav').fetchone()[0];assert mismatch==0
 save('cache_parse_complete.json',dict(rows=counts,funds=c.execute('SELECT count(distinct fund_id) FROM a').fetchone()[0],issues=issues,verified_against_existing_parquet=True));c.close();return out

def release(tag,daily,u,reports,origins,cross):
 D=R/tag;D.mkdir(exist_ok=True)
 if (D/'sealed_outputs.json').exists():return json.loads((D/'ready_manifest.json').read_text(encoding='utf-8'))
 stage('BUILDING_MONTHLY_RELEASE',release=tag)
 c=duckdb.connect(config={'threads':2,'memory_limit':'2GB','temp_directory':str(R/'spill')});c.execute("SET TimeZone='Europe/Moscow'")
 c.execute('CREATE TEMP VIEW daily AS FROM read_parquet('+q(daily)+')')
 # Reject conflicting same-date unit values rather than choosing by order.
 conflicts=c.execute('SELECT fund_id,value_date,count(distinct unit_nav) n FROM daily WHERE isfinite(unit_nav) AND unit_nav>0 GROUP BY ALL HAVING n>1').fetchdf();conflicts.to_parquet(D/'daily_conflicts.parquet',index=False)
 c.execute('''CREATE TEMP TABLE canonical_daily AS SELECT * EXCLUDE(rn) FROM (SELECT *,row_number() OVER(PARTITION BY fund_id,value_date ORDER BY raw_file,point_index) rn FROM daily WHERE unit_nav>0 AND isfinite(unit_nav)) WHERE rn=1 AND (fund_id,value_date) NOT IN (SELECT fund_id,value_date FROM daily GROUP BY fund_id,value_date HAVING count(distinct unit_nav)>1)''')
 c.execute('COPY canonical_daily TO '+q(D/'unit_values_daily.parquet')+' (FORMAT PARQUET,COMPRESSION ZSTD)')
 levels=c.execute("SELECT * EXCLUDE(rn) FROM (SELECT *,strftime(value_date,'%Y-%m') AS month,row_number() OVER(PARTITION BY fund_id,strftime(value_date,'%Y-%m') ORDER BY value_date DESC) rn FROM canonical_daily) WHERE rn=1").fetchdf();c.close()
 months=pd.period_range('2008-01','2026-10',freq='M');grid=pd.MultiIndex.from_product([u.fund_id,months.astype(str)],names=['fund_id','month']).to_frame(index=False)
 grid=grid.merge(levels,on=['fund_id','month'],how='left',validate='one_to_one').merge(cross,on='fund_id',how='left',validate='many_to_one')
 grid['month_end']=pd.PeriodIndex(grid.month,freq='M').to_timestamp(how='end').normalize();grid['source_age_days']=(grid.month_end-grid.value_date).dt.days
 # Registration is checked at actual dated disclosures. Outside evidence bounds remains a separate, explicit limitation.
 grid['dated_identity_verified']=grid.identity_unambiguous & grid.value_date.ge(grid.identity_observed_from)&grid.value_date.le(grid.identity_observed_to)
 grid['level_observed']=grid.unit_nav.notna() & grid.unit_nav.gt(0)&np.isfinite(grid.unit_nav)
 calendar_path=Path('./analysis_outputs/h1_macro_no_time_20261031/inputs/raw_external_2026-09-30/moex/indices/IMOEX_daily_history_2012-01-01_2026-08-31.csv')
 cal=pd.read_csv(calendar_path);datecol=next(x for x in cal.columns if x.upper()=='TRADEDATE');days=np.sort(pd.to_datetime(cal[datecol]).dropna().unique().astype('datetime64[ns]'))
 grid['trading_calendar_verified']=grid.value_date.ge(pd.Timestamp(days[0])) & grid.month_end.le(pd.Timestamp(days[-1]))
 grid['source_age_trading_days']=np.searchsorted(days,grid.month_end.to_numpy(dtype='datetime64[ns]'),side='right')-np.searchsorted(days,grid.value_date.to_numpy(dtype='datetime64[ns]'),side='right')
 grid.loc[~grid.trading_calendar_verified,'source_age_trading_days']=np.nan
 grid['level_fresh']=grid.level_observed & grid.source_age_days.ge(0)&grid.source_age_trading_days.between(0,3)&grid.trading_calendar_verified
 grid['level_verified']=grid.level_fresh & grid.dated_identity_verified & grid.identity_series_name_exact.fillna(False)
 grid=grid.sort_values(['fund_id','month']).reset_index(drop=True)
 for col in ['unit_nav','value_date','level_verified','month','raw_file','raw_sha256','point_index']:
  grid['previous_'+col]=grid.groupby('fund_id',sort=False)[col].shift(1)
 grid['month_adjacent']=pd.PeriodIndex(grid.month,freq='M').asi8- pd.PeriodIndex(grid.previous_month.fillna(grid.month),freq='M').asi8==1
 grid['fund_return_observed']=grid.unit_nav/grid.previous_unit_nav-1
 grid['return_verified']=grid.level_verified & grid.previous_level_verified.fillna(False)&grid.month_adjacent&np.isfinite(grid.fund_return_observed)
 grid['fund_return']=grid.fund_return_observed.where(grid.return_verified)
 grid['return_basis']='unit_NAV_change_cash_distributions_and_splits_not_fully_verified'
 grid['distributions_verified']=False;grid['splits_verified']=False;grid['total_return_verified']=False
 grid['return_missing_reason']=np.select([np.asarray(x.fillna(False),dtype=bool) for x in [~grid.level_observed,~grid.level_fresh,~grid.identity_unambiguous,~grid.dated_identity_verified,~grid.previous_level_verified.fillna(False),~grid.month_adjacent,~np.isfinite(grid.fund_return_observed)]],['no_observed_unit_value','stale_month_end_value','conflicting_registration_identity','outside_dated_registration_evidence','previous_month_endpoint_unverified','nonadjacent_month','nonfinite_return'],default='eligible_observed_NAV_return')
 assert not grid.duplicated(['fund_id','month']).any();assert grid.loc[grid.return_verified,'source_age_trading_days'].le(3).all()
 assert np.isfinite(grid.loc[grid.return_verified,'fund_return']).all();assert (grid.loc[grid.return_verified,'fund_return']>=-1).all()
 grid['series_id']='investfunds:'+grid.fund_id;grid['valuation_date']=grid.value_date;grid['unit_value']=grid.unit_nav;grid['source_path']=grid.raw_file;grid['identity_valid_from']=grid.identity_observed_from;grid['identity_valid_to']=grid.identity_observed_to;grid['identity_evidence']=grid.identity_source
 grid['start_valuation_date']=grid.previous_value_date;grid['end_valuation_date']=grid.value_date;grid['identity_verified']=grid.dated_identity_verified;grid['distribution_verified']=False;grid['split_verified']=False;grid['source_id']='investfunds_public_chart';grid['distribution_status']='unverified_completeness';grid['split_status']='unverified_completeness'
 grid.to_parquet(D/'fund_monthly_returns.parquet',index=False,compression='zstd');cross.to_parquet(D/'fund_series_crosswalk.parquet',index=False,compression='zstd')
 reports.to_parquet(D/'dated_identity_evidence.parquet',index=False,compression='zstd')
 # Coverage is computed per actual sample origin, with no month-t return observed after that origin.
 coverage=[]
 byfund={f:g.reset_index(drop=True) for f,g in grid.groupby('fund_id',sort=False)}
 for fund,gorig in origins.groupby('fund_id',sort=False):
  g=byfund[fund];idx={m:i for i,m in enumerate(g.month)};valid=g.return_verified.to_numpy(bool);prefix=np.r_[0,valid.cumsum()]
  for origin in gorig.itertuples(index=False):
   d=pd.Timestamp(origin.origin_date);m=str(d.to_period('M'));i=idx[m];tstar=i if g.month_end.iloc[i]<=d else i-1
   n60=int(prefix[tstar+1]-prefix[max(0,tstar-59)]);n36=int(prefix[tstar+1]-prefix[max(0,tstar-35)]);n12=int(prefix[tstar+1]-prefix[max(0,tstar-11)])
   for h in [1,2,3]:
    future=i+h<len(g) and bool(pd.notna(g.value_date.iloc[i]) and g.value_date.iloc[i]>=d) and bool(valid[i+1:i+h+1].all()) and bool((g.value_date.iloc[i+1:i+h+1]>d).all())
    coverage.append(dict(fund_id=fund,origin_date=d,month=m,training_end_month=g.month.iloc[tstar],horizon=h,n_past60=n60,n_past36=n36,n_past_last12=n12,future_complete=future,history_36in60=n60>=36 and n12>=9,history_24in36=n36>=24 and n12>=9,eligible_36in60=n60>=36 and n12>=9 and future,eligible_24in36=n36>=24 and n12>=9 and future,return_basis='observed_unit_NAV_change',total_return_verified=False))
 cv=pd.DataFrame(coverage).merge(u,on='fund_id',validate='many_to_one');cv.to_parquet(D/'history_eligibility_origins.parquet',index=False,compression='zstd')
 per=grid.groupby('fund_id').agg(observed_levels=('level_observed','sum'),verified_levels=('level_verified','sum'),strict_returns=('return_verified','sum'),first_observed=('value_date','min'),last_observed=('value_date','max')).reset_index().merge(cross,on='fund_id',validate='one_to_one');per.to_csv(D/'coverage_by_fund.csv',index=False,encoding='utf-8-sig')
 summary=dict(status='READY_OBSERVED_UNIT_NAV_RETURNS',release=tag,universe_funds=603,unit_value_funds=int(levels.fund_id.nunique()),positive_daily_values=pq.read_metadata(D/'unit_values_daily.parquet').num_rows,observed_monthly_levels=int(grid.level_observed.sum()),fresh_monthly_levels=int(grid.level_fresh.sum()),dated_identity_verified_levels=int(grid.level_verified.sum()),strict_monthly_returns=int(grid.return_verified.sum()),funds_with_strict_returns=int(grid.loc[grid.return_verified,'fund_id'].nunique()),ambiguous_identity_funds=int((~cross.identity_unambiguous).sum()),coverage=cv.groupby('horizon').agg(eligible36_origins=('eligible_36in60','sum'),eligible24_origins=('eligible_24in36','sum')).reset_index().to_dict('records'),eligible36_funds=int(cv.loc[cv.eligible_36in60,'fund_id'].nunique()),eligible24_funds=int(cv.loc[cv.eligible_24in36,'fund_id'].nunique()),minimum_endpoint_age_policy='last observed value inside calendar month, age<=3 actual MOEX index trading days, no interpolation',limitations=['Observed unit NAV changes, not verified total returns; distribution and split completeness not established','Dated identity bounds use exact same-provider fundID and consistent normalized registration across original disclosures','Some provider names can change; batch series must match the independently saved exact fund catalogue name','No inference from total NAV growth or securities quantities'])
 for n,v in [('counts.json',summary),('schema.json',dict(columns=[dict(name=n,type=str(t)) for n,t in grid.dtypes.items()])),('qa.json',dict(status='PASS',unique_fund_month=True,adjacent_months_only=True,strict_values_finite=True,endpoint_age_le3_trading_days=True,source_point_provenance=True,distributions_unknown_not_zero=True))]:save(tag+'/'+n,v)
 preview=grid.loc[grid.return_verified,['fund_id','fund_name','registration_number','month','value_date','unit_nav','previous_value_date','previous_unit_nav','fund_return','return_basis','raw_file','point_index']].head(20);preview.to_csv(D/'preview20.csv',index=False,encoding='utf-8-sig')
 files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in D.iterdir() if p.is_file()}
 ready=dict(ready=True,release=tag,root=str(D),return_file=str(D/'fund_monthly_returns.parquet'),crosswalk_file=str(D/'fund_series_crosswalk.parquet'),coverage_file=str(D/'history_eligibility_origins.parquet'),primary_return_column='fund_return',verification_column='return_verified',strict_returns_mean='Observed adjacent per-unit NAV returns; total return completeness separate',total_return_verified=False,counts=summary)
 save(tag+'/ready_manifest.json',ready);save(tag+'/sealed_outputs.json',dict(files=files,script_sha256=sha(__file__),sealed_utc=dt.datetime.now(dt.timezone.utc).isoformat()));save('ready_manifest.json',ready)
 stage('UNIT_NAV_SUBSET_READY',release=tag,strict_monthly_returns=summary['strict_monthly_returns'],eligible36_funds=summary['eligible36_funds']);return ready

def scan_archive(wanted):
 if (R/'archive_scan_complete.json').exists():return
 stage('SCANNING_RAW_DISCLOSURE_ARCHIVE')
 rows=[];matches=[];errors=[];parsed=0;unit_fields=0;pattern=re.compile(r'расч[её]тн.{0,40}стоимост.{0,30}па|стоимост.{0,20}инвестиционн.{0,10}па|количеств.{0,30}(?:выпущен.{0,12})?(?:инвестиционн.{0,10})?па[её]|unit.{0,15}(?:value|NAV)|units.{0,15}(?:issued|outstanding)',re.I)
 with tarfile.open(ARCHIVE,'r|gz') as t:
  for m in t:
   match=re.search(r'/xls/(\d+)/(\d{4}-\d{2}-\d{2})_',m.name)
   if not match or match[1] not in wanted or not m.isfile():continue
   body=t.extractfile(m).read()
   try:
    wb=xlrd.open_workbook(file_contents=body,on_demand=True,logfile=io.StringIO());hits=[];nav=None;nav_cell=None
    for sh in wb.sheets():
     for i in range(sh.nrows):
      vals=sh.row_values(i)
      for j,v in enumerate(vals):
       if not isinstance(v,str):continue
       if re.search(r'СЧА.{0,20}руб|стоимост.{0,12}чист.{0,8}актив',v,re.I):
        near=[(k,x) for k,x in enumerate(vals[j+1:],j+1) if isinstance(x,(int,float)) and x>0]
        if near:nav=near[0][1];nav_cell=sh.name+'!R'+str(i+1)+'C'+str(near[0][0]+1)
       if pattern.search(v):hits.append(dict(sheet=sh.name,row1=i+1,column1=j+1,label=v,row_values=vals))
    rows.append(dict(fund_id=match[1],report_date=match[2],member=m.name,member_sha256=hashlib.sha256(body).hexdigest(),sheets='|'.join(wb.sheet_names()),nav_rub=nav,nav_cell=nav_cell,unit_field_hits=len(hits)))
    if hits:matches.append(dict(member=m.name,fund_id=match[1],report_date=match[2],fields=hits));unit_fields+=len(hits)
    wb.release_resources();parsed+=1
   except Exception as e:errors.append(dict(member=m.name,error=repr(e)))
   if parsed%1000==0:stage('SCANNING_RAW_DISCLOSURE_ARCHIVE',workbooks_parsed=parsed,unit_field_hits=unit_fields,errors=len(errors))
 pd.DataFrame(rows).to_parquet(R/'raw_archive_disclosure_fields.parquet',index=False,compression='zstd');save('raw_archive_unit_field_matches.json',matches);save('archive_scan_complete.json',dict(workbooks_parsed=parsed,unit_field_hits=unit_fields,errors=errors,all_requested_funds_scanned=True,archive=str(ARCHIVE)))

class Fetcher:
 def __init__(self):self.session=requests.Session();self.last=0;self.count=0;self.errors=0
 def get(self,url,params=None):
  for attempt in range(3):
   time.sleep(max(0,1.1-(time.monotonic()-self.last)));self.last=time.monotonic()
   try:
    x=self.session.get(url,params=params,timeout=(12,45),headers={'User-Agent':'Academic-fund-NAV-research/1.0'});self.count+=1
    if x.status_code==429 or x.status_code>=500:raise RuntimeError('HTTP '+str(x.status_code))
    x.raise_for_status();return x
   except requests.exceptions.SSLError:raise
   except Exception:
    self.errors+=1
    if attempt==2:raise
    time.sleep(2**attempt)

def collect_extended(u,cross):
 stage('COLLECTING_SINGLE_ID_HISTORY_AND_PROFILES')
 f=Fetcher();raw=R/'collected_raw';raw.mkdir(exist_ok=True);log=[];profiles=[];errors=[];ssl_streak=0
 for n,fund in enumerate(u.fund_id,1):
  pay=raw/('pay_'+fund+'.json');html=raw/('profile_'+fund+'.html');url='https://investfunds.ru/funds/'+fund+'/'
  try:
   if not pay.exists():
    x=f.get(url,params=[('action','chartData'),('data_key','pay'),('date_from','01.01.2008'),('date_to','31.10.2026'),('currencyId','1'),('ids[]',fund)])
    body=x.json();assert isinstance(body,list);pay.write_bytes(x.content);log.append(dict(fund_id=fund,kind='pay',url=x.url,status=x.status_code,sha256=sha(pay),bytes=pay.stat().st_size))
   if not html.exists():
    x=f.get(url);html.write_bytes(x.content);log.append(dict(fund_id=fund,kind='profile',url=x.url,status=x.status_code,sha256=sha(html),bytes=html.stat().st_size))
   text=BeautifulSoup(html.read_bytes(),'html.parser').get_text(' ',strip=True);regs=re.findall(r'(?<!\d)\d{4}\s*[-–]\s*\d{8,10}(?:-\d+)?|(?<=Номер регистрации )\d{4,6}',text)
   income='Выплаты дохода' in text
   profiles.append(dict(fund_id=fund,source_url=url,profile_file=str(html),profile_sha256=sha(html),registration_candidates='|'.join(sorted(set(map(normreg,regs)))),income_section_present=income,distributions_status='public_income_section_present_need_event_parse' if income else 'no_income_section_reported_not_proof_of_zero',fetched_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
   ssl_streak=0
  except requests.exceptions.SSLError as e:
   errors.append(dict(fund_id=fund,error=repr(e),reason='TLS_verification_failed_no_bypass'));ssl_streak+=1
   if ssl_streak>=3:break
  except Exception as e:errors.append(dict(fund_id=fund,error=repr(e)))
  if n%10==0:save('extended_fetch_log.json',log);save('extended_fetch_errors.json',errors);pd.DataFrame(profiles).to_parquet(R/'fund_profile_evidence.parquet',index=False);stage('COLLECTING_SINGLE_ID_HISTORY_AND_PROFILES',funds_attempted=n,total=603,cached_pay_files=len(list(raw.glob('pay_*.json'))),cached_profile_files=len(list(raw.glob('profile_*.html'))),requests=f.count,errors=len(errors))
 save('extended_fetch_log.json',log);save('extended_fetch_errors.json',errors);pd.DataFrame(profiles).to_parquet(R/'fund_profile_evidence.parquet',index=False)
 # Build a second immutable release from strictly one-ID public requests; keep the first release intact.
 out=R/'extended_unit_nav_daily.parquet';writer=None
 try:
  for p in sorted(raw.glob('pay_*.json')):
   fund=p.stem.split('_',1)[1];payload=json.loads(p.read_text(encoding='utf-8'))
   if len(payload)!=1:continue
   pts=payload[0].get('data') or []
   if not pts:continue
   frame=pd.DataFrame(pts,columns=['timestamp_ms_source','unit_nav']);frame['fund_id']=fund;frame['series_name_source']=payload[0].get('name');frame['source_kind']='Investfunds_public_chart_secondary_singleID';frame['source_url']='https://investfunds.ru/funds/'+fund+'/?action=chartData&data_key=pay&date_from=01.01.2008&date_to=31.10.2026&currencyId=1&ids[]='+fund
   frame['raw_file']=str(p);frame['raw_sha256']=sha(p);frame['series_index']=np.int64(0);frame['point_index']=np.arange(len(frame),dtype='int64');frame['request_currency_id']=1;frame['currency']='RUB';frame['identity_series_name_exact']=True
   frame['value_date']=pd.to_datetime(frame.timestamp_ms_source,unit='ms',utc=True).dt.tz_convert('Europe/Moscow').dt.tz_localize(None).dt.normalize();frame['unit_nav']=pd.to_numeric(frame.unit_nav,errors='coerce').astype('float64');frame['timestamp_ms_source']=frame.timestamp_ms_source.astype('int64')
   table=pa.Table.from_pandas(frame,preserve_index=False)
   if writer is None:writer=pq.ParquetWriter(out,table.schema,compression='zstd')
   writer.write_table(table)
  if writer:writer.close();writer=None
 finally:
  if writer:writer.close()
 if out.exists():return out

def main():
 R.mkdir(parents=True,exist_ok=True)
 stage('VERIFYING_PRESERVED_INPUTS')
 source_manifest();u,reports,origins,cross=load_context();daily=cache_daily(u,cross);release('v1_existing_cache',daily,u,reports,origins,cross)
 scan_archive(set(u.fund_id))
 extended=collect_extended(u,cross)
 if extended is not None:
  # Extend dated identity upper bound only when a freshly saved profile confirms exactly the same registration.
  profiles=pd.read_parquet(R/'fund_profile_evidence.parquet');extra=dict(zip(profiles.fund_id,profiles.registration_candidates))
  cross2=cross.copy();cross2['identity_extended_by_fresh_profile']=cross2.apply(lambda x:bool(x.identity_unambiguous and x.registration_number in str(extra.get(x.fund_id,'')).split('|')),axis=1)
  cross2.loc[cross2.identity_extended_by_fresh_profile,'identity_observed_to']=pd.Timestamp('2026-11-04')
  release('v2_single_id_extended',extended,u,reports,origins,cross2)
 stage('COMPLETED',first_release_ready=True,extended_release_ready=extended is not None)
if __name__=='__main__':
 R.mkdir(exist_ok=True,parents=True);sys.stdout=(R/'collection.stdout.log').open('a',encoding='utf-8',buffering=1);sys.stderr=(R/'collection.stderr.log').open('a',encoding='utf-8',buffering=1)
 try:main()
 except Exception as e:save('failure.json',dict(error=repr(e),traceback=traceback.format_exc()));stage('FAILED',error=repr(e));raise
