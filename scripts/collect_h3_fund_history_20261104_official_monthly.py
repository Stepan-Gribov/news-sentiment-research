"""Resumable official monthly report collection with deterministic PDF-field provenance.
Output is a documented-return candidate, not automatic primary H3 certification.
"""
import pathlib,sys,re,json,hashlib,time,datetime as dt
from urllib.parse import urljoin
R=pathlib.Path('./analysis_outputs/h3_alpha_20261104/fund_history');sys.path.insert(0,str(R/'vendor'))
import requests,pandas as pd,numpy as np
from bs4 import BeautifulSoup
from pypdf import PdfReader
O=R/'official_monthly_returns';O.mkdir(exist_ok=True);raw=O/'raw';raw.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(n,v):
 p=O/n;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8');t.replace(p)
def norm(x):return re.sub(r'[^а-яa-z0-9]','',x.lower().replace('ё','е'))
MONTHS={'январь':1,'февраль':2,'март':3,'апрель':4,'май':5,'июнь':6,'июль':7,'август':8,'сентябрь':9,'октябрь':10,'ноябрь':11,'декабрь':12}
FUNDS={'403':'Арсагера — фонд акций','404':'Арсагера — фонд смешанных инвестиций','481':'Арсагера — акции 6.4','3435':'Арсагера — фонд облигаций КР 1.55'}
BASE='https://arsagera.ru/products/opifa_arsagera_-_fond_akcij/'
METHOD='https://arsagera.ru/kuda_i_kak_investirovat/klyuchevye_metodiki_upravleniya_kapitalom/poryadok_opredeleniya_dohodnosti_paevogo_investicionnogo_fonda/'
last=0;session=requests.Session();current_request={}
def get(url,path):
 global last
 if path.exists():return path.read_bytes()
 for attempt in range(3):
  time.sleep(max(0,5.0-(time.monotonic()-last)));last=time.monotonic()
  save('checkpoint.json',dict(stage='FETCHING_OFFICIAL_SOURCE',active_pid=__import__('os').getpid(),url=url,path=str(path),attempt=attempt+1,utc=dt.datetime.now(dt.timezone.utc).isoformat(),**current_request))
  try:
   x=session.get(url,timeout=(8,15),headers={'User-Agent':'Academic-fund-performance-research/1.0'});x.raise_for_status();path.write_bytes(x.content);return x.content
  except requests.exceptions.SSLError:raise
  except Exception as e:
   save('last_fetch_error.json',dict(url=url,path=str(path),attempt=attempt+1,error=repr(e),utc=dt.datetime.now(dt.timezone.utc).isoformat()))
   if attempt==2:raise
   time.sleep(2**attempt)
parent=(R/'official_sources/arsagera_fa_product.html').read_bytes();soup=BeautifulSoup(parent,'html.parser');products={}
for fund,name in FUNDS.items():
 links=[urljoin(BASE,a.get('href','')) for a in soup.find_all('a') if norm(a.get_text(' ',strip=True))==norm(name) and '/products/' in a.get('href','')]
 assert links,(fund,name);products[fund]=links[0]
nav=pd.read_parquet(R/'official_sources/arsagera_official_unit_nav.parquet');nav['month']=pd.to_datetime(nav.valuation_date).dt.to_period('M').astype(str);nav=nav.sort_values('valuation_date').groupby(['fund_id','month'],sort=False).tail(1);nav_idx={(str(x.fund_id),x.month):x for x in nav.itertuples(index=False)}
method_path=R/'official_sources/arsagera_return_method.html';assert method_path.exists()
manifest=[];records=[];errors=[]
for fund,url in products.items():
 current_request=dict(fund_id=fund,kind='product_page',reports=len(manifest),errors=len(errors))
 try:body=get(url,raw/('product_'+fund+'.html'))
 except Exception as e:errors.append(dict(fund_id=fund,error=repr(e)));continue
 ps=BeautifulSoup(body,'html.parser');targets={}
 for a in ps.find_all('a'):
  text=a.get_text(' ',strip=True);m=re.match(r'^(январь|февраль|март|апрель|май|июнь|июль|август|сентябрь|октябрь|ноябрь|декабрь)\s+(20\d{2})\b',text,re.I)
  if not m or int(m[2])<2013 or int(m[2])>2026:continue
  month=f'{int(m[2]):04d}-{MONTHS[m[1].lower()]:02d}';link=urljoin(url,a.get('href',''))
  if '/library/download/' in link:targets[month]=dict(url=link,label=text)
 save('report_links_'+fund+'.json',targets)
 for month,target in sorted(targets.items()):
  p=raw/(fund+'_'+month+'.pdf');current_request=dict(fund_id=fund,month=month,reports=len(manifest),errors=len(errors))
  try:
   get(target['url'],p);reader=PdfReader(p);found=None
   for pi,page in enumerate(reader.pages):
    text=page.extract_text(extraction_mode='layout') or '';lines=text.splitlines()
    for li,line in enumerate(lines):
     if not re.match(r'^\s*изменение\s+стоимости\s+пая\b',line,re.I):continue
     headers=[z for z in lines[max(0,li-7):li] if re.search(r'\b1\s*мес',z,re.I)]
     if not headers:continue
     header=headers[-1];h=re.search(r'\b1\s*мес\.?',header,re.I);center=(h.start()+h.end())/2
     tokens=list(re.finditer(r'[-−+]?\d+(?:[ \xa0]\d{3})*(?:[,.]\d+)?\s*%',line))
     if not tokens:continue
     token=min(tokens,key=lambda t:abs((t.start()+t.end())/2-center));distance=abs((token.start()+token.end())/2-center)
     if distance>12:continue
     number=token.group().replace('%','').replace('−','-').replace(' ','').replace('\xa0','').replace(',','.');value=float(number)
     found=dict(page1=pi+1,line1=li+1,header=header.strip(),value_line=line.strip(),selected_token=token.group(),column_distance=distance,reported_return_percent=value)
   if found is None:raise ValueError('No unambiguous monthly-return table field')
   curr=nav_idx.get((fund,month));prev=nav_idx.get((fund,str(pd.Period(month,freq='M')-1)))
   ratio=curr.unit_value/prev.unit_value-1 if curr is not None and prev is not None and curr.unit_value>0 and prev.unit_value>0 else None
   ret=found['reported_return_percent']/100
   record=dict(fund_id=fund,month=month,reported_monthly_return=ret,source_return_label='изменение стоимости пая; official monthly report',raw_file=str(p),source_url=target['url'],raw_sha256=sha(p),method_source_url=METHOD,method_source_path=str(method_path),method_sha256=sha(method_path),method_publication_date='2021-09-27',method_chronologically_documented=month>='2021-10',fund_price_end=curr.unit_value if curr is not None else None,fund_price_start=prev.unit_value if prev is not None else None,end_valuation_date=curr.valuation_date if curr is not None else None,start_valuation_date=prev.valuation_date if prev is not None else None,unit_NAV_ratio_return=ratio,rounding_tolerance_return=.00005001,consistent_with_observed_NAV_ratio=abs(ret-ratio)<=.00005001 if ratio is not None else False,published_return_precision_percent='as printed, commonly0.01percentagepoint',total_return_verified=False,requires_method_review=True,**found)
   records.append(record);manifest.append(dict(fund_id=fund,month=month,path=str(p),url=target['url'],bytes=p.stat().st_size,sha256=sha(p)))
  except Exception as e:errors.append(dict(fund_id=fund,month=month,url=target['url'],error=repr(e)))
  if True:
   save('checkpoint.json',dict(stage='COLLECTING_OFFICIAL_MONTHLY_RETURNS',active_pid=__import__('os').getpid(),reports=len(manifest),fund_id=fund,month=month,errors=len(errors),utc=dt.datetime.now(dt.timezone.utc).isoformat()))
   pd.DataFrame(records).to_parquet(O/'published_monthly_return_candidates.parquet',index=False,compression='zstd');save('fetch_errors.json',errors)
frame=pd.DataFrame(records);frame.to_parquet(O/'published_monthly_return_candidates.parquet',index=False,compression='zstd');save('source_manifest.json',manifest);save('fetch_errors.json',errors)
summary=dict(status='OFFICIAL_DOCUMENTED_RETURN_CANDIDATES_READY',reports=len(frame),funds=int(frame.fund_id.nunique()) if not frame.empty else 0,method_chronologically_documented_rows=int(frame.method_chronologically_documented.sum()) if not frame.empty else 0,monthly_ratio_agreement_rows=int(frame.consistent_with_observed_NAV_ratio.sum()) if not frame.empty else 0,needs_review_disagreement_rows=int((~frame.consistent_with_observed_NAV_ratio).sum()) if not frame.empty else 0,errors=len(errors),files=dict(data=str(O/'published_monthly_return_candidates.parquet'),manifest=str(O/'source_manifest.json')),method='Official 2021 method explicitly adds per-unit distributions when applicable and adjusts for unit splits; do not equate the printed report label alone with complete TR history',usage='Input source evidence for research-owner certification; candidate data do not automatically receive TR_VERIFIED',no_new_regressions=True)
save('summary.json',summary);save('checkpoint.json',dict(stage='COMPLETED',active_pid=__import__('os').getpid(),**summary));print(json.dumps(summary,ensure_ascii=False))
