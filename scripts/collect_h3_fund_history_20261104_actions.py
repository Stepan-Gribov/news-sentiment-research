"""Parse saved public payout/identity evidence. Never infer absence from an empty section."""
import pathlib,json,re,hashlib,datetime as dt
import pandas as pd
from bs4 import BeautifulSoup
R=pathlib.Path('./analysis_outputs/h3_alpha_20261104/fund_history')
O=R/'action_evidence';O.mkdir(exist_ok=True)
stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S');events=[];profiles=[];mentions=[]
def number(v):
 try:return float(re.sub(r'[^\d.,\-]','',v).replace(',','.'))
 except Exception:return None
for p in sorted((R/'collected_raw').glob('profile_*.html')):
 fund=p.stem.split('_',1)[1];body=p.read_bytes();h=hashlib.sha256(body).hexdigest();soup=BeautifulSoup(body,'html.parser');text=soup.get_text(' ',strip=True);count=0;tables=0
 for table in soup.find_all('table'):
  rows=table.find_all('tr')
  if not rows:continue
  headers=[x.get_text(' ',strip=True) for x in rows[0].find_all(['th','td'])];head=' '.join(headers)
  if not ('Сумма' in head and ('Фиксация' in head or 'Срок выплаты' in head)):continue
  tables+=1;currency='RUB' if 'RUB' in head or 'руб' in head.lower() else 'USD' if 'USD' in head else 'unknown'
  for rownum,tr in enumerate(rows[1:],1):
   vals=[x.get_text(' ',strip=True) for x in tr.find_all('td')]
   if len(vals)<4:continue
   amount=number(vals[3])
   if amount is None:continue
   key=fund+'\t'+'\t'.join(vals[:4])+'\t'+currency
   events.append(dict(event_id=hashlib.sha256(key.encode()).hexdigest(),fund_id=fund,period_label=vals[0],payment_date_text=vals[1],record_date_text=vals[2],cash_per_unit=amount,currency=currency,payment_date=pd.to_datetime(vals[1],format='%d.%m.%Y',errors='coerce'),record_date=pd.to_datetime(vals[2],format='%d.%m.%Y',errors='coerce'),ex_date=None,ex_date_verified=False,history_completeness_verified=False,source_path=str(p),source_sha256=h,source_table_header=head,table_row1=rownum+1,source_url='https://investfunds.ru/funds/'+fund+'/'))
   count+=1
 for x in re.finditer(r'дроблен|конвертац|выделен|присоединен',text,re.I):mentions.append(dict(fund_id=fund,keyword=x.group(),context=text[max(0,x.start()-80):x.end()+180],source_path=str(p),source_sha256=h))
 profiles.append(dict(fund_id=fund,income_section_present='Выплаты дохода' in text,income_tables=tables,known_cash_events=count,distributions_complete=False,split_history_complete=False,source_path=str(p),source_sha256=h))
ev=pd.DataFrame(events)
if not ev.empty:ev=ev.drop_duplicates('event_id');ev.to_parquet(O/('public_cash_events_'+stamp+'.parquet'),index=False,compression='zstd')
pd.DataFrame(profiles).to_parquet(O/('profile_action_status_'+stamp+'.parquet'),index=False,compression='zstd')
(O/('corporate_action_mentions_'+stamp+'.json')).write_text(json.dumps(mentions,ensure_ascii=False,indent=2),encoding='utf-8')
summary=dict(snapshot=stamp,profile_files=len(profiles),cash_event_rows=len(ev),funds_with_cash_events=int(ev.fund_id.nunique()) if not ev.empty else 0,explicit_ex_dates_verified=0,complete_cash_histories_verified=0,source='Saved public Investfunds pages, secondary',policy='Known distributions retained; payment/record dates are not silently treated as ex dates; no statement that unknown payout/split histories are empty',cash_file=str(O/('public_cash_events_'+stamp+'.parquet')) if not ev.empty else None,status_file=str(O/('profile_action_status_'+stamp+'.parquet')),corporate_action_mentions=len(mentions))
(O/('summary_'+stamp+'.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False))
