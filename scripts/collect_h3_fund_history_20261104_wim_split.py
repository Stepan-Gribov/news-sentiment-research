"""Verify known 100:1 unit split against the saved provider and official Q3 values."""
import pathlib,json,hashlib,requests,re
import pandas as pd,duckdb
from bs4 import BeautifulSoup
R=pathlib.Path('./analysis_outputs/h3_alpha_20261104/fund_history');O=R/'official_sources';p=O/'wim_Q3_2026.html';url='https://www.wimsavings.ru/ru/pif/scha/archive/index.php?action26=archive&part26=2026-3'
if not p.exists():
 x=requests.get(url,timeout=(10,35));x.raise_for_status();p.write_bytes(x.content)
soup=BeautifulSoup(p.read_bytes(),'html.parser');rows=[];day=None
for table in soup.find_all('table'):
 for tr in table.find_all('tr'):
  vals=[v.get_text(' ',strip=True) for v in tr.find_all(['td','th'])]
  for v in vals:
   if re.fullmatch(r'\d{2}\.\d{2}\.\d{4}',v):day=v
  for j,v in enumerate(vals):
   if 'Инвест.' in v and j+1<len(vals) and day:
    try:price=float(vals[j+1].replace(' ','').replace('\xa0','').replace(',','.'));rows.append(dict(date=pd.to_datetime(day,format='%d.%m.%Y'),official_adjusted_unit_value=price))
    except ValueError:pass
f=pd.DataFrame(rows).drop_duplicates();c=duckdb.connect();c.register('official',f)
parity=c.execute("SELECT date,official_adjusted_unit_value,unit_nav provider_unit_value,unit_nav-official_adjusted_unit_value difference FROM official JOIN read_parquet(?) p ON p.fund_id='214' AND p.value_date=official.date ORDER BY date",[str(R/'v1_existing_cache/unit_values_daily.parquet')]).fetchdf();assert len(parity)>20 and (parity.difference.abs()<=.011).all()
parity.to_parquet(O/'wim_split_price_comparison.parquet',index=False)
out=dict(fund_id='214',registration_number='0202-72729965',event_date='2026-08-14',new_units_per_old_unit=100,official_notice='Official fund chart states all historical unit values through2026-08-13 divided by100',source_path=str(p),source_url=url,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),notice_path=str(O/'wim_split_chart.html'),comparable_prices=len(parity),max_absolute_difference=float(parity.difference.abs().max()),comparison_tolerance=.011,known_split_already_adjusted_by_provider=True,action='Do not multiply these already rebased provider values by100 again. Preserve raw source values.',complete_split_history_verified=False)
(O/'wim_split_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False))
