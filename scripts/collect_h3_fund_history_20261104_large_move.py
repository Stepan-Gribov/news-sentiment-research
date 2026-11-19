import pathlib,requests,json,hashlib,datetime as dt,duckdb
from bs4 import BeautifulSoup
R=pathlib.Path('./analysis_outputs/h3_alpha_20261104/fund_history');O=R/'official_sources'
files=[]
for year in ['2023','2024']:
 p=O/('fund4091_official_'+year+'.html');url='https://www.mconsulting.ru/ru/disclosure/funds/cpifs/balanced-global-investments/info/'+year+'/'
 if not p.exists():
  x=requests.get(url,timeout=(10,35));x.raise_for_status();p.write_bytes(x.content)
 soup=BeautifulSoup(p.read_bytes(),'html.parser');rows=[]
 for table in soup.find_all('table'):
  if 'Расчетная стоимость пая' not in table.get_text(' ',strip=True):continue
  for tr in table.find_all('tr'):
   vals=[x.get_text(' ',strip=True) for x in tr.find_all(['td','th'])]
   if any(v in ['29.12.2023','31.01.2024'] for v in vals):rows.append(vals)
 files.append(dict(url=url,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),matching_rows=rows))
c=duckdb.connect();r=str(R/'v1_existing_cache/fund_monthly_returns.parquet');wim=c.execute("SELECT count(*),count(*) FILTER(WHERE level_observed) FROM read_parquet(?) WHERE fund_id='214'",[r]).fetchone()
out=dict(fund_id='4091',month='2024-01',classification='Official manager independently reports the 37.77 -> 848.32 unit-NAV change and +2146.02%; not evidence of a split by magnitude alone',action='Keep observed return, no magnitude-based deletion; do not claim investability through suspension',sources=files,wim_fund214_present_in_current_universe=bool(wim[0]),wim_observed_levels=wim[1],wim_split_notice='Official wimsavings.ru chart discloses 100-for-1 split 2026-08-14; no automatic application to unrelated fund IDs')
(O/'large_move_fund4091_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False))
