"""Seal the completed four-fund documentary cache under unchanged review rules.

Reads all 228 cached PDFs again. No network, new models or edits to v1 evidence.
"""
from pathlib import Path
import datetime as dt
import hashlib
import json
import re
import sys

ROOT=Path('./analysis_outputs/h3_alpha_20261104')
HISTORY=ROOT/'fund_history'
OUT=ROOT/'research_tr_evidence_arsagera'
sys.path.insert(0,str(HISTORY/'vendor'))
import pandas as pd
from pypdf import PdfReader

FUNDS={
 '403':('0363-75409054','арсагерафонд акций'),
 '404':('0364-75409132','арсагерафонд смешанных инвестиций'),
 '481':('0439-75408664','арсагераакции64'),
 '3435':('2721','арсагерафонд облигаций кр155')}
SOURCE_SHA='556a4870232ac91281493652a1a8bdb26acd7eb46925e872a46fab8a8cde7084'
MANIFEST_SHA='fee33f50201d4cf93ee4cee7297e3d355f4f0f107150f15903d0d5649f0e7327'
OLD_SHA='cb3054b3a1305a81d31dc1078caeef4d24489a821e6d24dd9625907b5190e4fd'

def sha(p):
 return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def norm(s):
 return re.sub(r'[^а-яa-z0-9]','',s.lower().replace('ё','е'))

def main():
 if sys.platform!='win32':raise SystemExit('PC-only cached documentary review')
 source=HISTORY/'official_monthly_returns'
 candidate=source/'published_monthly_return_candidates.parquet'
 manifest=source/'source_manifest.json'
 assert sha(candidate)==SOURCE_SHA and sha(manifest)==MANIFEST_SHA
 checkpoint=json.loads((source/'checkpoint.json').read_text(encoding='utf-8'))
 assert checkpoint['stage']=='COMPLETED' and checkpoint['reports']==228 and checkpoint['errors']==0
 c=pd.read_parquet(candidate);c['fund_id']=c.fund_id.astype(str)
 assert len(c)==228 and set(c.fund_id)==set(FUNDS) and not c.duplicated(['fund_id','month']).any()
 expected=list(pd.period_range('2022-01','2026-09',freq='M').astype(str))
 for fund in FUNDS:assert sorted(c.loc[c.fund_id.eq(fund),'month'])==expected
 sources=json.loads(manifest.read_text(encoding='utf-8'))
 lookup={(str(r['fund_id']),r['month']):r for r in sources}
 assert len(lookup)==228
 old=OUT/'cached_monthly_documentary_snapshot_20261104.json';assert sha(old)==OLD_SHA
 nav_path=HISTORY/'official_sources/arsagera_official_unit_nav.parquet'
 nav=pd.read_parquet(nav_path);nav['fund_id']=nav.fund_id.astype(str)
 nav['month']=pd.to_datetime(nav.valuation_date).dt.to_period('M').astype(str)
 nav=nav.sort_values('valuation_date').groupby(['fund_id','month']).tail(1)
 ni={(r.fund_id,r.month):r for r in nav.itertuples(index=False)}
 rows=[]
 for r in c.sort_values(['fund_id','month']).itertuples(index=False):
  path=Path(r.raw_file);before=sha(path);record=lookup[(r.fund_id,r.month)]
  assert before==r.raw_sha256==record['sha256']
  assert str(path)==str(Path(record['path'])) and r.source_url==record['url']
  pdf=PdfReader(path);found=[];all_dates=[];all_text=[]
  for pi,page in enumerate(pdf.pages,1):
   text=page.extract_text(extraction_mode='layout') or '';all_text.append(text)
   all_dates.extend(re.findall(r'\b\d{2}\.\d{2}\.\d{4}\b',text));lines=text.splitlines()
   for li,line in enumerate(lines):
    if not re.match(r'^\s*изменение\s+стоимости\s+пая\b',line,re.I):continue
    headers=[x for x in lines[max(0,li-7):li] if re.search(r'\b1\s*мес',x,re.I)]
    if not headers:continue
    header=headers[-1];mark=re.search(r'\b1\s*мес\.?',header,re.I);center=(mark.start()+mark.end())/2
    tokens=list(re.finditer(r'[-−+]?\d+(?:[ \xa0]\d{3})*(?:[,.]\d+)?\s*%',line))
    if not tokens:continue
    token=min(tokens,key=lambda t:abs((t.start()+t.end())/2-center))
    distance=abs((token.start()+token.end())/2-center)
    if distance>12:continue
    value=float(token.group().replace('%','').replace('−','-').replace(' ','').replace('\xa0','').replace(',','.'))/100
    found.append(dict(return_decimal=value,page=pi,line=li+1,header=header.strip(),
                      value_line=line.strip(),token=token.group(),column_distance=distance))
  assert len(found)==1,(r.fund_id,r.month,len(found))
  header_text='\n'.join(all_text[0].splitlines()[:8])
  assert norm(FUNDS[r.fund_id][1]) in norm(header_text),(r.fund_id,r.month,'fund header mismatch',header_text)
  end_date=pd.Period(r.month,freq='M').end_time.strftime('%d.%m.%Y')
  assert end_date in all_dates,(r.fund_id,r.month,'month-end date missing')
  assert sha(path)==before
  value=found[0]['return_decimal'];assert abs(value-float(r.reported_monthly_return))<1e-15
  cur=ni[(r.fund_id,r.month)];prev=ni[(r.fund_id,str(pd.Period(r.month,freq='M')-1))]
  start=str(pd.Timestamp(prev.valuation_date).date());end=str(pd.Timestamp(cur.valuation_date).date())
  assert start==str(pd.Timestamp(r.start_valuation_date).date()) and end==str(pd.Timestamp(r.end_valuation_date).date())
  ratio=float(cur.unit_value/prev.unit_value-1)
  rows.append(dict(fund_id=r.fund_id,registration_number=FUNDS[r.fund_id][0],month=r.month,
                   source_path=str(path),source_url=r.source_url,source_sha256=before,
                   date_in_pdf_verified=end_date,header_fund_name_verified=True,
                   actual_start_valuation_date=start,actual_end_valuation_date=end,
                   raw_nav_ratio_return=ratio,absolute_difference_from_nav_ratio=abs(value-ratio),**found[0]))
 assert sha(candidate)==SOURCE_SHA and sha(manifest)==MANIFEST_SHA and sha(old)==OLD_SHA
 old_rows=json.loads(old.read_text(encoding='utf-8'))['rows'];new_idx={(r['fund_id'],r['month']):r for r in rows}
 for r in old_rows:
  n=new_idx[(r['fund_id'],r['month'])]
  assert all(n[k]==r[k] for k in ['source_sha256','source_url','return_decimal','actual_start_valuation_date','actual_end_valuation_date'])
 method=HISTORY/'official_sources/arsagera_return_method.html'
 cross=HISTORY/'v3_action_aware/fund_series_crosswalk.parquet'
 payload=dict(version='v2_completed_four_fund_cache',created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
  scope='All228 completed cached PDFs re-read; no network, regressions or input changes',
  source_candidate_path=str(candidate),source_candidate_sha256=SOURCE_SHA,
  source_manifest_path=str(manifest),source_manifest_sha256=MANIFEST_SHA,
  previous_snapshot_path=str(old),previous_snapshot_sha256=OLD_SHA,previous55_exactly_preserved=True,
  method_source_path=str(method),method_sha256=sha(method),
  formula_image_path=str(OUT/'return_formula.png'),formula_image_sha256=sha(OUT/'return_formula.png'),
  unit_nav_source_path=str(nav_path),unit_nav_sha256=sha(nav_path),
  crosswalk_path=str(cross),crosswalk_sha256=sha(cross),count=len(rows),rows=rows,errors=[])
 target=OUT/'cached_monthly_documentary_snapshot_20261104_v2.json'
 if target.exists():raise SystemExit('v2 already exists; preserve it')
 target.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(dict(path=str(target),sha256=sha(target),rows=len(rows),funds=sorted(FUNDS),
                       first='2022-01',last='2026-09',old55_preserved=True,
                       max_ratio_difference=max(r['absolute_difference_from_nav_ratio'] for r in rows)),ensure_ascii=False))

if __name__=='__main__':main()
