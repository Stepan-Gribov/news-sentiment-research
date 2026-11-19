"""Bounded documentary collection for four Arsagera funds, run only on research-host.

No regressions, input rewriting, total-return classification or broad disk scan.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from urllib.parse import urljoin

ROOT = Path('./analysis_outputs/h3_alpha_20261104/research_tr_evidence_arsagera')
VENDOR = Path('./analysis_outputs/h3_alpha_20261104/fund_history/vendor')
sys.path.insert(0, str(VENDOR))
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

PAGES = {
    'fa': 'https://arsagera.ru/products/opifa_arsagera_-_fond_akcij/dokumenty_i_otchetnost_fonda/',
    'f4si': 'https://arsagera.ru/products/opifsi_arsagera_-_fond_smeshannyh_investicij/dokumenty_i_otchetnost_fonda/',
    'f64': 'https://arsagera.ru/products/ipifa_arsagera_-_akcii_64/dokumenty_i_otchetnost_fonda/',
    'fo': 'https://arsagera.ru/products/opifo_arsagera_fond_obligacij_kr_155/oficialnye_dokumenty_i_otchetnost/',
}
session = requests.Session()
session.headers['User-Agent'] = 'Academic-documentary-fund-method-review/1.0'
last_request = 0.0

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(name, value):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def fetch(url, relative):
    global last_request
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        time.sleep(max(0, 1.05 - (time.monotonic() - last_request)))
        last_request = time.monotonic()
        response = session.get(url, timeout=(10, 40))
        response.raise_for_status()
        path.write_bytes(response.content)
    return {'url': url, 'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size,
            'checked_at_utc': dt.datetime.now(dt.timezone.utc).isoformat()}

def index():
    records = []
    pages = []
    for code, url in PAGES.items():
        item = fetch(url, 'pages/' + code + '_disclosure.html')
        pages.append(dict(fund_code=code, **item))
        soup = BeautifulSoup(Path(item['path']).read_bytes(), 'html.parser')
        for a in soup.find_all('a', href=True):
            label = re.sub(r'\s+', ' ', a.get_text(' ', strip=True))
            href = urljoin(url, a['href'])
            if '/library/download/' not in href:
                continue
            if re.match(r'^Правила фонда\b', label, re.I):
                category = 'fund_rules_consolidated'
            elif re.search(r'Изменения и дополнения в Правила фонда', label, re.I):
                category = 'fund_rule_amendment'
            elif re.search(r'вступлен|дроблен|выплат[аыеу].{0,45}доход|изменени.{0,35}тип', label, re.I):
                category = 'event_or_effective_notice'
            elif re.match(r'^Сведения о паевом инвестиционном фонде', label, re.I):
                category = 'fund_identity'
            else:
                continue
            records.append(dict(fund_code=code, category=category, label=label,
                                url=href, parent_url=url, parent_sha256=item['sha256']))
    save('page_manifest.json', pages)
    save('selected_document_index.json', records)
    print(json.dumps({'documents': len(records), 'by_fund': {c: sum(r['fund_code']==c for r in records) for c in PAGES}}, ensure_ascii=False))

def documents():
    records = json.loads((ROOT / 'selected_document_index.json').read_text(encoding='utf-8'))
    output = []
    errors = []
    for r in records:
        stem = r['fund_code'] + '_' + hashlib.sha256(r['url'].encode()).hexdigest()[:12]
        try:
            item = fetch(r['url'], 'documents/' + stem + '.pdf')
            reader = PdfReader(item['path'])
            pages = [p.extract_text() or '' for p in reader.pages]
            save('text/' + stem + '.json', {'pages': pages})
            hits=[]
            for i,t in enumerate(pages,1):
                for pattern in [r'выплат.{0,30}доход',r'доход.{0,50}выплат',r'дроблен',r'вступ.{0,30}сил',r'номинал',r'0363',r'0364',r'0439',r'2721']:
                    for m in re.finditer(pattern,t,re.I|re.S):
                        hits.append(dict(page=i,pattern=pattern,text=re.sub(r'\s+',' ',t[max(0,m.start()-140):m.end()+260])))
            output.append({**r,**item,'stem':stem,'pages':len(pages),'text_path':str(ROOT/'text'/f'{stem}.json'),'hits':hits})
        except Exception as e:
            errors.append(dict(**r,error=repr(e)))
        save('document_manifest.json', output)
        save('download_errors.json', errors)
        save('checkpoint.json', {'documents_done':len(output),'errors':len(errors),'total':len(records),'last_fund':r['fund_code']})
    print(json.dumps({'documents_done':len(output),'errors':len(errors)},ensure_ascii=False))

def main():
    if sys.platform != 'win32':
        raise SystemExit('Run only on research-host; documentary outputs belong to its designated directory.')
    ROOT.mkdir(parents=True,exist_ok=True)
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['index','documents']);a=p.parse_args()
    {'index':index,'documents':documents}[a.mode]()

if __name__=='__main__':
    main()
