"""Read-only documentary audit of already downloaded Arsagera monthly PDFs.

No network, regressions, changes to the collector, or changes to sealed inputs.
Writes a separate documentary snapshot in the research review directory.
"""
from pathlib import Path
import datetime as dt
import hashlib
import json
import re
import sys

ROOT = Path('./analysis_outputs/h3_alpha_20261104')
HISTORY = ROOT / 'fund_history'
OUT = ROOT / 'research_tr_evidence_arsagera'
sys.path.insert(0, str(HISTORY / 'vendor'))
import pandas as pd
from pypdf import PdfReader

FUNDS = {'403': '0363-75409054', '404': '0364-75409132',
         '481': '0439-75408664', '3435': '2721'}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    if sys.platform != 'win32':
        raise SystemExit('Run on research-host only.')
    source = HISTORY / 'official_monthly_returns'
    nav_path = HISTORY / 'official_sources/arsagera_official_unit_nav.parquet'
    nav = pd.read_parquet(nav_path)
    nav['month'] = pd.to_datetime(nav.valuation_date).dt.to_period('M').astype(str)
    nav = nav.sort_values('valuation_date').groupby(['fund_id', 'month']).tail(1)
    nav_idx = {(str(r.fund_id), r.month): r for r in nav.itertuples(index=False)}
    rows, errors = [], []
    for fund, registration in FUNDS.items():
        links_path = source / f'report_links_{fund}.json'
        if not links_path.exists():
            continue
        links = json.loads(links_path.read_text(encoding='utf-8'))
        # Deliberately limit this audit to the previously observed, dated range.
        for month, target in sorted(links.items()):
            if not '2022-01' <= month <= '2026-07':
                continue
            path = source / 'raw' / f'{fund}_{month}.pdf'
            if not path.exists():
                continue
            try:
                before = sha(path)
                reader = PdfReader(path)
                candidates = []
                date_tokens = []
                for pi, page in enumerate(reader.pages, 1):
                    text = page.extract_text(extraction_mode='layout') or ''
                    date_tokens += re.findall(r'\b\d{2}\.\d{2}\.\d{4}\b', text)
                    lines = text.splitlines()
                    for li, line in enumerate(lines):
                        if not re.match(r'^\s*изменение\s+стоимости\s+пая\b', line, re.I):
                            continue
                        headers = [x for x in lines[max(0, li-7):li]
                                   if re.search(r'\b1\s*мес', x, re.I)]
                        if not headers:
                            continue
                        header = headers[-1]
                        mark = re.search(r'\b1\s*мес\.?', header, re.I)
                        center = (mark.start()+mark.end())/2
                        tokens = list(re.finditer(r'[-−+]?\d+(?:[ \xa0]\d{3})*(?:[,.]\d+)?\s*%', line))
                        if not tokens:
                            continue
                        token = min(tokens, key=lambda t: abs((t.start()+t.end())/2-center))
                        distance = abs((token.start()+token.end())/2-center)
                        if distance > 12:
                            continue
                        value = float(token.group().replace('%','').replace('−','-')
                                      .replace(' ','').replace('\xa0','').replace(',','.'))/100
                        candidates.append(dict(return_decimal=value, page=pi,
                                               line=li+1, header=header.strip(),
                                               value_line=line.strip(), token=token.group(),
                                               column_distance=distance))
                if len(candidates) != 1:
                    raise ValueError(f'Expected one unique monthly field, found {len(candidates)}')
                if before != sha(path):
                    raise ValueError('Source changed during read')
                expected_date = pd.Period(month, freq='M').end_time.strftime('%d.%m.%Y')
                if expected_date not in date_tokens:
                    raise ValueError('Expected calendar month-end date absent from PDF')
                c = nav_idx.get((fund, month))
                b = nav_idx.get((fund, str(pd.Period(month, freq='M')-1)))
                ratio = float(c.unit_value/b.unit_value-1) if c is not None and b is not None else None
                row = dict(fund_id=fund, registration_number=registration, month=month,
                           source_path=str(path), source_url=target['url'], source_sha256=before,
                           date_in_pdf_verified=expected_date,
                           actual_start_valuation_date=str(pd.Timestamp(b.valuation_date).date()) if b else None,
                           actual_end_valuation_date=str(pd.Timestamp(c.valuation_date).date()) if c else None,
                           raw_nav_ratio_return=ratio, **candidates[0])
                row['absolute_difference_from_nav_ratio'] = abs(row['return_decimal']-ratio) if ratio is not None else None
                rows.append(row)
            except Exception as e:
                errors.append(dict(fund_id=fund, month=month, error=repr(e)))
    method = HISTORY / 'official_sources/arsagera_return_method.html'
    crosswalk = HISTORY / 'v3_action_aware/fund_series_crosswalk.parquet'
    payload = dict(created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                   scope='Existing cached monthly PDFs only; no network or empirical estimates',
                   method_source_path=str(method), method_sha256=sha(method),
                   formula_image_path=str(OUT/'return_formula.png'),
                   formula_image_sha256=sha(OUT/'return_formula.png'),
                   unit_nav_source_path=str(nav_path), unit_nav_sha256=sha(nav_path),
                   crosswalk_path=str(crosswalk), crosswalk_sha256=sha(crosswalk),
                   count=len(rows), rows=rows, errors=errors)
    path = OUT / 'cached_monthly_documentary_snapshot_20261104.json'
    if path.exists():
        raise SystemExit('Snapshot already exists; do not overwrite reviewed evidence.')
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(path=str(path), sha256=sha(path), rows=len(rows),
                          funds=sorted({r['fund_id'] for r in rows}), errors=errors,
                          max_nav_difference=max((r['absolute_difference_from_nav_ratio'] or 0) for r in rows)), ensure_ascii=False))

if __name__ == '__main__':
    main()
