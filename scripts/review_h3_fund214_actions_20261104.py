"""Seal fund-214-only action evidence. Never change worker histories or releases.

Inputs are the separately preserved official pages/PDFs and fund-specific
comparisons under fund_action_review_214. Evidence collection steps are saved
there under code/. This review makes no fund-regression estimates.
"""
from pathlib import Path
import hashlib
import html
import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT=Path('./analysis_outputs/h3_alpha_20261104/fund_action_review_214')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def js(p,d):
    Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n',encoding='utf-8')


def text_html(p):
    return re.sub(r'\s+',' ',html.unescape(re.sub('<[^>]+>',' ',Path(p).read_text(encoding='utf-8')))).strip()


def main():
    checks=[]
    def check(name,condition):
        checks.append(dict(name=name,passed=bool(condition)))
    numeric=json.loads((ROOT/'numerical_comparison.json').read_text(encoding='utf-8'))
    x=json.loads((ROOT/'evidence/worker_crosswalk214.json').read_text(encoding='utf-8'))[0]
    params=text_html(ROOT/'raw/wim_fund_parameters.html')
    chart=text_html(ROOT/'raw/wim_split_chart.html')
    check('worker_identity_exact',x['fund_id']=='214' and x['registration_number']=='0202-72729965' and x['identity_unambiguous'])
    check('official_identity_same_registration','0202-72729965' in params and 'Накопительный резерв' in params)
    check('official_manager_WIM_Savings','ВИМ Сбережения' in params)
    reused=json.loads((ROOT/'reused_official_sources.json').read_text(encoding='utf-8'))
    for z in reused:
        check('reused_official_source_preserved:'+Path(z['file']).name,
              sha(z['file'])==z['sha256']==sha(ROOT/'raw'/Path(z['file']).name))
    pattern=r'14\.08\.2026.*?дробления.*?1:100.*?13\.08\.2026.*?разделены на 100'
    match=re.search(pattern,chart)
    check('official_split_and_retroactive_statement',match is not None)
    statement=match.group(0) if match else ''
    js(ROOT/'evidence/official_split_statement.json',dict(source_url='https://www.wimsavings.ru/ru/pif/chart/',
        source_path=str(ROOT/'raw/wim_split_chart.html'),source_sha256=sha(ROOT/'raw/wim_split_chart.html'),text=statement))
    # Original signed unit-NAV certificates independently establish denomination.
    worker=pd.read_parquet(ROOT/'evidence/worker_fund214_unit_values.parquet').set_index('value_date')
    certificates=[]
    for date,stem,divisor in [('2022-07-29','unit_certificate_29072022',100),
                             ('2026-07-31','unit_certificate_31072026',100),
                             ('2026-08-31','unit_certificate_31082026',1)]:
        pages=json.loads((ROOT/'pdf_text'/(stem+'.json')).read_text(encoding='utf-8'))
        found=[]
        for page in pages:
            t=re.sub(r'\s+',' ',page['text'])
            for m in re.finditer(r'расчетная стоимость инвестиционного пая паевого инвестиционного фонда\)\s*16\s*([\d ]+[.,]\d{2})',t,re.I):
                value=float(m[1].replace(' ','').replace(',','.'))
                found.append(dict(page_number=page['page_number'],value=value,excerpt=t[max(0,m.start()-80):m.end()+30]))
        check('certificate_unique_value:'+date,len(set(z['value'] for z in found))==1)
        value=found[0]['value']
        w=float(worker.loc[pd.Timestamp(date),'unit_nav'])
        check('certificate_scale_matches_worker:'+date,abs(value/divisor-w)<1e-10)
        certificates.append(dict(date=date,pdf_path=str(ROOT/'raw'/(stem+'.pdf')),sha256=sha(ROOT/'raw'/(stem+'.pdf')),
            original_unit_NAV=value,comparison_divisor=divisor,worker_unit_NAV=w,source_rows=found))
    js(ROOT/'evidence/original_unit_NAV_certificate_comparison.json',certificates)
    check('comparison_all_worker_dates_present',numeric['date_matches']==numeric['worker_observations']==3598)
    check('all_post_split_values_match',numeric['periods']['POST_SPLIT']['n']==12 and numeric['periods']['POST_SPLIT']['matches']==12)
    check('source_differences_disclosed',numeric['rounded_value_matches']==3572 and len(numeric['mismatches'])==26)
    check('no_second_split_correction',numeric['apply_additional_split_adjustment'] is False)
    income=json.loads((ROOT/'monthly_accrued_income_evidence.json').read_text(encoding='utf-8'))
    periods=sorted(z['month'] for z in income)
    check('income_reporting_contiguous_51_months',periods==pd.period_range('2022-07','2026-09',freq='M').astype(str).tolist())
    check('income_report_identity_all_verified',all(z['identity_verified'] for z in income))
    check('all_explicit_row07_values_zero',all(z['row07_candidates']>=1 and z['accrued_unit_income']==0 and
        all(q['value']==0 for q in z['excerpts']) for z in income))
    for z in income:
        check('income_PDF_hash:'+z['month'],sha(z['path'])==z['sha256'])
    inputs=json.loads((ROOT/'worker_inputs_before.json').read_text(encoding='utf-8'))
    for z in inputs:
        check('worker_input_unchanged:'+Path(z['path']).name,sha(z['path'])==z['sha256'])
    logs=[json.loads(x) for x in (ROOT/'fetch_log.jsonl').read_text(encoding='utf-8').splitlines()]
    for z in logs:
        check('official_download_hash:'+Path(z['path']).name,sha(z['path'])==z['sha256'])
    check('new_official_HTTPS_TLS_verification',all(z['url'].startswith('https://www.wimsavings.ru/') and z['tls_verified'] for z in logs))
    qa=dict(status='PASS' if all(x['passed'] for x in checks) else 'FAIL',checks_pass=sum(x['passed'] for x in checks),
        checks_fail=sum(not x['passed'] for x in checks),checks=checks)
    js(ROOT/'qa_review.json',qa)
    assert qa['checks_fail']==0,qa
    out=dict(status='COMPLETE_KNOWN_SPLIT_VERIFIED_DISTRIBUTION_HISTORY_LIMITED',reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
        evidence_root=str(ROOT),fund_id='214',identity=dict(name=x['fund_name'],registration_number='0202-72729965',
            manager='ООО ВИМ Сбережения',worker_crosswalk_registration_match=True,
            worker_dated_identity_observed_from=x['identity_observed_from'],worker_dated_identity_observed_to=x['identity_observed_to'],
            official_registration_date='2004-04-13',
            note='Do not confuse ООО ВИМ Сбережения with АО ВИМ Инвестиции or transfer this action to other fund IDs.'),
        split=dict(status='KNOWN_EVENT_CONFIRMED_ALREADY_BACKADJUSTED',effective_date='2026-08-14',
            units_after_per_unit_before=100,pre_event_unit_NAV_divisor=100,backadjusted_through_inclusive='2026-08-13',
            worker_history_already_backadjusted=True,official_chart_already_backadjusted=True,
            apply_additional_adjustment_to_worker=False,certificate_comparisons=certificates,
            official_source='https://www.wimsavings.ru/ru/pif/chart/',
            all_earlier_split_and_action_completeness='NOT_ESTABLISHED_BY_THIS_KNOWN_EVENT'),
        numerical_reconciliation={k:v for k,v in numeric.items() if k not in ['mismatches']},
        distribution_review=dict(status='EXPLICIT_ZERO_ACCRUAL_202207_202609_FULL_CASH_HISTORY_UNVERIFIED',
            official_monthly_growth_reports=51,period_from='2022-07',period_to='2026-09',
            explicit_zero_accrued_unit_income_reports=51,nonzero_accrued_unit_income_reports=0,
            earlier_worker_period_not_covered_by_these_reports='2012-01 through 2022-06',
            payout_cash_event_history_complete=False,ex_dates_verified=False,
            all_worker_history_distributions_verified=False,automatic_TR_VERIFIED_promotion=False,
            semantic_limit='Row 07 is accrued income on fund units, not a complete dated cash-payment or ex-date ledger. All 51 downloaded official reports explicitly show zero in every occurrence of this row; zero was not inferred from a missing row/list. This does not prove the absence of cash distributions throughout the full 2012-2026 history.',
            other_documents_reviewed='Five consolidated rule versions 20-24, amendments, five dated KIDs and official prior-name list. No blanket historical no-distribution assertion was inferred from the absence of a clause.'),
        downstream_instructions=['Keep worker observed unit NAV levels unchanged for this known split.',
            'For comparisons with original signed pre-14.08.2026 unit-NAV certificates, divide those original per-unit certificate values by 100 once; this is not an instruction to rescale total fund NAV.',
            'Do not divide the worker or official interactive-chart pre-split series by 100 again.',
            'Retain the 26 official-chart versus secondary-source differences (three calendar-month-end dates) as a separate unresolved source-quality issue; do not override worker values automatically.',
            'Do not promote the entire fund history to TR_VERIFIED or all-actions-complete from this single split or the partial zero-accrual evidence.'],
        worker_inputs_preserved=inputs,reused_official_source_provenance=reused,
        worker_release_unchanged=True,regressions_run=False,qa=qa,
        sources=dict(chart='https://www.wimsavings.ru/ru/pif/chart/',parameters='https://www.wimsavings.ru/ru/pif/parametres/basic-parameters/',
            rules='https://www.wimsavings.ru/ru/pif/parametres/rules/',financial_reports='https://www.wimsavings.ru/ru/pif/parametres/reporting/',
            KID='https://www.wimsavings.ru/ru/pif/kid/'))
    js(ROOT/'review.json',out)
    (ROOT/'README.md').write_text('''# Fund 214 action review — 04.11.2026

Confirmed by registration 0202-72729965: ВИМ — Накопительный резерв, ООО ВИМ Сбережения. The known 14.08.2026 split creates 100 units from one old unit. WIM explicitly states that its historical chart through 13.08.2026 is already divided by 100. Original signed unit-NAV certificates independently match the worker series after a single /100 comparison adjustment before the split, and directly after it. The worker and official chart series require NO second split adjustment.

Numerical comparison: 3,598 dated worker observations found on the official chart, 3,572 within half a kopeck. All 12 post-split observations match. Keep the remaining 26 source differences (including 2012-05-31, 2026-06-30 and 2026-07-31 month ends) unresolved; no worker values were changed. The signed 31.07.2026 unit value is 3,415.92, matching worker 34.1592 after /100. The signed 31.08.2026 value and worker value are both 33.66. Official chart display values have less precision and occasional additional differences.

Distribution evidence: all 51 official monthly growth reports from July 2022 through September 2026 explicitly show zero accrued fund-unit income (row 07), including every repeated occurrence of that row. This is direct zero-value evidence rather than an inference from empty event lists. Accrued income is not a complete cash-payment/ex-date ledger, and earlier 2012–June 2022 is outside these reports. Five consolidated rule versions and five dated KIDs were reviewed, but no blanket no-distribution assertion was inferred from missing clauses. The full history remains unverified for total-return distributions and other-action completeness; no automatic TR_VERIFIED promotion.

Raw official pages, PDFs, URL/retrieval-time/SHA log, page-numbered PDF text, exact numerical comparisons and the before/after worker hashes are preserved here. Collection and review scripts are in code/. The private pypdf 6.4.0 tool is isolated under tools/. All writes are inside this new evidence folder. Existing worker histories, factors, protected bulk resources and regressions were not modified.
''',encoding='utf-8')
    files={str(p.relative_to(ROOT)).replace('\\','/'):dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
        for p in ROOT.rglob('*') if p.is_file() and 'tools' not in p.relative_to(ROOT).parts and p.name!='ready_manifest.json'}
    js(ROOT/'ready_manifest.json',dict(status=out['status'],review_path=str(ROOT/'review.json'),review_sha256=sha(ROOT/'review.json'),
        sealed_at_utc=datetime.now(timezone.utc).isoformat(),known_split_already_adjusted=True,
        full_distribution_history_verified=False,files=files,qa_path=str(ROOT/'qa_review.json'),qa_sha256=sha(ROOT/'qa_review.json')))
    print(json.dumps(dict(status=out['status'],known_split_already_adjusted=True,apply_additional_adjustment=False,
        explicit_zero_accrual_months=51,full_cash_history_verified=False,qa_pass=qa['checks_pass'],qa_fail=qa['checks_fail']),ensure_ascii=False))


if __name__=='__main__':
    main()
