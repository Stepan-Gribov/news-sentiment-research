"""Review published return semantics on saved fund-214 evidence, without refits.

All writes go to the new published_return_review subdirectory. Parent evidence
and worker releases are verified by hash and never changed. Method-only HTTP
downloads are preserved separately; the 51 financial reports are not redownloaded.
"""
from pathlib import Path
import hashlib, html, json, re
from datetime import datetime, timezone
import numpy as np
import pandas as pd

PARENT=Path('./analysis_outputs/h3_alpha_20261104/fund_action_review_214')
OUT=PARENT/'published_return_review'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,d): Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n',encoding='utf-8')
def flat(s): return re.sub(r'\s+',' ',s).strip()
def number(s): return float(s.replace(' ','').replace('\xa0','').replace(',','.').replace('−','-'))

def main():
    OUT.mkdir(exist_ok=True)
    seal=json.loads((PARENT/'ready_manifest.json').read_text(encoding='utf-8'))
    checks=[]
    def check(k,v): checks.append(dict(name=k,passed=bool(v)))
    check('parent_sealed_files_match_before',all(sha(x['path'])==x['sha256'] for x in seal['files'].values()))
    reports=json.loads((PARENT/'monthly_accrued_income_evidence.json').read_text(encoding='utf-8'))
    chart=pd.read_parquet(PARENT/'evidence/official_wim_chart_points.parquet').set_index('value_date').official_chart_value
    cert=json.loads((PARENT/'evidence/original_unit_NAV_certificate_comparison.json').read_text(encoding='utf-8'))
    cert_by_month={str(pd.Timestamp(x['date']).to_period('M')):x for x in cert}
    worker_path=next(x['path'] for x in json.loads((PARENT/'worker_inputs_before.json').read_text()) if Path(x['path']).name=='fund_monthly_returns.parquet')
    w=pd.read_parquet(worker_path);w=w[w.fund_id.astype(str).eq('214')].set_index('month')
    mismatches=set(json.loads((PARENT/'numerical_comparison.json').read_text(encoding='utf-8'))['mismatching_month_end_dates'])
    data=[];excerpts=[]
    for report in sorted(reports,key=lambda x:x['month']):
        p=Path(report['path']);check('unchanged_report:'+report['month'],sha(p)==report['sha256'])
        pages=json.loads((PARENT/'pdf_text'/(p.stem+'.json')).read_text(encoding='utf-8'))
        t=flat(' '.join(z['text'] for z in pages));values={};locations={}
        anchors={1:r'результате сделок с имуществом',2:r'результате изменения справедливой стоимости имущества',
            3:r'Доход по имуществу, принадлежащему',4:r'Оплата по договорам',5:r'Сумма начисленных вознаграждений',
            6:r'Сумма начисленных расходов, связанных с доверительным управлением',7:r'Сумма начисленных дивидендов',
            8:r'Прочие доходы',9:r'Прочие расходы',10:r'имущества в результате размещения',
            11:r'имущества в результате выкупа',12:r'имущества в результате обмена инвестиционных паев других',
            13:r'имущества в результате обмена инвестиционных паев данного',
            14:r'Прирост .{0,30}стоимости имущества, принадлежащего'}
        for n in range(1,15):
            # Metric labels distinguish row codes from subrows and grouped money.
            match=re.search(anchors[n]+r'.{0,800}?\s'+f'{n:02}'+r'\s+([−-]?[\d \xa0]+[.,]\d{2})',t,re.I)
            values[n]=None if match is None else number(match[1])
            if match: locations[n]=t[max(0,match.start()-30):match.end()+20]
        check('main_table_all_14_rows:'+report['month'],all(v is not None for v in values.values()))
        expected=(values[1]+values[2]+values[3]-values[4]-values[5]-values[6]-values[7]+values[8]-values[9]
                  +values[10]+values[11]+values[12]+values[13])
        check('row14_accounting_identity:'+report['month'],abs(expected-values[14])<.03)
        check('row07_matches_prior_review:'+report['month'],values[7]==report['accrued_unit_income'])
        unit_metric_found=bool(re.search(r'расчетн(?:ая|ой) стоимость?и? инвестиционного пая|прирост.{0,60}расчетной стоимости',t,re.I))
        check('not_a_unit_NAV_return_table:'+report['month'],not unit_metric_found)
        month=pd.Period(report['month'],freq='M');start=(month-1).end_time.normalize();end=month.end_time.normalize()
        a,b=chart.get(start,np.nan),chart.get(end,np.nan)
        ca,cb=cert_by_month.get(str(month-1)),cert_by_month.get(str(month))
        sa=None if ca is None else ca['original_unit_NAV']/ca['comparison_divisor']
        sb=None if cb is None else cb['original_unit_NAV']/cb['comparison_divisor']
        ww=w.loc[str(month)] if str(month) in w.index else None
        row=dict(fund_id='214',month=str(month),report_date=report['report_date'],report_url=report['url'],report_path=str(p),
            report_sha256=report['sha256'],reported_property_change_RUB_row14=values[14],
            accounting_identity_residual_RUB=expected-values[14],accounting_identity_reconciled=abs(expected-values[14])<.03,
            reported_accrued_unit_income_RUB_row07=values[7],reported_net_unit_issues_and_redemptions_RUB=sum(values[i] for i in [10,11,12,13]),
            published_unit_return_pct_in_these_reports=None,unit_NAV_start_in_these_reports=None,unit_NAV_end_in_these_reports=None,
            report_missing_reason='ACCOUNTING_PROPERTY_CHANGE_RUB_NOT_UNIT_RETURN_OR_UNIT_NAV_ENDPOINT_REPORT',
            calendar_period_start=start,calendar_period_end=end,official_chart_published_date_start=start,
            official_chart_published_date_end=end,official_chart_unit_NAV_start=a,official_chart_unit_NAV_end=b,
            derived_chart_unit_NAV_return_m=b/a-1 if pd.notna(a) and pd.notna(b) else np.nan,
            chart_return_is_computed_not_a_literal_published_monthly_return=True,
            chart_actual_valuation_dates_verified=False,
            known_month_end_source_difference=start.strftime('%Y-%m-%d') in mismatches or end.strftime('%Y-%m-%d') in mismatches,
            signed_certificate_unit_NAV_start_adjusted=sa,signed_certificate_unit_NAV_end_adjusted=sb,
            signed_start_valuation_date=None if ca is None else ca['date'],signed_end_valuation_date=None if cb is None else cb['date'],
            derived_signed_certificate_unit_NAV_return_m=None if sa is None or sb is None else sb/sa-1,
            worker_unit_NAV_start=None if ww is None or pd.isna(ww.previous_unit_nav) else float(ww.previous_unit_nav),
            worker_unit_NAV_end=None if ww is None or pd.isna(ww.unit_nav) else float(ww.unit_nav),
            worker_start_valuation_date=None if ww is None or pd.isna(ww.previous_value_date) else ww.previous_value_date,
            worker_end_valuation_date=None if ww is None or pd.isna(ww.value_date) else ww.value_date,
            worker_observed_return_m=None if ww is None or pd.isna(ww.fund_return_observed) else float(ww.fund_return_observed),
            cash_distributions_RUB=None,ex_dates=None,
            separate_distribution_adjustment_declared=False,TR_VERIFIED_by_this_review=False,final_TR_rule_owner='research')
        data.append(row)
        excerpts.append(dict(month=str(month),source_url=report['url'],source_pdf_sha256=report['sha256'],
            property_change_row14=locations[14],accrued_income_row07=locations[7]))
    d=pd.DataFrame(data)
    check('51_unique_contiguous_months',d.month.tolist()==pd.period_range('2022-07','2026-09',freq='M').astype(str).tolist())
    check('missing_published_returns_remain_NA',d.published_unit_return_pct_in_these_reports.isna().all())
    check('all_chart_endpoint_candidates_exist',d.derived_chart_unit_NAV_return_m.notna().all())
    formula=' '.join(z['text'] for z in json.loads((OUT/'formula_text.json').read_text(encoding='utf-8')))
    check('explicit_NAV_growth_formula_in_official_PDF','(Х - Y) / Y x 100' in flat(formula))
    check('explicit_end_start_unit_NAV_definition','на конец периода' in formula and 'на начало периода' in formula)
    dynamics=flat(html.unescape(re.sub('<[^>]+>',' ',(OUT/'raw/wim_unit_dynamics.html').read_text(encoding='utf-8'))))
    check('current_fund_page_links_formula','/upload/Formula_SberPlus.pdf' in (OUT/'raw/wim_unit_dynamics.html').read_text(encoding='utf-8'))
    published=float(re.search(r'1 месяц\s+\d+\s+([−-]?[\d.,]+)%',dynamics)[1])
    period=re.search(r'5\s*[—–-]\s*за период\s+(\d{2}\.\d{2}\.\d{4})\s*[—–-]\s*(\d{2}\.\d{2}\.\d{4})',dynamics)
    start,end=[pd.to_datetime(x,format='%d.%m.%Y') for x in [period[1],period[2]]]
    computed=(chart.loc[end]/chart.loc[start]-1)*100
    check('literal_published_1month_value_matches_NAV_formula',abs(published-computed)<=.0050001)
    js_code=(OUT/'raw/wim_chart_page.js').read_text(encoding='utf-8')
    check('frontend_formula_consistent','(curValue - firstValue) / firstValue * 100' in js_code and '(maxPoint / minPoint - 1) * 100' in js_code)
    d.to_parquet(OUT/'published_return_candidates.parquet',index=False)
    d.to_csv(OUT/'published_return_candidates.csv',index=False)
    js(OUT/'exact_report_metric_excerpts.json',excerpts)
    js(OUT/'methodology_evidence.json',dict(title='Порядок определения прироста расчетной стоимости инвестиционного пая паевого инвестиционного фонда (доходности)',
        url='https://www.wimsavings.ru/upload/Formula_SberPlus.pdf',page=1,
        formula_pct='(X-Y)/Y*100',X='calculated fund-unit NAV at period end',Y='calculated fund-unit NAV at period start',
        applicability='Currently linked from the official fund-214 dynamics page; PDF retains predecessor manager ООО Сбережения Плюс naming.',
        distribution_term_or_reinvestment_not_declared=True,
        frontend_function_calcPercent='return (curValue - firstValue) / firstValue * 100;',
        literal_published_one_month_example=dict(label='1 месяц',start=str(start.date()),end=str(end.date()),published_percent=published,
            start_unit_NAV=float(chart.loc[start]),end_unit_NAV=float(chart.loc[end]),recomputed_percent=float(computed),
            calendar_month_return=False)))
    check('parent_sealed_files_match_after',all(sha(x['path'])==x['sha256'] for x in seal['files'].values()))
    failed=[x for x in checks if not x['passed']]
    core_failed=[x for x in failed if not x['name'].startswith('row14_accounting_identity:')]
    qa=dict(status='PASS_CORE_SEMANTICS_WITH_UNRECONCILED_ACCOUNTING_EXTRACTION' if failed and not core_failed else 'FAIL' if core_failed else 'PASS',
        checks_pass=sum(x['passed'] for x in checks),checks_fail=len(failed),core_semantic_checks_fail=len(core_failed),
        unresolved_accounting_months=d.loc[~d.accounting_identity_reconciled,'month'].tolist(),checks=checks)
    js(OUT/'qa.json',qa)
    assert not core_failed,[x['name'] for x in core_failed][:8]
    report=dict(status='COMPLETE_OFFICIAL_NAV_GROWTH_DEFINITION_VERIFIED_NO_DECLARED_DISTRIBUTION_ADJUSTED_SERIES',
        reviewed_at_utc=datetime.now(timezone.utc).isoformat(),fund_id='214',registration_number='0202-72729965',output_root=str(OUT),
        inputs='Same 51 already-preserved official monthly property-growth reports; no report redownload and no sealed-history changes.',
        published_accounting_metric=dict(name='Прирост «+» (уменьшение «-») стоимости имущества, принадлежащего акционерному инвестиционному фонду (составляющего паевой инвестиционный фонд)',
            row='14',units='RUB absolute amount, not percent',source='https://www.wimsavings.ru/ru/pif/parametres/reporting/',
            arithmetic_identity='01+02+03-04-05-06-07+08-09+10+11+12+13',arithmetic_verified_reports=int(d.accounting_identity_reconciled.sum()),
            arithmetic_unreconciled_months=d.loc[~d.accounting_identity_reconciled,'month'].tolist(),
            arithmetic_scope='Accounting extraction cross-check only. Unreconciled RUB rows are flagged and are not used to establish a unit return or TR definition.',
            unit_issues_redemptions_included=True,published_unit_returns_found_in_these_reports=0,
            published_unit_NAV_endpoints_found_in_these_reports=0),
        accrued_income_metric=dict(name='Сумма начисленных дивидендов по акциям акционерного инвестиционного фонда (дохода по инвестиционным паям паевого инвестиционного фонда)',
            row='07',units='RUB accrued income',explicit_zero_reports=51,cash_payment_ledger_equivalent=False),
        official_published_return_definition=dict(url='https://www.wimsavings.ru/upload/Formula_SberPlus.pdf',page=1,
            linked_from='https://www.wimsavings.ru/ru/pif/strategy/dynamics/',formula_percent='(unit_NAV_end/unit_NAV_start-1)*100',
            defined_metric='Growth in calculated unit NAV, as explicitly defined by the manager',
            explicit_distribution_addback_or_reinvestment_formula=False,declared_total_return_wealth_index_found=False,
            cash_ledger_absence_used_as_sole_exclusion=False,
            interpretation='Underlying portfolio income may already enter unit NAV; a separate distribution to fund-unit holders requires its own treatment. Price growth can equal economic total return if no fund-unit distributions occurred, but that is a separate period-specific determination for research.'),
        current_literal_one_month_example=dict(start=str(start.date()),end=str(end.date()),published_percent=published,
            recomputed_NAV_growth_percent=float(computed),is_calendar_month=False),
        candidate_table=dict(parquet=str(OUT/'published_return_candidates.parquet'),csv=str(OUT/'published_return_candidates.csv'),
            rows=51,key=['fund_id','month'],raw_report_percent_and_unit_endpoint_fields='NA: not published in these accounting reports; never zero-filled',
            chart_derived_return_column='derived_chart_unit_NAV_return_m',
            signed_certificate_derived_return_column='derived_signed_certificate_unit_NAV_return_m',
            derived_values_are_not_literal_published_monthly_returns=True),
        possible_ranges=dict(explicit_zero_accrual_reports=['2022-07','2026-09'],official_chart_adjacent_endpoint_candidates=['2022-07','2026-09'],
            worker_observed_return_overlap=['2022-07','2026-08'],
            both_original_signed_unit_NAV_certificates=[str(x) for x in d.loc[d.derived_signed_certificate_unit_NAV_return_m.notna(),'month']],
            chart_return_months_affected_by_known_EOM_source_difference=d.loc[d.known_month_end_source_difference,'month'].tolist(),
            unconditional_TR_VERIFIED_months=[],
            final_TR_rule='Research decides whether interval-specific payout evidence and genuine valuation dates suffice; code implements that rule. This review does not change it.'),
        semantic_flags=dict(ACCOUNTING_ASSET_GROWTH_NOT_UNIT_RETURN=True,OFFICIAL_METHOD_IS_UNIT_NAV_GROWTH=True,
            NO_DECLARED_DISTRIBUTION_ADJUSTED_INDEX_IN_REVIEWED_SOURCES=True,NO_AUTOMATIC_TR_PROMOTION=True,
            CASH_LEDGER_ABSENCE_ALONE_NOT_A_DISQUALIFIER=True,CHART_VALUES_ALREADY_SPLIT_ADJUSTED=True,
            CHART_CALENDAR_DATES_NOT_PROOF_OF_GENUINE_VALUATION_DATES=True,PUBLISHED_TRAILING_MONTH_NOT_CALENDAR_MONTH=True),
        qa={k:v for k,v in qa.items() if k!='checks'},parent_sealed_evidence_unchanged=True,worker_history_unchanged=True,regressions_run=False)
    js(OUT/'review.json',report)
    files={str(p.relative_to(OUT)).replace('\\','/'):dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
        for p in OUT.rglob('*') if p.is_file() and p.name!='ready_manifest.json'}
    js(OUT/'ready_manifest.json',dict(status=report['status'],files=files,qa=report['qa'],sealed_at_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(status=report['status'],candidate_rows=len(d),qa=report['qa'],published_one_month_percent=published,
        computed_one_month_percent=computed,signed_certificate_months=report['possible_ranges']['both_original_signed_unit_NAV_certificates']),ensure_ascii=False))

if __name__=='__main__': main()
