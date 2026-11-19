"""Finish the locked finite follow-up once, reusing completed monthly fits."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

BASE=Path('./analysis_outputs/title_name_regressions_20261026_w2_v2')
sys.path.insert(0,str(BASE/'code'))
import duckdb
import numpy as np
import pandas as pd
from title_name_regressions_core import sha256_file,freeze_json,IntegrityError,NotEstimable
from hypothesis_methods_20261029 import model,holm,save,clean
from hypothesis_extended_20261029 import target_panel,window_signal,quote_pairs,cash_labels,price_adjustment,CATALOG


def stage(root,name,**items):
    payload=dict(stage=name,updated_at_utc=datetime.now(timezone.utc).isoformat(),**items)
    save(root/'checkpoint.json',payload);print(json.dumps(clean(payload)),flush=True)


def read(root,name):
    return json.loads((root/name).read_text(encoding='utf-8-sig'))


def result_tables(root,summary):
    lines=['# Ограниченное исследование H1–H3 — 29 октября 2026', '',
        'Post-hoc exploration: варианты зафиксированы до новых коэффициентов, после знакомства с DW от 26 октября. Не причинный эффект и не гарантия подтверждения гипотез.', '',
        '| Модель | Исход | N | Фонды | Компании | Бумаги | Месяцы | Период | FE | Within R² |',
        '|---|---|---:|---:|---:|---:|---:|---|---|---:|']
    for record in summary['models']:
        d=record.get('diagnostics',{})
        if d:
            lines.append(f"| {record['model']} | {record['outcome']} | {d['nobs']} | {d['funds']} | {d['companies']} | {d['securities']} | {d['calendar_months']} | {d['period_start']}–{d['period_end']} | {' + '.join(d['fixed_effects'])} | {d['within_r2']:.6f} |")
        else:lines.append(f"| {record['model']} | {record['outcome']} | NA | | | | | {record.get('reason','')} | | |")
    lines+=['','## Все новостные коэффициенты','','SE/CI условны на готовой LLM-разметке и построенном сигнале; кластеризация по фонду и эмитенту.','',
        '| Модель | Часть сигнала | beta | SE | 95% CI | p | Единицы отображения | Эффект |',
        '|---|---|---:|---:|---|---:|---|---:|']
    def fmt(value):return 'NA' if value is None else f'{value:.6g}'
    for record in summary['models']:
        for row in record.get('coefficients',[]):
            if not row['term'].startswith('S_'):continue
            lines.append(f"| {record['model']} | {row['term']} | {fmt(row['beta'])} | {fmt(row['se'])} | [{fmt(row['ci95_low'])}; {fmt(row['ci95_high'])}] | {fmt(row['p'])} | {row['display_units']} | {fmt(row['effect_display'])} |")
    lines+=['','## Все зафиксированные тесты и Holm по семействам','',
        '| Семейство | Модель | Тест | p | Holm p | Размер семейства |','|---|---|---|---:|---:|---:|']
    for test in summary['tests']:
        lines.append(f"| {test['family']} | {test['model']} | {test['test']} | {fmt(test['p'])} | {fmt(test.get('p_holm'))} | {test.get('family_size_including_unavailable','')} |")
    lines+=['','## Цена, охват и ограничения','',json.dumps(summary['price_coverage_summary'],ensure_ascii=False,indent=2),
        '', 'M3 использует log(стоимость1/стоимость0) − log(цена1/цена0) только для продолжающихся позиций. Это ценово-скорректированный рост раскрытой стоимости, не наблюдаемое количество. Выплаты, accrual/valuation differences и splits ограничивают интерпретацию.',
        '', 'M4 объединяет реальные соседние раскрытия, включая квартальные; это иной интервальный estimand, не ряд синтетических месячных holdings. S построен заново на уникальных company/start/end, с точным предшествующим окном той же длины и контролем длительности.',
        '', 'H3 alpha и actual quantity не подменяются общей стоимостью фонда, V/P или будущей сырой доходностью.',
        '',json.dumps(summary['power_MDE'],ensure_ascii=False,indent=2)]
    (root/'tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();root=args.output
    if (root/'summary.json').exists():raise IntegrityError('Completed run is immutable; do not rerun')
    monthly=read(root,'monthly_results.json')
    if sha256_file(root/'config/analysis_plan.json')!=monthly['manifest']['plan_sha256']:raise IntegrityError('Plan changed after monthly estimates')
    for name,digest in monthly['manifest']['code_hashes'].items():
        if sha256_file(root/'code'/name)!=digest:raise IntegrityError('Monthly phase code changed: '+name)
    source_names=['fund_security_panel.parquet','holdings_inventory.parquet','all_true_intervals.parquet',
        'report_quality.parquet','event_representatives.parquet','scheduled_corporate_events.parquet']
    manifest=dict(plan_sha256=monthly['manifest']['plan_sha256'],assumptions_sha256=monthly['manifest']['assumptions_sha256'],
        base_files={name:sha256_file(BASE/name) for name in source_names},
        code_hashes={name:sha256_file(root/'code'/name) for name in ['hypothesis_extended_20261029.py','run_hypothesis_extended_20261029.py','hypothesis_methods_20261029.py']},
        created_at_utc=datetime.now(timezone.utc).isoformat(),network_data_downloads=False)
    freeze_json(root/'extended_manifest.json',manifest)
    stage(root,'READING_SEALED_MONTHLY_AND_INVENTORY')
    panel=pd.read_parquet(BASE/'fund_security_panel.parquet')
    inventory=pd.read_parquet(BASE/'holdings_inventory.parquet')
    intervals=pd.read_parquet(BASE/'all_true_intervals.parquet')
    reports=pd.read_parquet(BASE/'report_quality.parquet')
    facts=pd.read_parquet(BASE/'scheduled_corporate_events.parquet')
    chosen=intervals.loc[intervals.preliminary_eligible].copy()
    _,whole,monthly_audit=target_panel(inventory,chosen,facts)
    stage(root,'READING_EXISTING_QUOTES_ONCE',whole_portfolio_rows=len(whole))
    connection=duckdb.connect(CATALOG,read_only=True,config={'threads':2,'memory_limit':'2GB'})
    try:pairs,quote_audit=quote_pairs(whole,connection);cash,cash_names=cash_labels(connection)
    finally:connection.close()
    pairs.to_parquet(root/'observed_price_pairs.parquet',index=False,compression='zstd')
    quote_files=quote_audit.pop('quote_source_files',[])
    quote_hashes={name:sha256_file(name) for name in quote_files}
    freeze_json(root/'quote_sources_manifest.json',dict(files=quote_hashes,quote_pair_sha256=sha256_file(root/'observed_price_pairs.parquet')))
    full,coverage=price_adjustment(whole,pairs,cash)
    coverage.to_csv(root/'full_portfolio_price_coverage.csv',index=False)
    uncovered=full.loc[full.weight_open.gt(1e-12)&full.price_return.isna(),['interval_id','security_id','asset_type','weight_open']].copy()
    uncovered['source_label']=uncovered.security_id.map(cash_names)
    uncovered.to_parquet(root/'uncovered_opening_positions.parquet',index=False,compression='zstd')
    covered=full[['interval_id','security_id','AT_PRICE','eligible_full_price_AT']]
    adjusted=panel.merge(covered,on=['interval_id','security_id'],how='left',validate='one_to_one')
    with_prices=panel.merge(pairs,on=['security_id','start_date','end_date'],how='left',validate='many_to_one')
    gross=reports.set_index('report_id').gross_thousand
    with_prices['value_open']=with_prices.weight_open*with_prices.opening_report_id.map(gross)
    with_prices['value_close']=with_prices.weight_close*with_prices.closing_report_id.map(gross)
    continuing=with_prices.value_open.gt(0)&with_prices.value_close.gt(0)
    usable=continuing&with_prices.price_open.gt(0)&with_prices.price_close.gt(0)
    proxy=with_prices.loc[usable].copy()
    proxy['price_adjusted_value_growth']=np.log(proxy.value_close/proxy.value_open)-np.log(proxy.price_close/proxy.price_open)
    proxy.to_parquet(root/'continuing_price_adjusted_panel.parquet',index=False,compression='zstd')
    records=list(monthly['models'])
    for identifier,outcome,frame,family in [
        ('M2B_DW_RELATIVE_PRICE_MATCHED','DW',proxy,'relative_allocation'),
        ('M3_PRICE_ADJUSTED_VALUE_GROWTH','price_adjusted_value_growth',proxy,'relative_allocation'),
        ('M5_PRICE_AT_FULL_PORTFOLIO','AT_PRICE',adjusted.loc[adjusted.eligible_full_price_AT.eq(True)],'full_portfolio_price_AT'),
        ('M5B_DW_PRICE_AT_MATCHED','DW',adjusted.loc[adjusted.eligible_full_price_AT.eq(True)],'full_portfolio_price_AT')]:
        stage(root,'ESTIMATING_FIXED_PRICE_MODELS',model=identifier,candidate_rows=len(frame))
        record,_=model(frame,identifier=identifier,family=family,outcome=outcome,relative=identifier.startswith(('M2B','M3_')))
        records.append(record);save(root/(identifier+'.json'),record)
    stage(root,'BUILDING_ACTUAL_ADJACENT_DISCLOSURE_PANEL')
    eligible=(intervals.opening_full_portfolio&intervals.closing_full_portfolio&~intervals.opening_passive_explicit&~intervals.closing_passive_explicit&
        intervals.end_date.ge('2018-01-01')&intervals.end_date.le('2026-05-29')&intervals.start_date.ge('2012-05-01'))
    actual=intervals.loc[eligible].copy()
    expanded,_,expanded_audit=target_panel(inventory,actual,facts)
    expanded_audit.update(intervals=len(actual),monthly_calendar_intervals=int(((actual.end_date.dt.year-actual.start_date.dt.year)*12+
        actual.end_date.dt.month-actual.start_date.dt.month).eq(1).sum()),min_days=int((actual.end_date-actual.start_date).dt.days.min()),
        max_days=int((actual.end_date-actual.start_date).dt.days.max()))
    events=pd.read_parquet(BASE/'event_representatives.parquet')
    events=events.loc[~events.date_inferred.eq(True)].copy()
    stage(root,'EXACT_WINDOW_NEWS_SIGNAL',target_rows=len(expanded))
    windows,signal_audit=window_signal(expanded,events)
    windows.to_parquet(root/'exact_window_company_signal.parquet',index=False,compression='zstd')
    expanded=expanded.merge(windows[['company_id','start_date','end_date','S','Tone','lag_log_news_count','n_relevant','news_status']],
        on=['company_id','start_date','end_date'],how='left',validate='many_to_one')
    expanded.to_parquet(root/'actual_disclosure_panel.parquet',index=False,compression='zstd')
    stage(root,'ESTIMATING_FIXED_ACTUAL_WINDOW_MODEL',rows=len(expanded))
    record,_=model(expanded,identifier='M4_DW_EXACT_ACTUAL_WINDOWS',family='exact_disclosure_windows',extra_controls=['log_interval_days'])
    records.append(record);save(root/'M4_DW_EXACT_ACTUAL_WINDOWS.json',record)
    tests=holm([test for record in records for test in record['tests']])
    allmodels={r['model']:r for r in records}
    def hypothesis(identifier,test):
        return next((t['p'] for t in allmodels[identifier]['tests'] if t['test']==test),None)
    # Review families are additional conservative summaries, never new regressions.
    review_tests=[dict(family='R6_primary',model='M5_PRICE_AT_FULL_PORTFOLIO',test='H1_price_adjusted',p=hypothesis('M5_PRICE_AT_FULL_PORTFOLIO','H1_directional_intersection_union')),
        dict(family='R6_primary',model='M5_PRICE_AT_FULL_PORTFOLIO',test='H2_price_adjusted',p=hypothesis('M5_PRICE_AT_FULL_PORTFOLIO','H2_absolute_union_null'))]
    for h in [1,2,3]:review_tests.append(dict(family='R6_primary',model='H3_ALPHA',test='H3_alpha_h'+str(h),p=None,reason='No verified sufficient fund unit-return and factor history'))
    review_tests += [dict(family='R6_secondary',model='M4_DW_EXACT_ACTUAL_WINDOWS',test='H1_actual_windows',p=hypothesis('M4_DW_EXACT_ACTUAL_WINDOWS','H1_directional_intersection_union')),
        dict(family='R6_secondary',model='M4_DW_EXACT_ACTUAL_WINDOWS',test='H2_actual_windows',p=hypothesis('M4_DW_EXACT_ACTUAL_WINDOWS','H2_absolute_union_null')),
        dict(family='R6_secondary',model='QUANTITY_OBSERVED',test='H1_actual_quantity',p=None,reason='No verified actual security quantity/unit fields in registered holdings'),
        dict(family='R6_secondary',model='M1_DW_EQUITY_INTERACTION',test='joint_asset_interaction',p=hypothesis('M1_DW_EQUITY_INTERACTION','joint_zero_equity_interactions')),
        dict(family='R6_secondary',model='M2_DW_RELATIVE',test='H1_relative_allocation',p=hypothesis('M2_DW_RELATIVE','H1_directional_intersection_union'))]
    for family in ['R6_primary','R6_secondary']:
        ordered=sorted([t for t in review_tests if t['family']==family],key=lambda t:t['p'] if t['p'] is not None else 1)
        last=0
        for rank,t in enumerate(ordered):
            last=max(last,min(1,(5-rank)*(t['p'] if t['p'] is not None else 1)));t['p_holm']=last;t['family_size_including_unavailable']=5
    for t in review_tests:
        if t['p'] is None:t['p_holm']=None
    summary=dict(status='FINITE_HYPOTHESIS_RESEARCH_COMPLETED',client_date='2026-10-29',posthoc_exploratory=True,
        plan_sha256=manifest['plan_sha256'],plan='config/title_name_hypothesis_analysis_plan_20261029.json',output_root=str(root),
        models=records,tests=tests,review_primary_secondary_tests=review_tests,power_MDE=monthly['power_MDE'],
        reference_beta_max_abs_difference=monthly['reference_beta_max_abs_difference'],
        monthly_manifest=monthly['manifest'],extended_manifest=manifest,
        quote_sources_manifest_sha256=sha256_file(root/'quote_sources_manifest.json'),quote_source_files_hashed=len(quote_hashes),
        quote_coverage=quote_audit,price_coverage_summary=dict(monthly_intervals=len(coverage),fully_covered_intervals=int(coverage.eligible_full_price_AT.sum()),
            coverage_fraction_quantiles=coverage.coverage_fraction.quantile([0,.25,.5,.75,1]).to_dict(),uncovered_positions=len(uncovered),
            union_target_rows=len(panel),continuing_positive_rows=int(continuing.sum()),accepted_continuing_price_rows=len(proxy),
            entries_excluded_from_log=int(panel.full_entry.sum()),exits_excluded_from_log=int(panel.full_exit.sum()),
            allowed_identified_RUB_cash_bucket_count=len(cash)),
        expanded_panel=expanded_audit,exact_window_signal=signal_audit,
        source_preservation='Canonical inputs, live LLM and completed v2 untouched; all bulk derived outputs remain on research-host',
        H3_alpha=dict(status='NOT_ESTIMABLE',registered_fund_returns_period=['2024-12','2025-11'],minimum_prior_months=36,
                      reason='Insufficient verified linked unit-value/return history and appropriate equity/bond factors; raw fund return or DW alignment not promoted to alpha'),
        actual_quantity=dict(status='NOT_ESTIMABLE_PENDING_SOURCE_AUDIT',registered_fields='No quantity or unit fields in funds.holdings; V/P proxy is separately named'),
        limitations=['Already observed DW results motivated this follow-up; locked exploration is not original preregistration',
            'Price-based approximation omits payouts and unverified splits; reported valuation and clean market price may differ, especially for bonds',
            'Full-portfolio denominator is never replaced by a covered-subportfolio denominator',
            'Continuing positive/price-covered positions are a survivor-selected estimand; real entries/exits retained in DW but excluded from log proxy',
            'Exact-window branch includes monthly and quarterly/longer actual intervals and controls log duration; not a synthetic monthly panel',
            'Static issuer/unknown mandates/date provenance and incomplete human validation remain',
            'Conditional clustered SEs do not include LLM or generated-signal uncertainty; no causal or manager-skill claim'],
        completed_at_utc=datetime.now(timezone.utc).isoformat())
    # Assert protected base artifacts have not changed during the finite pass.
    if any(sha256_file(BASE/name)!=digest for name,digest in manifest['base_files'].items()):raise IntegrityError('A sealed source changed during analysis')
    save(root/'summary.json',summary)
    pd.DataFrame(tests+review_tests).to_csv(root/'tests.csv',index=False)
    pd.DataFrame([row for record in records for row in record['coefficients']]).to_csv(root/'coefficients.csv',index=False)
    result_tables(root,summary)
    stage(root,'COMPLETED',models=len(records),estimated=sum(r['status'].startswith('ESTIMATED') for r in records),
          summary_sha256=sha256_file(root/'summary.json'),tables_sha256=sha256_file(root/'tables.md'))


if __name__=='__main__':main()
