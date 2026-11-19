# -*- coding: utf-8 -*-
"""Presentation-only H2/H3 index; requires completed results and passed QA."""
from pathlib import Path
import collections
import csv
import datetime as dt
import hashlib
import html
import json
import math
from html.parser import HTMLParser
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'outputs/h23_hypotheses_20261103'
H2=BUNDLE/'h2'
H3=BUNDLE/'h3_mechanism'
CORE={'equity':'Акции','bond':'Облигации','mixed':'Смешанные'}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def e(x):return html.escape(str(x))
def finite(x):return isinstance(x,(int,float)) and math.isfinite(x)
def num(x,d=4):return f'{x:,.{d}f}'.replace(',',' ').replace('.',',') if finite(x) else '—'
def integer(x):return f'{int(x):,}'.replace(',',' ') if finite(x) else '—'
def pv(x):return '<0,000001' if finite(x) and x<1e-6 else num(x,6)
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,list):return [clean(v) for v in x]
    return None if isinstance(x,float) and not math.isfinite(x) else x


def main():
    # Check markers before opening any numerical output: no partial-run report.
    qa2_path=H2/'qa_independent.json'
    qa3_path=H3/('qa_independent_v2.json' if (H3/'qa_independent_v2.json').exists() else 'qa_independent.json')
    for folder,qa_path in [(H2,qa2_path),(H3,qa3_path)]:
        completion=json.loads((folder/'completion.json').read_text())
        if completion.get('status')!='COMPLETE':raise RuntimeError(f'{folder.name}: incomplete run')
        qa=json.loads(qa_path.read_text())
        if qa.get('fail_count')!=0 or not qa.get('status','').startswith('PASS'):
            raise RuntimeError(f'{folder.name}: QA has not passed')
    inputs=[H2/'summary.json',H3/'summary.json',H2/'coefficients.csv',H3/'coefficients.csv',
        qa2_path,qa3_path,H2/'completion.json',H3/'completion.json',
        H2/'plan.json',H3/'plan.json',H2/'source_manifest.json',H3/'source_manifest.json',
        BUNDLE/'h3_primary_data_readiness.json',H2/'review.html',H3/'review.html']
    # Optional post-result diagnosis is never read while incomplete or unverified.
    diagnostic_folder=BUNDLE/'h3_trade_sign_diagnostic'
    diagnostic_qa=diagnostic_folder/('qa_independent_v2.json' if (diagnostic_folder/'qa_independent_v2.json').exists() else 'qa_independent.json')
    diagnostic=None;diagnostic_block=''
    if (diagnostic_folder/'completion.json').exists() and diagnostic_qa.exists() and (diagnostic_folder/'review.html').exists():
        diagnostic_completion=json.loads((diagnostic_folder/'completion.json').read_text())
        diagnostic_check=json.loads(diagnostic_qa.read_text())
        if diagnostic_completion.get('status')=='COMPLETE' and diagnostic_check.get('fail_count')==0 and diagnostic_check.get('status','').startswith('PASS'):
            diagnostic_summary=json.loads((diagnostic_folder/'summary.json').read_text())
            assert diagnostic_summary['status']=='COMPLETE' and len(diagnostic_summary['models'])==9
            assert diagnostic_summary['analysis_role']=='POST_RESULT_SHARED_SIGN_DIAGNOSTIC_NOT_PRIMARY_NOT_ALPHA'
            assert diagnostic_summary['baseline_refit'] is False
            assert all(m.get('same_ordered_rows_as_baseline') is True for m in diagnostic_summary['models'])
            inputs.extend([diagnostic_folder/'completion.json',diagnostic_qa,diagnostic_folder/'summary.json',diagnostic_folder/'review.html'])
            diagnostic=dict(analysis_role=diagnostic_summary['analysis_role'],model_count=9,baseline_refit=False,
                all_same_ordered_rows_as_baseline=True,not_new_primary=True,not_fund_alpha=True,
                delta_beta_inference='Descriptive only; no crossmodel coefficient-change test',
                model_status_counts=dict(collections.Counter(m['status'] for m in diagnostic_summary['models'])),
                QA=dict(file=str(diagnostic_qa.relative_to(BUNDLE)),sha256=sha(diagnostic_qa),
                    status=diagnostic_check['status'],pass_count=diagnostic_check['pass_count'],fail_count=diagnostic_check['fail_count']),
                report='h3_trade_sign_diagnostic/review.html',comparison='h3_trade_sign_diagnostic/comparison.csv')
            diagnostic_block='''<h2>Отдельная диагностика после основного результата: общий знак s</h2><div class="notice"><p>В Y=100×s×E и X=s×Net входит один и тот же знак текущего расчётного изменения позиции. На ровно девяти прежних Q99_D99/core60/24 выборках добавлен обязательный контроль s, чтобы проверить постоянный mean-direction/carry компонент. Основная серия1080 не менялась; baseline не переоценивался. Изменение β описательное, без межмодельного теста.</p><p><a href="h3_trade_sign_diagnostic/review.html">Все9 baseline / +s результатов с SE, CI, p, joint и суммой</a>. Диагностика прошла отдельный QA; это не новая primary, не alpha фонда и не полная корректировка риска, купонов или корпоративных действий.</p></div>'''
    before={str(p.relative_to(BUNDLE)):sha(p) for p in inputs}
    h2=json.loads((H2/'summary.json').read_text());h3=json.loads((H3/'summary.json').read_text())
    q2=json.loads(qa2_path.read_text());q3=json.loads(qa3_path.read_text())
    qa3_relative=str(qa3_path.relative_to(BUNDLE))
    qa3_fix=(dict(selected_version=qa3_relative,scope='QA date-format comparison only',
        failed_v1_preserved_as='h3_mechanism/qa_independent.json',samples_or_coefficients_recomputed=False)
        if qa3_path.name=='qa_independent_v2.json' else None)
    qa3_note=('<p class="meta">Для DAR используется QA v2: исправлена только собственная нормализация формата дат при буквальном сравнении. Failed v1 сохранён отдельным артефактом <code>h3_mechanism/qa_independent.json</code>. Выборки и коэффициенты из-за этого исправления не пересчитывались.</p>' if qa3_fix else '')
    primary=json.loads((BUNDLE/'h3_primary_data_readiness.json').read_text())
    assert h2['status']==h3['status']=='COMPLETE'
    assert len(h2['models'])==432 and len(h3['models'])==1080
    assert h3['not_a_replacement_for_core_H3'] is True and h3['core_H3_alpha_status'].startswith('NOT_ESTIMABLE')
    assert h2['specification']['multiple_testing_adjustments'] is False and h3['specification']['multiple_testing_adjustments'] is False
    m2={m['model']:m for m in h2['models']};m3={m['model']:m for m in h3['models']}
    def read_coefs(folder):
        with (folder/'coefficients.csv').open(encoding='utf-8-sig',newline='') as s:
            return {(r['model'],r['term']):r for r in csv.DictReader(s)}
    coefficients2,coefficients3=read_coefs(H2),read_coefs(H3)
    def coef(source,label,term):
        r=source.get((label,term),{})
        return {k:float(r[k]) if r.get(k) not in (None,'') else None for k in ['beta','se_native','ci95_native_low','ci95_native_high','p_native']}
    def coefficient_cell(c):
        return f'<b>{num(c.get("beta"))}</b> (SE {num(c.get("se_native"))})<br><span class="small">95% CI [{num(c.get("ci95_native_low"))}; {num(c.get("ci95_native_high"))}]<br>p={pv(c.get("p_native"))}</span>'
    def native(folder,label):return f'<a href="{folder}/{e(label)}.summary.txt">TXT</a> · <a href="{folder}/{e(label)}.summary.html">HTML</a>'
    def table(headers,rows):
        return '<div class="scroll"><table><thead><tr>'+''.join(f'<th>{v}</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{v}</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    pooled={};pooled_rows=[]
    for cut in ['P99','P97_5','P95','P90']:
        name=f'{cut}__REFERENCE_AUGMENTED_P99__CURRENT_PLUS_LAG1';m=m2[name];t=h2['h2_tests'][name]['tests']['lag1']
        positive=coef(coefficients2,name,'net_positive_lag1');negative=coef(coefficients2,name,'net_negative_lag1')
        assert positive['beta']<0 and negative['beta']<0 and t['expected_directional_signs'] is False
        pooled[cut]=dict(model=name,nobs=m['nobs'],funds=m['funds'],companies=m['companies'],first_start_date=m.get('first_start_date'),
            last_end_date=m.get('last_end_date'),rsquared_within=m.get('rsquared_within'),
            positive_lag_coefficient=positive,negative_lag_coefficient=negative,lag_H2=t,
            nominal_abs_asymmetry_at5percent=finite(t.get('p_abs')) and t['p_abs']<.05)
        pooled_rows.append([cut,f'{integer(m["nobs"])} / {integer(m["funds"])} / {integer(m["companies"])}',
            coefficient_cell(positive),coefficient_cell(negative),f'<b>{pv(t.get("p_abs"))}</b>',
            f'{pv(t["difference_zero"].get("p_native"))} / {pv(t["sum_zero"].get("p_native"))}',native('h2',name)])
    nominal=[cut for cut,t in pooled.items() if t['nominal_abs_asymmetry_at5percent']]
    assert nominal==['P95','P90']
    core={};core_rows=[];core_coefficient_rows=[]
    terms={'net_positive':'Текущий Net⁺','net_negative':'Текущий Net⁻','net_positive_lag1':'Предыдущий Net⁺','net_negative_lag1':'Предыдущий Net⁻'}
    for focus,title in CORE.items():
        name=f'P99__FOCUS_{focus.upper()}_ROWS_60_INTERVALS_24__CURRENT_PLUS_LAG1';m=m2[name];t=h2['h2_tests'][name]['tests']
        cc={term:coef(coefficients2,name,term) for term in terms}
        core[focus]=dict(model=m,coefficients=cc,tests=t)
        assert all(t[scope].get('p_abs',1)>=.05 for scope in ['current','lag1','total_current_plus_lag1'])
        core_rows.append([title,f'{integer(m["nobs"])} / {integer(m["funds"])} / {integer(m["companies"])}',
            pv(t['current'].get('p_abs')),pv(t['lag1'].get('p_abs')),pv(t['total_current_plus_lag1'].get('p_abs')),
            num(m.get('rsquared_within'),6),f'{integer(m.get("achieved_min_fund_rows"))} / {integer(m.get("achieved_min_fund_intervals"))}',native('h2',name)])
        for term,title_term in terms.items():core_coefficient_rows.append([title,title_term,coefficient_cell(cc[term])])
    dar={};dar_rows=[]
    for focus,title in CORE.items():
        for horizon in [1,2,3]:
            name=f'Q99_D99__M{horizon}__FOCUS_{focus.upper()}_ROWS_60_INTERVALS_24__CURRENT_PLUS_LAG1';m=m3[name]
            current=coef(coefficients3,name,'align_current');lag=coef(coefficients3,name,'align_lag1')
            tests=h3['distributed_tests'].get(name) or {};joint=tests.get('joint_current_lag_zero',{});total=tests.get('sum_current_lag',{})
            dar[name]=dict(model=m,current_coefficient=current,lag_coefficient=lag,distributed_tests=tests)
            dar_rows.append([title,horizon,f'{integer(m["nobs"])} / {integer(m["funds"])} / {integer(m["companies"])}',
                coefficient_cell(current),coefficient_cell(lag),pv(joint.get('p_value')),coefficient_cell(total),
                num(m.get('rsquared_within'),6),native('h3_mechanism',name)])
    observed=primary['observed_fund_source'];links=primary['current_universe_linkage']
    assert observed['fund_series_passing36in60_and9in12']==0 and links['current_funds_with_eligible_alpha_history']==0
    primary_summary=dict(status=h3['core_H3_alpha_status'],eligible_fund_series=0,
        registered_return_rows=observed['rows'],registered_series_keys=observed['series_keys'],
        observed_unit_price_levels=observed['positive_unit_value_rows'],strict_observed_monthly_returns=observed['strict_observed_adjacent_returns'],
        max_observed_past_monthly_returns=observed['max_observed_returns_within60'],required_past_returns=36,window_months=60,
        max_even_undocumented_stored_returns=observed['max_existing_nonmissing_returns_per_series'],
        missing_dependencies=primary['exact_missing_dependencies'],source='h3_primary_data_readiness.json')
    def qa_summary(q,path):return dict(file=str(path.relative_to(BUNDLE)),sha256=before[str(path.relative_to(BUNDLE))],status=q['status'],pass_count=q['pass_count'],fail_count=q['fail_count'],
        independent_replication_models=q['independent_replication_models'],warning_counts=dict(collections.Counter(w['kind'] for w in q['warnings'])),limitations=q['limitations'])
    machine=dict(status='COMPLETE_VERIFIED_H2_AND_SUPPLEMENTARY_DAR_WITH_LIMITATIONS',created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        presentation_only=True,new_regressions=False,numeric_originals_unchanged=True,
        H2=dict(estimand='Absolute positive/negative sentiment-slope asymmetry for conditional estimated quantity change',
            native_inference='Unadjusted two-sided p; absolute-equality intersection-union p=max(p_difference,p_sum)',
            total_model_labels=432,pooled_lag_without_minimum_history=pooled,pooled_nominal_abs_asymmetry_cuts=nominal,
            pooled_expected_H1_directional_signs=False,core_P99_60rows_24intervals_distributed=core,
            no_H2_at5percent_in_displayed_core_models=True,QA=qa_summary(q2,qa2_path),report='h2/review.html'),
        H3_primary_alpha=primary_summary,
        supplementary_DAR=dict(role='POSITION_LEVEL_CASH_EXCESS_CLEAN_PRICE_MECHANISM_NOT_FUND_ALPHA',
            not_a_replacement_for_core_H3=True,outcome='100*sign(delta_quantity_units)*(future_clean_RUB_price_return-RUONIA_index_return)',
            current_alignment='sign(current deltaQ)*current Net',lag_alignment='sign(current deltaQ)*exact prior-interval Net',
            main_scenario='Q99_D99',main_support='60 rows AND24 actual intervals qualified before extra filters',
            horizons=[1,2,3],main_results=dar,total_model_labels=1080,scenario_definitions=h3['specification']['cutoffs'],
            quantity_direction_is_estimated=True,corporate_actions_comprehensively_adjusted=False,
            QA=qa_summary(q3,qa3_path),QA_format_only_fix=qa3_fix,report='h3_mechanism/review.html'),
        common_parameters=dict(fund_fixed_effects=True,time_fixed_effects=False,clusters=['fund_id','company_id'],
            time_clusters=False,weights=None,multiple_testing_adjustments=False,
            baseline_controls=h3['specification']['baseline_controls'],macro_controls=h3['specification']['macro_controls']),
        post_result_shared_sign_diagnostic=diagnostic,
        source_sha256=before,files=dict(index='index.html',summary='summary.json',H2='h2/review.html',DAR='h3_mechanism/review.html',primary_H3_readiness='h3_primary_data_readiness.json'))
    status_rows=[['H2: общая выборка, lag','Номинальная асимметрия при P95 и P90; оба lag-коэффициента отрицательны'],
        ['H2: отдельные типы фондов','В P99 60/24 ни текущая, ни lag, ни суммарная асимметрия незначима на5%'],
        ['H3: будущая alpha фонда','NOT_ESTIMABLE: 0 подходящих серий; максимум3 наблюдаемые доходности вместо36'],
        ['Дополнительный DAR','Отдельная ценовая проверка механизма; таблицы ниже не заменяют H3 alpha']]
    content=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>H2 и дополнительный DAR — результаты 03.11.2026</title>
<style>body{{margin:0;background:#f3f6f8;color:#172b3b;font:16px/1.55 system-ui,-apple-system,sans-serif}}main{{max-width:1420px;margin:auto;padding:34px 26px 75px}}h1{{font-size:34px;line-height:1.18}}h2{{font-size:24px;margin-top:34px}}p{{max-width:1100px}}a{{color:#07579b}}.scroll{{overflow:auto;background:white;border:1px solid #d6e0e8;border-radius:8px;margin:15px 0 25px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:10px 12px;text-align:right;white-space:nowrap;border-bottom:1px solid #e1e8ef;vertical-align:top}}th{{background:#e7eef4}}td:first-child,th:first-child{{text-align:left}}.small{{font-size:11px;color:#40576a}}.meta{{color:#536c7d;font-size:13px}}.notice{{background:#e8f0f7;border-left:5px solid #315c7a;padding:16px 21px;border-radius:7px}}.warn{{background:#fff0d8;border-color:#a66f20}}nav{{display:flex;gap:16px;flex-wrap:wrap;margin:20px 0}}details{{background:white;border:1px solid #d6e1e9;border-radius:7px;padding:15px 19px;margin:15px 0}}summary{{font-weight:650;cursor:pointer}}code{{overflow-wrap:anywhere}}@media(max-width:650px){{main{{padding:23px 13px}}h1{{font-size:27px}}}}</style><main><div class="meta">03 ноября 2026 · проверенные расчёты на домашнем ПК</div><h1>H2 и будущая ценовая динамика позиций</h1>
<nav><a href="h2/review.html">Полный отчёт H2</a><a href="h3_mechanism/review.html">Полный отчёт дополнительного DAR</a><a href="summary.json">Машиночитаемый summary</a></nav>
{table(['Проверка','Результат'],status_rows)}
<p>Все p двусторонние, без поправок на множественные тесты. Fund FE и кластеры фонд/компания; временных эффектов/кластеров нет. Значимый отдельный коэффициент не подтверждает автоматически гипотезу об асимметрии, направление реакции или исходную H3.</p>
<h2>H2 · общий lag-результат при четырёх отсечениях</h2><p>Общая REFERENCE-выборка без дополнительных min-history порогов; модель содержит текущую и предыдущую положительную/отрицательную части Net. N/F/C — наблюдения / фонды / компании. β — п.п. изменения расчётного количества на +1 части Net.</p>
{table(['Quantity cut','N / F / C','Предыдущий Net⁺: β / SE / CI / p','Предыдущий Net⁻: β / SE / CI / p','p lag-H2','p разности / суммы','Python'],pooled_rows)}
<p><strong>P95 p H2={pv(pooled['P95']['lag_H2']['p_abs'])}; P90 p H2={pv(pooled['P90']['lag_H2']['p_abs'])}.</strong> Это исследовательский результат общей смеси типов фондов. Оба lag-коэффициента отрицательны; отрицательная часть сильнее по модулю. Ожидаемая H1-направленность +/− не выполнена. При P99 и P97,5 formal H2 не достигает5%.</p>
<p>H2 проверяет |β⁺|=|β⁻|. Это объединение двух нулей: β⁺−β⁻=0 ИЛИ β⁺+β⁻=0; один intersection-union тест имеет p=max(p разности,p суммы). Максимум двух компонент здесь не является множественной поправкой.</p>
<h2>H2 · по типам фондов, P99 · 60 строк И 24 интервала</h2>
{table(['Тип фонда','N / F / C','H2 текущий p','H2 lag1 p','H2 сумма p','within R²','Факт. min строк / интервалов','Python'],core_rows)}
<details><summary>Все четыре коэффициента по типам: β, SE, 95% CI и native p</summary>{table(['Тип фонда','Сигнал','Коэффициент'],core_coefficient_rows)}</details>
<p>Исходная квалификация по истории выполнена до удаления строк без lag и отсечения; фактические минимумы не выдаются за исходные 24 периода. В этих core-группах H2 на5% не установлена; это не утверждение о незначимости всех pooled результатов. Различные p в pooled и группах не являются самостоятельным тестом различия slopes.</p>
<h2>Основная H3: будущая alpha фонда пока недоступна</h2><div class="notice warn"><p><strong>0 фондовых серий удовлетворяют принятой истории 36 прошлых месячных доходностей в окне60.</strong> В наблюдаемых данных максимум {integer(primary_summary['max_observed_past_monthly_returns'])}, даже в недокументированном сохранённом fund_ret максимум {integer(primary_summary['max_even_undocumented_stored_returns'])}. Рост общего NAV, доходность отдельной бумаги и DAR не подменяют alpha. <a href="h3_primary_data_readiness.json">Точный аудит доступных данных</a>.</p></div>
<h2>Дополнительный DAR · Q99_D99 · 60/24 · горизонты 1–3 месяца</h2><p>Y=100×sign(ΔQ)×(будущая ценовая доходность бумаги−доходность индекса RUONIA). Alignment текущего/предыдущего периода равен sign(текущегоΔQ)×Net соответствующего интервала. Ни Y, ни X не являются исходной fund alpha/NTS; общий множитель направления не позволяет отдельно идентифицировать торговое мастерство или причинный эффект новостей.</p>
{table(['Тип фонда','h, мес.','N / F / C','Текущий alignment: β / SE / CI / p','Lag alignment: β / SE / CI / p','p joint=0','Сумма: β / SE / CI / p','within R²','Python'],dar_rows)}
<p>Joint проверяет два коэффициента одновременно; сумма использует полную cross-covariance. Показаны все девять заранее обозначенных моделей, а не выбранный наиболее значимый горизонт. Все пять сценариев, 1080 статусов, точные потери и исходный Python output — в <a href="h3_mechanism/review.html">полном дополнительном отчёте DAR</a>.</p>
{diagnostic_block}
<div class="notice warn"><p>Цены — нормированные RUB clean prices без полного учёта купонов, выплат и корпоративных действий. Q восстановлено как V/P. Консолидация ВТБ демонстрирует разрыв необработанного ценового масштаба; она может менять и расчётное направление ΔQ. Q99_RAW оставлен диагностикой. DAR-отсечение не исправляет все такие эффекты; не каждый экстремум доказан ошибкой.</p></div>
<h2>Проверки и файлы</h2><p>H2: {integer(q2['pass_count'])} PASS / {integer(q2['fail_count'])} FAIL. Дополнительный DAR: {integer(q3['pass_count'])} PASS / {integer(q3['fail_count'])} FAIL. Независимо численно воспроизведены {len(q2['independent_replication_models'])} моделей H2 и {len(q3['independent_replication_models'])} моделей DAR; остальные прошли механические проверки. Предупреждения малых групп и неопределённой covariance сохранены.</p>
{qa3_note}<nav><a href="h2/qa_independent.json">QA H2</a><a href="{e(qa3_relative)}">QA DAR: {e(qa3_path.name)}</a><a href="h2/plan.json">План H2</a><a href="h3_mechanism/plan.json">План DAR</a><a href="h2/source_manifest.json">Источники H2</a><a href="h3_mechanism/source_manifest.json">Источники DAR</a><a href="h3_mechanism/data/outcome_trim_calibration.json">Калибровка |DAR|</a></nav>
<p class="meta">Эта страница форматирует завершённые результаты и не меняет исходные числа. Полные Parquet-панели остаются на домашнем ПК в соответствующих каталогах analysis_outputs; здесь находятся читаемые отчёты и небольшие таблицы результатов.</p></main></html>'''
    target=BUNDLE/'index.html';target.write_text(content,encoding='utf-8')
    class Parser(HTMLParser):
        def __init__(self):super().__init__();self.hrefs=[]
        def handle_starttag(self,tag,attrs):self.hrefs.extend(v for k,v in attrs if k=='href')
    parser=Parser();parser.feed(content)
    missing=[]
    for href in parser.hrefs:
        if href.startswith(('#','http:','https:')):continue
        resolved=(BUNDLE/unquote(href.split('#')[0])).resolve()
        if not resolved.is_relative_to(BUNDLE.resolve()):raise AssertionError('Link escapes bundle: '+href)
        if href!='summary.json' and not resolved.exists():missing.append(href)
    assert not missing,missing
    assert all(sha(BUNDLE/name)==value for name,value in before.items())
    machine['presentation_checks']=dict(all_folder_relative_links=True,local_links=len(parser.hrefs),missing_links=missing,
        H1_not_included=True,optional_shared_sign_diagnostic_included=diagnostic is not None,index_sha256=sha(target))
    (BUNDLE/'summary.json').write_text(json.dumps(clean(machine),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=machine['status'],index=str(target),summary=str(BUNDLE/'summary.json'),
        local_links=len(parser.hrefs),index_sha256=sha(target),summary_sha256=sha(BUNDLE/'summary.json')),ensure_ascii=False))


if __name__=='__main__':main()
