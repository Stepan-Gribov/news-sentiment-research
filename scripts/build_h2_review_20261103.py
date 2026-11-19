# -*- coding: utf-8 -*-
"""Readable H2 report from completed small results; never fits or reads bulk data."""
from pathlib import Path
import collections
import csv
import datetime as dt
import hashlib
import html
import json
import math
import re
from html.parser import HTMLParser
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/h23_hypotheses_20261103/h2'
REVIEW=ROOT/'reports/h2_result_review_20261103.json'
FOCI={'equity':'Акции','bond':'Облигации','mixed':'Смешанные','money_market':'Денежный рынок','other':'Прочие',None:'Общая выборка'}
CORE=['equity','bond','mixed']
VARIANTS={'CURRENT_FULL':'Текущий: вся выборка','CURRENT_COMMON':'Текущий: общие строки','CURRENT_PLUS_LAG1':'Текущий + lag1: общие строки'}
TERMS={'net_positive':'Текущий Net⁺','net_negative':'Текущий Net⁻','net_positive_lag1':'Предыдущий Net⁺','net_negative_lag1':'Предыдущий Net⁻'}
SCOPES={'current':'Текущий','lag1':'Предыдущий','total_current_plus_lag1':'Сумма текущего и предыдущего'}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def esc(x):return html.escape(str(x))
def finite(x):return isinstance(x,(int,float)) and math.isfinite(x)
def num(x,n=4):return f'{x:,.{n}f}'.replace(',',' ').replace('.',',') if finite(x) else '—'
def integer(x):return f'{int(x):,}'.replace(',',' ') if finite(x) else '—'
def pval(x):return '<0,000001' if finite(x) and x<1e-6 else num(x,6)
def date(x):return str(x)[:10] if x else '—'
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,list):return [clean(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):return None
    return x


def main():
    inputs=['summary.json','coefficients.csv','qa_independent.json','h2_tests.json','thresholds.json','mask_catalog.json','comparison.csv','sample_flow.csv','plan.json','source_manifest.json']
    before={n:sha(OUT/n) for n in inputs}
    summary=json.loads((OUT/'summary.json').read_text());qa=json.loads((OUT/'qa_independent.json').read_text())
    models=summary['models'];byname={m['model']:m for m in models};tests=summary['h2_tests'];spec=summary['specification']
    assert len(models)==len(byname)==432
    assert qa['fail_count']==0 and qa['pass_count']==19354
    with (OUT/'coefficients.csv').open(encoding='utf-8-sig',newline='') as f:coefs={(r['model'],r['term']):r for r in csv.DictReader(f)}
    def coefficient(label,term):
        r=coefs.get((label,term),{})
        return {k:float(r[k]) if r.get(k) not in (None,'') else None for k in ['beta','se_native','p_native','ci95_native_low','ci95_native_high']}
    def test(label,scope='current'):return tests.get(label,{}).get('tests',{}).get(scope,{})
    def label(focus,cut='P99',variant='CURRENT_PLUS_LAG1'):return f'{cut}__FOCUS_{focus.upper()}_ROWS_60_INTERVALS_24__{variant}'
    def native_links(name):return f'<a href="{esc(name)}.summary.txt">TXT</a> · <a href="{esc(name)}.summary.html">HTML</a>'
    def ci(c):return f'[{num(c.get("ci95_native_low"))}; {num(c.get("ci95_native_high"))}]'
    def table(headers,rows,cls=''):
        return f'<div class="table-wrap {cls}"><table><thead><tr>'+''.join(f'<th>{h}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{x}</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    minima={};iut_count=0
    for focus in CORE:
        candidates=[]
        for m in models:
            if m.get('fund_focus')!=focus:continue
            for scope,t in tests.get(m['model'],{}).get('tests',{}).items():
                if finite(t.get('p_abs')):candidates.append(dict(p=t['p_abs'],model=m['model'],scope=scope))
        assert candidates and all(x['p']>=.05 for x in candidates)
        minima[focus]=dict(minimum=min(candidates,key=lambda x:x['p']),finite_tests=len(candidates),significant_at_5pct=0)
    expected={'equity':.06625736056519807,'bond':.064715422585034,'mixed':.14262040940304083}
    for focus in CORE:assert math.isclose(minima[focus]['minimum']['p'],expected[focus],abs_tol=1e-12)
    for pack in tests.values():
        for t in pack.get('tests',{}).values():
            if finite(t.get('p_abs')):
                assert math.isclose(t['p_abs'],max(t['difference_zero']['p_native'],t['sum_zero']['p_native']),abs_tol=1e-12)
                iut_count+=1
    main_results={};coefficient_rows=[];test_rows=[];sample_rows=[]
    for focus in CORE:
        name=label(focus);m=byname[name]
        main_results[focus]=dict(model=m,coefficients={term:coefficient(name,term) for term in TERMS},tests=tests[name])
        for term,term_name in TERMS.items():
            c=coefficient(name,term)
            coefficient_rows.append([FOCI[focus],term_name,num(c['beta']),num(c['se_native']),ci(c),pval(c['p_native'])])
        for scope,scope_name in SCOPES.items():
            t=test(name,scope)
            test_rows.append([FOCI[focus],scope_name,pval(t.get('p_abs')),pval(t.get('difference_zero',{}).get('p_native')),
                pval(t.get('sum_zero',{}).get('p_native')),num(t.get('negative_absolute_minus_positive_absolute')),
                'Да' if t.get('expected_directional_signs') else 'Нет'])
        sample_rows.append([FOCI[focus],integer(m.get('nobs')),integer(m.get('funds')),integer(m.get('companies')),
            f'{date(m.get("first_start_date"))} — {date(m.get("last_end_date"))}',num(m.get('rsquared_within'),6),
            f'{integer(m.get("achieved_min_fund_rows"))} / {integer(m.get("achieved_min_fund_intervals"))}',native_links(name)])
    pooled_results={};pooled_coefficient_rows=[];pooled_test_rows=[]
    for cut in ['P99','P97_5','P95','P90']:
        name=f'{cut}__REFERENCE_AUGMENTED_P99__CURRENT_PLUS_LAG1';m=byname[name];t=test(name,'lag1')
        pooled_results[cut]=dict(model=m,lag1_test=t,native_two_sided_nominal_H2_significant=finite(t.get('p_abs')) and t['p_abs']<.05)
        for component,title in [('positive_coefficient','Предыдущий Net⁺'),('negative_coefficient','Предыдущий Net⁻')]:
            c=t[component]
            pooled_coefficient_rows.append([cut,f'{integer(m.get("nobs"))} / {integer(m.get("funds"))} / {integer(m.get("companies"))}',
                title,num(c.get('beta')),num(c.get('se_native')),ci(c),pval(c.get('p_native'))])
        pooled_test_rows.append([cut,pval(t.get('p_abs')),pval(t.get('difference_zero',{}).get('p_native')),
            pval(t.get('sum_zero',{}).get('p_native')),num(t.get('negative_absolute_minus_positive_absolute')),
            'Да' if t.get('expected_directional_signs') else 'Нет',num(m.get('rsquared_within'),6),native_links(name)])
    assert [cut for cut,record in pooled_results.items() if record['native_two_sided_nominal_H2_significant']]==['P95','P90']
    comparison_rows=[]
    for focus in CORE:
        for variant in VARIANTS:
            name=label(focus,variant=variant);m=byname[name]
            comparison_rows.append([FOCI[focus],VARIANTS[variant],integer(m.get('nobs')),integer(m.get('funds')),integer(m.get('companies')),
                num(m.get('rsquared_within'),6),pval(test(name).get('p_abs')),pval(test(name,'lag1').get('p_abs')),
                pval(test(name,'total_current_plus_lag1').get('p_abs')),native_links(name)])
    threshold_rows=[]
    for t in summary['thresholds']:
        threshold_rows.append([t['tag'],num(t['cutoff_percent'],6),integer(t['kept_from_P99_primary']),integer(t['removed_from_P99_primary']),
            integer(t['entries_kept']),integer(t['exits_kept'])])
    sensitivity_rows=[]
    for cut in ['P99','P97_5','P95','P90']:
        for focus in CORE:
            name=label(focus,cut);m=byname[name]
            sensitivity_rows.append([cut,FOCI[focus],integer(m.get('nobs')),integer(m.get('funds')),integer(m.get('companies')),
                pval(test(name).get('p_abs')),pval(test(name,'lag1').get('p_abs')),pval(test(name,'total_current_plus_lag1').get('p_abs')),
                num(m.get('rsquared_within'),6),f'{integer(m.get("achieved_min_fund_rows"))} / {integer(m.get("achieved_min_fund_intervals"))}',native_links(name)])
    master_rows=[]
    for m in models:
        name=m['model'];focus=m.get('fund_focus');warnings=[x['kind'] for x in qa['warnings'] if x.get('model')==name]
        master_rows.append(f'<tr data-cut="{m["cutoff_tag"]}" data-focus="{focus or "all"}" data-variant="{m["variant"]}"><td class="label">{esc(name)}</td>'+
            ''.join(f'<td>{x}</td>' for x in [FOCI[focus],esc(m['status']),integer(m.get('nobs')),integer(m.get('funds')),integer(m.get('companies')),
                num(m.get('rsquared_within'),6),pval(test(name).get('p_abs')),pval(test(name,'lag1').get('p_abs')),
                pval(test(name,'total_current_plus_lag1').get('p_abs')),
                f'{integer(m.get("achieved_min_fund_rows"))} / {integer(m.get("achieved_min_fund_intervals"))}',
                '<span class="tiny">'+esc(', '.join(sorted(set(warnings))))+'</span>',native_links(name)])+'</tr>')
    minimum_rows=[[FOCI[f],integer(minima[f]['finite_tests']),pval(minima[f]['minimum']['p']),
        SCOPES[minima[f]['minimum']['scope']],esc(minima[f]['minimum']['model']),'0'] for f in CORE]
    warning_counts=dict(collections.Counter(x['kind'] for x in qa['warnings']))
    warning_rows=[[esc(k),integer(v)] for k,v in warning_counts.items()]
    native_sections=[]
    for focus in CORE:
        name=label(focus);text=(OUT/(name+'.summary.txt')).read_text()
        native_sections.append(f'<details><summary>{FOCI[focus]} · P99 · 60/24 · текущий + lag1</summary><p>{native_links(name)}</p><pre>{esc(text)}</pre></details>')
    full_head=['Модель','Тип фонда','Статус','N','Фондов','Компаний','within R²','H2: текущий p','H2: lag1 p','H2: сумма p','Факт. min строк / интервалов','Предупреждения QA','Python output']
    full_table='<div class="table-wrap"><table id="allmodels"><thead><tr>'+''.join(f'<th>{h}</th>' for h in full_head)+'</tr></thead><tbody>'+''.join(master_rows)+'</tbody></table></div>'
    controls=', '.join(spec['baseline_controls']+spec['macro_controls'])
    content=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>H2 — асимметрия реакции на новости · 03.11.2026</title>
<style>body{{margin:0;background:#f4f6f8;color:#142332;font:16px/1.55 system-ui,-apple-system,sans-serif}}main{{max-width:1380px;margin:auto;padding:36px 26px 80px}}h1{{font-size:36px;line-height:1.16;margin:8px 0 18px}}h2{{font-size:24px;margin-top:38px}}h3{{margin-top:24px}}p{{max-width:1100px}}a{{color:#07549b}}.eyebrow{{text-transform:uppercase;letter-spacing:.12em;font-size:12px;color:#4b6374}}.lead{{font-size:19px}}.notice{{background:#e9f1f7;border-left:5px solid #315b78;padding:18px 22px;border-radius:6px}}.warn{{background:#fff2dc;border-color:#ad7624}}.table-wrap{{overflow:auto;border:1px solid #d7e0e7;border-radius:8px;background:white;margin:16px 0 24px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:10px 12px;text-align:right;border-bottom:1px solid #e4eaf0;vertical-align:top;white-space:nowrap}}th{{background:#e9eef3;font-weight:650;position:sticky;top:0;z-index:1}}td:first-child,th:first-child{{text-align:left}}tbody tr:last-child td{{border-bottom:0}}tbody tr:nth-child(even){{background:#fafbfd}}td.label{{max-width:360px;min-width:260px;white-space:normal;overflow-wrap:anywhere;font:12px/1.4 ui-monospace,monospace}}.tiny{{font-size:11px;max-width:180px;display:block;white-space:normal}}.meta{{color:#516579;font-size:14px}}.filters{{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}}input,select{{border:1px solid #9eafbd;border-radius:6px;padding:9px 12px;background:white;font:inherit}}input{{min-width:320px;flex:1}}details{{background:white;border:1px solid #d7e0e7;border-radius:8px;padding:15px 18px;margin:12px 0}}summary{{cursor:pointer;font-weight:650}}pre{{white-space:pre;overflow:auto;background:#f6f8fa;padding:18px;font:12px/1.45 ui-monospace,SFMono-Regular,monospace}}code{{overflow-wrap:anywhere;font-size:13px}}nav{{display:flex;gap:15px;flex-wrap:wrap;margin:22px 0}}.pill{{display:inline-block;padding:5px 10px;border:1px solid #9fb3c3;border-radius:20px;margin:4px 8px 4px 0;font-size:13px}}@media(max-width:650px){{main{{padding:24px 14px}}h1{{font-size:28px}}input{{min-width:200px}}}}@media print{{body{{background:white}}main{{max-width:none;padding:0}}.filters,nav{{display:none}}.table-wrap{{overflow:visible}}details pre{{font-size:9px}}}}</style>
<main><div class="eyebrow">03 ноября 2026 · готовые расчёты на домашнем ПК</div><h1>H2: отличается ли сила реакции<br>на положительные и отрицательные новости?</h1>
<div class="notice"><p class="lead"><strong>В фондах акций, облигаций и смешанных фондах H2 не подтверждена на уровне 5% ни в одной проверенной модели.</strong></p><p>Это вывод о различии абсолютных коэффициентов, а не об отсутствии любой связи новостей с изменением позиции. Значимость одного коэффициента отдельно не подтверждает H2.</p></div>
<p><span class="pill">432 обозначения моделей</span><span class="pill">348 оценены · 84 неоценимы</span><span class="pill">QA: 19 354 PASS · 0 FAIL</span><span class="pill">Без поправок на множественные тесты</span></p>
<div class="notice"><p><strong>Отдельный исследовательский результат общей выборки:</strong> lag-H2 значим при P95 (p={pval(pooled_results['P95']['lag1_test']['p_abs'])}) и P90 (p={pval(pooled_results['P90']['lag1_test']['p_abs'])}), без минимальных порогов истории. При P99/P97,5 порог 5% не достигнут. Оба lag-коэффициента отрицательны, поэтому это различие силы реакции, а не подтверждение ожидаемой направленности H1. <a href="#pooled">Четыре отсечения и точные коэффициенты</a>.</p></div>
<nav><a href="#main">Основные таблицы по типам</a><a href="#pooled">Общая выборка: lag-H2</a><a href="#compare">Сравнение выборок</a><a href="#cuts">Отсечение экстремумов</a><a href="#all">Все 432 модели</a><a href="#native">Вывод Python</a><a href="#quality">Проверки и ограничения</a></nav>
<p>Y — изменение расчётного количества бумаги в фонде за фактический интервал раскрытия, в процентах. Подтверждённый вход из нуля имеет условное Y=+100%, полный выход — −100%. Net⁺=max(Net,0); Net⁻=max(−Net,0): рост отрицательной части означает более негативные новости. Lag1 — сигнал той же компании за точный предыдущий отчётный интервал фонда, а не предыдущую сохранившуюся строку.</p>
<p class="meta">Во всех моделях fund FE и кластеры по фонду и компании; без time FE и временных кластеров. Наблюдения — фонд × бумага × фактический отчётный интервал; ранние интервалы не называются месяцами. Тип фонда определяется на начало интервала и может меняться во времени.</p>
<h2 id="main">P99 · не менее 60 строк И 24 интервалов · текущий сигнал + lag1</h2>
<p>Порог истории применён к исходной группе до отсечения и удаления строк без lag. После удаления фактический минимум интервалов ниже 24 у акционных и смешанных фондов — это показано в таблице.</p>
{table(['Тип фонда','N','Фондов','Компаний','Период','within R²','Факт. min строк / интервалов','Оригинал Python'],sample_rows)}
<h3>Четыре коэффициента сигнала</h3><p>β и доверительные интервалы — процентные пункты изменения расчётного количества на +1 соответствующей части Net. Показаны обычные двусторонние p.</p>
{table(['Тип фонда','Сигнал','β','SE','95% CI','p коэффициента'],coefficient_rows)}
<h3>Формальный тест H2</h3><p>H₀: |β⁺|=|β⁻| эквивалентна объединению β⁺−β⁻=0 ИЛИ β⁺+β⁻=0. Поэтому <strong>p H2=max(p разности, p суммы)</strong>: отвергнуть равенство модулей можно, только отвергнув оба компонента. Это один intersection-union тест одной гипотезы, а не поправка на множественные сравнения.</p>
{table(['Тип фонда','Экспозиция','p H2','p: β⁺−β⁻=0','p: β⁺+β⁻=0','|β⁻|−|β⁺|','Ожидаемые знаки +/−'],test_rows)}
<p>Для строки «сумма» сначала складываются текущий и lag-коэффициенты одного знака; тест использует полную ковариацию четырёх коэффициентов. Это не сумма четырёх модулей и не структурный долгосрочный эффект. Оба компонента, их β/SE/CI и матрицы ковариаций сохранены в <a href="h2_tests.json">полных метаданных H2</a>.</p>
<h2 id="pooled">Общая выборка без min-history: исследовательский lag-результат</h2>
<p>REFERENCE объединяет типы фондов и не требует дополнительных минимумов строк/интервалов. Здесь показан lag1 в модели с обеими текущими и обеими lag-частями Net; ограничения по исходным данным, макроконтролям и пригодности lag сохраняются. Окно исходных отчётов во всех четырёх моделях: 30.09.2013–29.05.2026. Fund FE, кластеры фонд/компания, без временных эффектов/кластеров.</p>
{table(['Отсечение','N / фондов / компаний','Коэффициент','β','SE','95% CI','p коэффициента'],pooled_coefficient_rows)}
{table(['Отсечение','p lag-H2','p: β⁺−β⁻=0','p: β⁺+β⁻=0','|β⁻|−|β⁺|','Ожидаемые знаки +/−','within R²','Python'],pooled_test_rows)}
<p><strong>P95 и P90 дают номинальное p H2&lt;0,05; P99 и P97,5 — нет.</strong> Все четыре результата показаны без поправок на множественные тесты. Значимость отрицательного lag-коэффициента отдельно — другой тест: например, при P99 он значим, хотя формальный lag-H2 не проходит 5%.</p>
<p>В обеих значимых общих моделях βlag⁺&lt;0 и βlag⁻&lt;0; отрицательная часть сильнее по модулю. Поэтому expected directional signs=false: это не положительная реакция на хорошие новости и отрицательная на плохие. Этот исследовательский результат общей смеси не заменяет основные таблицы по типам фондов. Различия p между pooled и отдельными типами сами по себе не являются тестом неоднородности коэффициентов.</p>
<h2 id="compare">Что меняет отбор общей выборки и добавление lag</h2><p>FULL использует все пригодные текущие наблюдения. COMMON удаляет строки без пригодного lag, сохраняя регрессию только с текущим сигналом. CURRENT_PLUS_LAG1 добавляет lag на <strong>тех же строках</strong>, что COMMON. Сравнение FULL→COMMON показывает влияние состава выборки; COMMON→DL — добавление lag.</p>
{table(['Тип фонда','Модель','N','Фондов','Компаний','within R²','H2 текущий p','H2 lag1 p','H2 сумма p','Python'],comparison_rows)}
<h2 id="cuts">Четыре заранее заданных отсечения количества</h2><p>Пороги рассчитаны один раз по 204 152 продолжающимся позициям исходной nonindex macro-complete выборки, до условных входов, типов фондов, support и lag. Исключаются только продолжающиеся позиции с Y строго выше порога; совпадения с порогом сохраняются. Все 10 329 подтверждённых входов и 9 460 выходов остаются при каждом отсечении. Это чувствительность к экстремумам, а не доказательство ошибки каждой исключённой строки.</p>
{table(['Отсечение','Верхний порог Y, %','Осталось от P99','Удалено от P99','Входов сохранено','Выходов сохранено'],threshold_rows)}
<h3>Те же группы 60/24 с текущим сигналом и lag1</h3>
{table(['Отсечение','Тип фонда','N','Фондов','Компаний','H2 текущий p','H2 lag1 p','H2 сумма p','within R²','Факт. min строк / интервалов','Python'],sensitivity_rows)}
<h3>Проверка всех вариантов core-групп</h3><p>Ниже минимум p H2 среди всех четырёх отсечений, семи масок истории и трёх вариантов модели, с учётом текущего, lag и суммарного теста. Минимумы показаны для полноты серии; выбранного по минимальному p основного результата нет.</p>
{table(['Тип фонда','Конечных H2-тестов','Минимальное p H2','Экспозиция минимума','Модель минимума','p&lt;0,05'],minimum_rows)}
<h2 id="all">Все 432 обозначения моделей</h2><p>36 исходных масок × 4 отсечения × 3 варианта. В таблице сохранены неоценимые модели, предупреждения и прямые ссылки на исходный Python output. Повторяющиеся точные выборки могут ссылаться на один fit; это не 432 независимых повторных оценивания.</p>
<div class="filters"><input id="search" aria-label="Поиск модели" placeholder="Поиск: модель, тип фонда, 60_INTERVALS_24…"><select id="cut" aria-label="Отсечение"><option value="">Все отсечения</option>P99</option><option>P97_5</option><option>P95</option><option>P90</option></select><select id="focus" aria-label="Тип фонда"><option value="">Все типы</option value="all">Общая выборка</option value="equity">Акции</option><option value="bond">Облигации</option><option value="mixed">Смешанные</option><option value="money_market">Денежный рынок</option><option value="other">Прочие</option></select><select id="variant" aria-label="Вариант"><option value="">Все варианты</option><option value="CURRENT_FULL">FULL</option><option value="CURRENT_COMMON">COMMON</option><option value="CURRENT_PLUS_LAG1">CURRENT + LAG1</option></select></div><p id="counter" class="meta">Показано: 432 / 432</p>{full_table}
<h2 id="native">Оригинальный вывод Python</h2>{''.join(native_sections)}
<h2 id="quality">Независимая проверка и ограничения</h2><p><strong>{esc(qa['status'])}: {integer(qa['pass_count'])} PASS, {integer(qa['fail_count'])} FAIL.</strong> Шесть заранее указанных моделей P99/P95 × акции/облигации/смешанные 60/24 DL независимо воспроизведены численно. Остальные модели прошли проверки масок, исходных значений, арифметики и контрастов; независимая переоценка всех моделей не заявляется.</p>
{table(['Тип предупреждения','Записей QA'],warning_rows)}
<p>Денежные и прочие фонды имеют мало кластеров или недостаточную историю; отдельные SE, контрасты и robust F там не определены. Они не заменены нулями, p=1 или исправленной ковариацией. Полные записи сохранены в <a href="qa_independent.json">QA JSON</a>; статус NOT_ESTIMABLE означает отсутствие пригодной оценки, а не нулевой эффект.</p>
<div class="notice warn"><p>Количество Q восстановлено как V/P и не является наблюдаемой сделкой. Некорректированные корпоративные действия могут менять и цену, и расчётное направление ΔQ. Известный пример — консолидация ВТБ в июле 2024; <a href="../h3_mechanism/data/vtbr_raw_price_break_examples.json">исходные примеры цен</a> сохранены. Отсечение верхнего положительного хвоста Y не гарантирует удаления всех таких случаев; не все экстремумы доказаны ошибками.</p><p>Исключены выявленные индексные фонды; полнота выявления всех индексных мандатов не гарантируется. Статический issuer crosswalk, LLM-метки и частичные closing reports несут неопределённость измерения. Макроконтроли не превращают ассоциацию в причинный эффект.</p></div>
<details><summary>Контрольные переменные, данные и воспроизводимость</summary><p><code>{esc(controls)}</code></p><p>Микро- и макроконтроли, исходные Q, новости и прежние результаты не менялись. Нет новостей/нет релевантных новостей/неполная классификация/нет предыдущего интервала не подменяются нейтральным нулём; lag-требование явно меняет выборку.</p><p>Полные использованные Parquet остаются на домашнем ПК: <code>{esc(spec['output_root'])}/samples/*.parquet</code>. Локально — только отчёты и небольшие таблицы результатов.</p><p><a href="plan.json">План до оценок</a> · <a href="source_manifest.json">Источники и SHA</a> · <a href="mask_catalog.json">Все маски и row hashes</a> · <a href="sample_flow.csv">Потери выборки</a> · <a href="thresholds.json">Пороги</a> · <a href="comparison.csv">Таблица сравнения</a> · <a href="coefficients.csv">Все коэффициенты</a> · <a href="h2_tests.csv">Все H2-тесты</a> · <a href="summary.json">Числовой summary</a></p><p class="meta">Поле independent_QA=PENDING в неизменённом числовом summary — прежний маркер момента расчёта. Более поздний фактический независимый QA находится в отдельном файле, на который ссылается этот отчёт.</p></details>
<p class="meta">Отчёт форматирует завершённые результаты. Новые модели, подбор спецификаций и изменение исходных данных при подготовке страницы не выполнялись.</p></main>
<script>const rs=[...document.querySelectorAll('#allmodels tbody tr')];function filter(){{const q=document.querySelector('#search').value.toLowerCase();let n=0;rs.forEach(r=>{{const ok=r.textContent.toLowerCase().includes(q)&&['cut','focus','variant'].every(k=>!document.getElementById(k).value||r.dataset[k]===document.getElementById(k).value);r.hidden=!ok;if(ok)n++;}});document.getElementById('counter').textContent=`Показано: ${{n}} / 432`;}}['search','cut','focus','variant'].forEach(k=>document.getElementById(k).addEventListener('input',filter));</script></html>'''
    target=OUT/'review.html';target.write_text(content,encoding='utf-8')
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):
            for key,val in attrs:
                if key=='href':self.links.append(val)
    parser=Links();parser.feed(content)
    local=[x for x in parser.links if not x.startswith(('#','http:','https:','mailto:'))]
    missing=[x for x in local if not (OUT/unquote(x.split('#')[0])).resolve().exists()]
    assert not missing,missing
    assert all(sha(OUT/n)==checksum for n,checksum in before.items())
    result=dict(status='PASS_H2_RESULT_REVIEW_WITH_PRESERVED_WARNINGS',utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        scope='Formatting and arithmetic checks of completed local small results only; no new fit or bulk data read',
        source_sha256=before,source_files_unchanged=True,report_path=str(target),report_sha256=sha(target),
        total_model_labels=len(models),model_status_counts=dict(collections.Counter(m['status'] for m in models)),
        unique_sample_design_fit_attempts=summary['unique_sample_design_fit_attempts'],
        H2_union_null='abs(beta_positive)=abs(beta_negative): difference zero OR sum zero',
        H2_test='Intersection-union p_abs=max(p_difference,p_sum), one hypothesis, not a multiplicity adjustment',
        verified_finite_IUT_p_values=iut_count,multiple_testing_adjustments=False,
        core_group_minimum_H2_p=minima,no_core_group_H2_significant_at5pct=True,
        pooled_reference_lag_results_all_four_cuts=pooled_results,
        pooled_native_nominal_H2_significant_cuts=['P95','P90'],
        pooled_nominal_result_scope='REFERENCE without minimum-history thresholds; lag absolute-magnitude asymmetry in distributed model, both lag coefficients negative; not directional H1 confirmation',
        main_P99_60rows_24intervals_current_plus_lag1=main_results,
        thresholds=summary['thresholds'],QA=dict(status=qa['status'],pass_count=qa['pass_count'],fail_count=qa['fail_count'],
            independent_replication_models=qa['independent_replication_models'],warning_counts=warning_counts,
            warning_records=qa['warnings'],limitations=qa['limitations']),
        report_checks=dict(local_link_count=len(local),missing_links=missing,all432models_in_master_table=len(master_rows)==432,
            native_TXT_HTML_links_preserved=True,undefined_inference_preserved=True,known_VTBR_quantity_direction_limitation_disclosed=True),
        conclusions_ru=['H2 не подтверждена на5% в акционных, облигационных и смешанных фондах ни в одной конечной проверенной модели.',
            'В общей REFERENCE-выборке без min-history lag-H2 имеет номинальное p<0,05 приP95иP90; приP99иP97,5 нет. Все4результата показаны, без множественных поправок.',
            'В значимых pooled lag-моделях оба коэффициента отрицательны; асимметрия модулей не подтверждает ожидаемую направленность H1.',
            'Отдельный значимый коэффициент положительного или отрицательного сигнала не подтверждает различие их абсолютной силы.',
            'Фактическая история после lag/trim может быть ниже исходного порога отбора; минимумы явно показаны.',
            'Строгое отсечение исследовательское; V/P и направление ΔQ могут искажаться корпоративными действиями.'])
    REVIEW.write_text(json.dumps(clean(result),ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],html=str(target),review=str(REVIEW),links=len(local),
        minima={f:minima[f]['minimum']['p'] for f in CORE},report_sha256=sha(target)),ensure_ascii=False))


if __name__=='__main__':main()
