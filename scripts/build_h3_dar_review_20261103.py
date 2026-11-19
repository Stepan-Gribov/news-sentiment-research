# -*- coding: utf-8 -*-
"""Format completed supplementary DAR results after separate QA; never fit."""
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
OUT=ROOT/'outputs/h23_hypotheses_20261103/h3_mechanism'
REVIEW=ROOT/'reports/h3_dar_result_review_20261103.json'
FOCUS={'equity':'Акции','bond':'Облигации','mixed':'Смешанные','money_market':'Денежный рынок','other':'Прочие',None:'Общая выборка'}
CORE=['equity','bond','mixed']
SCENARIOS=['Q99_RAW','Q99_D99','Q975_D975','Q95_D95','Q90_D90']
VARS={'CURRENT_COMMON':'Текущий: общие строки','CURRENT_PLUS_LAG1':'Текущий + lag1: общие строки'}
MOEX='https://www.moex.com/n71160'
CBR='https://www.cbr.ru/Content/Document/File/128244/RUONIA_AVG_UserGuide.pdf'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def e(x):return html.escape(str(x))
def finite(x):return isinstance(x,(int,float)) and math.isfinite(x)
def n(x,d=4):return f'{x:,.{d}f}'.replace(',',' ').replace('.',',') if finite(x) else '—'
def integer(x):return f'{int(x):,}'.replace(',',' ') if finite(x) else '—'
def pv(x):return '<0,000001' if finite(x) and x<1e-6 else n(x,6)
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,list):return [clean(v) for v in x]
    return None if isinstance(x,float) and not math.isfinite(x) else x


def main():
    # Guard against reading a partial overnight run or publishing before QA.
    completion=json.loads((OUT/'completion.json').read_text())
    if completion.get('status')!='COMPLETE':raise RuntimeError('Run not complete; no partial report produced')
    qa_name='qa_independent_v2.json' if (OUT/'qa_independent_v2.json').exists() else 'qa_independent.json'
    qa=json.loads((OUT/qa_name).read_text())
    if qa.get('fail_count')!=0 or not qa.get('status','').startswith('PASS'):raise RuntimeError('Independent QA has not passed')
    inputs=['summary.json','coefficients.csv',qa_name,'comparison.csv','distributed_tests.json',
        'sample_flow.csv','mask_catalog.json','outcome_audit.json','thresholds.json','plan.json','source_manifest.json','completion.json']
    before={name:sha(OUT/name) for name in inputs}
    summary=json.loads((OUT/'summary.json').read_text());spec=summary['specification']
    if summary.get('status')!='COMPLETE':raise RuntimeError('Summary not complete')
    assert summary['not_a_replacement_for_core_H3'] is True
    assert summary['core_H3_alpha_status'].startswith('NOT_ESTIMABLE')
    assert spec['multiple_testing_adjustments'] is False
    models=summary['models'];byname={m['model']:m for m in models};tests=summary['distributed_tests']
    assert len(models)==len(byname)==1080 and set(byname)==set(spec['planned_models'])
    assert [s['tag'] for s in spec['cutoffs']]==SCENARIOS
    with (OUT/'coefficients.csv').open(encoding='utf-8-sig',newline='') as f:
        coefficients={(r['model'],r['term']):r for r in csv.DictReader(f)}
    def coef(label,term):
        r=coefficients.get((label,term),{})
        return {k:float(r[k]) if r.get(k) not in (None,'') else None for k in ['beta','se_native','p_native','ci95_native_low','ci95_native_high']}
    def label(focus,h,scenario='Q99_D99',variant='CURRENT_PLUS_LAG1'):
        return f'{scenario}__M{h}__FOCUS_{focus.upper()}_ROWS_60_INTERVALS_24__{variant}'
    def pack(name):return tests.get(name) or {}
    def links(name):return f'<a href="{e(name)}.summary.txt">TXT</a> · <a href="{e(name)}.summary.html">HTML</a>'
    def ci(x):return f'[{n(x.get("ci95_native_low"))}; {n(x.get("ci95_native_high"))}]'
    def table(head,rows):
        return '<div class="scroll"><table><thead><tr>'+''.join(f'<th>{h}</th>' for h in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{x}</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    main_models={};sample_rows=[];coefficient_rows=[];contrast_rows=[];flow_rows=[];comparison_rows=[]
    for focus in CORE:
        for h in [1,2,3]:
            name=label(focus,h);m=byname[name];current=coef(name,'align_current');lag=coef(name,'align_lag1')
            test=pack(name);joint=test.get('joint_current_lag_zero',{});total=test.get('sum_current_lag',{})
            main_models[name]=dict(model=m,current=current,lag1=lag,tests=test)
            sample_rows.append([FOCUS[focus],h,integer(m.get('nobs')),integer(m.get('funds')),integer(m.get('companies')),
                f'{str(m.get("first_start_date") or "—")[:10]} — {str(m.get("last_end_date") or "—")[:10]}',
                n(m.get('rsquared_within'),6),f'{integer(m.get("achieved_min_fund_rows"))} / {integer(m.get("achieved_min_fund_intervals"))}',links(name)])
            coefficient_rows.append([FOCUS[focus],h,n(current['beta']),n(current['se_native']),ci(current),pv(current['p_native']),
                n(lag['beta']),n(lag['se_native']),ci(lag),pv(lag['p_native'])])
            contrast_rows.append([FOCUS[focus],h,n(joint.get('statistic')),pv(joint.get('p_value')),n(total.get('beta')),
                n(total.get('se_native')),ci(total),pv(total.get('p_native')),e(joint.get('status','NOT_AVAILABLE')),e(total.get('status','NOT_AVAILABLE'))])
            flow_rows.append([FOCUS[focus],h,integer(m.get('old_rows')),integer(m.get('n_cutoff_removed')),
                integer(m.get('n_future_or_lag_dropped')),integer(m.get('n_abs_dar_removed')),
                integer(m.get('n_abs_dar_removed_entries')),integer(m.get('n_abs_dar_removed_exits')),integer(m.get('nobs'))])
            for variant in VARS:
                other=label(focus,h,variant=variant);r=byname[other];c=coef(other,'align_current');l=coef(other,'align_lag1')
                assert r['row_hash']==m['row_hash'] and r.get('nobs')==m.get('nobs')
                comparison_rows.append([FOCUS[focus],h,VARS[variant],integer(r.get('nobs')),n(c['beta']),pv(c['p_native']),
                    n(l['beta']),pv(l['p_native']),n(r.get('rsquared_within'),6),links(other)])
    cutoff_rows=[]
    for cut in spec['cutoffs']:
        outcomes=cut.get('outcome_abs_cutoffs_by_horizon',{})
        cutoff_rows.append([cut['tag'],n(cut['cutoff_percent'],6),
            *[n(outcomes.get(str(h)),6) if outcomes else 'Без DAR trim' for h in [1,2,3]],e(cut['analysis_role'])])
    sensitivity_rows=[]
    for scenario in SCENARIOS:
        for focus in CORE:
            for h in [1,2,3]:
                name=label(focus,h,scenario);m=byname[name];c=coef(name,'align_current');l=coef(name,'align_lag1');t=pack(name)
                total=t.get('sum_current_lag',{})
                sensitivity_rows.append([scenario,FOCUS[focus],h,integer(m.get('nobs')),integer(m.get('funds')),integer(m.get('companies')),
                    n(c['beta']),pv(c['p_native']),n(l['beta']),pv(l['p_native']),pv(t.get('joint_current_lag_zero',{}).get('p_value')),
                    n(total.get('beta')),pv(total.get('p_native')),n(m.get('rsquared_within'),6),links(name)])
    warning_by_model=collections.defaultdict(list)
    for w in qa['warnings']:
        if w.get('model'):warning_by_model[w['model']].append(w['kind'])
    master=[]
    for m in models:
        name=m['model'];c=coef(name,'align_current');l=coef(name,'align_lag1');t=pack(name);total=t.get('sum_current_lag',{})
        cells=[e(name),FOCUS[m.get('fund_focus')],m['horizon'],e(m['status']),integer(m.get('nobs')),integer(m.get('funds')),integer(m.get('companies')),
            n(c['beta']),pv(c['p_native']),n(l['beta']),pv(l['p_native']),pv(t.get('joint_current_lag_zero',{}).get('p_value')),
            n(total.get('beta')),pv(total.get('p_native')),n(m.get('rsquared_within'),6),
            f'{integer(m.get("achieved_min_fund_rows"))} / {integer(m.get("achieved_min_fund_intervals"))}',
            '<span class="tiny">'+e(', '.join(sorted(set(warning_by_model[name]))))+'</span>',links(name)]
        master.append(f'<tr data-scenario="{m["cutoff_tag"]}" data-focus="{m.get("fund_focus") or "all"}" data-horizon="{m["horizon"]}" data-variant="{m["variant"]}">'+
            ''.join(f'<td class="model">{value}</td>' if i==0 else f'<td>{value}</td>' for i,value in enumerate(cells))+'</tr>')
    warnings=dict(collections.Counter(w['kind'] for w in qa['warnings']))
    qa_format_fix=(dict(selected_version=qa_name,scope='QA date-format comparison only',
        explanation='v1 converted Date32/Python date to Timestamp before literal comparison; v2 removes that QA-side mutation. Actual ns dates and numerical replications are unchanged.',
        failed_v1_preserved_as='qa_independent.json',samples_or_coefficients_recomputed=False)
        if qa_name=='qa_independent_v2.json' else None)
    qa_format_note=('<p class="meta">Используется QA v2: исправлено только сравнение форматов дат внутри самой проверки — без преобразования Date32/Python date в Timestamp перед буквальным сравнением. Failed v1 сохранён как отдельный артефакт <code>qa_independent.json</code>. Из-за этого исправления ни одна выборка или коэффициент не пересчитывались.</p>'
        if qa_format_fix else '')
    statuses=dict(collections.Counter(m['status'] for m in models))
    calibration=OUT/'data/outcome_trim_calibration.json'
    calibration_json=json.loads(calibration.read_text()) if calibration.exists() else None
    native=[]
    for focus in CORE:
        for h in [1,2,3]:
            name=label(focus,h)
            native.append(f'<details><summary>{FOCUS[focus]} · {h} мес. · Q99_D99 · 60/24 · текущий + lag1</summary><p>{links(name)}</p><pre>{e((OUT/(name+".summary.txt")).read_text())}</pre></details>')
    heads=['Модель','Тип фонда','h','Статус','N','Фондов','Компаний','β текущего','p текущего','β lag1','p lag1','p joint','β суммы','p суммы','within R²','Факт. min строк / периодов','QA','Python']
    master_table='<div class="scroll"><table id="models"><thead><tr>'+''.join(f'<th>{x}</th>' for x in heads)+'</tr></thead><tbody>'+''.join(master)+'</tbody></table></div>'
    content=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Дополнительный DAR · будущая доходность бумаг сверх RUONIA · 03.11.2026</title>
<style>body{{margin:0;background:#f4f6f8;color:#172d3c;font:16px/1.55 system-ui,-apple-system,sans-serif}}main{{max-width:1400px;margin:auto;padding:34px 25px 75px}}h1{{font-size:34px;line-height:1.18}}h2{{font-size:24px;margin-top:36px}}p{{max-width:1150px}}a{{color:#07549b}}.eyebrow,.meta{{color:#526b7c;font-size:13px}}.notice{{background:#e8f0f7;border-left:5px solid #315d7b;border-radius:6px;padding:15px 22px;margin:18px 0}}.warn{{background:#fff0d8;border-color:#a46c20}}.scroll{{overflow:auto;background:white;border:1px solid #d5dfe7;border-radius:8px;margin:16px 0 24px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:9px 11px;text-align:right;white-space:nowrap;border-bottom:1px solid #e3eaf0;vertical-align:top}}th{{background:#e8eef4;position:sticky;top:0}}td:first-child,th:first-child{{text-align:left}}td.model{{white-space:normal;overflow-wrap:anywhere;max-width:350px;min-width:270px;font-size:11px}}tr:nth-child(even){{background:#fafbfd}}.tiny{{font-size:10px;white-space:normal;display:block;max-width:160px}}nav,.filters{{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}}input,select{{font:inherit;padding:8px;border:1px solid #a1b3c1;border-radius:5px;background:white}}input{{min-width:260px;flex:1}}details{{background:white;border:1px solid #d7e1e8;border-radius:7px;padding:15px 18px;margin:12px 0}}summary{{font-weight:650;cursor:pointer}}pre{{overflow:auto;background:#f6f8fa;padding:15px;font:12px/1.5 ui-monospace,monospace}}code{{overflow-wrap:anywhere}}.pill{{display:inline-block;padding:5px 10px;border:1px solid #a1b4c3;border-radius:20px;margin:4px 8px 4px 0;font-size:13px}}@media(max-width:600px){{main{{padding:22px 13px}}h1{{font-size:27px}}}}</style><main>
<div class="eyebrow">03 ноября 2026 · дополнительный механизм · расчёты на домашнем ПК</div>
<h1>Связаны ли новости и направление изменения позиции<br>с будущей ценовой доходностью бумаги сверх RUONIA?</h1>
<div class="notice warn"><p><strong>Основная H3 о будущей alpha фонда остаётся неоценимой: подходящих серий — 0.</strong> В зарегистрированных данных максимум 3 наблюдаемых месячных доходности на фондовую серию вместо принятых 36; даже весь сохранённый, частично недокументированный fund_ret даёт максимум 10. Этот отчёт показывает отдельный дополнительный механизм DAR на уровне бумаги, а не alpha фонда.</p></div>
<p><span class="pill">1080 обозначений моделей</span><span class="pill">QA: {integer(qa['pass_count'])} PASS / {integer(qa['fail_count'])} FAIL</span><span class="pill">h=1, 2, 3 месяца</span><span class="pill">Без поправок на множественные тесты</span></p>
<nav><a href="#main">Q99_D99: основные таблицы</a><a href="#samples">Потери выборки</a><a href="#sensitivity">Все пять сценариев</a><a href="#all">Все 1080 моделей</a><a href="#native">Оригинал Python</a><a href="#limitations">Источники и ограничения</a></nav>
<h2>Как построены переменные</h2><p><strong>DAR=100 × sign(ΔQ) × (Pfuture/Pclosing − 1 − RF).</strong> Положительное значение означает: увеличенная позиция затем росла сильнее денежного ориентира либо уменьшенная позиция отставала от него. Это не наблюдаемая прибыль сделки: не используется размер сделки, цена её исполнения или денежный портфель фонда.</p>
<p>Текущая объясняющая переменная — sign(ΔQ)×Net текущего отчётного интервала. Lag1 — sign(ΔQ)×Net той же компании за точный предыдущий интервал фонда. Обе переменные используют направление текущего расчётного изменения Q. Y и X поэтому содержат общий множитель направления; результат не выделяет самостоятельно торговое мастерство фонда или причинный эффект новостей.</p>
<p>Целевая дата — фактическая дата закрытия отчёта плюс 1/2/3 календарных месяца. Будущая котировка строго после закрытия, не позже целевой даты и не старше 7 дней; используется тот же ISIN/SECID/BOARDID. Обе цены нормированы в RUB: акции CLOSE, облигации чистая CLOSE×FACEVALUE/100. Купоны, выплаты, амортизация и корпоративные действия комплексно не восстановлены.</p>
<p>RF=RUONIA_Index(целевая дата)/RUONIA_Index(дата закрытия)−1, с зафиксированным сдвигом дат 0. Готовый официальный накопленный индекс учитывает календарные дни; годовая ставка не делится на 12, пропуски не заполняются. Источник и методика: <a href="{CBR}">Банк России — руководство по индексу RUONIA</a>.</p>
<p class="meta">Fund FE; кластеры фонд/компания; без time FE и временных кластеров. Те же 6 микро- и 10 макроконтролей. Тип фонда — на начало фактического интервала, может меняться. Требования истории применены до дополнительных исключений; фактические минимумы ниже показаны явно.</p>
<h2 id="main">Q99_D99 · 60 строк И 24 интервала · текущий + lag1</h2>
<p>Эта заранее обозначенная дополнительная серия сохраняет прежний quantity-P99 и исключает будущий |DAR| выше его P99 на уникальных ценовых окнах. Она не выбрана по минимальному p и не подтверждает автоматически исходную H3.</p>
{table(['Тип фонда','h, мес.','N','Фондов','Компаний','Окно исходных отчётов','within R²','Факт. min строк / интервалов','Python'],sample_rows)}
<h3>Коэффициенты и доверительные интервалы</h3><p>β — п.п. будущей ценовой доходности сверх RUONIA с учётом направления ΔQ на +1 соответствующего alignment. Все p двусторонние, без корректировок.</p>
{table(['Тип фонда','h','β текущего','SE','95% CI','p текущего','β lag1','SE lag1','95% CI lag1','p lag1'],coefficient_rows)}
<h3>Совместный тест и сумма коэффициентов</h3>
{table(['Тип фонда','h','Wald χ²(2)','p current=lag=0','β current+lag','SE суммы','95% CI суммы','p суммы','Статус joint','Статус суммы'],contrast_rows)}
<p>Сумма использует полную cross-covariance текущего и lag-коэффициента. Это линейный контраст двух экспозиций, а не структурный долгосрочный эффект. Неопределённые контрасты показаны пропусками и статусом; covariance не исправляется. <a href="distributed_tests.json">Полные матрицы и контрасты</a>.</p>
<h3>Сравнение на одних строках</h3>{table(['Тип фонда','h','Вариант','N','β текущего','p текущего','β lag1','p lag1','within R²','Python'],comparison_rows)}
<h2 id="samples">Какие наблюдения исключены</h2>{table(['Тип фонда','h','Исходная маска','Удалено quantity','Нет future/lag','Удалено |DAR|','В том числе входов','Выходов','Итог N'],flow_rows)}
<p>Нулевое или неизвестное ΔQ не допускается к этому дополнительному outcome. Нет будущей котировки, длительное отсутствие торгов, погашение без будущей цены или отсутствующий индекс дают NA, а не ноль. Подтверждённые входы/выходы сохраняются при quantity-отсечении, но могут удаляться последующим отсечением |DAR|.</p>
<h2 id="sensitivity">Пять сценариев: количество и будущий outcome</h2><p>Пороги quantity рассчитаны до условных входов по первоначальным продолжающимся позициям. Пороги |DAR| рассчитаны отдельно по каждому горизонту на уникальных точных ценовых окнах, до типов фондов, support, lag и новых quantity-фильтров. Исключение строго выше порога; совпадения сохраняются. Конфликтов P0/P1/RF/|DAR| внутри ключа не было. <a href="data/outcome_trim_calibration.json">Калибровка, N окон и хеши</a>.</p>
{table(['Сценарий','Верхний quantity Y, %','|DAR| h1, п.п.','|DAR| h2, п.п.','|DAR| h3, п.п.','Роль'],cutoff_rows)}
<div class="notice warn"><p><strong>Q99_RAW — диагностика необработанных ценовых масштабов.</strong> Например, ВТБ: 0,02114 RUB в июне 2024 и 96,73 RUB в июле не образуют сопоставимую доходность без учёта консолидации 5000:1. Событие подтверждено <a href="{MOEX}">Московской биржей</a>. Известные raw-примеры уходят уже при DAR-P99; <a href="data/vtbr_raw_price_break_examples.json">исходные строки</a> сохранены. Quantity-P99 сам по себе эти будущие price-scale разрывы не удаляет.</p><p>Отсечение DAR не является полной корпоративной корректировкой: и направление ΔQ из V/P также может быть искажено. Не каждое исключённое экстремальное наблюдение доказано ошибкой.</p></div>
{table(['Сценарий','Тип фонда','h','N','Фондов','Компаний','β текущего','p текущего','β lag1','p lag1','p joint','β суммы','p суммы','within R²','Python'],sensitivity_rows)}
<h2 id="all">Все 1080 обозначений моделей и статусы</h2><p>36 исходных масок × 5 сценариев × 3 горизонта × 2 варианта. Повторные labels с одинаковой выборкой и дизайном могут использовать один fit. Показаны все статусы, в том числе неоценимые модели и неопределённая inference.</p>
<div class="filters"><input id="search" aria-label="Поиск модели" placeholder="Поиск модели или 60_INTERVALS_24"><select id="scenario" aria-label="Сценарий"><option value="">Все сценарии</option>{''.join(f'<option>{s}</option>' for s in SCENARIOS)}</select><select id="focus" aria-label="Тип фонда"><option value="">Все типы</option value="all">Общая выборка</option>{''.join(f'<option value="{f}">{FOCUS[f]}</option>' for f in FOCUS if f)}</select><select id="horizon" aria-label="Горизонт"><option value="">Все горизонты</option><option>1</option><option>2</option><option>3</option></select><select id="variant" aria-label="Вариант"><option value="">Оба варианта</option value="CURRENT_COMMON">Текущий</option><option value="CURRENT_PLUS_LAG1">Текущий + lag1</option></select></div><p id="counter" class="meta">Показано: 1080 / 1080</p>{master_table}
<h2 id="native">Оригинальный Python output Q99_D99 · 60/24</h2>{''.join(native)}
<h2 id="limitations">Проверяемость и ограничения</h2><p><strong>{e(qa['status'])}: {integer(qa['pass_count'])} PASS / {integer(qa['fail_count'])} FAIL.</strong> Независимо численно воспроизведены {len(qa['independent_replication_models'])} заранее обозначенных моделей; остальные прошли механические проверки данных, масок и статистической арифметики. Полная независимая переоценка всех 1080 labels не заявляется. <a href="{e(qa_name)}">Все проверки и предупреждения: {e(qa_name)}</a>.</p>{qa_format_note}
{table(['Предупреждение','Записей'],[[e(k),integer(v)] for k,v in warnings.items()])}
<p>Денежные и прочие узкие группы могут иметь мало кластеров, неоценимые модели или неопределённые SE/robust F. p не заполняются искусственными значениями; значимость одного горизонта или одной модели не выбирается как подтверждение всей серии.</p>
<p>Этот proxy относится к последующему движению цены выбранной бумаги относительно денежного ориентира. Он не является risk-adjusted alpha, доходностью всего фонда, наблюдаемым PnL, исходным fund-level NTS или причинным доказательством медленного включения новостей в цену. Цена облигации — чистая; отсутствие купонов и погашений особенно ограничивает её экономический смысл. Одно направление уменьшения позиции не означает наблюдаемую короткую продажу.</p>
<details><summary>Файлы, источники и воспроизводимость</summary><p><a href="plan.json">План до коэффициентов</a> · <a href="source_manifest.json">Источники и SHA</a> · <a href="comparison.csv">Все метрики</a> · <a href="coefficients.csv">Все коэффициенты</a> · <a href="sample_flow.csv">Потери выборки</a> · <a href="mask_catalog.json">Маски и row hashes</a> · <a href="outcome_audit.json">Аудит будущего outcome</a> · <a href="summary.json">Числовой summary</a></p><p>Исходный dataset и все точные Parquet-выборки остаются на PC: <code>{e(spec['output_root'])}/samples/*.parquet</code>. Эта страница использует только завершённые небольшие таблицы результатов; исходные Q, новости, цены, live LLM и прошлые расчёты не меняет.</p><p>Прежний summary marker independent_QA=PENDING сохранён как история момента расчёта. Этот отчёт использует более поздний фактический файл независимого QA.</p></details></main>
<script>const rs=[...document.querySelectorAll('#models tbody tr')];function filter(){{const q=document.querySelector('#search').value.toLowerCase();let n=0;rs.forEach(r=>{{const ok=r.textContent.toLowerCase().includes(q)&&['scenario','focus','horizon','variant'].every(k=>!document.getElementById(k).value||r.dataset[k]===document.getElementById(k).value);r.hidden=!ok;if(ok)n++;}});document.getElementById('counter').textContent=`Показано: ${{n}} / 1080`;}}['search','scenario','focus','horizon','variant'].forEach(k=>document.getElementById(k).addEventListener('input',filter));</script></html>'''
    target=OUT/'review.html';target.write_text(content,encoding='utf-8')
    class LinkParser(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):
            self.links.extend(v for k,v in attrs if k=='href')
    parser=LinkParser();parser.feed(content)
    local=[href for href in parser.links if not href.startswith(('#','https:','http:'))]
    missing=[href for href in local if not (OUT/unquote(href.split('#')[0])).resolve().exists()]
    assert not missing,missing
    assert all(sha(OUT/name)==value for name,value in before.items())
    report=dict(status='PASS_SUPPLEMENTARY_DAR_RESULT_REVIEW_WITH_PRESERVED_WARNINGS',created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        no_new_fits=True,no_bulk_data_read=True,source_sha256=before,source_files_unchanged=True,
        core_H3_alpha_status=summary['core_H3_alpha_status'],core_H3_eligible_fund_series=0,not_a_replacement_for_core_H3=True,
        model_labels=len(models),model_status_counts=statuses,main_scenario='Q99_D99',main_support='60 rows AND24 intervals before subsequent filters',
        main_models=main_models,scenarios=spec['cutoffs'],common_vs_distributed_same_rows_verified=True,
        multiple_testing_adjustments=False,QA=dict(file=qa_name,sha256=before[qa_name],status=qa['status'],pass_count=qa['pass_count'],fail_count=qa['fail_count'],
            independent_replication_models=qa['independent_replication_models'],warning_counts=warnings,warnings=qa['warnings'],limitations=qa['limitations']),
        QA_format_only_fix=qa_format_fix,
        unique_window_calibration_sha256=sha(calibration) if calibration.exists() else None,
        official_citations={'RUONIA_methodology':CBR,'VTBR_consolidation':MOEX},
        checks=dict(all1080statuses_visible=len(master)==1080,local_links=len(local),missing_links=missing,
            raw_Q99_diagnostic_labelled=True,quantity_direction_corporate_action_limitation_disclosed=True),
        report=str(target),report_sha256=sha(target))
    REVIEW.write_text(json.dumps(clean(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],html=str(target),review=str(REVIEW),model_labels=len(models),links=len(local)),ensure_ascii=False))


if __name__=='__main__':main()
