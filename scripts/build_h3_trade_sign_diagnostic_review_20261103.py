# -*- coding: utf-8 -*-
"""Presentation of nine completed shared-sign diagnostics, without new fits."""
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
OUT=ROOT/'outputs/h23_hypotheses_20261103/h3_trade_sign_diagnostic'
REVIEW=ROOT/'reports/h3_trade_sign_diagnostic_result_review_20261103.json'
FOCUS={'equity':'Акции','bond':'Облигации','mixed':'Смешанные'}


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
    if json.loads((OUT/'completion.json').read_text()).get('status')!='COMPLETE':raise RuntimeError('Diagnostic is not complete')
    qa_name='qa_independent_v2.json' if (OUT/'qa_independent_v2.json').exists() else 'qa_independent.json'
    qa=json.loads((OUT/qa_name).read_text())
    if qa.get('fail_count')!=0 or not qa.get('status','').startswith('PASS'):raise RuntimeError('Diagnostic QA has not passed')
    names=['completion.json','summary.json','coefficients.csv','comparison.csv','baseline_cache.json','distributed_tests.json','source_manifest.json','plan.json',qa_name]
    before={name:sha(OUT/name) for name in names}
    s=json.loads((OUT/'summary.json').read_text());spec=s['specification'];models=s['models']
    assert s['status']=='COMPLETE' and len(models)==9 and s['baseline_refit'] is False
    assert s['analysis_role']=='POST_RESULT_SHARED_SIGN_DIAGNOSTIC_NOT_PRIMARY_NOT_ALPHA'
    assert set(m['model'] for m in models)==set(spec['planned_models'])
    with (OUT/'coefficients.csv').open(encoding='utf-8-sig',newline='') as f:new={(r['model'],r['term']):r for r in csv.DictReader(f)}
    with (OUT/'comparison.csv').open(encoding='utf-8-sig',newline='') as f:comparison={r['model']:r for r in csv.DictReader(f)}
    fields=['beta','se_native','ci95_native_low','ci95_native_high','p_native']
    def metrics(row):
        return {k:float(row[k]) if row.get(k) not in (None,'') else None for k in fields}
    def cell(c):return f'<b>{n(c.get("beta"))}</b> (SE {n(c.get("se_native"))})<br><span class="small">CI [{n(c.get("ci95_native_low"))}; {n(c.get("ci95_native_high"))}]<br>p={pv(c.get("p_native"))}</span>'
    def number(row,key):
        try:return float(row[key])
        except (KeyError,TypeError,ValueError):return None
    def table(headers,rows):return '<div class="scroll"><table><thead><tr>'+''.join(f'<th>{h}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{v}</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table></div>'
    rows=[];contrast_rows=[];mean_rows=[];records={};native=[]
    for m in models:
        name=m['model'];oldname=m['baseline_model'];cache=s['baseline_cache'][oldname];old=cache['record'];r=comparison[name]
        assert m['same_ordered_rows_as_baseline'] is True and m['baseline_refit'] is False
        assert (m['nobs'],m['funds'],m['companies'])==(old['nobs'],old['funds'],old['companies'])
        baseline={x['term']:metrics(x) for x in cache['coefficients']}
        current=metrics(new.get((name,'align_current'),{}));lag=metrics(new.get((name,'align_lag1'),{}));sign=metrics(new.get((name,'trade_sign'),{}))
        newtest=s['distributed_tests'].get(name) or {};oldtest=cache.get('distributed_tests') or {}
        current_old=baseline['align_current'];lag_old=baseline['align_lag1'];oldsum=oldtest.get('sum_current_lag',{});newsum=newtest.get('sum_current_lag',{})
        records[name]=dict(model=m,baseline_model=old,baseline_current=current_old,baseline_lag=lag_old,
            controlled_current=current,controlled_lag=lag,trade_sign=sign,baseline_tests=oldtest,controlled_tests=newtest,
            descriptive_delta_current_beta=number(r,'descriptive_delta_current_beta'),descriptive_delta_lag_beta=number(r,'descriptive_delta_lag1_beta'),comparison=r)
        signcell='Поглощён fund FE; коэффициент не задан' if m.get('diagnostic_status')=='already_absorbed_equivalent' else cell(sign)
        source_links=f'<a href="{e(name)}.summary.txt">+s TXT</a> · <a href="{e(name)}.summary.html">+s HTML</a><br><a href="../h3_mechanism/{e(oldname)}.summary.txt">Baseline TXT</a>'
        rows.append([FOCUS[m['fund_focus']],m['horizon'],f'{integer(m["nobs"])} / {integer(m["funds"])} / {integer(m["companies"])}',
            f'{n(current_old["beta"])}<br>p={pv(current_old["p_native"])}',cell(current),
            f'{n(lag_old["beta"])}<br>p={pv(lag_old["p_native"])}',cell(lag),signcell,
            f'{n(number(r,"descriptive_delta_current_beta"))} / {n(number(r,"descriptive_delta_lag1_beta"))}',
            f'{n(old.get("rsquared_within"),6)} / {n(m.get("rsquared_within"),6)}',e(m.get('diagnostic_status')),source_links])
        contrast_rows.append([FOCUS[m['fund_focus']],m['horizon'],pv(oldtest.get('joint_current_lag_zero',{}).get('p_value')),
            pv(newtest.get('joint_current_lag_zero',{}).get('p_value')),f'{n(oldsum.get("beta"))}<br>p={pv(oldsum.get("p_native"))}',cell(newsum),
            e(newtest.get('joint_current_lag_zero',{}).get('status','NOT_AVAILABLE')),e(newsum.get('status','NOT_AVAILABLE'))])
        mean_rows.append([FOCUS[m['fund_focus']],m['horizon'],n(100*number(r,'mean_cash_excess_price_return')) if finite(number(r,'mean_cash_excess_price_return')) else '—',
            n(number(r,'mean_current_net')),n(number(r,'mean_lag_net')),integer(number(r,'positive_trade_rows')),integer(number(r,'negative_trade_rows'))])
        native.append(f'<details><summary>{FOCUS[m["fund_focus"]]} · {m["horizon"]} мес. · + trade_sign</summary><pre>{e((OUT/(name+".summary.txt")).read_text())}</pre></details>')
    index_nav='<a href="../index.html">Общий H2/H3 отчёт</a>' if (OUT.parent/'index.html').exists() else ''
    content=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Диагностика общего направления ΔQ · 9 моделей</title>
<style>body{{margin:0;background:#f4f6f8;color:#182e3e;font:16px/1.55 system-ui,-apple-system,sans-serif}}main{{max-width:1480px;margin:auto;padding:32px 24px 70px}}h1{{font-size:32px;line-height:1.2}}h2{{font-size:23px;margin-top:33px}}p{{max-width:1150px}}a{{color:#07569a}}.notice{{background:#fff0d9;border-left:5px solid #a26c24;padding:17px 22px;border-radius:7px}}.scroll{{overflow:auto;border:1px solid #d7e1e8;border-radius:7px;background:white;margin:16px 0 25px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:10px 12px;border-bottom:1px solid #e1e8ee;white-space:nowrap;text-align:right;vertical-align:top}}th{{background:#e8eef4}}td:first-child,th:first-child{{text-align:left}}.small,.meta{{color:#526b7d;font-size:12px}}details{{padding:15px 18px;background:white;border:1px solid #d7e1e9;border-radius:7px;margin:12px 0}}summary{{cursor:pointer;font-weight:650}}pre{{overflow:auto;background:#f6f8fa;padding:15px;font:12px/1.5 ui-monospace,monospace}}nav{{display:flex;gap:16px;flex-wrap:wrap}}@media(max-width:650px){{main{{padding:22px 13px}}h1{{font-size:26px}}}}</style><main>
<div class="meta">03 ноября 2026 · POST_RESULT_SHARED_SIGN_DIAGNOSTIC</div><h1>Общий знак направления позиции:<br>что меняет добавление trade_sign?</h1>
<nav>{index_nav}<a href="../h3_mechanism/review.html">Исходные 1080 DAR-моделей</a></nav>
<div class="notice"><p><strong>Отдельная диагностика после основного результата, не новая primary и не alpha фонда.</strong> На ровно девяти прежних Q99_D99/core60/24 выборках для h=1/2/3 добавлен один обязательный контроль s=sign(расчётногоΔQ). Исходные строки, Y, прежние X, 16 контролей, пороги, fund FE и кластеры фонд/компания сохранены. Прежние коэффициенты прочитаны из кэша без повторной оценки.</p></div>
<h2>Почему эта проверка нужна</h2><p>В исходном proxy Y=100×s×E, где E — будущая ценовая доходность сверх RUONIA; объясняющая переменная X=s×Net содержит тот же s. Даже при постоянном E=c результат Y=100c×s может быть связан с X через общий знак и средний уровень Net. Добавление s выделяет постоянный компонент направления/средней carry внутри этой модели.</p>
<p>Это не полное устранение риска, меняющегося carry, купонов, выплат, корпоративных действий или эндогенного выбора сделок. Сама Q и её направление оценены как V/P. Положительный коэффициент после добавления s не является целью отбора или доказательством alpha.</p>
<h2>Все девять результатов: baseline и добавление s</h2><p>N/F/C одинаковы с baseline: наблюдения / фонды / компании. Новые β/SE/CI — п.п. DAR на +1 соответствующего alignment. Δβ описательная; отдельный тест изменения коэффициента между моделями не проводился. Обычные двусторонние p, без множественных поправок.</p>
{table(['Тип фонда','h','N / F / C','Baseline текущий β / p','+s текущий β / SE / CI / p','Baseline lag β / p','+s lag β / SE / CI / p','s: β / SE / CI / p','Δβ current / lag','R² baseline / +s','Статус диагностики','Python'],rows)}
<h2>Совместный тест текущего и lag-коэффициента; их сумма</h2>{table(['Тип фонда','h','Baseline joint p','+s joint p','Baseline сумма β / p','+s сумма β / SE / CI / p','Статус joint','Статус суммы'],contrast_rows)}
<p>Сумма и совместный тест используют полную covariance внутри соответствующей модели. Они не являются тестом равенства коэффициентов между baseline и +s. Если s поглощён fund FE, сохраняется статус already_absorbed_equivalent; его коэффициент не заполняется нулём. Коллинеарная обязательная переменная даёт NOT_ESTIMABLE.</p>
<h2>Средние исходные компоненты</h2>{table(['Тип фонда','h','Средняя E, %','Средний текущий Net','Средний lag Net','s=+1 строк','s=−1 строк'],mean_rows)}
<p>Это описательные средние из тех же фиксированных выборок. Они не являются разложением доли объяснённого эффекта или причинной медиацией.</p>
<h2>Независимая проверка</h2><p><strong>{e(qa['status'])}: {integer(qa['pass_count'])} PASS / {integer(qa['fail_count'])} FAIL.</strong> Сохранены все девять моделей независимо от знаков и p, в том числе статусы неопределённой inference. <a href="{e(qa_name)}">Фактический QA</a>.</p>
<nav><a href="plan.json">План до коэффициентов этой диагностики</a><a href="comparison.csv">Сравнение</a><a href="coefficients.csv">Новые коэффициенты</a><a href="baseline_cache.json">Baseline cache</a><a href="distributed_tests.json">Контрасты</a><a href="source_manifest.json">Хеши неизменных исходников/выборок</a><a href="summary.json">Числовой summary</a></nav>
<h2>Оригинальный вывод Python</h2>{''.join(native)}<p class="meta">Страница только форматирует завершённые результаты. Основная серия 1080, её выборки и исходные числа этой диагностикой не заменяются.</p></main></html>'''
    target=OUT/'review.html';target.write_text(content,encoding='utf-8')
    class Parser(HTMLParser):
        def __init__(self):super().__init__();self.hrefs=[]
        def handle_starttag(self,tag,attrs):self.hrefs.extend(v for k,v in attrs if k=='href')
    parser=Parser();parser.feed(content)
    missing=[href for href in parser.hrefs if not (OUT/unquote(href.split('#')[0])).resolve().exists()]
    assert not missing,missing
    assert all(sha(OUT/name)==value for name,value in before.items())
    report=dict(status='PASS_POST_RESULT_SHARED_SIGN_PRESENTATION',created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        no_new_fits=True,baseline_refit=False,no_original_files_changed=True,analysis_role=s['analysis_role'],
        models=records,model_count=9,model_status_counts=dict(collections.Counter(m['status'] for m in models)),
        descriptive_delta_only=True,crossmodel_coefficient_change_test_performed=False,
        not_new_primary=True,not_fund_alpha=True,QA=dict(file=qa_name,sha256=before[qa_name],status=qa['status'],pass_count=qa['pass_count'],fail_count=qa['fail_count']),
        source_sha256=before,local_links=len(parser.hrefs),missing_links=missing,
        report=str(target),report_sha256=sha(target))
    REVIEW.write_text(json.dumps(clean(report),ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],html=str(target),review=str(REVIEW),models=9),ensure_ascii=False))


if __name__=='__main__':main()
