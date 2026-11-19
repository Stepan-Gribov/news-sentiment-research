#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Readable presentation of existing PC results; no estimation or data changes."""
from pathlib import Path
import csv
import html
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/h1_focus_joint_20261102'
rows = list(csv.DictReader((OUT / 'comparison.csv').open(encoding='utf-8-sig')))
qa = json.loads((OUT / 'qa_independent.json').read_text(encoding='utf-8'))
assert qa['fail_count'] == 0
labels = {'equity': 'Фонды акций', 'bond': 'Фонды облигаций',
          'mixed': 'Смешанные фонды', 'money_market': 'Денежный рынок', 'other': 'Прочие'}


def number(value, digits=4):
    try:
        return f'{float(value):,.{digits}f}'.replace(',', ' ').replace('.', ',')
    except (ValueError, TypeError):
        return '—'


def table(items):
    headings = ['Группа / порог истории', 'Строки', 'Фонды', 'Эмитенты', 'β Net',
                'SE', '95% ДИ', 'p двуст.', 'p Holm', 'Within R²', 'Python']
    result = ['<div class="scroll"><table><thead><tr>']
    result += [f'<th>{heading}</th>' for heading in headings]
    result += ['</tr></thead><tbody>']
    for row in items:
        model = row['model']
        rule = 'без порога истории'
        if '_ROWS_' in model:
            rule = model.split('_ROWS_')[1].replace('_INTERVALS_', ' строк и ') + ' интервалов'
        title = labels.get(row['fund_focus'], 'Все фонды') + ': ' + rule
        values = [title, number(row['nobs'], 0), number(row['funds'], 0),
                  number(row['companies'], 0), number(row['beta']), number(row['se_native']),
                  f"[{number(row['ci95_native_low'])}; {number(row['ci95_native_high'])}]",
                  number(row['p_native'], 6), number(row['p_native_holm_focus_unique'], 6),
                  number(row['rsquared_within'])]
        positive = row['p_native'] and float(row['p_native']) < .05 and float(row['beta']) > 0
        result.append('<tr class="positive">' if positive else '<tr>')
        result += [f'<td>{html.escape(value)}</td>' for value in values]
        result.append(f'<td><a href="{model}.summary.txt">TXT</a> · '
                      f'<a href="{model}.summary.html">HTML</a></td></tr>')
    result.append('</tbody></table></div>')
    return ''.join(result)


parts = ['''<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>H1: типы фондов и история наблюдений</title>
<style>body{font:16px/1.55 system-ui,sans-serif;color:#172b3a;max-width:1400px;
margin:32px auto;padding:0 24px;background:#f6f8fa}h1,h2{line-height:1.2}
section{background:white;border:1px solid #dce3e9;border-radius:10px;padding:22px;
margin:20px 0}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;
font-size:14px}td,th{padding:10px;border-bottom:1px solid #dae1e7;text-align:right;
white-space:nowrap}td:first-child,th:first-child{text-align:left;white-space:normal;
min-width:210px}th{background:#eaf0f5}.positive{background:#eaf8ed}.scroll{overflow-x:auto}
pre{overflow:auto;font-size:13px;background:#f1f4f6;padding:20px}a{color:#165c9b}
code{overflow-wrap:anywhere}.muted{color:#526373}.lead{font-size:20px}</style>
<h1>H1: типы фондов и совместные пороги истории</h1>
<p class="muted">Расчёт на research-host, 2 ноября 2026 года. 36 обозначений моделей,
20 новых уникальных оценок, 4 проверенных повторных использования прежних оценок.
Остальные — совпадающие или неоцениваемые выборки. Время расчёта 20,65 секунды.</p>
<section><p class="lead"><b>В фондах облигаций положительная связь имеет номинальное
двустороннее p=0,007129</b> при ≥60 строках и ≥24 отчётных интервалах на фонд
внутри этой группы.</p><p>β=1,087431; SE=0,404114; 95% ДИ [0,295360; 1,879501].
N=46 308; 66 фондов; 263 эмитента. Within R²=0,054062. Начало первого окна
30.09.2013, конец последнего 29.05.2026; наблюдения нерегулярные.</p>
<p>Во всех шести сочетаниях порогов облигационной группы номинальное p&lt;0,05.
Два варианта с 36 интервалами совпадают по строкам. Это разведочный подгрупповой
результат. После Holm по 25 уникальным текущим подгрупповым выборкам p=0,178214;
семейная значимость 5% не получена. Различие коэффициентов между типами фондов
отдельным тестом взаимодействия не проверялось.</p></section>
<section><h2>Что означает строка данных</h2>
<p><b>Фонд × конкретная ценная бумага × фактический интервал между двумя раскрытиями.</b>
Y — изменение расчётного количества бумаги. Для положительной начальной позиции:
100×(Q1/Q0−1). Q рассчитано из стоимости позиции и цены. Подтверждённый переход
0→положительное условно равен +100%; полное выбытие −100%.</p>
<p>Сентимент относится к <b>эмитенту × точному окну (d0,d1]</b>:
(Npositive−Nnegative)/(Npositive+Nneutral+Nnegative). Для бумаг одного эмитента
в одинаковом окне скор общий, в том числе между фондами. В исходной выборке
221 899 строк, но 9 227 уникальных сигналов компания–окно. Отсутствие пригодного
скора остаётся пропуском, не нулём.</p>
<p>147 219 исходных строк относятся к соседним календарным месяцам с интервалом
20–40 дней; 74 680 — к другим фактическим интервалам. Вся панель поэтому не
называется месячной. Тип фонда берётся из классификации начального отчёта и
может меняться по истории; это не класс конкретной бумаги. Числа фондов по
группам не складываются в число уникальных фондов.</p>
<p>Модель: Y = фиксированный эффект фонда + β×NetSentiment + 6 микроконтролей +
10 макроконтролей + ошибка. Кластеры: фонд и компания; временных эффектов и
временной кластеризации нет. Общий прежний верхний P99=322,777164% для
продолжающихся позиций не пересчитывался по группам. Выявленные индексные фонды
исключены прежним правилом; полная независимая верификация активности остальных
не заявляется.</p><p>Рост Net на 0,1 связан в модели 60/24 с ростом Y на 0,108743
процентного пункта. Это условная статистическая связь; Q — оценка количества,
а не фактически раскрытые сделки.</p></section>''']
parts += ['<section><h2>Типы фондов без дополнительных порогов истории</h2>',
          table([row for row in rows if row['model'].startswith('REFERENCE')
                 or row['model'].endswith('BASELINE')]),
          '''<p class="muted">У денежного рынка коэффициент отрицательный —
противоположен H1; всего 18 фондовых кластеров. В узких вариантах их 3–5 и
имеются проблемы SE отдельных controls и robust F. Эти результаты требуют
осторожности. Все предупреждения сохранены в QA.</p></section>''']
parts += ['<section><h2>Фонды облигаций: все проверенные сочетания</h2>',
          table([row for row in rows if row['fund_focus'] == 'bond']),
          '''<p>Число строк и интервалов считается один раз на полной облигационной
подвыборке каждого фонда, затем пороги применяются одновременно. 60 строк не
означает 60 отчётных дат. Формула Y, контроли и оценка ошибок одинаковы.</p></section>''']
parts += ['<section><h2>Оригинальный вывод Python: облигационные фонды, 60/24</h2><pre>',
          html.escape((OUT / 'FOCUS_BOND_ROWS_60_INTERVALS_24.summary.txt').read_text(encoding='utf-8')),
          '</pre></section><section><h2>Все 36 обозначений моделей</h2>', table(rows), '</section>']
parts += ['''<section><h2>Проверяемость</h2><p>Независимый QA: <b>628 PASS, 0 FAIL</b>.
Самостоятельная численная репликация within OLS и двухкластерной ковариации
выполнена для облигационной базовой модели и варианта 60/24. Для остальных
проверены данные, маски, числа наблюдений, статистики и поправки. Все 36 моделей
не заявляются независимо переоценёнными. Предупреждения относятся к малым группам
денежного рынка/прочим, а не к указанным облигационным моделям.</p>
<ul><li><a href="comparison.csv">Сводная таблица CSV</a> ·
<a href="coefficients.csv">Все коэффициенты и контроли</a></li>
<li><a href="focus_fund_support.csv">Число строк и интервалов по фондам внутри групп</a> ·
<a href="type_coverage.csv">Покрытие типов фондов</a></li>
<li><a href="issuer_period_example.csv">Пример общего скора в разных бумагах и фондах</a> ·
<a href="news_signal_level_audit.json">Аудит агрегации и формулы</a></li>
<li><a href="qa_independent.json">Независимый QA</a> · <a href="plan.json">План</a> ·
<a href="source_manifest.json">Источники и контрольные суммы</a> ·
<a href="multiplicity.json">Поправки Holm</a></li></ul>
<p>Точные построчные данные выборок находятся на ПК:
<code>./analysis_outputs/h1_focus_joint_20261102/samples/*.parquet</code>.
Защищённые источники, исходные Q, новости и прежние результаты сохранены.</p></section></html>''']
(OUT / 'review.html').write_text(''.join(parts), encoding='utf-8')
shutil.copy2(OUT / 'qa_independent.json', ROOT / 'reports/h1_focus_joint_independent_qa_20261102.json')
print(json.dumps({'status': 'OK', 'models': len(rows), 'QA_pass': qa['pass_count'],
                  'report': str(OUT / 'review.html')}, ensure_ascii=True))
