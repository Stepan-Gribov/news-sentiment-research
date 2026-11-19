#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Present already computed PC estimates; this file never fits a model."""
from pathlib import Path
import csv
import html
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/h1_sentiment_lag_20261103'
LABELS = {'equity': 'Акций', 'bond': 'Облигаций', 'mixed': 'Смешанные',
          'money_market': 'Денежного рынка', 'other': 'Прочие', '': 'Все типы'}


def read_csv(name):
    return list(csv.DictReader((OUT / name).open(encoding='utf-8-sig')))


def number(value, digits=4):
    try:
        return f'{float(value):,.{digits}f}'.replace(',', ' ').replace('.', ',')
    except (TypeError, ValueError):
        return '—'


def rule(model):
    name = model.split('__')[0]
    if '_ROWS_' in name:
        rows, periods = name.split('_ROWS_')[1].split('_INTERVALS_')
        return f'≥{rows} строк и ≥{periods} интервалов'
    return 'Без порога истории'


def table(headings, rows):
    result = ['<div class="scroll"><table><thead><tr>']
    result += [f'<th>{html.escape(h)}</th>' for h in headings]
    result += ['</tr></thead><tbody>']
    for values in rows:
        result.append('<tr>' + ''.join(f'<td>{v}</td>' for v in values) + '</tr>')
    result.append('</tbody></table></div>')
    return ''.join(result)


def lag_table(items):
    values = []
    for row in items:
        name = row['model']
        values.append([
            LABELS.get(row['fund_focus'], row['fund_focus']), html.escape(rule(name)),
            number(row['nobs'], 0), number(row['funds'], 0), number(row['companies'], 0),
            number(row['current_beta']), number(row['current_se_native']),
            number(row['current_p_native'], 6), number(row['lag1_beta']),
            number(row['lag1_se_native']), number(row['lag1_p_native'], 6),
            number(row['sum_beta']), number(row['sum_p'], 6), number(row['joint_p'], 6),
            number(row['rsquared_within']),
            f'<a href="{name}.summary.txt">TXT</a> · <a href="{name}.summary.html">HTML</a>'
        ])
    return table(['Фонды', 'Исходный отбор', 'N', 'Фондов', 'Эмитентов', 'β текущий',
                  'SE текущий', 'p текущий', 'β лаг', 'SE лаг', 'p лаг', 'β сумма',
                  'p сумма', 'p совместный', 'Within R²', 'Python'], values)


def main():
    summary = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    qa = json.loads((OUT / 'qa_independent.json').read_text(encoding='utf-8'))
    assert qa['fail_count'] == 0
    comparison = read_csv('comparison.csv')
    old = read_csv('old_metrics.csv')
    distributed = [r for r in comparison if r['variant'] == 'CURRENT_PLUS_LAG1']
    core = [r for r in distributed if r['fund_focus'] in ('equity', 'bond', 'mixed')
            and '_ROWS_60_INTERVALS_24__' in r['model']]
    parts = ['''<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>H1: текущий сентимент и лаг</title>
<style>body{font:16px/1.55 system-ui,sans-serif;color:#172b3a;max-width:1500px;
margin:30px auto;padding:0 24px;background:#f5f7fa}h1,h2{line-height:1.2}
section{background:white;border:1px solid #dce3e9;border-radius:10px;padding:22px;
margin:20px 0}table{border-collapse:collapse;width:100%;font-size:14px;
font-variant-numeric:tabular-nums}td,th{padding:9px;border-bottom:1px solid #dae1e7;
text-align:right;white-space:nowrap}td:first-child,th:first-child{text-align:left}
th{background:#eaf0f5}.scroll{overflow-x:auto}pre{overflow:auto;font-size:12px;
background:#f1f4f6;padding:18px}a{color:#165c9b}.muted{color:#526373}
code{overflow-wrap:anywhere}li{margin:8px 0}.formula{font-size:20px}</style>
<h1>H1: текущий сентимент и один отчётный лаг</h1>
<p class="muted">3 ноября 2026 года. Данные, агрегация и оценки выполнены на домашнем
ПК research-host по SSH. Здесь показаны обычные двусторонние p-values.</p>
<section><h2>Основное сравнение: фонды с исходной историей ≥60 строк и ≥24 интервалов</h2>''',
             lag_table(core)]
    parts.append('''<p>Отбор фондов зафиксирован на прежней полной выборке внутри каждого
типа. Затем удалены строки без пригодного лага. Минимальное фактическое число
строк/интервалов после этого может быть ниже исходного порога; все числа есть
в <a href="sample_flow.csv">таблице отбора</a>.</p></section>
<section><h2>Что добавлено в регрессию</h2>
<p class="formula">ΔQ% = α фонда + β₀ Net текущего интервала + β₁ Net предыдущего
интервала + γ контроли + ошибка</p>
<p>Строка — фонд × конкретная бумага × интервал между двумя отчётами. Net —
сентимент эмитента за точное окно (d0,d1]. Его лаг — сентимент того же эмитента
за непосредственно предыдущий интервал этого фонда (d−1,d0]. Предыдущая бумага
не обязана присутствовать в портфеле; даты и связь отчётов проверены по полному
календарю. Интервал не обязательно равен месяцу.</p>
<p>Y — прежнее изменение расчётного количества Q=V/P: 100×(Q1/Q0−1) при Q0&gt;0;
подтверждённый вход из нуля условно +100%, выбытие −100%. Прежний общий P99,
шесть микроконтролей и десять макроконтролей сохранены. Фиксированные эффекты
фондов; кластеры фонд/компания; временных эффектов и кластеров нет.</p>
<p>Сумма β₀+β₁ проверяется с учётом ковариации двух оценок. Совместный Wald проверяет
нулевую гипотезу β₀=β₁=0. Значимость совместного теста сама по себе не задаёт знак
связи. Лаговая связь также не устанавливает скорость включения новостей в цены.</p>
<p>Для каждого исходного отбора оценены две модели на одинаковых строках:
CURRENT_COMMON — только текущий Net; CURRENT_PLUS_LAG1 — текущий Net и лаг.
Так можно отделить изменение спецификации от изменения состава выборки.
Отсутствующий скор не заменён нулём; нулевой наблюдаемый Net сохранён.</p></section>
<section><h2>Акционные и смешанные фонды до добавления лага</h2>''')
    old_values = []
    for row in old:
        if row['fund_focus'] not in ('equity', 'mixed'):
            continue
        old_values.append([
            LABELS[row['fund_focus']], html.escape(rule(row['model'])),
            number(row['nobs'], 0), number(row['funds'], 0), number(row['companies'], 0),
            number(row['beta']), number(row['se_native']), number(row['p_native'], 6),
            f"[{number(row['ci95_native_low'])}; {number(row['ci95_native_high'])}]",
            number(row['rsquared_within'])])
    parts += [table(['Фонды', 'Отбор', 'N', 'Фондов', 'Эмитентов', 'β', 'SE', 'p',
                     '95% ДИ', 'Within R²'], old_values), '</section>',
              '<section><h2>Все лаговые модели и все исходные правила отбора</h2>',
              lag_table(distributed), '</section>',
              '<section><h2>Точный вывод Python для трёх групп, исходный отбор 60/24</h2>']
    for row in core:
        parts += [f"<h3>Фонды {LABELS[row['fund_focus']].lower()}</h3><pre>",
                  html.escape((OUT / (row['model'] + '.summary.txt')).read_text(encoding='utf-8')),
                  '</pre>']
    parts += ['</section><section><h2>Данные, контроль и полные результаты</h2>',
              f"<p>Независимый QA: <b>{qa['pass_count']} PASS, {qa['fail_count']} FAIL</b>. "
              'Область независимой численной репликации и основные предупреждения перечислены '
              'в <a href="qa_independent.json">полном QA</a>. '
              'Дополнительно в десяти моделях денежного рынка общий robust F имеет '
              'неопределённую ковариационную интерпретацию; отрицательный F и его '
              'техническое native p=1 не трактуются как корректный тест. Эти статусы '
              'сохранены в summary.json. Основные три группы 60/24 этой проблемы '
              'не имеют.</p>',
              '<ul><li><a href="comparison.csv">Все 72 обозначения моделей, включая парные без лага</a></li>',
              '<li><a href="coefficients.csv">Все коэффициенты, SE, p и 95% ДИ</a> · '
              '<a href="distributed_tests.csv">Сумма и совместный тест</a></li>',
              '<li><a href="old_metrics.csv">Все прежние исходные метрики</a> · '
              '<a href="sample_flow.csv">Отбор и потеря наблюдений</a> · '
              '<a href="lag_sample_flow.json">Причины недоступного лага</a></li>',
              '<li><a href="regression_preview.csv">Читаемый пример использованных строк</a> · '
              '<a href="data_audit.json">Аудит построения лага</a></li>',
              '<li><a href="summary.json">Полные метрики</a> · '
              '<a href="source_manifest.json">Источники и хеши</a> · '
              '<a href="plan.json">Зафиксированная спецификация</a></li></ul>',
              '<p>Полные данные на ПК: '
              '<code>./analysis_outputs/h1_sentiment_lag_20261103/</code>; '
              'точные выборки в <code>samples/*.parquet</code>, скоры и построчная связь лага '
              'в <code>data/lag_company_windows.parquet</code> и '
              '<code>data/row_lag_map.parquet</code>. Защищённые входы, исходные новости '
              'и прежние результаты сохранены.</p></section></html>']
    (OUT / 'review.html').write_text(''.join(parts), encoding='utf-8')
    print(json.dumps({'report': str(OUT / 'review.html'), 'models': len(comparison),
                      'core_models': len(core)}, ensure_ascii=True))


if __name__ == '__main__':
    main()
