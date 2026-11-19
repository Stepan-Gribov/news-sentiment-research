"""Read-only synthesis of the complete H2 search; does not estimate models."""
from pathlib import Path
import csv
import hashlib
import html
import json
import math

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/h2_significance_search_20261104'
GROUPS = ['pooled', 'equity', 'bond', 'mixed']
LABELS = dict(pooled='Общая', equity='Акции', bond='Облигации', mixed='Смешанные')


def number(row, key):
    try:
        value = float(row[key])
        return value if math.isfinite(value) else None
    except (KeyError, ValueError, TypeError):
        return None


def compact(row):
    keys = ['model_id', 'family', 'group', 'variant', 'control_mask', 'support', 'cutoff',
            'outcome_tag', 'controls_included', 'nobs', 'funds', 'companies', 'rsquared_within',
            'row_hash', 'outcome_column']
    out = {k: row.get(k) for k in keys}
    for scope in ['current', 'lag1', 'total_current_plus_lag1']:
        for suffix in ['beta_positive', 'beta_negative', 'p_abs', 'p_sum', 'p_difference',
                       'positive_se', 'negative_se', 'positive_ci_low', 'positive_ci_high',
                       'negative_ci_low', 'negative_ci_high']:
            out[scope + '_' + suffix] = number(row, scope + '_' + suffix)
    return out


def minimum(rows):
    valid = [r for r in rows if number(r, 'current_p_abs') is not None]
    return compact(min(valid, key=lambda r: (number(r, 'current_p_abs'), r['model_id']))) if valid else None


def render(rows, result):
    def fmt(value, digits=4):
        return '—' if value is None else f'{float(value):.{digits}f}'.replace('.', ',')
    def pcell(value):
        text = fmt(value)
        return '<b>' + text + '</b>' if value is not None and value < .05 else text
    def get(group, mask, variant):
        return next(r for r in rows if r['family'] == 'A' and r['group'] == group
                    and r['control_mask'] == str(mask) and r['variant'] == variant)
    shared = [get(g, 1, 'CURRENT_ONLY') for g in GROUPS]
    result['shared_hhi_current_only'] = [compact(r) for r in shared]
    parts = ['<!doctype html><html lang="ru"><meta charset="utf-8"><title>H2 — результаты поиска спецификаций</title>',
             '<style>body{font:16px system-ui;line-height:1.5;color:#203046;max-width:1180px;margin:35px auto;padding:0 22px}h1{font-size:29px}h2{font-size:22px;margin-top:30px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;border-bottom:1px solid #d7dfe7;text-align:right}th:first-child,td:first-child{text-align:left}thead{background:#eef3f7}b{color:#183b57}.scroll{overflow-x:auto}.equation{font:21px Georgia;padding:16px;background:#f1f5f8}a{color:#125f92}code{font-size:13px}small{color:#526175}</style>',
             '<h1>H2: что изменило статистическую значимость</h1>',
             '<p>Рассчитаны 660 обозначений моделей (652 уникальные спецификации) на домашнем ПК. Значимая текущая асимметрия найдена в общей выборке, фондах акций и смешанных фондах. Для облигационных фондов текущая H2 не достигла 5% ни в одном из проверенных вариантов.</p>',
             '<p>Один общий вариант для четырёх выборок: текущие положительная и отрицательная части Net, фиксированные эффекты фондов и контроль концентрации портфеля HHI. Лаг и остальные пять контролей прежней SIX-модели исключены. Данные, прежняя отсечка P99 и способ расчёта стандартных ошибок сохранены.</p>',
             '<div class="equation">g<sub>fst</sub> = α<sub>f</sub> + β<sub>+</sub>P<sub>jft</sub> + β<sub>−</sub>M<sub>jft</sub> + γ HHI<sub>ft</sub> + ε<sub>fst</sub></div>',
             '<p>P = max(Net, 0); M = max(−Net, 0). g — процентное изменение расчётного количества бумаг. HHI измерен на конец интервала. Границы наблюдаемых интервалов: 30.09.2013–29.05.2026. В модели сохранена константа; ошибки кластеризованы по фонду и эмитенту, без временных эффектов и временной кластеризации. Проверяется H2: |β<sub>+</sub>| = |β<sub>−</sub>|.</p>',
             '<h2>Одна спецификация: текущий сигнал + HHI</h2><div class="scroll"><table><thead><tr><th>Выборка</th><th>N / фондов / эмитентов</th><th>β позитив (SE)</th><th>β негатив (SE)</th><th>p H2</th><th>Within R²</th></tr></thead><tbody>']
    for r in shared:
        cells = [LABELS[r['group']], ' / '.join(f'{int(r[k]):,}'.replace(',', ' ') for k in ['nobs','funds','companies']),
                 fmt(number(r,'current_beta_positive'))+' ('+fmt(number(r,'current_positive_se'))+')',
                 fmt(number(r,'current_beta_negative'))+' ('+fmt(number(r,'current_negative_se'))+')',
                 pcell(number(r,'current_p_abs')), fmt(number(r,'rsquared_within'))]
        parts.append('<tr>'+''.join('<td>'+c+'</td>' for c in cells)+'</tr>')
    parts += ['</tbody></table></div>',
              '<p>В трёх значимых выборках оценка реакции на негатив больше по модулю. Оба точечных коэффициента отрицательны, но коэффициент позитивной части по отдельности незначим. Поэтому результат относится к H2 об асимметрии и не подтверждает рост позиции в ответ на позитивные новости.</p>',
              '<p>Увеличение модуля негативного Net на 0,1 связано с уменьшением g примерно на 0,241 п.п. в общей выборке, 0,331 п.п. в акционных и 0,338 п.п. в смешанных фондах. Это условная ассоциация, а не причинный эффект.</p>',
              '<details><summary>95% доверительные интервалы коэффициентов</summary><table><thead><tr><th>Выборка</th><th>β позитив: 95% CI</th><th>β негатив: 95% CI</th></tr></thead><tbody>']
    for r in shared:
        parts.append('<tr><td>'+LABELS[r['group']]+'</td>'+''.join('<td>['+fmt(number(r,'current_'+s+'_ci_low'))+'; '+fmt(number(r,'current_'+s+'_ci_high'))+']</td>' for s in ['positive','negative'])+'</tr>')
    parts += ['</tbody></table></details><h2>Что дало изменение результата</h2><table><thead><tr><th>Модель: p текущей H2</th>'+''.join('<th>'+LABELS[g]+'</th>' for g in GROUPS)+'</tr></thead><tbody>']
    steps = [(63,'CURRENT_PLUS_LAG1','Прежние шесть контролей + лаг'), (63,'CURRENT_ONLY','Шесть контролей, без лага'),
             (1,'CURRENT_PLUS_LAG1','Только HHI + лаг'), (1,'CURRENT_ONLY','Только HHI, без лага')]
    for mask, variant, label in steps:
        parts.append('<tr><td>'+label+'</td>'+''.join('<td>'+pcell(number(get(g,mask,variant),'current_p_abs'))+'</td>' for g in GROUPS)+'</tr>')
    parts += ['</tbody></table><p>Для акционных фондов требуются оба изменения: исключение лага само по себе и сокращение контролей при сохранении лага не дают 5%. Удалены bond_indicator, доходность IMOEX, RVI и индикаторы периодов после марта 2014 / сентября 2018. Следовательно, найденная значимость зависит от более узкого набора контролей; учёт общих рыночных условий стал слабее.</p>',
              '<h2>Какие проверки не дали значимой текущей H2</h2>',
              '<ul><li>128 вариантов с прежними шестью контролями, четырьмя порогами выбросов и четырьмя требованиями к истории: ни одного p текущей H2 ниже 0,05.</li>',
              '<li>12 вариантов с шестью контролями и лагом: только продолжающиеся позиции, логарифмическое изменение количества, симметричное изменение количества. Ни одного значимого текущего теста.</li>',
              '<li>Единой проверенной спецификации со значимой текущей H2 сразу во всех четырёх группах нет. Облигационная выборка остаётся незначимой.</li></ul>',
              '<p>У продолжающихся позиций исходная процентная шкала сохраняется; логарифмическая и симметричная шкалы проверяют другие числовые зависимости. Для симметричной шкалы вход и выход равны +200 и −200. Эти варианты сохранены отдельно и не подменяют исходную H2.</p>',
              '<h2>Проверяемость</h2><p>Это исследовательский поиск после просмотра прежних результатов. Все модели сохранены, включая незначимые. Показаны обычные двусторонние p без Holm и иных поправок. Выбранная после поиска спецификация не является независимым подтверждающим тестом.</p>',
              '<p>Независимая проверка: <a href="qa_independent_v2.json">8 452 PASS / 0 FAIL</a>, 61 модель воспроизведена отдельным построчным расчётом коэффициентов и кластерной ковариации. Проверены все 660 записей, 1 332 контраста H2, состав выборок и исходные контрольные суммы. Расхождений в регрессионных результатах нет.</p>',
              '<p>Полная ковариация используется в составном тесте равенства модулей: p H2 = max(p разности, p суммы). Это один тест H2, а не поправка на число моделей. Численная реализация сверяется с <a href="https://bashtage.github.io/linearmodels/panel/panel/linearmodels.panel.model.PanelOLS.fit.html">PanelOLS</a>; источники, фактические строки и контрольные суммы сохранены.</p>',
              '<p><a href="review.html">Все результаты и сравнения</a> · <a href="all_models.csv">Все 660 моделей — CSV</a> · <a href="all_h2_tests.csv">Все тесты H2</a> · <a href="decision_summary.json">Сводка сравнения</a> · <a href="shared_hhi_native.json">Четыре общие модели: параметры и ковариация</a></p><ul>']
    for r in shared:
        filename = r['model_id'] + '.summary.txt'
        location = 'native/' if (OUT/'native'/filename).exists() else 'shared_hhi/native/'
        parts.append('<li>'+LABELS[r['group']]+': <a href="'+location+html.escape(filename)+'">полный Python output</a></li>')
    parts += ['</ul><small>Отдельные использованные панели и строки масок находятся на research-host: ./analysis_outputs/h2_significance_search_20261104/data. Прежние документы H1/H2 не изменены.</small></html>']
    (OUT/'practical_summary.html').write_text('\n'.join(parts), encoding='utf-8')


def run():
    source = OUT / 'all_models.csv'
    with source.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 660 and len({r['model_id'] for r in rows}) == 660
    baselines = [compact(next(r for r in rows if r['family'] == 'A' and r['group'] == group
                 and r['control_mask'] == '63' and r['variant'] == 'CURRENT_PLUS_LAG1')) for group in GROUPS]
    best_raw = [minimum([r for r in rows if r['family'] in ['A', 'B'] and r['group'] == group]) for group in GROUPS]
    best_raw_with_lag = [minimum([r for r in rows if r['family'] in ['A', 'B'] and r['group'] == group
                                 and r['variant'] == 'CURRENT_PLUS_LAG1']) for group in GROUPS]
    common = {}
    for row in rows:
        family = row['family']
        settings = (row['control_mask'],) if family == 'A' else ((row['support'], row['cutoff'])
                    if family == 'B' else (row['outcome_tag'],))
        key = (family, row['variant'], *settings)
        common.setdefault(key, {})[row['group']] = row
    ranking = []
    for key, records in common.items():
        if set(records) != set(GROUPS):
            continue
        pvalues = [number(records[g], 'current_p_abs') for g in GROUPS]
        if any(p is None for p in pvalues):
            continue
        ranking.append(dict(settings=list(key), max_current_p=max(pvalues),
                            groups_significant=sum(p < .05 for p in pvalues),
                            models=[compact(records[g]) for g in GROUPS]))
    ranking.sort(key=lambda r: (r['max_current_p'], r['settings']))
    counts = []
    for family in ['A', 'B', 'C']:
        for group in GROUPS:
            subset = [r for r in rows if r['family'] == family and r['group'] == group]
            counts.append(dict(family=family, group=group, labels=len(subset),
                               finite=sum(number(r, 'current_p_abs') is not None for r in subset),
                               current_significant=sum(number(r, 'current_p_abs') is not None and number(r, 'current_p_abs') < .05 for r in subset)))
    result = dict(status='DESCRIPTIVE_SYNTHESIS_NO_NEW_FITS',
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  interpretation='Minimum ordinary p values selected after an exploratory search; no multiple-testing adjustment. Alternative outcome tests are not the original raw-percent H2.',
                  baselines=baselines, best_raw_current=best_raw, best_raw_current_with_lag=best_raw_with_lag,
                  alternative_outcomes=[compact(r) for r in rows if r['family'] == 'C'],
                  current_significance_counts=counts, common_specification_ranking=ranking)
    render(rows, result)
    (OUT / 'decision_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(best_raw_current=best_raw, best_raw_current_with_lag=best_raw_with_lag,
                         common_top=ranking[:3], counts=counts), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
