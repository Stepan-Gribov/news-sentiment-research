"""Render a separate readable H1 review from saved results; no estimation."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path


def read_json(root, name, default=None):
    path=root/name
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else default


def number(value, digits=2):
    try:
        value=float(value)
    except (TypeError,ValueError):
        return '—'
    if not math.isfinite(value):
        return 'не определено'
    return f'{value:,.{digits}f}'.replace(',','\u202f').replace('.',',')


def count(value):
    return number(value,0)


def escaped(value):
    return html.escape(str(value))


LABELS={
    'ORIGINAL_MICRO':'Прежняя модель: историческая проверка',
    'NONINDEX_MICRO':'Микроконтроли, выявленные индексные исключены',
    'BASELINE_COMMON':'Микроконтроли: сопоставимая выборка',
    'NET_MACRO_COMMON':'Микро + макроконтроли: основная модель',
    'NET_MACRO_FOCUS_EQUITY':'Фонды акций',
    'NET_MACRO_FOCUS_BOND':'Фонды облигаций',
    'NET_MACRO_FOCUS_MIXED':'Смешанные фонды',
    'NET_MACRO_FOCUS_MONEY_MARKET':'Фонды денежного рынка',
    'NET_MACRO_FOCUS_OTHER':'Прочие фонды',
}


def result_table(rows, names, *, holm=False):
    headers=['Модель / тип фонда','N','Фондов','Эмитентов','β, п.п.','SE','95% CI','p','R² within']
    if holm:
        headers.insert(-1,'p Holm')
    out=['<div class="scroll"><table><thead><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr></thead><tbody>']
    for name in names:
        row=rows.get(name,{})
        if row.get('status')!='ESTIMATED':
            out.append('<tr><td>'+escaped(LABELS.get(name,name))+'</td><td colspan="9">Не оценена: '+escaped(row.get('reason','нет результата'))+'</td></tr>')
            continue
        values=[LABELS.get(name,name),count(row.get('nobs')),count(row.get('funds')),
            count(row.get('companies')),number(row.get('beta')),number(row.get('se_native')),
            '['+number(row.get('ci95_native_low'))+'; '+number(row.get('ci95_native_high'))+']',
            number(row.get('p_native'),4),number(row.get('rsquared_within'),6)]
        if holm:
            values.insert(-1,number(row.get('p_native_holm_focus_family'),4))
        out.append('<tr>'+''.join('<td>'+escaped(v)+'</td>' for v in values)+'</tr>')
    return '\n'.join(out+['</tbody></table></div>'])


def render(root):
    summary=read_json(root,'summary.json')
    universe=read_json(root,'universe_metadata.json',{})
    exceptions=read_json(root,'exception_summary.json',{})
    diagnostics=read_json(root,'residual_diagnostics.json',{})
    qa_name='qa_independent_v2.json' if (root/'qa_independent_v2.json').exists() else 'qa_independent.json'
    qa=read_json(root,qa_name,{})
    with (root/'comparison.csv').open(encoding='utf-8-sig',newline='') as stream:
        rows={r['model']:r for r in csv.DictReader(stream)}
    main=rows['NET_MACRO_COMMON']
    flow=summary['sample_flow']
    models={m['model']:m for m in summary['models']}
    main_model=models['NET_MACRO_COMMON']
    first=str(main_model['first_start_date']).split()[0]
    last=str(main_model['last_end_date']).split()[0]
    indexed=int(universe.get('index_status_counts_funds',{}).get('index',0))
    indexed_used=int(flow.get('index_status_funds',{}).get('index',0))
    grouped=['NET_MACRO_FOCUS_'+x for x in ['EQUITY','BOND','MIXED','MONEY_MARKET','OTHER']]
    chunks=['<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>H1 — проверенный новый прогон</title>',
        '<style>body{font:16px/1.55 system-ui,-apple-system,sans-serif;color:#1b2735;background:#f4f6f8;margin:0}main{max-width:1380px;margin:24px auto;padding:32px;background:white;border-radius:12px}h1{font-size:29px;margin-top:0}h2{font-size:21px;margin-top:30px}p{max-width:1100px}.lead{font-size:18px}.summary{padding:18px 22px;background:#edf4f9;border-left:4px solid #376c91}.note{padding:14px 18px;background:#fff7e5;border-left:4px solid #c99724}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px 11px;border-bottom:1px solid #dce2e8;text-align:right;white-space:nowrap}th{background:#eef2f5}th:first-child,td:first-child{text-align:left;white-space:normal}a{color:#135d90}code{font-size:13px;overflow-wrap:anywhere}details{margin-top:14px}.small{font-size:14px;color:#526171}ul{padding-left:24px}li{margin:6px 0}@media(max-width:650px){main{margin:0;padding:20px;border-radius:0}}</style></head><body><main>',
        '<h1>H1: новости и изменение позиции фонда</h1>',
        '<p class="lead">Новый прогон с макроконтролями и исключением выявленных индексных фондов. '+
        'Основная модель: <strong>'+count(main['nobs'])+' наблюдений, '+count(main['funds'])+' фондов, '+count(main['companies'])+' эмитентов</strong>; фактические окна '+escaped(first)+' — '+escaped(last)+'.</p>',
        '<div class="summary"><strong>Основная H1 статистически не подтверждена.</strong> β = '+number(main['beta'])+' п.п., SE = '+number(main['se_native'])+', p = '+number(main['p_native'],4)+
        '; 95% CI ['+number(main['ci95_native_low'])+'; '+number(main['ci95_native_high'])+']. Коэффициент положительный, но интервал включает ноль.</div>',
        '<p>Зависимая переменная: <strong>100 × (Q₁ / Q₀ − 1)</strong>, где Q — расчётное количество бумаг и Q₀ &gt; 0. β показывает изменение этого показателя в процентных пунктах при росте NetSentiment на 1. Это не изменение числа бумаг в штуках. NetSentiment находится между −1 и +1.</p>',
        '<p>Фиксированные эффекты: <strong>фонд</strong>. Стандартные ошибки: два кластера — <strong>фонд и компания</strong>. Временных фиксированных эффектов и кластеризации по времени нет. Макроблок: IMOEX, его доходность за фактический интервал, RUONIA, USD/RUB, RVI, рыночный поток последнего завершённого месяца, число фондов в полном исходном наборе интервалов и три исторических индикатора.</p>',
        '<p>Индексные фонды исключались за всю историю фонда при наличии любого подтверждения индексного статуса в предоставленных источниках: выявлено <strong>'+count(indexed)+'</strong> во всех метаданных; <strong>'+count(indexed_used)+'</strong> присутствовали в прежней регрессионной выборке. Это удалило '+count(flow['original_rows']-flow['nonindex_rows'])+' строк. Классификация остальных фондов по источнику не равнозначна проверке активного инвестиционного мандата.</p>',
        '<h2>Что изменили макроконтроли</h2>',
        result_table(rows,['BASELINE_COMMON','NET_MACRO_COMMON']),
        '<p>Эти две модели используют <strong>одинаковые строки</strong>. Добавление макроконтролей почти не изменило β и p-value. Смена знака относительно прежнего полного прогона связана прежде всего с переходом к выборке, для которой доступны макроданные, а не с добавлением регрессоров внутри этой сопоставимой выборки.</p>',
        '<details><summary>Переход от прежней выборки к новой</summary>',
        result_table(rows,['ORIGINAL_MICRO','NONINDEX_MICRO','BASELINE_COMMON']),
        '<p class="small">ORIGINAL_MICRO сохранена только для воспроизведения прежнего результата; она включает индексные фонды. NONINDEX_MICRO исключает выявленные индексные фонды. BASELINE_COMMON дополнительно ограничена доступностью макроконтролей. Эти переходы нельзя приписывать только влиянию макропеременных.</p></details>',
        '<h2>Разделение по типу фонда</h2>',result_table(rows,grouped,holm=True),
        '<p>Тип берётся на начало интервала из категории фактического раскрытия; при её отсутствии используется только полный корректный открывающий портфель с порогом преобладания 80%. Это <strong>тип фонда, а не тип отдельной бумаги</strong>. Один фонд может менять тип между периодами, поэтому числа фондов в группах могут пересекаться.</p>']
    mixed=rows.get('NET_MACRO_FOCUS_MIXED',{})
    wald=summary.get('focus_interaction_wald',{})
    chunks.append('<p>У смешанных фондов оценка отрицательная: p = '+number(mixed.get('p_native'),4)+', после поправки Holm по трём заранее заданным основным группам p = '+number(mixed.get('p_native_holm_focus_family'),4)+'. Это не подтверждение положительного направления H1. На уровне 5% после поправки этот результат не значим; на уровне 10% остаётся предварительным сигналом.</p>')
    if wald.get('status')=='ESTIMATED':
        chunks.append('<p>В совместной модели тест равенства склонов для фондов акций, облигаций и смешанных фондов: Wald χ²('+count(wald['df'])+') = '+number(wald['statistic'],4)+', <strong>p = '+number(wald['p_value'],4)+'</strong>. Убедительных различий склонов не установлено. Это дополнительная модель с общими коэффициентами контролей, '+count(wald['rows'])+' строками, '+count(wald['funds'])+' фондами и '+count(wald['companies'])+' эмитентами.</p>')
    else:
        chunks.append('<p>Совместный тест различий склонов не интерпретируется: '+escaped(wald.get('status','нет оценки'))+'.</p>')
    chunks.extend(['<h2>Как обработаны новости и исключения</h2>',
        '<p>Отсутствующий сентимент <strong>не заменяется нулём</strong>. Ниже аудит '+count(exceptions.get('rows'))+' кандидатных строк фонд–бумага–интервал, включая нецелевые позиции. Это не N основной регрессии и не число уникальных компаний.</p>',
        '<div class="scroll"><table><thead><tr><th>Состояние</th><th>Строк</th><th>Обработка</th></tr></thead><tbody>'])
    labels={
        'observed_nonzero_sentiment':('Релевантные новости с ненулевым балансом','Наблюдаемый скор; дальнейшие фильтры позиции и контролей сохраняются'),
        'observed_zero_sentiment':('Настоящий ноль по оценённым релевантным новостям','Ноль сохраняется и допускается в регрессию'),
        'issuer_unmapped_or_ambiguous':('Эмитент не определён или неоднозначен','Пропуск; сюда также попадают нецелевые/некорпоративные позиции'),
        'no_news_in_period':('Нет подобранных датированных новостей за окно','Пропуск; это отсутствие кандидатов в выбранном title-name/no-GAZ покрытии'),
        'incomplete_or_error_classification':('Неполная классификация или ошибка','Пропуск, а не нейтральная новость'),
        'all_candidates_irrelevant':('Все кандидаты нерелевантны','Пропуск: нет релевантного знаменателя'),
        'no_dated_news_in_period_undated_company_candidates':('Датированных кандидатов нет; есть кандидаты без даты','Пропуск: недатированные события нельзя присвоить интервалу'),
        'issuer_outside_declared_news_universe':('Эмитент вне утверждённого новостного покрытия','Пропуск с отдельной причиной'),
    }
    for item in exceptions.get('disjoint_reason_counts',[]):
        label,handling=labels.get(item['reason'],(item['reason'],'См. аудит'))
        chunks.append('<tr><td>'+escaped(label)+'</td><td>'+count(item['rows'])+'</td><td>'+escaped(handling)+'</td></tr>')
    chunks.append('</tbody></table></div><p>Все строки сохранены в аудитах и исходных портфелях. В регрессию входят только наблюдаемые допустимые скоры и определённый исход. Данный тест описывает позиции с доступными релевантными новостями, а не реакцию всех позиций на отсутствие новостей.</p>')
    diag=diagnostics.get('NET_MACRO_COMMON',{})
    extreme=diag.get('flags',{}).get('growth_over_10000_percent',{})
    chunks.extend(['<h2>Почему точность остаётся низкой</h2>',
        '<p>В основной модели '+count(extreme.get('rows'))+' строк с приростом расчётного Q свыше 10 000% дают <strong>'+number(100*extreme.get('residual_SS_fraction',float('nan')))+'% суммы квадратов остатков</strong>. На 10 крупнейших остатков приходится '+number(100*diag.get('largest_10_residual_SS_fraction',float('nan')))+'%. Экстремальные строки не удалялись. Это показывает концентрацию ошибки прогноза, но само по себе не доказывает причину смещения β и не обосновывает удаление строк ради p-value.</p>',
        '<p>Q вычислено как стоимость позиции / цена бумаги; для облигаций остаётся ограничение сопоставимости стоимости позиции с чистой ценой и НКД. Очень маленькая начальная позиция может давать огромный относительный прирост. Следующая содержательная проверка — происхождение таких строк и устойчивость заранее выбранного определения изменения позиции.</p>',
        '<h2>Ограничения вывода и проверка</h2>'])
    undefined=main_model.get('negative_covariance_diagonal_terms',[])
    if undefined:
        chunks.append('<div class="note">В основной двухкластерной ковариации отрицательны оценки дисперсии для '+', '.join('<code>'+escaped(x)+'</code>' for x in undefined)+'. Их SE, CI и p-value не определены и не интерпретируются. Дисперсия NetSentiment положительна; показанные для него SE и p конечны и независимо воспроизведены. Матрица не исправлялась принудительно до положительно полуопределённой.</div>')
    invalid_f=[]
    for model in summary['models']:
        f=model.get('native_robust_F',{})
        statistic=f.get('stat')
        if isinstance(statistic,(int,float)) and (not math.isfinite(statistic) or statistic<0):
            invalid_f.append((model['model'],statistic))
    if invalid_f:
        chunks.append('<p>Отдельно не интерпретируются отрицательные общие robust F: '+
            '; '.join(escaped(name)+' = '+number(value,4) for name,value in invalid_f)+
            '. Выведенные пакетом для них p = 1 не являются корректным свидетельством отсутствия общего эффекта. Это не p-value отдельного NetSentiment. Исходные Python summaries сохранены без изменений.</p>')
    chunks.append('<p>Строгая дополнительная выборка с отдельно подтверждённым активным/неиндексным мандатом пуста: соответствующая модель <strong>не оценена</strong>. Это недостаточность такой верификации в метаданных, а не доказательство того, что активных фондов нет. В основной модели все '+count(main['funds'])+' фондов имеют исходную классификацию как неиндексные категории, но не отдельное подтверждение активного управления.</p>')
    if qa:
        chunks.append('<p>Независимая проверка: <strong>'+escaped(qa.get('status','статус не указан'))+'</strong>; '+count(qa.get('pass_count'))+' проверок пройдено, '+count(qa.get('fail_count'))+' ошибок. Проверялись исходники, строки выборок и независимый расчёт общей модели. Предупреждения о неопределённой инференции контролей сохранены. <a href="'+escaped(qa_name)+'">Полный QA</a>.</p>')
    else:
        chunks.append('<p>Файл независимого QA пока не доставлен; статус проверки не заявляется.</p>')
    chunks.extend(['<h2>Данные и исходные Python-отчёты</h2>',
        '<ul><li><a href="NET_MACRO_COMMON.summary.html">Основная модель — native PanelOLS summary</a> · <a href="NET_MACRO_COMMON.summary.txt">TXT</a></li>',
        '<li><a href="report.html">Все неизменённые native summaries</a> · <a href="coefficients.csv">Все коэффициенты CSV</a> · <a href="comparison.csv">Сравнение моделей CSV</a></li>',
        '<li><a href="regression_preview.csv">200 фактически использованных строк: ключи, Y, NetSentiment, контроли и тип фонда</a></li>',
        '<li><a href="sample_flow.json">Отбор наблюдений</a> · <a href="universe_metadata.json">Классификация фондов</a> · <a href="exception_summary.json">Новостные исключения</a></li>',
        '<li><a href="macro_metadata.json">Правила макроконтролей</a> · <a href="source_manifest.json">Файлы и SHA256</a> · <a href="focus_interaction_wald.json">Совместный Wald</a></li></ul>',
        '<p class="small">Полные использованные выборки и построчный аудит исключений остаются на домашнем ПК: <code>'+escaped(summary['specification']['output_root'])+'</code>, подпапка <code>samples</code> и файл <code>exception_rows.csv.gz</code>. Новые оценки рассчитаны на ПК по SSH. Эта страница только форматирует сохранённые результаты.</p>',
        '</main></body></html>'])
    target=root/'review.html'
    target.write_text('\n'.join(chunks),encoding='utf-8')
    return target


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]/'outputs/h1_macro_no_time_20261031')
    args=parser.parse_args()
    print(render(args.root))
