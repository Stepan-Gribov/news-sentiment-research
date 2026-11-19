"""Format sealed H3 outputs; never estimate models or alter source releases."""
from pathlib import Path
import csv
import hashlib
import html
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/h3_alpha_20261104'
DIAG = BASE / 'run_d0440fad412ee5e9'
TR = BASE / 'provider_TR4_b00ab703609fb58f'
OUT = BASE / 'control_final_report'
OUT.mkdir(exist_ok=True)

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def read_json(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def fmt(value, scale=1, digits=3):
    if value in ('', None):
        return 'NA'
    return f'{float(value)*scale:.{digits}f}'

def table(headers, rows):
    e = lambda x: html.escape(str(x))
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+e(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+e(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'

groups = {'pooled':'Общая', 'equity':'Акций', 'bond':'Облигаций', 'mixed':'Смешанных'}
families = {'PRIMARY':'Основная', 'HISTORY36_MIN24':'История 36/24', 'BROAD_OBSERVED_NTS':'Расширенное новостное покрытие'}
data = {}
for label, path in [('diagnostic', DIAG), ('provider_TR', TR)]:
    data[label] = {name:read_csv(path/(name+'.csv')) for name in ['models','coefficients','tests']}
    data[label]['manifest_sha256'] = sha(path/'manifest.json')
    data[label]['summary'] = read_json(path/'summary.json')
    data[label]['path'] = str(path)
qa = read_json(ROOT/'reports/h3_alpha_independent_qa_20261104.json')
method = read_json(ROOT/'reports/h3_alpha_interpretation_20261104.json')
report = {'created_utc':datetime.now(timezone.utc).isoformat(), 'purpose':'Control synthesis of immutable results; no regression rerun', 'external_QA':qa, 'method_review_status':method['status'], 'runs':data, 'units':'Coefficient/SE/CI tables in basis points of mean monthly performance per +1 score; source CSV decimals retained', 'H2_unchanged':True}
report['method_review'] = method
ADDENDUM = TR/'addendum_106d6a7dbd5e'
if ADDENDUM.exists():
    report['TR_actual_fitted_coverage'] = read_csv(ADDENDUM/'fitted_samples_actual_date_coverage.csv')
    report['TR_primary_68_to_16'] = read_csv(ADDENDUM/'primary68_to16_flow.csv')
    report['TR_primary_NTS_reason_counts'] = read_csv(ADDENDUM/'primary_NTS_combined_reason_counts.csv')
(OUT/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')

parts = ['<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>H3 — результаты и проверяемость</title><style>body{font:15px/1.5 system-ui,sans-serif;color:#19232d;max-width:1400px;margin:32px auto;padding:0 24px}h1{font-size:28px}h2{font-size:21px;margin-top:32px}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}td,th{padding:7px 9px;border-bottom:1px solid #dce2e7;text-align:right;white-space:nowrap}th{background:#edf3f6}td:first-child,th:first-child{text-align:left}.scroll{overflow:auto}.note{background:#f1f5f7;border-left:4px solid #466477;padding:14px 18px}a{color:#12568b}code{overflow-wrap:anywhere}details{margin:16px 0}pre{white-space:pre-wrap}@media print{body{font-size:11px;margin:0;padding:0}table{font-size:9px}h2{break-after:avoid}.scroll{overflow:visible}tr{break-inside:avoid}}</style><body>']
parts += ['<h1>H3: новостная согласованность ребалансировки и будущая доходность фонда</h1>', '<p>Расчёты выполнены на домашнем ПК через SSH. Основной результат с полной доходностью ограничен малым покрытием; широкая панель отдельно показывает факторно-скорректированное изменение стоимости пая.</p>', '<div class="note"><b>Вывод:</b> положительная H3 не подтверждена. На широкой диагностической панели текущий скор, скор предыдущих новостей, их сумма и совместный тест незначимы на 5%. На официальной доходности в основной выборке недостаточно наблюдений. Дополнительные варианты дают отдельные отрицательные связи, но лишь для 2–4 фондов одной УК.</div>', '<p><b>Статус внешнего QA:</b> '+html.escape(qa['status'])+'. Широкая панель: '+str(qa['checks_pass'])+' PASS / '+str(qa['checks_fail'])+' FAIL. Подробности: <a href="../../../reports/h3_alpha_independent_qa_20261104.json">отчёт независимой проверки</a>.</p>', '<h2>Спецификация</h2><p><code>alpha(f,t,h) = fund_FE + βcurrent × NTS_current + βlag × NTS_lagnews + γ × past_alpha + error</code></p>', '<p>NTS измеряет согласованность текущей ребалансировки с новостями: покупки при позитивных и продажи при негативных новостях повышают скор. NTS_lagnews использует те же текущие изменения позиций и новости предыдущего отчётного интервала; это не лаг прошлых сделок. Одна строка — фонд × отчётная дата; фондовый outcome не размножается по бумагам.</p>', '<p>Alpha — средняя месячная избыточная доходность следующих h полных месяцев за вычетом прогнозируемой экспозиции на MCFTR и RGBITR. Безрисковая ставка — RUONIA. Beta оцениваются только на прошлых данных: окно 60 месяцев, минимум 36 пригодных, не менее 9 из последних 12. Прошлый intercept не вычитается из outcome и включён отдельным контролем. h=2/3 — средняя, не накопленная доходность.</p>', '<p>Фиксированные эффекты фондов; CR1-кластеризация по фонду; двусторонние p и 95% CI по t(G−1). Временных эффектов и временной кластеризации нет. Поправки Holm нет. R² ниже — within. Коэффициенты, SE и CI выражены в базисных пунктах на +1 к скору: 100 б.п. = 1 процентный пункт.</p>']

def coefficient_table(label, term):
    bundle=data[label]
    ms={r['model_id']:r for r in bundle['models']}
    rows=[]
    for c in bundle['coefficients']:
        if c['term']!=term or c['variant']!='CURRENT_PLUS_LAGNEWS':continue
        m=ms[c['model_id']]
        row=[groups[c['group']],c['horizon'],c['nobs'],c['funds'],fmt(c['beta'],10000,2),fmt(c['SE'],10000,2),'['+fmt(c['CI_low'],10000,2)+'; '+fmt(c['CI_high'],10000,2)+']',fmt(c['p'],digits=4),fmt(m['within_R2'],digits=4)]
        if label=='provider_TR':row.insert(0,families.get(c['sensitivity'],c['sensitivity']))
        rows.append(row)
    headers=['Группа','h','N','Фондов','β, б.п.','SE','95% CI','p','R² within']
    if label=='provider_TR':headers.insert(0,'Вариант')
    return table(headers,rows)

parts += ['<h2>Широкая диагностическая панель: текущий скор</h2><p>57 793 исходных месячных изменения стоимости пая, 603 фонда. После новостных, исторических и регрессионных фильтров общая h1: 6 045 наблюдений / 351 фонд. Отчётные даты в моделях: 31.12.2013–29.05.2026; будущие месяцы с января 2014 до июня / июля / августа 2026 для h=1/2/3. История выплат полностью не подтверждена: это диагностический outcome, а не подтверждённая total-return alpha.</p>',coefficient_table('diagnostic','NTS_Q_current'),'<h2>Широкая панель: скор предыдущих новостей</h2>',coefficient_table('diagnostic','NTS_Q_lagnews')]

primary=[m for m in data['provider_TR']['models'] if m['sensitivity']=='PRIMARY' and m['variant']=='CURRENT_PLUS_LAGNEWS']
parts += ['<h2>Официальная доходность: основная спецификация</h2><p>Собраны и документально проверены 228 опубликованных месячных доходностей: четыре фонда «Арсагеры» × 57 месяцев, январь 2022 — сентябрь 2026. Доходность следует методике провайдера: начисленный доход на пай и нормализация дроблений. Это не независимая реконструкция реинвестирования на фактические даты выплат. Предшествующие месяцы не достраивались из неподтверждённого NAV.</p><p>После требования 36 прошлых месяцев alpha определена на 68 фонд-дата для каждого горизонта; с основным NTS остаётся 16. Ни одна основная модель не проходит заранее установленный минимум 30 строк и двух фондов. NA означает нехватку данных, а не нулевой эффект.</p>',table(['Группа','h','Доступных строк','Фондов','β / SE / CI / p / R²','Причина'],[[groups[m['group']],m['horizon'],m['nobs'],m['funds'],'NA','N < 30'+('; один фонд' if int(m['funds'])<2 else '')] for m in primary]),'<h2>Официальная доходность: дополнительные варианты</h2><p><b>История 36/24:</b> минимум 24 месяца из последних 36 вместо 36/60. <b>Расширенное новостное покрытие:</b> минимум одна компания, без основного порога покрытия 50%; остальные правила и 36/60 сохраняются. Эти варианты заранее зафиксированы, не заменяют основную модель. Всего оценено 9 из 90 предусмотренных обозначений; остальные статусы сохранены. Всего 2–4 кластера одной УК: выводы крайне неустойчивы и не распространяются на рынок.</p><p><b>Текущий скор:</b></p>',coefficient_table('provider_TR','NTS_Q_current'),'<p><b>Скор предыдущих новостей:</b></p>',coefficient_table('provider_TR','NTS_Q_lagnews')]

ts=[t for t in data['provider_TR']['tests'] if t['test'] not in ['NTS_Q_current=0','NTS_Q_lagnews=0']]
parts += ['<h2>Дополнительные тесты официальной доходности</h2>',table(['Вариант','Группа','h','Тест','β суммы, б.п.','SE','95% CI','p','Статус'],[[families.get(t['sensitivity'],t['sensitivity']),groups[t['group']],t['horizon'],t['test'],fmt(t.get('beta'),10000,2),fmt(t.get('SE'),10000,2),'['+fmt(t.get('CI_low'),10000,2)+'; '+fmt(t.get('CI_high'),10000,2)+']',fmt(t.get('p'),digits=5),t['status']] for t in ts]),'<p>При расширенном новостном покрытии на h1 сумма коэффициентов отрицательна: общая p=0,01337, акции p=0,04062. В общей группе совместный тест p=0,00415. Это не положительный эффект H3. Совместный тест в акциях с двумя кластерами не идентифицирован и оставлен NA. Отдельные текущие и прошлые коэффициенты ни в одном из девяти вариантов не достигают 5% при t(G−1).</p>']

parts += ['<h2>Проверяемость и файлы</h2><p>Все оценённые и неоценённые варианты, точные выборки, исходные значения, ковариации и причины исключения сохранены. Базовые ресурсы, live Gemma, H1/H2 и прежние версии не изменены. Использованные большие панели находятся только на ПК.</p><ul><li><a href="../run_d0440fad412ee5e9/review.html">Широкая панель: все Python-выводы</a> · <a href="../run_d0440fad412ee5e9/coefficients.csv">коэффициенты CSV</a> · <a href="../run_d0440fad412ee5e9/tests.csv">тесты CSV</a></li><li><a href="../provider_TR4_b00ab703609fb58f/review.html">Официальная доходность: все модели и Python-выводы</a> · <a href="../provider_TR4_b00ab703609fb58f/coefficients.csv">коэффициенты CSV</a> · <a href="../provider_TR4_b00ab703609fb58f/models.csv">все 90 статусов CSV</a></li><li><a href="summary.json">Объединённый машиночитаемый отчёт и QA</a></li><li><a href="../../../config/h3_alpha_spec_20261104.json">Замороженная спецификация</a> · <a href="../../../reports/h3_alpha_interpretation_20261104.json">Методический review</a></li></ul><p>В запечатанных ранних summary/README может оставаться историческое “QA pending”; актуальное заключение находится в отдельном независимом QA, связанном выше. Исходный Python-вывод сохраняет p по residual df; для таблиц этого отчёта используются сохранённые основные p/CI по G−1.</p>', '<p><b>Ограничения:</b> неполная история выплат у широкой панели; один управляющий и мало кластеров у официального поднабора; расчётные количества бумаг, влияние притоков/оттоков и чистых цен облигаций; неполная human validation новостных меток; неопределённость сгенерированных NTS/alpha не учтена полностью; кластеризация только по фонду не учитывает общие межфондовые шоки. Незначимость не доказывает отсутствия эффекта.</p><details><summary>Точные источники широкой панели и хеши</summary><pre>'+html.escape(json.dumps(data['diagnostic']['summary']['input_manifest'],ensure_ascii=False,indent=2))+'</pre></details><p>Удалённые результаты: <code>./analysis_outputs/h3_alpha_20261104/outputs/run_d0440fad412ee5e9</code> и <code>.../provider_TR4_b00ab703609fb58f</code>. Внутри regression — точные used samples, в корне — training/future membership и manifests.</p></body></html>']
(OUT/'index.html').write_text('\n'.join(parts),encoding='utf-8')
document = (OUT/'index.html').read_text()
extra = '<h2>Окончательная проверка</h2><p>Независимый QA широкой панели: 51 127 PASS / 0 FAIL. QA официального поднабора: '+str(qa.get('provider_TR',{}).get('checks_pass','в работе'))+' PASS / '+str(qa.get('provider_TR',{}).get('checks_fail','NA'))+' FAIL. Численная проверка подтверждает воспроизводимость расчётов, а не истинность H3 или надёжность вывода при малом числе фондов.</p><p>Сумма βcurrent + βlag описывает совместное увеличение двух скоров на одной текущей ребалансировке. Это не накопленный эффект сделок за два периода.</p>'
if ADDENDUM.exists():
    extra += '<p><a href="../provider_TR4_b00ab703609fb58f/addendum_106d6a7dbd5e/primary_NTS_exact_reason_counts.csv">Точные причины перехода от 68 alpha к 16 основным строкам</a>: 52 фонд-периода не проходят основной фильтр NTS. <a href="../provider_TR4_b00ab703609fb58f/addendum_106d6a7dbd5e/fitted_samples_actual_date_coverage.csv">Даты и пути к девяти фактически оценённым выборкам</a>.</p>'
    cov = report['TR_actual_fitted_coverage']
    extra += table(['Вариант','Группа','h','N','Фондов','Первая дата','Последняя дата','Будущий период'],[[families[r['sensitivity']],groups[r['group']],r['horizon'],r['N'],r['funds'],r['first_origin'],r['last_origin'],r['first_future_month']+' — '+r['last_future_month']] for r in cov])
(OUT/'index.html').write_text(document.replace('</body>',extra+'</body>'),encoding='utf-8')
print(json.dumps({'output':str(OUT/'index.html'),'diagnostic_models':sum(m['status']=='ESTIMATED' for m in data['diagnostic']['models']),'TR_models':sum(m['status']=='ESTIMATED' for m in data['provider_TR']['models']),'TR_primary_NA':len(primary),'QA_status':qa['status']},ensure_ascii=False))
