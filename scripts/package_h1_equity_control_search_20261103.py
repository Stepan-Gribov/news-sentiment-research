"""Readable H1 search outputs, no regression or data transformation."""
from pathlib import Path
import gzip,hashlib,html,json,shutil,zipfile
ROOT=Path('./analysis_outputs/h1_equity_control_search_20261103')
LABELS={
'opening_weight':'Начальная доля актива в фонде',
'closing_log_NAV':'Логарифм СЧА на конечную дату',
'closing_HHI':'Концентрация портфеля HHI на конечную дату',
'log_news_count_interval':'Логарифм числа релевантных новостей за интервал',
'log_interval_days':'Логарифм длительности интервала',
'bond_indicator':'Индикатор облигационной бумаги',
'Imoex':'Уровень индекса Мосбиржи на конечную дату',
'imoex_ret':'Доходность индекса Мосбиржи за интервал',
'Ruonia':'Ставка RUONIA на конечную дату',
'USD_RUB':'Курс USD/RUB на конечную дату',
'RVI':'Индекс волатильности RVI на конечную дату',
'Flow_billion_RUB':'Приток/отток за последний полностью завершённый экономический месяц, млрд руб.',
'unique_funds':'Число наблюдаемых фондов на точную конечную дату в полном исходном наборе интервалов',
'dummy_mar2014':'Индикатор: конечная дата ≥ 01.03.2014',
'dummy_feb2022':'Индикатор: конечная дата ≥ 01.02.2022',
'dummy_sep2018':'Индикатор: конечная дата ≥ 01.09.2018'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
with (ROOT/'all_models.csv').open('rb')as src,gzip.open(ROOT/'all_models.csv.gz','wb',compresslevel=6)as out:shutil.copyfileobj(src,out)
(ROOT/'controls_dictionary.json').write_text(json.dumps(LABELS,ensure_ascii=False,indent=2),encoding='utf-8')
p=ROOT/'review.html';s=p.read_text(encoding='utf-8')
if '<h2>Названия контролей</h2>' in s:s=s.split('<h2>Названия контролей</h2>')[0]+'</html>'
if '<b>Все 131 072 модели' not in s:s=s.replace('<h2>Проверяемость</h2>','<h2>Проверяемость</h2><p><a href="all_models.csv.gz"><b>Все 131 072 модели: полный CSV, сжатый gzip</b></a> · <a href="controls_dictionary.json">Расшифровка названий контролей</a></p>')
x=json.loads((ROOT/'summary.json').read_text(encoding='utf-8'))
if '<h2>Главное сравнение</h2>' not in s:
 chosen=[r for r in x['shortlist']if r['control_mask']in[65535,42148]]
 def fmt(v):return f'{v:.5f}'if isinstance(v,(float,int))else '—'
 top='<h2>Главное сравнение</h2><p><b>Один набор из шести контролей даёт p&lt;0,05 для текущего сентимента и его лага в одной модели.</b> Все строки выборки и правила расчёта ошибок одинаковы. Это результат выбора после полной сетки.</p><table><tr><th>Модель</th><th>Контроли</th><th>Текущий β (SE)</th><th>p текущего</th><th>Лаг β (SE)</th><th>p лага</th><th>Within R²</th></tr>'
 for r in chosen:
  top+='<tr><td>'+('Без лага'if r['variant']=='CURRENT_ONLY'else'Текущий + лаг 1')+'</td><td>'+('Все 16'if r['control_mask']==65535 else'Шесть отобранных')+'</td><td>'+fmt(r['current_beta'])+' ('+fmt(r['current_se'])+')</td><td>'+fmt(r['current_p'])+'</td><td>'+(fmt(r.get('lag_beta'))+' ('+fmt(r.get('lag_se'))+')'if r.get('lag_beta')is not None else'—')+'</td><td>'+fmt(r.get('lag_p'))+'</td><td>'+fmt(r['rsquared_within'])+'</td></tr>'
 best=next(r for r in chosen if r['control_mask']==42148 and r['variant']=='CURRENT_PLUS_LAG1')
 top+='</table><p>Сумма текущего и лага: β='+fmt(best['sum_beta'])+', SE='+fmt(best['sum_se'])+', 95% CI ['+fmt(best['sum_ci_low'])+'; '+fmt(best['sum_ci_high'])+'], p='+fmt(best['sum_p'])+'. Совместный тест current=lag=0: χ²(2)='+fmt(best['joint_stat'])+', p='+fmt(best['joint_p'])+'.</p><p><b>Шесть контролей:</b> концентрация HHI, индикатор облигационной бумаги, доходность IMOEX за интервал, RVI, индикаторы конечной даты ≥01.03.2014 и ≥01.09.2018. В этом наборе исключены начальная доля актива, размер фонда и ещё восемь контролей; полный список ниже.</p><p>Оригинальный Python output лучшей общей модели: <a href="native/CURRENT_PLUS_LAG1__42148.summary.txt">TXT</a> · <a href="native/CURRENT_PLUS_LAG1__42148.summary.html">HTML</a>.</p>'
 s=s.replace('<h2>Сводка всех моделей</h2>',top+'<h2>Сводка всех моделей</h2>')

s=s.replace('</html>','<h2>Названия контролей</h2><table><tr><th>Столбец</th><th>Смысл</th></tr>'+''.join('<tr><td>'+html.escape(k)+'</td><td>'+html.escape(v)+'</td></tr>'for k,v in LABELS.items())+'</table></html>')
p.write_text(s,encoding='utf-8')
(ROOT/'README.txt').write_text('H1, фонды акций — полный перебор контрольных переменных, 03.11.2026\n\nОткрыть review.html.\nВсе 131 072 модели: all_models.csv.gz (распаковать gzip, затем открыть CSV).\nНабор шести контролей mask42148 дает обычные p<0.05 для current и lag в одной модели.\nЭто отбор после полной сетки. Никаких поправок на множественные проверки по указанию пользователя нет.\n\nОдна общая выборка для всех моделей: 60 903 строки / 91 фонд / 136 эмитентов.\nFund FE, кластеры фонд и компания; без time FE и временных кластеров.\nТочные построчные данные используются без изменений и остаются на домашнем ПК:\n'+str(ROOT/'used_sample.parquet')+'\nИсточники, SHA256, rowhash: input_manifest.json.\nКонтроли: controls_dictionary.json.\nТип фонда (equity focus) не равен типу каждой отдельной бумаги; bond_indicator может варьировать.\n',encoding='utf-8')
files=[ROOT/'all_models.csv.gz',ROOT/'shortlist.csv',ROOT/'native_coefficients.csv',ROOT/'control_inclusion_summary.csv',ROOT/'controls_dictionary.json',ROOT/'summary.json',ROOT/'input_manifest.json',ROOT/'frozen_plan.json',ROOT/'native_verification.json',ROOT/'review.html',ROOT/'README.txt']+list((ROOT/'native').glob('*'))+list((ROOT/'code').glob('*'))+[f for f in ROOT.glob('*.json')if 'qa' in f.name.lower()]
manifest={str(f.relative_to(ROOT)).replace('\\','/'):{'bytes':f.stat().st_size,'sha256':sha(f)}for f in files}
(ROOT/'download_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');files.append(ROOT/'download_manifest.json')
archive=ROOT/'h1_equity_control_search_20261103.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6)as z:
 for f in files:z.write(f,str(f.relative_to(ROOT)))
print(json.dumps({'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),'gzip_bytes':(ROOT/'all_models.csv.gz').stat().st_size,'gzip_sha256':sha(ROOT/'all_models.csv.gz')},ensure_ascii=True))
