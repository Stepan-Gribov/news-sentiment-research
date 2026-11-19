"""Package the H1 ablation report; no estimation or row transformations."""
from pathlib import Path
import hashlib,html,json,shutil,zipfile
p=Path('./analysis_outputs/h1_control_ablation_20261103')
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
s=json.loads((p/'summary.json').read_text(encoding='utf-8'))
cov=json.loads((p/'all_model_covariances.json').read_text(encoding='utf-8'))
s['covariance_export']={k:v for k,v in cov.items()if k!='models'};s['covariance_export']['file']='all_model_covariances.json';s['covariance_export']['sha256']=sha(p/'all_model_covariances.json')
qa=p/'qa_independent.json'
if qa.exists():
 q=json.loads(qa.read_text(encoding='utf-8'))
 assert q['status']=='PASS'
 s['status']='COMPLETE_VERIFIED';s['independent_QA']={'status':'PASS','file':'qa_independent.json','sha256':sha(qa),'PASS':q.get('pass_count',q.get('PASS')),'FAIL':q.get('fail_count',q.get('FAIL')),'scope':q.get('numerical_scope',q.get('scope'))}
 status=json.loads((p/'status.json').read_text(encoding='utf-8'));status['status']=s['status'];status['independent_QA']=s['independent_QA'];(p/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
(p/'summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
r=p/'review.html';text=r.read_text(encoding='utf-8')
if '<h2>Без лага</h2>'not in text:
 labels={'equity':'Акций','bond':'Облигаций','mixed':'Смешанные'}
 def f(v):return f'{v:.5f}'if isinstance(v,(float,int))else'—'
 top='<h2>Без лага</h2><table><tr><th>Фонды</th><th>Контроли</th><th>β текущего</th><th>SE</th><th>p</th><th>95% CI</th><th>Within R²</th></tr>'
 for x in s['models']:
  if x['variant']!='CURRENT_ONLY':continue
  top+='<tr><td>'+labels[x['focus']]+'</td><td>'+x['control_set']+'</td><td>'+f(x['current_beta'])+'</td><td>'+f(x['current_se'])+'</td><td>'+f(x['current_p'])+'</td><td>['+f(x['current_ci_low'])+'; '+f(x['current_ci_high'])+']</td><td>'+f(x['rsquared_within'])+'</td></tr>'
 top+='</table>'
 text=text.replace('<h2>Можно ли убрать const?</h2>',top+'<h2>Можно ли убрать const?</h2>')
if qa.exists()and'Независимый QA'not in text:
 qahtml='<p><b>Независимый QA: '+str(s['independent_QA']['PASS'])+' PASS / '+str(s['independent_QA']['FAIL'])+' FAIL.</b> <a href="qa_independent.json">Проверки и численные реплики</a>.</p>'
 text=text.replace('<h2>Текущий сентимент + лаг 1</h2>',qahtml+'<h2>Текущий сентимент + лаг 1</h2>')
if '<a href="all_model_covariances.json">'not in text:text=text.replace('<h2>Все Python outputs</h2>','<p><a href="all_model_covariances.json">Полные native covariance matrices всех 33 моделей</a> · <a href="no_const_models.csv">Модели без явной константы</a> · <a href="no_const_coefficients.csv">Их полные коэффициенты</a></p><h2>Все Python outputs</h2>')
r.write_text(text,encoding='utf-8')
(p/'README.txt').write_text('H1 control ablation, 03.11.2026\nОткрыть review.html.\nВсе 30 основные модели: comparison.csv; полные коэффициенты: coefficients.csv.\nКонтроли: ALL16, DROP_OPENING_WEIGHT, DROP_NEWS_COUNT, DROP_BOTH (14), EQUITY_SELECTED_SIX.\nSIX отличается от DROP_BOTH.\n3 проверки const: no_const_comparison.json; явная константа убрана, fixed effects фондов сохранены.\nНаблюдения и правила отсечки не менялись.\nБольшие источники и построчные проверки представлений остаются на research-host.\nТочные paths, SHA256, rowhash: samples_manifest.json.\nNative источники и полный covariance: all_model_covariances.json.\nВсе p обычные native, без поправок; набор SIX перенесён из предыдущего поиска в акционных фондах.\n',encoding='utf-8')
files=[f for f in p.iterdir()if f.is_file()and f.suffix in ['.csv','.json','.html','.txt']and f.name!='download_manifest.json']+[f for f in (p/'native').glob('*')if f.is_file()]+[f for f in (p/'code').glob('*')if f.is_file()]
manifest={str(f.relative_to(p)).replace('\\','/'):{'bytes':f.stat().st_size,'sha256':sha(f)}for f in files}
(p/'download_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');files.append(p/'download_manifest.json')
z=p/'h1_control_ablation_20261103.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED,compresslevel=6)as archive:
 for f in files:archive.write(f,str(f.relative_to(p)))
print(json.dumps({'status':s['status'],'files':len(files),'summary_sha256':sha(p/'summary.json'),'archive_bytes':z.stat().st_size,'archive_sha256':sha(z)},ensure_ascii=True))
