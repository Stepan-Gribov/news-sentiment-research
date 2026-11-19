"""Attach independent QA without changing fitted values."""
from pathlib import Path
import hashlib,json,shutil
p=Path('./analysis_outputs/h1_equity_control_search_20261103')
qa=json.loads((p/'qa_independent.json').read_text(encoding='utf-8'))
assert qa['status']=='PASS'and qa['fail_count']==0
qasha=hashlib.sha256((p/'qa_independent.json').read_bytes()).hexdigest()
summary=json.loads((p/'summary.json').read_text(encoding='utf-8'))
if not(p/'summary_pre_qa.json').exists():shutil.copy2(p/'summary.json',p/'summary_pre_qa.json')
summary['status']='COMPLETE_VERIFIED'
summary['independent_QA']={'status':'PASS','PASS':qa['pass_count'],'FAIL':qa['fail_count'],'selected_numerical_replica_count':qa['selected_numerical_replica_count'],'exhaustive_grid_repeated':qa['exhaustive_grid_repeated'],'file':'qa_independent.json','sha256':qasha,'scope':'All 131072 grid rows checked structurally; 19 models independently reproduced numerically, including all nine runner shortlist models and ten fixed random masks.'}
(p/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
status=json.loads((p/'status.json').read_text(encoding='utf-8'));status.update(status='COMPLETE_VERIFIED',independent_QA=summary['independent_QA'])
(p/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
r=p/'review.html';s=r.read_text(encoding='utf-8')
if 'Независимая проверка: 103 PASS'not in s:
 s=s.replace('<h2>Главное сравнение</h2>','<p><b>Независимая проверка: 103 PASS / 0 FAIL.</b> Численно воспроизведены 19 моделей (все девять отобранных + десять заранее фиксированных случайных масок); вся сетка повторно не оценивалась. <a href="qa_independent.json">Полный QA</a>.</p><h2>Главное сравнение</h2>')
r.write_text(s,encoding='utf-8')
print(json.dumps({'status':summary['status'],'QA_sha256':qasha},ensure_ascii=True))
