"""Thirty frozen native H1 control ablations plus three intercept representations."""
from __future__ import annotations
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ.setdefault(k,'1')
import argparse,hashlib,html,importlib.util,json,sys,time,traceback,zipfile
from pathlib import Path
import numpy as np,pandas as pd
from linearmodels.panel import PanelOLS


def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def clean(v):
 if isinstance(v,dict):return {str(k):clean(x)for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x)for x in v]
 if isinstance(v,np.ndarray):return clean(v.tolist())
 if isinstance(v,np.generic):return clean(v.item())
 if isinstance(v,float)and not np.isfinite(v):return None
 return v
def save(p,v):p.write_text(json.dumps(clean(v),ensure_ascii=False,indent=2),encoding='utf-8')
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m

def fit(d,controls,distributed,explicit_const,base):
 declared=(['net_sentiment_lag1']if distributed else[])+list(controls);terms,dropped=base.select_regressors(d,declared)
 targets=['net_sentiment']+(['net_sentiment_lag1']if distributed else[])
 assert set(targets).issubset(terms)
 index=pd.MultiIndex.from_arrays([d.fund_id.astype(str),pd.to_datetime(d.end_date)],names=['fund_id','report_date'])
 x=d[terms].astype(float).copy();x.index=index
 if explicit_const:x.insert(0,'const',1.)
 y=pd.Series(d.quantity_change_pct.to_numpy(float),index=index,name='quantity_change_pct')
 clusters=pd.DataFrame({'fund_cluster':pd.Categorical(d.fund_id.astype(str)).codes,'company_cluster':pd.Categorical(d.company_id.astype(str)).codes},index=index)
 model=PanelOLS(y,x,entity_effects=True,time_effects=False,other_effects=None,singletons=True,drop_absorbed=False,check_rank=True)
 result=model.fit(cov_type='clustered',clusters=clusters,cluster_entity=False,cluster_time=False,group_debias=True,debiased=True,auto_df=False,count_effects=True)
 assert result.nobs==len(d)and result.resids.index.equals(index)
 return result,terms,dropped

def export_result(d,res,terms,dropped,name,focus,controlset,variant,root,lag,explicit_const=True):
 (root/'native').mkdir(exist_ok=True)
 (root/'native'/(name+'.summary.txt')).write_text(str(res.summary),encoding='utf-8')
 (root/'native'/(name+'.summary.html')).write_text(res.summary.as_html(),encoding='utf-8')
 ci=res.conf_int();co=[]
 for term in res.params.index:
  co.append(dict(model=name,focus=focus,control_set=controlset,variant=variant,explicit_const=explicit_const,term=term,beta=float(res.params[term]),se=float(res.std_errors[term]),p=float(res.pvalues[term]),ci_low=float(ci.loc[term].iloc[0]),ci_high=float(ci.loc[term].iloc[1])))
 rec=dict(model=name,focus=focus,control_set=controlset,variant=variant,explicit_const=explicit_const,nobs=int(res.nobs),funds=int(d.fund_id.nunique()),companies=int(d.company_id.nunique()),df_resid=int(res.df_resid),df_model=int(res.df_model),rsquared=float(res.rsquared),rsquared_within=float(res.rsquared_within),first_start_date=str(d.start_date.min()),last_end_date=str(d.end_date.max()),retained_terms=terms,dropped_controls=dropped,negative_covariance_diagonal_terms=[str(k)for k,v in zip(res.params.index,np.diag(res.cov))if v<0])
 for prefix,target in [('current','net_sentiment'),('lag','net_sentiment_lag1'),('const','const')]:
  if target not in res.params:continue
  for key,val in [('beta',res.params[target]),('se',res.std_errors[target]),('p',res.pvalues[target]),('ci_low',ci.loc[target].iloc[0]),('ci_high',ci.loc[target].iloc[1])]:rec[prefix+'_'+key]=float(val)
 if variant=='CURRENT_PLUS_LAG1':
  tests=lag.distributed_tests(res);t=tests['sum_current_lag'];rec.update(sum_beta=t['beta'],sum_se=t['se_native'],sum_p=t['p_native'],sum_ci_low=t['ci95_native_low'],sum_ci_high=t['ci95_native_high'],joint_stat=tests['joint_current_lag_zero']['statistic'],joint_p=tests['joint_current_lag_zero']['p_value'],sum_status=t['status'],joint_status=tests['joint_current_lag_zero']['status'],target_covariance=tests['target_covariance'])
 return rec,co

def nocheck(d,withres,nores,checkid,root):
 # PanelOLS fitted_values exclude estimated entity effects in both representations.
 withfit=withres.fitted_values.iloc[:,0].to_numpy();nofit=nores.fitted_values.iloc[:,0].to_numpy()
 withef=withres.estimated_effects.iloc[:,0].to_numpy();noef=nores.estimated_effects.iloc[:,0].to_numpy()
 wr=withres.resids.to_numpy();nr=nores.resids.to_numpy();wt=withfit+withef;nt=nofit+noef
 terms=[t for t in withres.params.index if t!='const'];checks={}
 for name,a,b in [('slopes',withres.params[terms],nores.params[terms]),('slope_SE',withres.std_errors[terms],nores.std_errors[terms]),('slope_p',withres.pvalues[terms],nores.pvalues[terms]),('slope_covariance',withres.cov.loc[terms,terms],nores.cov.loc[terms,terms]),('residuals',wr,nr),('combined_prediction_fitted_plus_FE',wt,nt)]:
  av=np.asarray(a,float);bv=np.asarray(b,float);checks[name]={'PASS':bool(np.allclose(av,bv,rtol=3e-7,atol=3e-8,equal_nan=True)),'max_abs_difference':float(np.nanmax(np.abs(av-bv)))}
 checks['df_resid']={'PASS':withres.df_resid==nores.df_resid,'with_const':int(withres.df_resid),'without_const':int(nores.df_resid)}
 checks['R2_within']={'PASS':bool(np.isclose(withres.rsquared_within,nores.rsquared_within,rtol=1e-9,atol=1e-12)),'with_const':float(withres.rsquared_within),'without_const':float(nores.rsquared_within)}
 const=float(withres.params['const']);deltafit=withfit-nofit;deltaef=withef-noef
 checks['intercept_normalization_fitted_shift']={'PASS':bool(np.allclose(deltafit,const,rtol=3e-7,atol=3e-8)),'fitted_difference_mean':float(deltafit.mean()),'const':const}
 checks['intercept_normalization_FE_opposite_shift']={'PASS':bool(np.allclose(deltaef,-const,rtol=3e-7,atol=3e-8)),'FE_difference_mean':float(deltaef.mean()),'negative_const':-const}
 (root/'representation_checks').mkdir(exist_ok=True)
 frame=d[['row_id','fund_id','company_id','security_id','start_date','end_date']].copy()
 for k,v in dict(with_const_fitted_excluding_FE=withfit,without_const_fitted_excluding_FE=nofit,with_const_FE=withef,without_const_FE=noef,with_const_total_prediction=wt,without_const_total_prediction=nt,with_const_residual=wr,without_const_residual=nr).items():frame[k]=v
 frame.to_parquet(root/'representation_checks'/(checkid+'.parquet'),index=False,compression='zstd')
 return dict(check_id=checkid,status='PASS'if all(c['PASS']for c in checks.values())else'FAIL',checks=checks,const_beta=const,const_p=float(withres.pvalues['const']),scope='Explicit const removed only; fund fixed effects retained. Individual fitted_values differ in normalization, total fitted + FE and residuals agree.')

def render(root,s):
 labels={'equity':'Акций','bond':'Облигаций','mixed':'Смешанные'}
 def f(v):return f'{v:.5f}'if isinstance(v,(float,int))else'—'
 rows=s['models'];txt='<!doctype html><html lang="ru"><meta charset="utf-8"><title>H1 — удаление двух контролей и константа</title><style>body{font:15px system-ui;margin:32px;color:#16253d}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border:1px solid #cbd2df}th{background:#eef3fa}p{max-width:1100px}code{background:#eef3fa}</style><h1>H1: отдельное удаление контролей</h1><p>30 native PanelOLS моделей: три типа фондов × два варианта сентимента × пять наборов контролей. Строки внутри каждого типа фонда одинаковые: акции 60 903 / облигации 39 039 / смешанные 57 131. Fund FE; кластеры фонд и компания; без time FE и временной кластеризации. Обычные двусторонние p без поправок.</p><p><b>DROP_BOTH</b> исключает только начальную долю актива (opening_weight) и логарифм числа новостей (log_news_count_interval), оставляя 14 контролей. <b>EQUITY_SELECTED_SIX</b> — другой набор из предыдущего поиска: HHI, bond_indicator, imoex_ret, RVI, dummy_mar2014, dummy_sep2018. Он не равен DROP_BOTH.</p>'
 txt+='<h2>Текущий сентимент + лаг 1</h2><table><tr><th>Фонды</th><th>Контроли</th><th>Текущий β (SE)</th><th>p</th><th>Лаг β (SE)</th><th>p</th><th>Within R²</th><th>Python output</th></tr>'
 for r in rows:
  if r['variant']!='CURRENT_PLUS_LAG1':continue
  txt+='<tr><td>'+labels[r['focus']]+'</td><td>'+r['control_set']+'</td><td>'+f(r['current_beta'])+' ('+f(r['current_se'])+')</td><td>'+f(r['current_p'])+'</td><td>'+f(r.get('lag_beta'))+' ('+f(r.get('lag_se'))+')</td><td>'+f(r.get('lag_p'))+'</td><td>'+f(r['rsquared_within'])+'</td><td><a href="native/'+r['model']+'.summary.txt">TXT</a> · <a href="native/'+r['model']+'.summary.html">HTML</a></td></tr>'
 txt+='</table><h2>Можно ли убрать const?</h2><p>Константу можно не показывать явно, сохраняя fund FE. Это меняет нормировку фиксированных эффектов. Незначимый p константы не является основанием обнулять фиксированные эффекты фондов. Для трёх моделей выполнена прямая проверка с const и без неё.</p><table><tr><th>Проверка</th><th>Статус</th><th>β const</th><th>p const</th><th>max Δ slopes</th><th>max Δ prediction + FE</th></tr>'
 for r in s['no_const_checks']:
  txt+='<tr><td>'+html.escape(r['check_id'])+'</td><td>'+r['status']+'</td><td>'+f(r['const_beta'])+'</td><td>'+f(r['const_p'])+'</td><td>'+f(r['checks']['slopes']['max_abs_difference'])+'</td><td>'+f(r['checks']['combined_prediction_fitted_plus_FE']['max_abs_difference'])+'</td></tr>'
 txt+='</table><p><a href="comparison.csv">Все 30 моделей CSV</a> · <a href="coefficients.csv">Все коэффициенты</a> · <a href="no_const_comparison.json">Все проверки const</a> · <a href="summary.json">Сводка JSON</a> · <a href="samples_manifest.json">Точные данные и хеши</a> · <a href="frozen_plan.json">План до расчётов</a></p><h2>Все Python outputs</h2><ul>'
 for r in rows+s['no_const_models']:txt+='<li>'+r['model']+': <a href="native/'+r['model']+'.summary.txt">TXT</a> · <a href="native/'+r['model']+'.summary.html">HTML</a></li>'
 txt+='</ul><p>Это дальнейшая проверка чувствительности к контролям. Выборки, правила P99, условные входы +100%, выходы −100%, support и классификация fund focus сохранены. Штуки расчётные и не полностью скорректированы на корпоративные действия. Никаких новых LLM оценок нет. Построчные проверки представления с const/без const остаются на домашнем ПК.</p></html>'
 (root/'review.html').write_text(txt,encoding='utf-8')

def execute(a):
 start=time.perf_counter();plan=json.loads(Path(a.plan).read_text(encoding='utf-8'));root=Path(plan['output_root']);root.mkdir(parents=True,exist_ok=True)
 if (root/'summary.json').exists():raise FileExistsError('Completed outputs exist')
 save(root/'frozen_plan.json',plan)
 def checkpoint(status,**kw):save(root/'status.json',dict(status=status,elapsed_seconds=time.perf_counter()-start,**kw));print(json.dumps(dict(status=status,**kw)),flush=True)
 checkpoint('VERIFYING_SOURCES')
 base=load(a.base,'h1ablationbase');lag=load(a.lag,'h1ablationlag');samples={};manifest={}
 for focus,src in plan['sources'].items():
  assert sha(src['path'])==src['sha256'];d=pd.read_parquet(src['path']).sort_values(['fund_id','end_date','security_id','interval_id'],kind='stable').reset_index(drop=True)
  rh=hashlib.sha256('\n'.join(d.row_id.astype(str)).encode()).hexdigest()
  assert (len(d),d.fund_id.nunique(),d.company_id.nunique(),rh)==(src['rows'],src['funds'],src['companies'],src['row_hash'])
  assert np.isfinite(d[['quantity_change_pct']+plan['targets']+plan['controls_all16']].astype(float)).all().all()
  samples[focus]=d;manifest[focus]={**src,'first_start_date':str(d.start_date.min()),'last_end_date':str(d.end_date.max()),'bytes':Path(src['path']).stat().st_size,'columns':list(d.columns)}
 save(root/'samples_manifest.json',dict(samples=manifest,plan_sha256=sha(a.plan),runner_sha256=sha(__file__),bulk_location='Referenced exact source Parquet on research-host; no new bulk copy or download.'))
 records=[];coefficients=[];cache={};nativefits=0
 for focus,d in samples.items():
  for distributed,variant in [(False,'CURRENT_ONLY'),(True,'CURRENT_PLUS_LAG1')]:
   for controlset,controls in plan['control_sets'].items():
    name=focus.upper()+'__'+variant+'__'+controlset;checkpoint('FITTING',model=name,completed=nativefits)
    res,terms,dropped=fit(d,controls,distributed,True,base);rec,co=export_result(d,res,terms,dropped,name,focus,controlset,variant,root,lag)
    rec['controls_included']='|'.join(controls);rec['source_row_hash']=manifest[focus]['row_hash'];records.append(rec);coefficients+=co;cache[(focus,controlset,variant)]=res;nativefits+=1
 checks=[];nocoeff=[];norecord=[]
 for spec in plan['representation_checks']:
  focus=spec['focus'];controlset=spec['controls'];variant=spec['variant'];d=samples[focus];name=focus.upper()+'__'+variant+'__'+controlset+'__NO_EXPLICIT_CONST';checkpoint('CONST_REPRESENTATION_CHECK',model=name)
  res,terms,dropped=fit(d,plan['control_sets'][controlset],True,False,base);rec,co=export_result(d,res,terms,dropped,name,focus,controlset,variant,root,lag,False)
  norecord.append(rec);nocoeff+=co;checks.append(nocheck(d,cache[(focus,controlset,variant)],res,name,root));nativefits+=1
 if not all(x['status']=='PASS'for x in checks):raise AssertionError('Const representation checks failed')
 pd.DataFrame(records).to_csv(root/'comparison.csv',index=False,encoding='utf-8-sig');pd.DataFrame(coefficients).to_csv(root/'coefficients.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame(norecord).to_csv(root/'no_const_models.csv',index=False,encoding='utf-8-sig');pd.DataFrame(nocoeff).to_csv(root/'no_const_coefficients.csv',index=False,encoding='utf-8-sig')
 save(root/'no_const_comparison.json',{'status':'PASS','checks':checks,'numerical_check_count':sum(len(c['checks'])for c in checks)})
 summary={'status':'COMPLETE_PENDING_INDEPENDENT_QA','plan':plan,'plan_sha256':sha(a.plan),'runner_sha256':sha(__file__),'models':records,'no_const_models':norecord,'no_const_checks':checks,'native_fit_count':nativefits,'elapsed_seconds':time.perf_counter()-start,'all_pvalues':'ordinary native two-sided, no multiple-testing adjustment','comparison_rule':'Same exact rows per focus for every control set and both variants. No new filters or cuts.'}
 save(root/'summary.json',summary);render(root,summary);checkpoint(summary['status'],native_fits=nativefits,representation_checks=len(checks))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--base',required=True);p.add_argument('--lag',required=True);a=p.parse_args()
 try:execute(a)
 except Exception:traceback.print_exc();raise
