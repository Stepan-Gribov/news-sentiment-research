"""Finite 16-control all-subsets H1 search, exact cluster sufficient statistics."""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse, hashlib, html, importlib.util, json, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
try:
 from threadpoolctl import threadpool_limits
except ImportError:
 from contextlib import nullcontext
 def threadpool_limits(limits): return nullcontext()


def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x

def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2),encoding='utf-8')

def load(p,name):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

class Solver:
 def __init__(self,d,plan,root):
  self.d=d;self.names=plan['targets']+plan['controls'];self.n=len(d);self.f=d.fund_id.nunique()
  raw=d[[plan['outcome']]+self.names].astype(float)
  within=(raw-raw.groupby(d.fund_id.astype(str),sort=False).transform('mean')).to_numpy()
  self.scales=np.sqrt(np.mean(within*within,axis=0))
  if not np.isfinite(within).all() or np.any(self.scales<=0):raise AssertionError('Nonfinite/absorbed full design')
  self.z=within/self.scales;self.x=self.z[:,1:];self.y=self.z[:,0]
  self.gram=self.x.T@self.x;self.xy=self.x.T@self.y;self.yy=float(self.y@self.y)
  self.eigen=np.linalg.eigvalsh(self.gram)
  if self.eigen.min()<=0 or self.eigen.max()/self.eigen.min()>1e12:raise AssertionError('Full design rank/conditioning')
  self.ii,self.jj=np.triu_indices(self.z.shape[1]);q=len(self.ii);self.T=np.zeros((q,q))
  self.cluster_counts={}
  groups=[d.fund_id.astype(str),d.company_id.astype(str),pd.MultiIndex.from_frame(d[['fund_id','company_id']].astype(str))]
  for sign,key,g in zip([1,1,-1],['fund','company','intersection'],groups):
   codes,labels=pd.factorize(g,sort=False);ng=len(labels);self.cluster_counts[key]=ng
   moments=np.empty((ng,q))
   for j,(a,b) in enumerate(zip(self.ii,self.jj)):moments[:,j]=np.bincount(codes,weights=self.z[:,a]*self.z[:,b],minlength=ng)
   self.T+=sign*ng/(ng-1)*(self.n-1)/self.n*(moments.T@moments)
  np.savez_compressed(root/'cluster_sufficient_statistics.npz',gram=self.gram,xy=self.xy,yy=self.yy,scales=self.scales,T=self.T,ii=self.ii,jj=self.jj,n=self.n,funds=self.f)

 def fit_many(self,masks,distributed):
  nt=2 if distributed else 1;m=len(masks);q=len(self.ii)
  W=np.zeros((nt,m,q));B=np.zeros((m,nt));DF=np.zeros(m,dtype=int);RS=np.zeros(m)
  for row,mask in enumerate(masks):
   ix=[0]+([1]if distributed else[])+[2+j for j in range(16)if mask>>j&1]
   gram=self.gram[np.ix_(ix,ix)];bread=np.linalg.inv(gram);b=bread@self.xy[ix]
   fullb=np.zeros(18);fullb[ix]=b;r=np.r_[1.,-fullb]
   DF[row]=self.n-self.f-len(ix);RS[row]=float((2*b@self.xy[ix]-b@gram@b)/self.yy)
   for t in range(nt):
    factor=self.scales[0]/self.scales[1+t]
    B[row,t]=b[t]*factor;v=np.zeros(19);v[1+np.asarray(ix)]=bread[t]*factor
    w=v[self.ii]*r[self.jj]+v[self.jj]*r[self.ii]
    diag=self.ii==self.jj;w[diag]*=.5;W[t,row]=w
  WT=[w@self.T for w in W]
  var=np.column_stack([np.einsum('ij,ij->i',WT[t],W[t])*self.n/DF for t in range(nt)])
  se=np.sqrt(np.where(var>0,var,np.nan));tstat=B/se;p=2*stats.t.sf(np.abs(tstat),DF[:,None]);crit=stats.t.ppf(.975,DF)
  records=[]
  for j,mask in enumerate(masks):
   rec={'control_mask':int(mask),'n_controls':int(mask).bit_count(),'nobs':self.n,'funds':self.f,'companies':self.d.company_id.nunique(),
    'df_resid':int(DF[j]),'rsquared_within':float(RS[j]),'current_beta':B[j,0],'current_variance':var[j,0],'current_se':se[j,0],'current_p':p[j,0],
    'current_ci_low':B[j,0]-crit[j]*se[j,0],'current_ci_high':B[j,0]+crit[j]*se[j,0],
    'current_status':'FINITE'if np.isfinite(p[j,0])else'UNDEFINED_VARIANCE'}
   if distributed:
    cov=float(WT[0][j]@W[1,j]*self.n/DF[j]);sumvar=float(var[j].sum()+2*cov);sumbeta=float(B[j].sum())
    sumse=np.sqrt(sumvar)if sumvar>0 else np.nan;det=float(var[j].prod()-cov*cov)
    jointstat=(B[j,0]**2*var[j,1]-2*B[j,0]*B[j,1]*cov+B[j,1]**2*var[j,0])/det if det>0 and(var[j]>0).all()else np.nan
    rec.update(lag_beta=B[j,1],lag_variance=var[j,1],lag_se=se[j,1],lag_p=p[j,1],lag_ci_low=B[j,1]-crit[j]*se[j,1],lag_ci_high=B[j,1]+crit[j]*se[j,1],
     lag_status='FINITE'if np.isfinite(p[j,1])else'UNDEFINED_VARIANCE',current_lag_covariance=cov,sum_beta=sumbeta,sum_variance=sumvar,sum_se=sumse,
     sum_p=2*stats.t.sf(abs(sumbeta/sumse),DF[j])if np.isfinite(sumse)else np.nan,sum_ci_low=sumbeta-crit[j]*sumse,sum_ci_high=sumbeta+crit[j]*sumse,
     joint_stat=jointstat,joint_p=stats.chi2.sf(jointstat,2)if np.isfinite(jointstat)and jointstat>=0 else np.nan)
   records.append(rec)
  return pd.DataFrame(records)


def native(d,controls,distributed,lag,base,root,label):
 _,res,terms,dropped=lag.fit_native(d,controls,base,distributed)
 (root/'native').mkdir(exist_ok=True)
 (root/'native'/f'{label}.summary.txt').write_text(str(res.summary),encoding='utf-8')
 (root/'native'/f'{label}.summary.html').write_text(res.summary.as_html(),encoding='utf-8')
 rows=[];ci=res.conf_int()
 for term in res.params.index:
  rows.append(dict(model=label,term=term,beta=float(res.params[term]),se=float(res.std_errors[term]),p=float(res.pvalues[term]),ci_low=float(ci.loc[term].iloc[0]),ci_high=float(ci.loc[term].iloc[1])))
 return res,rows


def verify_native(rec,res,distributed):
 targets=['net_sentiment']+(['net_sentiment_lag1']if distributed else[]);checks=[]
 for prefix,target in zip(['current','lag'],targets):
  for suffix,value in [('beta',res.params[target]),('se',res.std_errors[target]),('p',res.pvalues[target])]:
   actual=rec[prefix+'_'+suffix];ok=np.isclose(actual,value,rtol=2e-7,atol=2e-8,equal_nan=True)
   checks.append({'metric':prefix+'_'+suffix,'fast':actual,'native':float(value),'PASS':bool(ok)})
 if int(res.df_resid)!=int(rec['df_resid']):raise AssertionError('Native df mismatch')
 if not np.isclose(res.rsquared_within,rec['rsquared_within'],rtol=1e-8,atol=1e-10):raise AssertionError('Native R2 mismatch')
 if not all(c['PASS']for c in checks):raise AssertionError('Native coefficient/covariance mismatch: '+str(checks))
 return checks


def select(frame,variant):
 reasons={}
 def add(mask,reason):reasons.setdefault(int(mask),[]).append(reason)
 add(0,'NO_CONTROLS');add(65535,'ALL_CONTROLS')
 for target in ['current']+(['lag','sum']if variant=='CURRENT_PLUS_LAG1'else[]):
  valid=frame[np.isfinite(frame[target+'_p'])]
  if len(valid):add(valid.sort_values([target+'_p','control_mask']).iloc[0].control_mask,'MIN_P_'+target.upper()+'_ANY_SIGN')
  positive=valid[valid[target+'_beta']>0]
  if len(positive):add(positive.sort_values([target+'_p','control_mask']).iloc[0].control_mask,'MIN_P_'+target.upper()+'_POSITIVE')
 valid=frame[np.isfinite(frame.current_p)]
 if len(valid):add(valid.sort_values(['current_p','control_mask'],ascending=[False,True]).iloc[0].control_mask,'MAX_P_CURRENT')
 if variant=='CURRENT_PLUS_LAG1':
  valid=frame[np.isfinite(frame.joint_p)]
  if len(valid):add(valid.sort_values(['joint_p','control_mask']).iloc[0].control_mask,'MIN_P_JOINT')
 return reasons


def execute(a):
 start=time.perf_counter();plan=json.loads(Path(a.plan).read_text(encoding='utf-8'));root=Path(plan['output_root']);root.mkdir(parents=True,exist_ok=True)
 if (root/'summary.json').exists():raise FileExistsError('Existing completed output; refusing overwrite')
 def status(s,**kw):save(root/'status.json',dict(status=s,elapsed_seconds=time.perf_counter()-start,**kw));print(json.dumps(clean(dict(status=s,**kw)),ensure_ascii=True),flush=True)
 save(root/'frozen_plan.json',plan);status('LOADING')
 d=pd.read_parquet(plan['input_panel']).sort_values(['fund_id','end_date','security_id','interval_id'],kind='stable').reset_index(drop=True)
 rh=hashlib.sha256('\n'.join(d.row_id.astype(str)).encode()).hexdigest()
 assert (len(d),d.fund_id.nunique(),d.company_id.nunique(),rh)==(plan['expected_rows'],plan['expected_funds'],plan['expected_companies'],plan['expected_row_hash'])
 if not np.isfinite(d[[plan['outcome']]+plan['targets']+plan['controls']].astype(float)).all().all():raise AssertionError('Incomplete row')
 manifest={'panel':plan['input_panel'],'sha256':sha(plan['input_panel']),'row_hash':rh,'rows':len(d),'funds':d.fund_id.nunique(),'companies':d.company_id.nunique(),'first_start_date':str(d.start_date.min()),'last_end_date':str(d.end_date.max()),'plan_sha256':sha(a.plan),'runner_sha256':sha(__file__)}
 save(root/'input_manifest.json',manifest)
 d.to_parquet(root/'used_sample.parquet',index=False,compression='zstd')
 base=load(a.base,'h1basecs');lag=load(a.lag,'h1lagcs')
 with threadpool_limits(limits=1):
  status('BUILDING_SUFFICIENT_STATISTICS');solver=Solver(d,plan,root)
  checks=[];allrows=[];selected=[];nativeco=[];benchmark={}
  for distributed,variant in [(False,'CURRENT_ONLY'),(True,'CURRENT_PLUS_LAG1')]:
   status('NATIVE_BASELINE',variant=variant)
   single=solver.fit_many([65535],distributed).iloc[0].to_dict();res,co=native(d,plan['controls'],distributed,lag,base,root,variant+'__65535');checks+=verify_native(single,res,distributed)
   benchstart=time.perf_counter();solver.fit_many(list(range(1024)),distributed);bench=time.perf_counter()-benchstart
   projected=bench*64;benchmark[variant]={'1024_seconds':bench,'projected_65536_seconds':projected}
   status('BENCHMARK_COMPLETE',variant=variant,**benchmark[variant])
   if projected>600:save(root/'benchmark.json',benchmark);raise RuntimeError('Benchmark projection exceeds10min: root decision required')
   status('FULL_GRID_RUNNING',variant=variant)
   gridstart=time.perf_counter();frame=solver.fit_many(list(range(65536)),distributed);frame.insert(0,'variant',variant)
   frame['controls_included']=frame.control_mask.map(lambda m:'|'.join(c for j,c in enumerate(plan['controls'])if m>>j&1))
   frame['controls_excluded']=frame.control_mask.map(lambda m:'|'.join(c for j,c in enumerate(plan['controls'])if not(m>>j&1)))
   frame.to_csv(root/(variant+'.all_models.csv'),index=False,encoding='utf-8-sig')
   benchmark[variant]['full_grid_seconds']=time.perf_counter()-gridstart
   reasons=select(frame,variant)
   for mask,why in reasons.items():
    controls=[c for j,c in enumerate(plan['controls'])if mask>>j&1];label=variant+f'__{mask:05d}'
    if mask==65535:nativeres=res;nativeco.extend([{**c,'model':label}for c in co])
    else:nativeres,c=native(d,controls,distributed,lag,base,root,label);nativeco.extend(c)
    rec=frame.loc[frame.control_mask==mask].iloc[0].to_dict();checks+=verify_native(rec,nativeres,distributed)
    rec.update(model=label,selection_reasons='|'.join(why));selected.append(rec)
   allrows.append(frame);status('VARIANT_COMPLETE',variant=variant,full_grid_seconds=benchmark[variant]['full_grid_seconds'],shortlist_count=len(reasons))
  allframe=pd.concat(allrows,ignore_index=True);allframe.to_csv(root/'all_models.csv',index=False,encoding='utf-8-sig')
  selected=pd.DataFrame(selected);selected.to_csv(root/'shortlist.csv',index=False,encoding='utf-8-sig');pd.DataFrame(nativeco).to_csv(root/'native_coefficients.csv',index=False,encoding='utf-8-sig')
  summaries=[]
  for variant,frame in allframe.groupby('variant',sort=False):
   rec={'variant':variant,'models':len(frame),'N':len(d),'funds':d.fund_id.nunique(),'companies':d.company_id.nunique()}
   for target in ['current']+(['lag','sum']if variant=='CURRENT_PLUS_LAG1'else[]):
    valid=np.isfinite(frame[target+'_p']);pos=frame[target+'_beta']>0;sig=frame[target+'_p']<.05
    rec[target]={'finite':int(valid.sum()),'undefined':int((~valid).sum()),'positive':int(pos.sum()),'p_lt_05':int(sig.sum()),'positive_p_lt_05':int((pos&sig).sum()),'negative_p_lt_05':int((~pos&sig).sum()),'min_p':float(frame[target+'_p'].min()),'median_p':float(frame[target+'_p'].median())}
   summaries.append(rec)
  influence=[]
  for variant,frame in allframe.groupby('variant',sort=False):
   for j,control in enumerate(plan['controls']):
    for included in [False,True]:
     sub=frame[((frame.control_mask.to_numpy()>>j)&1).astype(bool)==included]
     influence.append({'variant':variant,'control':control,'included':included,'models':len(sub),'current_beta_median':sub.current_beta.median(),'current_p_median':sub.current_p.median(),'current_p_min':sub.current_p.min(),'current_positive_p_lt_05':int(((sub.current_p<.05)&(sub.current_beta>0)).sum())})
  pd.DataFrame(influence).to_csv(root/'control_inclusion_summary.csv',index=False,encoding='utf-8-sig')
  save(root/'native_verification.json',{'PASS':sum(x['PASS']for x in checks),'FAIL':sum(not x['PASS']for x in checks),'checks':checks})
  summary={'status':'COMPLETE_PENDING_INDEPENDENT_QA','plan':plan,'manifest':manifest,'benchmark':benchmark,'counts':summaries,'shortlist':selected.to_dict('records'),'native_verification':{'PASS':sum(x['PASS']for x in checks),'FAIL':sum(not x['PASS']for x in checks)},'cluster_counts':solver.cluster_counts,'full_gram_eigenvalue_min':solver.eigen.min(),'full_gram_condition':solver.eigen.max()/solver.eigen.min(),'elapsed_seconds':time.perf_counter()-start,'all_results_file':'all_models.csv','exact_used_sample':'used_sample.parquet'}
  save(root/'summary.json',summary)
  cols=['variant','control_mask','selection_reasons','n_controls','current_beta','current_se','current_p','lag_beta','lag_se','lag_p','sum_beta','sum_p','rsquared_within','controls_excluded']
  report='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>H1 — перебор контролей, фонды акций</title><style>body{font:15px system-ui;margin:35px;color:#17233a}table{border-collapse:collapse;font-size:12px}td,th{padding:7px;border:1px solid #cbd2dc;text-align:left}th{background:#eef3fa}code{background:#eef3fa}p{max-width:1100px}</style><h1>H1: фонды акций — полный перебор 16 контролей</h1>'''
  report+=f'<p>Одинаковые {len(d):,} строк / {d.fund_id.nunique()} фонд / {d.company_id.nunique()} эмитентов. Fund FE; кластеры фонд и компания; без time FE / time clustering. Два дизайна × 65 536 наборов. Все обычные двусторонние p без поправок. Выбор моделей сделан после просмотра полной сетки; это разведочный поиск.</p>'
  report+='<h2>Сводка всех моделей</h2><pre>'+html.escape(json.dumps(clean(summaries),ensure_ascii=False,indent=2))+'</pre>'
  report+='<h2>Контрольные модели и крайние результаты</h2>'+selected[[c for c in cols if c in selected]].to_html(index=False,float_format=lambda v:f'{v:.6g}')
  report+='<h2>Проверяемость</h2><p><a href="shortlist.csv">Отобранные модели CSV</a> · <a href="native_coefficients.csv">Все коэффициенты отобранных native моделей</a> · <a href="control_inclusion_summary.csv">Включение каждого контроля</a> · <a href="summary.json">Полные параметры JSON</a> · <a href="input_manifest.json">Источники и хеши</a> · <a href="frozen_plan.json">План до расчёта</a> · <a href="native_verification.json">Сверка fast / PanelOLS</a></p><p>Все 131 072 результата и полный used sample остаются на домашнем ПК: <code>'+html.escape(str(root))+'</code>. Файлы all_models.csv / used_sample.parquet. Никакие отсечки и строки между моделями не менялись.</p><ul>'
  for rec in selected.to_dict('records'):
   label=rec['model'];report+='<li>'+html.escape(label)+': <a href="native/'+label+'.summary.txt">Python TXT</a> · <a href="native/'+label+'.summary.html">Python HTML</a></li>'
  report+='</ul><p>Штуки расчётные (стоимость / котировка); вход +100% условный, выход −100%. Нет полного corporate-action adjustment. Порог 60/24 — отбор до пропусков лага; на этой общей выборке минимум 21 интервал. Closing controls относятся к конечной дате, не являются гарантированно предопределёнными.</p></html>'
  (root/'review.html').write_text(report,encoding='utf-8')
  status('COMPLETE_PENDING_INDEPENDENT_QA',models=len(allframe),native_models=len(selected))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--base',required=True);p.add_argument('--lag',required=True);a=p.parse_args()
 try:execute(a)
 except Exception:
  traceback.print_exc();raise
