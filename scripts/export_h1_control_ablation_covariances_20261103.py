"""Native covariance export only; unchanged frozen H1 ablation models."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ.setdefault(k,'1')
from pathlib import Path
import importlib.util,json,sys,time
import numpy as np,pandas as pd
ROOT=Path('./analysis_outputs/h1_control_ablation_20261103')
spec=importlib.util.spec_from_file_location('ablationexport',ROOT/'code/run_h1_control_ablation_20261103.py');m=importlib.util.module_from_spec(spec);sys.modules['ablationexport']=m;spec.loader.exec_module(m)
plan=json.loads((ROOT/'frozen_plan.json').read_text(encoding='utf-8'));summary=json.loads((ROOT/'summary.json').read_text(encoding='utf-8'))
base=m.load('./analysis_outputs/h1_macro_no_time_20261031/code/run_h1_net_fund_fe_20261030.py','ablationcovbase')
start=time.perf_counter();out={};cache={};co=pd.concat([pd.read_csv(ROOT/'coefficients.csv'),pd.read_csv(ROOT/'no_const_coefficients.csv')])
for rec in summary['models']+summary['no_const_models']:
 focus=rec['focus']
 if focus not in cache:cache[focus]=pd.read_parquet(plan['sources'][focus]['path']).sort_values(['fund_id','end_date','security_id','interval_id'],kind='stable').reset_index(drop=True)
 d=cache[focus];res,terms,dropped=m.fit(d,plan['control_sets'][rec['control_set']],rec['variant']=='CURRENT_PLUS_LAG1',rec['explicit_const'],base)
 prior=co[co.model.eq(rec['model'])].set_index('term')
 for key,value in [('beta',res.params),('se',res.std_errors),('p',res.pvalues)]:
  assert np.allclose(prior.loc[value.index,key],value,rtol=2e-8,atol=2e-10,equal_nan=True)
 out[rec['model']]={'term_order':list(res.params.index),'params':res.params.tolist(),'std_errors':res.std_errors.tolist(),'p_values':res.pvalues.tolist(),'covariance':res.cov.to_numpy().tolist(),'nobs':int(res.nobs),'df_resid':int(res.df_resid),'df_model':int(res.df_model),'rsquared_within':float(res.rsquared_within),'explicit_const':rec['explicit_const'],'focus':focus,'control_set':rec['control_set'],'controls':plan['control_sets'][rec['control_set']],'source':plan['sources'][focus],'retained_terms':terms,'dropped_controls':dropped}
m.save(ROOT/'all_model_covariances.json',{'status':'COMPLETE','models':out,'native_covariance_export_fit_count':len(out),'elapsed_seconds':time.perf_counter()-start,'fitted_values_unchanged_verified_against_coefficient_CSV':True,'exporter_sha256':m.sha(__file__)})
print(json.dumps({'status':'COMPLETE','covariance_models':len(out),'elapsed_seconds':time.perf_counter()-start}),flush=True)
