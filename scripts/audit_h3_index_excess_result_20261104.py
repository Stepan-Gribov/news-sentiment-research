"""Compact independent source-leg/sample/LSDV-CR1 QA of sealed24 index cells.
No production imports, new model variants or score-panel reconstruction.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib,datetime as dt,time,traceback
import numpy as np,pandas as pd,duckdb
from scipy import stats
from qa_h3_index_excess_independent_20261104 import expected_sample,regression_replica,planned_cells,Y,P,SPEC_SHA,SCORES_SHA,BENCHMARK_SHA
from qa_h3_relaxed_news_independent_20261104 import ordered_sample_sha256

BASE=Path('./analysis_outputs');RUN=BASE/'h3_index_excess_20261104/outputs/21bd79859fefaa8e'
OUT=BASE/'h3_index_excess_20261104/qa/independent_21bd79859fefaa8e_20261104';OUT.mkdir(parents=True,exist_ok=True)
EXPECTED_RUN_SHA='d0721c8a87067d5788ca12552212834daa5d7a2b4b18aa7417a5d10636b1cc51'
START=time.monotonic();checks=[];details={}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def clean(v):
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,np.ndarray):return clean(v.tolist())
 if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
 if isinstance(v,np.integer):return int(v)
 if isinstance(v,np.bool_):return bool(v)
 if isinstance(v,(Path,pd.Timestamp,dt.date)):return str(v)
 return v
def save(name,v):
 p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(clean(v),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def stage(s,**kw):save('checkpoint.json',dict(stage=s,pid=os.getpid(),elapsed_seconds=time.monotonic()-START,**kw));print(s,kw,flush=True)
def ck(n,v,**kw):checks.append(dict(check=n,status='PASS' if bool(v) else 'FAIL',**kw))
def eq(n,a,b,rtol=1e-8,atol=1e-11):
 a=np.asarray(a,float);b=np.asarray(b,float);same=a.shape==b.shape and np.allclose(a,b,rtol=rtol,atol=atol,equal_nan=True);delta=abs(a-b) if a.shape==b.shape else np.array([])
 ck(n,same,max_absolute_difference=float(np.nanmax(delta)) if np.isfinite(delta).any() else 0.)
def q(p):return "'"+str(p).replace('\\','/').replace("'","''")+"'"
def records(m,prefix):
 for group in ['inputs','code','files']:
  for name,z in m.get(group,{}).items():
   p=Path(z['path']);ck(prefix+':'+group+':'+name,p.stat().st_size==z['bytes'] and sha(p)==z['sha256'])

def main():
 stage('PIN_SEALED21bd_SPEC_SCORES_AND_BENCHMARKS')
 m=json.loads((RUN/'manifest.json').read_text(encoding='utf-8'));ck('correct_final_run',sha(RUN/'manifest.json')==EXPECTED_RUN_SHA);ck('frozen_spec',m['spec_sha256']==SPEC_SHA)
 ck('frozen_score_and_benchmark_inputs',m['inputs']['scores']['sha256']==SCORES_SHA and m['inputs']['benchmark_monthly']['sha256']==BENCHMARK_SHA)
 records(m,'before');c=duckdb.connect(config={'threads':2,'memory_limit':'2GB','temp_directory':str(OUT/'spill')})
 def view(name,path):c.execute('CREATE TEMP VIEW '+name+' AS FROM read_parquet('+q(path)+')')
 def sq(n,sql):k=int(c.execute(sql).fetchone()[0]);ck(n,k==0,violations=k)
 def bad(a,b,tol='1e-12'):return '(coalesce(isfinite('+a+'),false)<>coalesce(isfinite('+b+'),false) OR (coalesce(isfinite('+a+'),false) AND abs('+a+'-'+b+')>'+tol+'))'
 for n,k in [('o','scores'),('f','fund_returns'),('b','benchmark_monthly')]:view(n,m['inputs'][k]['path'])
 for n,path in [('a','origin_outcome_control_ledger.parquet'),('l','monthly_outcome_control_legs.parquet'),('p','index_relative_analysis_panel.parquet'),('ba','benchmark_monthly_adapter.parquet')]:view(n,RUN/path)
 for n,keys in [('o','fund_id,interval_id'),('f','fund_id,month'),('b','series_id,month'),('a','fund_id,interval_id,horizon'),('l','fund_id,interval_id,kind,month'),('p','fund_id,interval_id,horizon')]:sq('unique:'+n,'SELECT count(*) FROM (SELECT '+keys+',count(*) n FROM '+n+' GROUP BY ALL HAVING n>1)')
 ck('all28976_origins_three_outcomes_four_legs',c.execute('SELECT count(*) FROM o').fetchone()[0]==28976 and c.execute('SELECT count(*) FROM a').fetchone()[0]==86928 and c.execute('SELECT count(*) FROM l').fetchone()[0]==115904)
 sourcecols=[x[0] for x in c.execute('DESCRIBE b').fetchall()]
 sq('benchmark_adapter_all_raw_fields_unchanged','SELECT count(*) FROM b JOIN ba USING(series_id,month) WHERE '+' OR '.join('b."'+x+'" IS DISTINCT FROM ba."'+x+'"' for x in sourcecols if x not in ['series_id','month']))
 sq('benchmark_aliases_no_duplicates_or_new_gate','SELECT count(*) FROM ba WHERE benchmark_id IS DISTINCT FROM series_id OR benchmark_verified IS DISTINCT FROM return_verified')
 stage('EVERY_FUTURE_AND_LATEST_PAST_SOURCE_LEG')
 fc=[x[0] for x in c.execute('DESCRIBE f').fetchall()];bc=sourcecols
 fpub='try_cast(publication_timestamp AS TIMESTAMPTZ)' if 'publication_timestamp' in fc else 'NULL::TIMESTAMPTZ'
 bpub='try_cast(publication_timestamp AS TIMESTAMPTZ)' if 'publication_timestamp' in bc else 'NULL::TIMESTAMPTZ'
 c.execute('''CREATE TEMP TABLE ff AS SELECT *,coalesce(diagnostic_eligible,false) AND coalesce(identity_verified,false)
  AND coalesce(isfinite(fund_return),false) AND fund_return>=-1 AND start_valuation_date IS NOT NULL AND end_valuation_date IS NOT NULL
  AND start_valuation_date<=end_valuation_date AND return_period_end IS NOT NULL AND strftime(return_period_end,'%Y-%m')=month qa_known,'''+fpub+' qa_publication FROM f')
 c.execute('''CREATE TEMP TABLE bb AS SELECT *,coalesce(return_verified,false) AND coalesce(isfinite(benchmark_return),false) AND benchmark_return>-1
  AND index_start_date IS NOT NULL AND index_end_date IS NOT NULL AND index_start_date<=index_end_date AND return_period_end IS NOT NULL
  AND strftime(return_period_end,'%Y-%m')=month qa_known,'''+bpub+' qa_publication FROM b')
 c.execute("""CREATE TEMP TABLE oo AS SELECT *,CASE fund_focus WHEN 'equity' THEN 'IMOEX' WHEN 'mixed' THEN 'IMOEX' WHEN 'bond' THEN 'RGBITR' END qa_benchmark,
  strftime(CASE WHEN last_day(end_date)<=end_date THEN date_trunc('month',end_date) ELSE date_trunc('month',end_date)-INTERVAL 1 MONTH END,'%Y-%m') qa_last_allowed FROM o""")
 c.execute("""CREATE TEMP TABLE future_keys AS SELECT oo.*,k.j,strftime(date_trunc('month',end_date)+k.j*INTERVAL 1 MONTH,'%Y-%m') AS month_key FROM oo CROSS JOIN range(1,4) k(j)""")
 c.execute('''CREATE TEMP TABLE ef AS SELECT o.fund_id,o.interval_id,o.end_date,o.identity_origin_verified,o.qa_benchmark,o.j,o.month_key,
  f.fund_id IS NOT NULL fund_exists,b.series_id IS NOT NULL index_exists,coalesce(f.qa_known,false) fknown,coalesce(b.qa_known,false) bknown,
  f.fund_return,b.benchmark_return,f.start_valuation_date fstart,f.end_valuation_date fend,f.return_period_end fperiod,
  b.index_start_date bstart,b.index_end_date bend,b.return_period_end bperiod,
  coalesce(f.qa_known,false) AND coalesce(b.qa_known,false) AND o.qa_benchmark IS NOT NULL complete,
  CASE WHEN f.qa_known AND b.qa_known AND o.qa_benchmark IS NOT NULL THEN f.fund_return-b.benchmark_return END component
  FROM future_keys o LEFT JOIN ff f ON f.fund_id=o.fund_id AND f.month=o.month_key LEFT JOIN bb b ON b.series_id=o.qa_benchmark AND b.month=o.month_key''')
 sq('every_future_leg_source_return_date_and_eligibility','''SELECT count(*) FROM l JOIN ef e ON l.fund_id=e.fund_id AND l.interval_id=e.interval_id AND l.month=e.month_key
  WHERE l.kind='FUTURE' AND (l.benchmark_id IS DISTINCT FROM e.qa_benchmark OR l.fund_month_exists IS DISTINCT FROM e.fund_exists OR l.benchmark_month_exists IS DISTINCT FROM e.index_exists OR l.fund_month_known IS DISTINCT FROM e.fknown OR l.benchmark_month_known IS DISTINCT FROM e.bknown OR l.component_verified IS DISTINCT FROM e.complete
  OR l.fund_start_valuation_date IS DISTINCT FROM e.fstart OR l.fund_end_valuation_date IS DISTINCT FROM e.fend OR l.benchmark_start_valuation_date IS DISTINCT FROM e.bstart OR l.benchmark_end_valuation_date IS DISTINCT FROM e.bend OR l.fund_return_period_end IS DISTINCT FROM e.fperiod OR l.benchmark_return_period_end IS DISTINCT FROM e.bperiod OR '''+bad('l.fund_return','e.fund_return')+' OR '+bad('l.benchmark_return','e.benchmark_return')+' OR '+bad('l.index_excess_component','e.component')+')')
 sq('future_exact_month_keys','SELECT count(*) FROM ((SELECT fund_id,interval_id,month FROM l WHERE kind=\'FUTURE\' EXCEPT ALL SELECT fund_id,interval_id,month_key FROM ef) UNION ALL (SELECT fund_id,interval_id,month_key FROM ef EXCEPT ALL SELECT fund_id,interval_id,month FROM l WHERE kind=\'FUTURE\'))')
 c.execute('''CREATE TEMP TABLE ep AS SELECT o.fund_id,o.interval_id,f.month AS past_month,f.fund_return-b.benchmark_return AS past_value,
  f.start_valuation_date fstart,f.end_valuation_date fend,f.return_period_end fperiod,b.index_start_date bstart,b.index_end_date bend,b.return_period_end bperiod
  FROM oo o JOIN ff f ON f.fund_id=o.fund_id AND f.qa_known JOIN bb b ON b.series_id=o.qa_benchmark AND b.month=f.month AND b.qa_known
  WHERE last_day(strptime(f.month,'%Y-%m'))<=o.end_date AND f.return_period_end<=o.end_date AND f.start_valuation_date<=o.end_date AND f.end_valuation_date<=o.end_date
  AND b.return_period_end<=o.end_date AND b.index_start_date<=o.end_date AND b.index_end_date<=o.end_date
  AND (f.qa_publication IS NULL OR f.qa_publication<=timezone('Europe/Moscow',o.end_date))
  AND (b.qa_publication IS NULL OR b.qa_publication<=timezone('Europe/Moscow',o.end_date))
  QUALIFY row_number() OVER(PARTITION BY o.fund_id,o.interval_id ORDER BY f.month DESC)=1''')
 sq('all_past_legs_latest_eligible_month_and_value','''SELECT count(*) FROM l JOIN oo o USING(fund_id,interval_id) LEFT JOIN ep e USING(fund_id,interval_id)
  WHERE l.kind='PAST' AND (l.month IS DISTINCT FROM coalesce(e.past_month,o.qa_last_allowed) OR l.component_verified IS DISTINCT FROM (e.past_month IS NOT NULL) OR '''+bad('l.index_excess_component','e.past_value')+')')
 sq('known_past_dates_match_source','''SELECT count(*) FROM l JOIN ep e USING(fund_id,interval_id) WHERE l.kind='PAST' AND (l.fund_start_valuation_date IS DISTINCT FROM e.fstart OR l.fund_end_valuation_date IS DISTINCT FROM e.fend OR l.fund_return_period_end IS DISTINCT FROM e.fperiod OR l.benchmark_start_valuation_date IS DISTINCT FROM e.bstart OR l.benchmark_end_valuation_date IS DISTINCT FROM e.bend OR l.benchmark_return_period_end IS DISTINCT FROM e.bperiod)''')
 c.execute('''CREATE TEMP TABLE ex0 AS SELECT o.fund_id,o.interval_id,o.end_date,o.universe_eligible,o.identity_origin_verified,o.qa_benchmark,o.qa_last_allowed,h.h,
  count(*) FILTER(WHERE f.complete) complete_months,count(*) FILTER(WHERE f.fknown) fund_months,count(*) FILTER(WHERE f.bknown) index_months,
  avg(f.component) raw_Y,max(CASE WHEN f.j=1 THEN f.fstart END) fund_anchor
  FROM oo o CROSS JOIN range(1,4) h(h) JOIN ef f ON f.fund_id=o.fund_id AND f.interval_id=o.interval_id AND f.j<=h.h GROUP BY ALL''')
 c.execute('''CREATE TEMP TABLE ex AS SELECT x.*,e.past_month,CASE WHEN x.identity_origin_verified THEN e.past_value END AS past_value,
  date_diff('month',strptime(e.past_month,'%Y-%m'),x.end_date) control_age,
  date_diff('month',strptime(e.past_month,'%Y-%m'),strptime(x.qa_last_allowed,'%Y-%m')) control_gap,
  coalesce(x.identity_origin_verified,false) AND x.qa_benchmark IS NOT NULL AND x.complete_months=x.h AND x.fund_anchor IS NOT NULL AND x.fund_anchor>=x.end_date valid_Y,
  CASE WHEN x.identity_origin_verified AND x.qa_benchmark IS NOT NULL AND x.complete_months=x.h AND x.fund_anchor>=x.end_date THEN x.raw_Y END value_Y
  FROM ex0 x LEFT JOIN ep e USING(fund_id,interval_id)''')
 sq('all86928_outcomes_and_controls_from_source','''SELECT count(*) FROM a JOIN ex e ON a.fund_id=e.fund_id AND a.interval_id=e.interval_id AND a.horizon=e.h
  WHERE a.benchmark_id IS DISTINCT FROM e.qa_benchmark OR a.past_reference_month IS DISTINCT FROM e.past_month OR a.past_latest_allowed_month IS DISTINCT FROM e.qa_last_allowed OR '''+bad('a.'+Y,'e.value_Y')+' OR '+bad('a.index_excess_return','e.value_Y')+' OR '+bad('a.'+P,'e.past_value')+' OR '+bad('a.control_age_months','e.control_age')+' OR '+bad('a.control_gap_months','e.control_gap')+''' OR a.past_control_known IS DISTINCT FROM coalesce(isfinite(e.past_value),false) OR a.outcome_known IS DISTINCT FROM e.valid_Y OR a.future_months_complete IS DISTINCT FROM e.complete_months OR a.fund_future_months_eligible IS DISTINCT FROM e.fund_months OR a.benchmark_future_months_eligible IS DISTINCT FROM e.index_months''')
 sq('calendar_end_control_and_gap_flags','''SELECT count(*) FROM a JOIN oo o USING(fund_id,interval_id) WHERE a.past_reference_month IS NOT NULL AND
  (last_day(strptime(a.past_reference_month,'%Y-%m'))>a.origin_date OR a.control_uses_older_available_month IS DISTINCT FROM (a.past_reference_month<o.qa_last_allowed))''')
 sq('origin_outcome_keys_complete','SELECT count(*) FROM ((SELECT fund_id,interval_id,horizon FROM a EXCEPT ALL SELECT fund_id,interval_id,h FROM ex) UNION ALL (SELECT fund_id,interval_id,h FROM ex EXCEPT ALL SELECT fund_id,interval_id,horizon FROM a))')
 sq('past_and_future_origin_dates_unchanged','SELECT count(*) FROM a JOIN o USING(fund_id,interval_id) WHERE a.origin_date IS DISTINCT FROM o.end_date')
 oc=[x[0] for x in c.execute('DESCRIBE o').fetchall()];pc={x[0] for x in c.execute('DESCRIBE p').fetchall()}
 sq('all_frozen_NTS_scores_statuses_dates_and_universe_unchanged','SELECT count(*) FROM p JOIN o USING(fund_id,interval_id) WHERE '+' OR '.join('p."'+x+'" IS DISTINCT FROM o."'+x+'"' for x in oc if x in pc and x not in ['fund_id','interval_id']))
 sq('canonical_NTS_aliases','SELECT count(*) FROM p WHERE '+bad('NTS_Q_current','NTS_Q_current_relaxed','0')+' OR '+bad('NTS_Q_lagnews','NTS_Q_lagnews_relaxed','0'))
 details['leg_coverage']=dict(origins=28976,outcomes=86928,source_legs=115904,latest_available_past_pairs=int(c.execute('SELECT count(*) FROM ep').fetchone()[0]))
 details['past_control_cases']=c.execute('''SELECT count(*) FILTER(WHERE h=1 AND past_value IS NOT NULL AND control_gap>0) older_available_control_origins,max(control_gap) max_control_gap_months FROM ex''').fetchdf().to_dict('records')[0]
 details['source_leg_examples']=c.execute('''SELECT e.fund_id,e.interval_id,e.end_date,e.qa_benchmark,e.past_month,e.past_value,e.value_Y,e.h FROM ex e WHERE e.valid_Y AND e.past_value IS NOT NULL AND e.h=3 ORDER BY e.fund_id,e.interval_id LIMIT 3''').fetchdf().to_dict('records')
 stage('24_FULL_DUMMY_FE_CR1_AND_EXACT_SAMPLES')
 pool=c.execute('''SELECT o.*,x.h horizon,x.value_Y AS mean_monthly_index_excess_return,x.past_value AS past_index_excess_return,
  x.qa_benchmark benchmark_id,x.complete_months,x.fund_months,x.index_months,x.fund_anchor,x.valid_Y,x.past_month,x.control_age,x.control_gap,
  o.NTS_Q_current_relaxed AS NTS_Q_current,o.NTS_Q_lagnews_relaxed AS NTS_Q_lagnews FROM ex x JOIN o USING(fund_id,interval_id)''').fetchdf()
 models=json.loads((RUN/'all_model_details.json').read_text(encoding='utf-8'));ck('exact24_fixed_grid',len(models)==24 and {(x['group'],x['horizon'],x['variant']) for x in models}==set(planned_cells()))
 flow=pd.read_csv(RUN/'sample_flow.csv');replicas=[];samples=[];used_first=None
 for model in models:
  mid=model['model_id'];g=pool.loc[pool.horizon.eq(model['horizon'])].copy();expected,removed=expected_sample(g,distributed=model['distributed'],group=model['group'])
  path=RUN/'regression'/mid;used=pd.read_parquet(path/'used_sample.parquet').sort_values(['fund_id','end_date','interval_id']).reset_index(drop=True)
  ck('exact_sample_keys:'+mid,not used.duplicated(['fund_id','interval_id']).any() and used[['fund_id','interval_id']].equals(expected[['fund_id','interval_id']]))
  n=len(used);G=used.fund_id.nunique();ck('sample_N_G_hash_singletons:'+mid,n==model['nobs'] and G==model['funds'] and len(removed)==model['singleton_rows_removed'] and ordered_sample_sha256(used)==model['sample_hash'])
  eq('sample_source_values:'+mid,used[[Y,P,'NTS_Q_current','NTS_Q_lagnews']],expected[[Y,P,'NTS_Q_current','NTS_Q_lagnews']],rtol=1e-10,atol=1e-12)
  ck('sample_benchmark_origin_assignment:'+mid,np.array_equal(used.benchmark_id.to_numpy(),expected.benchmark_id.to_numpy()))
  ck('sample_past_month_latest:'+mid,np.array_equal(used.past_reference_month.to_numpy(),expected.past_month.to_numpy()))
  ck('FE_one_way_cluster_and_main_df:'+mid,model['fund_FE'] and not model['time_FE'] and model['clusters']==['fund_id'] and model['reference_df']==G-1 and model['weak_cluster_inference']==(G<30) and model['significance_stars_suppressed']==(G<30) and not model['multiple_testing_adjustments'])
  fit=regression_replica(used,distributed=model['distributed']);ck('estimability:'+mid,fit['status']==model['status'])
  if model['status']!='ESTIMATED':continue
  names=fit['terms'];native=pd.DataFrame(model['coefficients']).set_index('term').loc[names];cov=pd.read_csv(path/'full_covariance.csv',index_col=0,encoding='utf-8-sig').loc[names,names]
  eq('LSDV_beta:'+mid,fit['beta'],native.beta,rtol=1e-7,atol=1e-10);eq('CR1_full_covariance:'+mid,fit['covariance'],cov,rtol=1e-6,atol=1e-11)
  ck('full_rank_residual_df_absorbed_control:'+mid,fit['rank']==model['K_full'] and fit['residual_df']==model['residual_df'] and fit['absorbed_columns']==model['absorbed_columns'])
  eq('CR1_correction:'+mid,[fit['CR1_correction']],[model['CR1_correction']]);eq('within_R2:'+mid,[fit['within_R2']],[model['within_R2']])
  for field in ['SE','t','p','CI_low','CI_high']:eq('tGminus1_'+field+':'+mid,fit[field],native[field],rtol=1e-6,atol=1e-9)
  tests={x['test']:x for x in model['target_tests']}
  for j,term in enumerate(['NTS_Q_current']+(['NTS_Q_lagnews'] if model['distributed'] else []),start=1):
   z=tests[term+'=0'];eq('individual_target:'+mid+':'+term,[z[k] for k in ['beta','SE','t','p','CI_low','CI_high']],[fit['beta'][j]]+[fit[k][j] for k in ['SE','t','p','CI_low','CI_high']],rtol=1e-6,atol=1e-9)
  if model['distributed']:
   z=tests['current+lagnews=0'];t=fit['sum_test'];eq('sum_full_covariance:'+mid,[z[k] for k in ['beta','SE','t','p','CI_low','CI_high']],[t[k] for k in ['beta','SE','t','p','CI_low','CI_high']],rtol=1e-6,atol=1e-9)
   z=tests['joint_current_lagnews=0'];t=fit['joint_test'];ck('joint_identification_guard:'+mid,z['status']==t['status'])
   if t['status']=='defined':eq('joint_F_p:'+mid,[z['F'],z['p']],[t['F'],t['p']],rtol=1e-6,atol=1e-9)
   else:ck('undefined_joint_honest_NA:'+mid,not np.isfinite([z['F'],z['p']]).any())
  base=g.universe_eligible.eq(True)
  if model['group']!='pooled':base&=g.fund_focus.eq(model['group'])
  assigned=base&g.benchmark_id.notna();fund_complete=assigned&g.fund_months.eq(model['horizon'])&g.fund_anchor.notna()&g.fund_anchor.ge(g.end_date)
  out=fund_complete&g.valid_Y;controlled=out&np.isfinite(g[P]);current=controlled&g.current_eligible;both=current&g.lagnews_eligible
  row=flow.loc[flow.model_id.eq(mid)].iloc[0]
  ck('source_flow_no_alpha_history_gate:'+mid,all(int(row[k])==int(v.sum()) for k,v in [('inherited_universe',base),('assigned_benchmark',assigned),('complete_future_fund_with_anchor',fund_complete),('complete_future_fund_and_index',out),('past_control',controlled),('current_known',current),('both_known',both)]) and row.before_singletons==n+len(removed) and row.final_N==n)
  oldroot=Path(m['inputs']['old_relaxed_model_manifest']['path']).parent;oldid=f"UNIT_VALUE_KNOWN_ADJUSTMENTS__{model['group']}__h{model['horizon']}__{model['variant']}"
  old=pd.read_parquet(oldroot/'regression'/oldid/'used_sample.parquet',columns=['row_id']);oldids=set(old.row_id.astype(str));ids=set(used.row_id.astype(str))
  ck('old_alpha_overlap_metadata:'+mid,row.old_relaxed_alpha_N==len(oldids) and row.overlap_old==len(ids&oldids) and row.added_vs_old==len(ids-oldids) and row.removed_vs_old==len(oldids-ids))
  samples.append(dict(model_id=mid,n=n,funds=int(G),sample_hash=model['sample_hash'],known_current_unknown_lag=int((used.current_eligible&~used.lagnews_eligible).sum()),old_alpha_overlap=len(ids&oldids),added_vs_alpha=len(ids-oldids),older_available_control=int(used.control_uses_older_available_month.sum())))
  replicas.append(dict(model_id=mid,n=n,funds=int(G),rank=fit['rank'],residual_df=fit['residual_df'],reference_df=G-1,sample_hash=model['sample_hash'],max_beta_difference=float(np.max(abs(fit['beta']-native.beta.to_numpy()))),max_cov_difference=float(np.max(abs(fit['covariance']-cov.to_numpy()))),independent_full_dummy_fit=True))
  if model['group']=='pooled' and model['horizon']==1 and not model['distributed']:used_first=used[['fund_id','interval_id','end_date']]
  stage('INDEX_CELL_INDEPENDENTLY_REPLICATED',completed=len(replicas),model_id=mid)
 ck('all24_models_independently_replicated',len(replicas)==24)
 if used_first is not None:
  c.register('used_first',used_first)
  history=c.execute('''SELECT u.fund_id,u.interval_id,count(f.month) n_history FROM used_first u LEFT JOIN ff f ON f.fund_id=u.fund_id AND f.qa_known AND last_day(strptime(f.month,'%Y-%m'))<=u.end_date AND f.return_period_end<=u.end_date GROUP BY ALL''').fetchdf()
  details['fewer_than36_history_witness_count']=int(history.n_history.lt(36).sum());ck('valid_observations_without36_history_are_retained',details['fewer_than36_history_witness_count']>0)
 details.update(regression_replicas=replicas,all24_samples=samples,run_manifest_sha256=EXPECTED_RUN_SHA,spec_sha256=SPEC_SHA)
 stage('FINAL_SOURCE_AND_RESULT_HASHES');records(m,'after');ck('run_manifest_unchanged',sha(RUN/'manifest.json')==EXPECTED_RUN_SHA)

try:
 main()
 report=dict(status='PASS' if all(x['status']=='PASS' for x in checks) else 'FAIL',audit_type='EXTERNAL_INDEPENDENT_INDEX_EXCESS24',run=str(RUN),audit_code_sha256=sha(__file__),production_modules_imported=False,executor_checks_counted_as_external=False,checks_pass=sum(x['status']=='PASS' for x in checks),checks_fail=sum(x['status']=='FAIL' for x in checks),checks=checks,details=details,completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),runtime_seconds=time.monotonic()-START,limitations=['NTS reused by sealed hash; no655k-position or oldalpha audit repeated.','Publication vintage is only constrained where truly known; source dating does not prove a real-time trading strategy.','IMOEX price return and RGBITR total return have different conventions; broad fund return is not certified payout-complete total return.','Index subtraction is descriptive performance, not estimated beta/factor alpha or causal managerial skill.'])
 save('independent_qa.json',report);save('summary.json',{k:v for k,v in report.items() if k!='checks'});stage('COMPLETED',status=report['status'],checks_pass=report['checks_pass'],checks_fail=report['checks_fail'])
 if report['checks_fail']:print('FAILURES',json.dumps(clean([x for x in checks if x['status']=='FAIL']),ensure_ascii=False));raise SystemExit(1)
except Exception:
 save('execution_failure.json',dict(traceback=traceback.format_exc(),checks=checks,details=details));stage('EXECUTION_FAILED');raise
