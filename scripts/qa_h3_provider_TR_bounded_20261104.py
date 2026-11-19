"""Bounded audit of a separately sealed, research-approved four-fund TR run.
No production estimators imported; three source-based past/future alpha replications.
"""
from pathlib import Path
import json,hashlib,datetime,sys
import numpy as np,pandas as pd
from scipy import linalg
B=Path('./analysis_outputs/h3_alpha_20261104');R=Path(sys.argv[1]);EXPECTED_SNAPSHOT_SHA=sys.argv[2];OUT=B/'qa/external_independent_run_d0440fad412ee5e9_20261104/provider_TR_bounded';OUT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[];examples=[]
def ck(n,v,**kw):checks.append(dict(check=n,status='PASS' if bool(v) else 'FAIL',**kw))
def eq(n,a,b):ck(n,np.allclose(np.asarray(a,float),np.asarray(b,float),rtol=1e-9,atol=1e-11,equal_nan=True),max_absolute_difference=float(np.nanmax(np.abs(np.asarray(a,float)-np.asarray(b,float)))))
m=json.loads((R/'manifest.json').read_text(encoding='utf-8'));before=sha(R/'manifest.json')
for group in ['inputs','files']:
 for key,z in m[group].items():ck(group+'_SHA:'+key,sha(z['path'])==z['sha256'])
snapshot=Path(m['inputs']['snapshot']['path']);ck('exact_approved_snapshot_SHA',sha(snapshot)==EXPECTED_SNAPSHOT_SHA)
s=pd.DataFrame(json.loads(snapshot.read_text(encoding='utf-8'))['rows']).sort_values(['fund_id','month']);r=pd.read_parquet(R/'approved_monthly_provider_returns.parquet').sort_values(['fund_id','month'])
ck('exact_approved_keys_no_backfill',len(r)==len(s) and not r.duplicated(['fund_id','month']).any() and r[['fund_id','month']].astype(str).equals(s[['fund_id','month']].astype(str)) and r.month.ge('2022-01').all())
ck('four_funds_exact_approved_registrations',r.fund_id.nunique()==4 and r[['fund_id','registration_number']].astype(str).equals(s[['fund_id','registration_number']].astype(str)))
eq('approved_printed_return_preserved',r.fund_return,s.return_decimal)
ck('no_NAV_promotion_or_second_adjustment',r.raw_NAV_history_promoted_TR.eq(False).all() and r.additional_payout_or_split_adjustment_applied.eq(False).all())
ck('cash_ledger_completeness_not_invented',r.cash_event_history_complete.eq(False).all())
for z in s.itertuples(index=False):ck('snapshot_PDF_SHA:'+z.month,sha(z.source_path)==z.source_sha256)
a=pd.read_parquet(R/'lookback60_min36_alpha.parquet');training=pd.read_parquet(R/'lookback60_min36_training.parquet');f=pd.read_parquet(m['inputs']['factors']['path'])
ck('no_pre2022_training_backfill',training.month.ge('2022-01').all())
ck('unique_alpha_and_training_rows',not a.duplicated(['fund_id','interval_id','horizon']).any() and not training.duplicated(['training_id','month']).any())
g=r.merge(f[['month','risk_free','equity_return','bond_return','factors_verified']],on='month',validate='many_to_one');g['period']=pd.PeriodIndex(g.month,freq='M');groups={str(k):v.sort_values('period').set_index('period',drop=False) for k,v in g.groupby('fund_id')}
eligible=a.loc[a.horizon.eq(3)&a.alpha.notna()].sort_values(['fund_id','origin_date']);selected=eligible.groupby('fund_id').head(1).interval_id.head(3).tolist();ck('three_distinct_eligible_origins',len(set(selected))==3)
for interval in selected:
 rows=a[a.interval_id.eq(interval)].sort_values('horizon');v=rows.iloc[0];origin=pd.Timestamp(v.origin_date);g=groups[str(v.fund_id)];month=origin.to_period('M');last=month if month.end_time.normalize()<=origin else month-1
 t=g[(g.index>=last-59)&(g.index<=last)&g.return_verified.eq(True)&g.factors_verified.eq(True)&pd.to_datetime(g.start_valuation_date).le(origin)&pd.to_datetime(g.end_valuation_date).le(origin)]
 y=(t.fund_return-t.risk_free).to_numpy(float);X=np.column_stack([np.ones(len(t)),t.equity_return-t.risk_free,t.bond_return-t.risk_free]);b=linalg.lstsq(X,y,lapack_driver='gelsy')[0]
 rec=training[training.training_id.eq(v.training_id)]
 ck('past_month_members:'+interval,rec.month.tolist()==t.month.tolist() and len(t)>=36 and (t.index>=last-11).sum()>=9 and t.end_valuation_date.max()<=origin)
 eq('past_beta:'+interval,b,[v.past_alpha,v.beta_equity_return,v.beta_bond_return]);eq('past_design:'+interval,rec[['fund_excess','x_equity_return','x_bond_return']],np.column_stack([y,X[:,1:]]))
 legsout=[]
 for z in rows.itertuples(index=False):
  months=pd.period_range(month+1,month+z.horizon,freq='M');legs=g.reindex(months)
  ck('complete_future_months:'+interval+':h'+str(z.horizon),len(legs)==z.horizon and legs.return_verified.eq(True).all() and legs.factors_verified.eq(True).all() and pd.Timestamp(legs.start_valuation_date.iloc[0])>=origin)
  abnormal=(legs.fund_return-legs.risk_free).to_numpy(float)-np.column_stack([legs.equity_return-legs.risk_free,legs.bond_return-legs.risk_free])@b[1:]
  eq('future_alpha_no_intercept_subtraction:'+interval+':h'+str(z.horizon),[abnormal.mean()],[z.alpha])
  legsout.append(dict(horizon=int(z.horizon),months=legs.month.tolist(),fund_return=legs.fund_return.tolist(),risk_free=legs.risk_free.tolist(),equity_return=legs.equity_return.tolist(),bond_return=legs.bond_return.tolist(),abnormal=abnormal.tolist(),alpha_recomputed=float(abnormal.mean()),alpha_saved=float(z.alpha)))
 examples.append(dict(interval_id=interval,origin_date=str(origin.date()),training_last_allowed=str(last),training_months=t.month.tolist(),betas=b.tolist(),future_legs=legsout))
models=json.loads((R/'all_models.json').read_text(encoding='utf-8'))
ck('all90_frozen_TR_labels',len(models)==90)
for z in models:
 if z['status']=='ESTIMATED':
  ck('few_cluster_inference_fragile:'+z['model_id'],2<=z['funds']<=4 and z['nobs']>=30 and z['reference_df']==z['funds']-1 and z['weak_cluster_inference'] and z['significance_stars_suppressed'])
 else:
  ck('NA_reason_and_no_inference:'+z['model_id'],z['status']=='NOT_ESTIMABLE' and bool(z.get('reason')) and not z.get('coefficients') and not z.get('target_tests'))
ck('source_run_unchanged',sha(R/'manifest.json')==before)
report=dict(status='PASS' if all(x['status']=='PASS' for x in checks) else 'FAIL',audit_type='BOUNDED_EXTERNAL_PROVIDER_TR_REPLICATION',production_estimator_imports=False,run=R.name,manifest_sha256=before,snapshot_sha256=sha(snapshot),checks_pass=sum(x['status']=='PASS' for x in checks),checks_fail=sum(x['status']=='FAIL' for x in checks),checks=checks,origin_examples=examples,unverified=['Relies on research-approved immutable published-return snapshot; does not independently recertify the documentary extraction.','Bounded source/alpha audit: new provider-TR panel regression coefficient/covariance fits are not independently reestimated here; their small-cluster/NA flags are checked.'],completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(OUT/'qa.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({k:report[k] for k in ['status','run','checks_pass','checks_fail','manifest_sha256']},ensure_ascii=False))
if report['checks_fail']:print('FAILURES',[x for x in checks if x['status']=='FAIL']);raise SystemExit(1)
