"""Bounded executor QA for the separately approved four-fund TR release."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from h3_alpha_io_20261104 import ROOT,sha,file_record,save_json,utc

ready=json.loads((ROOT/'runtime/provider_TR_ready.json').read_text(encoding='utf-8'));root=Path(ready['output'])
seal=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
for record in list(seal['inputs'].values())+list(seal['files'].values()):file_record(record['path'],record['sha256'])
approval=json.loads(Path(seal['inputs']['documentary_approval']['path']).read_text(encoding='utf-8'))
snapshot=json.loads(Path(seal['inputs']['snapshot']['path']).read_text(encoding='utf-8'))
r=pd.read_parquet(root/'approved_monthly_provider_returns.parquet')
f=pd.read_parquet(seal['inputs']['factors']['path']).set_index('month')
checks=[]
def check(name,passed,**data):checks.append(dict(check=name,passed=bool(passed),**data))
check('exact_approved_return_rows',len(r)==sum(x['rows'] for x in approval['funds'].values())
      and np.array_equal(r.fund_return.to_numpy(),np.array([x['return_decimal'] for x in snapshot['rows']])))
check('exact_approved_funds',set(r.fund_id.astype(str))==set(approval['funds']))
check('no_NAV_substitution_or_double_adjustment',not np.array_equal(r.fund_return,r.raw_nav_ratio_return)
      and not r.additional_payout_or_split_adjustment_applied.any())
check('endpoint_staleness_identity_currency',r.start_age_trading_days.between(0,3).all()
      and r.end_age_trading_days.between(0,3).all() and r.identity_verified.all() and r.currency_verified_RUB.all())
check('printed_precision_preserved',r.printed_percentage_decimal_places.eq(2).all() and r.rounding_half_width_decimal.eq(.00005).all())
r=r.set_index(['fund_id','month'])
errors=[]
for label,minimum,credit in [('lookback60_min36',36,False),('lookback36_min24',24,False),('lookback60_min36_credit',36,True)]:
    a=pd.read_parquet(root/(label+'_alpha.parquet'));m=pd.read_parquet(root/(label+'_training.parquet'))
    finite=a.loc[a.alpha.notna()];maximum=0.
    for row in finite.itertuples(index=False):
        train=m.loc[m.training_id.eq(row.training_id)]
        terms=['equity_return','bond_return']+(['credit_factor'] if credit else [])
        x=np.column_stack([np.ones(len(train))]+[train['x_'+c].to_numpy(float) for c in terms])
        beta=np.linalg.solve(x.T@x,x.T@train.fund_excess.to_numpy(float))
        months=list(pd.period_range(row.future_first_month,row.future_last_month,freq='M').astype(str))
        future=np.array([r.loc[(row.fund_id,month),'fund_return'] for month in months])-f.loc[months].risk_free.to_numpy(float)
        for c,b in zip(terms,beta[1:]):future-=b*(f.loc[months,c].to_numpy(float)-(0 if c=='credit_factor' else f.loc[months].risk_free.to_numpy(float)))
        maximum=max(maximum,abs(float(future.mean())-row.alpha))
        if pd.PeriodIndex(train.month,freq='M').end_time.normalize().max()>pd.Timestamp(row.origin_date):errors.append('future_training')
        if row.n_past<minimum or row.n_past_recent<9:errors.append('history_gate')
    check('independent_alpha_'+label,maximum<1e-10,alpha_rows=len(finite),max_abs_error=maximum)
    check('future_anchor_'+label,(pd.to_datetime(finite.future_anchor_date)>=pd.to_datetime(finite.origin_date)).all())
check('all_past_training_gates',not errors)
models=pd.read_csv(root/'models.csv')
families=models.groupby('sensitivity').size().to_dict()
check('all90_fixed_cells_families',families=={'PRIMARY':24,'HISTORY36_MIN24':12,'BROAD_OBSERVED_NTS':12,
    'NONOVERLAPPING_PHASES':24,'CORPORATE_BOND_SPREAD':9,'CORPORATE_SAME_SAMPLE_TWO_FACTOR':9}
    and not models.model_id.duplicated().any())
estimated=models.loc[models.status.eq('ESTIMATED')]
check('all_estimated_models_flagged_fragile_no_stars',estimated.funds.between(2,4).all()
    and estimated.weak_cluster_inference.eq(True).all() and estimated.significance_stars_suppressed.eq(True).all())
for group in ['pooled','equity','bond','mixed']:
    for h in [1,2,3]:
        pair=models.loc[models.sensitivity.eq('PRIMARY')&models.group.eq(group)&models.horizon.eq(h)]
        check(f'primary_current_distributed_same_rows_{group}_h{h}',len(pair)==2 and pair.sample_hash.nunique()==1)
for p in (root/'regression').rglob('fit.json'):
    info=json.loads(p.read_text(encoding='utf-8'))
    if info['status']!='ESTIMATED':continue
    check('native_within_CR1_'+info['model_id'],info['QA']['status']=='PASS')
    coef=pd.read_csv(p.parent/'coefficients_main_tGminus1.csv')
    check('reference_df_'+info['model_id'],coef.reference_df.eq(info['funds']-1).all())
    if info['funds']==2 and info['distributed']:
        check('two_cluster_joint_wald_not_faked_'+info['model_id'],info['target_tests'][-1]['status']!='defined')
result=dict(status='PASS' if all(x['passed'] for x in checks) else 'FAIL',scope='Executor QA; independent DB/research review separate',
    checked_utc=utc(),input_root=str(root),run_manifest_sha256=sha(root/'manifest.json'),
    checks_pass=sum(x['passed'] for x in checks),checks_fail=sum(not x['passed'] for x in checks),checks=checks)
destination=ROOT/'qa'/(root.name+'_'+sha(__file__)[:12]);destination.mkdir(exist_ok=True,parents=True)
save_json(destination/'qa.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
if result['checks_fail']:raise SystemExit(2)
