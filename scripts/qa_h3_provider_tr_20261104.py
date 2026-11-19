"""Second-path validation of the bounded official provider-return extension."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from h3_alpha_io_20261104 import ROOT,sha,file_record,save_json,utc

ready=json.loads((ROOT/'runtime/provider_TR_ready.json').read_text(encoding='utf-8'));root=Path(ready['output'])
seal=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
for record in list(seal['inputs'].values())+list(seal['files'].values()):file_record(record['path'],record['sha256'])
snapshot=json.loads(Path(seal['inputs']['snapshot']['path']).read_text(encoding='utf-8'))
r=pd.read_parquet(root/'approved_monthly_provider_returns.parquet').set_index('month')
f=pd.read_parquet(seal['inputs']['factors']['path']).set_index('month')
checks=[]
def check(name,passed,**data):checks.append(dict(check=name,passed=bool(passed),**data))
check('exactly_approved55_provider_returns',len(r)==55 and np.array_equal(r.fund_return.to_numpy(),np.array([x['return_decimal'] for x in snapshot['rows']])))
check('published_returns_not_replaced_by_NAV_ratios',not np.array_equal(r.fund_return.to_numpy(),r.raw_nav_ratio_return.to_numpy()))
check('no_second_income_or_split_adjustment',not r.additional_payout_or_split_adjustment_applied.any())
check('both_valuation_endpoint_staleness_gates',r.start_age_trading_days.between(0,3).all() and r.end_age_trading_days.between(0,3).all())
check('all55_identity_and_currency_verified',r.identity_verified.all() and r.currency_verified_RUB.all() and r.return_verified.all())
check('rounding_precision_preserved',r.printed_percentage_decimal_places.eq(2).all() and r.rounding_half_width_decimal.eq(.00005).all())
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
        future=r.loc[months].fund_return.to_numpy(float)-f.loc[months].risk_free.to_numpy(float)
        for c,b in zip(terms,beta[1:]):
            future-=b*(f.loc[months,c].to_numpy(float)-(0 if c=='credit_factor' else f.loc[months].risk_free.to_numpy(float)))
        maximum=max(maximum,abs(float(future.mean())-row.alpha))
        if pd.PeriodIndex(train.month,freq='M').end_time.normalize().max()>pd.Timestamp(row.origin_date):errors.append('future_training')
        if row.n_past<minimum or row.n_past_recent<9:errors.append('history_gate')
    check('independent_alpha_reconstruction_'+label,maximum<1e-10,alpha_rows=len(finite),max_abs_error=maximum)
    check('future_anchor_'+label,(pd.to_datetime(finite.future_anchor_date)>=pd.to_datetime(finite.origin_date)).all())
check('all_training_cutoffs_and_minimum_history',not errors)
models=pd.read_csv(root/'models.csv')
check('all90_provider_sensitivity_cells_accounted',len(models)==90 and not models.model_id.duplicated().any())
check('no_invalid_one_fund_clustered_inference',models.status.eq('NOT_ESTIMABLE').all() and models.funds.le(1).all())
result=dict(status='PASS' if all(x['passed'] for x in checks) else 'FAIL',scope='Executor second-path QA; independent research review remains separate',
    checked_utc=utc(),input_root=str(root),run_manifest_sha256=sha(root/'manifest.json'),
    checks_pass=sum(x['passed'] for x in checks),checks_fail=sum(not x['passed'] for x in checks),checks=checks)
destination=ROOT/'qa'/(root.name+'_'+sha(__file__)[:12]);destination.mkdir(exist_ok=True,parents=True)
save_json(destination/'qa.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
if result['checks_fail']:raise SystemExit(2)
