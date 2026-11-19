"""Synthetic-only tests for the action-aware v3 handoff adapter."""
import json
import numpy as np
import pandas as pd
from run_h3_alpha_20261104 import prepare_returns


def tests():
    months=pd.period_range('2020-04','2020-07',freq='M')
    raw=pd.DataFrame(dict(fund_id='f',month=months.astype(str),
        start_valuation_date=(months-1).end_time.normalize(),end_valuation_date=months.end_time.normalize(),
        identity_valid_from=pd.Timestamp('2010-01-01'),identity_valid_to=pd.Timestamp('2026-11-04'),
        identity_verified=True,unit_value=101.,previous_unit_nav=100.,currency='RUB',
        total_return_verified=False,distribution_verified=False,split_verified=False,
        known_distribution_date_unresolved=[False,True,True,False],return_verified=[True,False,False,True],
        fund_return=[.01,np.nan,np.nan,.01]))
    cross=pd.DataFrame({'fund_id':['f']})
    cash=pd.DataFrame([dict(fund_id='f',record_date=pd.Timestamp('2020-05-15'),payment_date=pd.Timestamp('2020-06-05'))])
    snapshot=dict(known_cash_exclusion_scope='source_flagged_record_payment_months')
    d=prepare_returns(raw,cross,cash,snapshot)
    checks={
        'known_record_and_payment_months_excluded':list(d.diagnostic_eligible)==[True,False,False,True],
        'not_entire_fund_history_excluded':int(d.diagnostic_eligible.sum())==2,
        'unknown_payout_completeness_not_promoted_TR':not d.TR_eligible.any(),
        'source_values_retained':d.unit_value.equals(raw.unit_value),
        'event_month_reason_explicit':d.loc[d.known_unresolved_cash_event,'adapter_exclusion_reason'].eq('known_distribution_record_payment_month_unresolved').all(),
    }
    bad=raw.copy();bad.loc[0,'known_distribution_date_unresolved']=True
    try:prepare_returns(bad,cross,cash,snapshot)
    except AssertionError:rejected=True
    else:rejected=False
    checks['source_raw_event_month_disagreement_rejected']=rejected
    assert all(checks.values()),checks
    return dict(status='PASS',synthetic_only=True,checks_pass=len(checks),checks_fail=0,
                checks={name:bool(value) for name,value in checks.items()})


if __name__=='__main__':print(json.dumps(tests(),indent=2))
