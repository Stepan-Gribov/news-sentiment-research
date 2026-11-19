"""Synthetic timing/absence/no-long-history tests, no research coefficients."""
import json
import numpy as np
import pandas as pd
from h3_index_excess_core_20261104 import index_excess_outcomes


def run():
    origins=pd.DataFrame([dict(fund_id='f',interval_id='i',end_date=pd.Timestamp('2025-01-31'),
        identity_origin_verified=True,fund_focus='equity',row_id='origin')])
    months=pd.period_range('2024-12','2025-04',freq='M')
    funds=pd.DataFrame([dict(fund_id='f',month=str(m),fund_return=r,diagnostic_eligible=True,identity_verified=True,
        start_valuation_date=(m-1).end_time.normalize(),end_valuation_date=m.end_time.normalize(),
        return_period_end=m.end_time.normalize()) for m,r in zip(months,[.01,.015,.03,.04,.05])])
    benchmark=pd.DataFrame([dict(benchmark_id=b,month=str(m),benchmark_return=.01 if b=='IMOEX' else .02,
        benchmark_verified=True,start_valuation_date=(m-1).end_time.normalize(),
        end_valuation_date=m.end_time.normalize(),return_period_end=m.end_time.normalize())
        for b in ['IMOEX','RGBITR'] for m in months])
    kwargs=dict(focus_benchmarks={'equity':'IMOEX','mixed':'IMOEX','bond':'RGBITR'},
        past_month_rule='latest_eligible_complete_calendar_month_at_or_before_origin',require_benchmark_future_anchor=False)
    results=[]
    def test(name,ok):
        if not ok:raise AssertionError(name)
        results.append(name)
    a,legs=index_excess_outcomes(origins,funds,benchmark,**kwargs)
    test('only_one_past_return_needed_not36or60',a.outcome_known.all() and a.past_control_known.all())
    test('future_horizons_arithmetic_mean_not_cumulative',np.allclose(a.index_excess_return,[.02,.025,.03]))
    test('monthend_past_control_exact_month',a.past_reference_month.eq('2025-01').all() and np.allclose(a.past_index_excess_return,.005))
    mid=origins.assign(end_date=pd.Timestamp('2025-01-15'))
    am,_=index_excess_outcomes(mid,funds,benchmark,**kwargs)
    test('midmonth_origin_uses_prior_complete_month',am.past_reference_month.eq('2024-12').all() and np.allclose(am.past_index_excess_return,0))
    missing=funds.loc[funds.month.ne('2025-01')]
    an,_=index_excess_outcomes(origins,missing,benchmark,**kwargs)
    test('missing_latest_month_uses_older_observed_pair_with_age',an.past_reference_month.eq('2024-12').all() and an.control_gap_months.eq(1).all() and np.allclose(an.past_index_excess_return,0))
    missing_future=funds.loc[funds.month.ne('2025-03')]
    ax,_=index_excess_outcomes(origins,missing_future,benchmark,**kwargs)
    test('incomplete_future_horizon_not_partial_mean',ax.outcome_known.tolist()==[True,False,False])
    bad=funds.copy();bad.loc[bad.month.eq('2025-02'),'diagnostic_eligible']=False
    ab,_=index_excess_outcomes(origins,bad,benchmark,**kwargs)
    test('finite_but_unverified_fund_return_remains_unknown',ab.index_excess_return.isna().all())
    early=funds.copy();early.loc[early.month.eq('2025-02'),'start_valuation_date']=pd.Timestamp('2025-01-30')
    ae,_=index_excess_outcomes(origins,early,benchmark,**kwargs)
    test('actual_future_anchor_not_before_origin',ae.index_excess_return.isna().all())
    late=funds.copy();late.loc[late.month.eq('2025-01'),'end_valuation_date']=pd.Timestamp('2025-02-01')
    al,_=index_excess_outcomes(origins,late,benchmark,**kwargs)
    test('past_source_endpoint_after_origin_selects_earlier_verified_pair',al.past_reference_month.eq('2024-12').all())
    for focus,expected in [('mixed',.02),('bond',.01)]:
        ac,_=index_excess_outcomes(origins.assign(fund_focus=focus),funds,benchmark,**kwargs)
        test('origin_focus_benchmark_'+focus,np.isclose(ac.index_excess_return.iloc[0],expected))
    ai,_=index_excess_outcomes(origins.assign(identity_origin_verified=False),funds,benchmark,**kwargs)
    test('unknown_origin_identity_not_repaired',ai.index_excess_return.isna().all() and ai.past_index_excess_return.isna().all())
    no_past=funds.loc[funds.month.ge('2025-02')]
    az,_=index_excess_outcomes(origins,no_past,benchmark,**kwargs)
    test('no_eligible_past_pair_stays_NA',az.past_index_excess_return.isna().all() and az.outcome_known.all())
    published=funds.copy();published['publication_timestamp']=pd.NaT
    published.loc[published.month.eq('2025-01'),'publication_timestamp']=pd.Timestamp('2025-02-01')
    ap,_=index_excess_outcomes(origins,published,benchmark,**kwargs)
    test('known_past_publication_after_origin_not_used',ap.past_reference_month.eq('2024-12').all())
    early_index=benchmark.copy();early_index.loc[early_index.month.eq('2025-02'),'start_valuation_date']=pd.Timestamp('2025-01-30')
    ak,_=index_excess_outcomes(origins,funds,early_index,**kwargs)
    test('earlier_index_last_trading_day_does_not_add_future_anchor_gate',ak.outcome_known.all() and not ak.future_benchmark_anchor_verified.any())
    from run_h3_index_excess_20261104 import normalize_benchmarks
    duplicate_aliases=benchmark.assign(series_id=benchmark.benchmark_id,return_verified=benchmark.benchmark_verified,
        index_start_date=benchmark.start_valuation_date,index_end_date=benchmark.end_valuation_date)
    aliases={'series_id':'benchmark_id','return_verified':'benchmark_verified','index_start_date':'start_valuation_date','index_end_date':'end_valuation_date'}
    normalized=normalize_benchmarks(duplicate_aliases,aliases)
    test('existing_identical_benchmark_aliases_not_duplicated',normalized.columns.is_unique and len(normalized)==len(benchmark))
    conflict=duplicate_aliases.copy();conflict.loc[0,'benchmark_id']='WRONG'
    rejected=False
    try:normalize_benchmarks(conflict,aliases)
    except ValueError:rejected=True
    test('conflicting_benchmark_aliases_rejected',rejected)
    return dict(status='PASS',passed=len(results),failed=0,checks=results)


if __name__=='__main__':print(json.dumps(run(),indent=2))
