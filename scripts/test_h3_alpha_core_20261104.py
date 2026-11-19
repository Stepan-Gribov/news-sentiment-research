"""Synthetic, local-only acceptance tests; never reads research holdings/history."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from h3_alpha_core_20261104 import (nts_from_positions,rolling_future_alpha,
                                   join_fund_alpha,paired_nts_q)


def tests():
    checks=[]
    def check(name, value):
        checks.append(dict(check=name,passed=bool(value)))
        if not value:raise AssertionError(name)
    rng=np.random.default_rng(20261104)
    months=pd.period_range('2010-01','2022-12',freq='M')
    rf=np.full(len(months),.004)
    equity=rng.normal(.01,.04,len(months));bond=rng.normal(.006,.015,len(months))
    factors=pd.DataFrame(dict(month=months.astype(str),risk_free=rf,equity_return=equity,
                              bond_return=bond,factors_verified=True))
    actual_alpha=.003
    returns=pd.DataFrame(dict(fund_id='f',month=months.astype(str),return_verified=True,identity_verified=True,
        fund_return=rf+actual_alpha+1.1*(equity-rf)+.4*(bond-rf)))
    origins=pd.DataFrame([dict(fund_id='f',interval_id='i',end_date=pd.Timestamp('2020-05-15'))])
    kwargs=dict(factor_definitions={'equity_return':'minus_rf','bond_return':'minus_rf'},
                lookback_months=60,minimum_history=36,minimum_recent=9,recent_months=12)
    alpha,members=rolling_future_alpha(origins,returns,factors,**kwargs)
    check('three_horizons',len(alpha)==3 and list(alpha.horizon)==[1,2,3])
    check('past_only_training_end_midmonth',set(alpha.training_last_observed)=={'2020-04'})
    check('full_sixty_calendar_month_window',alpha.n_past.eq(60).all())
    check('correct_equity_beta',np.allclose(alpha.beta_equity_return,1.1,atol=1e-11))
    check('correct_bond_beta',np.allclose(alpha.beta_bond_return,.4,atol=1e-11))
    check('training_intercept_not_subtracted_forward',np.allclose(alpha.alpha,actual_alpha,atol=1e-11))
    check('past_intercept_saved',np.allclose(alpha.past_alpha,actual_alpha,atol=1e-11))
    check('training_membership_once_not_per_horizon',len(members)==60 and members.training_id.nunique()==1)
    changed=returns.copy();changed.loc[changed.month.ge('2020-05'),'fund_return']+=.17
    alpha_changed,_=rolling_future_alpha(origins,changed,factors,**kwargs)
    check('future_mutation_cannot_change_betas',np.array_equal(alpha.beta_equity_return,alpha_changed.beta_equity_return))
    check('future_mutation_cannot_change_training_intercept',np.array_equal(alpha.past_alpha,alpha_changed.past_alpha))
    check('future_mutation_changes_only_outcome',np.allclose(alpha_changed.alpha-alpha.alpha,.17,atol=1e-11))
    outside=returns.copy();outside.loc[outside.month.lt('2015-05'),'fund_return']+=.6
    outside_alpha,_=rolling_future_alpha(origins,outside,factors,**kwargs)
    check('prewindow_mutation_has_no_effect',np.array_equal(alpha.alpha,outside_alpha.alpha))
    missing=returns.loc[returns.month.ne('2020-08')]
    incomplete,_=rolling_future_alpha(origins,missing,factors,**kwargs)
    check('incomplete_h3_not_partial_average',pd.isna(incomplete.loc[incomplete.horizon.eq(3),'alpha']).all())
    check('shorter_complete_horizons_kept',incomplete.loc[incomplete.horizon.le(2),'status'].eq('eligible').all())
    factor_gap=factors.loc[factors.month.ne('2020-06')]
    missing_factors,_=rolling_future_alpha(origins,returns,factor_gap,**kwargs)
    check('missing_future_factors_never_zero',missing_factors.alpha.isna().all())
    short=returns.loc[returns.month.ge('2020-02')]
    insufficient,_=rolling_future_alpha(origins,short,factors,**kwargs)
    check('three_observations_not_claimed_alpha',insufficient.alpha.isna().all() and insufficient.n_past.eq(3).all())
    recent_gap=returns.loc[~returns.month.isin(['2020-01','2020-02','2020-03','2020-04'])]
    too_few_recent,_=rolling_future_alpha(origins,recent_gap,factors,**kwargs)
    check('recent_history_gate',too_few_recent.alpha.isna().all() and too_few_recent.n_past_recent.eq(8).all())
    unverified=returns.copy();unverified['identity_verified']=False
    invalid_identity,_=rolling_future_alpha(origins,unverified,factors,**kwargs)
    check('dated_identity_gate',invalid_identity.n_past.eq(0).all() and invalid_identity.alpha.isna().all())
    rank_fail=factors.copy();rank_fail['bond_return']=rank_fail.equity_return
    bad_rank,_=rolling_future_alpha(origins,returns,rank_fail,**kwargs)
    check('rank_failure_explicit',bad_rank.alpha.isna().all() and bad_rank.status.str.contains('rank').all())
    calendar=pd.DataFrame([dict(fund_id='f',interval_id=f'i{k}',opening_report_id=f'r{k}',
        closing_report_id=f'r{k+1}',start_date=pd.Timestamp('2020-01-31')+pd.offsets.MonthEnd(k),
        end_date=pd.Timestamp('2020-02-29')+pd.offsets.MonthEnd(k)) for k in range(4)])
    positions=pd.DataFrame([
        dict(row_id='a',fund_id='f',interval_id='i0',company_id='c1',activity=.1,net=.5),
        dict(row_id='b',fund_id='f',interval_id='i0',company_id='c2',activity=.2,net=-.25),
        dict(row_id='c',fund_id='f',interval_id='i0',company_id='c3',activity=.3,net=0.),
        dict(row_id='d',fund_id='f',interval_id='i1',company_id='c1',activity=.2,net=np.nan),
        dict(row_id='e',fund_id='f',interval_id='i2',company_id='c1',activity=.1,net=.5),
        dict(row_id='g',fund_id='f',interval_id='i2',company_id='c2',activity=.1,net=-.5),
        dict(row_id='h',fund_id='f',interval_id='i2',company_id='c3',activity=np.nan,net=.5)])
    nts=nts_from_positions(positions,calendar,activity='activity',signal='net',minimum_companies=2,
        minimum_trade_coverage=.5,require_all_activity=True)
    by=nts.set_index('interval_id')
    check('one_fund_period_row_before_alpha_join',len(nts)==len(calendar))
    check('true_balanced_NTS_zero_preserved',by.loc['i0','NTS_Q']==0)
    check('zero_signal_mass_excluded_but_coverage_denominator_kept',np.isclose(by.loc['i0','nts_activity_coverage'],.5))
    check('missing_news_is_NA_not_neutral',pd.isna(by.loc['i1','NTS_Q']))
    check('missing_trade_activity_not_zero',pd.isna(by.loc['i2','NTS_Q']) and by.loc['i2','nts_positions_activity_missing']==1)
    check('empty_interval_retained',by.loc['i3','nts_status']=='no_corporate_position_candidates')
    check('lag_preserves_real_zero',by.loc['i1','NTS_Q_lag1']==0)
    check('lag_does_not_skip_missing_predecessor',pd.isna(by.loc['i2','NTS_Q_lag1']))
    join_nts=nts.loc[nts.interval_id.eq('i0')].copy();join_nts['end_date']=origins.end_date.iloc[0]
    join_alpha=alpha.copy();join_alpha['interval_id']='i0'
    joined=join_fund_alpha(join_nts,join_alpha)
    check('alpha_not_multiplied_by_security_count',len(joined)==3)
    duplicate=pd.concat([positions,positions.iloc[[0]]],ignore_index=True)
    try:
        nts_from_positions(duplicate,calendar,activity='activity',signal='net',minimum_companies=2,
                           minimum_trade_coverage=.5)
    except ValueError:duplicate_rejected=True
    else:duplicate_rejected=False
    check('duplicate_position_rows_rejected',duplicate_rejected)
    pair_positions=pd.DataFrame([
        dict(row_id='p1',fund_id='f',interval_id='i0',company_id='c1',B_Q=100.,inherited_target=True,
             net_sentiment=.5,net_sentiment_lag1=-.5,eligible_for_score=True),
        dict(row_id='p2',fund_id='f',interval_id='i0',company_id='c2',B_Q=-100.,inherited_target=True,
             net_sentiment=-.5,net_sentiment_lag1=.5,eligible_for_score=True),
        dict(row_id='p3',fund_id='f',interval_id='i0',company_id='c3',B_Q=100.,inherited_target=False,
             net_sentiment=np.nan,net_sentiment_lag1=np.nan,eligible_for_score=True),
        dict(row_id='p4',fund_id='f',interval_id='i0',company_id='c4',B_Q=50.,inherited_target=True,
             net_sentiment=0.,net_sentiment_lag1=0.,eligible_for_score=True),
        dict(row_id='p5',fund_id='f',interval_id='i0',company_id='c5',B_Q=np.nan,inherited_target=True,
             net_sentiment=.8,net_sentiment_lag1=.8,eligible_for_score=True)])
    pair,contrib=paired_nts_q(pair_positions,calendar)
    p0=pair.set_index('interval_id').loc['i0']
    check('paired_current_score',p0.NTS_Q_current==1.)
    check('paired_lag_is_current_trade_previous_news_not_fund_lag',p0.NTS_Q_lagnews==-1.)
    check('common_E_includes_observed_neutral_news',p0.common_E_positions==3)
    check('U_includes_positions_without_news',p0.gross_B_Q_all_measurable_corporate==350.)
    check('unknown_quantity_not_zero',p0.corporate_B_Q_missing==1)
    check('coverage_unweighted_by_abs_net',np.isclose(p0.news_trade_coverage_current,200/350))
    scaled=pair_positions.copy();scaled['B_Q']*=1000
    rebased,_=paired_nts_q(scaled,calendar)
    check('RUB_rebasing_invariant',np.allclose(pair.NTS_Q_current,rebased.NTS_Q_current,equal_nan=True))
    missing_one=pair_positions.copy();missing_one.loc[missing_one.row_id.eq('p2'),'net_sentiment_lag1']=np.nan
    insufficient_pair,_=paired_nts_q(missing_one,calendar)
    check('both_scores_same_E_when_lag_missing',insufficient_pair.NTS_Q_current.isna().all())
    broad,_=paired_nts_q(missing_one,calendar,minimum_companies=1,minimum_coverage=0)
    check('predeclared_broad_branch_not_silently_primary',broad.loc[broad.interval_id.eq('i0'),'NTS_Q_current'].eq(1).all())
    affected=pair_positions.copy();affected.loc[affected.row_id.eq('p2'),'eligible_for_score']=False
    excluded,_=paired_nts_q(affected,calendar)
    check('known_event_excludes_score_not_U',excluded.loc[excluded.interval_id.eq('i0'),'gross_B_Q_all_measurable_corporate'].eq(350).all()
          and excluded.NTS_Q_current.isna().all())
    dated=returns.copy();dated['start_valuation_date']=(pd.PeriodIndex(dated.month,freq='M')-1).end_time.normalize()
    dated['end_valuation_date']=pd.PeriodIndex(dated.month,freq='M').end_time.normalize()
    may29=origins.copy();may29['end_date']=pd.Timestamp('2020-05-29')
    may_result,_=rolling_future_alpha(may29,dated,factors,**kwargs)
    check('May29_never_uses_May31_training_return',may_result.training_last_observed.eq('2020-04').all())
    dated.loc[dated.month.eq('2020-06'),'start_valuation_date']=pd.Timestamp('2020-05-28')
    preorigin,_=rolling_future_alpha(may29,dated,factors,**kwargs)
    check('stale_future_anchor_before_origin_rejected',preorigin.alpha.isna().all()
          and preorigin.status.eq('future_anchor_precedes_origin_or_missing').all())
    unidentified=may29.copy();unidentified['identity_origin_verified']=False
    identity_origin,_=rolling_future_alpha(unidentified,returns,factors,**kwargs)
    check('dated_identity_required_at_forecast_origin',identity_origin.alpha.isna().all()
          and identity_origin.status.eq('unverified_dated_identity_at_origin').all())
    from build_h3_nts_q_20261104 import select_origins,quantity_contributions
    multiple=calendar.copy();multiple.loc[0,'end_date']=pd.Timestamp('2020-02-15')
    later=multiple.iloc[[0]].copy();later['interval_id']='i0latest';later['closing_report_id']='r-latest';later['end_date']=pd.Timestamp('2020-02-29')
    selected,decisions=select_origins(pd.concat([multiple,later],ignore_index=True))
    check('latest_origin_chosen_before_score_gate_no_earlier_fallback','i0latest' in set(selected.interval_id) and 'i0' not in set(selected.interval_id))
    conflict=later.copy();conflict['interval_id']='i0conflict';conflict['closing_report_id']='r-conflict'
    selected_conflict,_=select_origins(pd.concat([multiple,later,conflict],ignore_index=True))
    check('same_date_conflict_excludes_month_without_fallback',not selected_conflict.origin_month.eq('2020-02').any())
    def position(**changes):
        row=dict(security_id='s',isin='RU0007288411',Q0=10.,Q1=1000.,P0=10000.,P1=100.,V0=100000.,V1=100000.,
            age0=0.,age1=0.,quote0='2024-03-29',quote1='2024-04-30',start_date='2024-03-29',end_date='2024-04-30',
            observed0=True,observed1=True,opening_full_portfolio=True,closing_full_portfolio=True,currency0='RUB',currency1='RUB',
            identity_conflict=False,class_conflict=False)
        row.update(changes);return row
    actions={'events':[dict(isin='RU0007288411',event_date='2024-04-04',post_quote_from='2024-04-08',
        closing_to_opening_quantity_ratio=.01,source_url='fixture')]}
    empty_events=pd.DataFrame(columns=['security_id','scheduled_maturity_date'])
    legs=quantity_contributions(pd.DataFrame([position(),position(security_id='entry',Q0=np.nan,V0=np.nan,
        P0=np.nan,quote0=None,observed0=False),position(security_id='exit',Q1=np.nan,V1=np.nan,
        P1=np.nan,quote1=None,observed1=False)]),actions,empty_events)
    check('verified_split_does_not_create_trade',legs.B_Q.iloc[0]==0.)
    check('entry_uses_closing_value_without_preentry_quote',legs.B_Q.iloc[1]==100000. and legs.confirmed_entry.iloc[1])
    check('exit_uses_opening_price_without_exit_quote',legs.B_Q.iloc[2]==-100000. and legs.confirmed_exit.iloc[2])
    unknown=quantity_contributions(pd.DataFrame([position(observed0=False,Q0=np.nan,P0=np.nan,V0=np.nan,
        opening_full_portfolio=False)]),actions,empty_events)
    check('partial_opening_absence_not_zero',unknown.B_Q.isna().all())
    return dict(status='PASS',synthetic_only=True,passed=len(checks),failed=0,checks=checks,
                alpha_max_error=float(abs(alpha.alpha-actual_alpha).max()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report');a=p.parse_args();report=tests()
    if a.report:Path(a.report).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
