"""Pure H3 kernels: fund-period aggregation and past-only factor adjustment.

No network, source mutation, security-level alpha regression or proxy fallback.
Definitions/thresholds are supplied by the separately frozen research spec.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd


def require(frame, columns, label):
    absent = sorted(set(columns) - set(frame))
    if absent:
        raise ValueError(f'{label}: missing columns {absent}')


def unique(frame, columns, label):
    require(frame, columns, label)
    if frame[columns].isna().any().any() or frame.duplicated(columns).any():
        raise ValueError(f'{label}: null or duplicated keys {columns}')


def truth(series):
    return series.eq(True).fillna(False).astype(bool)


def numeric(series):
    return pd.to_numeric(series, errors='coerce').astype(float)


def digest_lines(values):
    return hashlib.sha256('\n'.join(map(str, values)).encode('utf-8')).hexdigest()


def nts_from_positions(positions, calendar, *, activity, signal, minimum_companies,
                       minimum_trade_coverage, zero_tolerance=1e-12,
                       require_all_activity=False, signal_valid_column=None):
    """One row per actual fund interval, retaining empty and ineligible intervals.

    `activity` must already be the dimensionally comparable, frozen activity
    measure. Raw counts of heterogeneous securities must not be pooled as RUB.
    Unknown activity/news is never filled with zero. Coverage uses known activity
    over the full supplied corporate-position universe, not only signal-covered
    rows; the unknown-activity count is separately retained.
    """
    require(positions, ['row_id', 'fund_id', 'interval_id', 'company_id', activity, signal], 'NTS positions')
    require(calendar, ['fund_id', 'interval_id', 'opening_report_id', 'closing_report_id',
                       'start_date', 'end_date'], 'fund calendar')
    unique(positions, ['row_id'], 'NTS positions')
    unique(calendar, ['fund_id', 'interval_id'], 'fund calendar')
    p = positions.copy()
    p['_activity'] = numeric(p[activity])
    p['_signal'] = numeric(p[signal])
    valid_news = np.isfinite(p._signal) & p._signal.abs().le(1 + 1e-12)
    if signal_valid_column:
        require(p, [signal_valid_column], 'NTS eligibility')
        valid_news &= truth(p[signal_valid_column])
    known_activity = np.isfinite(p._activity)
    p['_news_eligible'] = (valid_news & known_activity & p.company_id.notna()
                           & p._signal.abs().gt(zero_tolerance) & p._activity.abs().gt(zero_tolerance))
    p['_known_activity'] = known_activity
    p['_mass'] = p._activity.abs().where(known_activity)
    p['_eligible_mass'] = p._activity.abs().where(p._news_eligible)
    p['_numerator'] = (p._activity * p._signal).where(p._news_eligible)
    p['_denominator'] = (p._activity.abs() * p._signal.abs()).where(p._news_eligible)
    p['_news_measured'] = valid_news
    output = []
    for (fund, interval), g in p.groupby(['fund_id', 'interval_id'], observed=True, sort=False):
        numerator = float(g._numerator.sum(min_count=1))
        denominator = float(g._denominator.sum(min_count=1))
        total_mass = float(g._mass.sum(min_count=1))
        covered_mass = float(g._eligible_mass.sum(min_count=1))
        if not np.isfinite(covered_mass) and g._news_eligible.sum() == 0:
            covered_mass = 0.0  # Empty-set count/mass, not an imputed security activity.
        coverage = covered_mass / total_mass if np.isfinite(total_mass) and total_mass > zero_tolerance else np.nan
        companies = int(g.loc[g._news_eligible, 'company_id'].nunique())
        unknown = int((~g._known_activity).sum())
        reasons = []
        if require_all_activity and unknown: reasons.append('incomplete_corporate_activity')
        if not np.isfinite(denominator) or denominator <= zero_tolerance: reasons.append('zero_or_missing_denominator')
        if companies < minimum_companies: reasons.append('insufficient_eligible_companies')
        if not np.isfinite(coverage) or coverage < minimum_trade_coverage: reasons.append('insufficient_measured_trade_coverage')
        score = numerator / denominator if not reasons else np.nan
        if np.isfinite(score) and abs(score) > 1 + 1e-10: raise AssertionError('NTS is outside [-1,1]')
        output.append(dict(fund_id=fund, interval_id=interval, NTS_Q=score,
            nts_numerator=numerator, nts_denominator=denominator,
            nts_gross_activity_known=total_mass, nts_covered_activity=covered_mass,
            nts_activity_coverage=coverage, nts_positions_total=len(g),
            nts_positions_activity_missing=unknown, nts_positions_signal_measured=int(g._news_measured.sum()),
            nts_positions_eligible=int(g._news_eligible.sum()), nts_companies_eligible=companies,
            nts_status='defined' if not reasons else ';'.join(reasons),
            position_sample_hash=digest_lines(sorted(g.row_id.astype(str)))))
    columns=['fund_id','interval_id','NTS_Q','nts_numerator','nts_denominator','nts_gross_activity_known',
             'nts_covered_activity','nts_activity_coverage','nts_positions_total','nts_positions_activity_missing',
             'nts_positions_signal_measured','nts_positions_eligible','nts_companies_eligible','nts_status','position_sample_hash']
    aggregate=pd.DataFrame(output, columns=columns)
    result=calendar.merge(aggregate,on=['fund_id','interval_id'],how='left',validate='one_to_one')
    result['nts_status']=result.nts_status.fillna('no_corporate_position_candidates')
    return add_exact_fund_lag(result)


def add_exact_fund_lag(frame):
    unique(frame,['fund_id','interval_id'],'NTS calendar')
    unique(frame,['fund_id','closing_report_id'],'NTS predecessor calendar')
    fields=['fund_id','interval_id','closing_report_id','end_date','NTS_Q','nts_status']
    previous=frame[fields].rename(columns={
        'interval_id':'previous_interval_id','closing_report_id':'previous_closing_report_id',
        'end_date':'previous_end_date','NTS_Q':'NTS_Q_lag1','nts_status':'nts_lag1_status'})
    d=frame.merge(previous,left_on=['fund_id','opening_report_id'],
        right_on=['fund_id','previous_closing_report_id'],how='left',validate='one_to_one')
    has_previous=d.previous_interval_id.notna()
    if not pd.to_datetime(d.loc[has_previous,'start_date']).eq(pd.to_datetime(d.loc[has_previous,'previous_end_date'])).all():
        raise AssertionError('NTS lag does not use the immediately preceding actual interval')
    d['nts_lag1_status']=d.nts_lag1_status.fillna('no_previous_actual_interval')
    return d


def paired_nts_q(contributions, calendar, *, minimum_companies=2, minimum_coverage=.5):
    """Frozen H3: CURRENT trades aligned with current and previous NEWS.

    Both scores use the identical inherited eligible set E. The denominator U
    includes every measurable corporate quantity-value change in the interval,
    including rows excluded from the inherited target/news sample. This is NOT
    a lag of a prior fund-level trading score.
    """
    require(contributions,['row_id','fund_id','interval_id','company_id','B_Q',
                           'inherited_target','net_sentiment','net_sentiment_lag1'], 'paired NTS')
    unique(contributions,['fund_id','interval_id','row_id'],'paired NTS contributions')
    unique(calendar,['fund_id','interval_id'],'paired NTS calendar')
    p=contributions.copy()
    for c in ['B_Q','net_sentiment','net_sentiment_lag1']:p[c]=numeric(p[c])
    valid=(truth(p.inherited_target)&np.isfinite(p.B_Q)&p.company_id.notna()
           &np.isfinite(p.net_sentiment)&np.isfinite(p.net_sentiment_lag1)
           &p.net_sentiment.abs().le(1+1e-12)&p.net_sentiment_lag1.abs().le(1+1e-12))
    if 'eligible_for_score' in p:
        valid &= truth(p.eligible_for_score)
    p['NTS_common_E']=valid
    p['current_contribution']=(p.B_Q*p.net_sentiment).where(valid)
    p['lagnews_contribution']=(p.B_Q*p.net_sentiment_lag1).where(valid)
    result=[]
    for (fund,interval),g in p.groupby(['fund_id','interval_id'],observed=True,sort=False):
        e=g.loc[g.NTS_common_E];gross_u=float(g.B_Q.abs().sum(min_count=1))
        gross_e=float(e.B_Q.abs().sum())
        epsilon=1e-12*max(1.,gross_e)
        row=dict(fund_id=fund,interval_id=interval,gross_B_Q_all_measurable_corporate=gross_u,
            gross_B_Q_E=gross_e,corporate_positions_total=len(g),corporate_B_Q_missing=int(g.B_Q.isna().sum()),
            common_E_positions=len(e),common_E_companies=int(e.company_id.nunique()),
            common_E_hash=digest_lines(sorted(e.row_id.astype(str))),
            net_purchase_balance=float(e.B_Q.sum()/gross_e) if gross_e>0 else np.nan)
        for label,column in [('current','net_sentiment'),('lagnews','net_sentiment_lag1')]:
            weights=e.B_Q.abs()*e[column].abs()
            numerator=float((e.B_Q*e[column]).sum())
            denominator=float(weights.sum())
            companies=int(e.loc[weights.gt(0),'company_id'].nunique())
            covered=float(e.loc[e[column].ne(0),'B_Q'].abs().sum())
            coverage=covered/gross_u if np.isfinite(gross_u) and gross_u>0 else np.nan
            reason=[]
            if denominator<=epsilon:reason.append('zero_signal_or_zero_rebalancing')
            if companies<minimum_companies:reason.append('insufficient_contributing_companies')
            if not np.isfinite(coverage) or coverage<minimum_coverage:reason.append('low_measurable_news_trade_coverage')
            score=numerator/denominator if not reason else np.nan
            if np.isfinite(score) and abs(score)>1+1e-10:raise AssertionError('Invalid paired NTS range')
            row.update({f'{label}_numerator':numerator,f'{label}_denominator':denominator,
                f'company_count_{label}':companies,f'news_trade_coverage_{label}':coverage,
                f'NTS_Q_{label}':score,f'nts_{label}_status':';'.join(reason) if reason else 'defined'})
        row['nts_pair_eligible']=np.isfinite(row['NTS_Q_current']) and np.isfinite(row['NTS_Q_lagnews'])
        result.append(row)
    aggregate=pd.DataFrame(result)
    if aggregate.empty:
        aggregate=pd.DataFrame(columns=['fund_id','interval_id','NTS_Q_current','NTS_Q_lagnews',
                                        'nts_pair_eligible','nts_current_status','nts_lagnews_status'])
    d=calendar.merge(aggregate,on=['fund_id','interval_id'],how='left',validate='one_to_one')
    d['nts_pair_eligible']=truth(d.nts_pair_eligible)
    for c in ['nts_current_status','nts_lagnews_status']:d[c]=d[c].fillna('no_corporate_position_candidates')
    return d,p


def monthly_alpha_inputs(fund_returns, factors, factor_columns):
    require(fund_returns,['fund_id','month','fund_return','return_verified'],'fund returns')
    require(factors,['month','risk_free','factors_verified']+list(factor_columns),'factors')
    unique(fund_returns,['fund_id','month'],'fund returns')
    unique(factors,['month'],'factors')
    d=fund_returns.merge(factors,on='month',how='left',validate='many_to_one',suffixes=('','_factor'))
    d['period']=pd.PeriodIndex(d.month,freq='M')
    if 'return_period_end' not in d:
        d['return_period_end']=d.period.dt.to_timestamp(how='end').dt.normalize()
    else: d['return_period_end']=pd.to_datetime(d.return_period_end)
    for c in ['start_valuation_date','end_valuation_date']:
        if c in d:d[c]=pd.to_datetime(d[c])
    numeric_fields=['fund_return','risk_free']+list(factor_columns)
    for field in numeric_fields: d[field]=numeric(d[field])
    d['fund_observation_eligible']=truth(d.return_verified)&np.isfinite(d.fund_return)&d.fund_return.ge(-1)
    if 'identity_verified' in d: d['fund_observation_eligible'] &= truth(d.identity_verified)
    d['factor_observation_eligible']=truth(d.factors_verified)&np.isfinite(d[['risk_free']+list(factor_columns)]).all(axis=1)
    d['eligible']=d.fund_observation_eligible & d.factor_observation_eligible
    return d


def rolling_future_alpha(origins, fund_returns, factors, *, factor_definitions,
                         lookback_months, minimum_history, minimum_recent, recent_months,
                         horizons=(1,2,3), condition_max=10000,
                         training_end_rule='latest_complete_month_at_or_before_origin',
                         aggregation='mean_monthly_abnormal', include_memberships=True):
    """Past-only rolling betas; training intercept is NOT subtracted forward.

    Forecast months are the full calendar months AFTER the origin calendar month.
    No endpoint/availability estimate is invented; caller must label its return
    basis and any publication-vintage limitations in the input manifest.
    """
    require(origins,['fund_id','interval_id','end_date'],'alpha origins')
    unique(origins,['fund_id','interval_id'],'alpha origins')
    if training_end_rule not in ['previous_calendar_month','latest_complete_month_at_or_before_origin']:
        raise ValueError('Unsupported frozen training-end rule')
    if aggregation!='mean_monthly_abnormal': raise ValueError('Unsupported alpha aggregation; freeze and implement explicitly')
    allowed={'minus_rf','already_excess_or_spread'}
    if not factor_definitions or set(factor_definitions.values())-allowed: raise ValueError('Invalid factor definitions')
    d=monthly_alpha_inputs(fund_returns,factors,factor_definitions)
    factors_for_month=factors.copy();factors_for_month['period']=pd.PeriodIndex(factors_for_month.month,freq='M')
    factors_for_month['factors_verified']=truth(factors_for_month.factors_verified)
    for c in ['risk_free']+list(factor_definitions):factors_for_month[c]=numeric(factors_for_month[c])
    factors_for_month=factors_for_month.set_index('period')
    groups={str(f):g.sort_values('period').set_index('period') for f,g in d.groupby('fund_id',observed=True)}
    output=[];members=[]
    for origin in origins.itertuples(index=False):
        fund=str(origin.fund_id);date=pd.Timestamp(origin.end_date).normalize();t=date.to_period('M')
        last=t-1 if training_end_rule=='previous_calendar_month' or t.end_time.normalize()>date else t
        first=last-lookback_months+1
        g=groups.get(fund)
        train=None;beta=None;state='eligible';condition=np.nan;recent=0;n_past=0
        training_id=hashlib.sha256(json.dumps([fund,str(origin.interval_id),str(date),lookback_months,
            minimum_history,minimum_recent,recent_months,list(factor_definitions.items()),training_end_rule],ensure_ascii=False).encode()).hexdigest()
        if g is None: state='no_verified_fund_return_series'
        else:
            source_dates_ok=g.return_period_end.le(date)
            if 'end_valuation_date' in g:source_dates_ok &= g.end_valuation_date.le(date)
            if 'start_valuation_date' in g:source_dates_ok &= g.start_valuation_date.le(date)
            train=g.loc[(g.index>=first)&(g.index<=last)&g.eligible&source_dates_ok].copy()
            n_past=len(train);recent=int((train.index>=last-recent_months+1).sum())
            if n_past<minimum_history or recent<minimum_recent: state='insufficient_past_history'
            else:
                xx=np.column_stack([(train[c]-train.risk_free).to_numpy(float) if mode=='minus_rf'
                    else train[c].to_numpy(float) for c,mode in factor_definitions.items()])
                sd=xx.std(axis=0,ddof=1)
                if not np.isfinite(sd).all() or (sd<=0).any(): state='factor_rank_failure'
                else:
                    standardized=np.column_stack([np.ones(n_past),(xx-xx.mean(axis=0))/sd])
                    condition=float(np.linalg.cond(standardized))
                    x=np.column_stack([np.ones(n_past),xx])
                    if np.linalg.matrix_rank(x)<x.shape[1] or condition>condition_max:
                        state='factor_rank_or_condition_failure'
                    else:
                        yy=(train.fund_return-train.risk_free).to_numpy(float)
                        beta=np.linalg.lstsq(x,yy,rcond=None)[0]
                        if include_memberships:
                            members.extend(dict(training_id=training_id,fund_id=fund,interval_id=origin.interval_id,
                                month=str(m),fund_excess=float(y),**{f'x_{c}':float(v) for c,v in zip(factor_definitions,row)})
                                for m,y,row in zip(train.index,yy,xx))
        identity=getattr(origin,'identity_origin_verified',True)
        if pd.isna(identity) or identity!=True:
            beta=None;state='unverified_dated_identity_at_origin'
        for h in horizons:
            future_months=pd.period_range(t+1,t+h,freq='M')
            row=dict(fund_id=fund,interval_id=origin.interval_id,origin_date=date,origin_month=str(t),horizon=int(h),
                alpha=np.nan,past_alpha=np.nan,n_past=n_past,n_past_recent=recent,
                minimum_history=minimum_history,lookback_months=lookback_months,
                training_first_allowed=str(first),training_last_allowed=str(last),
                training_last_observed=str(train.index.max()) if train is not None and len(train) else None,
                training_id=training_id,training_condition_number=condition,
                future_first_month=str(t+1),future_last_month=str(t+h),future_months_complete=0,
                fund_future_months_eligible=0,factor_future_months_eligible=0,status=state,
                ignored_origin_month_tail_days=int((t.end_time.normalize()-date).days),
                future_anchor_date=None,future_anchor_verified=False)
            for field in factor_definitions: row['beta_'+field]=np.nan
            if g is not None:
                existing=future_months.isin(g.index)
                forward=g.reindex(future_months)
                fund_ok=forward.fund_observation_eligible.eq(True)
                factor_ok=[]
                for month in future_months:
                    if month not in factors_for_month.index: factor_ok.append(False);continue
                    ff=factors_for_month.loc[month]
                    factor_ok.append(bool(ff.factors_verified==True and np.isfinite(
                        [ff.risk_free]+[ff[c] for c in factor_definitions]).all()))
                row['fund_future_months_eligible']=int(fund_ok.sum())
                row['factor_future_months_eligible']=int(sum(factor_ok))
                row['future_months_complete']=int((forward.eligible.eq(True)&existing).sum())
                if 'start_valuation_date' in forward:
                    anchor=forward.start_valuation_date.iloc[0]
                    row['future_anchor_date']=anchor
                    row['future_anchor_verified']=bool(pd.notna(anchor) and anchor>=date)
                else:
                    # Generic synthetic callers may omit dates; the actual runner
                    # is required to provide observed endpoints by its adapter.
                    row['future_anchor_verified']=True
            else: forward=None
            if beta is not None:
                row['past_alpha']=float(beta[0])
                for field,b in zip(factor_definitions,beta[1:]): row['beta_'+field]=float(b)
                if row['future_months_complete']!=h:
                    row['status']='incomplete_future_horizon'
                elif not row['future_anchor_verified']:
                    row['status']='future_anchor_precedes_origin_or_missing'
                else:
                    xf=np.column_stack([(forward[c]-forward.risk_free).to_numpy(float) if mode=='minus_rf'
                        else forward[c].to_numpy(float) for c,mode in factor_definitions.items()])
                    abnormal=(forward.fund_return-forward.risk_free).to_numpy(float)-xf@beta[1:]
                    row['alpha']=float(abnormal.mean())
                    row['status']='eligible'
            output.append(row)
    return pd.DataFrame(output),pd.DataFrame(members)


def join_fund_alpha(nts, alpha):
    unique(nts,['fund_id','interval_id'],'fund-period NTS')
    unique(alpha,['fund_id','interval_id','horizon'],'fund-period alpha')
    merged=alpha.merge(nts,on=['fund_id','interval_id'],how='left',validate='many_to_one')
    if len(merged)!=len(alpha): raise AssertionError('Fund alpha was multiplied by security rows')
    if not pd.to_datetime(merged.origin_date).eq(pd.to_datetime(merged.end_date)).all():
        raise AssertionError('Fund alpha joined to a different origin date')
    return merged
