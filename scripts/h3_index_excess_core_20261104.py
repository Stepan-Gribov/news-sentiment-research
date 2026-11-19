"""Future fund-minus-index returns. No alpha, beta estimation or history cut."""
from __future__ import annotations
import numpy as np
import pandas as pd


def yes(x):return bool(pd.notna(x) and x==True)
def finite(x):return bool(pd.notna(x) and np.isfinite(float(x)))
def date(x):return pd.Timestamp(x).normalize() if pd.notna(x) else pd.NaT


def index_excess_outcomes(origins,fund_returns,benchmarks,*,focus_benchmarks,
                          past_month_rule,require_benchmark_future_anchor,
                          horizons=(1,2,3)):
    """Latest eligible past month and complete full-calendar-month future legs.

    Missing future months are never replaced. The one past control may use an
    older observed matched month, with its age saved and no age cutoff.
    Benchmarks are fixed by focus at the origin,
    including pooled rows, across the past control and every future leg.
    """
    if past_month_rule!='latest_eligible_complete_calendar_month_at_or_before_origin':
        raise ValueError('Unsupported unfrozen past-control month rule')
    if fund_returns.duplicated(['fund_id','month']).any():raise ValueError('Duplicate fund month')
    if benchmarks.duplicated(['benchmark_id','month']).any():raise ValueError('Duplicate index month')
    if origins.duplicated(['fund_id','interval_id']).any():raise ValueError('Duplicate selected origin')
    funds={(str(f),str(m)):r for f,m,r in zip(fund_returns.fund_id,fund_returns.month,fund_returns.to_dict('records'))}
    indices={(str(b),str(m)):r for b,m,r in zip(benchmarks.benchmark_id,benchmarks.month,benchmarks.to_dict('records'))}
    candidates={}
    for (fund,month),r in funds.items():
        if yes(r.get('diagnostic_eligible')) and yes(r.get('identity_verified')) and finite(r.get('fund_return')):
            candidates.setdefault(fund,[]).append(month)
    for fund in candidates:candidates[fund].sort(reverse=True)
    outputs=[];legs=[]
    for origin in origins.itertuples(index=False):
        fund=str(origin.fund_id);d1=date(origin.end_date);t=d1.to_period('M')
        benchmark=focus_benchmarks.get(origin.fund_focus)
        past=t if t.end_time.normalize()<=d1 else t-1
        identity=yes(origin.identity_origin_verified)
        def leg(period,kind,save_leg=True):
            f=funds.get((fund,str(period)),{});b=indices.get((str(benchmark),str(period)),{})
            f_return=f.get('fund_return',np.nan);b_return=b.get('benchmark_return',np.nan)
            f_start=date(f.get('start_valuation_date'));f_end=date(f.get('end_valuation_date'))
            b_start=date(b.get('start_valuation_date'));b_end=date(b.get('end_valuation_date'))
            f_period_end=date(f.get('return_period_end'));b_period_end=date(b.get('return_period_end'))
            fund_known=(yes(f.get('diagnostic_eligible')) and yes(f.get('identity_verified'))
                and finite(f_return) and float(f_return)>=-1 and pd.notna(f_start) and pd.notna(f_end)
                and f_start<=f_end and pd.notna(f_period_end) and f_period_end.to_period('M')==period)
            index_known=(yes(b.get('benchmark_verified')) and finite(b_return) and float(b_return)>-1
                and pd.notna(b_start) and pd.notna(b_end) and b_start<=b_end
                and pd.notna(b_period_end) and b_period_end.to_period('M')==period)
            reasons=[]
            if not f:reasons.append('MISSING_FUND_MONTH')
            elif not fund_known:reasons.append('FUND_MONTH_UNVERIFIED')
            if not b:reasons.append('MISSING_BENCHMARK_MONTH')
            elif not index_known:reasons.append('BENCHMARK_MONTH_UNVERIFIED')
            if benchmark is None:reasons.append('UNKNOWN_ORIGIN_FOCUS')
            if kind=='PAST':
                if not (pd.notna(f_period_end) and f_period_end<=d1 and pd.notna(f_start) and f_start<=d1 and pd.notna(f_end) and f_end<=d1):
                    reasons.append('PAST_FUND_ENDPOINT_AFTER_ORIGIN_OR_MISSING')
                if not (pd.notna(b_period_end) and b_period_end<=d1 and pd.notna(b_start) and b_start<=d1 and pd.notna(b_end) and b_end<=d1):
                    reasons.append('PAST_BENCHMARK_ENDPOINT_AFTER_ORIGIN_OR_MISSING')
                for source,label in [(f,'FUND'),(b,'BENCHMARK')]:
                    publication=source.get('publication_timestamp')
                    if pd.notna(publication):
                        publication=pd.Timestamp(publication)
                        cutoff=d1.tz_localize('Europe/Moscow').tz_convert('UTC') if publication.tzinfo else d1
                        if publication>cutoff:reasons.append('PAST_'+label+'_KNOWN_PUBLICATION_AFTER_ORIGIN')
            row=dict(fund_id=fund,interval_id=origin.interval_id,origin_date=d1,origin_row_id=origin.row_id,
                kind=kind,month=str(period),benchmark_id=benchmark,fund_focus_at_origin=origin.fund_focus,
                fund_return=f_return,benchmark_return=b_return,fund_month_exists=bool(f),benchmark_month_exists=bool(b),
                fund_month_known=fund_known,benchmark_month_known=index_known,
                fund_start_valuation_date=f_start,fund_end_valuation_date=f_end,
                benchmark_start_valuation_date=b_start,benchmark_end_valuation_date=b_end,
                fund_return_period_end=f_period_end,benchmark_return_period_end=b_period_end,
                component_verified=not reasons,index_excess_component=float(f_return)-float(b_return) if not reasons else np.nan,
                reason=';'.join(reasons) if reasons else 'KNOWN',
                fund_adapter_exclusion_reason=f.get('adapter_exclusion_reason'),fund_return_basis=f.get('return_basis'),
                fund_source_path=f.get('source_path'),fund_raw_sha256=f.get('raw_sha256'),
                fund_previous_raw_sha256=f.get('previous_raw_sha256'),fund_source_json_pointer=f.get('source_json_pointer'),
                fund_previous_point_index=f.get('previous_point_index'),fund_identity_verified=f.get('identity_verified'),
                fund_total_return_verified=f.get('total_return_verified'),fund_distribution_status=f.get('distribution_status'),
                fund_split_status=f.get('split_status'),benchmark_source_id=b.get('source_id'),
                fund_publication_timestamp=f.get('publication_timestamp'),benchmark_publication_timestamp=b.get('publication_timestamp'),
                fund_publication_vintage_unknown=pd.isna(f.get('publication_timestamp')),
                benchmark_publication_vintage_unknown=pd.isna(b.get('publication_timestamp')),
                fund_index_endpoint_dates_equal=bool(pd.notna(f_start) and pd.notna(b_start) and f_start==b_start and f_end==b_end))
            if save_leg:legs.append(row)
            return row
        backward=None;selected_past=None;checked=0
        for month in candidates.get(fund,[]):
            if month>str(past):continue
            candidate=leg(pd.Period(month,freq='M'),'PAST',save_leg=False);checked+=1
            if candidate['component_verified']:
                backward=candidate;selected_past=pd.Period(month,freq='M');break
        if backward is None:
            backward=leg(past,'PAST',save_leg=False)
            backward['component_verified']=False;backward['index_excess_component']=np.nan
            backward['reason']='NA_NO_ELIGIBLE_PAST_MATCHED_MONTH;'+backward['reason']
        legs.append(backward)
        future=[leg(t+k,'FUTURE') for k in range(1,max(horizons)+1)]
        f_anchor=future[0]['fund_start_valuation_date'];b_anchor=future[0]['benchmark_start_valuation_date']
        fund_anchor_ok=bool(pd.notna(f_anchor) and f_anchor>=d1)
        index_anchor_ok=bool(pd.notna(b_anchor) and b_anchor>=d1)
        for h in horizons:
            forward=future[:h];complete=sum(r['component_verified'] for r in forward)
            reasons=[]
            if not identity:reasons.append('UNVERIFIED_DATED_IDENTITY_AT_ORIGIN')
            if benchmark is None:reasons.append('UNKNOWN_ORIGIN_FOCUS')
            if complete!=h:reasons.append('INCOMPLETE_FUTURE_HORIZON')
            if not fund_anchor_ok:reasons.append('FUND_FUTURE_ANCHOR_BEFORE_ORIGIN_OR_MISSING')
            if require_benchmark_future_anchor and not index_anchor_ok:reasons.append('INDEX_FUTURE_ANCHOR_BEFORE_ORIGIN_OR_MISSING')
            outcome=float(np.mean([r['index_excess_component'] for r in forward])) if not reasons else np.nan
            outputs.append(dict(fund_id=fund,interval_id=origin.interval_id,origin_date=d1,horizon=h,
                benchmark_id=benchmark,index_outcome_id='H3INDEX:'+fund+':'+str(origin.interval_id)+':h'+str(h),
                index_excess_return=outcome,mean_monthly_index_excess_return=outcome,
                past_index_excess_return=backward['index_excess_component'] if identity else np.nan,
                past_reference_month=str(selected_past) if selected_past is not None else None,
                past_latest_allowed_month=str(past),past_candidates_checked=checked,
                control_age_months=t.ordinal-selected_past.ordinal if selected_past is not None else np.nan,
                control_gap_months=past.ordinal-selected_past.ordinal if selected_past is not None else np.nan,
                control_uses_older_available_month=bool(selected_past is not None and selected_past<past),
                past_status=backward['reason'] if identity else 'UNVERIFIED_DATED_IDENTITY_AT_ORIGIN',
                future_first_month=str(t+1),future_last_month=str(t+h),future_months_complete=complete,
                fund_future_months_eligible=sum(r['fund_month_known'] for r in forward),
                benchmark_future_months_eligible=sum(r['benchmark_month_known'] for r in forward),
                future_fund_anchor_date=f_anchor,future_benchmark_anchor_date=b_anchor,
                future_fund_anchor_verified=fund_anchor_ok,future_benchmark_anchor_verified=index_anchor_ok,
                ignored_origin_month_tail_days=int((t.end_time.normalize()-d1).days),
                outcome_status=';'.join(reasons) if reasons else 'KNOWN_INDEX_RELATIVE_RETURN',
                past_control_known=finite(backward['index_excess_component']) and identity,
                outcome_known=not reasons))
    return pd.DataFrame(outputs),pd.DataFrame(legs)
