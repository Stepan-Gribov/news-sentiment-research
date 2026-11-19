"""Past-only factor betas and complete future monthly alpha horizons."""
from __future__ import annotations

import numpy as np
import pandas as pd

from title_name_regressions_core import require_columns, unique_keys


def future_fund_alpha(origins,fund_returns,factors,*,window=60,min_past=36,min_last12=9,
                      horizons=(1,2,3),condition_max=10000):
    require_columns(origins,["fund_id","month"],"alpha origins")
    unique_keys(origins,["fund_id","month"],"alpha origins")
    require_columns(fund_returns,["fund_id","month","fund_return","return_verified"],"fund returns")
    require_columns(factors,["month","risk_free","equity_return","bond_return","factors_verified"],"alpha factors")
    unique_keys(fund_returns,["fund_id","month"],"fund returns")
    unique_keys(factors,["month"],"alpha factors")
    d=fund_returns.merge(factors,on="month",how="left",validate="many_to_one")
    d["period"]=pd.PeriodIndex(d.month,freq="M")
    d["eligible"]=(d.return_verified.eq(True)&d.factors_verified.eq(True)
                    &np.isfinite(d[["fund_return","risk_free","equity_return","bond_return"]]).all(axis=1))
    groups={fund:g.sort_values("period").set_index("period") for fund,g in d.groupby("fund_id",observed=True)}
    rows=[]
    for origin in origins.itertuples(index=False):
        t=pd.Period(origin.month,freq="M")
        g=groups.get(origin.fund_id)
        state="eligible"
        ntrain=nrecent=0
        coefficients=None
        condition=np.nan
        if g is None:
            state="missing_fund_return_crosswalk"
        else:
            train=g.loc[(g.index>=t-window+1)&(g.index<=t)&g.eligible]
            ntrain=len(train)
            nrecent=int((train.index>=t-11).sum())
            if ntrain<min_past or nrecent<min_last12:
                state="insufficient_past_history"
            else:
                x=np.column_stack([train.equity_return-train.risk_free,train.bond_return-train.risk_free])
                sd=x.std(axis=0,ddof=1)
                if not np.isfinite(sd).all() or (sd<=0).any():
                    state="factor_rank_failure"
                else:
                    standardized=np.column_stack([np.ones(ntrain),(x-x.mean(axis=0))/sd])
                    condition=float(np.linalg.cond(standardized))
                    design=np.column_stack([np.ones(ntrain),x])
                    if np.linalg.matrix_rank(design)<3 or condition>condition_max:
                        state="factor_rank_or_condition_failure"
                    else:
                        y=(train.fund_return-train.risk_free).to_numpy(float)
                        coefficients=np.linalg.lstsq(design,y,rcond=None)[0]
        for h in horizons:
            row=dict(fund_id=origin.fund_id,month=str(t),horizon=h,alpha=np.nan,
                     past_alpha=np.nan,beta_equity=np.nan,beta_bond=np.nan,n_past=ntrain,
                     n_past_last12=nrecent,condition_number=condition,status=state)
            if coefficients is not None:
                row.update(past_alpha=float(coefficients[0]),beta_equity=float(coefficients[1]),beta_bond=float(coefficients[2]))
                months=pd.period_range(t+1,t+h,freq="M")
                if not months.isin(g.index).all() or not g.reindex(months).eligible.eq(True).all():
                    row["status"]="incomplete_future_horizon"
                else:
                    future=g.reindex(months)
                    abnormal=(future.fund_return-future.risk_free
                              -coefficients[1]*(future.equity_return-future.risk_free)
                              -coefficients[2]*(future.bond_return-future.risk_free))
                    # The fitted historical intercept is a control, not subtracted.
                    row["alpha"]=float(abnormal.mean())
            rows.append(row)
    return pd.DataFrame(rows)
