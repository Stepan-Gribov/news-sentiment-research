"""Bounded exact-window and observed-price analysis; no data collection."""
import hashlib
import json
from pathlib import Path
import re

import duckdb
import numpy as np
import pandas as pd

from title_name_regressions_core import build_weight_panel,aggregate_news_windows,unique_keys,IntegrityError,NotEstimable
from title_name_regressions_crosswalk import maturity_interval_flags
from title_name_regressions_estimation import within_transform,fixed_effect_rank

BASE=Path('./analysis_outputs/title_name_regressions_20261026_w2_v2')
CATALOG='./local/research-data-service/research.duckdb'


def target_panel(inventory,intervals,facts):
    endpoint_ids=set(intervals.opening_report_id)|set(intervals.closing_report_id)
    inventory=inventory.loc[inventory.report_id.isin(endpoint_ids)].copy()
    empty=pd.DataFrame(dict(interval_id=pd.Series(dtype=str),security_id=pd.Series(dtype=str),
                             total_return=pd.Series(dtype=float),return_coverage_complete=pd.Series(dtype=bool)))
    universe=build_weight_panel(inventory,intervals,empty)
    flags=[]
    for boundary in ['opening_report_id','closing_report_id']:
        joined=intervals[['interval_id',boundary]].merge(inventory[['report_id','security_id','target_security_candidate',
            'residence_documented_by_disclosure']],left_on=boundary,right_on='report_id',how='inner',validate='one_to_many')
        flags.append(joined[['interval_id','security_id','target_security_candidate','residence_documented_by_disclosure']])
    flag=pd.concat(flags,ignore_index=True).groupby(['interval_id','security_id'],observed=True)[
        ['target_security_candidate','residence_documented_by_disclosure']].all().reset_index()
    universe=universe.merge(flag,on=['interval_id','security_id'],how='left',validate='one_to_one')
    eligible=universe.target_security_candidate.eq(True)&universe.residence_documented_by_disclosure.eq(True)&universe.company_id.notna()
    maturity=maturity_interval_flags(universe,facts)
    audit=dict(all_union_rows=len(universe),target_before_maturity=int(eligible.sum()),maturity_targets_removed=int((eligible&maturity).sum()))
    panel=universe.loc[eligible&~maturity].copy()
    panel['DW']=panel.weight_close-panel.weight_open
    panel['opening_weight']=panel.weight_open
    panel['opening_log_NAV']=np.log(panel.opening_NAV.where(panel.opening_NAV.gt(0)))
    panel['fund_security_id']=panel.fund_id.astype(str)+'|'+panel.security_id
    panel['asset_class']=panel.asset_type
    panel['log_interval_days']=np.log((panel.end_date-panel.start_date).dt.days)
    audit.update(target_after_maturity=len(panel),funds=int(panel.fund_id.nunique()),companies=int(panel.company_id.nunique()),
                 securities=int(panel.security_id.nunique()),calendar_months=int(panel.month.nunique()))
    return panel,universe,audit


def window_signal(panel,events):
    windows=panel[['company_id','start_date','end_date']].drop_duplicates().reset_index(drop=True)
    windows['interval_id']=['w_'+hashlib.blake2b(f'{a.isoformat()}\t{b.isoformat()}'.encode(),digest_size=12).hexdigest()
                             for a,b in zip(windows.start_date,windows.end_date)]
    windows['news_start']=windows.start_date;windows['news_end']=windows.end_date
    current,coverage=aggregate_news_windows(windows,events)
    lag=windows.copy();duration=lag.end_date-lag.start_date
    lag['news_start']=lag.start_date-duration;lag['news_end']=lag.start_date
    previous,_=aggregate_news_windows(lag,events)
    current['lag_Tone']=previous.Tone.to_numpy()
    current['lag_log_news_count']=np.log1p(previous.n_relevant.to_numpy())
    current['log_news_count']=np.log1p(current.n_relevant)
    current['log_interval_days']=np.log((current.end_date-current.start_date).dt.days)
    current['month']=current.end_date.dt.to_period('M').astype(str)
    for source in ['rbc','prime','kommersant']:
        counted,_=aggregate_news_windows(windows,events.loc[events.source.eq(source)])
        current['source_share_'+source]=counted.n_relevant.to_numpy()/current.n_relevant.replace(0,np.nan)
    names=['lag_Tone','log_news_count','source_share_rbc','source_share_prime','source_share_kommersant','log_interval_days']
    good=current[['Tone']+names].notna().all(axis=1)&np.isfinite(current[['Tone']+names]).all(axis=1)
    fit=current.loc[good].copy();n_complete=len(fit)
    while len(fit):
        keep=fit.groupby('company_id',observed=True).company_id.transform('size').ge(6)&fit.groupby('month',observed=True).company_id.transform('nunique').ge(10)
        if keep.all():break
        fit=fit.loc[keep].copy()
    current['S']=np.nan
    if not len(fit): return current,dict(status='NOT_ESTIMABLE',reason='Exact-window signal minima fail',coverage=coverage)
    transformed,iterations=within_transform(fit[['Tone']+names].to_numpy(float),[fit.company_id,fit.month])
    y,x=transformed[:,0],transformed[:,1:];kept=[];dropped=[]
    for j,name in enumerate(names):
        if np.max(np.abs(x[:,j]))<1e-10:dropped.append(name);continue
        trial=x[:,kept+[j]];trial=trial/np.maximum(np.linalg.norm(trial,axis=0),1e-30)
        if np.linalg.matrix_rank(trial,tol=1e-9)<len(kept)+1:dropped.append(name)
        else:kept.append(j)
    fe_rank,components=fixed_effect_rank(fit.company_id,fit.month)
    if len(fit)<=fe_rank+len(kept): raise NotEstimable('No exact-window signal residual degrees of freedom')
    residual=y-x[:,kept]@np.linalg.lstsq(x[:,kept],y,rcond=None)[0]
    sigma=float(residual.std(ddof=1))
    if not np.isfinite(sigma) or sigma<=1e-12: raise NotEstimable('Degenerate exact-window signal')
    current.loc[fit.index,'S']=residual/sigma
    return current,dict(status='ESTIMATED_WINDOW_SIGNAL',natural_unit='unique company/start/end, not fund-expanded',
        input_windows=len(current),complete_windows=n_complete,usable_windows=len(fit),companies=int(fit.company_id.nunique()),
        closing_months=int(fit.month.nunique()),regressors=[names[j] for j in kept],dropped_nuisance=dropped,
        residual_sigma=sigma,fe_rank=fe_rank,fe_components=components,within_iterations=iterations,
        current_window='(start,end]',lag_window='(start-(end-start),start]',min_company_windows=6,
        min_companies_per_closing_month=10,date_and_classification_coverage=coverage,
        quarter_and_longer_windows_included_as_actual_intervals_not_called_monthly=True)


def quote_pairs(universe,connection):
    requests=universe.loc[universe.security_id.str.startswith('isin:')&
        (universe.weight_open.gt(1e-12)|universe.weight_close.gt(1e-12)),
        ['security_id','start_date','end_date']].drop_duplicates().reset_index(drop=True)
    requests['requested_isin']=requests.security_id.str[5:]
    isins=requests[['requested_isin']].drop_duplicates();connection.register('requested_instruments',isins)
    quotes=connection.execute("SELECT q.REQUESTED_ISIN requested_isin,q.SECID secid,q.BOARDID boardid,try_cast(q.TRADEDATE AS DATE) trade_date,"
        "q.CLOSE price,q.VALUE trading_value,q.CURRENCYID currency,q.FACEVALUE facevalue,q.FACEUNIT faceunit,q.SOURCE_MARKET market,q.source_file "
        "FROM market.security_quotes q INNER JOIN requested_instruments r ON q.REQUESTED_ISIN=r.requested_isin "
        "WHERE try_cast(q.TRADEDATE AS DATE) BETWEEN DATE '2017-01-01' AND DATE '2026-05-29' "
        "AND q.CLOSE>0 AND q.SECID IS NOT NULL AND q.BOARDID IS NOT NULL").fetchdf()
    mapping=connection.execute("SELECT requested_isin,primary_boardid FROM refs.moex_mapping WHERE status='resolved'").fetchdf().drop_duplicates()
    primary=mapping.drop_duplicates('requested_isin').set_index('requested_isin').primary_boardid.to_dict()
    quotes['trade_date']=pd.to_datetime(quotes.trade_date)
    # Reject genuinely conflicting source observations rather than picking a price.
    key=['requested_isin','secid','boardid','trade_date']
    conflict=quotes.groupby(key,observed=True).price.transform('nunique').gt(1)
    n_conflict=int(conflict.sum());quotes=quotes.loc[~conflict].drop_duplicates(key)
    candidates=[];grouped={isin:g for isin,g in requests.groupby('requested_isin',observed=True)}
    for (isin,secid,board),g in quotes.groupby(['requested_isin','secid','boardid'],observed=True):
        group=grouped[isin];g=g.sort_values('trade_date');dates=g.trade_date.to_numpy(dtype='datetime64[ns]')
        start=group.start_date.to_numpy(dtype='datetime64[ns]');end=group.end_date.to_numpy(dtype='datetime64[ns]')
        left=np.searchsorted(dates,start,side='right')-1;right=np.searchsorted(dates,end,side='right')-1
        exists=(left>=0)&(right>=0)
        ix=np.flatnonzero(exists)
        if not len(ix):continue
        opening=g.iloc[left[ix]].reset_index(drop=True);closing=g.iloc[right[ix]].reset_index(drop=True)
        a=(start[ix]-opening.trade_date.to_numpy(dtype='datetime64[ns]'))/np.timedelta64(1,'D')
        b=(end[ix]-closing.trade_date.to_numpy(dtype='datetime64[ns]'))/np.timedelta64(1,'D')
        currency_ok=opening.currency.fillna('').str.upper().isin(['SUR','RUB','RUR'])&closing.currency.fillna('').str.upper().isin(['SUR','RUB','RUR'])
        bond=opening.market.eq('bonds')|closing.market.eq('bonds')
        face_ok=(opening.facevalue.gt(0)&closing.facevalue.gt(0)&opening.faceunit.fillna('').str.upper().isin(['SUR','RUB','RUR'])&
                 closing.faceunit.fillna('').str.upper().isin(['SUR','RUB','RUR']))
        valid=(a<=7)&(b<=7)&currency_ok.to_numpy()&(~bond|face_ok).to_numpy()
        if not valid.any():continue
        part=group.iloc[ix].copy().reset_index(drop=True)
        p0=opening.price.to_numpy()*np.where(bond,opening.facevalue.to_numpy()/100,1)
        p1=closing.price.to_numpy()*np.where(bond,closing.facevalue.to_numpy()/100,1)
        part['price_open']=p0;part['price_close']=p1;part['price_return']=p1/p0-1
        part['secid']=secid;part['boardid']=board;part['quote_open_date']=opening.trade_date.to_numpy();part['quote_close_date']=closing.trade_date.to_numpy()
        part['quote_open_age_days']=a;part['quote_close_age_days']=b
        part['primary_board_preferred']=board==primary.get(isin)
        part['opening_trading_value']=opening.trading_value.fillna(0).to_numpy()
        candidates.append(part.loc[valid].copy())
    columns=['security_id','start_date','end_date','price_open','price_close','price_return','secid','boardid',
             'quote_open_date','quote_close_date','quote_open_age_days','quote_close_age_days']
    if not candidates:return pd.DataFrame(columns=columns),dict(quote_rows=len(quotes),conflicting_quote_rows=n_conflict,requested_pairs=len(requests),accepted_pairs=0)
    all_candidates=pd.concat(candidates,ignore_index=True).sort_values(
        ['primary_board_preferred','opening_trading_value','secid','boardid'],ascending=[False,False,True,True],kind='stable')
    accepted=all_candidates.drop_duplicates(['security_id','start_date','end_date'])
    unique_keys(accepted,['security_id','start_date','end_date'],'observed quote pairs')
    return accepted[columns],dict(quote_rows=len(quotes),conflicting_quote_rows=n_conflict,requested_pairs=len(requests),
        accepted_pairs=len(accepted),instruments_with_pairs=int(accepted.security_id.nunique()),max_quote_age_days=7,
        quote_basis='Positive observed RUB CLOSE; bond CLOSE*historical_FACEVALUE/100; no distributions or split reconstruction',
        quote_source_files=sorted(quotes.source_file.dropna().unique().tolist()))


def cash_labels(connection):
    categories=connection.execute("select distinct parent_category,parent_subcategory from funds.category_totals where table_type='asset'").fetchdf().fillna('')
    leaf=connection.execute("select distinct registration_number_security,parent_category,parent_subcategory,item_name from funds.holdings "
        "where parent_category ilike '%денеж%' or parent_category ilike '%депозит%' or parent_category ilike '%вклад%'").fetchdf().fillna('')
    rows=[]
    for r in categories.to_dict('records'):
        a,b=r['parent_category'],r['parent_subcategory'];key='documented_bucket:'+hashlib.blake2b(f'{a}\t{b}'.encode(),digest_size=12).hexdigest()
        rows.append((key,a+' '+b))
    for r in leaf.to_dict('records'):
        code=str(r['registration_number_security']).strip().upper()
        if code:key='reg:'+code
        else:key='unidentified:'+hashlib.blake2b(f"{r['parent_category']}\t{r['parent_subcategory']}\t{r['item_name']}".encode(),digest_size=12).hexdigest()
        rows.append((key,r['parent_category']+' '+r['parent_subcategory']+' '+r['item_name']))
    allowed=set();ambiguous=set();labels={}
    for key,label in rows:
        labels.setdefault(key,[]).append(label)
        cash=bool(re.search(r'денежн.*средств|средств.*счет|депозит|вклад',label,re.I))
        rub=bool(re.search(r'рубл|RUB|RUR|валют[аеуы]?\s+Российск|национальн',label,re.I))
        foreign=bool(re.search(r'иностранн.*валют|USD|EUR|CNY|доллар|евро|юан',label,re.I))
        disclosed=bool(re.search(r'\d+(?:[.,]\d+)?\s*%',label))
        if cash and rub and not foreign and not disclosed:allowed.add(key)
        elif cash:ambiguous.add(key)
    allowed-=ambiguous
    return allowed,{key:' | '.join(sorted(set(value))) for key,value in labels.items()}


def price_adjustment(universe,pairs,allowed_cash):
    full=universe.merge(pairs,on=['security_id','start_date','end_date'],how='left',validate='many_to_one')
    full['price_return_source']=np.where(full.price_return.notna(),'OBSERVED_RUB_PRICE',
        np.where(full.security_id.isin(allowed_cash),'USER_IDENTIFIED_RUB_CASH_DEPOSIT_ZERO','UNCOVERED'))
    assumed=full.price_return.isna()&full.security_id.isin(allowed_cash)
    full.loc[assumed,'price_return']=0.0
    required=full.weight_open.gt(1e-12)
    full['covered_opening_weight']=np.where(~required|full.price_return.notna(),full.weight_open,0)
    full['uncovered_opening_weight']=np.where(required&full.price_return.isna(),full.weight_open,0)
    full['opening_growth']=np.where(required,full.weight_open*(1+full.price_return),0)
    group=full.groupby('interval_id',observed=True)
    coverage=group.agg(opening_weight=('weight_open','sum'),closing_weight=('weight_close','sum'),
        uncovered_weight=('uncovered_opening_weight','sum'),covered_weight=('covered_opening_weight','sum'),
        bh_denominator=('opening_growth','sum'),positions=('security_id','size'))
    coverage['coverage_fraction']=coverage.covered_weight/coverage.opening_weight
    coverage['eligible_full_price_AT']=coverage.uncovered_weight.le(1e-12)&coverage.opening_weight.sub(1).abs().le(1e-8)&coverage.closing_weight.sub(1).abs().le(1e-8)&coverage.bh_denominator.gt(0)
    full=full.merge(coverage[['bh_denominator','eligible_full_price_AT']],on='interval_id',how='left',validate='many_to_one')
    full['AT_PRICE']=(full.weight_close-full.opening_growth/full.bh_denominator).where(full.eligible_full_price_AT)
    return full,coverage.reset_index()
