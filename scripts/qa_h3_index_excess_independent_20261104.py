"""Prepared compact independent QA for24 index-relative H3 cells.

No data collection, score rebuilding or production imports. The final adapter
awaits the executor's sealed run. The optional self-check uses synthetic data.
The auditor's own prior LSDV/CR1 code is reused only for linear algebra.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np,pandas as pd
from qa_h3_relaxed_news_independent_20261104 import independent_full_dummy_cr1,expected_model_sample

SPEC_SHA='b0d982e5bf57866a2ea6d0fca5e57186a54076d4359f5ebe42ec752224dbd74b'
SCORES_SHA='b91618a88d33e3ed0a705ac5e30c7bd34542a4266a8097b99791c7d09a8b2f12'
BENCHMARK_SHA='3b553cc5b9cb292472330895a4ffd88f3e011ae9def66b9a3d939f70bad5f356'
BENCHMARK_MANIFEST_SHA='a7ef0535a9ecebd5fe3c0d6cbfb9c6745529301cdfb34bce12cfe59289fa009e'
ASSIGNMENT={'equity':'IMOEX','mixed':'IMOEX','bond':'RGBITR'}
Y='mean_monthly_index_excess_return';P='past_index_excess_return'


def origin_replica(origin,fund_returns,benchmarks):
    """All h=1/2/3 plus one latest available past matched monthly return.

    Benchmark is chosen once from origin-dated focus. Missing months are not
    filled, and no old alpha/risk-free/factor/history variable is referenced.
    """
    date=pd.Timestamp(origin['end_date']).normalize();month=date.to_period('M')
    benchmark=ASSIGNMENT.get(origin['fund_focus'])
    result=dict(fund_id=str(origin['fund_id']),interval_id=origin['interval_id'],origin_date=str(date.date()),benchmark_id=benchmark,outcomes={},past_control=np.nan,past_month=None,control_age_months=None)
    if benchmark is None:return dict(result,status='unknown_origin_focus_no_benchmark')
    f=fund_returns.loc[fund_returns.fund_id.astype(str).eq(str(origin['fund_id']))].copy()
    b=benchmarks.loc[benchmarks.series_id.eq(benchmark)].copy()
    if f.month.duplicated().any() or b.month.duplicated().any():raise ValueError('Duplicate fund/index month')
    fund_columns=['fund_id','month','fund_return','diagnostic_eligible','start_valuation_date','end_valuation_date','return_period_end']
    index_columns=['series_id','month','benchmark_return','return_verified','index_start_date','index_end_date']
    if 'publication_timestamp' in f:fund_columns.append('publication_timestamp')
    if 'publication_timestamp' in b:index_columns.append('publication_timestamp')
    f=f[fund_columns].rename(columns={x:x+'_fund' for x in ['start_valuation_date','end_valuation_date','return_period_end','publication_timestamp']})
    b=b[index_columns].rename(columns={'return_verified':'benchmark_verified','publication_timestamp':'publication_timestamp_index'})
    d=f.merge(b,on='month',how='outer',suffixes=('_fund','_index'),validate='one_to_one')
    # Source columns are named explicitly so index flags cannot stand in for
    # fund quality or inherit unrelated-factor completeness accidentally.
    for name in ['start_valuation_date_fund','end_valuation_date_fund','return_period_end_fund','index_start_date','index_end_date']:
        d[name]=pd.to_datetime(d[name])
    d['period']=pd.PeriodIndex(d.month,freq='M')
    known=(d.diagnostic_eligible.eq(True)&d.benchmark_verified.eq(True)&np.isfinite(d[['fund_return','benchmark_return']]).all(axis=1)&d.fund_return.ge(-1)&d.benchmark_return.gt(-1))
    d['matched_ok']=known;d['relative_return']=d.fund_return-d.benchmark_return
    # Latest completed calendar month, with every observed source date <=origin.
    past=known & d.period.map(lambda m:m.end_time.normalize()<=date)
    for name in ['return_period_end_fund','start_valuation_date_fund','end_valuation_date_fund','index_start_date','index_end_date']:
        past &= d[name].notna()&d[name].le(date)
    # Only genuine publication timestamps count; archive download timestamps
    # are never invented as historical public-availability observations.
    for name in ['publication_timestamp_fund','publication_timestamp_index']:
        if name in d:
            pub=pd.to_datetime(d[name]);past &= pub.isna()|pub.le(date)
    candidates=d.loc[past].sort_values('period')
    if len(candidates):
        last=candidates.iloc[-1];result.update(past_control=float(last.relative_return),past_month=last.month,control_age_months=month.ordinal-last.period.ordinal,past_fund_return=float(last.fund_return),past_benchmark_return=float(last.benchmark_return),past_source_dates={k:str(last[k]) for k in ['return_period_end_fund','start_valuation_date_fund','end_valuation_date_fund','index_start_date','index_end_date']})
    bymonth=d.set_index('period')
    for h in [1,2,3]:
        periods=pd.period_range(month+1,month+h,freq='M');legs=bymonth.reindex(periods)
        complete=legs.matched_ok.eq(True).all();anchor=legs.start_valuation_date_fund.iloc[0]
        anchor_ok=pd.notna(anchor) and pd.Timestamp(anchor)>=date
        value=float(legs.relative_return.mean()) if complete and anchor_ok else np.nan
        result['outcomes'][h]=dict(value=value,status='eligible' if np.isfinite(value) else 'incomplete_future_months' if not complete else 'future_fund_anchor_precedes_origin_or_missing',months=periods.astype(str).tolist(),fund_returns=legs.fund_return.tolist(),index_returns=legs.benchmark_return.tolist(),relative_returns=legs.relative_return.tolist(),fund_anchor=None if pd.isna(anchor) else str(anchor),fund_anchor_verified=bool(anchor_ok))
    result['status']='past_control_available' if np.isfinite(result['past_control']) else 'no_eligible_past_control'
    return result


def expected_sample(panel,*,distributed,group):
    # This is an algebraic column alias of the NEW one-month index control.
    # It neither reads historical past_alpha nor imposes its history gate.
    d=panel.copy();d['past_alpha']=d[P]
    return expected_model_sample(d,outcome=Y,current='NTS_Q_current',lag='NTS_Q_lagnews',distributed=distributed,universe_column='universe_eligible',group=group)


def regression_replica(sample,*,distributed):
    d=sample.copy();d['past_alpha']=d[P]
    terms=['NTS_Q_current']+(['NTS_Q_lagnews'] if distributed else [])+['past_alpha']
    fit=independent_full_dummy_cr1(d,outcome=Y,terms=terms)
    if 'terms' in fit:fit['terms']=[P if x=='past_alpha' else x for x in fit['terms']]
    if 'absorbed_columns' in fit:fit['absorbed_columns']=[P if x=='past_alpha' else x for x in fit['absorbed_columns']]
    return fit


def planned_cells():
    return [(g,h,v) for g in ['pooled','equity','bond','mixed'] for h in [1,2,3] for v in ['CURRENT_ONLY_OWN_AVAILABLE','CURRENT_PLUS_LAGNEWS']]


def self_check():
    months=pd.period_range('2026-01','2026-08',freq='M')
    end=lambda m:pd.bdate_range(m.start_time,m.end_time)[-1]
    f=pd.DataFrame([dict(fund_id='f',month=str(m),fund_return=.04,diagnostic_eligible=True,start_valuation_date=end(m-1),end_valuation_date=end(m),return_period_end=m.end_time.normalize()) for m in months])
    records=[]
    for series,ret in [('IMOEX',.01),('RGBITR',.02)]:
        for m in months:records.append(dict(series_id=series,month=str(m),benchmark_return=ret,return_verified=True,index_start_date=end(m-1),index_end_date=end(m),start_valuation_date=end(m-1),end_valuation_date=end(m),return_period_end=m.end_time.normalize()))
    b=pd.DataFrame(records)
    origin=dict(fund_id='f',interval_id='i',end_date='2026-05-29',fund_focus='equity')
    x=origin_replica(origin,f,b)
    assert x['past_month']=='2026-04'
    assert np.isclose(x['outcomes'][3]['value'],.03)
    assert origin_replica(dict(origin,fund_focus='bond'),f,b)['benchmark_id']=='RGBITR'
    assert np.isclose(origin_replica(dict(origin,fund_focus='bond'),f,b)['outcomes'][1]['value'],.02)
    assert origin_replica(dict(origin,fund_focus='mixed'),f,b)['benchmark_id']=='IMOEX'
    assert origin_replica(dict(origin,fund_focus='unknown'),f,b)['benchmark_id'] is None
    assert origin_replica(dict(origin,end_date='2026-05-31'),f,b)['past_month']=='2026-05'
    assert np.isnan(origin_replica(dict(origin,end_date='2026-05-31'),f,b)['outcomes'][1]['value'])
    gap=origin_replica(origin,f.loc[~f.month.eq('2026-07')],b)
    assert np.isfinite(gap['outcomes'][1]['value']) and np.isnan(gap['outcomes'][2]['value'])
    gap_index=origin_replica(origin,f,b.loc[~(b.series_id.eq('IMOEX')&b.month.eq('2026-07'))])
    assert np.isfinite(gap_index['outcomes'][1]['value']) and np.isnan(gap_index['outcomes'][2]['value'])
    old=origin_replica(origin,f.loc[~f.month.isin(['2026-04','2026-05'])],b)
    assert old['past_month']=='2026-03' and old['control_age_months']==2
    short=f.loc[f.month.isin(['2026-04','2026-06','2026-07','2026-08'])].copy()
    short['alpha']=np.nan;short['past_alpha']=np.nan;short['factors_verified']=False;short['n_past']=0
    assert origin_replica(origin,short,b)['past_month']=='2026-04' and np.isfinite(origin_replica(origin,short,b)['outcomes'][3]['value'])
    publication=f.copy();publication['publication_timestamp']=pd.NaT
    publication.loc[publication.month.eq('2026-04'),'publication_timestamp']=pd.Timestamp('2026-05-30')
    assert origin_replica(origin,publication,b)['past_month']=='2026-03'
    panel=pd.DataFrame(dict(fund_id=['a','a','b','b'],interval_id=['1','2','3','4'],end_date=pd.date_range('2020-01-01',periods=4),universe_eligible=True,fund_focus='equity',NTS_Q_current=[0.,.5,0.,-.5],NTS_Q_lagnews=np.nan,mean_monthly_index_excess_return=.03,past_index_excess_return=.01))
    own,_=expected_sample(panel,distributed=False,group='pooled');both,_=expected_sample(panel,distributed=True,group='pooled')
    assert len(own)==4 and both.empty
    assert len(planned_cells())==24
    import ast
    tree=ast.parse(Path(__file__).read_text(encoding='utf-8'));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='self_check')
    return dict(status='PASS',synthetic_contract_checks=sum(isinstance(n,ast.Assert) for n in ast.walk(fn)),production_models_run=0,actual24cell_QA_started=False,spec_sha256=SPEC_SHA)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-check',action='store_true');args=p.parse_args()
    if not args.self_check:p.error('Final adapter awaits sealed24-cell output; preparation only supports --self-check')
    print(json.dumps(self_check()))
