"""Independent raw-to-normalized/SQL endpoint QA; no fund models or DB mutations."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd

ROOT=Path('./analysis_outputs/h3_alpha_20261104/factors')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def js(p,d):
    Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n',encoding='utf-8')


def main():
    m=json.loads((ROOT/'collection_manifest.json').read_text(encoding='utf-8'))
    checks=[]
    def check(name,passed,detail=None):
        checks.append(dict(name=name,passed=bool(passed),detail=detail))
        if not passed:
            print('FAIL',name,detail,flush=True)
    def close(a,b):
        return np.allclose(np.asarray(a,dtype=float),np.asarray(b,dtype=float),atol=2e-13,rtol=2e-12,equal_nan=True)
    for rel,f in m['files'].items():
        p=Path(f['path']);check('file_SHA:'+rel,p.exists() and p.stat().st_size==f['bytes'] and sha(p)==f['sha256'])
    for f in m['reused_sources']:
        check('preserved_source_SHA:'+f['path'],sha(f['path'])==f['sha256'])
    http=[]
    for log in [ROOT/'raw_requests.jsonl',ROOT/'raw_requests_additional.jsonl']:
        if log.exists():
            http.extend(json.loads(x) for x in log.read_text(encoding='utf-8').splitlines())
    for item in http:
        check('HTTP_response_SHA:'+item['path'],sha(item['path'])==item['sha256'])
    check('all_HTTP_TLS_verified',all(x['tls_verify'] is True for x in http))
    check('official_HTTPS_sources_only',all(x['url'].startswith(('https://iss.moex.com/','https://www.moex.com/',
        'https://fs.moex.com/','https://www.cbr.ru/')) for x in http))
    all_daily=[]
    for code,cov in m['series'].items():
        if code=='RUONIA':
            continue
        md=json.loads((ROOT/f'raw/moex/{code}/metadata.json').read_text(encoding='utf-8'))
        desc={r[0]:r[2] for r in md['description']['data']}
        check(code+':official_TR_RUB_identity',desc['SECID']==code and desc['CALCMODE'] in ['TR','TRN','TRB'] and desc['CURRENCYID']=='RUB')
        raw=[];cursor=[]
        for p in sorted((ROOT/f'raw/moex/{code}').glob('history_*.json')):
            j=json.loads(p.read_text(encoding='utf-8'))
            cols=j['history']['columns'];raw.extend(dict(zip(cols,r)) for r in j['history']['data'])
            cursor.append(dict(zip(j['history.cursor']['columns'],j['history.cursor']['data'][0])))
        check(code+':pagination_total',len(raw)==cov['rows']==cov['raw_total'] and all(x['TOTAL']==len(raw) for x in cursor))
        check(code+':page_starts_contiguous',[x['INDEX'] for x in cursor]==list(range(0,len(raw),100)))
        r=pd.DataFrame(raw).sort_values('TRADEDATE')
        d=pd.read_parquet(ROOT/f'normalized/{code}_daily.parquet').sort_values('date')
        check(code+':raw_dates_and_closes_exact',r.TRADEDATE.tolist()==d.date.dt.strftime('%Y-%m-%d').tolist() and close(r.CLOSE,d.index_value))
        check(code+':unique_positive_sorted',not d.date.duplicated().any() and d.date.is_monotonic_increasing and d.index_value.gt(0).all() and np.isfinite(d.index_value).all())
        check(code+':raw_board_session_identity',r.BOARDID.eq(cov['board']).all() and r.SECID.eq(code).all() and r.TRADINGSESSION.astype(str).eq('3').all())
        check(code+':daily_returns',close(d.daily_return,d.index_value/d.index_value.shift()-1))
        check(code+':asof_cap',d.date.max()<=pd.Timestamp(m['as_of']))
        all_daily.append(d)
    daily=pd.concat(all_daily,ignore_index=True)
    monthly=pd.read_parquet(ROOT/'normalized/moex_monthly_endpoints.parquet').sort_values(['series_id','month'])
    c=duckdb.connect(':memory:',config={'threads':1})
    c.register('daily',daily);c.register('monthly',monthly)
    # SQL group maxima + a dated prior-calendar-month join are independent of the collector's Python loop.
    expected=c.execute('''WITH e AS (
      SELECT series_id,strftime(date,'%Y-%m') AS "month",max(date) end_date,
             arg_max(index_value,date) end_value,count(*) obs
      FROM daily GROUP BY 1,2)
      SELECT e.series_id,e.month,e.end_date,e.end_value,e.obs,p.end_date start_date,p.end_value start_value,
             CASE WHEN last_day(e.end_date)<=CAST(? AS DATE) THEN e.end_value/p.end_value-1 END ret
      FROM e LEFT JOIN e p ON e.series_id=p.series_id
       AND strptime(p.month,'%Y-%m')=strptime(e.month,'%Y-%m')-INTERVAL '1 month'
      ORDER BY 1,2''',[m['as_of']]).df()
    z=expected.merge(monthly,on=['series_id','month'],validate='1:1')
    check('SQL_monthly_all_observed_rows_present',len(z)==len(expected))
    check('SQL_monthly_return_arithmetic',close(z.ret,z.return_m))
    check('SQL_monthly_end_dates',(z.end_date==z.index_end_date).all())
    check('SQL_monthly_end_values',close(z.end_value,z.index_end_value))
    check('SQL_monthly_start_dates',((z.start_date==z.index_start_date)|(z.start_date.isna()&z.index_start_date.isna())).all())
    check('SQL_monthly_start_values',close(z.start_value,z.index_start_value))
    check('SQL_monthly_observation_counts',(z.obs==z.n_observations).all())
    check('monthly_unique',not monthly.duplicated(['series_id','month']).any())
    check('partial_months_not_returns',monthly.loc[~monthly.month_complete,'return_m'].isna().all())
    check('no_missing_month_bridging',monthly.loc[monthly.status.ne('OK'),'return_m'].isna().all())
    check('legacy_not_forward_filled',monthly.loc[(monthly.series_id=='RUCBITR')&(monthly.month>'2023-05'),'return_m'].isna().all())
    check('new_corporate_not_backfilled',monthly.loc[(monthly.series_id=='RUCBTRNS')&(monthly.month<'2019-01'),'return_m'].isna().all())
    rf=pd.read_parquet(ROOT/'normalized/ruonia_index_daily.parquet').sort_values('date')
    raw_old=Path(m['reused_sources'][0]['path'])
    rr=[]
    for path in [raw_old,ROOT/'raw/cbr/RuoniaSVXML_extension.xml']:
        for e in ET.fromstring(path.read_bytes()).findall('.//ra'):
            rr.append((e.findtext('DT')[:10],float(e.findtext('RUONIA_Index'))))
    lookup={}
    for date,value in rr:
        check('RUONIA_raw_duplicate_consistent:'+date,date not in lookup or abs(lookup[date]-value)<2e-14)
        lookup[date]=value
    check('RUONIA_entire_raw_to_normalized',rf.date.dt.strftime('%Y-%m-%d').tolist()==sorted(lookup)
        and close(rf.ruonia_index,[lookup[x] for x in sorted(lookup)]))
    check('RUONIA_no_calendar_gaps',rf.date.diff().dropna().dt.days.eq(1).all())
    check('RUONIA_positive_finite_monotonic',rf.ruonia_index.gt(0).all() and np.isfinite(rf.ruonia_index).all() and rf.ruonia_index.diff().dropna().ge(0).all())
    rfm=pd.read_parquet(ROOT/'normalized/ruonia_monthly_both_conventions.parquet')
    c.register('rf',rf);c.register('rfm',rfm)
    rcheck=c.execute('''SELECT a.*,b.ruonia_index v0,e.ruonia_index v1,
       CASE WHEN month_end<=CAST(? AS DATE) THEN e.ruonia_index/b.ruonia_index-1 END expected_rf
       FROM rfm a LEFT JOIN rf b ON b.date=a.rf_index_start_date
        LEFT JOIN rf e ON e.date=a.rf_index_end_date ORDER BY month,shift_days''',[m['as_of']]).df()
    check('RUONIA_exact_ratio_SQL',close(rcheck.rf_m,rcheck.expected_rf))
    check('RUONIA_exact_endpoint_values',close(rcheck.rf_index_start_value,rcheck.v0) and close(rcheck.rf_index_end_value,rcheck.v1))
    s1=rfm[rfm.shift_days==1]
    check('primary_close_date_shift1',s1.rf_index_start_date.eq(s1.month_start+pd.Timedelta(days=1)).all()
        and s1.rf_index_end_date.eq(s1.month_end+pd.Timedelta(days=1)).all())
    s0=rfm[rfm.shift_days==0]
    check('shift0_separate_diagnostic_dates',s0.rf_index_start_date.eq(s0.month_start).all() and s0.rf_index_end_date.eq(s0.month_end).all())
    outputs={}
    for fn,shift in [('factors_monthly.parquet',1),('factors_monthly_shift0_DIAGNOSTIC.parquet',0)]:
        f=pd.read_parquet(ROOT/fn)
        check(fn+':unique_month_shift',not f.month.duplicated().any() and f.shift_days.eq(shift).all())
        q=f.merge(rfm[rfm.shift_days==shift][['month','rf_m']],on='month',suffixes=('','_expected'),validate='1:1')
        check(fn+':RF_matches_declared_convention',close(q.rf_m,q.rf_m_expected))
        for code in ['MCFTR','MCFTRR','RGBITR','RUCBITR','RUCBTRNS']:
            q=f.merge(monthly[monthly.series_id==code][['month','return_m']],on='month',validate='1:1')
            check(fn+':'+code+'_matches_endpoints',close(q[code+'_return_m'],q.return_m))
        for out,x,y in [('equity_excess_m','MCFTR_return_m','rf_m'),('bond_excess_m','RGBITR_return_m','rf_m'),
            ('equity_resident_net_excess_m','MCFTRR_return_m','rf_m'),('credit_factor_m','RUCBTRNS_return_m','RGBITR_return_m'),
            ('legacy_credit_factor_m','RUCBITR_return_m','RGBITR_return_m')]:
            check(fn+':'+out,close(f[out],f[x]-f[y]))
        check(fn+':completeness_flags',f.primary_complete.eq(f[['rf_m','MCFTR_return_m','RGBITR_return_m']].notna().all(axis=1)).all())
        check(fn+':partial_2026Nov_missing',f.loc[f.month=='2026-11',['rf_m','MCFTR_return_m','RGBITR_return_m','credit_factor_m']].isna().all().all())
        full=f[f.primary_complete]
        outputs[fn]=dict(rows=len(f),first_complete=full.month.min(),last_complete=full.month.max(),complete_months=len(full),sha256=sha(ROOT/fn))
    f=pd.read_parquet(ROOT/'factors_monthly.parquet')
    check('primary_minimum_target_2013_2026Oct',f.loc[f.month.between('2013-01','2026-10'),'primary_complete'].sum()==166)
    # Provide factor-only feasibility, never a claim of sufficient fund unit-return history.
    rows=[];ff=f.copy();ff['period']=pd.PeriodIndex(ff.month,freq='M');ff=ff.set_index('period',drop=False)
    for t in ff.index:
        train=ff.loc[(ff.index>=t-59)&(ff.index<=t)&ff.primary_complete]
        last12=int(((train.index>=t-11)&(train.index<=t)).sum())
        x=train[['equity_excess_m','bond_excess_m']].to_numpy()
        condition=None;rank=0
        if len(x)>2 and (x.std(axis=0)>0).all():
            X=np.column_stack([np.ones(len(x)),(x-x.mean(axis=0))/x.std(axis=0)])
            rank=int(np.linalg.matrix_rank(X));condition=float(np.linalg.cond(X))
        row=dict(month=str(t),past60_complete_factor_months=len(train),last12_complete_factor_months=last12,
            factor_design_rank=rank,standardized_condition_number=condition,
            factor_only_training_60_min36_ready=bool(len(train)>=36 and last12>=9 and rank==3 and condition<=10000))
        for h in [1,2,3]:
            future=[ff.loc[t+k,'primary_complete'] if t+k in ff.index else False for k in range(1,h+1)]
            row[f'future_{h}m_factors_complete']=all(future)
        rows.append(row)
    availability=pd.DataFrame(rows)
    availability.to_csv(ROOT/'factor_history_availability.csv',index=False)
    gaps=daily[daily.calendar_gap_days.ge(10)][['series_id','previous_observed_date','date','calendar_gap_days']]
    gaps.to_csv(ROOT/'long_observed_trading_gaps.csv',index=False)
    # This is a comparison, not permission to replace the primary convention.
    a=pd.read_parquet(ROOT/'factors_monthly.parquet')[['month','rf_m']]
    b=pd.read_parquet(ROOT/'factors_monthly_shift0_DIAGNOSTIC.parquet')[['month','rf_m']]
    compare=a.merge(b,on='month',suffixes=('_primary_shift1','_diagnostic_shift0'))
    compare['difference_bps']=(compare.rf_m_primary_shift1-compare.rf_m_diagnostic_shift0)*10000
    compare.to_csv(ROOT/'ruonia_convention_comparison.csv',index=False)
    c.close()
    fail=sum(not x['passed'] for x in checks)
    report=dict(status='PASS' if not fail else 'FAIL',checked_at_utc=datetime.now(timezone.utc).isoformat(),
        checks_pass=len(checks)-fail,checks_fail=fail,checks=checks,monthly_outputs=outputs,
        long_trading_gaps=gaps.to_dict('records'),
        factor_only_latest_origins={str(h):availability.loc[availability[f'future_{h}m_factors_complete'],'month'].max() for h in [1,2,3]},
        rf_convention_max_absolute_difference_bps=float(compare.difference_bps.abs().max()),
        limitations=['RUONIA begins January 2010, first complete RF month February 2010; 2008-2009 RF unavailable.',
            'RUCBITR stops May 2023; RUCBTRNS starts December 2018. Separate series; no splice.',
            'November 2026 is partial and all monthly factor returns are missing.',
            'Factor-only readiness does not establish fund-return, distribution adjustment or complete H3 readiness.'])
    js(ROOT/'qa_independent.json',report)
    print(json.dumps({k:report[k] for k in ['status','checks_pass','checks_fail','monthly_outputs','factor_only_latest_origins','rf_convention_max_absolute_difference_bps']},ensure_ascii=False),flush=True)
    if fail:
        raise SystemExit(1)


if __name__=='__main__':
    main()
