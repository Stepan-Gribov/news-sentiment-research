"""Bounded official provider-convention TR extension: approved55 rows only.

No live collector, raw-NAV backfill, second cash/split adjustment, new source
collection or change to frozen H3 formulas. One fund cannot support inference.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from h3_alpha_core_20261104 import rolling_future_alpha,truth,numeric,unique
from h3_alpha_io_20261104 import ROOT,sha,file_record,save_json,seal_directory,checkpoint,utc
from run_h3_alpha_20261104 import estimate_grid

SNAP_SHA='cb3054b3a1305a81d31dc1078caeef4d24489a821e6d24dd9625907b5190e4fd'
SPEC_SHA='565a726fd3858b94b2bfb3dbf43442d8cab4812c79749fd2ea24db806c44e811'
BRANCH='TR_PROVIDER_ACCRUED_INCOME_CONVENTION'


def run(approval_path=None):
    parent=ROOT/'outputs/run_d0440fad412ee5e9';code=Path(__file__).parent
    approval=json.loads(Path(approval_path).read_text(encoding='utf-8')) if approval_path else None
    path=Path(approval['snapshot']) if approval else ROOT/'research_tr_evidence_arsagera/cached_monthly_documentary_snapshot_20261104.json'
    approved_sha=approval['snapshot_sha256'] if approval else SNAP_SHA
    approved_funds=approval['funds'] if approval else {'403':{'api_code':'fa','registration_number':'0363-75409054','first_month':'2022-01','last_month':'2026-07','rows':55}}
    approved_rows=sum(x['rows'] for x in approved_funds.values())
    snapshot=json.loads(path.read_text(encoding='utf-8'))
    inputs=dict(snapshot=file_record(path,approved_sha),spec=file_record(code/'h3_alpha_spec_20261104.json',SPEC_SHA),
        v3=file_record(parent/'manifest.json','d61349f8275762eee3c224a8118d69eec81fb08107d48a6fc00f63b324a09628'),
        returns_code=file_record(__file__),core=file_record(code/'h3_alpha_core_20261104.py'),
        grid_helper=file_record(code/'run_h3_alpha_20261104.py'),FE_helper=file_record(code/'fit_h3_alpha_fe_20261104.py'))
    if approval:
        if approval['status']!='RESEARCH_CERTIFIED_IMMUTABLE_SNAPSHOT':raise AssertionError('Uncertified source forbidden')
        inputs['documentary_approval']=file_record(approval_path)
        inputs['authority_review']=file_record(approval['authority_review'])
    for label,pin,path_key in [('method','method_sha256','method_source_path'),('formula','formula_image_sha256','formula_image_path'),
        ('unit_endpoint_evidence','unit_nav_sha256','unit_nav_source_path'),('crosswalk','crosswalk_sha256','crosswalk_path')]:
        inputs[label]=file_record(snapshot[path_key],snapshot[pin])
    calendar=ROOT/'factors/normalized/MCFTR_daily.parquet'
    inputs['calendar']=file_record(calendar,'6373ddcdfa68964a740b83d4218a000e3d47f60a0b42669f49f9258cc9c2b07f')
    inputs['factors']=file_record(parent/'monthly_factors_adapter.parquet')
    inputs['origins']=file_record(parent/'origin_eligibility.parquet')
    v3=json.loads((parent/'input_manifest.json').read_text(encoding='utf-8'))['inputs']
    inputs['broad_NTS']=file_record(v3['nts_broad']['path'],v3['nts_broad']['sha256'])
    rows=pd.DataFrame(snapshot['rows'])
    unique(rows,['fund_id','month'],'approved published monthly returns')
    if len(rows)!=approved_rows or set(rows.fund_id.astype(str))!=set(approved_funds):
        raise AssertionError('Snapshot exceeds or differs from the bounded documentary approval')
    for fund,bounds in approved_funds.items():
        sample=rows.loc[rows.fund_id.astype(str).eq(fund)]
        if len(sample)!=bounds['rows'] or list(sorted(sample.month))!=list(pd.period_range(bounds['first_month'],bounds['last_month'],freq='M').astype(str)):
            raise AssertionError('Approved per-fund monthly range mismatch '+fund)
        if set(sample.registration_number.astype(str))!={bounds['registration_number']}:raise AssertionError('Approved registration mismatch')
    # Verify every approved PDF hash without accessing the live candidate file.
    for row in rows.itertuples(index=False):file_record(row.source_path,row.source_sha256)
    r=rows.copy();r['fund_id']=r.fund_id.astype(str)
    r['start_valuation_date']=pd.to_datetime(r.actual_start_valuation_date)
    r['end_valuation_date']=pd.to_datetime(r.actual_end_valuation_date)
    r['fund_return']=numeric(r.return_decimal)  # Published printed field; never the NAV ratio.
    token=pd.to_numeric(r.token.str.replace('%','',regex=False).str.replace(',','.',regex=False),errors='raise')/100
    if not np.allclose(token,r.fund_return,rtol=0,atol=1e-15):raise AssertionError('Published token/decimal mismatch')
    r['source_series_id']=[f"arsagera:{approved_funds[fund]['api_code']}:{fund}:official_monthly_accrued_income_split_normalized:{approved_funds[fund]['first_month']}_{approved_funds[fund]['last_month']}" for fund in r.fund_id]
    r['return_convention']='Published arithmetic monthly return including accrued unit income and source split normalization; monthly chaining, not audited payment-date reinvestment'
    r['printed_percentage_decimal_places']=2;r['rounding_half_width_decimal']=.00005
    r['cash_event_history_complete']=False;r['raw_NAV_history_promoted_TR']=False
    r['additional_payout_or_split_adjustment_applied']=False
    cross=pd.read_parquet(snapshot['crosswalk_path']);cross['fund_id']=cross.fund_id.astype(str)
    fields=['fund_id','registration_number','identity_unambiguous','identity_observed_from','identity_observed_to']
    r=r.merge(cross[fields],on='fund_id',how='left',validate='many_to_one',suffixes=('','_crosswalk'))
    r['identity_verified']=truth(r.identity_unambiguous)&r.registration_number.eq(r.registration_number_crosswalk)
    r['identity_verified']&=r.start_valuation_date.ge(pd.to_datetime(r.identity_observed_from))&r.end_valuation_date.le(pd.to_datetime(r.identity_observed_to))
    nav=pd.read_parquet(snapshot['unit_nav_source_path']);nav['fund_id']=nav.fund_id.astype(str)
    currency=nav.groupby('fund_id',observed=True).currency.agg(lambda x:sorted(set(x.dropna().astype(str).str.upper())))
    r['currency_verified_RUB']=r.fund_id.map(currency).map(lambda x:x==['RUB'])
    date_columns=[c for c in pq.read_schema(calendar).names if c.lower() in ['tradedate','date','trade_date']]
    if len(date_columns)!=1:raise AssertionError('Unambiguous official trading-date column required')
    dates=np.sort(pd.to_datetime(pd.read_parquet(calendar,columns=date_columns)[date_columns[0]]).dropna().unique())
    period=pd.PeriodIndex(r.month,freq='M')
    r['return_period_end']=period.end_time.normalize()
    endpoint_ok=pd.Series(True,index=r.index)
    for name,months in [('start',period-1),('end',period)]:
        observed=pd.to_datetime(r[name+'_valuation_date'])
        eom=months.end_time.normalize()
        age=np.searchsorted(dates,eom,side='right')-np.searchsorted(dates,observed,side='right')
        r[name+'_age_trading_days']=age
        endpoint_ok&=observed.notna()&(observed<=eom)&(age>=0)&(age<=3)
        endpoint_ok&=observed.dt.to_period('M').astype(str).to_numpy()==months.astype(str)
        endpoint_ok&=observed.ge(pd.Timestamp(dates.min()))&pd.Series(eom<=pd.Timestamp(dates.max()),index=r.index)
    r['return_verified']=truth(r.identity_verified)&r.currency_verified_RUB&endpoint_ok&np.isfinite(r.fund_return)&r.fund_return.gt(-1)
    r['return_gate_reason']=np.select([~r.identity_verified,~r.currency_verified_RUB,~endpoint_ok],
        ['dated_identity_unverified','currency_unverified','endpoint_month_or_staleness_failure'],default='eligible_approved_provider_convention')
    version=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()[:16]
    out=ROOT/'outputs'/(('provider_TR4_' if approval else 'provider_TR_')+version);out.mkdir(exist_ok=True,parents=True)
    if (out/'manifest.json').exists():raise RuntimeError('Approved snapshot version already complete; no duplicate run')
    copied={}
    for key in ['returns_code','core','grid_helper','FE_helper']:
        src=Path(inputs[key]['path']);dest=out/'code'/src.name;dest.parent.mkdir(exist_ok=True)
        if not dest.exists():shutil.copy2(src,dest)
        copied[key]=file_record(dest,inputs[key]['sha256'])
    inputs.update(copied)
    save_json(out/'input_manifest.json',dict(inputs=inputs,provider_approval=approved_funds,spec_sha256=SPEC_SHA))
    r.to_parquet(out/'approved_monthly_provider_returns.parquet',index=False,compression='zstd')
    factors=pd.read_parquet(parent/'monthly_factors_adapter.parquet')
    origins=pd.read_parquet(parent/'origin_eligibility.parquet');origins=origins.loc[origins.fund_id.astype(str).isin(approved_funds)].copy()
    broad=pd.read_parquet(inputs['broad_NTS']['path']);broad=broad.loc[broad.fund_id.astype(str).isin(approved_funds)].copy()
    # Dated identity, mandate and source metadata remain exactly as in v3.
    for c in origins.columns:
        if c not in broad:broad[c]=broad.interval_id.map(origins.set_index('interval_id')[c])
    for c in ['identity_origin_verified','universe_eligible','row_id']:
        broad[c]=broad.interval_id.map(origins.set_index('interval_id')[c])
    def alpha(history,minimum,credit=False):
        f=factors.copy();definitions={'equity_return':'minus_rf','bond_return':'minus_rf'}
        if credit:f['factors_verified']=truth(f.corporate_sensitivity_complete);definitions['credit_factor']='already_excess_or_spread'
        a,m=rolling_future_alpha(origins[['fund_id','interval_id','end_date','identity_origin_verified']],r,f,
            factor_definitions=definitions,lookback_months=history,minimum_history=minimum,
            minimum_recent=9,recent_months=12,horizons=(1,2,3),condition_max=10000)
        label=f'lookback{history}_min{minimum}'+('_credit' if credit else '')
        a.to_parquet(out/(label+'_alpha.parquet'),index=False,compression='zstd')
        m.to_parquet(out/(label+'_training.parquet'),index=False,compression='zstd')
        return a
    checkpoint('PROVIDER_TR_ALPHA',output=str(out),source_months=len(r),eligible_months=int(r.return_verified.sum()),source_funds=len(approved_funds))
    main=alpha(60,36);short=alpha(36,24);credit=alpha(60,36,True)
    results=estimate_grid(origins,main,out/'regression/PRIMARY',BRANCH)
    reference={}
    for result in results:
        if result['variant']=='CURRENT_PLUS_LAGNEWS':
            used=pd.read_parquet(out/'regression/PRIMARY'/result['model_id']/'used_sample.parquet')
            reference[(result['group'],result['horizon'])]=set(used.row_id) if len(used) else set()
    results+=estimate_grid(origins,short,out/'regression/HISTORY36_MIN24',BRANCH,'HISTORY36_MIN24',variants=('CURRENT_PLUS_LAGNEWS',),reference_ids=reference)
    results+=estimate_grid(broad,main,out/'regression/BROAD_OBSERVED_NTS',BRANCH,'BROAD_OBSERVED_NTS',variants=('CURRENT_PLUS_LAGNEWS',),reference_ids=reference)
    for phase in range(3):
        results+=estimate_grid(origins,main,out/f'regression/NONOVERLAPPING_PHASE{phase}',BRANCH,'NONOVERLAPPING_PHASES',
            variants=('CURRENT_PLUS_LAGNEWS',),phase=phase,horizons=(2,3),reference_ids=reference)
    results += [dict(x,model_id=x['model_id'].replace('__PRIMARY__','__NONOVERLAPPING_PHASES__'),sensitivity='NONOVERLAPPING_PHASES',
        reuse_of_primary_model_id=x['model_id'],phase=0) for x in results.copy() if x['sensitivity']=='PRIMARY' and x['variant']=='CURRENT_PLUS_LAGNEWS' and x['horizon']==1]
    results+=estimate_grid(origins,credit,out/'regression/CORPORATE_BOND_SPREAD',BRANCH,'CORPORATE_BOND_SPREAD',groups=['pooled','bond','mixed'],variants=('CURRENT_PLUS_LAGNEWS',),reference_ids=reference)
    keys=credit.loc[credit.alpha.notna(),['fund_id','interval_id','horizon']].assign(credit_eligible=True)
    same=main.merge(keys,on=['fund_id','interval_id','horizon'],how='left',validate='one_to_one');same.loc[~truth(same.credit_eligible),'alpha']=np.nan
    results+=estimate_grid(origins,same,out/'regression/CORPORATE_SAME_SAMPLE_TWO_FACTOR',BRANCH,'CORPORATE_SAME_SAMPLE_TWO_FACTOR',groups=['pooled','bond','mixed'],variants=('CURRENT_PLUS_LAGNEWS',),reference_ids=reference)
    if len(results)!=90:raise AssertionError('Exactly90 fixed provider cells required')
    estimated=[x for x in results if x['status']=='ESTIMATED']
    if any(x['funds']<2 or not x['weak_cluster_inference'] or not x['significance_stars_suppressed'] for x in estimated):
        raise AssertionError('Invalid cluster count or missing fragile-inference qualification')
    coverage=[]
    for label,frame in [('PRIMARY_60_36',main),('HISTORY36_MIN24',short),('CREDIT_60_36',credit)]:
        for (fund,h),g in frame.groupby(['fund_id','horizon'],observed=True):
            eligible=g.loc[g.alpha.notna()];nts=eligible.interval_id.isin(set(origins.loc[origins.nts_pair_eligible&origins.universe_eligible,'interval_id']))
            coverage.append(dict(series=str(fund),branch=label,horizon=int(h),source_months=int(r.fund_id.eq(str(fund)).sum()),eligible_source_months=int((r.fund_id.eq(str(fund))&r.return_verified).sum()),
                origins_total=len(g),alpha_defined=len(eligible),alpha_with_primary_NTS=int(nts.sum()),
                first_eligible_origin=str(pd.to_datetime(eligible.origin_date).min().date()) if len(eligible) else None,
                last_eligible_origin=str(pd.to_datetime(eligible.origin_date).max().date()) if len(eligible) else None,
                last_future_month=eligible.future_last_month.max() if len(eligible) else None,
                inference_status='MANAGER_SPECIFIC_FE_INFERENCE_MAX4_CLUSTERS_FRAGILE' if approval else 'NOT_ESTIMABLE_ONE_APPROVED_FUND'))
    pd.DataFrame(coverage).to_csv(out/'provider_TR_alpha_coverage.csv',index=False,encoding='utf-8-sig')
    for label,frame in [('primary',main),('history36',short),('credit',credit)]:
        frame.groupby(['horizon','status'],dropna=False).size().rename('origins').reset_index().to_csv(out/f'{label}_alpha_NA_flow.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{k:v for k,v in x.items() if not isinstance(v,(list,dict))} for x in results]).to_csv(out/'models.csv',index=False,encoding='utf-8-sig')
    save_json(out/'all_models.json',results)
    coef=[];tests=[]
    for result in results:
        tags={k:result[k] for k in ['model_id','sensitivity','group','horizon','variant','nobs','funds']}
        coef += [dict(tags,**x) for x in result.get('coefficients',[])]
        tests += [dict(tags,**x) for x in result.get('target_tests',[])]
    pd.DataFrame(coef).to_csv(out/'coefficients.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(tests).to_csv(out/'tests.csv',index=False,encoding='utf-8-sig')
    summary=dict(status='PROVIDER_TR_CALCULATED_MANAGER_SPECIFIC_FRAGILE_INFERENCE' if estimated else 'PROVIDER_TR_ALPHA_COMPUTED_NO_CLUSTERED_INFERENCE',output=str(out),source_snapshot_sha256=approved_sha,
        source_series_ids=sorted(r.source_series_id.unique()),approved_source_months=len(r),eligible_source_months=int(r.return_verified.sum()),funds=len(approved_funds),
        model_labels=90,estimated_hypothesis_models=len(estimated),spec_unchanged=True,NTS_unchanged=True,diagnostic_v3_reused_not_rerun=True,
        source_convention=r.return_convention.iloc[0],coverage=coverage,independent_QA_pending=True,
        limit='Only documentary-certified printed returns; max4 same-manager funds. No population generalization, significance stars, NAV backfill, ratio replacement, second adjustment or payment-date reinvestment claim.')
    save_json(out/'summary.json',summary)
    text=f'Официальная TR-конвенция: {len(approved_funds)} фонда, {len(r)} опубликованных доходностей. Все фонды одной УК; максимум4 кластера — крайне неустойчивая инференция, без звёздочек и обобщений на рынок. Alpha посчитана без подстановки NAV и повторных корректировок.'
    (out/'review.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:18px system-ui;margin:32px}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}</style><h1>Официальная TR-конвенция</h1><p>'+text+'</p>'+pd.DataFrame(coverage).to_html(index=False)+'<h2>Фиксированные ячейки</h2>'+pd.DataFrame([{k:v for k,v in x.items() if k in ['model_id','group','horizon','sensitivity','status','nobs','funds','reason']} for x in results]).to_html(index=False),encoding='utf-8')
    for rec in inputs.values():file_record(rec['path'],rec['sha256'])
    seal_directory(out,dict(status=summary['status'],inputs=inputs,summary='summary.json'))
    export=ROOT/'exports'/out.name;export.mkdir(exist_ok=True,parents=True)
    bundle=export/'provider_TR_small.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in out.glob('*'):
            if p.is_file() and p.suffix in ['.json','.csv','.html']:archive.write(p,p.name)
        for p in (out/'regression').rglob('native.summary.txt'):archive.write(p,str(p.relative_to(out)).replace('\\','/'))
        for p in (out/'regression').rglob('*'):
            if p.is_file() and (p.name.endswith('.summary.html') or p.suffix=='.csv'):archive.write(p,str(p.relative_to(out)).replace('\\','/'))
    save_json(export/'export_manifest.json',dict(bundle=file_record(bundle),run_manifest_sha256=sha(out/'manifest.json'),source=str(out)))
    save_json(ROOT/'runtime/provider_TR_ready.json',dict(output=str(out),summary=summary,manifest_sha256=sha(out/'manifest.json'),bundle=str(bundle)))
    checkpoint('PROVIDER_TR_COMPLETE_NO_INFERENCE',output=str(out),summary=summary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--approval');a=p.parse_args();run(a.approval)
