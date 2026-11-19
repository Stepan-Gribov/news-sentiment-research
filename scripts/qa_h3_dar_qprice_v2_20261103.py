"""Independent QA of supplementary quantity-direction cash-excess clean-price DAR.

Research data are read on research-host only. No PanelOLS or regression runner is
imported. The only research write is the separately named QA JSON report.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def row_hash(frame):
    return hashlib.sha256('\n'.join(frame.row_id.astype(str)).encode()).hexdigest()


def truth(values):
    return values.astype(str).str.lower().str.strip().isin(['true', '1', 'yes'])


def finite(frame, columns):
    return np.isfinite(frame[columns].apply(pd.to_numeric, errors='coerce').to_numpy(float)).all(axis=1)


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (np.integer, np.bool_)): return value.item()
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value, (Path, pd.Timestamp)): return str(value)
    return value


def within_ols(frame, terms, outcome='dar_q_cash_price_pp'):
    """Fund-demeaned OLS and independent two-way CGM sandwich."""
    groups = frame.fund_id.astype(str)
    x = frame[terms].astype(float)
    y = frame[outcome].astype(float)
    xw = (x-x.groupby(groups, sort=False).transform('mean')).to_numpy(float)
    yw = (y-y.groupby(groups, sort=False).transform('mean')).to_numpy(float)
    scale = np.linalg.norm(xw, axis=0)
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Retained term lacks finite within variation')
    z = xw/scale
    beta_z, _, rank, singular = np.linalg.lstsq(z, yw, rcond=None)
    if rank != len(terms): raise ValueError('Within design is rank deficient')
    residual = yw-z@beta_z
    bread = np.linalg.inv(z.T@z)
    scores = z*residual[:, None]

    def meat(keys):
        codes, labels = pd.factorize(keys, sort=False)
        n, g = len(codes), len(labels)
        if g < 2: raise ValueError('Fewer than two clusters')
        sums = np.zeros((g, scores.shape[1]))
        np.add.at(sums, codes, scores)
        return sums.T@sums * (g/(g-1)) * ((n-1)/n), g

    fund_meat, funds = meat(groups)
    companies = frame.company_id.astype(str)
    company_meat, company_count = meat(companies)
    joint_meat, joint_count = meat(pd.MultiIndex.from_arrays([groups, companies]))
    n, k = z.shape
    df = n-funds-k
    if df <= 0: raise ValueError('Nonpositive residual degrees of freedom')
    covariance = bread@(fund_meat+company_meat-joint_meat)@bread * n/df
    covariance = covariance/scale[:, None]/scale[None, :]
    ssr, tss = float(residual@residual), float(yw@yw)
    return dict(beta=beta_z/scale, covariance=(covariance+covariance.T)/2,
        nobs=n, funds=funds, companies=company_count, joint_clusters=joint_count,
        df_resid=df, rank=rank, condition_number=float(singular.max()/singular.min()),
        within_SSR=ssr, within_TSS=tss, within_R2=1-ssr/tss if tss else np.nan)


REPLICAS=[f"Q99_D99__M{h}__FOCUS_{g}_ROWS_60_INTERVALS_24__CURRENT_PLUS_LAG1" for h in [1,3] for g in ["EQUITY","BOND","MIXED"]]
LITERAL_DATE_FIELDS=['future_target_date','p0_quote_date','future_quote_date']


def restore_literal_date_columns(frame,snapshot):
    """Restore source Date32 objects after auxiliary timestamp arithmetic."""
    result=frame.copy()
    keys=pd.MultiIndex.from_frame(result[['horizon','row_id']])
    original=snapshot.set_index(['horizon','row_id']).reindex(keys)
    for field in LITERAL_DATE_FIELDS:
        values=original[field].reset_index(drop=True);values.index=result.index
        result[field]=values
    return result


def equal_values(left,right):
    try:
        pd.testing.assert_frame_equal(left.reset_index(drop=True),right.reset_index(drop=True),check_exact=True,check_dtype=False)
        return True
    except AssertionError:return False


def audit(args):
    if os.name!='nt':raise RuntimeError('Actual-data QA must run on research-host')
    from scipy import stats
    planpath=Path(args.plan);plan=json.loads(planpath.read_text(encoding='utf-8'));root=Path(args.root or plan['output_root'])
    output=Path(args.output or root/'qa_independent_v2.json')
    if output.exists():raise FileExistsError('Refusing to overwrite an existing QA report')
    if 'databaseresources'in[p.lower()for p in output.parts]:raise ValueError('Protected output')
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'));coef=pd.read_csv(root/'coefficients.csv')
    prior=json.loads(Path(plan['focus_summary']).read_text(encoding='utf-8'));checks,warnings,details=[],[],{}
    def check(name,passed,**evidence):checks.append(dict(name=name,status='PASS'if bool(passed)else'FAIL',**clean(evidence)))
    def close(a,b,atol=1e-7):return bool(np.isclose(float(a),float(b),rtol=args.rtol,atol=atol))
    def numeric(s):return pd.to_numeric(pd.Series(s),errors='coerce').to_numpy(dtype=float,na_value=np.nan)
    def same_num(a,b,atol=0.):return bool(np.allclose(numeric(a),numeric(b),rtol=0,atol=atol,equal_nan=True))
    def text_values(s):return s.astype('string').fillna('<NA>')
    check('frozen_plan_hash',sha(planpath)==summary['plan_sha256'])
    check('same_frozen_specification',summary.get('specification')==plan)
    check('predeclared_independent_models',args.models==plan['independent_qa_models'])
    check('supplement_only_core_alpha_not_estimated',plan['not_a_replacement_for_core_H3']is True and plan['core_H3_alpha_status'].startswith('NOT_ESTIMABLE')and summary['not_a_replacement_for_core_H3']is True and summary['core_H3_alpha_status']==plan['core_H3_alpha_status'])
    check('native_unadjusted_coefficient_fields',plan['multiple_testing_adjustments']is False and set(coef.columns)=={'model','outcome','term','beta','se_native','t_native','p_native','ci95_native_low','ci95_native_high'})
    before={}
    for key,expected in plan['expected_source_sha256'].items():
        before[key]=sha(plan[key]);check('frozen_source:'+key,before[key]==expected,sha256=before[key])
    source=pd.read_parquet(plan['input_panel']).sort_values(['fund_id','end_date','security_id','interval_id'],kind='stable').reset_index(drop=True)
    mapping=pd.read_parquet(plan['lag_map']);future=pd.read_parquet(plan['future_outcomes']);prices=pd.read_parquet(plan['future_prices'])
    check('source_N_unique_ids',len(source)==plan['expected_primary_rows']and not source.row_id.duplicated().any())
    check('future_full_row_horizon_universe',len(future)==len(source)*3 and not future.duplicated(['row_id','horizon']).any()and set(future.horizon)=={1,2,3})
    future=future.sort_values(['horizon','row_id'],kind='stable').reset_index(drop=True);prices=prices.sort_values(['horizon','row_id'],kind='stable').reset_index(drop=True)
    check('raw_price_inputs_preserved_by_RF_attachment',len(prices)==len(future)and equal_values(future[list(prices.columns)],prices));del prices
    literal_date_snapshot=future[['horizon','row_id']+LITERAL_DATE_FIELDS].copy()
    aligned_map=mapping.set_index('row_id').reindex(source.row_id).reset_index(drop=True)
    check('lagmap_unique_exact_source',not mapping.row_id.duplicated().any()and set(mapping.row_id)==set(source.row_id))
    check('lagmap_current_binding',np.array_equal(source.fund_id.astype(str),aligned_map.fund_id.astype(str))and np.array_equal(source.company_id.astype(str),aligned_map.company_id.astype(str))and np.array_equal(source.interval_id.astype(str),aligned_map.current_interval_id.astype(str)))
    base=source.copy()
    for field in aligned_map:
        if field.startswith('lag_')or field=='net_sentiment_lag1':base[field]=aligned_map[field].to_numpy()
    lag_eligible=truth(aligned_map.lag_eligible_primary)&np.isfinite(pd.to_numeric(aligned_map.net_sentiment_lag1,errors='coerce'))
    base['complete_current_lag1']=np.asarray(lag_eligible,bool)
    rf=pd.read_parquet(plan['rf_index']);rf['date']=pd.to_datetime(rf['date']).dt.normalize()
    check('RF_index_daily_positive_unique',rf.date.notna().all()and not rf.date.duplicated().any()and np.isfinite(rf.ruonia_index).all()and rf.ruonia_index.gt(0).all()and len(rf)==len(pd.date_range(rf.date.min(),rf.date.max(),freq='D')))
    check('RF_shift_frozen_zero',plan['rf_index_shift_days']==0 and future.rf_date_shift_days.eq(0).all())
    index=rf.set_index('date').ruonia_index
    future['current_end_date']=pd.to_datetime(future.current_end_date);future['future_target_date']=pd.to_datetime(future.future_target_date)
    future['p0_quote_date']=pd.to_datetime(future.p0_quote_date);future['future_quote_date']=pd.to_datetime(future.future_quote_date)
    ids=source.set_index('row_id');source_rows=ids.reindex(future.row_id)
    for field in ['fund_id','company_id','security_id','quantity_open','quantity_close','delta_quantity_units','opening_report_id','closing_report_id']:
        if pd.api.types.is_numeric_dtype(source_rows[field]):ok=same_num(future[field],source_rows[field])
        else:ok=np.array_equal(text_values(future[field]),text_values(source_rows[field]))
        check('future_identity_source:'+field,ok)
    check('future_actual_origin_dates',np.array_equal(future.current_end_date.to_numpy(dtype='datetime64[ns]'),pd.to_datetime(source_rows.end_date).to_numpy(dtype='datetime64[ns]')))
    for horizon in [1,2,3]:
        part=future.loc[future.horizon.eq(horizon)]
        check('exact_calendar_horizon:'+str(horizon),np.array_equal(part.future_target_date.to_numpy(dtype='datetime64[ns]'),(part.current_end_date+pd.DateOffset(months=horizon)).to_numpy(dtype='datetime64[ns]')))
    delta=pd.to_numeric(source_rows.delta_quantity_units,errors='coerce').reset_index(drop=True);direction=np.sign(delta)
    nonzero=np.isfinite(delta)&delta.ne(0)
    check('estimated_delta_from_Q',same_num(source.delta_quantity_units,source.quantity_close-source.quantity_open,1e-7))
    check('trade_direction_literal',same_num(future.trade_sign,direction))
    origin_age=(future.current_end_date-future.p0_quote_date).dt.days;future_age=(future.future_target_date-future.future_quote_date).dt.days
    rub=['RUB','RUR','SUR']
    price_eligible=(future.requested_isin.notna()&future.secid.notna()&future.boardid.notna()
        &np.isfinite(future.p0_price_rub)&future.p0_price_rub.gt(0)&future.p0_quote_date.le(future.current_end_date)&origin_age.between(0,7)
        &future.p0_currency.isin(rub)&np.isfinite(future.future_price_rub)&future.future_price_rub.gt(0)
        &future.future_quote_date.gt(future.current_end_date)&future.future_quote_date.le(future.future_target_date)&future_age.between(0,7)
        &future.future_currency.isin(rub)&future.future_source_market.eq(future.p0_source_market))
    check('independent_price_eligibility',np.array_equal(truth(future.price_eligible),price_eligible))
    check('quote_age_fields',same_num(future.loc[price_eligible,'p0_quote_age_days'],origin_age.loc[price_eligible])and same_num(future.loc[price_eligible,'future_quote_age_days'],future_age.loc[price_eligible]))
    expected_return=(future.future_price_rub/future.p0_price_rub-1).where(price_eligible)
    check('future_clean_RUB_price_return',same_num(future.future_price_return,expected_return,1e-11))
    start_values=index.reindex(pd.DatetimeIndex(future.current_end_date)).to_numpy(float);end_values=index.reindex(pd.DatetimeIndex(future.future_target_date)).to_numpy(float)
    rf_eligible=np.isfinite(start_values)&np.isfinite(end_values)&(start_values>0)&(end_values>0)
    cash=end_values/start_values-1
    check('RF_exact_endpoint_dates',np.array_equal(pd.to_datetime(future.rf_index_start_date).to_numpy(dtype='datetime64[ns]'),future.current_end_date.to_numpy(dtype='datetime64[ns]'))and np.array_equal(pd.to_datetime(future.rf_index_end_date).to_numpy(dtype='datetime64[ns]'),future.future_target_date.to_numpy(dtype='datetime64[ns]')))
    check('RF_exact_endpoint_values',same_num(future.rf_index_start_value,start_values)and same_num(future.rf_index_end_value,end_values))
    check('RF_return_no_fill_or_annualization',np.array_equal(truth(future.rf_eligible),rf_eligible)and same_num(future.cash_return,cash,1e-12)and same_num(future.rf_return,cash,1e-12))
    outcome_eligible=price_eligible&nonzero&rf_eligible
    expected_y=(100*direction*(expected_return-pd.Series(cash,index=future.index))).where(outcome_eligible)
    check('independent_outcome_eligibility',np.array_equal(truth(future.outcome_eligible),outcome_eligible))
    check('cash_excess_direction_outcome_formula',same_num(future.dar_q_cash_price_pp,expected_y,1e-8))
    source_current=numeric(source_rows.net_sentiment);source_lag=numeric(aligned_map.set_index(source.row_id).reindex(future.row_id).net_sentiment_lag1)
    expected_current=(pd.Series(direction.to_numpy()*source_current,index=future.index)).where(nonzero)
    expected_lag=(pd.Series(direction.to_numpy()*source_lag,index=future.index)).where(nonzero&np.isfinite(source_lag))
    check('current_alignment_nonzero_only',same_num(future.align_current,expected_current))
    check('lag_alignment_same_current_trade_direction',same_num(future.align_lag1,expected_lag))
    check('no_trade_not_zero_outcome',future.loc[~nonzero,'dar_q_cash_price_pp'].isna().all()and future.loc[~nonzero,['align_current','align_lag1']].isna().all().all()and not truth(future.loc[~nonzero,'outcome_eligible']).any())
    # Bind the origin identity to the unchanged closing/opening portfolio ledger.
    position_columns=['fund_id','report_id','security_id','unit_price_rub','price_isin','secid','boardid','quote_date']
    positions=pd.read_parquet(plan['positions_source'],columns=position_columns)
    needed_keys=pd.MultiIndex.from_arrays([pd.concat([source.fund_id,source.fund_id],ignore_index=True),pd.concat([source.opening_report_id,source.closing_report_id],ignore_index=True),pd.concat([source.security_id,source.security_id],ignore_index=True)])
    position_keys=pd.MultiIndex.from_frame(positions[['fund_id','report_id','security_id']]);positions=positions.loc[position_keys.isin(needed_keys)].copy()
    check('used_position_ledger_unique_keys',not positions.duplicated(['fund_id','report_id','security_id']).any())
    pindex=positions.set_index(['fund_id','report_id','security_id'])
    def endpoint(report):
        keys=pd.MultiIndex.from_arrays([source.fund_id,source[report],source.security_id],names=['fund_id','report_id','security_id'])
        return pindex.reindex(keys).reset_index(drop=True)
    closing,opening=endpoint('closing_report_id'),endpoint('opening_report_id')
    use_closing=np.isfinite(pd.to_numeric(closing.unit_price_rub,errors='coerce'))&pd.to_numeric(closing.unit_price_rub,errors='coerce').gt(0)
    fallback=~use_closing&opening[['price_isin','secid','boardid']].notna().all(axis=1)
    expected_kind=np.where(use_closing,'closing_position_quote',np.where(fallback,'opening_identity_market_fallback','no_price_identity'))
    one=future.loc[future.horizon.eq(1)].set_index('row_id').reindex(source.row_id).reset_index()
    check('origin_price_source_kind',np.array_equal(one.p0_source_kind,expected_kind))
    for field,outfield in [('price_isin','requested_isin'),('secid','secid'),('boardid','boardid')]:
        expected=closing[field].where(use_closing,opening[field])
        check('exact_closing_or_opening_identity:'+field,np.array_equal(text_values(one[outfield]),text_values(expected)))
    check('closing_origin_price_verbatim',same_num(one.loc[use_closing,'p0_price_rub'],closing.loc[use_closing,'unit_price_rub']))
    check('closing_origin_quote_date_verbatim',np.array_equal(pd.to_datetime(one.loc[use_closing,'p0_quote_date']).to_numpy(dtype='datetime64[ns]'),pd.to_datetime(closing.loc[use_closing,'quote_date']).to_numpy(dtype='datetime64[ns]')))
    del positions,pindex,closing,opening
    # Spot-check twenty distinct price windows against referenced local files.
    candidate=future.loc[outcome_eligible].copy();candidate['window_key']=candidate[['horizon','requested_isin','secid','boardid','current_end_date','future_target_date']].astype(str).agg('\t'.join,axis=1)
    candidate=candidate.sort_values(['horizon','p0_source_kind','p0_source_market','trade_sign','row_id'],kind='stable').drop_duplicates('window_key')
    selected=[];known=candidate.loc[candidate.secid.eq('VTBR')&candidate.p0_quote_date.lt(pd.Timestamp('2024-07-11'))&candidate.future_quote_date.ge(pd.Timestamp('2024-07-11'))]
    if len(known):selected.append(known.index[0])
    for _,part in candidate.groupby(['horizon','p0_source_kind','p0_source_market','trade_sign'],sort=True,dropna=False):
        if len(selected)<20 and part.index[0]not in selected:selected.append(part.index[0])
    for index_value in candidate.index:
        if len(selected)>=20:break
        if index_value not in selected:selected.append(index_value)
    raw_frames={};raw_hashes={};spot=[]
    for index_value in selected:
        row=candidate.loc[index_value];record=dict(row_id=row.row_id,horizon=int(row.horizon),isin=row.requested_isin,secid=row.secid,boardid=row.boardid,legs=[])
        for prefix,target in [('p0',row.current_end_date),('future',row.future_target_date)]:
            path=str(row[prefix+'_quote_source_file'])
            try:
                if path not in raw_frames:raw_hashes[path]=sha(path);raw_frames[path]=pd.read_parquet(path)
                raw=raw_frames[path];columns={str(c).upper():c for c in raw.columns}
                required=['SECID','BOARDID','TRADEDATE','CLOSE']
                if not set(required).issubset(columns):raise ValueError('Referenced source lacks direct quote columns')
                mask=raw[columns['SECID']].astype(str).eq(str(row.secid))&raw[columns['BOARDID']].astype(str).eq(str(row.boardid))
                if 'REQUESTED_ISIN'in columns:mask&=raw[columns['REQUESTED_ISIN']].astype(str).str.upper().str.strip().eq(str(row.requested_isin))
                dates=pd.to_datetime(raw[columns['TRADEDATE']],errors='coerce');mask&=dates.eq(pd.Timestamp(row[prefix+'_quote_date']))
                found=raw.loc[mask];close_values=pd.to_numeric(found[columns['CLOSE']],errors='coerce')
                check('raw_quote_spot:'+str(row.row_id)+':'+str(row.horizon)+':'+prefix,len(found)>0 and np.isclose(close_values,numeric([row[prefix+'_raw_close']])[0],rtol=1e-11,atol=1e-12).any())
                expected_price=float(row[prefix+'_raw_close'])*float(row[prefix+'_facevalue'])/100 if row[prefix+'_source_market']=='bonds'else float(row[prefix+'_raw_close'])
                check('raw_quote_normalization:'+str(row.row_id)+':'+str(row.horizon)+':'+prefix,close(expected_price,row['p0_price_rub'if prefix=='p0'else'future_price_rub'],1e-10))
                record['legs'].append(dict(kind=prefix,path=path,status='CHECKED',quote_date=str(row[prefix+'_quote_date'])))
            except (OSError,ValueError,KeyError,ImportError)as error:
                warnings.append(dict(kind='RAW_QUOTE_SPOTCHECK_UNAVAILABLE',row_id=row.row_id,horizon=int(row.horizon),leg=prefix,path=path,reason=str(error)));record['legs'].append(dict(kind=prefix,path=path,status='UNAVAILABLE'))
        spot.append(record)
    details['raw_quote_spot_checks']=spot
    if len(known):warnings.append(dict(kind='KNOWN_UNADJUSTED_CORPORATE_ACTION',security='VTBR',effective_date='2024-07-11',consolidation_ratio='5000:1',crossing_unique_windows=len(known),source='https://www.vtb.ru/ir/shares/listings/',note='Raw clean-price and estimated quantity changes are not corporate-action adjusted. This flag does not change any source or frozen sample.'))
    # Independently calibrate absolute-outcome cuts on unique exact price windows.
    calibration=json.loads(Path(plan['outcome_trim_calibration']).read_text(encoding='utf-8'));calibration_by_h={r['horizon']:r for r in calibration['horizons']};outcome_cuts={}
    group_keys=['horizon','requested_isin','secid','boardid','current_end_date','future_target_date']
    candidate['abs_dar']=candidate.dar_q_cash_price_pp.abs()
    # Use every eligible row before deduplication, not the spot-check representatives.
    eligible_frame=future.loc[outcome_eligible].copy();eligible_frame['abs_dar']=eligible_frame.dar_q_cash_price_pp.abs()
    uniqueness=eligible_frame.groupby(group_keys,dropna=False,observed=True)[['abs_dar','p0_price_rub','future_price_rub','cash_return','p0_quote_date','future_quote_date']].nunique(dropna=False)
    check('unique_price_window_no_conflicting_values',uniqueness.eq(1).all().all())
    unique=eligible_frame.groupby(group_keys,dropna=False,observed=True).agg(abs_dar=('abs_dar','first'),rows=('row_id','size')).reset_index()
    unique['price_window_id']=[hashlib.sha256((str(int(h))+'\t'+str(i)+'\t'+str(s)+'\t'+str(b)+'\t'+pd.Timestamp(d0).strftime('%Y-%m-%d')+'\t'+pd.Timestamp(d1).strftime('%Y-%m-%d')).encode()).hexdigest()for h,i,s,b,d0,d1 in unique[group_keys].itertuples(index=False,name=None)]
    for horizon in [1,2,3]:
        sub=unique.loc[unique.horizon.eq(horizon)];saved=calibration_by_h[horizon]
        check('outcome_calibration_population:'+str(horizon),len(sub)==saved['unique_price_windows']and int(sub.rows.sum())==saved['eligible_source_rows']and hashlib.sha256('\n'.join(sorted(sub.price_window_id)).encode()).hexdigest()==saved['eligible_window_id_hash'])
        for quantile in [.99,.975,.95,.90]:
            value=float(sub.abs_dar.quantile(quantile,interpolation='linear'));outcome_cuts[(horizon,quantile)]=value
            observed=next(c for c in saved['quantiles']if c['quantile']==quantile)
            check('unique_outcome_quantile:'+str(horizon)+':'+str(quantile),close(value,observed['cutoff_abs_dar_pp'],1e-10),independent=value)
    reference=pd.read_parquet(plan['cutoff_reference_sample'],columns=['row_id','quantity_open','quantity_close','quantity_change_pct'])
    population=reference.loc[reference.quantity_open.gt(0)&reference.quantity_close.gt(0)]
    check('quantity_cut_reference_universe',len(reference)==213612 and len(population)==plan['cutoff_calibration_continuing_rows'])
    continuing=source.quantity_open.gt(0)&source.quantity_close.gt(0);entries=source.quantity_open.eq(0)&source.quantity_close.gt(0);exits=source.quantity_open.gt(0)&source.quantity_close.eq(0)
    quantity_masks={}
    for cut in plan['cutoffs']:
        value=float(population.quantity_change_pct.quantile(cut['quantile'],interpolation='linear'));quantity_masks[cut['tag']]=~continuing|source.quantity_change_pct.le(cut['cutoff_percent'])
        check('quantity_quantile:'+cut['tag'],close(value,cut['cutoff_percent'],1e-10))
        check('quantity_cuts_keep_entries_exits:'+cut['tag'],quantity_masks[cut['tag']].loc[entries|exits].all())
        if cut['outcome_abs_quantile']is not None:
            for horizon in [1,2,3]:check('frozen_outcome_cutoff:'+cut['tag']+':'+str(horizon),close(cut['outcome_abs_cutoffs_by_horizon'][str(horizon)],outcome_cuts[(horizon,cut['outcome_abs_quantile'])],1e-10))
    support=source.groupby(['fund_focus','fund_id'],observed=True).agg(fund_rows=('row_id','size'),fund_intervals=('interval_id','nunique'),fund_end_dates=('end_date','nunique'))
    saved_support=pd.read_csv(plan['focus_support'],dtype={'fund_id':str}).set_index(['fund_focus','fund_id'])
    for field in support:check('old_focus_support_fixed:'+field,np.array_equal(support[field],saved_support.reindex(support.index)[field]))
    old_masks={'REFERENCE_AUGMENTED_P99':pd.Series(True,index=source.index)}
    for group in plan['focus_groups']:
        belongs=source.fund_focus.eq(group);old_masks['FOCUS_'+group.upper()+'_BASELINE']=belongs;sg=support.loc[group]
        for rows in [30,60]:
            for periods in [12,24,36]:old_masks[f'FOCUS_{group.upper()}_ROWS_{rows}_INTERVALS_{periods}']=belongs&source.fund_id.isin(sg.index[sg.fund_rows.ge(rows)&sg.fund_intervals.ge(periods)])
    for name,mask in old_masks.items():check('old_mask_unchanged:'+name,row_hash(source.loc[mask])==prior['samples'][name]['row_hash'])
    records={r['model']:r for r in summary['models']};controls=plan['baseline_controls']+plan['macro_controls'];variants=plan['variants']
    expected_names={cut['tag']+'__M'+str(h)+'__'+name+'__'+v for cut in plan['cutoffs']for h in plan['horizons']for name in old_masks for v in variants}
    check('all_one_thousand_eighty_labels',len(expected_names)==1080 and set(records)==set(plan['planned_models'])==expected_names)
    native_fields=['beta','se_native','t_native','p_native','ci95_native_low','ci95_native_high'];verified_samples={};seen_designs={};datasets={};inference={}
    def verify_tests(name,beta,cov,df,test,prefix):
        beta=np.asarray(beta,float);cov=np.asarray(cov,float);joint=test['joint_current_lag_zero'];total=test['sum_current_lag']
        check(prefix+'target_names_and_no_repair:'+name,test['target_terms']==['align_current','align_lag1']and test['covariance_repair']is False)
        if np.isfinite(cov).all()and np.isfinite(beta).all():
            eigenvalues=np.linalg.eigvalsh((cov+cov.T)/2)
            check(prefix+'target_cov_eigenvalues:'+name,np.allclose(joint['target_covariance_eigenvalues'],eigenvalues,rtol=args.rtol,atol=1e-10))
            if eigenvalues.min()>0:
                w=float(beta@np.linalg.solve(cov,beta));p=stats.chi2.sf(w,2)
                check(prefix+'joint_chi_square2:'+name,joint['status']=='ESTIMATED'and joint['df']==2 and 'chi2'in joint['distribution'].lower()and close(w,joint['statistic'])and close(p,joint['p_value'],1e-10))
            else:check(prefix+'undefined_nonPD_joint:'+name,joint['status']!='ESTIMATED'and joint['p_value']is None)
            estimate=float(beta.sum());variance=float(cov.sum())
            check(prefix+'sum_full_crosscovariance:'+name,close(total['beta'],estimate)and close(total['variance'],variance)and close(test['cov_current_lag'],cov[0,1]))
            if variance>0 and df>0:
                se=float(np.sqrt(variance));critical=stats.t.ppf(.975,df);p=2*stats.t.sf(abs(estimate/se),df)
                check(prefix+'sum_t_CI:'+name,total['status']=='ESTIMATED'and total['native_df_resid']==df and close(total['se_native'],se)and close(total['p_native'],p,1e-10)and close(total['ci95_native_low'],estimate-critical*se)and close(total['ci95_native_high'],estimate+critical*se))
            else:check(prefix+'undefined_nonpositive_sumvar:'+name,total['status']!='ESTIMATED'and total['p_native']is None)
        else:check(prefix+'nonfinite_covariance_not_repaired:'+name,joint['status']!='ESTIMATED'and total['status']!='ESTIMATED')
    for horizon in plan['horizons']:
        part=future.loc[future.horizon.eq(horizon)].set_index('row_id').reindex(source.row_id).reset_index()
        part=restore_literal_date_columns(part,literal_date_snapshot)
        frame=base.copy()
        for field in part:
            if field not in frame:frame[field]=part[field].to_numpy()
        common_eligible=truth(part.outcome_eligible)&lag_eligible&np.isfinite(part.align_current)&np.isfinite(part.align_lag1)
        frame['complete_dar_and_lag']=np.asarray(common_eligible,bool)
        for cut in plan['cutoffs']:
            absolute_cut=cut['outcome_abs_cutoffs_by_horizon'].get(str(horizon));outcome_keep=pd.Series(True,index=source.index)if absolute_cut is None else part.dar_q_cash_price_pp.abs().le(absolute_cut)
            for base_name,oldmask in old_masks.items():
                quantity_mask=oldmask&quantity_masks[cut['tag']];pre_dar=quantity_mask&common_eligible;mask=pre_dar&outcome_keep;expected=frame.loc[mask];pair=[]
                for variant in variants:
                    name=cut['tag']+'__M'+str(horizon)+'__'+base_name+'__'+variant;record=records[name];info=summary['samples'][name];key_hash=row_hash(expected)
                    check('fixed_horizon_common_mask_N_hash:'+name,info['row_hash']==key_hash and info['rows']==len(expected)and info['horizon']==horizon)
                    check('model_N_funds_companies:'+name,record['nobs']==len(expected)and record['funds']==expected.fund_id.nunique()and record['companies']==expected.company_id.nunique())
                    check('declared_outcome_scenario_metadata:'+name,record['outcome_abs_quantile']==cut['outcome_abs_quantile']and ((record['outcome_abs_cutoff']is None and absolute_cut is None)or(record['outcome_abs_cutoff']is not None and absolute_cut is not None and close(record['outcome_abs_cutoff'],absolute_cut,1e-10))))
                    check('drop_counts_in_declared_order:'+name,record['old_rows']==int(oldmask.sum())and record['trimmed_rows']==int(quantity_mask.sum())and record['n_cutoff_removed']==int((oldmask&~quantity_masks[cut['tag']]).sum())and record['n_future_or_lag_dropped']==int((quantity_mask&~common_eligible).sum())and record['n_abs_dar_removed']==int((pre_dar&~outcome_keep).sum()))
                    check('outcome_cut_removed_entries_exits:'+name,record['n_abs_dar_removed_entries']==int((pre_dar&~outcome_keep&truth(source.entry_zero_to_positive)).sum())and record['n_abs_dar_removed_exits']==int((pre_dar&~outcome_keep&truth(source.complete_exit)).sum()))
                    sample_key=(horizon,key_hash)
                    if sample_key not in verified_samples:
                        sample=pd.read_parquet(info['parquet']);check('actual_saved_row_order:'+name,sample.row_id.reset_index(drop=True).equals(expected.row_id.reset_index(drop=True)))
                        check('all_original_quantity_and_news_values_preserved:'+name,equal_values(sample[list(source.columns)],expected[list(source.columns)]))
                        extra=[c for c in expected if c not in source];check('all_future_price_RF_alignment_fields_preserved:'+name,equal_values(sample[extra],expected[extra]))
                        verified_samples[sample_key]=info['parquet']
                    else:check('sample_file_reuse_same_horizon_only:'+name,verified_samples[sample_key]==info['parquet'])
                    pair.append((info['horizon'],info['row_hash']));design_key=(variant,horizon,key_hash);current=coef.loc[coef.model.eq(name)].set_index('term')
                    if design_key in seen_designs:
                        check('exact_horizon_design_alias:'+name,record['exact_sample_alias_of']==seen_designs[design_key]);original=coef.loc[coef.model.eq(seen_designs[design_key])].set_index('term')
                        check('alias_native_values_unchanged:'+name,set(current.index)==set(original.index)and np.allclose(current.reindex(original.index)[native_fields],original[native_fields],rtol=1e-12,atol=1e-10,equal_nan=True))
                    else:seen_designs[design_key]=name
                    if record['status']!='ESTIMATED':check('not_estimated_has_no_coefficients:'+name,len(current)==0);inference[name]=dict(status='NOT_ESTIMATED');warnings.append(dict(model=name,kind='NOT_ESTIMATED',reason=record.get('reason')));continue
                    required=plan['required_targets_by_variant'][variant];terms=record['retained_terms'];df=record['native_df_resid'];options=record['covariance_options']
                    check('DAR_outcome_and_alignment_targets_only:'+name,record['dependent']==plan['dependent']and set(required).issubset(terms)and set(terms).issubset(required+controls)and 'net_sentiment'not in terms and 'net_sentiment_lag1'not in terms)
                    check('native_terms_df:'+name,set(current.index)==set(['const']+terms)and df==len(expected)-expected.fund_id.nunique()-len(terms))
                    check('same_fund_FE_two_way_covariance:'+name,record['included_effects']==['Entity']and record['time_effects']is False and all(options.get(k)==v for k,v in dict(cov_type='clustered',clusters=['fund_id','company_id'],debiased=True,auto_df=False,count_effects=True,group_debias=True,cluster_time=False).items()))
                    valid_mask=np.isfinite(current[native_fields].to_numpy(float)).all(axis=1)&current.se_native.gt(0).to_numpy();valid=current.loc[valid_mask];t=valid.beta/valid.se_native;critical=stats.t.ppf(.975,df)
                    check('all_defined_native_t_p_CI:'+name,np.allclose(t,valid.t_native,rtol=args.rtol,atol=1e-7)and np.allclose(2*stats.t.sf(abs(t),df),valid.p_native,rtol=args.rtol,atol=1e-10)and np.allclose(valid.beta-critical*valid.se_native,valid.ci95_native_low,rtol=args.rtol,atol=1e-7)and np.allclose(valid.beta+critical*valid.se_native,valid.ci95_native_high,rtol=args.rtol,atol=1e-7))
                    if(~valid_mask).any():warnings.append(dict(model=name,kind='UNDEFINED_COEFFICIENT_INFERENCE',terms=current.index[~valid_mask].tolist()))
                    if min(record['funds'],record['companies'])<30:warnings.append(dict(model=name,kind='LOW_CLUSTER_COUNT',funds=record['funds'],companies=record['companies']))
                    if record.get('overall_robust_F_inference',{}).get('status')=='UNDEFINED_COVARIANCE':warnings.append(dict(model=name,kind='UNDEFINED_OVERALL_ROBUST_F',native=record['overall_robust_F_inference']))
                    inference[name]={target:'FINITE'if target in valid.index else'UNDEFINED'for target in required}
                    if variant=='CURRENT_PLUS_LAG1':
                        test=summary['distributed_tests'][name];block=np.asarray(test['target_covariance'],float);betas=current.loc[required,'beta'].to_numpy(float)
                        verify_tests(name,betas,block,df,test,'mechanical_')
                    if name in args.models:datasets[name]=(info['parquet'],record)
                check('current_and_lag_variant_identical_horizon_rows:'+cut['tag']+':'+str(horizon)+':'+base_name,len(pair)==2 and pair[0]==pair[1])
    check('unique_horizon_sample_file_count',summary['unique_horizon_sample_files']==len(verified_samples))
    check('unique_horizon_variant_design_count',summary['unique_horizon_sample_designs']==len(seen_designs))
    for name in args.models:
        if name not in datasets:details[name]=dict(numerical_replication_status='NOT_AVAILABLE');warnings.append(dict(model=name,kind='NUMERICAL_REPLICA_NOT_ESTIMATED'));continue
        path,record=datasets[name];sample=pd.read_parquet(path);terms=record['retained_terms'];required=['align_current','align_lag1'];result=within_ols(sample,terms,outcome=plan['dependent']);native=coef.loc[coef.model.eq(name)].set_index('term');begin=len(checks)
        check('independent_df:'+name,result['df_resid']==record['native_df_resid']);check('independent_within_R2:'+name,close(result['within_R2'],record['rsquared_within'],1e-10))
        for target in required:
            j=terms.index(target);beta=float(result['beta'][j]);variance=float(result['covariance'][j,j]);se=float(np.sqrt(variance))if variance>0 else np.nan;found=native.loc[target]
            check('independent_alignment_beta:'+name+':'+target,close(beta,found.beta,1e-8),independent=beta,native=found.beta)
            if np.isfinite(se):
                t=beta/se;p=2*stats.t.sf(abs(t),result['df_resid']);critical=stats.t.ppf(.975,result['df_resid'])
                check('independent_CGM_SE:'+name+':'+target,close(se,found.se_native),independent=se,native=found.se_native)
                check('independent_native_t_p_CI:'+name+':'+target,close(t,found.t_native)and close(p,found.p_native,1e-10)and close(beta-critical*se,found.ci95_native_low)and close(beta+critical*se,found.ci95_native_high))
            else:check('undefined_alignment_variance_preserved:'+name+':'+target,not np.isfinite(found.se_native)or found.se_native<=0)
        indices=[terms.index(target)for target in required];block=result['covariance'][np.ix_(indices,indices)];betas=result['beta'][indices];saved=summary['distributed_tests'][name]
        check('independent_target_covariance_cross_term:'+name,np.allclose(block,np.asarray(saved['target_covariance'],float),rtol=args.rtol,atol=1e-8,equal_nan=True));verify_tests(name,betas,block,result['df_resid'],saved,'independent_')
        details[name]={key:value for key,value in result.items()if key not in['beta','covariance']};details[name].update(coefficients=dict(zip(terms,result['beta'])),numerical_replication_status='PASS'if all(c['status']=='PASS'for c in checks[begin:])else'FAIL')
    for key,value in before.items():check('frozen_source_preserved_after_QA:'+key,sha(plan[key])==value)
    for path,value in raw_hashes.items():check('spot_checked_raw_source_unchanged:'+path,sha(path)==value)
    details['data_scope']=dict(source_rows=len(source),future_rows=len(future),no_trade_source_rows=int(source.delta_quantity_units.eq(0).sum()),eligible_by_horizon={str(h):int((outcome_eligible&future.horizon.eq(h)).sum())for h in[1,2,3]},unique_price_windows=len(unique),raw_price_windows_spot_checked=len(selected),all_model_labels=1080,unique_horizon_sample_files_verified=len(verified_samples))
    failed=sum(c['status']=='FAIL'for c in checks)
    result=dict(status='FAIL'if failed else('PASS_WITH_WARNINGS'if warnings else'PASS'),qa_version=2,literal_date_metadata_restored_before_exact_comparison=True,created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),plan_sha256=sha(planpath),pass_count=len(checks)-failed,fail_count=failed,checks=checks,warnings=warnings,native_inference_status=inference,details=details,independent_replication_models=args.models,mechanical_checks_only_models=[name for name in records if name not in args.models],limitations=['Supplementary position-level cash-excess clean-price proxy, not fund alpha, fund NTS or actual trade PnL.','Clean prices and estimated quantities are not adjusted for coupons, distributions, amortization or corporate actions. Known VTBR scale change remains raw and is separately flagged.','Cash benchmark endpoints follow frozen report/target dates; asset prices are permitted as-of observations up to seven days earlier.','Both outcome and alignment contain current trade direction; the design does not separately identify fund trading skill or causal news effects.','Outcome calibration uses unique eligible price windows in the existing P99 quantity baseline, before additional quantity, fund/support and lag gates.','Only six named distributed models are independently numerically replicated; other labels receive exact-mask/data/native-arithmetic checks.'])
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(dict(status=result['status'],pass_count=result['pass_count'],fail_count=failed)))


def fixture():
    rng=np.random.default_rng(3811);n=2400;fund=np.repeat(np.arange(24),100);current=rng.normal(size=n);lag=rng.normal(size=n)
    frame=pd.DataFrame(dict(fund_id=fund,company_id=np.tile(np.arange(12),200),align_current=current,align_lag1=lag,quantity_change_pct=rng.normal(size=n),dar_q_cash_price_pp=.7*current-.3*lag+fund*.4+rng.normal(scale=.05,size=n)))
    result=within_ols(frame,['align_current','align_lag1']);assert np.allclose(result['beta'],[.7,-.3],atol=.01)
    assert result['df_resid']==2374
    assert pd.Timestamp('2020-01-31')+pd.DateOffset(months=1)==pd.Timestamp('2020-02-29')
    assert np.isclose(100*(-1)*((95/100-1)-.01),6.)
    delta=pd.Series([0.,2.,-3.]);alignment=(np.sign(delta)*.4).where(delta.ne(0));assert pd.isna(alignment.iloc[0])and np.isclose(alignment.iloc[1],.4)and np.isclose(alignment.iloc[2],-.4)
    literal=pd.DataFrame(dict(horizon=[1,3],row_id=['a','a'],future_target_date=[dt.date(2020,2,29),dt.date(2020,4,30)],p0_quote_date=[dt.date(2020,1,31)]*2,future_quote_date=[dt.date(2020,2,28),dt.date(2020,4,30)]))
    normalized=literal.iloc[::-1].reset_index(drop=True).copy()
    for field in LITERAL_DATE_FIELDS:normalized[field]=pd.to_datetime(normalized[field])
    assert isinstance(normalized.future_target_date.iloc[0],pd.Timestamp)
    restored=restore_literal_date_columns(normalized,literal)
    assert all(type(restored[field].iloc[0])is dt.date for field in LITERAL_DATE_FIELDS)
    assert equal_values(restored,literal.iloc[::-1].reset_index(drop=True))
    for field in LITERAL_DATE_FIELDS:assert np.array_equal(pd.to_datetime(restored[field]).to_numpy(dtype='datetime64[ns]'),normalized[field].to_numpy(dtype='datetime64[ns]'))
    beta=np.array([.8,-.3]);cov=np.array([[.16,.02],[.02,.09]]);assert np.isclose(cov.sum(),.29)and np.isclose(beta@np.linalg.solve(cov,beta),5.828571428571429)
    print(json.dumps(dict(status='PASS',scope='synthetic_only',checks=7)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan');parser.add_argument('--root');parser.add_argument('--output');parser.add_argument('--models',nargs='+',default=REPLICAS);parser.add_argument('--rtol',type=float,default=1e-5);parser.add_argument('--fixture-only',action='store_true');args=parser.parse_args();fixture()if args.fixture_only else audit(args)

