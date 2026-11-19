"""Independent QA of exact previous-report-window sentiment and paired H1 models.

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


def within_ols(frame, terms):
    """Fund-demeaned OLS and independent two-way CGM sandwich."""
    groups = frame.fund_id.astype(str)
    x = frame[terms].astype(float)
    y = frame.quantity_change_pct.astype(float)
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


REPLICAS=[f"FOCUS_{group}_ROWS_60_INTERVALS_24__{variant}" for group in ["EQUITY","BOND","MIXED"] for variant in ["CURRENT_COMMON","CURRENT_PLUS_LAG1"]]


def equal_values(left,right):
    try:
        pd.testing.assert_frame_equal(left.reset_index(drop=True),right.reset_index(drop=True),check_exact=True,check_dtype=False)
        return True
    except AssertionError:return False


def raw_counts(events,company,start,end):
    part=events.loc[events.company_id.eq(company)&events.event_date.gt(start)&events.event_date.le(end)]
    relevant=truth(part.valid)&part.relevance.eq('relevant')
    n=len(part);valid=int(truth(part.valid).sum());nr=int(relevant.sum())
    counts=dict(n_candidates=n,n_attempted=int(truth(part.attempted).sum()),n_valid=valid,n_relevant=nr,
        n_positive=int((relevant&part.sentiment_label.eq('positive')).sum()),
        n_negative=int((relevant&part.sentiment_label.eq('negative')).sum()),
        n_neutral=int((relevant&part.sentiment_label.eq('neutral')).sum()))
    eligible=n>0 and valid==n and nr>0
    counts.update(n_members=n,n_errors_or_unattempted=n-valid,n_irrelevant=valid-nr,
        n_error=counts['n_attempted']-valid,n_unattempted=n-counts['n_attempted'])
    state='no_title_candidates'if n==0 else('partial_or_error_coverage'if valid!=n else('no_relevant_news'if nr==0 else'complete_relevant_news'))
    observed=(counts['n_positive']-counts['n_negative'])/nr if nr else np.nan
    return part,dict(**counts,eligible_primary=eligible,strict_complete=n>0 and n==valid,
        net_sentiment=observed if eligible else np.nan,net_sentiment_observed=observed,state=state,
        sample_hash=hashlib.sha256('\n'.join(sorted(part.pair_id.astype(str))).encode()).hexdigest()if n else None)


def audit(args):
    if os.name!='nt':raise RuntimeError('Actual research-data QA must run on research-host')
    from scipy import stats
    planpath=Path(args.plan);plan=json.loads(planpath.read_text(encoding='utf-8'))
    root=Path(args.root or plan['output_root']);output=Path(args.output or root/'qa_independent.json')
    if output.exists():raise FileExistsError('Refusing to overwrite an existing QA report')
    if 'databaseresources'in[p.lower()for p in output.parts]:raise ValueError('Protected output')
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    coef=pd.read_csv(root/'coefficients.csv');prior=json.loads(Path(plan['focus_summary']).read_text(encoding='utf-8'))
    checks,warnings,details=[],[],{}
    def check(name,passed,**evidence):checks.append(dict(name=name,status='PASS'if bool(passed)else'FAIL',**clean(evidence)))
    def close(a,b,atol=1e-7):return bool(np.isclose(float(a),float(b),rtol=args.rtol,atol=atol,equal_nan=False))
    def same_num(a,b,atol=0.):
        left=pd.to_numeric(pd.Series(a),errors='coerce').to_numpy(dtype=float,na_value=np.nan)
        right=pd.to_numeric(pd.Series(b),errors='coerce').to_numpy(dtype=float,na_value=np.nan)
        return bool(np.allclose(left,right,rtol=0,atol=atol,equal_nan=True))
    check('frozen_plan_hash',sha(planpath)==summary['plan_sha256'])
    check('same_specification',summary.get('specification')==plan)
    check('predeclared_independent_models',args.models==plan['independent_qa_models'])
    check('unadjusted_inference_only',plan.get('multiple_testing_adjustments')is False and set(coef.columns)=={'model','outcome','term','beta','se_native','t_native','p_native','ci95_native_low','ci95_native_high'})
    check('fund_FE_no_time_cluster',plan['fund_FE']is True and plan['time_FE']is False and plan['cluster_time']is False and plan['clusters']==['fund_id','company_id'])
    before={}
    for key,expected in plan['expected_source_sha256'].items():
        before[key]=sha(plan[key]);check('frozen_source:'+key,before[key]==expected,sha256=before[key])
    source=pd.read_parquet(plan['input_panel']).sort_values(['fund_id','end_date','security_id','interval_id'],kind='stable').reset_index(drop=True)
    mapping=pd.read_parquet(plan['lag_map']);windows=pd.read_parquet(plan['lag_windows'])
    intervals=pd.read_parquet(plan['interval_source'])
    original_net=pd.read_parquet(plan['net_source']);actual=original_net.loc[original_net.period_kind.eq('actual_report_interval')].copy()
    for frame,start,end in [(source,'start_date','end_date'),(intervals,'start_date','end_date'),(windows,'period_start','period_end'),(actual,'period_start','period_end')]:
        frame[start]=pd.to_datetime(frame[start]);frame[end]=pd.to_datetime(frame[end])
    check('source_rows_unique',len(source)==plan['expected_primary_rows']and source.row_id.notna().all()and not source.row_id.duplicated().any())
    check('lagmap_exact_source_ids',set(mapping.row_id)==set(source.row_id)and mapping.row_id.notna().all()and not mapping.row_id.duplicated().any())
    m=mapping.set_index('row_id').reindex(source.row_id).reset_index()
    for field,mapped in [('fund_id','fund_id'),('company_id','company_id'),('interval_id','current_interval_id')]:check('map_current_identity:'+field,np.array_equal(source[field].astype(str),m[mapped].astype(str)))
    for field,mapped in [('start_date','current_start_date'),('end_date','current_end_date')]:check('map_current_date:'+field,np.array_equal(source[field].to_numpy(dtype='datetime64[ns]'),pd.to_datetime(m[mapped]).to_numpy(dtype='datetime64[ns]')))
    check('full_calendar_unique_interval_ids',not intervals.interval_id.duplicated().any())
    check('full_calendar_unambiguous_previous_end',not intervals.duplicated(['fund_id','end_date']).any())
    calendar_current=source[['row_id','fund_id','interval_id']].merge(intervals[['fund_id','interval_id','start_date','end_date','opening_report_id','closing_report_id']],on=['fund_id','interval_id'],how='left',validate='many_to_one',sort=False)
    for field in ['start_date','end_date','opening_report_id','closing_report_id']:check('current_full_calendar_binding:'+field,np.array_equal(calendar_current[field].astype(str),source[field].astype(str)))
    previous=intervals[['fund_id','interval_id','start_date','end_date','opening_report_id','closing_report_id']].rename(columns={c:'previous_'+c for c in ['interval_id','start_date','end_date','opening_report_id','closing_report_id']})
    expected=source[['row_id','fund_id','company_id','start_date','opening_report_id']].merge(previous,left_on=['fund_id','start_date'],right_on=['fund_id','previous_end_date'],how='left',validate='many_to_one',sort=False)
    has_previous=expected.previous_interval_id.notna()
    check('previous_calendar_report_chain',expected.loc[has_previous,'previous_closing_report_id'].astype(str).eq(expected.loc[has_previous,'opening_report_id'].astype(str)).all())
    for field,mapped in [('previous_interval_id','lag_interval_id'),('previous_opening_report_id','lag_opening_report_id'),('previous_closing_report_id','lag_closing_report_id')]:
        check('independent_predecessor:'+field,expected[field].fillna('<NA>').astype(str).equals(m[mapped].fillna('<NA>').astype(str)))
    for field,mapped in [('previous_start_date','lag_start_date'),('previous_end_date','lag_end_date')]:check('independent_predecessor_dates:'+field,np.array_equal(pd.to_datetime(expected[field]).to_numpy(dtype='datetime64[ns]'),pd.to_datetime(m[mapped]).to_numpy(dtype='datetime64[ns]'),equal_nan=True))
    check('no_future_or_overlapping_lag',expected.loc[has_previous,'previous_start_date'].lt(expected.loc[has_previous,'previous_end_date']).all()and expected.loc[has_previous,'previous_end_date'].eq(source.loc[has_previous,'start_date']).all())
    keys=['company_id','period_start','period_end']
    check('actual_net_unique_company_window',not actual.duplicated(keys).any())
    check('lag_windows_actual_only_unique',windows.period_kind.eq('actual_report_interval').all()and not windows.duplicated(keys).any()and not windows.company_period_id.duplicated().any())
    requested=expected.loc[has_previous,['company_id','previous_start_date','previous_end_date']].rename(columns={'previous_start_date':'period_start','previous_end_date':'period_end'}).drop_duplicates()
    check('lag_window_keys_exact_requested',set(map(tuple,requested[keys].itertuples(index=False,name=None)))==set(map(tuple,windows[keys].itertuples(index=False,name=None))))
    existing=windows.merge(actual[keys+['company_period_id']],on=keys,how='left',suffixes=('','_old'),validate='one_to_one',sort=False)
    is_existing=existing.company_period_id_old.notna();new=existing.loc[~is_existing].copy()
    check('exactly_eighty_five_missing_windows',len(new)==85,actual=len(new))
    check('existing_source_kind',existing.loc[is_existing,'lag_source_kind'].eq('existing_actual_window').all())
    check('recomputed_source_kind',new.lag_source_kind.eq('recomputed_missing_actual_window').all())
    check('existing_company_period_ids_verbatim',existing.loc[is_existing,'company_period_id'].eq(existing.loc[is_existing,'company_period_id_old']).all())
    matched=actual.set_index('company_period_id').reindex(existing.loc[is_existing,'company_period_id'])
    for field in ['net_sentiment','net_sentiment_observed','classification_coverage','n_candidates','n_attempted','n_valid','n_relevant','n_positive','n_negative','n_neutral','n_members']:
        check('existing_scores_counts_verbatim:'+field,same_num(existing.loc[is_existing,field],matched[field]))
    for field in ['state','eligible_primary','strict_complete','sample_hash']:
        check('existing_window_metadata_verbatim:'+field,np.array_equal(existing.loc[is_existing,field].fillna('<NA>').astype(str),matched[field].fillna('<NA>').astype(str)))
    events=pd.read_parquet(plan['canonical_events']);events.event_date=pd.to_datetime(events.event_date,errors='coerce')
    members_path=Path(plan['missing_window_members']);members=pd.read_parquet(members_path);members.event_date=pd.to_datetime(members.event_date)
    check('canonical_pairs_unique',not events.pair_id.duplicated().any()and not events.duplicated(['article_id','company_id']).any())
    if 'is_representative'in events:check('canonical_representatives_only',truth(events.is_representative).all())
    check('new_window_members_unique',not members.duplicated(['company_period_id','pair_id']).any()and set(members.company_period_id).issubset(set(new.company_period_id)))
    for _,window in new.iterrows():
        cid=window.company_period_id;part,counts=raw_counts(events,window.company_id,window.period_start,window.period_end)
        expected_id=hashlib.sha256(('actual_report_interval\t'+str(window.company_id)+'\t'+window.period_start.strftime('%Y-%m-%d')+'\t'+window.period_end.strftime('%Y-%m-%d')).encode()).hexdigest()
        check('recomputed_exact_window_id:'+cid,cid==expected_id)
        chosen=members.loc[members.company_period_id.eq(cid)]
        check('raw_canonical_membership:'+cid,len(chosen)==len(part)and set(chosen.pair_id)==set(part.pair_id))
        for field in ['n_candidates','n_attempted','n_valid','n_relevant','n_positive','n_negative','n_neutral','n_errors_or_unattempted','n_irrelevant','n_error','n_unattempted','n_members']:check('raw_counts:'+cid+':'+field,int(window[field])==counts[field])
        check('raw_primary_score:'+cid,same_num([window.net_sentiment],[counts['net_sentiment']])and bool(window.eligible_primary)==counts['eligible_primary']and bool(window.strict_complete)==counts['strict_complete']and window.state==counts['state'])
        digest=None if pd.isna(window.sample_hash)else window.sample_hash
        check('raw_member_hash:'+cid,digest==counts['sample_hash'])
        if len(chosen):
            canonical=part.set_index('pair_id').reindex(chosen.pair_id)
            check('member_source_attributes:'+cid,all(np.array_equal(chosen[field].fillna('<NA>').astype(str),canonical[field].fillna('<NA>').astype(str))for field in ['article_id','company_id','event_date','attempted','valid','relevance','sentiment_label']))
            check('member_denominator_and_boundaries:'+cid,np.array_equal(truth(chosen.included_in_net_denominator),truth(chosen.valid)&chosen.relevance.eq('relevant'))and chosen.event_date.gt(window.period_start).all()and chosen.event_date.le(window.period_end).all())
    overlap_path=Path(plan['lag_data_audit']).parent/'overlap_parity.json';overlap=json.loads(overlap_path.read_text(encoding='utf-8'))
    overlap_ids=overlap['company_period_ids'];check('twenty_overlap_windows',len(overlap_ids)==20 and overlap['status']=='PASS')
    for cid in overlap_ids:
        window=actual.set_index('company_period_id').loc[cid];_,counts=raw_counts(events,window.company_id,window.period_start,window.period_end)
        check('overlap_raw_counts_score_and_hash:'+cid,all(int(window[field])==counts[field]for field in ['n_candidates','n_attempted','n_valid','n_relevant','n_positive','n_negative','n_neutral'])and same_num([window.net_sentiment],[counts['net_sentiment']])and(None if pd.isna(window.sample_hash)else window.sample_hash)==counts['sample_hash'])
    # Join by exact bounds, independently of holdings or filtered-row ordering.
    joined=expected.merge(windows,left_on=['company_id','previous_start_date','previous_end_date'],right_on=keys,how='left',validate='many_to_one',sort=False)
    eligible=has_previous&truth(joined.eligible_primary)&np.isfinite(pd.to_numeric(joined.net_sentiment,errors='coerce'))&pd.to_numeric(joined.n_relevant,errors='coerce').gt(0)
    expected_score=pd.to_numeric(joined.net_sentiment,errors='coerce').where(eligible)
    check('independent_lag_eligibility',np.array_equal(truth(m.lag_eligible_primary),eligible))
    check('independent_lag_score_no_fill',same_num(m.net_sentiment_lag1,expected_score))
    check('lag_window_ids',joined.company_period_id.fillna('<NA>').astype(str).equals(m.lag_company_period_id.fillna('<NA>').astype(str)))
    check('lag_issuer_identity',m.loc[has_previous,'lag_company_id'].astype(str).eq(source.loc[has_previous,'company_id'].astype(str)).all())
    expected_kind=joined.lag_source_kind.where(has_previous,'no_previous_interval')
    expected_reason=joined.state.where(~eligible,'eligible').where(has_previous,'no_previous_interval')
    check('lag_provenance_kind',expected_kind.fillna('<NA>').astype(str).equals(m.lag_source_kind.fillna('<NA>').astype(str)))
    check('lag_missing_reasons',expected_reason.fillna('<NA>').astype(str).equals(m.lag_reason.fillna('<NA>').astype(str)))
    for field in ['n_candidates','n_attempted','n_valid','n_relevant','n_positive','n_negative','n_neutral','n_errors_or_unattempted','n_irrelevant','n_error','n_unattempted','n_undated_company_candidates']:
        check('lag_row_counts:'+field,same_num(m['lag_'+field],joined[field]))
    check('lag_row_member_hash',m.lag_window_sample_hash.fillna('<NA>').astype(str).equals(joined.sample_hash.fillna('<NA>').astype(str)))
    valid_map=m.loc[eligible].copy();valid_map['score']=expected_score.loc[eligible].to_numpy()
    check('lag_signal_shared_by_exact_lag_key',valid_map.groupby('lag_company_period_id').score.nunique().eq(1).all())
    check('lag_signal_same_within_fund_issuer_interval',valid_map.groupby(['fund_id','company_id','current_interval_id']).score.nunique().eq(1).all())
    full=source.copy()
    for field in m:
        if field.startswith('lag_')or field=='net_sentiment_lag1':full[field]=m[field].to_numpy()
    full['complete_current_lag1']=np.asarray(eligible,bool)
    support=source.groupby(['fund_focus','fund_id'],observed=True).agg(fund_rows=('row_id','size'),fund_intervals=('interval_id','nunique'),fund_end_dates=('end_date','nunique'))
    saved_support=pd.read_csv(plan['focus_support'],dtype={'fund_id':str}).set_index(['fund_focus','fund_id'])
    for field in support:check('old_qualification_support_fixed:'+field,np.array_equal(saved_support.reindex(support.index)[field],support[field]))
    old_masks={'REFERENCE_AUGMENTED_P99':pd.Series(True,index=source.index)}
    for group in plan['focus_groups']:
        belongs=source.fund_focus.eq(group);old_masks['FOCUS_'+group.upper()+'_BASELINE']=belongs;sg=support.loc[group]
        for rows in [30,60]:
            for periods in [12,24,36]:old_masks[f'FOCUS_{group.upper()}_ROWS_{rows}_INTERVALS_{periods}']=belongs&source.fund_id.isin(sg.index[sg.fund_rows.ge(rows)&sg.fund_intervals.ge(periods)])
    records={r['model']:r for r in summary['models']};controls=plan['baseline_controls']+plan['macro_controls']
    check('all_seventy_two_models',set(records)==set(plan['planned_models'])=={name+'__'+v for name in old_masks for v in ['CURRENT_COMMON','CURRENT_PLUS_LAG1']})
    check('unique_coefficient_keys',not coef.duplicated(['model','term']).any())
    native_fields=['beta','se_native','t_native','p_native','ci95_native_low','ci95_native_high'];datasets={};seen={};inference={}
    def check_distributed(name,beta,cov,df,test,prefix):
        cov=np.asarray(cov,float);beta=np.asarray(beta,float)
        check(prefix+'target_terms:'+name,test['target_terms']==['net_sentiment','net_sentiment_lag1'])
        check(prefix+'no_covariance_repair:'+name,test['covariance_repair']is False)
        joint=test['joint_current_lag_zero'];total=test['sum_current_lag']
        check(prefix+'joint_chi_square_two_df:'+name,joint['df']==2 and 'chi2'in joint['distribution'].lower())
        if np.isfinite(cov).all()and np.isfinite(beta).all():
            eig=np.linalg.eigvalsh((cov+cov.T)/2)
            check(prefix+'cov_eigenvalues:'+name,same_num(joint['target_covariance_eigenvalues'],eig,1e-9))
            if eig.min()>0:
                w=float(beta@np.linalg.solve(cov,beta));p=float(stats.chi2.sf(w,2))
                check(prefix+'joint_Wald:'+name,joint['status']=='ESTIMATED'and joint['df']==2 and close(w,joint['statistic'])and close(p,joint['p_value'],1e-10),independent_stat=w,independent_p=p)
            else:check(prefix+'joint_undefined_nonPD:'+name,joint['status']!='ESTIMATED'and joint['p_value']is None)
            variance=float(cov.sum());value=float(beta.sum())
            check(prefix+'sum_beta_crosscov_variance:'+name,close(total['beta'],value)and close(total['variance'],variance)and close(test['cov_current_lag'],cov[0,1])and close(test['var_current'],cov[0,0])and close(test['var_lag'],cov[1,1]))
            if variance>0 and df>0:
                se=float(np.sqrt(variance));critical=stats.t.ppf(.975,df);p=2*stats.t.sf(abs(value/se),df)
                check(prefix+'sum_native_t_CI:'+name,total['status']=='ESTIMATED'and total['native_df_resid']==df and close(total['se_native'],se)and close(total['ci95_native_low'],value-critical*se)and close(total['ci95_native_high'],value+critical*se)and close(total['p_native'],p,1e-10))
            else:check(prefix+'sum_undefined_nonpositive_variance:'+name,total['status']!='ESTIMATED'and total['p_native']is None)
        else:check(prefix+'nonfinite_covariance_undefined:'+name,joint['status']!='ESTIMATED'and total['status']!='ESTIMATED')
    for oldname,oldmask in old_masks.items():
        check('old_mask_exact_before_lag:'+oldname,row_hash(source.loc[oldmask])==prior['samples'][oldname]['row_hash']and int(oldmask.sum())==prior['samples'][oldname]['rows'])
        expected=full.loc[oldmask&eligible];pair=[]
        for variant in ['CURRENT_COMMON','CURRENT_PLUS_LAG1']:
            name=oldname+'__'+variant;record=records[name];info=summary['samples'][name];sample=pd.read_parquet(info['parquet']);pair.append(sample.row_id)
            distributed=variant=='CURRENT_PLUS_LAG1';required=['net_sentiment','net_sentiment_lag1']if distributed else['net_sentiment']
            check('exact_common_rows_no_requalification:'+name,sample.row_id.reset_index(drop=True).equals(expected.row_id.reset_index(drop=True)))
            check('source_Y_X_Q_preserved:'+name,equal_values(sample[list(source.columns)],expected[list(source.columns)]))
            lagcols=[c for c in full if c.startswith('lag_')or c in['net_sentiment_lag1','complete_current_lag1']]
            check('sample_lag_provenance_preserved:'+name,equal_values(sample[lagcols],expected[lagcols]))
            check('sample_metadata_N_hash:'+name,len(sample)==info['rows']and row_hash(sample)==info['row_hash'])
            check('model_N_funds_companies:'+name,record['nobs']==len(sample)and record['funds']==sample.fund_id.nunique()and record['companies']==sample.company_id.nunique())
            check('old_support_qualifiers_not_recounted:'+name,record['support_qualifiers_fixed_before_lag_drop']is True and record['old_nobs']==int(oldmask.sum())and record['n_lag_missing']==int((oldmask&~eligible).sum()))
            key=(variant,info['row_hash']);current=coef.loc[coef.model.eq(name)].set_index('term')
            if key in seen:
                check('same_variant_only_alias:'+name,record['exact_common_sample_alias_of']==seen[key])
                representative=coef.loc[coef.model.eq(seen[key])].set_index('term')
                check('alias_native_coefficients:'+name,set(current.index)==set(representative.index)and np.allclose(current.reindex(representative.index)[native_fields],representative[native_fields],rtol=1e-12,atol=1e-10,equal_nan=True))
            else:seen[key]=name
            if record['status']!='ESTIMATED':
                check('not_estimated_no_coefficients:'+name,len(current)==0);inference[name]=dict(status='NOT_ESTIMATED',reason=record.get('reason'));warnings.append(dict(model=name,kind='NOT_ESTIMATED',reason=record.get('reason')));continue
            terms=record['retained_terms'];df=record['native_df_resid'];options=record['covariance_options']
            check('required_sentiment_targets_retained:'+name,set(required).issubset(terms)and set(terms).issubset(required+controls)and(not distributed or len(required)==2))
            check('native_terms_df:'+name,set(current.index)==set(['const']+terms)and df==len(sample)-sample.fund_id.nunique()-len(terms))
            check('unchanged_fund_FE_covariance:'+name,record['included_effects']==['Entity']and record['time_effects']is False and all(options.get(k)==v for k,v in dict(cov_type='clustered',clusters=['fund_id','company_id'],debiased=True,auto_df=False,count_effects=True,group_debias=True,cluster_time=False).items()))
            defined=np.isfinite(current[native_fields].to_numpy(float)).all(axis=1)&current.se_native.gt(0).to_numpy();valid=current.loc[defined]
            t=valid.beta/valid.se_native;critical=stats.t.ppf(.975,df)
            check('all_defined_native_t_p_CI:'+name,np.allclose(t,valid.t_native,rtol=args.rtol,atol=1e-7)and np.allclose(2*stats.t.sf(abs(t),df),valid.p_native,rtol=args.rtol,atol=1e-10)and np.allclose(valid.beta-critical*valid.se_native,valid.ci95_native_low,rtol=args.rtol,atol=1e-7)and np.allclose(valid.beta+critical*valid.se_native,valid.ci95_native_high,rtol=args.rtol,atol=1e-7))
            if(~defined).any():warnings.append(dict(model=name,kind='UNDEFINED_COEFFICIENT_INFERENCE',terms=current.index[~defined].tolist()))
            if min(record['funds'],record['companies'])<30:warnings.append(dict(model=name,kind='LOW_CLUSTER_COUNT',funds=record['funds'],companies=record['companies']))
            inference[name]={term:'FINITE'if term in valid.index else'UNDEFINED'for term in required}
            if distributed:
                test=summary['distributed_tests'][name];cov=np.asarray(test['target_covariance'],float);beta=current.loc[required,'beta'].to_numpy(float)
                check('native_covariance_target_diagonal:'+name,all(not np.isfinite(current.loc[term,'se_native'])or close(current.loc[term,'se_native']**2,cov[j,j])for j,term in enumerate(required)))
                check_distributed(name,beta,cov,df,test,'mechanical_')
                if test['joint_current_lag_zero']['status']!='ESTIMATED':warnings.append(dict(model=name,kind='UNDEFINED_JOINT_TARGET_INFERENCE'))
                if test['sum_current_lag']['status']!='ESTIMATED':warnings.append(dict(model=name,kind='UNDEFINED_SUM_INFERENCE'))
            if name in args.models:datasets[name]=sample
        check('paired_models_identical_ordered_rows:'+oldname,pair[0].equals(pair[1]))
    for name in args.models:
        if name not in datasets:details[name]=dict(numerical_replication_status='NOT_AVAILABLE');warnings.append(dict(model=name,kind='NUMERICAL_REPLICA_NOT_ESTIMATED'));continue
        sample,record=datasets[name],records[name];terms=record['retained_terms'];result=within_ols(sample,terms);native=coef.loc[coef.model.eq(name)].set_index('term');required=['net_sentiment','net_sentiment_lag1']if name.endswith('CURRENT_PLUS_LAG1')else['net_sentiment']
        begin=len(checks);check('independent_df:'+name,result['df_resid']==record['native_df_resid']);check('independent_within_R2:'+name,close(result['within_R2'],record['rsquared_within'],1e-10))
        for term in required:
            j=terms.index(term);beta=float(result['beta'][j]);var=float(result['covariance'][j,j]);se=float(np.sqrt(var))if var>0 else np.nan;target=native.loc[term]
            check('independent_beta:'+name+':'+term,close(beta,target.beta,1e-8),independent=beta,native=target.beta)
            if np.isfinite(se):
                t=beta/se;p=2*stats.t.sf(abs(t),result['df_resid']);critical=stats.t.ppf(.975,result['df_resid'])
                check('independent_CGM_SE:'+name+':'+term,close(se,target.se_native),independent=se,native=target.se_native)
                check('independent_native_t_p_CI:'+name+':'+term,close(t,target.t_native)and close(p,target.p_native,1e-10)and close(beta-critical*se,target.ci95_native_low)and close(beta+critical*se,target.ci95_native_high))
            else:check('independent_undefined_target_variance:'+name+':'+term,not np.isfinite(target.se_native))
        if len(required)==2:
            indices=[terms.index(t)for t in required];block=result['covariance'][np.ix_(indices,indices)];betas=result['beta'][indices];saved=summary['distributed_tests'][name]
            check('independent_full_target_covariance:'+name,np.allclose(block,saved['target_covariance'],rtol=args.rtol,atol=1e-8))
            check_distributed(name,betas,block,result['df_resid'],saved,'independent_')
        details[name]={key:value for key,value in result.items()if key not in['beta','covariance']};details[name].update(coefficients=dict(zip(terms,result['beta'])),numerical_replication_status='PASS'if all(c['status']=='PASS'for c in checks[begin:])else'FAIL')
    for key,value in before.items():check('source_unchanged_after_QA:'+key,sha(plan[key])==value)
    details['lag_data']=dict(source_rows=len(source),previous_interval_absent=int((~has_previous).sum()),required_company_windows=len(windows),existing_windows=int(is_existing.sum()),recomputed_windows=len(new),recomputed_member_rows=len(members),eligible_lag_rows=int(eligible.sum()),missing_lag_rows=int((~eligible).sum()),genuine_zero_lag_rows=int((eligible&expected_score.eq(0)).sum()),conditional_entries_with_eligible_lag=int((eligible&truth(source.entry_zero_to_positive)).sum()),support_qualification_before_drop=True)
    failed=sum(c['status']=='FAIL'for c in checks)
    result=dict(status='FAIL'if failed else('PASS_WITH_WARNINGS'if warnings else'PASS'),created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),plan_sha256=sha(planpath),pass_count=len(checks)-failed,fail_count=failed,checks=checks,warnings=warnings,native_inference_status=inference,details=details,independent_replication_models=args.models,mechanical_checks_only_models=[name for name in records if name not in args.models],limitations=['Lag is the previous actual fund-report interval, not necessarily one month; missing or ineligible scores remain absent.','The six named models are numerically replicated by within OLS and CGM covariance; other models receive exact-sample and mechanical inference checks.','Raw canonical membership and counts are independently checked for 85 new windows and 20 source windows, not all historical article labels.','Support qualification is fixed on the previous focus baseline; achieved support can fall after lag-completeness exclusion.','Coefficient sums are linear associations across two exposure windows, not structural long-run multipliers or causal effects.'])
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(dict(status=result['status'],pass_count=result['pass_count'],fail_count=failed)))


def fixture():
    rng=np.random.default_rng(311);n=1200;fund=np.repeat(np.arange(24),50)
    current=rng.normal(size=n);lag=rng.normal(size=n);control=rng.normal(size=n)
    frame=pd.DataFrame(dict(fund_id=fund,company_id=np.tile(np.arange(12),100),net_sentiment=current,net_sentiment_lag1=lag,control=control,quantity_change_pct=.8*current-.3*lag+.2*control+fund*.4+rng.normal(scale=.05,size=n)))
    result=within_ols(frame,['net_sentiment','net_sentiment_lag1','control']);assert np.allclose(result['beta'],[.8,-.3,.2],atol=.01)
    assert result['df_resid']==1173
    b=np.array([.8,-.3]);c=np.array([[.16,.02],[.02,.09]])
    assert np.isclose(c.sum(),.29)and np.isclose(b.sum(),.5)and np.isclose(b@np.linalg.solve(c,b),5.828571428571429)
    events=pd.DataFrame(dict(company_id=['c']*4,event_date=pd.to_datetime(['2020-01-31','2020-02-01','2020-02-29','2020-03-01']),pair_id=['p0','p1','p2','p3'],attempted=[True]*4,valid=[True]*4,relevance=['relevant']*4,sentiment_label=['positive','positive','negative','negative']))
    part,counts=raw_counts(events,'c',pd.Timestamp('2020-01-31'),pd.Timestamp('2020-02-29'))
    assert set(part.pair_id)=={'p1','p2'}and counts['net_sentiment']==0 and counts['eligible_primary']
    print(json.dumps(dict(status='PASS',scope='synthetic_only',checks=4)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan');parser.add_argument('--root');parser.add_argument('--output')
    parser.add_argument('--models',nargs='+',default=REPLICAS);parser.add_argument('--rtol',type=float,default=1e-5);parser.add_argument('--fixture-only',action='store_true');args=parser.parse_args()
    fixture()if args.fixture_only else audit(args)

