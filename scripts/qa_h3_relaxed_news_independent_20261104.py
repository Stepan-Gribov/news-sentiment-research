"""Prepared independent audit primitives for H3 relaxed-news, NOT an executor.

No production module imports or automatic PC/data access. The final run adapter
must bind these primitives to the frozen research spec and sealed new outputs.
The research spec is pinned below; the output adapter awaits a sealed new run.
--self-check exercises only synthetic edge cases.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import linalg, stats

PRIOR_MANIFEST_SHA256 = {
    'panels/nts_ee5a8bf9b450137a/manifest.json': '598e05ce14d7b2942fdc3056633a882feb51c3883aebc9c81e1c4f322db80695',
    'outputs/run_d0440fad412ee5e9/manifest.json': 'd61349f8275762eee3c224a8118d69eec81fb08107d48a6fc00f63b324a09628',
    'outputs/provider_TR4_b00ab703609fb58f/manifest.json': 'af25eb9dc57288ef54d2b0839477f0ced6b9539708bd3a88a8cbd2257b8e98c3',
}
RELAXED_SPEC_SHA256 = '36a64070c5dfcba749d7baa1f15b639f9801a39fcac12142e372a94a2554699f'
FROZEN_BRANCHES = ['UNIT_VALUE_KNOWN_ADJUSTMENTS','TR_PROVIDER_ACCRUED_INCOME_CONVENTION']
FROZEN_VARIANTS = ['CURRENT_ONLY_OWN_AVAILABLE','CURRENT_PLUS_LAGNEWS']


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def classify_news_window(row, *, mapping_known, window_found,
                         date_bounds_verified, undated_policy):
    """Independent raw-count evidence, never fillna(0).

    Research must choose the scope of undated candidates explicitly. Neither
    policy here silently changes the analysis policy before the spec is frozen.
    """
    if undated_policy not in {'dated_archive_scope','block_absence_zero_if_undated'}:
        raise ValueError('Undated-news policy must be explicitly bound to frozen spec')
    def unknown(reason):
        return dict(score=np.nan, known=False, reason=reason, zero_by_absence=False)
    if not mapping_known:
        return unknown('company_mapping_unknown')
    if not window_found:
        return unknown('window_not_found_not_no_news')
    if not date_bounds_verified:
        return unknown('window_dates_unverified')
    names = ['n_candidates','n_valid','n_relevant','n_positive','n_negative','n_neutral','n_errors_or_unattempted','n_undated_company_candidates']
    values = [row.get(k,np.nan) for k in names]
    if not np.isfinite(values).all() or min(values) < 0 or any(float(v)!=int(v) for v in values):
        return unknown('missing_or_invalid_raw_counts')
    c,v,r,p,n,z,error,undated = map(int,values)
    if p+n+z != r or r>v or v>c or error!=c-v:
        return unknown('inconsistent_raw_counts')
    if error>0 or v<c:
        return unknown('failed_or_unattempted_classifications')
    if r==0:
        if undated_policy=='block_absence_zero_if_undated' and undated>0:
            return unknown('undated_candidates_prevent_absence_claim')
        return dict(score=0.,known=True,reason='no_dated_title_candidates' if c==0 else 'all_classified_candidates_irrelevant',zero_by_absence=True)
    return dict(score=(p-n)/r,known=True,reason='complete_classified_relevant_news',zero_by_absence=False,undated_candidate_count=undated)


def classify_zero_movement(values, *, scope_complete):
    """Optional whole-scope absence flag, NEVER an NTS eligibility gate.

    A partial sum cannot prove whole-scope absence; the observed-subportfolio
    score below can still be zero on its own nonempty usable subset.
    """
    q = np.asarray(values,dtype=float)
    if not scope_complete:
        return dict(known_zero=False,reason='position_scope_incomplete')
    if not len(q):
        return dict(known_zero=True,reason='confirmed_empty_position_scope')
    if not np.isfinite(q).all():
        return dict(known_zero=False,reason='unknown_quantity_or_amount')
    return dict(known_zero=bool(np.all(q==0)),reason='all_measured_zero' if np.all(q==0) else 'measured_nonzero_movement')


def expected_model_sample(panel, *, outcome, current, lag, distributed,
                          universe_column, group=None):
    """Current-only eligibility does not consult the lag or old paired gate."""
    needed = [outcome,'past_alpha',current] + ([lag] if distributed else [])
    finite = np.isfinite(panel[needed].apply(pd.to_numeric,errors='coerce').to_numpy(float)).all(axis=1)
    keep = panel[universe_column].eq(True) & finite
    if group is not None and group!='pooled':
        keep &= panel.fund_focus.eq(group)
    before = panel.loc[keep].sort_values(['fund_id','end_date','interval_id']).reset_index(drop=True)
    if before.duplicated(['fund_id','interval_id']).any():
        raise ValueError('Duplicate fund/interval in one-horizon model sample')
    singleton = before.groupby('fund_id').interval_id.transform('size').lt(2)
    return before.loc[~singleton].reset_index(drop=True), before.loc[singleton].reset_index(drop=True)


def ordered_sample_sha256(frame):
    values = frame.fund_id.astype(str)+'\t'+frame.interval_id.astype(str)
    return hashlib.sha256('\n'.join(values).encode()).hexdigest()


def planned_cells(outcomes):
    if len(outcomes)!=2 or len(set(outcomes))!=2:
        raise ValueError('Two distinct outcome labels must come from the frozen spec')
    return list(itertools.product(outcomes,['pooled','equity','bond','mixed'],[1,2,3],FROZEN_VARIANTS))


def independent_relaxed_score(positions, *, window):
    """Independent arithmetic on normalized source-evidence columns.

    Adapter requirements: one row per union-held corporate security; retain
    unknown Q/P rows as applicable; exclude only explicit inherited non-news
    exclusions. Scope states must distinguish documented OUTSIDE_NEWS_UNIVERSE
    from UNKNOWN_MAPPING. A successful-label join cannot establish membership.
    Original sentiment is never overwritten when movement alone proves a zero.
    """
    g=positions.loc[positions.known_non_news_excluded.eq(False)].copy()
    inside=g.scope_state.eq('IN_NEWS_UNIVERSE')
    unknown_scope=g.scope_state.eq('UNKNOWN_MAPPING')
    scope=g.loc[inside]
    result=dict(in_scope_positions=len(scope),outside_news_universe_positions=int(g.scope_state.eq('OUTSIDE_NEWS_UNIVERSE').sum()),unknown_mapping_positions=int(unknown_scope.sum()))
    if scope.empty:
        return dict(result,score=np.nan,status='NA_NO_APPLICABLE_SCOPE',numerator=np.nan,denominator=np.nan,known_zero_flags={})
    q=pd.to_numeric(scope.B_Q,errors='coerce').to_numpy(float)
    net=pd.to_numeric(scope[window+'_net'],errors='coerce').to_numpy(float)
    window_exists=scope[window+'_window_exists'].eq(True).to_numpy()
    news_known=scope[window+'_known'].eq(True).to_numpy() & window_exists & np.isfinite(net) & (abs(net)<=1+1e-12)
    absence=scope[window+'_absence'].eq(True).to_numpy()
    qknown=np.isfinite(q)
    # Exact zero movement proves a zero product even with unknown sentiment;
    # near-zero aggregate evidence is recorded separately below.
    product_known=qknown & window_exists & ((q==0)|news_known)
    products=np.full(len(scope),np.nan)
    products[product_known & (q==0)]=0.
    usable=qknown & news_known
    products[usable]=q[usable]*net[usable]
    numerator=float(np.sum(products[product_known]))
    denominator=float(np.sum(abs(products[product_known])))
    gross=float(np.sum(abs(q[product_known])))
    epsilon=1e-12*max(1.,gross)
    unresolved=int((~product_known).sum())+int(unknown_scope.sum())
    result.update(numerator=numerator,denominator=denominator,epsilon=epsilon,known_product_positions=int(product_known.sum()),unknown_contribution_positions=unresolved,quantity_unknown_positions=int((~qknown).sum()),company_count=int(scope.loc[product_known & (np.nan_to_num(abs(products))>0),'company_id'].nunique()))
    flags=dict(no_measured_movement=False,confirmed_news_absence=False,observed_zero_signal_denominator=False,offsetting_alignment=False)
    if denominator>epsilon:
        score=numerator/denominator
        flags['offsetting_alignment']=abs(numerator)<=epsilon
        status='OBSERVED_OFFSETTING_ALIGNMENT' if flags['offsetting_alignment'] else 'OBSERVED_PARTIAL_SUBPORTFOLIO' if unresolved else 'OBSERVED_ALIGNMENT'
        return dict(result,score=score,status=status,known_zero_flags=flags)
    if not product_known.any():
        return dict(result,score=np.nan,status='NA_INCOMPLETE_ZERO_EVIDENCE',known_zero_flags=flags)
    q_observed=q[product_known]
    flags['no_measured_movement']=bool(np.sum(abs(q_observed))<=epsilon)
    flags['confirmed_news_absence']=bool(np.any(news_known & absence & product_known) and np.all(((q==0)|(news_known & absence))[product_known]))
    flags['observed_zero_signal_denominator']=bool(denominator<=epsilon and np.any(product_known & news_known & ~absence & (net==0)))
    return dict(result,score=0.,status='OBSERVED_SUBPORTFOLIO_ZERO',known_zero_flags=flags,exact_zero_movement=bool(np.all(q_observed==0)),observed_subset_only=True)


def independent_full_dummy_cr1(sample, *, outcome, terms):
    """Direct LSDV OLS/CR1, weighted-average intercept, Student t(G-1).

    The adapter compares all coefficients/covariance/inference and explicit NA
    with native output. This helper never creates production model artifacts.
    """
    n = len(sample)
    codes, funds = pd.factorize(sample.fund_id.astype(str),sort=True)
    G = len(funds)
    result = dict(nobs=n,funds=G,reference_df=G-1,sample_hash=ordered_sample_sha256(sample))
    if n<30 or G<2:
        return dict(result,status='NOT_ESTIMABLE',reason='fewer_than_30_rows_or_two_funds_after_singletons')
    X = sample[terms].to_numpy(float)
    Y = sample[outcome].to_numpy(float)
    if not np.isfinite(np.column_stack([Y,X])).all():
        raise ValueError('Model adapter supplied unknown inputs')
    within = X-np.vstack([X[codes==j].mean(0) for j in range(G)])[codes]
    absorbed = np.max(abs(within),axis=0)<=1e-12*np.maximum(1,np.max(abs(X),axis=0))
    dropped = [term for term,drop in zip(terms,absorbed) if drop]
    if any(term!='past_alpha' for term in dropped):
        return dict(result,status='NOT_ESTIMABLE',reason='required_NTS_target_absorbed_by_fund_FE',absorbed_columns=dropped)
    names = [term for term,drop in zip(terms,absorbed) if not drop]
    X = sample[names].to_numpy(float)
    k = len(names)
    D = np.column_stack([np.eye(G)[codes],X])
    rank = np.linalg.matrix_rank(D)
    if rank!=G+k or n-rank<=0:
        return dict(result,status='NOT_ESTIMABLE',reason='rank_or_residual_df_failure',rank=int(rank))
    b = linalg.lstsq(D,Y,lapack_driver='gelsy')[0]
    residual = Y-D@b
    bread = linalg.inv(D.T@D)
    scores = np.vstack([D[codes==j].T@residual[codes==j] for j in range(G)])
    correction = G/(G-1)*(n-1)/(n-G-k)
    covD = correction*bread@scores.T@scores@bread
    transform = np.zeros((k+1,G+k))
    transform[0,:G] = np.bincount(codes)/n
    transform[1:,G:] = np.eye(k)
    beta = transform@b
    covariance = transform@covD@transform.T
    diagonal=np.diag(covariance)
    se = np.sqrt(np.where(diagonal>0,diagonal,np.nan))
    t = beta/se
    p = 2*stats.t.sf(abs(t),G-1)
    q = stats.t.ppf(.975,G-1)
    WY = Y-np.array([Y[codes==j].mean() for j in range(G)])[codes]
    result.update(status='ESTIMATED',rank=int(rank),residual_df=n-rank,CR1_correction=correction,
        terms=['const']+names,beta=beta,covariance=covariance,SE=se,t=t,p=p,
        CI_low=beta-q*se,CI_high=beta+q*se,absorbed_columns=dropped,
        within_R2=1-residual@residual/(WY@WY),residuals=residual)
    if len([x for x in names if x!='past_alpha'])==2:
        block=covariance[1:3,1:3]
        total=beta[1:3].sum()
        var=block.sum()
        if np.isfinite(var) and var>0:
            ss=np.sqrt(var)
            result['sum_test']=dict(beta=total,SE=ss,t=total/ss,p=2*stats.t.sf(abs(total/ss),G-1),CI_low=total-q*ss,CI_high=total+q*ss,status='defined')
        else:
            result['sum_test']=dict(status='undefined_nonpositive_or_nonfinite_variance')
        if G>=3 and np.linalg.matrix_rank(block)==2 and np.linalg.eigvalsh(block).min()>0:
            F=beta[1:3]@linalg.solve(block,beta[1:3])/2
            result['joint_test']=dict(F=F,p=stats.f.sf(F,2,G-1),df_num=2,df_denom=G-1,status='defined')
        else:
            result['joint_test']=dict(F=np.nan,p=np.nan,status='undefined_target_covariance_not_positive_definite')
    return result


def independent_alpha(monthly, *, origin_date, horizon, lookback=60, minimum=36):
    """Two-factor past-only reconstruction for a supplied source series."""
    d=monthly.copy()
    d.index=pd.PeriodIndex(d.month,freq='M')
    if d.index.duplicated().any():
        raise ValueError('Duplicate source fund/month rows')
    d=d.sort_index()
    date=pd.Timestamp(origin_date).normalize()
    m=date.to_period('M')
    last=m if m.end_time.normalize()<=date else m-1
    first=last-lookback+1
    finite=np.isfinite(d[['fund_return','risk_free','equity_return','bond_return']]).all(axis=1)
    ok=d.return_verified.eq(True)&d.factors_verified.eq(True)&finite
    dates_ok=pd.to_datetime(d.end_valuation_date).le(date)&pd.to_datetime(d.start_valuation_date).le(date)
    if 'return_period_end' in d:
        dates_ok &= pd.to_datetime(d.return_period_end).le(date)
    t=d.loc[(d.index>=first)&(d.index<=last)&ok & dates_ok]
    if len(t)<minimum or (t.index>=last-11).sum()<9:
        return dict(alpha=np.nan,status='insufficient_past_history',training_months=t.month.tolist())
    xx=np.column_stack([t.equity_return-t.risk_free,t.bond_return-t.risk_free])
    X=np.column_stack([np.ones(len(t)),xx])
    scale=xx.std(0,ddof=1)
    if not np.isfinite(scale).all() or (scale<=0).any():
        return dict(alpha=np.nan,status='factor_rank_failure')
    condition=np.linalg.cond(np.column_stack([np.ones(len(t)),(xx-xx.mean(0))/scale]))
    if np.linalg.matrix_rank(X)!=3 or condition>10000:
        return dict(alpha=np.nan,status='factor_rank_or_condition_failure')
    b=linalg.lstsq(X,(t.fund_return-t.risk_free).to_numpy(float),lapack_driver='gelsy')[0]
    future=pd.period_range(m+1,m+horizon,freq='M')
    legs=d.reindex(future)
    complete=ok.reindex(future).eq(True).all()
    anchor=legs.start_valuation_date.iloc[0]
    if not complete or pd.isna(anchor) or pd.Timestamp(anchor)<date:
        return dict(alpha=np.nan,status='future_incomplete_or_anchor_unverified',past_alpha=b[0],betas=b,training_months=t.month.tolist())
    future_X=np.column_stack([legs.equity_return-legs.risk_free,legs.bond_return-legs.risk_free])
    abnormal=(legs.fund_return-legs.risk_free).to_numpy(float)-future_X@b[1:]
    return dict(alpha=float(abnormal.mean()),past_alpha=b[0],betas=b,training_months=t.month.tolist(),training_first_allowed=str(first),training_last_allowed=str(last),condition_number=condition,future_months=legs.month.tolist(),future_abnormal=abnormal,status='eligible')


def self_check():
    base=dict(n_candidates=0,n_valid=0,n_relevant=0,n_positive=0,n_negative=0,n_neutral=0,n_errors_or_unattempted=0,n_undated_company_candidates=0)
    def news(row=base,**kw):
        args=dict(mapping_known=True,window_found=True,date_bounds_verified=True,undated_policy='block_absence_zero_if_undated')
        args.update(kw)
        return classify_news_window(row,**args)
    assert news()['score']==0
    assert not news(mapping_known=False)['known']
    assert not news(window_found=False)['known']
    failed={**base,'n_candidates':1,'n_errors_or_unattempted':1}
    assert not news(failed)['known']
    irrelevant={**base,'n_candidates':1,'n_valid':1}
    assert news(irrelevant)['score']==0 and news(irrelevant)['zero_by_absence']
    balanced={**base,'n_candidates':2,'n_valid':2,'n_relevant':2,'n_positive':1,'n_negative':1}
    assert news(balanced)['score']==0 and not news(balanced)['zero_by_absence']
    undated={**base,'n_undated_company_candidates':1}
    assert not news(undated,undated_policy='block_absence_zero_if_undated')['known']
    assert news({**balanced,'n_undated_company_candidates':1})['known']
    assert not classify_zero_movement([np.nan,np.nan],scope_complete=True)['known_zero']
    assert not classify_zero_movement([0.,np.nan],scope_complete=True)['known_zero']
    assert classify_zero_movement([0.,0.],scope_complete=True)['known_zero']
    assert not classify_zero_movement([0.,0.],scope_complete=False)['known_zero']
    panel=pd.DataFrame(dict(fund_id=['a','a','b','b'],interval_id=['1','2','3','4'],end_date=pd.date_range('2020-01-01',periods=4),past_alpha=[.01]*4,alpha=[.02]*4,current=[0.,.5,0.,-.5],lag=[np.nan]*4,universe=[True]*4))
    current,_=expected_model_sample(panel,outcome='alpha',current='current',lag='lag',distributed=False,universe_column='universe')
    distributed,_=expected_model_sample(panel,outcome='alpha',current='current',lag='lag',distributed=True,universe_column='universe')
    assert len(current)==4 and distributed.empty
    assert len(planned_cells(['diagnostic','provider_TR']))==48
    positions=pd.DataFrame(dict(company_id=['x','y'],known_non_news_excluded=[False,False],scope_state=['IN_NEWS_UNIVERSE']*2,B_Q=[0.,0.],current_net=[np.nan,np.nan],current_known=[False,False],current_absence=[False,False],current_window_exists=[True,True]))
    assert independent_relaxed_score(positions,window='current')['status']=='OBSERVED_SUBPORTFOLIO_ZERO'
    positions['B_Q']=[0.,np.nan]
    partial_zero=independent_relaxed_score(positions,window='current')
    assert partial_zero['status']=='OBSERVED_SUBPORTFOLIO_ZERO' and partial_zero['unknown_contribution_positions']==1
    positions['B_Q']=[10.,20.];positions['current_net']=[0.,0.];positions['current_known']=True;positions['current_absence']=True
    assert independent_relaxed_score(positions,window='current')['status']=='OBSERVED_SUBPORTFOLIO_ZERO'
    positions['B_Q']=[np.nan,np.nan]
    assert independent_relaxed_score(positions,window='current')['status']=='NA_INCOMPLETE_ZERO_EVIDENCE'
    positions['B_Q']=[10.,20.];positions['current_net']=[1.,np.nan];positions['current_known']=[True,False]
    partial=independent_relaxed_score(positions,window='current')
    assert partial['score']==1 and partial['company_count']==1 and partial['status']=='OBSERVED_PARTIAL_SUBPORTFOLIO'
    positions['scope_state']=['OUTSIDE_NEWS_UNIVERSE']*2
    assert independent_relaxed_score(positions,window='current')['status']=='NA_NO_APPLICABLE_SCOPE'
    positions['scope_state']=['IN_NEWS_UNIVERSE']*2;positions['B_Q']=[0.,0.];positions['current_window_exists']=False
    assert independent_relaxed_score(positions,window='current')['status']=='NA_INCOMPLETE_ZERO_EVIDENCE'
    import ast
    module=ast.parse(Path(__file__).read_text(encoding='utf-8'))
    fn=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='self_check')
    count=sum(isinstance(n,ast.Assert) for n in ast.walk(fn))
    return dict(status='PASS',synthetic_semantic_checks=count,production_models_run=0,new_run_numeric_QA_started=False,frozen_spec_sha256=RELAXED_SPEC_SHA256)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check',action='store_true')
    args=parser.parse_args()
    if not args.self_check:
        parser.error('Prepared primitives only: final adapter awaits frozen spec and sealed new root. Use --self-check for synthetic evidence cases.')
    print(json.dumps(self_check()))
