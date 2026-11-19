"""Pure independent current/lag observed-subportfolio scores, frozen V2."""
from __future__ import annotations
import numpy as np
import pandas as pd
from h3_alpha_core_20261104 import numeric,truth,unique,digest_lines

ABSENCE={'ZERO_NO_CANDIDATES_IN_FROZEN_CORPUS','ZERO_ALL_CANDIDATES_IRRELEVANT'}


def classify_news(windows):
    d=windows.copy()
    count=['n_candidates','n_valid','n_relevant','n_errors_or_unattempted','n_undated_company_candidates']
    for c in count:d[c]=numeric(d[c])
    d['value']=np.nan;d['news_status']='NA_SOURCE_OR_WINDOW_UNAVAILABLE'
    present=truth(d.source_present)&truth(d.window_valid)&truth(d.mapping_in_scope)
    # V3 preserves all formerly eligible dated-event Net values. Undated counts
    # prohibit NEW absence-based zeros, not an already observed finite score.
    clean=present&d[[c for c in count if c!='n_undated_company_candidates']].notna().all(axis=1)&d.n_errors_or_unattempted.eq(0)
    complete=clean&d.n_valid.eq(d.n_candidates)
    none=complete&d.n_candidates.eq(0)&d.n_undated_company_candidates.eq(0)
    irrelevant=complete&d.n_candidates.gt(0)&d.n_relevant.eq(0)&d.n_undated_company_candidates.eq(0)
    measured=complete&d.n_relevant.gt(0)&np.isfinite(numeric(d.net_sentiment))&numeric(d.net_sentiment).abs().le(1+1e-12)
    d.loc[none,'news_status']='ZERO_NO_CANDIDATES_IN_FROZEN_CORPUS'
    d.loc[irrelevant,'news_status']='ZERO_ALL_CANDIDATES_IRRELEVANT'
    d.loc[none|irrelevant,'value']=0.
    d.loc[measured,'news_status']='OBSERVED_SENTIMENT'
    d.loc[measured,'value']=numeric(d.loc[measured,'net_sentiment'])
    d.loc[present&~clean,'news_status']='NA_UNRESOLVED_CLASSIFICATION_OR_UNDATED'
    d.loc[complete&d.n_relevant.eq(0)&~d.n_undated_company_candidates.eq(0),'news_status']='NA_UNDATED_PREVENTS_NEW_ABSENCE_ZERO'
    d.loc[~truth(d.mapping_in_scope),'news_status']='OUTSIDE_NEWS_UNIVERSE'
    d.loc[d.company_id.isna(),'news_status']='NA_COMPANY_IDENTITY'
    d.loc[~truth(d.window_valid)&truth(d.mapping_in_scope),'news_status']='NA_INVALID_OR_MISSING_WINDOW_CHAIN'
    d.loc[d.company_id.isna(),'value']=np.nan
    d.loc[d.company_id.isna(),'news_status']='NA_COMPANY_IDENTITY'
    d['news_known']=np.isfinite(d.value)
    return d


def independent_scores(contributions,origins):
    """Unknown excluded rows neither become0 nor veto supported subset0."""
    d=contributions.copy()
    unique(d,['fund_id','interval_id','security_id'],'relaxed contributions')
    unique(origins,['fund_id','interval_id'],'relaxed origins')
    d['B_Q']=numeric(d.B_Q)
    physical=truth(d.quantity_in_scope)&truth(d.quantity_action_eligible)&np.isfinite(d.B_Q)
    mapped=truth(d.mapping_in_scope)&d.company_id.notna()
    for label in ['current','lagnews']:
        known=truth(d[label+'_news_known'])&np.isfinite(numeric(d[label+'_news_value']))
        window=truth(d[label+'_window_valid'])
        no_move=d.B_Q.eq(0)
        usable=physical&mapped&window&((known&truth(d[label+'_source_present']))|no_move)
        d[label+'_usable_E']=usable
        d[label+'_product']=np.nan;d[label+'_weight']=np.nan
        d.loc[usable&known,label+'_product']=(d.B_Q*numeric(d[label+'_news_value'])).loc[usable&known]
        d.loc[usable&known,label+'_weight']=(d.B_Q.abs()*numeric(d[label+'_news_value']).abs()).loc[usable&known]
        d.loc[usable&no_move,label+'_product']=0.
        d.loc[usable&no_move,label+'_weight']=0.
        d[label+'_zero_product_from_measured_no_movement_unknown_news']=usable&no_move&~known
        d[label+'_unknown_contribution']=truth(d.quantity_in_scope)&~usable
    records=[]
    for (fund,interval),g in d.groupby(['fund_id','interval_id'],observed=True,sort=False):
        row=dict(fund_id=fund,interval_id=interval,corporate_position_rows=len(g),
            gross_B_Q_all_measurable_corporate=float(g.B_Q.abs().sum(min_count=1)),
            outside_news_universe_count=int((~truth(g.mapping_in_scope)).sum()),
            excluded_nonnews_count=int((~truth(g.quantity_in_scope)).sum()),
            restored_news_only_position_count=int(truth(g.restored_news_only).sum()))
        for label in ['current','lagnews']:
            e=g.loc[g[label+'_usable_E']];gross=float(e.B_Q.abs().sum());epsilon=1e-12*max(1.,gross)
            numerator=float(e[label+'_product'].sum());denominator=float(e[label+'_weight'].sum())
            observed=len(e)>0
            if not observed:score=np.nan;status='NA_NO_USABLE_OBSERVED_SUBPORTFOLIO'
            elif denominator<=epsilon:score=0.;status='OBSERVED_SUBPORTFOLIO_ZERO'
            else:
                score=numerator/denominator
                status='OBSERVED_OFFSETTING_ALIGNMENT' if numerator==0 else 'OBSERVED_RATIO'
            if np.isfinite(score) and abs(score)>1+1e-10:raise AssertionError('Score outside[-1,1]')
            covered=float(e.loc[numeric(e[label+'_news_value']).ne(0)&e[label+'_news_known'].eq(True),'B_Q'].abs().sum())
            total=row['gross_B_Q_all_measurable_corporate']
            zero=observed and denominator<=epsilon
            row.update({f'NTS_Q_{label}_relaxed':score,f'{label}_status':status,
                f'{label}_numerator':numerator,f'{label}_denominator':denominator,f'{label}_epsilon':epsilon,
                f'{label}_usable_positions':len(e),f'{label}_gross_B_Q_E':gross,
                f'{label}_company_count':int(e.loc[e[label+'_weight'].gt(0),'company_id'].nunique()),
                f'{label}_coverage':covered/total if np.isfinite(total) and total>0 else np.nan,
                f'{label}_unknown_contribution_count':int(g[label+'_unknown_contribution'].sum()),
                f'{label}_zero_no_measured_movement':bool(zero and e.B_Q.eq(0).all()),
                f'{label}_zero_has_confirmed_news_absence':bool(zero and e[label+'_news_status'].isin(ABSENCE).any()),
                f'{label}_zero_has_observed_neutral_balanced_signal':bool(zero and (e[label+'_news_status'].eq('OBSERVED_SENTIMENT')&numeric(e[label+'_news_value']).eq(0)).any()),
                f'{label}_zero_has_unknown_news_no_movement':bool(zero and e[label+'_zero_product_from_measured_no_movement_unknown_news'].any()),
                f'{label}_E_hash':digest_lines(sorted(e.position_id.astype(str)))})
        records.append(row)
    score=pd.DataFrame(records)
    if score.empty:score=pd.DataFrame(columns=['fund_id','interval_id'])
    output=origins.merge(score,on=['fund_id','interval_id'],how='left',validate='one_to_one')
    for label in ['current','lagnews']:
        col='NTS_Q_'+label+'_relaxed'
        if col not in output:output[col]=np.nan
        output[label+'_eligible']=np.isfinite(numeric(output[col]))
        status=label+'_status'
        if status not in output:output[status]='NA_NO_APPLICABLE_SCOPE'
        output[status]=output[status].fillna('NA_NO_APPLICABLE_SCOPE')
    return output,d
