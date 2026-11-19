"""Synthetic contracts: no hidden lag, coverage or completeness gate."""
import json
import numpy as np
import pandas as pd
from h3_relaxed_news_core_20261104 import independent_scores,classify_news

def tests():
    checks={}
    def row(i,**kw):
        d=dict(fund_id='f',interval_id='i',security_id=i,position_id=i,company_id='c',B_Q=100.,
            quantity_in_scope=True,quantity_action_eligible=True,mapping_in_scope=True,restored_news_only=False,
            current_source_present=True,lagnews_source_present=True,current_window_valid=True,lagnews_window_valid=True,
            current_news_known=True,lagnews_news_known=True,current_news_value=.5,lagnews_news_value=-.5,
            current_news_status='OBSERVED_SENTIMENT',lagnews_news_status='OBSERVED_SENTIMENT',
            eligible_for_score=False,NTS_common_E=False)
        d.update(kw);return d
    origin=pd.DataFrame([dict(fund_id='f',interval_id='i')])
    s,p=independent_scores(pd.DataFrame([row('a',lagnews_news_known=False,lagnews_news_value=np.nan)]),origin)
    checks['current_does_not_require_lag_or_legacy_mutated_gate']=s.current_eligible.all() and s.NTS_Q_current_relaxed.iloc[0]==1 and not s.lagnews_eligible.any()
    s,_=independent_scores(pd.DataFrame([row('a'),row('b',company_id='d',quantity_in_scope=False,B_Q=10000.)]),origin)
    checks['one_company_and_sub50coverage_allowed']=s.current_eligible.all() and s.current_company_count.iloc[0]==1 and s.current_coverage.iloc[0]<.5
    s,p=independent_scores(pd.DataFrame([row('a',current_news_value=0.),row('b',company_id='d',B_Q=np.nan)]),origin)
    checks['observed_subset_zero_unknown_row_not_veto']=s.NTS_Q_current_relaxed.iloc[0]==0 and s.current_status.iloc[0]=='OBSERVED_SUBPORTFOLIO_ZERO' and s.current_unknown_contribution_count.iloc[0]==1
    checks['unknown_quantity_product_stays_NA']=pd.isna(p.current_product.iloc[1])
    s,_=independent_scores(pd.DataFrame([row('a',B_Q=np.nan)]),origin)
    checks['all_unknown_not_zero']=s.NTS_Q_current_relaxed.isna().all()
    s,p=independent_scores(pd.DataFrame([row('a',B_Q=0.,current_news_known=False,current_news_value=np.nan)]),origin)
    checks['measured_no_move_zero_without_sentiment_imputation']=s.NTS_Q_current_relaxed.iloc[0]==0 and pd.isna(p.current_news_value.iloc[0]) and s.current_zero_no_measured_movement.iloc[0]
    s,_=independent_scores(pd.DataFrame([row('a',B_Q=0.,lagnews_window_valid=False)]),origin)
    checks['missing_previous_interval_not_zero']=s.NTS_Q_lagnews_relaxed.isna().all()
    s,p=independent_scores(pd.DataFrame([row('a',B_Q=0.,current_source_present=False,current_news_known=False,current_news_value=np.nan)]),origin)
    checks['measured_zero_valid_interval_does_not_require_news_record']=s.NTS_Q_current_relaxed.iloc[0]==0 and pd.isna(p.current_news_value.iloc[0])
    s,_=independent_scores(pd.DataFrame([row('a',B_Q=0.,mapping_in_scope=False)]),origin)
    checks['outside_mapping_not_zero']=s.NTS_Q_current_relaxed.isna().all()
    news=pd.DataFrame([dict(company_id='c',source_present=True,window_valid=True,mapping_in_scope=True,
        n_candidates=0,n_valid=0,n_relevant=0,n_errors_or_unattempted=0,n_undated_company_candidates=0,net_sentiment=np.nan),
        dict(company_id='d',source_present=True,window_valid=True,mapping_in_scope=True,n_candidates=2,n_valid=1,
        n_relevant=1,n_errors_or_unattempted=1,n_undated_company_candidates=0,net_sentiment=.5)])
    classified=classify_news(news)
    checks['explicit_no_candidates_is_known_zero']=classified.value.iloc[0]==0
    checks['incomplete_LLM_not_known_zero']=pd.isna(classified.value.iloc[1])
    old=news.iloc[[1]].copy();old['n_valid']=2;old['n_errors_or_unattempted']=0;old['n_undated_company_candidates']=7
    preserved=classify_news(old)
    checks['legacy_known_Net_not_gated_by_undated_audit']=preserved.value.iloc[0]==.5 and preserved.news_status.iloc[0]=='OBSERVED_SENTIMENT'
    absence=news.iloc[[0]].copy();absence['n_undated_company_candidates']=1
    checks['undated_blocks_new_absence_zero_only']=classify_news(absence).value.isna().all()
    missing_company=news.iloc[[0]].copy();missing_company['company_id']=None
    unknown=classify_news(missing_company)
    checks['missing_company_clears_numeric_value_and_known_flag']=unknown.value.isna().all() and not unknown.news_known.any()
    assert all(checks.values()),checks
    return dict(status='PASS',synthetic_only=True,checks_pass=len(checks),checks_fail=0,checks={k:bool(v) for k,v in checks.items()})
if __name__=='__main__':print(json.dumps(tests(),indent=2))
