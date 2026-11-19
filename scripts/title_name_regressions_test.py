"""Meaningful synthetic checks, including an independent dummy-OLS calculation."""
from __future__ import annotations

import tempfile
import unittest
import json
import hashlib
import os
import sqlite3
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from title_name_regressions_core import (
    IntegrityError, NotEstimable, aggregate_news_windows, attach_exact_month_lags,
    build_weight_panel, freeze_json, interval_total_return, nts_by_interval,
    report_intervals, validate_issuer_bridge,
)
from title_name_regressions_estimation import (
    fit_panel_fe, fixed_effect_rank, h2_absolute_reaction_test, prune_singletons,
    fit_signal, signal_control_plan,
)
from title_name_regressions_news import collapse_events
from title_name_regressions_alpha import future_fund_alpha
from title_name_regressions_data import normalize_holdings,valid_isin
from title_name_regressions_integrity import (PREPARATION_MODULES,PREPARED_FILES,preparation_identity,
    seal_preparation,validate_preparation,verified_llm_binding,APPROVED_PROMPT_VERSION,APPROVED_RESULT_MODEL)
from title_name_regressions_news import read_labels,monthly_states


def make_intervals(full_close=True):
    return report_intervals(pd.DataFrame([
        dict(fund_id="f1", report_date="2020-01-30", report_id="r1", full_portfolio=True),
        dict(fund_id="f1", report_date="2020-03-05", report_id="r2", full_portfolio=full_close),
    ]))


def holding(security, weight, date, company="company_a", **extra):
    return dict(fund_id="f1", report_date=date, security_id=security, company_id=company,
                weight=weight, is_leaf=True, asset_type="cash" if security=="CASH" else "stock", **extra)


class PortfolioChecks(unittest.TestCase):
    def setUp(self):
        self.intervals = make_intervals()
        self.iid = self.intervals.interval_id.iloc[0]

    def returns(self, mapping):
        return pd.DataFrame([dict(interval_id=self.iid, security_id=k, total_return=v,
                                  return_coverage_complete=np.isfinite(v)) for k,v in mapping.items()])

    def test_no_trade_has_zero_at_despite_weight_drift(self):
        opening = dict(A=.4, B=.4, CASH=.2)
        closing = dict(A=.5/1.1, B=.4/1.1, CASH=.2/1.1)
        h = pd.DataFrame([holding(k,v,"2020-01-30",company=None if k!="A" else "ca") for k,v in opening.items()]
                         + [holding(k,v,"2020-03-05",company=None if k!="A" else "ca") for k,v in closing.items()])
        p = build_weight_panel(h,self.intervals,self.returns(dict(A=.25,B=0,CASH=0)))
        np.testing.assert_allclose(p.AT,0,atol=1e-12)
        self.assertGreater(abs(p.loc[p.security_id.eq("A"),"raw_delta_weight"].iloc[0]),.05)
        np.testing.assert_allclose(p.portfolio_bh_denominator,1.1)

    def test_entry_exit_and_unmapped_opening_asset_are_retained(self):
        h = pd.DataFrame([holding("A",.4,"2020-01-30"),holding("B",.4,"2020-01-30",None),
                          holding("CASH",.2,"2020-01-30",None),holding("A",.7,"2020-03-05"),
                          holding("CASH",.2,"2020-03-05",None),holding("NEW",.1,"2020-03-05","new_company")])
        p = build_weight_panel(h,self.intervals,self.returns(dict(A=0,B=0,CASH=0))).set_index("security_id")
        self.assertAlmostEqual(p.loc["B","AT"],-.4)
        self.assertAlmostEqual(p.loc["NEW","AT"],.1)
        self.assertTrue(p.loc["B","full_exit"])
        self.assertTrue(p.loc["NEW","full_entry"])
        self.assertTrue(pd.isna(p.loc["NEW","total_return"]))
        self.assertAlmostEqual(p.AT.sum(),0)

    def test_one_unknown_opening_return_blocks_the_whole_bh_denominator(self):
        h = pd.DataFrame([holding(k,v,d) for d in ["2020-01-30","2020-03-05"] for k,v in dict(A=.4,B=.4,CASH=.2).items()])
        p = build_weight_panel(h,self.intervals,self.returns(dict(A=.1,B=np.nan,CASH=0)))
        self.assertTrue(p.AT.isna().all())
        self.assertTrue(p.raw_delta_weight.eq(0).all())
        self.assertTrue(p.at_gate_reason.eq("missing_opening_asset_total_return").all())
        np.testing.assert_allclose(p.portfolio_return_weight_coverage,.6)

    def test_residual_is_not_assumed_to_earn_zero(self):
        h = pd.DataFrame([holding(k,v,d) for d in ["2020-01-30","2020-03-05"] for k,v in dict(A=.5,B=.3).items()])
        unknown = build_weight_panel(h,self.intervals,self.returns(dict(A=0,B=0)))
        self.assertIn("__RESIDUAL__",unknown.security_id.tolist())
        self.assertTrue(unknown.AT.isna().all())
        known = build_weight_panel(h,self.intervals,self.returns(dict(A=0,B=0,__RESIDUAL__=0)))
        np.testing.assert_allclose(known.AT,0,atol=1e-12)

    def test_incomplete_report_does_not_turn_missing_position_into_sale(self):
        h = pd.DataFrame([holding("A",.5,"2020-01-30"),holding("B",.5,"2020-01-30"),holding("A",.5,"2020-03-05")])
        p = build_weight_panel(h,make_intervals(False),self.returns(dict(A=0,B=0))).set_index("security_id")
        self.assertTrue(pd.isna(p.loc["B","weight_close"]))
        self.assertFalse(p.loc["B","full_exit"])
        self.assertTrue(p.AT.isna().all())

    def test_missing_reported_weight_is_not_filled(self):
        h = pd.DataFrame([holding("A",np.nan,"2020-01-30"),holding("A",1.0,"2020-03-05")])
        p = build_weight_panel(h,self.intervals,self.returns(dict(A=0)))
        self.assertTrue(p.weight_open.isna().all())
        self.assertTrue(p.AT.isna().all())

    def test_categories_are_not_leaf_positions(self):
        h = pd.DataFrame([holding("A",1,d) for d in ["2020-01-30","2020-03-05"]])
        category = h.iloc[[0]].copy()
        category["security_id"]="ALL_STOCKS"
        category["is_leaf"]=False
        p = build_weight_panel(pd.concat([h,category]),self.intervals,self.returns(dict(A=0)))
        self.assertEqual(len(p),1)
        self.assertEqual(p.security_id.iloc[0],"A")

    def test_duplicate_leaf_security_cannot_expand_the_join(self):
        h = pd.DataFrame([holding("A",1,d) for d in ["2020-01-30","2020-03-05"]])
        h = pd.concat([h,h.iloc[[0]]])
        with self.assertRaises(IntegrityError):
            build_weight_panel(h,self.intervals,self.returns(dict(A=0)))

    def test_percent_weights_fail_portfolio_reconciliation(self):
        h = pd.DataFrame([holding("A",100,d) for d in ["2020-01-30","2020-03-05"]])
        p = build_weight_panel(h,self.intervals,self.returns(dict(A=0)))
        self.assertTrue(p.AT.isna().all())
        self.assertTrue(p.at_gate_reason.eq("portfolio_denominator_not_reconciled").all())

    def test_true_report_dates_are_preserved(self):
        self.assertEqual(self.intervals.start_date.iloc[0],pd.Timestamp("2020-01-30"))
        self.assertEqual(self.intervals.end_date.iloc[0],pd.Timestamp("2020-03-05"))

    def test_unselected_report_revisions_raise(self):
        reports = pd.DataFrame([dict(fund_id="f",report_date="2020-01-31",report_id=k,full_portfolio=True) for k in ["v1","v2"]])
        with self.assertRaises(IntegrityError): report_intervals(reports)

    def test_split_and_dividend_return_units(self):
        self.assertEqual(interval_total_return(100,50,2,0,coverage_complete=True),0)
        self.assertAlmostEqual(interval_total_return(100,50,2,5,coverage_complete=True),.05)
        self.assertTrue(np.isnan(interval_total_return(100,50,2,0,coverage_complete=False)))


class NewsAndKeyChecks(unittest.TestCase):
    def candidates(self):
        return pd.DataFrame([
            dict(pair_id="p1",article_id="a1",company_id="c",event_date="2020-02-10",attempted=True,valid=True,relevance="relevant",sentiment_label="neutral"),
            dict(pair_id="p2",article_id="a2",company_id="d",event_date="2020-02-10",attempted=True,valid=True,relevance="irrelevant",sentiment_label=None),
        ])

    def targets(self):
        return pd.DataFrame([dict(interval_id="i",company_id=k,news_start="2020-01-31",news_end="2020-02-29") for k in ["c","d","z"]])

    def test_no_news_irrelevant_and_neutral_remain_distinct(self):
        p,_ = aggregate_news_windows(self.targets(),self.candidates())
        p=p.set_index("company_id")
        self.assertEqual(p.loc["c","Tone"],0)
        self.assertTrue(pd.isna(p.loc["d","Tone"]))
        self.assertTrue(pd.isna(p.loc["z","Tone"]))
        self.assertEqual(p.loc["d","news_status"],"no_relevant_news")
        self.assertEqual(p.loc["z","news_status"],"no_title_candidates")

    def test_partial_llm_coverage_is_not_neutral(self):
        c=self.candidates().iloc[[0]].copy()
        c["sentiment_label"]="negative"
        missing=c.iloc[[0]].copy()
        missing["pair_id"],missing["article_id"]="p3","a3"
        missing["attempted"],missing["valid"],missing["relevance"],missing["sentiment_label"]=False,False,None,None
        p,_=aggregate_news_windows(self.targets(),pd.concat([c,missing]))
        row=p.loc[p.company_id.eq("c")].iloc[0]
        self.assertEqual(row.tone_observed,-1)
        self.assertTrue(pd.isna(row.Tone))
        self.assertEqual(row.n_errors_or_unattempted,1)

    def test_left_boundary_open_right_boundary_closed(self):
        c=self.candidates().iloc[[0]].copy()
        c["event_date"]="2020-01-31"
        p,_=aggregate_news_windows(self.targets(),c)
        self.assertEqual(int(p.n_expected.sum()),0)
        c["event_date"]="2020-02-29"
        p,_=aggregate_news_windows(self.targets(),c)
        self.assertEqual(int(p.n_expected.sum()),1)

    def test_dateless_pair_is_audited_not_assigned_a_month(self):
        c=self.candidates(); c.loc[0,"event_date"]=None
        p,audit=aggregate_news_windows(self.targets(),c)
        self.assertEqual(audit["undated_pairs"],1)
        self.assertEqual(p.loc[p.company_id.eq("c"),"n_expected"].iloc[0],0)

    def test_invalid_valid_label_contract_raises(self):
        c=self.candidates(); c.loc[0,"sentiment_label"]=None
        with self.assertRaises(IntegrityError): aggregate_news_windows(self.targets(),c)

    def test_no_silent_many_to_many_issuer_bridge(self):
        bridge=pd.DataFrame(dict(isin=["RU1","RU1"],company_id=["a","b"]))
        with self.assertRaises(IntegrityError): validate_issuer_bridge(bridge)
        good=pd.DataFrame(dict(isin=["RU1","RU2"],company_id=["a","a"]))
        self.assertEqual(len(validate_issuer_bridge(good)),2)

    def test_calendar_gap_is_not_forward_filled(self):
        panel=pd.DataFrame(dict(company_id=["c","c"],month=["2020-03","2020-04"]))
        controls=pd.DataFrame(dict(company_id=["c","c"],month=["2020-02","2020-04"],value=[2,999]))
        out=attach_exact_month_lags(panel,controls,["company_id"],["value"])
        self.assertEqual(out.value_lag1.iloc[0],2)
        self.assertTrue(pd.isna(out.value_lag1.iloc[1]))

    def test_no_active_trade_does_not_create_zero_nts(self):
        p=pd.DataFrame(dict(fund_id=["f"]*3,interval_id=["i"]*3,company_id=["c1","c2","c3"],S=[1,-1,0],AT=[0,0,.5]))
        out=nts_by_interval(p)
        self.assertTrue(pd.isna(out.NTS.iloc[0]))
        p["AT"]=[.2,-.1,.1]
        out=nts_by_interval(p)
        self.assertAlmostEqual(out.NTS.iloc[0],1)

    def test_nts_uses_signal_magnitudes_not_only_their_sign(self):
        p=pd.DataFrame(dict(fund_id=["f","f"],interval_id=["i","i"],company_id=["c1","c2"],S=[2,1],AT=[.2,-.1]))
        out=nts_by_interval(p)
        self.assertAlmostEqual(out.NTS.iloc[0],.6)

    def test_nts_requires_distinct_companies_and_all_corporate_trade_coverage(self):
        p=pd.DataFrame(dict(fund_id=["f"]*3,interval_id=["i"]*3,company_id=["a","b","c"],
                            S=[1,-1,np.nan],AT=[.1,-.1,.8]))
        out=nts_by_interval(p)
        self.assertEqual(out.nts_status.iloc[0],"low_active_trade_coverage")
        self.assertTrue(pd.isna(out.NTS.iloc[0]))
        p.loc[2,"AT"]=.01;p.loc[1,"company_id"]="a"
        self.assertEqual(nts_by_interval(p).nts_status.iloc[0],"insufficient_companies")
        p.loc[1,"company_id"]="b";p.loc[2,"AT"]=np.nan
        self.assertEqual(nts_by_interval(p).nts_status.iloc[0],"corporate_AT_coverage_incomplete")

    def test_frozen_payload_cannot_change(self):
        with tempfile.TemporaryDirectory(prefix="title_name_regressions_") as tmp:
            p=Path(tmp)/"freeze.json"
            freeze_json(p,dict(spec_hash="one"))
            freeze_json(p,dict(spec_hash="one"))
            with self.assertRaises(IntegrityError): freeze_json(p,dict(spec_hash="two"))


class EstimationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(7261)
        rows=[]
        ne,nt=11*8,8
        entity=rng.normal(size=ne)
        time=rng.normal(size=nt)
        fshock=rng.normal(scale=.2,size=(11,nt))
        cshock=rng.normal(scale=.3,size=(8,nt))
        for f in range(11):
            for c in range(8):
                for t in range(nt):
                    x,z=rng.normal(size=2)
                    y=.8*x-.5*z+entity[f*8+c]+time[t]+fshock[f,t]+cshock[c,t]+rng.normal()
                    rows.append(dict(fund_id=f,company_id=c,fund_security_id=f"{f}:{c}",month=t,x=x,z=z,y=y))
        cls.frame=pd.DataFrame(rows)

    def test_two_way_fe_and_clusters_match_independent_explicit_dummy_ols(self):
        import statsmodels.api as sm
        from statsmodels.stats.sandwich_covariance import cov_cluster_2groups
        p=self.frame.copy()
        result=fit_panel_fe(p,"y",["x","z"])
        dummy=pd.get_dummies(p[["fund_security_id","month"]].astype(str),drop_first=True,dtype=float)
        design=sm.add_constant(pd.concat([p[["x","z"]],dummy],axis=1))
        independent=sm.OLS(p.y,design).fit()
        covariance=cov_cluster_2groups(independent,p.fund_id.to_numpy(),p.company_id.to_numpy(),use_correction=True)[0]
        locations=[design.columns.get_loc(x) for x in ["x","z"]]
        np.testing.assert_allclose(result.beta,independent.params[["x","z"]],rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(result.covariance,covariance[np.ix_(locations,locations)],rtol=1e-8,atol=1e-10)
        self.assertEqual(result.diagnostics["residual_df"],int(independent.df_resid))
        self.assertEqual(result.diagnostics["fund_clusters"],11)
        self.assertEqual(result.diagnostics["company_clusters"],8)

    def test_fully_absorbed_and_collinear_regressors_fail_explicitly(self):
        p=self.frame.copy();p["macro"]=p.month
        with self.assertRaises(NotEstimable):fit_panel_fe(p,"y",["x","macro"])
        allowed=fit_panel_fe(p,"y",["x","macro"],allow_absorbed=["macro"])
        self.assertEqual(allowed.diagnostics["absorbed"],["macro"])
        p["x2"]=2*p.x
        with self.assertRaises(NotEstimable):fit_panel_fe(p,"y",["x","x2"])

    def test_nuisance_collinearity_never_removes_a_target(self):
        p=self.frame.copy();p["x2"]=2*p.x
        result=fit_panel_fe(p,"y",["x","z","x2"],allow_collinear=["x2"])
        self.assertEqual(result.names,["x","z"])
        self.assertEqual(result.diagnostics["collinear_nuisance_dropped"],["x2"])
        with self.assertRaises(NotEstimable):
            fit_panel_fe(p,"y",["x","x2","z"],allow_collinear=["z"])

    def test_three_way_cluster_inclusion_exclusion_against_explicit_dummies(self):
        import statsmodels.api as sm
        from statsmodels.stats.sandwich_covariance import cov_cluster
        from itertools import combinations
        p=self.frame.copy()
        result=fit_panel_fe(p,"y",["x","z"],extra_cluster="month")
        dummy=pd.get_dummies(p[["fund_security_id","month"]].astype(str),drop_first=True,dtype=float)
        design=sm.add_constant(pd.concat([p[["x","z"]],dummy],axis=1))
        independent=sm.OLS(p.y,design).fit()
        covariance=np.zeros((design.shape[1],design.shape[1]))
        for count in [1,2,3]:
            for names in combinations(["fund_id","company_id","month"],count):
                groups=pd.factorize(pd.MultiIndex.from_frame(p[list(names)]),sort=False)[0]
                covariance+=(-1)**(count+1)*cov_cluster(independent,groups,use_correction=True)
        locations=[design.columns.get_loc(x) for x in ["x","z"]]
        np.testing.assert_allclose(result.covariance,covariance[np.ix_(locations,locations)],rtol=1e-8,atol=1e-10)
        self.assertEqual(result.diagnostics["inference_df"],7)

    def test_singletons_are_removed_recursively(self):
        frame=pd.DataFrame(dict(e=["a","b","b","c","c"],t=[1,1,2,2,3]))
        out,removed,rounds=prune_singletons(frame,["e","t"])
        self.assertEqual(len(out),0)
        self.assertEqual(removed,5)
        self.assertGreater(rounds,1)

    def test_disconnected_fe_rank(self):
        rank,components=fixed_effect_rank(["a","a","b","b"],[1,1,2,2])
        self.assertEqual(components,2)
        self.assertEqual(rank,2)

    def test_h2_uses_correct_restriction_and_marks_unexpected_signs(self):
        result=fit_panel_fe(self.frame,"y",["x","z"])
        result.names=["S_positive","S_negative"]
        result.beta=np.array([.8,-.5]); result.covariance=np.array([[.04,.01],[.01,.09]])
        test=h2_absolute_reaction_test(result)
        self.assertTrue(test["expected_h1_signs"])
        self.assertAlmostEqual(test["sum_restriction"]["contrast"],.3)
        self.assertAlmostEqual(test["sum_restriction"]["se"]**2,.15)
        self.assertEqual(test["p_abs"],max(test["sum_restriction"]["p"],test["difference_restriction"]["p"]))
        result.beta=np.array([.8,.5])
        test=h2_absolute_reaction_test(result)
        self.assertFalse(test["expected_h1_signs"])
        self.assertAlmostEqual(test["difference_restriction"]["se"]**2,.11)
        self.assertEqual(test["p_abs"],max(test["sum_restriction"]["p"],test["difference_restriction"]["p"]))


class EventDuplicateChecks(unittest.TestCase):
    def collapse(self,texts,days,valid=None):
        valid=[True]*len(texts) if valid is None else valid
        with tempfile.TemporaryDirectory(prefix="title_name_regressions_events_") as tmp:
            p=Path(tmp)/"input.jsonl"
            rows=[];offset=0
            with p.open("wb") as handle:
                for i,(text,day,ok) in enumerate(zip(texts,days,valid)):
                    payload=dict(pair_id=f"p{i}",text=text)
                    line=(json.dumps(payload)+"\n").encode()
                    rows.append(dict(pair_id=f"p{i}",article_id=f"a{i}",company_id="c",source="prime",
                        title="company posts annual operating report",input_offset=offset,
                        event_timestamp=pd.Timestamp(day,tz="UTC"),valid=ok,
                        pair_state="relevant_positive" if ok else "classification_error"))
                    handle.write(line);offset+=len(line)
            return collapse_events(pd.DataFrame(rows),p)

    def test_invalid_earliest_representative_is_not_replaced(self):
        representatives,audit=self.collapse(["same report","same report"],["2020-01-01","2020-01-02"],[False,True])
        self.assertEqual(len(representatives),1)
        self.assertFalse(representatives.valid.iloc[0])
        self.assertTrue(audit.conflicting_member_labels.all())

    def test_no_transitive_72_hour_chaining(self):
        representatives,_=self.collapse(["same"]*3,["2020-01-01","2020-01-03","2020-01-05"])
        self.assertEqual(len(representatives),2)

    def test_short_nonidentical_and_empty_texts_are_not_collapsed(self):
        words=" ".join(f"word{i}" for i in range(48))
        representatives,_=self.collapse([words,words+" extra"],["2020-01-01","2020-01-02"])
        self.assertEqual(len(representatives),2)
        representatives,_=self.collapse(["",""] ,["2020-01-01","2020-01-02"],[False,False])
        self.assertEqual(len(representatives),2)

    def test_near_duplicate_threshold_and_material_update(self):
        words=[f"word{i}" for i in range(100)]
        a=" ".join(words)
        b=a+" additional"
        updated=" ".join(words[:50]+[f"different{i}" for i in range(50)])
        representatives,_=self.collapse([a,b,updated],["2020-01-01","2020-01-02","2020-01-03"])
        self.assertEqual(len(representatives),2)


class SignalAndAlphaChecks(unittest.TestCase):
    def test_fixed_control_coverage_selection(self):
        p=pd.DataFrame(dict(Tone=np.arange(10),lag_Tone=np.arange(10),a=[1]*9+[np.nan],b=[1]*8+[np.nan]*2))
        plan=signal_control_plan(p,["a","b","unavailable"])
        self.assertEqual(plan["selected"],["a"])

    def test_signal_has_unit_pooled_sd_and_respects_unique_natural_unit(self):
        rng=np.random.default_rng(710)
        p=pd.MultiIndex.from_product([range(12),range(10)],names=["company_id","month"]).to_frame(index=False)
        p["Tone"]=rng.normal(size=len(p));p["lag_Tone"]=rng.normal(size=len(p));p["log_news_count"]=rng.normal(size=len(p))
        for source in ["rbc","prime","kommersant"]:p["source_share_"+source]=rng.uniform(0,.2,size=len(p))
        out,diagnostics=fit_signal(p)
        self.assertAlmostEqual(out.S.std(ddof=1),1)
        np.testing.assert_allclose(out.groupby("company_id").S.mean(),0,atol=1e-9)
        np.testing.assert_allclose(out.groupby("month").S.mean(),0,atol=1e-9)
        self.assertEqual(diagnostics["n_final"],120)
        with self.assertRaises(IntegrityError):fit_signal(pd.concat([p,p.iloc[[0]]]))

    def alpha_data(self):
        rng=np.random.default_rng(281)
        months=pd.period_range("2010-01","2015-12",freq="M").astype(str)
        e=rng.normal(.01,.03,len(months));b=rng.normal(.003,.008,len(months));rf=np.full(len(months),.002)
        factors=pd.DataFrame(dict(month=months,risk_free=rf,equity_return=e,bond_return=b,factors_verified=True))
        funds=pd.DataFrame(dict(fund_id="f",month=months,fund_return=rf+.01+.4*(e-rf)+.2*(b-rf),return_verified=True))
        origins=pd.DataFrame(dict(fund_id=["f"],month=[months[50]]))
        return origins,funds,factors

    def test_alpha_does_not_subtract_past_intercept_or_leak_future_into_betas(self):
        origins,funds,factors=self.alpha_data()
        initial=future_fund_alpha(origins,funds,factors)
        np.testing.assert_allclose(initial.alpha,.01,atol=1e-10)
        np.testing.assert_allclose(initial.past_alpha,.01,atol=1e-10)
        funds.loc[funds.month.gt(origins.month.iloc[0]),"fund_return"]+=.1
        changed=future_fund_alpha(origins,funds,factors)
        np.testing.assert_allclose(changed.beta_equity,initial.beta_equity,atol=1e-10)
        np.testing.assert_allclose(changed.beta_bond,initial.beta_bond,atol=1e-10)
        np.testing.assert_allclose(changed.alpha,.11,atol=1e-10)

    def test_short_registered_history_is_not_an_alpha(self):
        origins,funds,factors=self.alpha_data()
        funds=funds.iloc[40:]
        result=future_fund_alpha(origins,funds,factors)
        self.assertTrue(result.alpha.isna().all())
        self.assertTrue(result.status.eq("insufficient_past_history").all())

    def test_future_horizons_require_every_calendar_month(self):
        origins,funds,factors=self.alpha_data()
        missing=str(pd.Period(origins.month.iloc[0],freq="M")+2)
        funds=funds.loc[funds.month.ne(missing)]
        result=future_fund_alpha(origins,funds,factors).set_index("horizon")
        self.assertTrue(np.isfinite(result.loc[1,"alpha"]))
        self.assertTrue(pd.isna(result.loc[2,"alpha"]))
        self.assertTrue(pd.isna(result.loc[3,"alpha"]))


class SourceNormalizationChecks(unittest.TestCase):
    def fixture(self,ambiguous=False):
        dates=["2018-01-31","2018-02-28"]
        report=pd.DataFrame([dict(report_id=str(i),fund_id="f",report_date=d,registration_number="fund-reg",
            fund_name_xls="active-name",fund_type_category="открытый,акций",NAV=1000.,gross_thousand=1.,obligations_thousand=0.) for i,d in enumerate(dates)])
        leaf=pd.DataFrame([dict(report_id=str(i),fund_id="f",report_date=d,item_name="Сбербанк",
            registration_number_security="10301481B",parent_category="Акции российских эмитентов",parent_subcategory="обыкновенные акции",
            amount_thousand_rub=.4,share_pct=40.) for i,d in enumerate(dates)])
        totals=pd.DataFrame([dict(report_id=str(i),row_type=kind,parent_category=cat,parent_subcategory=sub,
            item_name=cat,amount_thousand_rub=amount) for i in range(2) for kind,cat,sub,amount in [
                ("asset_category","Акции российских эмитентов","",.4),
                ("asset_category","Денежные средства","",.6),
                ("asset_subcategory","Денежные средства","рубли",.6)]])
        securities=pd.DataFrame(dict(isin=["RU0009029540"],security_code=["10301481B"]))
        bridge=pd.DataFrame(dict(isin=["RU0009029540"]*(2 if ambiguous else 1),company_id=["c","conflicting"] if ambiguous else ["c"]))
        market=pd.DataFrame(dict(requested_isin=["RU0009029540"],secid=["SBER"],type=["common_share"],primary_boardid=["TQBR"]))
        spec=dict(holdings_and_weights=dict(reconciliation_tolerance_fraction=.001,rounding_gap_max_fraction=.0001),
                  time_and_scope=dict(baseline_closing_month_start="2018-01",baseline_closing_month_end_cap="2026-05"))
        return (report,leaf,totals,securities,bridge,market),spec

    def test_cash_category_and_subcategory_never_double_count(self):
        inputs,spec=self.fixture()
        with patch("title_name_regressions_data.read_holdings_sources",return_value=inputs):
            h,r,intervals,conflicts,diag,market=normalize_holdings("unused",spec)
        self.assertTrue(r.full_portfolio.all())
        np.testing.assert_allclose(h.groupby("report_id").weight.sum(),1)
        np.testing.assert_allclose(r.HHI,.4**2+.6**2)
        self.assertEqual(int(h.target_security_candidate.sum()),2)
        self.assertEqual(diag["documented_residual_buckets"],2)
        self.assertTrue(intervals.preliminary_eligible.all())

    def test_ambiguous_issuer_is_excluded_but_kept_in_asset_coverage(self):
        inputs,spec=self.fixture(True)
        with patch("title_name_regressions_data.read_holdings_sources",return_value=inputs):
            h,r,intervals,conflicts,diag,market=normalize_holdings("unused",spec)
        self.assertEqual(len(conflicts),2)
        self.assertFalse(h.target_security_candidate.any())
        np.testing.assert_allclose(h.groupby("report_id").weight.sum(),1)

    def test_conflicting_lot_categories_keep_value_but_exclude_target(self):
        inputs,spec=self.fixture()
        report,leaf,totals,securities,bridge,market=inputs
        extra=leaf.copy();extra["parent_category"]="Прочее";extra["amount_thousand_rub"]=.1
        leaf=pd.concat([leaf,extra],ignore_index=True)
        report["gross_thousand"]=1.1;report["NAV"]=1100.
        totals=pd.concat([totals,pd.DataFrame([dict(report_id=str(i),row_type="asset_category",parent_category="Прочее",
            parent_subcategory="",item_name="Прочее",amount_thousand_rub=.1) for i in range(2)])],ignore_index=True)
        with patch("title_name_regressions_data.read_holdings_sources",return_value=(report,leaf,totals,securities,bridge,market)):
            h,r,intervals,conflicts,diag,market=normalize_holdings("unused",spec)
        self.assertFalse(h.target_security_candidate.any())
        self.assertTrue(r.full_portfolio.all())
        np.testing.assert_allclose(h.groupby("report_id").weight.sum(),1)
        self.assertEqual(diag["report_security_groups_with_conflicting_source_asset_class"],2)

    def test_isin_check_digit(self):
        self.assertTrue(valid_isin("RU0009029540"))
        self.assertFalse(valid_isin("RU0009029541"))
        self.assertFalse(valid_isin("10301481B"))


class RunnerAndReportingChecks(unittest.TestCase):
    def test_eligible_panel_ignores_negative_positions_in_rejected_reports(self):
        from run_title_name_regressions import make_portfolio_panel
        inputs,spec=SourceNormalizationChecks().fixture()
        with patch("title_name_regressions_data.read_holdings_sources",return_value=inputs):
            inventory,reports,intervals,*_=normalize_holdings("unused",spec)
        bad=inventory.iloc[[0]].copy()
        bad["report_id"]="rejected";bad["report_date"]=pd.Timestamp("2015-01-01");bad["weight"]=-.5
        inventory=pd.concat([inventory,bad],ignore_index=True)
        with tempfile.TemporaryDirectory(prefix="title_name_regressions_runner_") as tmp:
            root=Path(tmp)
            inventory.to_parquet(root/"holdings_inventory.parquet",index=False)
            intervals.to_parquet(root/"all_true_intervals.parquet",index=False)
            panel,diagnostics=make_portfolio_panel(root)
        self.assertEqual(len(panel),1)
        self.assertAlmostEqual(panel.DW.iloc[0],0)
        self.assertTrue(panel.AT.isna().all())
        self.assertEqual(diagnostics["target_rows_with_documented_Russian_category"],1)

    def test_primary_exclusion_priority_is_mutually_exclusive(self):
        from title_name_regressions_reporting import primary_reasons
        p=pd.DataFrame(dict(a=[True,True,False,False],b=[True,False,True,False]))
        reasons=primary_reasons(p,[("first",p.a),("second",p.b)],"accepted")
        self.assertEqual(reasons.tolist(),["first","first","second","accepted"])

    def signal_fixture(self,last="2020-08"):
        rng=np.random.default_rng(2601);rows=[]
        for company in range(12):
            for month in pd.period_range("2020-01",last,freq="M"):
                positive=int(rng.integers(1,4));day=month.end_time.normalize()
                for article in range(4):
                    key=f"{company}-{month}-{article}"
                    rows.append(dict(pair_id="p"+key,article_id="a"+key,company_id=str(company),event_date=day,
                        published_at=day.isoformat(),date_inferred=False,attempted=True,valid=True,relevance="relevant",
                        sentiment_label="positive" if article<positive else "negative",source="interfax"))
        candidates=pd.DataFrame(rows)
        base,_=monthly_states(candidates,candidates.company_id.unique(),"2020-01",last)
        base["log_news_count"]=np.log1p(base.n_relevant)
        spec=dict(signal_model=dict(candidate_lag_company_controls=[],minimum_company_usable_months=6,
            minimum_companies_per_month=10),time_and_scope=dict(signal_history_start="2020-01"))
        return candidates,base,spec

    def fit_fixture_variants(self,last="2020-08",integrity_failure=False):
        import run_title_name_regressions as runner
        candidates,base,spec=self.signal_fixture(last)
        objects={"news_candidates.parquet":candidates,"event_representatives.parquet":candidates,
                 "company_month_states.parquet":base}
        with patch.object(pd,"read_parquet",side_effect=lambda p:objects[p.name].copy()), \
             patch.object(runner,"announce"),patch.object(runner,"parquet"),patch.object(runner,"save"):
            if integrity_failure:
                with patch.object(runner,"fit_signal",side_effect=IntegrityError("corrupt key")):
                    return runner.make_signal_variants(Path("/synthetic-no-write"),spec,{"last_baseline_month":last},[])
            return runner.make_signal_variants(Path("/synthetic-no-write"),spec,{"last_baseline_month":last},[])

    def test_failed_endpoint_variant_does_not_abort_valid_main_signal(self):
        signals,status=self.fit_fixture_variants()
        self.assertEqual(status["main"]["status"],"ESTIMATED_SIGNAL")
        self.assertTrue(signals["main"].S.notna().any())
        self.assertEqual(status["exclude_endpoint_days"]["status"],"NOT_ESTIMABLE")
        self.assertTrue(signals["exclude_endpoint_days"].S.isna().all())
        self.assertIn("no_company_controls",status)

    def test_missing_main_signal_keeps_estimable_raw_tone(self):
        signals,status=self.fit_fixture_variants("2020-05")
        self.assertEqual(status["main"]["status"],"NOT_ESTIMABLE")
        self.assertTrue(signals["main"].S.isna().all())
        natural=signals["main"].loc[signals["main"].Tone.notna(),["company_id","month","Tone"]]
        frames=[];rng=np.random.default_rng(30)
        for fund in range(3):
            d=natural.copy();d["fund_id"]=str(fund)
            d["fund_security_id"]=d.fund_id+"|"+d.company_id
            d["plus"]=d.Tone.clip(lower=0);d["minus"]=(-d.Tone).clip(lower=0)
            d["y"]=.2*d.plus-.3*d.minus+rng.normal(scale=.05,size=len(d))
            frames.append(d)
        result=fit_panel_fe(pd.concat(frames,ignore_index=True),"y",["plus","minus"])
        self.assertGreater(result.diagnostics["nobs"],0)

    def test_structural_signal_integrity_error_remains_fatal(self):
        with self.assertRaises(IntegrityError):self.fit_fixture_variants(integrity_failure=True)


class RevisionIntegrityChecks(unittest.TestCase):
    def make_cache(self,root):
        code=root/"code";code.mkdir()
        for name in PREPARATION_MODULES:(code/name).write_text("fixed code "+name,encoding="utf-8")
        for name in PREPARED_FILES:(root/name).write_text("derived "+name,encoding="utf-8")
        spec=root/"spec.json";audit=root/"audit.json";source=root/"raw.csv"
        spec.write_text("spec",encoding="utf-8");audit.write_text("audit",encoding="utf-8");source.write_text("source",encoding="utf-8")
        binding=dict(database_path=str(root/"results.sqlite3"),logical_results_sha256="a"*64)
        catalog=dict(views=["SELECT source"])
        identity=preparation_identity(spec,audit,[source],catalog,binding,code)
        state=dict(preparation_provenance=seal_preparation(root,identity))
        return state,identity,(spec,audit,[source],catalog,binding,code)

    def test_unchanged_preparation_resumes(self):
        with tempfile.TemporaryDirectory(prefix="title_name_integrity_") as temp:
            root=Path(temp);state,identity,args=self.make_cache(root)
            self.assertTrue(validate_preparation(root,state,preparation_identity(*args)))

    def test_changed_preparation_dependency_rejects_cache(self):
        with tempfile.TemporaryDirectory(prefix="title_name_integrity_") as temp:
            root=Path(temp);state,identity,args=self.make_cache(root)
            (args[-1]/PREPARATION_MODULES[0]).write_text("changed code",encoding="utf-8")
            with self.assertRaises(IntegrityError):validate_preparation(root,state,preparation_identity(*args))

    def test_changed_cached_output_rejects_cache(self):
        with tempfile.TemporaryDirectory(prefix="title_name_integrity_") as temp:
            root=Path(temp);state,identity,args=self.make_cache(root)
            (root/PREPARED_FILES[0]).write_text("tampered",encoding="utf-8")
            with self.assertRaises(IntegrityError):validate_preparation(root,state,identity)

    def test_changed_source_with_same_size_and_mtime_rejects_cache(self):
        with tempfile.TemporaryDirectory(prefix="title_name_integrity_") as temp:
            root=Path(temp);state,identity,args=self.make_cache(root)
            source=args[2][0];before=source.stat();source.write_text("SOURCE",encoding="utf-8")
            os.utime(source,ns=(before.st_atime_ns,before.st_mtime_ns))
            with self.assertRaises(IntegrityError):validate_preparation(root,state,preparation_identity(*args))

    def test_changed_catalog_or_audit_rejects_cache(self):
        with tempfile.TemporaryDirectory(prefix="title_name_integrity_") as temp:
            root=Path(temp);state,identity,args=self.make_cache(root)
            args[3]["views"]=["SELECT other_source"]
            with self.assertRaises(IntegrityError):validate_preparation(root,state,preparation_identity(*args))
            args[3]["views"]=["SELECT source"];args[1].write_text("different audit",encoding="utf-8")
            with self.assertRaises(IntegrityError):validate_preparation(root,state,preparation_identity(*args))

    def test_legacy_cache_is_rejected_not_restamped(self):
        from run_title_name_regressions import prepare
        with tempfile.TemporaryDirectory(prefix="title_name_legacy_") as temp:
            root=Path(temp);(root/"preparation.json").write_text(json.dumps(dict(spec_sha256="old",sources=[])),encoding="utf-8")
            with self.assertRaises(IntegrityError):prepare({},root)

    def ledger_fixture(self,root):
        path=root/"results.sqlite3"
        row=dict(pair_id="p",article_id="a",company_id="c",valid=True,text_sha256="text-hash",
            relevance="relevant",sentiment_label="positive",prompt_version=APPROVED_PROMPT_VERSION,model=APPROVED_RESULT_MODEL)
        body=json.dumps(row);db=sqlite3.connect(path)
        db.execute("CREATE TABLE results(pair_id TEXT,task_index INTEGER,valid INTEGER,result_json TEXT)")
        db.execute("INSERT INTO results VALUES(?,?,?,?)",("p",1,1,body));db.commit();db.close()
        digest=hashlib.sha256(json.dumps(["p",1,1,body],ensure_ascii=False,separators=(",",":")).encode()+b"\n").hexdigest()
        index=pd.DataFrame([dict(pair_id="p",article_id="a",company_id="c",text_sha256="text-hash",input_issue=None)])
        return path,row,index,digest

    def test_exact_audited_logical_result_digest_passes(self):
        with tempfile.TemporaryDirectory(prefix="title_name_ledger_") as temp:
            path,row,index,digest=self.ledger_fixture(Path(temp))
            labels=read_labels(path,index,expected_logical_sha256=digest)
            self.assertEqual(labels.attrs["logical_results_sha256"],digest)

    def test_changed_label_or_model_or_prompt_rejects_old_audit(self):
        for field,value in [("sentiment_label","negative"),("model","other-model"),("prompt_version","other-prompt")]:
            with self.subTest(field=field),tempfile.TemporaryDirectory(prefix="title_name_ledger_") as temp:
                path,row,index,digest=self.ledger_fixture(Path(temp));row[field]=value
                db=sqlite3.connect(path);db.execute("UPDATE results SET result_json=?",(json.dumps(row),));db.commit();db.close()
                with self.assertRaises(IntegrityError):read_labels(path,index,expected_logical_sha256=digest)

    def test_input_with_self_consistent_new_manifest_rejects_frozen_R1(self):
        observed=dict(input_sha256="a"*64,mapping_pairs=1,prompt_version=APPROVED_PROMPT_VERSION,
                      prompt_sha256="b"*64,model_file="model.gguf")
        spec=dict(observed_input_audit=observed,resources=dict(llm_database="unused"))
        audit=dict(audit_execution_complete=True,status="AUDIT_COMPLETE",llm=dict(status="READY",integrity_failures=[],input=dict(sha256="a"*64)))
        with self.assertRaises(IntegrityError):verified_llm_binding(spec,audit,dict(input_sha256="c"*64,total=1),
            dict(input_sha256="c"*64),input_manifest_path="unused",run_manifest_path="unused")

    def test_nonpositive_or_nonfinite_variance_never_exports_confidence_interval(self):
        from title_name_regressions_estimation import PanelResult
        for variance in [0.,-.1,np.inf,np.nan]:
            with self.subTest(variance=variance):
                model=PanelResult(["x"],np.array([.2]),np.array([[variance]]),pd.DataFrame(),np.array([]),dict(inference_df=10))
                row=model.table()[0]
                self.assertEqual(row["inference_status"],"NOT_ESTIMABLE_COVARIANCE")
                self.assertTrue(all(np.isnan(row[k]) for k in ["se","ci95_low","ci95_high","p"]))
        model.covariance=np.array([[.01]])
        self.assertEqual(model.table()[0]["inference_status"],"ESTIMABLE")


if __name__=="__main__":
    unittest.main(verbosity=2)
