"""Synthetic W2 integration contract tests; no empirical panel is run."""
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import pandas as pd

from title_name_regressions_core import IntegrityError,sha256_file
from title_name_regressions_crosswalk import (registration_key,select_exact_edges,merged_registration_map,load_approved_w2,
    affected_report_dates,scheduled_maturity_facts,maturity_interval_flags)


class CrosswalkContractChecks(unittest.TestCase):
    def fixture(self):
        row=dict(registration_code_normalized="10301481B",isin="RU0009029540",mapping_status="EXACT_CURRENT_CONFIRMED",
            company_join_status="UNIQUE_STATIC",company_ids_static=json.dumps(["c"]),
            source_evidence=json.dumps([dict(source_url="https://iss.moex.com/iss/securities.json",source_sha256="a"*64)]),
            security_types=json.dumps(["common_share"]),historical_baseline_eligible=False,
            identity_valid_from=None,identity_valid_to=None)
        return pd.DataFrame([row]),pd.DataFrame(dict(isin=["RU0009029540"],company_id=["c"]))

    def test_normalization_is_typography_only_not_homoglyph_or_suffix_matching(self):
        self.assertEqual(registration_key(" 1–01–00077–A "),"1-01-00077-A")
        self.assertNotEqual(registration_key("10401000В"),registration_key("10401000B"))
        self.assertNotEqual(registration_key("10301481B-01"),registration_key("10301481B"))

    def test_unique_current_link_is_explicitly_not_historically_validated(self):
        frame,bridge=self.fixture();edges,audit=select_exact_edges(frame,bridge,version="test-v1")
        self.assertEqual(len(edges),1)
        self.assertFalse(edges.historical_identity_verified.any())
        self.assertEqual(edges.w2_version.iloc[0],"test-v1")

    def test_conflicting_or_ambiguous_edges_are_excluded(self):
        for field,value in [("mapping_status","CONFLICT"),("company_join_status","AMBIGUOUS")]:
            frame,bridge=self.fixture();frame[field]=value
            edges,audit=select_exact_edges(frame,bridge,version="test-v1")
            self.assertEqual(len(edges),0);self.assertNotEqual(audit.reason.iloc[0],"ACCEPTED_CURRENT_STATIC_ONLY")

    def test_duplicate_code_never_expands_the_holdings_join(self):
        frame,bridge=self.fixture();frame=pd.concat([frame,frame],ignore_index=True)
        with self.assertRaises(IntegrityError):select_exact_edges(frame,bridge,version="test-v1")

    def test_foreign_company_mapping_or_missing_evidence_is_not_accepted(self):
        for field,value in [("company_ids_static",'["other"]'),("source_evidence","[]")]:
            frame,bridge=self.fixture();frame[field]=value
            self.assertEqual(len(select_exact_edges(frame,bridge,version="test-v1")[0]),0)

    def test_new_historical_validity_claim_requires_separate_review(self):
        frame,bridge=self.fixture();frame["identity_valid_from"]="2000-01-01"
        with self.assertRaises(IntegrityError):select_exact_edges(frame,bridge,version="test-v1")

    def test_explicit_unresolved_identity_code_is_blocked(self):
        frame,bridge=self.fixture()
        edges,audit=select_exact_edges(frame,bridge,version="test-v1",blocked_codes=["10301481B"])
        self.assertEqual(len(edges),0)
        self.assertEqual(audit.reason.iloc[0],"UNRESOLVED_TEMPORAL_OR_IDENTITY_REVIEW")

    def test_old_and_new_identifier_contradiction_is_blocked_not_overwritten(self):
        frame,bridge=self.fixture();edges,_=select_exact_edges(frame,bridge,version="test-v1")
        old=pd.DataFrame(dict(security_code=["10301481B"],isin=["RU0009029557"]))
        mapping,conflicts=merged_registration_map(old,edges)
        self.assertIn("10301481B",conflicts);self.assertNotIn("10301481B",mapping)

    def overlay_fixture(self,*,ambiguous=False,contradict_company=False,temporal=False):
        import title_name_regressions_test as base_tests
        from title_name_regressions_data import normalize_holdings
        inputs,spec=base_tests.SourceNormalizationChecks().fixture(ambiguous)
        report,leaf,totals,securities,bridge,market=inputs
        securities=securities.iloc[0:0]
        frame,_=self.fixture()
        if contradict_company:frame["company_ids_static"]='["other"]'
        exclusions=pd.DataFrame([dict(registration_code_normalized="10301481B",report_date="2018-01-31")] if temporal else [],
                                columns=["registration_code_normalized","report_date"])
        overlay=dict(frame=frame,metadata=dict(version="test-v1"),blocked_codes=[],temporal_exclusions=exclusions)
        with patch("title_name_regressions_data.read_holdings_sources",return_value=(report,leaf,totals,securities,bridge,market)):
            outputs=normalize_holdings("unused",spec,crosswalk=overlay)
        return outputs,overlay

    def test_overlay_resolves_code_and_preserves_entire_portfolio_amount(self):
        (h,r,intervals,*_),overlay=self.overlay_fixture()
        self.assertEqual(int(h.target_security_candidate.sum()),2)
        self.assertTrue(r.full_portfolio.all())
        self.assertTrue(h.loc[h.w2_static_link_used.eq(True),"isin"].eq("RU0009029540").all())
        self.assertTrue(h.groupby("report_id").weight.sum().sub(1).abs().lt(1e-12).all())
        self.assertIn("selection_audit",overlay)

    def test_ambiguous_company_keeps_identifier_and_portfolio_but_no_target(self):
        (h,r,intervals,*_),overlay=self.overlay_fixture(ambiguous=True)
        self.assertFalse(h.target_security_candidate.any())
        self.assertTrue(h.loc[h.w2_static_link_used.eq(True),"isin"].eq("RU0009029540").all())
        self.assertTrue(r.full_portfolio.all())

    def test_company_contradiction_cannot_become_target(self):
        (h,r,intervals,*_),overlay=self.overlay_fixture(contradict_company=True)
        self.assertFalse(h.target_security_candidate.any())
        self.assertTrue(r.full_portfolio.all())

    def test_temporal_conflict_excludes_only_actual_report_date(self):
        (h,r,intervals,*_),overlay=self.overlay_fixture(temporal=True)
        target=h.loc[h.bucket_kind.eq("instrument_leaf")].set_index("report_date").target_security_candidate
        self.assertFalse(target.loc[pd.Timestamp("2018-01-31")])
        self.assertTrue(target.loc[pd.Timestamp("2018-02-28")])
        self.assertTrue(r.full_portfolio.all())

    def test_R4_three_early_dates_do_not_ban_later_same_codes(self):
        codes=["1-02-06556-A","1-03-02472-A","1-06-10167-F"]
        dates=["2011-09-30","2008-09-30","2008-09-30"]
        exclusions=pd.DataFrame(dict(registration_code_normalized=codes,report_date=dates))
        holdings=pd.DataFrame([dict(registration_match_key=code,report_date=date) for code,early in zip(codes,dates) for date in [early,"2018-01-31"]])
        self.assertEqual(affected_report_dates(holdings,exclusions).tolist(),[True,False,True,False,True,False])

    def test_only_official_MATDATE_is_screened_not_yield_or_trading_start(self):
        frame,_=self.fixture()
        facts=[dict(field="MATDATE",source_title="Дата погашения",value="2020-02-15",source_url="source",source_sha256="a"*64),
               dict(field="ISSUEDATE",source_title="Дата начала торгов",value="2010-01-01"),
               dict(field="BUYBACKDATE",source_title="Дата к которой рассчитывается доходность",value="2020-02-20"),
               dict(field="EARLYREPAYMENT",source_title="Возможен досрочный выкуп",value=0)]
        frame["dated_source_facts"]=json.dumps(facts,ensure_ascii=False)
        observed=scheduled_maturity_facts(frame)
        self.assertEqual(len(observed),1)
        self.assertEqual(observed.scheduled_maturity_date.iloc[0],pd.Timestamp("2020-02-15"))
        self.assertEqual(observed.interpretation.iloc[0],"SCHEDULED_NOT_CONFIRMED_ACTUAL_REDEMPTION")

    def test_scheduled_maturity_screen_uses_actual_interval_without_expansion(self):
        panel=pd.DataFrame(dict(interval_id=["before","contains","after"],security_id=["isin:RU0009029540"]*3,
            start_date=pd.to_datetime(["2019-12-31","2020-01-31","2020-02-29"]),
            end_date=pd.to_datetime(["2020-01-31","2020-02-29","2020-03-31"])))
        facts=pd.DataFrame(dict(security_id=["isin:RU0009029540"]*2,scheduled_maturity_date=pd.to_datetime(["2020-02-15"]*2)))
        self.assertEqual(maturity_interval_flags(panel,facts).tolist(),[False,True,False])
        self.assertEqual(len(panel),3)

    def release_fixture(self,root):
        folder=root/"test-v1";folder.mkdir()
        def write(path,value):path.write_text(json.dumps(value),encoding="utf-8")
        def stamp(path):return dict(path=str(path),sha256=sha256_file(path))
        frame,_=self.fixture();frame.to_parquet(folder/"crosswalk.parquet",index=False)
        write(folder/"conflicts.json",[]);write(folder/"temporal_review_queue.json",[])
        summary=dict(version="test-v1",status="COMPLETE_WITH_LIMITS",collection_complete=True,counts={},temporal_gate="No historical validity asserted")
        write(folder/"summary.json",summary)
        qa=dict(status="PASS",summary_sha256=sha256_file(folder/"summary.json"),checks=[dict(check="synthetic",passed=True)],public_sources_verified=1)
        write(folder/"independent_qa.json",qa)
        release=dict(status="PASS",summary=stamp(folder/"summary.json"),qa=stamp(folder/"independent_qa.json"),
            artifacts=[stamp(folder/name) for name in ["crosswalk.parquet","conflicts.json","temporal_review_queue.json"]])
        write(folder/"release_manifest.json",release);write(root/"handoff.json",dict(ready_for_handoff=True))
        config=dict(ready_for_handoff=True,scope="EXPLORATORY_CURRENT_STATIC_ONLY",w2_version="test-v1",
            handoff=stamp(root/"handoff.json"),release=stamp(folder/"release_manifest.json"))
        review=dict(status="COMPLETE_ACCEPTED_FOR_PROVISIONAL_DW_WITH_FINDINGS",ready_for_handoff=True,
            crosswalk_release=dict(version="test-v1"),specification=dict(sha256="1"*64),temporal_queue_holdings_check=dict(records=[]))
        write(root/"r4.json",review)
        config["r4_review"]=stamp(root/"r4.json");config["spec_sha256"]="1"*64
        write(root/"input.json",config)
        return root/"input.json",folder

    def test_pinned_verified_release_loads_with_explicit_static_scope(self):
        with tempfile.TemporaryDirectory(prefix="w2_contract_") as temp:
            path,folder=self.release_fixture(Path(temp));data=load_approved_w2(path)
            self.assertEqual(data["metadata"]["version"],"test-v1")
            self.assertFalse(data["metadata"]["historical_identity_verified"])

    def test_changed_release_artifact_rejects_before_panel(self):
        with tempfile.TemporaryDirectory(prefix="w2_contract_") as temp:
            path,folder=self.release_fixture(Path(temp));(folder/"conflicts.json").write_text("[1]",encoding="utf-8")
            with self.assertRaises(IntegrityError):load_approved_w2(path)

    def test_unready_handoff_is_not_accepted_even_with_valid_pinned_hash(self):
        with tempfile.TemporaryDirectory(prefix="w2_contract_") as temp:
            path,folder=self.release_fixture(Path(temp));config=json.loads(path.read_text())
            handoff=Path(config["handoff"]["path"]);handoff.write_text('{"ready_for_handoff": false}',encoding="utf-8")
            config["handoff"]["sha256"]=sha256_file(handoff);path.write_text(json.dumps(config),encoding="utf-8")
            with self.assertRaises(IntegrityError):load_approved_w2(path)


if __name__=="__main__":unittest.main(verbosity=2)
