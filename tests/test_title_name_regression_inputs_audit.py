"""Synthetic tests for W1 identity, exclusion and preservation guarantees."""
from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_title_name_regression_inputs.py"
SPEC = importlib.util.spec_from_file_location("w1_audit", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def write_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


class AuditSafetyTests(unittest.TestCase):
    def test_large_disclosure_json_field_is_read_without_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"wide.csv.gz"
            value="x"*200000
            write_csv(path,[{"fund_id":"f1","asset_rows_json":value}])
            self.assertEqual(next(audit.rows(path))["asset_rows_json"],value)

    def test_pair_id_matches_observed_runner_example(self):
        self.assertEqual(audit.stable_pair_id("a_2b9653c131c8e35ec1c947a3b1f9eb20", "c_0c6a71c3e06621a192602bc4"),
                         "p_481e0210d44e87a5af16acfe43562cf3")
        self.assertNotEqual(audit.stable_pair_id("a", "b"), audit.stable_pair_id("b", "a"))

    def test_wal_visible_but_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp)/"live.sqlite3"
            writer = sqlite3.connect(db)
            try:
                writer.execute("PRAGMA journal_mode=WAL")
                writer.execute("PRAGMA wal_autocheckpoint=0")
                writer.execute("CREATE TABLE results (x INTEGER)")
                writer.execute("INSERT INTO results VALUES (7)")
                writer.commit()
                reader = audit.read_only_sqlite(db)
                try:
                    self.assertEqual(reader.execute("SELECT x FROM results").fetchall(), [(7,)])
                    with self.assertRaises(sqlite3.OperationalError):
                        reader.execute("INSERT INTO results VALUES (8)")
                finally:
                    reader.close()
            finally:
                writer.close()

    def test_errors_and_irrelevance_are_not_neutral(self):
        self.assertEqual(audit.outcome({"valid":False,"input_issue":"empty_archived_text"},0), "input_error")
        self.assertEqual(audit.outcome({"valid":False,"sentiment_label":"neutral"},0), "generation_error")
        self.assertEqual(audit.outcome({"valid":True,"relevance":"irrelevant","sentiment_label":None},1), "valid_irrelevant")
        self.assertEqual(audit.outcome({"valid":True,"relevance":"irrelevant","sentiment_label":"neutral"},1), "schema_error")
        self.assertEqual(audit.outcome({"valid":True,"relevance":"relevant","sentiment_label":"neutral"},1), "valid_relevant")
        self.assertEqual(audit.outcome({"valid":True,"input_issue":"empty_archived_text"},1), "schema_error")
        self.assertEqual(audit.outcome({"valid":False,"relevance":"relevant","sentiment_label":"neutral"},1), "schema_error")

    def test_exact_dates_and_calendar_gaps(self):
        self.assertIsNone(audit.parse_date("2020-01-01invalid"))
        self.assertIsNone(audit.parse_date("2020-02-30"))
        report = audit.fund_intervals({("f",date(2020,1,15)),("f",date(2020,1,31)),("f",date(2020,2,29)),("f",date(2020,5,20)),("g",date(2021,1,1))})
        self.assertEqual(report["month_gap_distribution"], {"0":1,"1":1,"3":1})
        self.assertEqual(report["missing_calendar_months_between_reports"], 2)
        self.assertEqual(report["consecutive_exact_calendar_month_end_intervals"],1)
        self.assertEqual(report["exact_calendar_month_end_closing_2018_01_to_2026_05"],1)
        self.assertNotEqual(audit.timestamp_key("2020-01-01T11:00:00"),audit.timestamp_key("2020-01-01T12:00:00"))

    def test_equity_words_in_a_fund_name_are_not_equity_categories(self):
        base={"table_type":"asset","row_type":"asset_item","parent_category":"Акции российских эмитентов"}
        self.assertTrue(audit.is_equity_leaf(base))
        self.assertFalse(audit.is_equity_leaf(dict(base,parent_category="ПРОМСВЯЗЬ - АКЦИИ")))
        self.assertFalse(audit.is_equity_leaf(dict(base,parent_category="FinEx CASH EQUIVALENTS UCITS ETF (RUB Hedged Shares)")))

    def test_protected_sources_and_existing_reports_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in (root/"data",root/"run"):
                with self.assertRaises(ValueError):
                    audit.write_report(folder/"bad.json",{},root/"data",root/"run")
            target=root/"reports/ok.json"
            audit.write_report(target,{"ok":1},root/"data",root/"run")
            with self.assertRaises(FileExistsError):
                audit.write_report(target,{},root/"data",root/"run")
            self.assertEqual(json.loads(target.read_text()), {"ok":1})

    def test_leaf_positions_do_not_multiply_news_or_accept_ambiguous_isins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            base={"fund_id":"f1","source_path":"report1.xls","report_date":"2020-01-31","table_type":"asset","row_type":"asset_item",
                  "parent_category":"Акции","registration_number_security":"R1","item_name":"Share 1","share_pct":"10","amount_thousand_rub":"100","net_asset_value_rub":"1000000"}
            records=[base,dict(base,row_type="asset_category"),dict(base,registration_number_security="AMB"),dict(base,registration_number_security="R2"),dict(base,parent_category="Облигации")]
            write_csv(root/"funds/mac-20261008/fund_holdings_long.csv.gz",records)
            dictionary=root/"dict.csv"
            write_csv(dictionary,[{"security_code":"R1","isin":"RU0000000001"}, {"security_code":"R2","isin":"RU0000000002"},
                                  {"security_code":"AMB","isin":"RU0000000001"},{"security_code":"AMB","isin":"RU0000000003"}])
            result=audit.audit_holdings(root,{"RU0000000001":{"c1"},"RU0000000002":{"c2","c3"}},{"c1"},[dictionary])
            self.assertEqual(result["counts"]["all_rows"],5)
            self.assertEqual(result["counts"]["leaf_asset_rows"],4)
            self.assertEqual(result["counts"]["equity_leaf_rows"],3)
            self.assertEqual(result["counts"]["equity_unambiguous_company_rows"],1)
            self.assertEqual(result["counts"]["equity_ambiguous_isin"],1)
            self.assertEqual(result["counts"]["equity_ambiguous_company"],1)
            self.assertEqual(result["eligible_relevant_companies_in_holdings"],1)
            self.assertEqual(result["all_asset_leaf_identity_bridge"]["counts"]["unique_company_rows"],2)


class AuditFullInputTests(unittest.TestCase):
    def fixture(self, root):
        run=root/"run"; run.mkdir()
        scripts=root/"scripts"; scripts.mkdir()
        runner=scripts/"68_run_gemma_title_full.py"
        runner.write_text("def pair_id(article_id, company_id):\n    return 'synthetic'\n")
        comparison=scripts/"43_compare_sentiment_models.py"
        comparison.write_text("PROMPT_VERSION = 'v1'\nSYSTEM_PROMPT = 'fixture'\n")
        inputs=[]; mapping=[]; results=[]
        cases=[("relevant","positive",True,None),("irrelevant",None,True,None),(None,None,False,"empty_archived_text"),(None,None,False,None)]
        for i,(relevance,polarity,valid,issue) in enumerate(cases,1):
            aid,cid=f"a{i}",f"c{i}"
            body="" if issue else "Full fixture article"
            h=hashlib.sha256(body.encode()).hexdigest()
            row={"article_id":aid,"company_id":cid,"pair_id":audit.stable_pair_id(aid,cid),"source":"rbc","published_at":"2020-01-10T12:00:00",
                 "match_scope":"title","match_signal":"name","text":body,"text_sha256":h}
            if issue: row["input_issue"]=issue
            inputs.append(row)
            mapping.append({"article_id":aid,"company_id":cid,"media":"rbc","date":"2020-01-10T12:00:00","match_signal":"name"})
            result={k:v for k,v in row.items() if k!="text"}
            result.update(valid=valid,relevance=relevance,sentiment_label=polarity,short_rationale="fixture")
            if not issue:
                result.update(model="gemma4_e4b_q4qat",prompt_version="v1",input_sha256=h,input_characters=len(body),text_truncated=False)
            results.append(result)
        raw=b"".join((json.dumps(row)+"\n").encode() for row in inputs)
        (run/"input.jsonl").write_bytes(raw)
        manifest={"input_sha256":hashlib.sha256(raw).hexdigest(),"runner_script_sha256":hashlib.sha256(runner.read_bytes()).hexdigest(),
                  "comparison_script_sha256":hashlib.sha256(comparison.read_bytes()).hexdigest(),"prompt_version":"v1","prompt_sha256":hashlib.sha256(b"fixture").hexdigest()}
        for name in ("run_manifest.json","input_manifest.json"):
            (run/name).write_text(json.dumps(manifest))
        db=sqlite3.connect(run/"results.sqlite3")
        db.execute("CREATE TABLE results(pair_id TEXT PRIMARY KEY, task_index INTEGER UNIQUE, valid INTEGER, result_json TEXT, elapsed_seconds REAL)")
        db.executemany("INSERT INTO results VALUES (?,?,?,?,?)",[(row["pair_id"],i,int(row["valid"]),json.dumps(row),0) for i,row in enumerate(results,1)])
        db.commit();db.close()
        return run,mapping

    def audit_fixture(self,root,run,mapping):
        with patch.object(audit,"rows",return_value=iter(mapping)),patch.object(audit,"schema",return_value={"rows":len(mapping),"columns":list(mapping[0])}):
            return audit.audit_llm(root/"resources",run,root)[0]

    def test_all_pairs_attempted_can_contain_explicit_excluded_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run,mapping=self.fixture(root)
            result=self.audit_fixture(root,run,mapping)
            self.assertEqual(result["status"],"READY")
            self.assertEqual(result["counts"]["results"],4)
            self.assertEqual(result["counts"]["input_error"],1)
            self.assertEqual(result["counts"]["generation_error"],1)
            self.assertEqual(result["counts"]["valid_relevant"],1)
            self.assertEqual(result["eligible_relevant_companies"],1)
            self.assertEqual(result["integrity_failures"],[])

    def test_corrupt_prefix_hash_is_detected_without_rewriting_sqlite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run,mapping=self.fixture(root)
            db=sqlite3.connect(run/"results.sqlite3")
            row=json.loads(db.execute("SELECT result_json FROM results WHERE task_index=1").fetchone()[0]);row["input_sha256"]="0"*64
            db.execute("UPDATE results SET result_json=? WHERE task_index=1",(json.dumps(row),));db.commit();db.close()
            before=(run/"results.sqlite3").read_bytes()
            result=self.audit_fixture(root,run,mapping)
            self.assertEqual(result["status"],"BLOCKED")
            self.assertIn("input:result_prefix_hash_mismatch",result["integrity_failures"])
            self.assertEqual(before,(run/"results.sqlite3").read_bytes())


if __name__ == "__main__":
    unittest.main()
