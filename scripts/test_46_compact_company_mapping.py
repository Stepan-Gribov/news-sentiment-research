"""Canonical IDs, description preservation and no repeated article rows."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

spec = importlib.util.spec_from_file_location("compact", Path(__file__).with_name("46_build_company_registry_and_article_ids.py"))
compact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compact)


class CompactTests(unittest.TestCase):
    def test_canonical_ids_and_features_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            registry = [{"company_id": "c_A", "company": "Company A", "industry": "industry", "product": "product", "geography": "Russia", "status": "active"},
                        {"company_id": "c_B", "company": "Company B", "industry": "industry", "product": "product B", "geography": "Russia", "status": "active"}]
            pq.write_table(pa.Table.from_pylist(registry), p / "registry.parquet")
            pairs = [{"article_id": aid, "date": None, "media": "rbc", "title": aid, "company_id": cid, "match_signal": signal, "article_url": "https://example.com/" + aid}
                     for aid, cid, signal in [("a_1", "c_B", "industry"), ("a_1", "c_A", "name;product"), ("a_2", "c_A", "name")]]
            schema = pa.schema([(k, pa.timestamp("ns") if k == "date" else pa.string()) for k in pairs[0]])
            pq.write_table(pa.Table.from_pylist(pairs, schema=schema), p / "pairs.parquet")
            argv = ["compact", "--article-company-map", str(p / "pairs.parquet"), "--company-registry-input", str(p / "registry.parquet"),
                    "--company-registry", str(p / "out/registry.parquet"), "--article-ids-output", str(p / "out/articles.parquet"), "--batch-size", "1"]
            with patch.object(sys, "argv", argv):
                compact.main()
            result = pq.read_table(p / "out/articles.parquet").to_pylist()
            preserved = pq.read_table(p / "out/registry.parquet").to_pylist()
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["company_ids"], "c_A,c_B")
            self.assertEqual(result[0]["company_count"], 2)
            self.assertEqual(json.loads(result[0]["company_match_signals_json"])[0]["match_signal"], "name;product")
            self.assertEqual(preserved[0]["geography"], "Russia")
            self.assertEqual(preserved[0]["company_name"], "Company A")
            self.assertEqual([r["company_id"] for r in preserved], ["c_A", "c_B"])
            self.assertTrue((p / "pairs.parquet").is_file())

    def test_metadata_conflicts_rejected(self):
        first = dict(article_id="a_1", company_id="c_A", match_signal="name", date=None, media="rbc", title="Title", article_url="url")
        second = dict(first, company_id="c_B", title="Different title")
        with self.assertRaises(ValueError):
            compact.merge_article_rows([first, second])


if __name__ == "__main__":
    unittest.main()
