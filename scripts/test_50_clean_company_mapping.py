"""Regression tests for conservative company cleanup; no production file writes."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pyarrow as pa
import pyarrow.parquet as pq

spec = importlib.util.spec_from_file_location("cleaner", Path(__file__).with_name("50_clean_company_mapping.py"))
cleaner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleaner)


class CleanupTests(unittest.TestCase):
    def test_scope(self):
        for name in ["Санкт-Петербург", "Краснодарский край", "Минфин Магаданской области", "Московская Биржа", "СПБ Биржа"]:
            self.assertTrue(cleaner.excluded(name), name)
        for name in ["Магнит", "Полюс", "Открытие Холдинг", "ВЭБ.РФ", "Роснано", "Автодор (Государственная компания)", "Банк Санкт-Петербург"]:
            self.assertFalse(cleaner.excluded(name), name)

    def test_identity_safety(self):
        self.assertEqual(cleaner.base("РБК"), cleaner.base("РБК (бывш. РБК-ТВ)"))
        self.assertEqual(cleaner.base("АО Ювелит"), cleaner.base("Ювелит"))
        self.assertNotEqual(cleaner.base("Банк Открытие"), cleaner.base("Открытие Холдинг"))
        self.assertNotEqual(cleaner.base("РБК"), cleaner.base("РБКИС"))
        self.assertNotEqual(cleaner.base("Азот (Кемерово)"), cleaner.base("Азот (Березники)"))

    def test_registry_and_bad_shared_isin(self):
        rows = [
            ["Магнит", 1, "RU000A0JKQU8"], ["Полюс", 1, "RU000A0JNAA8"],
            ["Открытие Холдинг", 1, "RU000A0JABC1"],
            ["Открытие Холдинг (бывш. ФК Открытие)", 1, "RU000A0JABC1"],
            ["РБК", 1, "RU000A0JR6A6"], ["РБК (бывш. РБК-ТВ)", 1, "RU000A0JR6A6"],
            ["Санкт-Петербург", 1, "RU000A0JXYZ1"], ["СПБ Биржа", 1, "RU000A0JQ9P9"],
            ["МБСП", 1, "RU0005097020"], ["Южно-Уральский никелевый комбинат", 1, "RU0005097020"],
            ["Наука-Связь (ООО)", 1, "RU000A0JKAA1"], ["Наука-Связь (ОАО)", 1, "RU000A0JKAA2"],
        ]
        class Book:
            def __getitem__(self, key): return self
            def iter_rows(self, **kwargs): return iter([r + [None] * (10 - len(r)) for r in rows])
            def close(self): pass
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory) / "blacklist.csv"
            policy.write_text("rule_scope,company_id,company_name,match_term,action,reason,keep_match_if\n", encoding="utf-8")
            with patch.object(cleaner.openpyxl, "load_workbook", return_value=Book()):
                _, _, canonical, crosswalk, merges, conflicts, rules = cleaner.build_registry(Path("fake.xlsx"), policy)
        self.assertEqual(len(canonical), 10)
        self.assertEqual(len(merges), 2)
        self.assertEqual(sum(r["status"] == "excluded" for r in canonical.values()), 2)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(sum(r["action"] == "exclude_all_signals" for r in rules), 2)
        rbc = [r for r in crosswalk if cleaner.base(r["old_company"]) == "рбк"]
        self.assertEqual(len({r["company_id"] for r in rbc}), 1)

    def test_full_pair_cleanup_and_reconciliation(self):
        source_companies = [["Магнит", 1, "RU000A0JKQU8"], ["Полюс", 1, "RU000A0JNAA8"],
                            ["Открытие Холдинг", 1, "RU000A0JABC1"],
                            ["РБК", 1, "RU000A0JR6A6"], ["РБК (бывш. РБК-ТВ)", 1, "RU000A0JR6A6"],
                            ["Санкт-Петербург", 1, "RU000A0JXYZ1"]]
        class Book:
            def __getitem__(self, key): return self
            def iter_rows(self, **kwargs): return iter([r + [None] * (10 - len(r)) for r in source_companies])
            def close(self): pass
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            policy = p / "blacklist.csv"
            policy.write_text("rule_scope,company_id,company_name,match_term,action,reason,keep_match_if\ngeneric_keyword,,,новости,suppress_keyword_globally,generic,\n", encoding="utf-8")
            with patch.object(cleaner.openpyxl, "load_workbook", return_value=Book()):
                _, records, *_ = cleaner.build_registry(p / "fake.xlsx", policy)
            ids = {r["company"]: cid for cid, r in records.items()}
            def row(aid, name, terms):
                return dict(article_id=aid, date=None, media="rbc", title="Example", company=name,
                            company_id=ids[name], match_signal=";".join(terms), matched_terms=json.dumps(terms), article_url=f"https://example.com/{aid}")
            rows = [row("a1", "РБК", {"name": ["РБК"]}),
                    row("a1", "РБК (бывш. РБК-ТВ)", {"industry": ["медиа"]}),
                    row("a1", "Санкт-Петербург", {"name": ["Санкт-Петербург"]}),
                    row("a2", "РБК", {"product": ["новости"]}),
                    row("a3", "Магнит", {"name": ["Магнит"]})]
            schema = pa.schema([(k, pa.timestamp("ns") if k == "date" else pa.string()) for k in rows[0]])
            pq.write_table(pa.Table.from_pylist(rows, schema=schema), p / "source.parquet")
            original = (p / "source.parquet").read_bytes()
            argv = ["cleaner", "--companies", str(p / "fake.xlsx"), "--mapping", str(p / "source.parquet"),
                    "--blacklist", str(policy), "--output-dir", str(p / "clean")]
            with patch.object(cleaner.openpyxl, "load_workbook", return_value=Book()), patch.object(sys, "argv", argv):
                cleaner.main()
            result = pq.read_table(p / "clean/article_company_map_clean.parquet").to_pylist()
            audit = json.loads((p / "clean/cleanup_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["match_signal"], "name;industry")
            self.assertEqual(audit["duplicate_pairs_collapsed"], 1)
            self.assertEqual(audit["excluded_entity_pairs"], 1)
            self.assertEqual(audit["empty_after_ambiguous_or_generic_filter"], 1)
            self.assertEqual((p / "source.parquet").read_bytes(), original)
            self.assertTrue((p / "clean/SUCCESS").exists())
            argv[argv.index("--mapping") + 1] = str(p / "clean/article_company_map_clean.parquet")
            argv[argv.index("--output-dir") + 1] = str(p / "clean_second_pass")
            with patch.object(cleaner.openpyxl, "load_workbook", return_value=Book()), patch.object(sys, "argv", argv):
                cleaner.main()
            audit2 = json.loads((p / "clean_second_pass/cleanup_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit2["original_raw_pairs"], 5)
            self.assertEqual(audit2["total_duplicate_pairs_collapsed"], 1)
            self.assertEqual(audit2["clean_pairs"], 2)

    def test_explicit_publisher_exclusion_keeps_other_issuers(self):
        rows = [["Магнит", 1, "RU000A0JKQU8"], ["Полюс", 1, "RU000A0JNAA8"],
                ["Открытие Холдинг", 1, "RU000A0JABC1"],
                ["РБК", 1, "RU000A0JR6A6"], ["РБК (бывш. РБК-ТВ)", 1, "RU000A0JR6A6"],
                ["РБКИС", 1, "RU000A0JXYZ1"]]
        class Book:
            def __getitem__(self, key): return self
            def iter_rows(self, **kwargs): return iter([r + [None] * (10 - len(r)) for r in rows])
            def close(self): pass
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory) / "blacklist.csv"
            policy.write_text("rule_scope,company_id,company_name,match_term,action,reason,keep_match_if\n"
                              "entity,,РБК,,exclude_all_signals,publisher_noise,\n", encoding="utf-8")
            with patch.object(cleaner.openpyxl, "load_workbook", return_value=Book()):
                _, _, canonical, crosswalk, *_ = cleaner.build_registry(Path("fake.xlsx"), policy)
        rbc = [r for r in crosswalk if cleaner.base(r["old_company"]) == "рбк"]
        self.assertEqual(len(rbc), 2)
        self.assertTrue(all(r["status"] == "excluded" for r in rbc))
        self.assertTrue(all(r["exclusion_reason"] == "publisher_noise" for r in rbc))
        self.assertTrue(all(r["status"] == "active" for r in canonical.values()
                            if cleaner.base(r["company"]) != "рбк"))


if __name__ == "__main__":
    unittest.main()
