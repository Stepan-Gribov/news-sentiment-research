import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

scripts = Path(__file__).parent
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

runner = module("full_runner", scripts / "68_run_gemma_title_full.py")
comparison = module("comparison", scripts / "43_compare_sentiment_models.py")
mapper = module("mapper", scripts / "45_map_articles_to_companies.py")


class FullRunnerTests(unittest.TestCase):
    def test_stable_target_pair_id(self):
        self.assertEqual(runner.pair_id("a1", "c1"), runner.pair_id("a1", "c1"))
        self.assertNotEqual(runner.pair_id("a1", "c1"), runner.pair_id("a1", "c2"))

    def test_sqlite_projection_regenerates_committed_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            connection = sqlite3.connect(":memory:")
            connection.execute("CREATE TABLE results(task_index INTEGER,result_json TEXT)")
            connection.executemany("INSERT INTO results VALUES (?,?)", [(2, '{"pair_id":"p2"}'), (1, '{"pair_id":"p1"}')])
            path = Path(folder) / "results.jsonl"
            path.write_text("partial uncommitted projection")
            runner.regenerate_jsonl(connection, path)
            self.assertEqual([json.loads(line)["pair_id"] for line in path.read_text().splitlines()], ["p1", "p2"])

    def test_missing_text_is_explicit_not_neutral(self):
        result = runner.classify(dict(pair_id="p", text="", input_issue="empty_archived_text"), "unused", comparison)
        self.assertFalse(result["valid"])
        self.assertIsNone(result["sentiment_label"])
        self.assertEqual(result["input_issue"], "empty_archived_text")

    def test_full_input_join_and_safe_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source_dir = root / "map"
            source_dir.mkdir()
            (source_dir / "scripts").mkdir()
            shutil.copy2(scripts / "45_map_articles_to_companies.py", source_dir / "scripts/45_map_articles_to_companies.py")
            (root / "config/company_lists").mkdir(parents=True)
            (root / "config/company_lists/active_company_mapping.json").write_text(json.dumps(dict(run_root=str(source_dir))))
            date = datetime(2020, 1, 1)
            articles = [dict(media="rbc", link="https://example.org/a", title="Газпром увеличил прибыль", date=date, text="Газпром увеличил прибыль."),
                        dict(media="prime", link="https://example.org/b", title="ВТБ опубликовал отчет", date=date, text="")]
            pq.write_table(pa.Table.from_pylist(articles), root / "articles.parquet", row_group_size=1)
            records = []
            companies = [dict(company_id="gas_company", company_name="Газпром"), dict(company_id="bank", company_name="ВТБ")]
            for article, company in zip(articles, companies):
                records.append(dict(article_id=mapper.make_article_id(article["media"], article["link"], article["date"], article["title"], article["text"]),
                    date=date, media=article["media"], title=article["title"], company=company["company_name"], company_id=company["company_id"],
                    match_signal="name", matched_terms=json.dumps({"name": [company["company_name"]]}), article_url=article["link"]))
            pq.write_table(pa.Table.from_pylist(records, schema=mapper.OUTPUT_SCHEMA), source_dir / "map.parquet")
            pq.write_table(pa.Table.from_pylist(companies), source_dir / "registry.parquet")
            (source_dir / "SUCCESS").write_text("done")
            config = root / "mapping.json"
            config.write_text(json.dumps(dict(directory=str(source_dir), completion_marker="SUCCESS", article_company_map="map.parquet", company_registry="registry.parquet")))
            run = root / "run"
            run.mkdir()
            args = argparse.Namespace(run_dir=run, project_root=root, articles=root / "articles.parquet", mapping_config=config)
            manifest = runner.prepare(args, comparison, lambda **values: None)
            self.assertEqual(manifest["total"], 2)
            self.assertEqual(manifest["input_issues"], {"empty_archived_text": 1})
            self.assertEqual(runner.prepare(args, comparison, lambda **values: None), manifest)
            data = [json.loads(line) for line in (run / "input.jsonl").read_text().splitlines()]
            self.assertEqual(data[0]["text"], articles[0]["text"])
            self.assertEqual(data[0]["target_company"], "Газпром")
            self.assertEqual(data[0]["text_sha256"], hashlib.sha256(articles[0]["text"].encode()).hexdigest())


if __name__ == "__main__":
    unittest.main()
