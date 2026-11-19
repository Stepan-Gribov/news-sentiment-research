import importlib.util
import unittest
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

spec = importlib.util.spec_from_file_location("title_map", Path(__file__).with_name("60_build_title_name_mapping.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TitleNameTests(unittest.TestCase):
    def setUp(self):
        self.mapper_path = Path(__file__).with_name("45_map_articles_to_companies.py")
        self.registry = [dict(company_id="gazprom", company="Газпром", aliases="Газпром"),
                         dict(company_id="otkritie", company="Открытие Холдинг", aliases="Открытие Холдинг"),
                         dict(company_id="vtb", company="ВТБ", aliases="ВТБ; Банк ВТБ"),
                         dict(company_id="gaz", company="ГАЗ", aliases="ГАЗ")]
        mapper = module.load_matcher(self.mapper_path)
        patterns = module.name_patterns(mapper, self.registry)
        module.worker_init(str(self.mapper_path), patterns)

    def row(self, title, cid="gazprom"):
        return dict(media="rbc", link="https://example.org/1", date=None, company_id=cid, company=cid, title=title,
                    match_signal="name;industry", matched_terms='{"name":["Газпром"],"industry":["газ"]}')

    def test_title_inflection_and_ids_preserved(self):
        output = module.select_rows([self.row("Прибыль Газпрома выросла")])
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["article_id"], module.MATCHER.make_article_id("rbc", "https://example.org/1", None, "Прибыль Газпрома выросла", ""))
        self.assertEqual(output[0]["company_id"], "gazprom")
        self.assertEqual(output[0]["match_signal"], "name")
        self.assertNotIn("industry", output[0]["matched_terms"])

    def test_body_name_and_industry_not_sufficient(self):
        self.assertEqual(module.select_rows([self.row("Газ и нефть подорожали")]), [])
        self.assertEqual(module.select_rows([self.row("Газпромбанк увеличил прибыль")]), [])

    def test_ambiguous_otkritie_excluded(self):
        self.assertEqual(module.select_rows([self.row("Открытие нового салона", "otkritie")]), [])
        self.assertEqual(len(module.select_rows([self.row("Открытие Холдинг опубликовал отчет", "otkritie")])), 1)

    def test_alias_case_punctuation(self):
        output = module.select_rows([self.row("Банк «втб» опубликовал отчет", "vtb")])
        self.assertEqual(len(output), 1)

    def test_gaz_company_is_excluded_but_gazprom_remains(self):
        self.assertFalse(module.title_hits("ГАЗ увеличил производство"))
        self.assertFalse(module.title_hits("Экспорт газа вырос"))
        self.assertIn("gazprom", module.title_hits("Газпром увеличил прибыль"))

    def test_end_to_end_parallel_no_source_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            clean = root / "clean"
            scripts = root / "scripts"
            clean.mkdir()
            scripts.mkdir()
            shutil.copy2(self.mapper_path, scripts / self.mapper_path.name)
            (clean / "SUCCESS").write_text("Existing map complete")
            pq.write_table(pa.Table.from_pylist(self.registry), clean / "registry.parquet")
            rows = [self.row("Прибыль Газпрома выросла"), self.row("Газ и нефть подорожали"),
                    self.row("Открытие нового салона", "otkritie"), self.row("ВТБ выплатит дивиденды", "vtb")]
            for index, row in enumerate(rows):
                row["article_id"] = f"article_{index}"
                row.update(date=None, media="rbc", article_url=f"https://example.org/{index}")
                row["link"] = row["article_url"]
            pq.write_table(pa.Table.from_pylist(rows, schema=module.MATCHER.OUTPUT_SCHEMA), clean / "map.parquet")
            pq.write_table(pa.Table.from_pylist([{key: row.get(key) for key in ["media", "title", "link", "date", "text"]} for row in rows]), root / "articles.parquet")
            config = root / "active.json"
            config.write_text(json.dumps(dict(version="unchanged", active_directory=str(clean), run_root=str(root),
                                             completion_marker="SUCCESS", article_company_map="map.parquet", company_registry="registry.parquet")))
            protected = [config, clean / "map.parquet", clean / "registry.parquet"]
            before = [(p.stat().st_size, p.stat().st_mtime_ns) for p in protected]
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("60_build_title_name_mapping.py")),
                "--active-config", str(config), "--articles", str(root / "articles.parquet"), "--output-dir", str(root / "additional"), "--workers", "2", "--batch-size", "2"],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            audit = json.loads((root / "additional/audit.json").read_text())
            self.assertEqual(audit["title_name_pairs"], 2)
            self.assertEqual(audit["unique_articles"], 2)
            self.assertTrue((root / "additional/SUCCESS").is_file())
            self.assertEqual(before, [(p.stat().st_size, p.stat().st_mtime_ns) for p in protected])


if __name__ == "__main__":
    unittest.main()
