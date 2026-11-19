"""Read-only output QA and sample checks against the frozen title matcher."""
import importlib.util
import json
import pathlib
import random
import sys

root = pathlib.Path(".")
sys.path.insert(0, str(root / "news_data/company_mapping_runs/run_20260930_01/python_packages_20260930"))
import pyarrow.parquet as pq

run = root / "news_data/company_mapping_runs/run_20261008_title_name_01"
if not (run / "SUCCESS").is_file():
    raise SystemExit("Additional mapping is not complete")
audit = json.loads((run / "audit.json").read_text(encoding="utf-8"))
config = json.loads((root / "config/company_lists/active_company_mapping.json").read_text(encoding="utf-8"))
assert config["version"] == audit["source_mapping_version"]
print(json.dumps({"audit": audit, "main_mapping_still_active": config["version"]}), flush=True)
spec = importlib.util.spec_from_file_location("title_map", run / "scripts/60_build_title_name_mapping.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
mapper_path = pathlib.Path(config["run_root"]) / "scripts/45_map_articles_to_companies.py"
mapper = module.load_matcher(mapper_path)
registry = pq.read_table(run / "company_registry.parquet").to_pylist()
for row in registry:
    row["company"] = row["company_name"]
module.worker_init(str(mapper_path), module.name_patterns(mapper, registry))
file = pq.ParquetFile(run / "article_company_map_title_name.parquet")
assert file.metadata.num_rows == audit["title_name_pairs"]
rng = random.Random(20261008)
checked = 0
examples = []
for group in rng.sample(range(file.metadata.num_row_groups), min(5, file.metadata.num_row_groups)):
    rows = file.read_row_group(group).to_pylist()
    for row in rng.sample(rows, min(20, len(rows))):
        assert row["match_signal"] == "name"
        assert set(json.loads(row["matched_terms"])) == {"name"}
        assert row["company_id"] in module.title_hits(row["title"])
        if row["article_url"]:
            assert row["article_id"] == mapper.make_article_id(row["media"], row["article_url"], row["date"], row["title"], "")
        checked += 1
        if len(examples) < 3:
            examples.append({key: row[key] for key in ["title", "company", "media", "article_url"]})
print(json.dumps({"validated_random_pairs": checked, "examples": examples}, ensure_ascii=True), flush=True)
