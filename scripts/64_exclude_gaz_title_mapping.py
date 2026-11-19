"""Remove only the GAZ issuer from a new additional-map version, not source data."""
import collections
import csv
import json
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(".")
sys.path.insert(0, str(ROOT / "news_data/company_mapping_runs/run_20260930_01/python_packages_20260930"))
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

source = ROOT / "news_data/company_mapping_runs/run_20261008_title_name_01"
destination = ROOT / "news_data/company_mapping_runs/run_20261008_title_name_no_gaz_02"
excluded_id = "c_90995d77be411a67f2ef2b72"
if not (source / "SUCCESS").is_file():
    raise SystemExit("Existing additional mapping is incomplete")
if destination.exists():
    raise SystemExit("Output version already exists; refusing to overwrite")
if shutil.disk_usage(ROOT).free < 1024**3:
    raise SystemExit("Insufficient disk headroom")
protected = [source / "article_company_map_title_name.parquet", source / "company_registry.parquet", source / "audit.json", ROOT / "config/company_lists/active_company_mapping.json"]
stamps = {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in protected}
registry = pq.read_table(source / "company_registry.parquet")
removed_company = registry.filter(pc.equal(registry["company_id"], excluded_id)).to_pylist()
if len(removed_company) != 1 or removed_company[0]["company_name"].casefold() != "газ":
    raise SystemExit("Excluded ID does not identify exactly the intended GAZ issuer")
started = time.monotonic()
destination.mkdir(exist_ok=False)
table = pq.read_table(source / "article_company_map_title_name.parquet")
old_counts = collections.Counter(table["company_id"].to_pylist())
original_pairs = table.num_rows
filtered = table.filter(pc.not_equal(table["company_id"], excluded_id))
new_counts = collections.Counter(filtered["company_id"].to_pylist())
expected = dict(old_counts)
removed_pairs = expected.pop(excluded_id)
if dict(new_counts) != expected or filtered.num_rows != original_pairs - removed_pairs:
    raise RuntimeError("Other company counts changed")
pq.write_table(filtered, destination / "article_company_map_title_name.parquet", compression="snappy", row_group_size=100_000)
registry_input = destination / ".registry_input.parquet"
pq.write_table(registry.filter(pc.not_equal(registry["company_id"], excluded_id)), registry_input, compression="snappy")
with (destination / "compact.log").open("x", encoding="utf-8") as log:
    subprocess.run([sys.executable, str(source / "scripts/46_build_company_registry_and_article_ids.py"),
        "--article-company-map", str(destination / "article_company_map_title_name.parquet"),
        "--company-registry-input", str(registry_input), "--company-registry", str(destination / "company_registry.parquet"),
        "--article-ids-output", str(destination / "article_company_ids.parquet")], check=True, stdout=log, stderr=subprocess.STDOUT)
compact = json.loads((destination / "article_company_ids.parquet.audit.json").read_text(encoding="utf-8"))
if compact["article_company_pairs_in_output"] != filtered.num_rows or compact["duplicate_source_pairs_collapsed"]:
    raise RuntimeError("Compact output reconciliation failed")
compact_ids = pq.read_table(destination / "article_company_ids.parquet", columns=["company_ids"])["company_ids"]
if pc.any(pc.match_substring(compact_ids, excluded_id)).as_py():
    raise RuntimeError("Excluded company remains in compact map")
if any((pathlib.Path(path).stat().st_size, pathlib.Path(path).stat().st_mtime_ns) != stamp for path, stamp in stamps.items()):
    raise RuntimeError("Protected source changed")
registry_input.unlink()
names = {row["company_id"]: row["company_name"] for row in registry.to_pylist()}
with (destination / "company_match_counts_title_name.csv").open("x", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["company_id", "company", "matches"])
    writer.writeheader()
    writer.writerows(dict(company_id=cid, company=names[cid], matches=count) for cid, count in new_counts.most_common())
audit = json.loads((source / "audit.json").read_text(encoding="utf-8"))
audit.update(title_name_pairs=filtered.num_rows, unique_articles=compact["unique_articles_output"], companies_matched=len(new_counts),
    companies_in_registry=registry.num_rows-1, media_pairs=dict(collections.Counter(filtered["media"].to_pylist())),
    missing_date_pairs=filtered["date"].null_count, excluded_company_id=excluded_id, excluded_company="ГАЗ",
    excluded_pairs=removed_pairs, original_title_name_pairs=original_pairs, previous_title_mapping=str(source),
    all_other_company_counts_unchanged=True, source_unchanged=True, active_mapping_not_replaced=True,
    elapsed_seconds=round(time.monotonic()-started, 2), operation="exclude_only_GAZ_from_additional_title_mapping")
for field, function in [("min_date", pc.min), ("max_date", pc.max)]:
    value = function(filtered["date"]).as_py()
    audit[field] = value.isoformat() if value else None
def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
save(destination / "audit.json", audit)
save(destination / "progress.json", {**audit, "stage": "completed"})
config = json.loads((source / "mapping_config.json").read_text(encoding="utf-8"))
config.update(directory=str(destination), excluded_company_ids=[excluded_id], previous_directory=str(source))
save(destination / "mapping_config.json", config)
save(destination / "company_exclusions.json", {"scope": "additional_title_name", "excluded_company_ids": [excluded_id], "excluded_company_names": ["ГАЗ"], "reason": "user_request_20261008_ambiguous_gaz"})
(destination / "README.md").write_text("# Title/name-only mapping without GAZ\n\nOnly the GAZ issuer is removed from pair, compact and company-registry outputs. All other company counts are unchanged. Previous additional mapping, main mapping and source news remain intact. Stable article/company IDs and title-only matching are retained.\n", encoding="utf-8")
(destination / "SUCCESS").write_text("GAZ excluded; every other company reconciled; source data preserved.\n", encoding="utf-8")
print(json.dumps({"result": audit, "top_3": [dict(company=names[cid], matches=count) for cid, count in new_counts.most_common(3)]}, ensure_ascii=True), flush=True)
