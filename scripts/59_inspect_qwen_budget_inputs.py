"""Read-only inspection of active mapping size and the current LLM contract."""
import json
import pathlib
import sys

root = pathlib.Path(".")
sys.path.insert(0, str(root / "news_data/company_mapping_runs/run_20260930_01/python_packages_20260930"))
import pyarrow.parquet as pq

config_path = root / "config/company_lists/active_company_mapping.json"
config = json.loads(config_path.read_text(encoding="utf-8"))
print(json.dumps({"active_mapping_version": config["version"]}, ensure_ascii=True), flush=True)
active = pathlib.Path(config["active_directory"])
for name in ["cleanup_audit.json", "SUCCESS"]:
    path = active / name
    if path.exists():
        content = path.read_text(encoding="utf-8")
        if name.endswith(".json"):
            audit = json.loads(content)
            content = {key: audit[key] for key in ["raw_pairs", "clean_pairs", "active_companies", "source_unchanged"] if key in audit}
        print(json.dumps({"path": str(path), "content": content}, ensure_ascii=True), flush=True)
for field in ["article_company_map", "company_registry"]:
    path = active / config[field]
    print(json.dumps({"parquet": str(path), "rows": pq.ParquetFile(path).metadata.num_rows}), flush=True)
for path in [root / "news_data/company_mapping_runs/run_20261008_safe_otkritie_01/progress.json"]:
    if path.exists():
        print(json.dumps({"path": str(path), "content": path.read_text(encoding="utf-8")}, ensure_ascii=True), flush=True)
path = root / "scripts/43_compare_sentiment_models.py"
text = path.read_text(encoding="utf-8")
start = text.find("PROMPT_VERSION")
end = text.find("def ", start)
print(json.dumps({"prompt_contract": text[start:end]}, ensure_ascii=True), flush=True)
start = text.find("def build_user_prompt")
end = text.find("\ndef ", start + 1)
print(json.dumps({"user_prompt_function": text[start:end]}, ensure_ascii=True), flush=True)
