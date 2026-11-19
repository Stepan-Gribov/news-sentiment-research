import json
import os
from pathlib import Path
import shutil
import subprocess
import sqlite3
import sys

root = Path(".")
run = root / "news_data/llm_runs/gemma_title_name_no_gaz_20261008_01"
mode = sys.argv[1]
if mode == "prepare":
    pointer = root / "config/company_lists/additional_title_name_mapping.json"
    config = json.loads(pointer.read_text(encoding="utf-8"))
    if not (Path(config["directory"]) / "SUCCESS").is_file() or "no_gaz_02" not in config["directory"]:
        raise SystemExit("Required additional mapping without GAZ is not active/complete")
    if shutil.disk_usage(root).free < 8*1024**3:
        raise SystemExit("Insufficient disk headroom")
    run.mkdir(parents=True, exist_ok=False)
    (run / "scripts").mkdir()
    shutil.copy2(pointer, run / "mapping_config_frozen.json")
    print(json.dumps({"run_dir": str(run), "frozen_mapping": config["directory"], "free_gib": round(shutil.disk_usage(root).free/1024**3, 2)}), flush=True)
elif mode == "test":
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "news_data/company_mapping_runs/run_20260930_01/python_packages_20260930")
    environment["PYTHONUTF8"] = "1"
    subprocess.run([sys.executable, str(run / "scripts/test_68_gemma_full.py")], env=environment, check=True)
elif mode == "start":
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(run / "scripts/68_start_gemma_title_full_20261008.ps1")], check=True)
elif mode == "status":
    for name in ["progress.json", "runner.stderr.log", "runner.stdout.log", "server.stderr.log", "server.stdout.log"]:
        path = run / name
        if path.exists():
            print(json.dumps({"file": name, "content": path.read_text(encoding="utf-8", errors="replace")[-6500:]}, ensure_ascii=True), flush=True)
    for name in ["input.jsonl.partial", "input.jsonl", "gemma_results.jsonl", "results.sqlite3"]:
        path = run / name
        if path.exists():
            print(json.dumps({"file": name, "bytes": path.stat().st_size}), flush=True)
elif mode == "snapshot":
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(root / "scripts/69_gemma_title_progress.ps1")], check=True)
elif mode == "results_check":
    connection = sqlite3.connect((run / "results.sqlite3").resolve().as_uri()+"?mode=ro", uri=True)
    rows = []
    for (body,) in connection.execute("SELECT result_json FROM results WHERE valid=0 LIMIT 5"):
        value = json.loads(body)
        rows.append({key: value.get(key) for key in ["pair_id", "input_issue", "first_error", "retry_error"]})
    print(json.dumps({"invalid_examples": rows}, ensure_ascii=True), flush=True)
    connection.close()
else:
    raise SystemExit("Unknown mode")
