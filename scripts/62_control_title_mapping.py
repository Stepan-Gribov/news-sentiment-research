"""Prepare/test/start/check only the additional title/name mapping on Windows."""
import json
import csv
import os
import pathlib
import shutil
import subprocess
import sys

root = pathlib.Path(".")
run = root / "news_data/company_mapping_runs/run_20261008_title_name_01"
mode = sys.argv[1]
if mode == "prepare":
    if shutil.disk_usage(root).free < 3 * 1024**3:
        raise SystemExit("Insufficient disk headroom")
    run.mkdir(exist_ok=False)
    (run / "scripts").mkdir(exist_ok=False)
    print(json.dumps({"prepared": str(run), "free_gib": round(shutil.disk_usage(root).free/1024**3, 2)}), flush=True)
elif mode == "test":
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "news_data/company_mapping_runs/run_20260930_01/python_packages_20260930")
    environment["PYTHONUTF8"] = "1"
    subprocess.run([sys.executable, str(run / "scripts/test_60_title_name_mapping.py")], env=environment, check=True)
elif mode == "start":
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(run / "scripts/60_start_title_name_20261008.ps1")], check=True)
elif mode == "status":
    for name in ["progress.json", "audit.json", "runner.stderr.log", "runner.stdout.log", "compact.log"]:
        path = run / name
        if path.exists():
            print(json.dumps({"file": name, "content": path.read_text(encoding="utf-8", errors="replace")[-4500:]}, ensure_ascii=True), flush=True)
    result = subprocess.run(["powershell", "-NoProfile", "-Command", "Get-ScheduledTask -TaskName 'CompanyTitleName_20261008_01' | Select-Object TaskName,State | ConvertTo-Json"], capture_output=True, text=True, errors="replace")
    print(result.stdout, flush=True)
elif mode == "top":
    pointer = root / "config/company_lists/additional_title_name_mapping.json"
    if pointer.is_file():
        run = pathlib.Path(json.loads(pointer.read_text(encoding="utf-8"))["directory"])
    if not (run / "SUCCESS").is_file():
        raise SystemExit("Mapping is not complete")
    with (run / "company_match_counts_title_name.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: (-int(row["matches"]), row["company_id"]))
    print(json.dumps([dict(company=row["company"], matches=int(row["matches"])) for row in rows[:20]], ensure_ascii=True))
else:
    raise SystemExit("Unknown mode")
