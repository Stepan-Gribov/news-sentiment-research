"""Read-only preflight for the full additional-title Gemma run."""
import json
import pathlib
import sqlite3
import subprocess

root = pathlib.Path(".")
for path in [root / "models/llm/gemma-4-E4B_q4_0-it.gguf", pathlib.Path("./local/AppData/Local/Microsoft/WinGet/Packages/ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe/llama-server.exe")]:
    print(json.dumps({"file": str(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else None}), flush=True)
for label, command in [
    ("gpu", "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json"),
    ("memory", "Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize,FreePhysicalMemory | ConvertTo-Json"),
    ("llm_processes", "Get-CimInstance Win32_Process | Where-Object {$_.Name -match 'llama|python' -and $_.CommandLine -match 'llama|sentiment|gemma|target_llm'} | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json"),
    ("llm_ports", "Get-NetTCPConnection -State Listen | Where-Object {$_.LocalPort -ge 8080 -and $_.LocalPort -le 8099} | Select-Object LocalAddress,LocalPort,OwningProcess | ConvertTo-Json"),
]:
    value = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True, errors="replace")
    print(json.dumps({"check": label, "stdout": value.stdout, "stderr": value.stderr}, ensure_ascii=True), flush=True)
connection = sqlite3.connect((root / "news_search/news.sqlite3").resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
connection.execute("PRAGMA query_only=ON")
print(json.dumps({"article_schema": connection.execute("PRAGMA table_info(articles)").fetchall(), "indices": connection.execute("PRAGMA index_list(articles)").fetchall()}, ensure_ascii=True), flush=True)
for row in connection.execute("PRAGMA index_list(articles)").fetchall():
    print(json.dumps({"index_name": row[1], "columns": connection.execute('PRAGMA index_info("' + row[1].replace('"', '""') + '")').fetchall()}), flush=True)
print(json.dumps({"article_sample": connection.execute("SELECT id,source,url,title,length(text),published_at FROM articles LIMIT 2").fetchall()}, ensure_ascii=True), flush=True)
connection.close()
log = root / "news_data/validation/sber_random300_20261006_01/server.stderr.log"
if log.exists():
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    print(json.dumps({"previous_gemma_gpu_log": [line for line in lines if any(term in line.lower() for term in ["vulkan", "cuda", "device", "offload", "buffer size"])][:50]}, ensure_ascii=True), flush=True)
