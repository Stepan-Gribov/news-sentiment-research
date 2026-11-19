from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def process_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0 and str(pid) in result.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def runtime_pid(root: Path) -> int:
    path = root / "_runtime" / "download.pid"
    try:
        return int(path.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return 0


def scheduled_task_state(task_name: str) -> str:
    if os.name != "nt" or not task_name:
        return ""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", f"(Get-ScheduledTask -TaskName '{task_name}' -ErrorAction SilentlyContinue).State"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def nonempty(values: list[object]) -> int:
    return sum(bool(str(value or "").strip()) for value in values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show one progress snapshot for the full Frank Media archive.")
    parser.add_argument("--root", type=Path, default=Path("data/frank_full_archive"))
    parser.add_argument("--task-name", default="FrankMediaFullArchive")
    args = parser.parse_args()

    links_dir = args.root / "sitemap_links"
    pages_dir = args.root / "pages"
    link_files = sorted(path for path in links_dir.glob("*.csv") if not path.name.startswith("_"))
    link_rows = 0
    unique_links: set[str] = set()
    dates: list[str] = []
    for path in link_files:
        frame = pd.read_csv(path, encoding="utf-16", sep="\t", usecols=lambda name: name in {"link", "date"})
        link_rows += len(frame)
        if "link" in frame:
            unique_links.update(value for value in frame["link"].fillna("").astype(str) if value.strip())
        if "date" in frame:
            dates.extend(value for value in frame["date"].fillna("").astype(str) if value.strip())

    page_files = sorted(path for path in pages_dir.glob("*.parquet") if not path.name.startswith("."))
    page_rows = html_rows = text_rows = error_rows = unreadable = 0
    for path in page_files:
        try:
            parquet = pq.ParquetFile(path)
            page_rows += parquet.metadata.num_rows
            available = set(parquet.schema_arrow.names)
            columns = [name for name in ("page_source", "page_source_error", "article_text_extracted") if name in available]
            if columns:
                table = parquet.read(columns=columns)
                values = {name: table.column(name).to_pylist() for name in columns}
                html_rows += nonempty(values.get("page_source", []))
                text_rows += nonempty(values.get("article_text_extracted", []))
                error_rows += nonempty(values.get("page_source_error", []))
        except Exception:
            unreadable += 1

    pid = runtime_pid(args.root)
    task_state = scheduled_task_state(args.task_name)
    running = process_is_running(pid) or task_state == "Running"
    print(f"checked: {datetime.now().astimezone().isoformat(timespec='seconds')}")
    print(f"root: {args.root.resolve()}")
    print(f"process: {'RUNNING' if running else 'NOT RUNNING'}" + (f" (PID {pid})" if pid else ""))
    if task_state:
        print(f"scheduled task {args.task_name}: {task_state}")
    print(f"index files: {len(link_files):,}")
    print(f"index rows: {link_rows:,}")
    print(f"unique publication URLs: {len(unique_links):,}")
    print(f"index date range: {min(dates, default='none')} .. {max(dates, default='none')}")
    print(f"page parts: {len(page_files):,}")
    print(f"page rows: {page_rows:,}")
    print(f"downloaded HTML: {html_rows:,}")
    print(f"extracted text: {text_rows:,}")
    print(f"page errors recorded: {error_rows:,}")
    print(f"unreadable page parts: {unreadable:,}")


if __name__ == "__main__":
    main()
