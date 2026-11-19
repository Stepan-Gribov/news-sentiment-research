from __future__ import annotations

import argparse
import os
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

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


def collect_snapshot(root: Path, fast: bool) -> dict[str, object]:
    files = sorted(root.rglob("*.parquet")) if root.exists() else []
    rows = 0
    bytes_total = 0
    statuses: Counter[str] = Counter()
    text_rows = 0
    unreadable: list[str] = []

    for path in files:
        bytes_total += path.stat().st_size
        try:
            parquet = pq.ParquetFile(path)
            rows += parquet.metadata.num_rows
            if not fast:
                table = parquet.read(columns=["parse_status", "text_length"])
                statuses.update(str(value or "missing") for value in table.column("parse_status").to_pylist())
                text_rows += sum(int(value or 0) > 0 for value in table.column("text_length").to_pylist())
        except Exception as exc:
            unreadable.append(f"{path}: {exc}")

    pid_path = root / "_runtime" / "download.pid"
    pid = 0
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError:
            pass

    partition_months: list[str] = []
    for path in files:
        year = next((part[5:] for part in path.parts if part.startswith("year=")), "")
        month = next((part[6:] for part in path.parts if part.startswith("month=")), "")
        if year and month:
            partition_months.append(f"{year}-{month}")

    newest_mtime = max((path.stat().st_mtime for path in files), default=0.0)
    return {
        "pid": pid,
        "running": process_is_running(pid),
        "files": len(files),
        "rows": rows,
        "bytes": bytes_total,
        "text_rows": text_rows,
        "statuses": statuses,
        "unreadable": unreadable,
        "first_month": min(partition_months, default=""),
        "last_month": max(partition_months, default=""),
        "newest_mtime": newest_mtime,
    }


def print_snapshot(root: Path, snapshot: dict[str, object], *, fast: bool, rate: float | None) -> None:
    pid = int(snapshot["pid"])
    print(f"checked: {datetime.now().astimezone().isoformat(timespec='seconds')}")
    print(f"root: {root.resolve()}")
    print(f"process: {'RUNNING' if snapshot['running'] else 'NOT RUNNING'}" + (f" (PID {pid})" if pid else ""))
    print(f"parquet parts: {int(snapshot['files']):,}")
    print(f"saved rows: {int(snapshot['rows']):,}")
    print(f"disk size: {int(snapshot['bytes']) / (1024 ** 3):.3f} GiB")
    if snapshot["first_month"]:
        print(f"saved month range: {snapshot['first_month']} .. {snapshot['last_month']}")
    if snapshot["newest_mtime"]:
        latest = datetime.fromtimestamp(float(snapshot["newest_mtime"])).astimezone()
        age = max(0, int(time.time() - float(snapshot["newest_mtime"])))
        print(f"latest part: {latest.isoformat(timespec='seconds')} ({age:,} sec ago)")
    if rate is not None:
        print(f"recent save rate: {rate:.2f} rows/sec ({rate * 3600:,.0f} rows/hour)")
    if not fast:
        print(f"rows with text: {int(snapshot['text_rows']):,}")
        statuses = snapshot["statuses"]
        print("statuses: " + (", ".join(f"{key}={value:,}" for key, value in sorted(statuses.items())) or "none"))
    unreadable = snapshot["unreadable"]
    print(f"unreadable parts: {len(unreadable):,}")
    for error in unreadable[:5]:
        print(f"  {error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Show progress of the Prime archive download.")
    parser.add_argument("--root", type=Path, default=Path("data/prime_archive"))
    parser.add_argument("--fast", action="store_true", help="Count rows without reading status columns.")
    parser.add_argument("--watch", action="store_true", help="Keep refreshing until interrupted with Ctrl+C.")
    parser.add_argument("--interval", type=float, default=60.0, help="Watch refresh interval in seconds.")
    args = parser.parse_args()
    if args.interval < 5:
        raise SystemExit("--interval must be at least 5 seconds")

    previous_rows: int | None = None
    previous_time: float | None = None
    while True:
        started = time.time()
        snapshot = collect_snapshot(args.root, args.fast)
        now = time.time()
        rate = None
        if previous_rows is not None and previous_time is not None and now > previous_time:
            rate = max(0, int(snapshot["rows"]) - previous_rows) / (now - previous_time)
        print_snapshot(args.root, snapshot, fast=args.fast, rate=rate)
        if not args.watch:
            break
        print(f"next refresh in {args.interval:.0f} sec; Ctrl+C to stop monitoring\n", flush=True)
        previous_rows = int(snapshot["rows"])
        previous_time = now
        time.sleep(max(0.0, args.interval - (time.time() - started)))


if __name__ == "__main__":
    main()
