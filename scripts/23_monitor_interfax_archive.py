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
    publication_dates: list[str] = []
    article_ids: set[int] = set()

    for path in files:
        bytes_total += path.stat().st_size
        try:
            parquet = pq.ParquetFile(path)
            rows += parquet.metadata.num_rows
            if not fast:
                table = parquet.read(
                    columns=["article_id", "parse_status", "text_length", "publication_date"]
                )
                statuses.update(str(value or "missing") for value in table.column("parse_status").to_pylist())
                text_rows += sum(int(value or 0) > 0 for value in table.column("text_length").to_pylist())
                article_ids.update(int(value) for value in table.column("article_id").to_pylist() if value)
                publication_dates.extend(
                    str(value) for value in table.column("publication_date").to_pylist() if value
                )
        except Exception as exc:
            unreadable.append(f"{path}: {exc}")

    pid_path = root / "_runtime" / "download.pid"
    pid = 0
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError:
            pass

    groups = [
        part[8:]
        for path in files
        for part in path.parts
        if part.startswith("sitemap=")
    ]
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
        "first_group": min(groups, default=""),
        "last_group": max(groups, default=""),
        "newest_mtime": newest_mtime,
        "unique_articles": len(article_ids),
        "min_date": min(publication_dates, default=""),
        "max_date": max(publication_dates, default=""),
    }


def print_snapshot(
    root: Path,
    snapshot: dict[str, object],
    *,
    fast: bool,
    rate: float | None,
    expected_rows: int | None,
) -> None:
    pid = int(snapshot["pid"])
    print(f"checked: {datetime.now().astimezone().isoformat(timespec='seconds')}")
    print(f"root: {root.resolve()}")
    print(f"process: {'RUNNING' if snapshot['running'] else 'NOT RUNNING'}" + (f" (PID {pid})" if pid else ""))
    print(f"parquet parts: {int(snapshot['files']):,}")
    print(f"saved rows: {int(snapshot['rows']):,}")
    if expected_rows:
        remaining = max(0, expected_rows - int(snapshot["rows"]))
        completion = min(100.0, 100.0 * int(snapshot["rows"]) / expected_rows)
        print(f"expected candidates: {expected_rows:,}")
        print(f"completion: {completion:.2f}% (remaining {remaining:,})")
    print(f"disk size: {int(snapshot['bytes']) / (1024 ** 3):.3f} GiB")
    if snapshot["first_group"]:
        print(f"saved sitemap range: {snapshot['first_group']} .. {snapshot['last_group']}")
    if snapshot["newest_mtime"]:
        latest = datetime.fromtimestamp(float(snapshot["newest_mtime"])).astimezone()
        age = max(0, int(time.time() - float(snapshot["newest_mtime"])))
        print(f"latest part: {latest.isoformat(timespec='seconds')} ({age:,} sec ago)")
    if rate is not None:
        print(f"recent save rate: {rate:.2f} rows/sec ({rate * 3600:,.0f} rows/hour)")
        if expected_rows and rate > 0:
            remaining = max(0, expected_rows - int(snapshot["rows"]))
            print(f"ETA at recent rate: {remaining / rate / 3600:.1f} hours")
    if not fast:
        print(f"unique article IDs: {int(snapshot['unique_articles']):,}")
        print(f"duplicate rows: {int(snapshot['rows']) - int(snapshot['unique_articles']):,}")
        print(f"publication date range: {snapshot['min_date'] or 'none'} .. {snapshot['max_date'] or 'none'}")
        print(f"rows with text: {int(snapshot['text_rows']):,}")
        statuses = snapshot["statuses"]
        print("statuses: " + (", ".join(f"{key}={value:,}" for key, value in sorted(statuses.items())) or "none"))
    unreadable = snapshot["unreadable"]
    print(f"unreadable parts: {len(unreadable):,}")
    for error in unreadable[:5]:
        print(f"  {error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Show progress of the Interfax archive download.")
    parser.add_argument("--root", type=Path, default=Path("data/interfax_archive"))
    parser.add_argument("--fast", action="store_true", help="Read Parquet metadata only.")
    parser.add_argument("--watch", action="store_true", help="Refresh until interrupted with Ctrl+C.")
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--expected-rows", type=int, help="Optional candidate total for percent and ETA.")
    args = parser.parse_args()
    if args.interval < 5:
        raise SystemExit("--interval must be at least 5 seconds")
    if args.expected_rows is not None and args.expected_rows < 1:
        raise SystemExit("--expected-rows must be positive")

    previous_rows: int | None = None
    previous_time: float | None = None
    while True:
        started = time.time()
        snapshot = collect_snapshot(args.root, args.fast)
        now = time.time()
        rate = None
        if previous_rows is not None and previous_time is not None and now > previous_time:
            rate = max(0, int(snapshot["rows"]) - previous_rows) / (now - previous_time)
        print_snapshot(
            args.root,
            snapshot,
            fast=args.fast,
            rate=rate,
            expected_rows=args.expected_rows,
        )
        if not args.watch:
            break
        print(f"next refresh in {args.interval:.0f} sec; Ctrl+C to stop monitoring\n", flush=True)
        previous_rows = int(snapshot["rows"])
        previous_time = now
        time.sleep(max(0.0, args.interval - (time.time() - started)))


if __name__ == "__main__":
    main()
