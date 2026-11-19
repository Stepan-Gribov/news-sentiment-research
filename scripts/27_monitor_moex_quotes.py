"""Show progress for 26_download_moex_quotes_by_isin.py."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path


def read_statuses(path: Path) -> list[dict]:
    rows = []
    for item in sorted(path.glob("*.json")):
        try:
            rows.append(json.loads(item.read_text(encoding="utf-8")))
        except Exception:
            continue
    return rows


def snapshot(root: Path) -> None:
    mapping = read_statuses(root / "mapping_cache")
    history = read_statuses(root / "history_status")
    mapping_counts = Counter(row.get("status", "unknown") for row in mapping)
    history_counts = Counter(row.get("status", "unknown") for row in history)
    history_rows = sum(int(row.get("rows", 0) or 0) for row in history)
    latest = max(
        [path.stat().st_mtime for path in (root / "history_status").glob("*.json")]
        if (root / "history_status").exists()
        else [0]
    )
    print(f"checked: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    print(f"root: {root}")
    print(f"mapping: {len(mapping)} {dict(mapping_counts)}")
    print(f"history: {len(history)} {dict(history_counts)} rows={history_rows:,}")
    if latest:
        print(f"latest history status: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(latest))}")
    summary = root / "summary.json"
    if summary.exists():
        try:
            print(f"summary: {json.loads(summary.read_text(encoding='utf-8'))}")
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    while True:
        snapshot(args.root)
        if not args.watch:
            return
        time.sleep(max(args.interval, 1))


if __name__ == "__main__":
    main()
