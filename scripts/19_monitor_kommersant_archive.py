from __future__ import annotations

import argparse
import os
import subprocess
from collections import Counter
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Show progress of the Kommersant archive download.")
    parser.add_argument("--root", type=Path, default=Path("data/kommersant_archive"))
    parser.add_argument("--fast", action="store_true", help="Count rows only; do not read status columns.")
    args = parser.parse_args()

    files = sorted(args.root.rglob("*.parquet")) if args.root.exists() else []
    rows = 0
    bytes_total = 0
    statuses: Counter[str] = Counter()
    text_rows = 0
    unreadable = []

    for path in files:
        bytes_total += path.stat().st_size
        try:
            parquet = pq.ParquetFile(path)
            rows += parquet.metadata.num_rows
            if not args.fast:
                table = parquet.read(columns=["parse_status", "text_length"])
                status_values = table.column("parse_status").to_pylist()
                lengths = table.column("text_length").to_pylist()
                statuses.update(str(value or "missing") for value in status_values)
                text_rows += sum(int(value or 0) > 0 for value in lengths)
        except Exception as exc:
            unreadable.append(f"{path}: {exc}")

    pid_path = args.root / "_runtime" / "download.pid"
    pid = 0
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError:
            pass

    print(f"root: {args.root.resolve()}")
    print(f"process: {'RUNNING' if process_is_running(pid) else 'NOT RUNNING'}" + (f" (PID {pid})" if pid else ""))
    print(f"parquet parts: {len(files):,}")
    print(f"saved rows: {rows:,}")
    print(f"disk size: {bytes_total / (1024 ** 3):.3f} GiB")
    if not args.fast:
        print(f"rows with text: {text_rows:,}")
        print("statuses: " + (", ".join(f"{key}={value:,}" for key, value in sorted(statuses.items())) or "none"))
    print(f"unreadable parts: {len(unreadable):,}")
    for error in unreadable[:5]:
        print(f"  {error}")


if __name__ == "__main__":
    main()

