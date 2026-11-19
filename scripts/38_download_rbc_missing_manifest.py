"""Download a stable RBC missing-only manifest into an isolated delta archive."""
from __future__ import annotations

import argparse
import time
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401

from dissertation_pipeline.news.rbc_archive import create_browser_pool, download_content_part


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=6)
    parser.add_argument("--rps", type=float, default=6.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--part-retry-base-seconds", type=float, default=15.0)
    parser.add_argument("--part-retry-max-seconds", type=float, default=180.0)
    args = parser.parse_args()
    if args.processes < 1 or args.rps <= 0:
        raise SystemExit("processes and rps must be positive")

    manifest = pd.read_parquet(args.manifest)
    required = {"material_id", "url", "sitemap_group", "part_number"}
    missing_columns = required.difference(manifest.columns)
    if missing_columns:
        raise SystemExit(f"manifest missing columns: {sorted(missing_columns)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runtime = args.out_dir / "_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "download.pid").write_text(f"{__import__('os').getpid()}\n", encoding="ascii")

    groups = list(manifest.groupby(["sitemap_group", "part_number"], sort=True))
    totals = {"parts": 0, "rows": 0, "skipped": 0, "ok": 0, "metadata_only": 0, "not_found": 0, "errors": 0}
    executor = None

    def reset_pool() -> None:
        nonlocal executor
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
            executor = None

    try:
        for index, ((group, part_number), frame) in enumerate(groups, start=1):
            raw_candidates = frame.drop(columns=["part_number"], errors="ignore").to_dict("records")
            candidates = [
                {key: "" if pd.isna(value) else value for key, value in candidate.items()}
                for candidate in raw_candidates
            ]
            path = args.out_dir / f"sitemap={group}" / f"part-{int(part_number):05d}.parquet"
            attempt = 1
            while True:
                try:
                    if executor is None:
                        executor = create_browser_pool(
                            processes=args.processes,
                            requests_per_second=args.rps,
                            timeout=args.timeout,
                            retries=args.retries,
                        )
                    stats = download_content_part(
                        executor,
                        candidates,
                        path,
                        force=False,
                        output_format="parquet",
                    )
                    break
                except (BrokenProcessPool, RuntimeError) as exc:
                    reset_pool()
                    delay = min(
                        args.part_retry_max_seconds,
                        args.part_retry_base_seconds * (2 ** (attempt - 1)),
                    )
                    print(f"{path.relative_to(args.out_dir)} | {exc}; retrying in {delay:.0f}s", flush=True)
                    time.sleep(delay)
                    attempt += 1
            totals["parts"] += 1
            totals["rows"] += len(candidates)
            for key, value in stats.items():
                if key in totals:
                    totals[key] += value
            print(
                f"{index:,}/{len(groups):,} {path.relative_to(args.out_dir)} rows={len(candidates):,} "
                f"ok={stats['ok']:,} metadata_only={stats['metadata_only']:,} "
                f"404={stats['not_found']:,} errors={stats['errors']:,} skipped={stats['skipped']:,}",
                flush=True,
            )
    finally:
        reset_pool()
    print("Finished")
    print(" ".join(f"{key}={value:,}" for key, value in totals.items()))


if __name__ == "__main__":
    main()
