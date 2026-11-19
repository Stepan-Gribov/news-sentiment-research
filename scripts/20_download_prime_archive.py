from __future__ import annotations

import argparse
import importlib.util
import os
import time
from datetime import date
from pathlib import Path

import _bootstrap  # noqa: F401

from dissertation_pipeline.news.prime_archive import (
    HttpClient,
    chunked,
    discover_month_candidates,
    discover_sitemap_urls,
    download_part,
)


def valid_month(value: str) -> str:
    try:
        year, month = value.split("-")
        if len(year) != 4 or len(month) != 2 or int(year) < 2010 or not 1 <= int(month) <= 12:
            raise ValueError
        return value
    except ValueError as exc:
        raise argparse.ArgumentTypeError("month must have YYYY-MM format and year >= 2010") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the public Prime archive from its official monthly article sitemaps."
    )
    parser.add_argument("--start-month", type=valid_month, default="2010-01")
    parser.add_argument("--end-month", type=valid_month, default=date.today().strftime("%Y-%m"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/prime_archive"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--rps", type=float, default=3.0, help="Global GET rate across all workers.")
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--output-format", choices=["parquet", "jsonl.gz"], default="parquet")
    parser.add_argument("--max-articles", type=int, help="Cap the total rows for a smoke test.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--part-retry-base-seconds", type=float, default=60.0)
    parser.add_argument("--part-retry-max-seconds", type=float, default=900.0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--user-agent",
        default="Mozilla/5.0 (compatible; DissertationResearchArchive/1.0)",
    )
    return parser.parse_args()


def ensure_parquet_engine() -> None:
    if importlib.util.find_spec("pyarrow") or importlib.util.find_spec("fastparquet"):
        return
    raise SystemExit("Parquet support is missing: python -m pip install pyarrow")


def main() -> None:
    args = parse_args()
    if args.output_format == "parquet":
        ensure_parquet_engine()
    if args.workers < 1 or args.chunk_size < 1 or args.rps <= 0:
        raise SystemExit("--workers, --chunk-size and --rps must be positive")
    if args.max_articles is not None and args.max_articles < 1:
        raise SystemExit("--max-articles must be positive")
    if args.start_month > args.end_month:
        raise SystemExit("--start-month must not be after --end-month")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pid_path = args.pid_file or args.out_dir / "_runtime" / "download.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(f"{os.getpid()}\n", encoding="ascii")
    client = HttpClient(
        requests_per_second=args.rps,
        retries=args.retries,
        timeout=args.timeout,
        user_agent=args.user_agent,
    )
    sitemap_urls = discover_sitemap_urls(
        client,
        start_month=args.start_month,
        end_month=args.end_month,
    )
    if not sitemap_urls:
        raise SystemExit("No Prime monthly sitemaps matched the requested interval")

    totals = {
        "candidates": 0,
        "parts": 0,
        "skipped": 0,
        "ok": 0,
        "metadata_only": 0,
        "not_found": 0,
        "errors": 0,
    }
    remaining = args.max_articles

    for month, sitemap_url in sitemap_urls:
        candidates = discover_month_candidates(client, month, sitemap_url)
        if remaining is not None:
            candidates = candidates[: max(0, remaining)]
        if not candidates:
            continue

        year, month_number = month.split("-")
        month_dir = args.out_dir / f"year={year}" / f"month={month_number}"
        print(f"\n{month}: candidates={len(candidates):,}", flush=True)
        totals["candidates"] += len(candidates)

        for part_number, part in enumerate(chunked(candidates, args.chunk_size)):
            suffix = ".parquet" if args.output_format == "parquet" else ".jsonl.gz"
            path = month_dir / f"part-{part_number:05d}{suffix}"
            part_attempt = 1
            while True:
                try:
                    stats = download_part(
                        client,
                        part,
                        path,
                        workers=args.workers,
                        force=args.force,
                        output_format=args.output_format,
                    )
                    break
                except RuntimeError as exc:
                    if args.fail_fast:
                        raise
                    retry_delay = min(
                        args.part_retry_max_seconds,
                        args.part_retry_base_seconds * (2 ** (part_attempt - 1)),
                    )
                    print(
                        f"{path.relative_to(args.out_dir)} | transient failure: {exc}; "
                        f"retrying in {retry_delay:.0f}s (attempt {part_attempt + 1})",
                        flush=True,
                    )
                    time.sleep(retry_delay)
                    part_attempt += 1

            totals["parts"] += 1
            for key, value in stats.items():
                totals[key] += value
            print(
                f"{path.relative_to(args.out_dir)} | rows={len(part):,} "
                f"ok={stats['ok']:,} metadata_only={stats['metadata_only']:,} "
                f"404={stats['not_found']:,} errors={stats['errors']:,} skipped={stats['skipped']:,}",
                flush=True,
            )

        if remaining is not None:
            remaining -= len(candidates)
            if remaining <= 0:
                break

    print("\nFinished", flush=True)
    print(" ".join(f"{key}={value:,}" for key, value in totals.items()), flush=True)
    print(f"output={args.out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
