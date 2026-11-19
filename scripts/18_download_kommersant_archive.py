from __future__ import annotations

import argparse
import importlib.util
import os
import time
from datetime import date
from pathlib import Path

import _bootstrap  # noqa: F401

from dissertation_pipeline.news.kommersant_archive import (
    DOC_URL_TEMPLATE,
    HttpClient,
    chunked,
    discover_month_candidates,
    discover_sitemap_urls,
    download_part,
)
from dissertation_pipeline.paths import PATHS


def valid_month(value: str) -> str:
    try:
        year, month = value.split("-")
        if len(year) != 4 or len(month) != 2 or int(year) < 1990 or not 1 <= int(month) <= 12:
            raise ValueError
        return value
    except ValueError as exc:
        raise argparse.ArgumentTypeError("month must have YYYY-MM format") from exc


def parse_args() -> argparse.Namespace:
    current_month = date.today().strftime("%Y-%m")
    parser = argparse.ArgumentParser(
        description=(
            "Download the Kommersant archive as extracted, resumable Parquet parts. "
            "The default sitemap mode avoids requests for millions of unused numeric IDs."
        )
    )
    parser.add_argument("--mode", choices=["sitemap", "id-range"], default="sitemap")
    parser.add_argument("--start-month", type=valid_month, default="1990-01")
    parser.add_argument("--end-month", type=valid_month, default=current_month)
    parser.add_argument("--id-from", type=int, default=1)
    parser.add_argument("--id-to", type=int, help="Required for --mode id-range.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PATHS.news_data_dir / "kommersant_archive",
        help="Output dataset directory (default: $DISS_SUPP_ROOT/news_data/kommersant_archive).",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--rps", type=float, default=1.0, help="Global GET requests per second across all workers.")
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument(
        "--output-format",
        choices=["parquet", "jsonl.gz"],
        default="parquet",
        help="Parquet is recommended; jsonl.gz works without pyarrow.",
    )
    parser.add_argument("--max-articles", type=int, help="Smoke-test cap across the whole run.")
    parser.add_argument("--force", action="store_true", help="Overwrite already complete parts.")
    parser.add_argument("--pid-file", type=Path, help="Optional process-ID file for external monitoring.")
    parser.add_argument(
        "--part-retry-base-seconds",
        type=float,
        default=60.0,
        help="Initial pause before retrying a part with transient request failures.",
    )
    parser.add_argument(
        "--part-retry-max-seconds",
        type=float,
        default=900.0,
        help="Maximum pause between retries of a transiently failed part.",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Exit instead of retrying a transiently failed part.")
    parser.add_argument(
        "--user-agent",
        default="Mozilla/5.0 (compatible; DissertationResearchArchive/1.0)",
    )
    return parser.parse_args()


def ensure_parquet_engine() -> None:
    if importlib.util.find_spec("pyarrow") or importlib.util.find_spec("fastparquet"):
        return
    raise SystemExit(
        "Parquet support is missing. Install the project dependencies first: "
        "python3 -m pip install -r requirements.txt"
    )


def id_range_groups(args: argparse.Namespace):
    if args.id_to is None:
        raise SystemExit("--id-to is required with --mode id-range")
    if args.id_from < 1 or args.id_to < args.id_from:
        raise SystemExit("invalid ID range")
    step = max(args.chunk_size, 1) * 40
    for first in range(args.id_from, args.id_to + 1, step):
        last = min(args.id_to, first + step - 1)
        candidates = [
            {
                "doc_id": doc_id,
                "url": DOC_URL_TEMPLATE.format(doc_id=doc_id),
                "sitemap_month": "",
                "sitemap_lastmod": "",
            }
            for doc_id in range(first, last + 1)
        ]
        yield f"ids_{first:09d}_{last:09d}", candidates


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

    if args.mode == "sitemap":
        sitemap_urls = discover_sitemap_urls(
            client,
            start_month=args.start_month,
            end_month=args.end_month,
        )

        def groups():
            for month, url in sitemap_urls:
                yield month, discover_month_candidates(client, month, url)

        group_iter = groups()
    else:
        group_iter = id_range_groups(args)

    totals = {"candidates": 0, "parts": 0, "skipped": 0, "ok": 0, "metadata_only": 0, "not_found": 0, "errors": 0}
    remaining = args.max_articles

    for group_name, candidates in group_iter:
        if remaining is not None:
            candidates = candidates[: max(0, remaining)]
        if not candidates:
            continue

        if args.mode == "sitemap":
            year, month = group_name.split("-")
            group_dir = args.out_dir / f"year={year}" / f"month={month}"
        else:
            group_dir = args.out_dir / "id_ranges" / group_name

        print(f"\n{group_name}: candidates={len(candidates):,}", flush=True)
        totals["candidates"] += len(candidates)
        for part_number, part in enumerate(chunked(candidates, args.chunk_size)):
            suffix = ".parquet" if args.output_format == "parquet" else ".jsonl.gz"
            path = group_dir / f"part-{part_number:05d}{suffix}"
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
                        f"retrying part in {retry_delay:.0f}s (attempt {part_attempt + 1})",
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
                f"404={stats['not_found']:,} skipped={stats['skipped']:,}",
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
