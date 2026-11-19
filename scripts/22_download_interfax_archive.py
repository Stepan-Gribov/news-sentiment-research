from __future__ import annotations

import argparse
import importlib.util
import os
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from dissertation_pipeline.news.interfax_archive import (
    HttpClient,
    LocalRateLimiter,
    chunked,
    create_process_pool,
    discover_group_candidates,
    discover_sitemap_urls,
    download_part,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download public Interfax.ru editorial articles from the official sitemaps."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/interfax_archive"))
    parser.add_argument("--processes", type=int, default=6)
    parser.add_argument("--rps", type=float, default=4.0, help="Aggregate GET rate across all processes.")
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--min-map", type=int, default=1)
    parser.add_argument("--max-map", type=int)
    parser.add_argument("--exclude-latest", action="store_true")
    parser.add_argument("--newest-first", action="store_true")
    parser.add_argument("--output-format", choices=["parquet", "jsonl.gz"], default="parquet")
    parser.add_argument("--max-articles", type=int, help="Cap total candidates for a smoke test.")
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
    if args.processes < 1 or args.chunk_size < 1 or args.rps <= 0:
        raise SystemExit("--processes, --chunk-size and --rps must be positive")
    if args.min_map < 1 or (args.max_map is not None and args.max_map < args.min_map):
        raise SystemExit("invalid sitemap number range")
    if args.max_articles is not None and args.max_articles < 1:
        raise SystemExit("--max-articles must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pid_path = args.pid_file or args.out_dir / "_runtime" / "download.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(f"{os.getpid()}\n", encoding="ascii")

    discovery_client = HttpClient(
        limiter=LocalRateLimiter(min(args.rps, 1.0)),
        retries=args.retries,
        timeout=args.timeout,
        user_agent=args.user_agent,
    )
    sitemap_urls = discover_sitemap_urls(
        discovery_client,
        min_map=args.min_map,
        max_map=args.max_map,
        include_latest=not args.exclude_latest,
        newest_first=args.newest_first,
    )
    if not sitemap_urls:
        raise SystemExit("No Interfax sitemaps matched the requested options")

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

    with create_process_pool(
        processes=args.processes,
        requests_per_second=args.rps,
        retries=args.retries,
        timeout=args.timeout,
        user_agent=args.user_agent,
    ) as executor:
        for group, sitemap_url in sitemap_urls:
            candidates = discover_group_candidates(discovery_client, group, sitemap_url)
            if remaining is not None:
                candidates = candidates[: max(0, remaining)]
            if not candidates:
                continue

            group_dir = args.out_dir / f"sitemap={group}"
            print(f"\n{group}: candidates={len(candidates):,}", flush=True)
            totals["candidates"] += len(candidates)

            for part_number, part in enumerate(chunked(candidates, args.chunk_size)):
                suffix = ".parquet" if args.output_format == "parquet" else ".jsonl.gz"
                path = group_dir / f"part-{part_number:05d}{suffix}"
                part_attempt = 1
                while True:
                    try:
                        stats = download_part(
                            executor,
                            part,
                            path,
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
