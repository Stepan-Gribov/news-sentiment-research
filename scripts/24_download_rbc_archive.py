from __future__ import annotations

import argparse
import importlib.util
import os
import time
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import _bootstrap  # noqa: F401

from dissertation_pipeline.news.interfax_archive import HttpClient, LocalRateLimiter, chunked
from dissertation_pipeline.news.rbc_archive import (
    create_browser_pool,
    discover_group_candidates,
    discover_sitemap_urls,
    download_content_part,
    write_inventory_part,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an RBC sitemap inventory or an authorized text archive.")
    parser.add_argument("--mode", choices=["inventory", "content"], default="inventory")
    parser.add_argument("--permission-confirmed", action="store_true")
    parser.add_argument(
        "--technical-smoke-test",
        action="store_true",
        help="Allow at most five article pages for parser verification without representing permission.",
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--processes", type=int, default=2, help="Headless Chrome worker processes in content mode.")
    parser.add_argument("--rps", type=float, default=1.0, help="Aggregate article navigation rate.")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--min-map", type=int, default=0)
    parser.add_argument("--max-map", type=int)
    parser.add_argument("--newest-first", action="store_true")
    parser.add_argument("--output-format", choices=["parquet", "jsonl.gz"], default="parquet")
    parser.add_argument("--max-articles", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--part-retry-base-seconds", type=float, default=60.0)
    parser.add_argument("--part-retry-max-seconds", type=float, default=900.0)
    parser.add_argument(
        "--user-agent",
        default="Mozilla/5.0 (compatible; DissertationResearchInventory/1.0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "content":
        if args.technical_smoke_test:
            if args.max_articles is None or args.max_articles > 5:
                raise SystemExit("--technical-smoke-test requires --max-articles between 1 and 5")
        elif not args.permission_confirmed:
            raise SystemExit(
                "RBC explicitly requires official permission for automated extraction. "
                "After permission is obtained, rerun with --permission-confirmed."
            )
    if args.output_format == "parquet" and not (
        importlib.util.find_spec("pyarrow") or importlib.util.find_spec("fastparquet")
    ):
        raise SystemExit("Parquet support is missing: python -m pip install pyarrow")
    if args.processes < 1 or args.rps <= 0 or args.chunk_size < 1:
        raise SystemExit("--processes, --rps and --chunk-size must be positive")
    if args.min_map < 0 or (args.max_map is not None and args.max_map < args.min_map):
        raise SystemExit("invalid sitemap range")
    if args.max_articles is not None and args.max_articles < 1:
        raise SystemExit("--max-articles must be positive")

    out_dir = args.out_dir or Path(f"data/rbc_archive_{args.mode}")
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = out_dir / "_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "download.pid").write_text(f"{os.getpid()}\n", encoding="ascii")

    sitemap_client = HttpClient(
        limiter=LocalRateLimiter(1.0),
        retries=6,
        timeout=args.timeout,
        user_agent=args.user_agent,
    )
    sitemap_urls = discover_sitemap_urls(
        sitemap_client,
        min_map=args.min_map,
        max_map=args.max_map,
        newest_first=args.newest_first,
    )
    totals = {"candidates": 0, "parts": 0, "skipped": 0, "inventory_only": 0, "ok": 0, "metadata_only": 0, "not_found": 0, "errors": 0}
    remaining = args.max_articles

    # A Chromium worker can occasionally exit unexpectedly (for example, when
    # ChromeDriver itself crashes).  A ProcessPoolExecutor becomes permanently
    # unusable after that.  Keeping the same broken pool caused an infinite
    # retry loop on one part.  Lazily recreate the pool and retry the part so a
    # full archive run can resume without losing its already written files.
    executor = None

    def reset_browser_pool() -> None:
        nonlocal executor
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
            executor = None

    try:
        for group, sitemap_url in sitemap_urls:
            candidates = discover_group_candidates(sitemap_client, group, sitemap_url)
            if remaining is not None:
                candidates = candidates[: max(0, remaining)]
            if not candidates:
                continue
            print(f"\n{group}: candidates={len(candidates):,} mode={args.mode}", flush=True)
            totals["candidates"] += len(candidates)
            group_dir = out_dir / f"sitemap={group}"
            for part_number, part in enumerate(chunked(candidates, args.chunk_size)):
                suffix = ".parquet" if args.output_format == "parquet" else ".jsonl.gz"
                path = group_dir / f"part-{part_number:05d}{suffix}"
                part_attempt = 1
                while True:
                    try:
                        if args.mode == "content":
                            if executor is None:
                                executor = create_browser_pool(
                                    processes=args.processes,
                                    requests_per_second=args.rps,
                                    timeout=args.timeout,
                                    retries=args.retries,
                                )
                            stats = download_content_part(
                                executor,
                                part,
                                path,
                                force=args.force,
                                output_format=args.output_format,
                            )
                        else:
                            stats = write_inventory_part(
                                part,
                                path,
                                force=args.force,
                                output_format=args.output_format,
                            )
                        break
                    except BrokenProcessPool as exc:
                        reset_browser_pool()
                        if args.fail_fast:
                            raise
                        delay = min(
                            args.part_retry_max_seconds,
                            args.part_retry_base_seconds * (2 ** (part_attempt - 1)),
                        )
                        print(
                            f"{path.relative_to(out_dir)} | browser worker exited ({exc}); "
                            f"recreated pool, retrying in {delay:.0f}s",
                            flush=True,
                        )
                        time.sleep(delay)
                        part_attempt += 1
                    except RuntimeError as exc:
                        if args.fail_fast:
                            raise
                        delay = min(
                            args.part_retry_max_seconds,
                            args.part_retry_base_seconds * (2 ** (part_attempt - 1)),
                        )
                        print(f"{path.relative_to(out_dir)} | {exc}; retrying in {delay:.0f}s", flush=True)
                        time.sleep(delay)
                        part_attempt += 1
                totals["parts"] += 1
                for key, value in stats.items():
                    totals[key] += value
                print(
                    f"{path.relative_to(out_dir)} | rows={len(part):,} "
                    f"inventory={stats['inventory_only']:,} ok={stats['ok']:,} "
                    f"metadata_only={stats['metadata_only']:,} 404={stats['not_found']:,} "
                    f"errors={stats['errors']:,} skipped={stats['skipped']:,}",
                    flush=True,
                )
            if remaining is not None:
                remaining -= len(candidates)
                if remaining <= 0:
                    break
    finally:
        reset_browser_pool()

    print("\nFinished", flush=True)
    print(" ".join(f"{key}={value:,}" for key, value in totals.items()), flush=True)
    print(f"output={out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
