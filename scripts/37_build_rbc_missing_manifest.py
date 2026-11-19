"""Build a stable manifest of RBC articles missing from an existing archive."""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

import _bootstrap  # noqa: F401

from dissertation_pipeline.news.interfax_archive import HttpClient, LocalRateLimiter, chunked
from dissertation_pipeline.news.rbc_archive import discover_group_candidates, discover_sitemap_urls


def iso_date(value: object) -> str:
    return str(value or "").strip()[:10]


def load_completed_ids(root: Path) -> set[str]:
    if not root.exists() or not any(root.rglob("*.parquet")):
        return set()
    table = ds.dataset(root, format="parquet", exclude_invalid_files=True).to_table(
        columns=["requested_material_id", "parse_status"]
    )
    result: set[str] = set()
    for material_id, status in zip(
        table.column("requested_material_id").to_pylist(),
        table.column("parse_status").to_pylist(),
    ):
        if material_id and str(status or "") != "request_error":
            result.add(str(material_id))
    return result


def in_research_window(candidate: dict[str, object], date_from: str, date_to_exclusive: str) -> bool:
    url_date = iso_date(candidate.get("url_date"))
    lastmod = iso_date(candidate.get("sitemap_lastmod"))
    if url_date:
        return date_from <= url_date < date_to_exclusive
    # For undated rbcfreenews URLs, lastmod is safe for the upper bound but can
    # be much newer than publication, so do not use it to exclude older pages.
    return not lastmod or lastmod < date_to_exclusive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--date-from", default="2008-09-30")
    parser.add_argument("--date-to-exclusive", default="2026-07-01")
    parser.add_argument("--sitemap-rps", type=float, default=8.0)
    args = parser.parse_args()
    date.fromisoformat(args.date_from)
    date.fromisoformat(args.date_to_exclusive)
    if args.date_from >= args.date_to_exclusive or args.sitemap_rps <= 0:
        raise SystemExit("invalid date window or sitemap rate")

    completed = load_completed_ids(args.archive_root)
    print(f"completed_material_ids={len(completed):,}", flush=True)
    client = HttpClient(
        limiter=LocalRateLimiter(args.sitemap_rps),
        retries=6,
        timeout=45,
        user_agent="Mozilla/5.0 (compatible; DissertationResearchInventory/1.0)",
    )
    groups = discover_sitemap_urls(client, min_map=0, max_map=None, newest_first=False)
    rows: list[dict[str, object]] = []
    candidates_total = eligible_total = 0
    for group_index, (group, sitemap_url) in enumerate(groups, start=1):
        candidates = discover_group_candidates(client, group, sitemap_url)
        candidates_total += len(candidates)
        for part_number, part in enumerate(chunked(candidates, 50)):
            for candidate in part:
                if not in_research_window(candidate, args.date_from, args.date_to_exclusive):
                    continue
                eligible_total += 1
                material_id = str(candidate["material_id"])
                if material_id in completed:
                    continue
                rows.append({**candidate, "part_number": part_number})
        if group_index % 50 == 0 or group_index == len(groups):
            print(
                f"sitemaps={group_index:,}/{len(groups):,} candidates={candidates_total:,} "
                f"eligible={eligible_total:,} missing={len(rows):,}",
                flush=True,
            )

    manifest = pd.DataFrame(rows).drop_duplicates("material_id", keep="first")
    if not manifest.empty:
        manifest = manifest.sort_values(["sitemap_group", "part_number", "material_id"], kind="stable")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(args.output, index=False, compression="zstd")
    print(f"sitemaps_total={len(groups):,}")
    print(f"candidates_total={candidates_total:,}")
    print(f"eligible_candidates={eligible_total:,}")
    print(f"missing_unique={len(manifest):,}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
