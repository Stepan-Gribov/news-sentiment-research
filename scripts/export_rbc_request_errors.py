"""Export persisted RBC request failures from an archive without changing it."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="RBC content archive directory")
    parser.add_argument("--output", type=Path, required=True, help="CSV report to create")
    args = parser.parse_args()

    parts = sorted(args.root.rglob("*.parquet"))
    columns = [
        "requested_material_id",
        "url",
        "final_url",
        "parse_status",
        "http_status",
        "error",
        "sitemap_group",
        "sitemap_lastmod",
        "fetched_at_utc",
        "browser_attempts",
    ]
    failed: list[pd.DataFrame] = []
    for index, part in enumerate(parts, start=1):
        frame = pd.read_parquet(part, columns=columns)
        bad = frame.loc[frame["parse_status"].fillna("").eq("request_error")].copy()
        if not bad.empty:
            bad.insert(0, "archive_part", str(part.relative_to(args.root)))
            failed.append(bad)
        if index % 1000 == 0:
            print(f"scanned {index:,}/{len(parts):,} parts", flush=True)

    result = pd.concat(failed, ignore_index=True) if failed else pd.DataFrame(columns=["archive_part", *columns])
    result = result.sort_values(["sitemap_group", "requested_material_id"], kind="stable")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"parts_scanned={len(parts):,}")
    print(f"request_error_rows={len(result):,}")
    print(f"unique_urls={result['url'].nunique():,}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
