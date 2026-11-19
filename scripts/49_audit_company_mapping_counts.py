#!/usr/bin/env python3
"""Write sorted company match counts and source date coverage without changing inputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import unicodedata
from pathlib import Path

import openpyxl
import pyarrow.parquet as pq


def normalize(value):
    return unicodedata.normalize("NFKC", str(value or "")).casefold().replace("ё", "е")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "company_match_counts_sorted.csv"
    json_path = args.output_dir / "mapping_and_date_audit.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError("Choose a new output directory")

    records = {}
    workbook = openpyxl.load_workbook(
        args.run_dir / "fund_asset_ru_isin_by_company_classification_template.xlsx",
        read_only=True, data_only=True,
    )
    for row in workbook["По компаниям"].iter_rows(min_row=2, max_col=5, values_only=True):
        name = str(row[0] or "").strip()
        if not name:
            continue
        identity = f"{normalize(name)}\t{normalize(row[2])}".encode("utf-8")
        cid = "c_" + hashlib.blake2b(identity, digest_size=12).hexdigest()
        records.setdefault(cid, {
            "company_id": cid, "company": name, "matches": 0,
            "with_name": 0, "industry_product_only": 0, "missing_date_matches": 0,
            "kommersant": 0, "prime": 0, "interfax": 0, "rbc": 0,
            "min_date": None, "max_date": None,
        })
    workbook.close()

    mapping = pq.ParquetFile(args.run_dir / "article_company_map.parquet")
    for batch in mapping.iter_batches(
        columns=["company_id", "company", "match_signal", "media", "date"],
        batch_size=250_000,
    ):
        import pyarrow as pa
        grouped = pa.Table.from_batches([batch]).group_by(
            ["company_id", "company", "match_signal", "media"]
        ).aggregate([
            ("company_id", "count"), ("date", "count"),
            ("date", "min"), ("date", "max"),
        ])
        for entry in grouped.to_pylist():
            record = records[entry["company_id"]]
            count = entry["company_id_count"]
            record["matches"] += count
            record["missing_date_matches"] += count - entry["date_count"]
            record["with_name" if "name" in entry["match_signal"].split(";") else "industry_product_only"] += count
            if entry["media"] in record:
                record[entry["media"]] += count
            for key, value, fn in (
                ("min_date", entry["date_min"], min),
                ("max_date", entry["date_max"], max),
            ):
                if value is not None:
                    record[key] = value if record[key] is None else fn(record[key], value)

    rows = sorted(records.values(), key=lambda r: (-r["matches"], r["company"], r["company_id"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
        for key in ("min_date", "max_date"):
            row[key] = row[key].isoformat() if row[key] is not None else ""
    assert sum(row["matches"] for row in rows) == mapping.metadata.num_rows

    articles = pq.read_table(
        args.run_dir / "articles_all_from_news_sqlite.parquet", columns=["media", "date"]
    ).group_by("media").aggregate([
        ("media", "count"), ("date", "count"), ("date", "min"), ("date", "max"),
    ])
    date_coverage = []
    for row in articles.to_pylist():
        date_coverage.append({
            "media": row["media"], "articles": row["media_count"],
            "dated_articles": row["date_count"],
            "missing_date_articles": row["media_count"] - row["date_count"],
            "min_date": row["date_min"].isoformat() if row["date_min"] else None,
            "max_date": row["date_max"].isoformat() if row["date_max"] else None,
        })
    fields = ["rank", "company", "company_id", "matches", "with_name", "industry_product_only",
              "kommersant", "prime", "interfax", "rbc", "missing_date_matches", "min_date", "max_date"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "mapping_path": str(args.run_dir / "article_company_map.parquet"),
        "company_counts_csv": str(csv_path), "blacklist_applied": False,
        "company_registry_ids": len(rows), "companies_with_matches": sum(r["matches"] > 0 for r in rows),
        "zero_match_companies": sum(r["matches"] == 0 for r in rows),
        "total_matches": sum(r["matches"] for r in rows),
        "matches_with_name": sum(r["with_name"] for r in rows),
        "industry_product_only": sum(r["industry_product_only"] for r in rows),
        "source_date_coverage": sorted(date_coverage, key=lambda r: r["media"]),
        "top_20": rows[:20],
    }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
