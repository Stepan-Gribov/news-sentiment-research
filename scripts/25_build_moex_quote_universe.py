"""Build the MOEX quote universe from legacy Y and exact InvestFunds holdings.

The output contains the union of syntactically valid ISINs from both sources,
with source-specific and combined date coverage. Both large inputs are read in
chunks so the script can run on an ordinary workstation.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


ISIN_RE = re.compile(r"\b([A-Z]{2}[A-Z0-9]{9}\d)\b", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the union of ISINs in Y and exact fund holdings."
    )
    parser.add_argument("--y-file", type=Path, required=True)
    parser.add_argument("--holdings-long", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunksize", type=int, default=200_000)
    return parser.parse_args()


def empty_source_stats() -> dict[str, Any]:
    return {"first_date": None, "last_date": None, "rows": 0}


def merge_grouped(
    target: dict[str, dict[str, Any]], grouped: pd.DataFrame
) -> None:
    for isin, row in grouped.iterrows():
        stats = target[isin]
        first_date = row["first_date"]
        last_date = row["last_date"]
        if stats["first_date"] is None or first_date < stats["first_date"]:
            stats["first_date"] = first_date
        if stats["last_date"] is None or last_date > stats["last_date"]:
            stats["last_date"] = last_date
        stats["rows"] += int(row["rows"])


def scan_y(
    path: Path, chunksize: int
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = defaultdict(empty_source_stats)
    source_min = None
    source_max = None
    source_rows = 0
    valid_rows = 0
    reader = pd.read_csv(
        path,
        sep="\t",
        encoding="utf-16",
        usecols=["date", "ISIN конечного актива"],
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in reader:
        dates = pd.to_datetime(chunk["date"], errors="coerce")
        valid_dates = dates.dropna()
        if not valid_dates.empty:
            current_min = valid_dates.min()
            current_max = valid_dates.max()
            source_min = current_min if source_min is None else min(source_min, current_min)
            source_max = current_max if source_max is None else max(source_max, current_max)
        source_rows += len(chunk)
        isins = (
            chunk["ISIN конечного актива"]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.extract(ISIN_RE, expand=False)
        )
        valid = pd.DataFrame({"isin": isins, "date": dates}).dropna()
        valid_rows += len(valid)
        if not valid.empty:
            grouped = valid.groupby("isin")["date"].agg(
                first_date="min", last_date="max", rows="size"
            )
            merge_grouped(stats, grouped)
    return dict(stats), {
        "rows": source_rows,
        "valid_isin_rows": valid_rows,
        "first_date": source_min,
        "last_date": source_max,
    }


def scan_exact_holdings(
    path: Path, chunksize: int
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = defaultdict(empty_source_stats)
    source_min = None
    source_max = None
    source_rows = 0
    asset_item_rows = 0
    valid_rows = 0
    reader = pd.read_csv(
        path,
        compression="infer",
        encoding="utf-8-sig",
        usecols=[
            "report_date",
            "table_type",
            "row_type",
            "item_name",
            "registration_number_security",
        ],
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in reader:
        source_rows += len(chunk)
        chunk = chunk[
            chunk["table_type"].eq("asset") & chunk["row_type"].eq("asset_item")
        ].copy()
        asset_item_rows += len(chunk)
        if chunk.empty:
            continue
        dates = pd.to_datetime(chunk["report_date"], errors="coerce")
        valid_dates = dates.dropna()
        if not valid_dates.empty:
            current_min = valid_dates.min()
            current_max = valid_dates.max()
            source_min = current_min if source_min is None else min(source_min, current_min)
            source_max = current_max if source_max is None else max(source_max, current_max)
        identifier_text = (
            chunk["registration_number_security"].fillna("").astype(str)
            + " "
            + chunk["item_name"].fillna("").astype(str)
        )
        isins = identifier_text.str.upper().str.extract(ISIN_RE, expand=False)
        valid = pd.DataFrame({"isin": isins, "date": dates}).dropna()
        valid_rows += len(valid)
        if not valid.empty:
            grouped = valid.groupby("isin")["date"].agg(
                first_date="min", last_date="max", rows="size"
            )
            merge_grouped(stats, grouped)
    return dict(stats), {
        "rows": source_rows,
        "asset_item_rows": asset_item_rows,
        "valid_isin_rows": valid_rows,
        "first_date": source_min,
        "last_date": source_max,
    }


def iso_date(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).date().isoformat()


def main() -> None:
    args = parse_args()
    y_stats, y_summary = scan_y(args.y_file, args.chunksize)
    exact_stats, exact_summary = scan_exact_holdings(args.holdings_long, args.chunksize)

    rows = []
    all_isins = sorted(set(y_stats) | set(exact_stats))
    for isin in all_isins:
        y = y_stats.get(isin, empty_source_stats())
        exact = exact_stats.get(isin, empty_source_stats())
        first_values = [value for value in (y["first_date"], exact["first_date"]) if value is not None]
        last_values = [value for value in (y["last_date"], exact["last_date"]) if value is not None]
        rows.append(
            {
                "requested_isin": isin,
                "first_date": iso_date(min(first_values)),
                "last_date": iso_date(max(last_values)),
                "source_y": int(isin in y_stats),
                "source_exact_holdings": int(isin in exact_stats),
                "y_first_date": iso_date(y["first_date"]),
                "y_last_date": iso_date(y["last_date"]),
                "exact_first_date": iso_date(exact["first_date"]),
                "exact_last_date": iso_date(exact["last_date"]),
                "y_rows": y["rows"],
                "exact_rows": exact["rows"],
            }
        )
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(args.output)

    summary = {
        "y": {key: iso_date(value) if "date" in key else value for key, value in y_summary.items()},
        "exact_holdings": {
            key: iso_date(value) if "date" in key else value
            for key, value in exact_summary.items()
        },
        "y_unique_isins": len(y_stats),
        "exact_holdings_unique_isins": len(exact_stats),
        "union_unique_isins": len(all_isins),
        "only_in_y": len(set(y_stats) - set(exact_stats)),
        "only_in_exact_holdings": len(set(exact_stats) - set(y_stats)),
        "earliest_quote_request_date": min(
            value
            for value in (y_summary["first_date"], exact_summary["first_date"])
            if value is not None
        ).date().isoformat(),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
