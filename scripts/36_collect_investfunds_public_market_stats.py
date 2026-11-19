"""Collect Investfunds' public market-level PIF count and net-flow time series.

The JSON responses are preserved verbatim. CSV/Parquet files are a long-form
representation of the source points only; this script does not calculate
returns, growth rates, or other analytical variables.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests


DEFAULT_ROOT = Path("./market_data/raw_external_2026-09-30")
PAGE_URL = "https://investfunds.ru/funds-statistics/"
COUNT_SERIES = ["0", "1", "2", "3", "5"]
FLOW_SERIES = ["4", "4001", "4003", "4004", "4006", "4023"]
START_DATE = date(2012, 1, 1)
END_DATE = date(2026, 8, 31)


def request_payload(start: date, end: date, key: str, view: str, item_ids: list[str]) -> tuple[bytes, Any]:
    params = {
        "action": "chartData",
        "date_start": start.strftime("%d.%m.%Y"),
        "date_end": end.strftime("%d.%m.%Y"),
        "data_key": key,
        "data_view": view,
        "data_items": "_".join(item_ids),
    }
    response = requests.get(
        PAGE_URL,
        params=params,
        headers={"User-Agent": "PIF-research-raw-data-collection/1.0", "X-Requested-With": "XMLHttpRequest"},
        timeout=60,
    )
    response.raise_for_status()
    return response.content, response.json()


def unpack_series(payload: Any, item_ids: list[str]) -> dict[str, list[list[Any]]]:
    if isinstance(payload, dict):
        return {str(key): value for key, value in payload.items() if isinstance(value, list)}
    if not isinstance(payload, list):
        return {}
    if len(item_ids) == 1 and len(payload) == 1 and isinstance(payload[0], list):
        return {item_ids[0]: payload[0]}
    if len(payload) == len(item_ids):
        return {item_id: values for item_id, values in zip(item_ids, payload) if isinstance(values, list)}
    return {}


def normalize_points(payload: Any, item_ids: list[str], key: str, view: str, start: date, end: date) -> list[dict[str, Any]]:
    records = []
    for item_id, points in unpack_series(payload, item_ids).items():
        for point in points:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                records.append({
                    "data_key_source": key,
                    "data_view_source": view,
                    "series_id_source": item_id,
                    "timestamp_ms_source": point[0],
                    "value_source": point[1],
                    "requested_period_start": start.isoformat(),
                    "requested_period_end": end.isoformat(),
                })
    return records


def save_raw(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(body)


def load_or_fetch(path: Path, start: date, end: date, key: str, view: str, ids: list[str], resume: bool) -> Any:
    if path.exists():
        if not resume:
            raise FileExistsError(f"Refusing to replace existing raw response: {path}")
        return json.loads(path.read_text(encoding="utf-8"))
    body, payload = request_payload(start, end, key, view, ids)
    save_raw(path, body)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    folder = args.output_dir / "investfunds" / "public_market_statistics_raw"
    folder.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for year in range(2012, 2027):
        start = date(year, 1, 1)
        end = min(date(year, 12, 31), END_DATE)
        if start > END_DATE:
            break
        path = folder / f"fund_count_by_type_{year}.json"
        payload = load_or_fetch(path, start, end, "quant", "ft", COUNT_SERIES, args.resume)
        year_rows = normalize_points(payload, COUNT_SERIES, "quant", "ft", start, end)
        records.extend(year_rows)
        print(f"[Investfunds fund count] {year}: {len(year_rows)} source points", flush=True)

    flow_path = folder / "monthly_net_inflow_outflow_market_2012-01-01_2026-08-31.json"
    flow_payload = load_or_fetch(flow_path, START_DATE, END_DATE, "inflow", "", FLOW_SERIES, args.resume)
    flow_rows = normalize_points(flow_payload, FLOW_SERIES, "inflow", "", START_DATE, END_DATE)
    records.extend(flow_rows)

    frame = pd.DataFrame(records)
    csv_path = folder / "fund_count_and_market_net_flows_public_source_points.csv"
    parquet_path = folder / "fund_count_and_market_net_flows_public_source_points.parquet"
    for path in (csv_path, parquet_path):
        if path.exists():
            raise FileExistsError(f"Refusing to replace existing normalized output: {path}")
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    frame.to_parquet(parquet_path, index=False, compression="zstd")

    summary = {
        "source_url": PAGE_URL,
        "requested_period": [START_DATE.isoformat(), END_DATE.isoformat()],
        "daily_fund_count_source_points": int(frame[frame["data_key_source"].eq("quant")].shape[0]),
        "monthly_market_flow_source_points": int(frame[frame["data_key_source"].eq("inflow")].shape[0]),
        "unique_series": frame[["data_key_source", "series_id_source"]].drop_duplicates().to_dict("records"),
        "csv": csv_path.name,
        "parquet": parquet_path.name,
        "raw_responses": sorted(p.name for p in folder.glob("*.json")),
    }
    summary_path = folder / "public_market_statistics_collection_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to replace existing summary: {summary_path}")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
