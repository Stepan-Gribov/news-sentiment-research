"""Download MOEX daily histories for the ISIN universe used by the PIF panel.

The script is deliberately resumable.  Mapping responses and per-ISIN histories
are stored separately, so an interrupted run only retries incomplete items.
The input universe is a compact CSV built from Y_interpolated_updated.csv and
must contain a ``requested_isin`` column.

MOEX ISS endpoints used here:
  /iss/securities.json?q=<ISIN>                         ISIN -> SECID mapping
  /iss/history/engines/stock/markets/<market>/securities/<SECID>.json
                                                        daily trading history
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ISS_ROOT = "https://iss.moex.com/iss"
ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}\d$", re.IGNORECASE)
PAGE_LIMIT = 100
DEFAULT_FROM = "2012-03-31"
DEFAULT_TO = "2026-06-30"
DEFAULT_RPS = 2.0
DEFAULT_WORKERS = 4
BOARD_RANK = {
    "TQBR": 1,
    "TQTF": 2,
    "TQOB": 3,
    "TQCB": 4,
    "TQIR": 5,
    "TQPI": 6,
    "TQTD": 7,
    "EQRP": 8,
    "FQBR": 9,
}


class RateLimiter:
    """One aggregate request budget shared by all worker threads."""

    def __init__(self, requests_per_second: float) -> None:
        self.interval = 1.0 / max(float(requests_per_second), 0.01)
        self._next_allowed = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_allowed)
            self._next_allowed = slot + self.interval
        delay = slot - now
        if delay > 0:
            time.sleep(delay)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download resumable MOEX histories for a PIF ISIN universe."
    )
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--from-date", default=DEFAULT_FROM)
    parser.add_argument("--to-date", default=DEFAULT_TO)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--rps", type=float, default=DEFAULT_RPS)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--stage", choices=["mapping", "history", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temp, index=False, encoding="utf-8-sig")
    temp.replace(path)


def atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(temp, index=False, compression="zstd")
    temp.replace(path)


def atomic_history_frame(df: pd.DataFrame, out_dir: Path, isin: str) -> tuple[Path, str]:
    """Prefer Parquet, with a CSV.GZ fallback for minimal Python installs."""
    parquet_path = out_dir / "histories" / f"{isin}.parquet"
    try:
        atomic_parquet(df, parquet_path)
        return parquet_path, "parquet"
    except (ImportError, ModuleNotFoundError):
        csv_path = out_dir / "histories" / f"{isin}.csv.gz"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        temp = csv_path.with_suffix(csv_path.suffix + ".tmp")
        df.to_csv(temp, index=False, compression="gzip", encoding="utf-8")
        temp.replace(csv_path)
        return csv_path, "csv.gz"


def load_universe(path: Path, limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "requested_isin" not in df.columns:
        raise ValueError(f"Universe must contain requested_isin: {path}")
    df["requested_isin"] = df["requested_isin"].astype(str).str.strip().str.upper()
    df = df[df["requested_isin"].map(lambda value: bool(ISIN_RE.fullmatch(value)))].copy()
    df = df.drop_duplicates("requested_isin").sort_values("requested_isin").reset_index(drop=True)
    if limit is not None:
        df = df.head(max(limit, 0)).copy()
    return df


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "dissertation-moex-quotes/1.0"})
    return session


def request_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    limiter: RateLimiter,
    retries: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        limiter.wait()
        try:
            response = session.get(url, params=params, timeout=45)
            if response.status_code == 429 or response.status_code >= 500:
                raise requests.HTTPError(f"HTTP {response.status_code}")
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # network/API failures are recorded per item
            last_error = exc
            if attempt < retries:
                time.sleep(min(30.0, (2 ** (attempt - 1)) + random.random()))
    raise RuntimeError(f"request failed after {retries} attempts: {last_error}")


def market_from_row(row: dict[str, Any]) -> str:
    text = " ".join(str(row.get(key, "")) for key in ("group", "type", "primary_boardid")).lower()
    if "bond" in text or str(row.get("primary_boardid", "")).upper().startswith("TQO"):
        return "bonds"
    return "shares"


def resolve_one(
    isin: str,
    mapping_path: Path,
    limiter: RateLimiter,
    retries: int,
    force: bool,
) -> dict[str, Any]:
    if mapping_path.exists() and not force:
        try:
            cached = json.loads(mapping_path.read_text(encoding="utf-8"))
            if cached.get("status") in {"resolved", "unresolved"}:
                return cached
        except Exception:
            pass

    try:
        session = make_session()
        payload = request_json(
            session,
            f"{ISS_ROOT}/securities.json",
            {"q": isin, "iss.meta": "off", "limit": 50},
            limiter,
            retries,
        )
        block = payload.get("securities", {})
        columns = block.get("columns", [])
        records = [dict(zip(columns, row)) for row in block.get("data", [])]
        exact = [
            row for row in records
            if str(row.get("isin", "")).strip().upper() == isin
        ]
        if not exact:
            result = {"requested_isin": isin, "status": "unresolved", "checked_at": utc_now()}
        else:
            exact.sort(
                key=lambda row: (
                    -int(bool(row.get("is_traded"))),
                    BOARD_RANK.get(str(row.get("primary_boardid", "")).upper(), 999),
                    str(row.get("secid", "")),
                )
            )
            row = exact[0]
            result = {
                "requested_isin": isin,
                "status": "resolved",
                "secid": str(row.get("secid", "")),
                "engine": "stock",
                "market": market_from_row(row),
                "shortname": row.get("shortname"),
                "name": row.get("name"),
                "type": row.get("type"),
                "group": row.get("group"),
                "is_traded": row.get("is_traded"),
                "primary_boardid": row.get("primary_boardid"),
                "marketprice_boardid": row.get("marketprice_boardid"),
                "checked_at": utc_now(),
            }
    except Exception as exc:
        result = {
            "requested_isin": isin,
            "status": "error",
            "error": str(exc),
            "checked_at": utc_now(),
        }
    atomic_json(mapping_path, result)
    return result


def write_mapping_table(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = [
        "requested_isin", "status", "secid", "engine", "market", "shortname", "name",
        "type", "group", "is_traded", "primary_boardid", "marketprice_boardid",
        "error", "checked_at",
    ]
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    atomic_csv(frame[columns].sort_values("requested_isin"), out_dir / "mapping.csv")


def resolve_universe(
    universe: pd.DataFrame,
    out_dir: Path,
    workers: int,
    rps: float,
    retries: int,
    force: bool,
) -> list[dict[str, Any]]:
    mapping_dir = out_dir / "mapping_cache"
    limiter = RateLimiter(rps)
    rows: list[dict[str, Any]] = []
    total = len(universe)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(workers, 1)) as executor:
        futures = {
            executor.submit(
                resolve_one,
                isin,
                mapping_dir / f"{isin}.json",
                limiter,
                retries,
                force,
            ): isin
            for isin in universe["requested_isin"]
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            rows.append(future.result())
            if index % 100 == 0 or index == total:
                counts = pd.Series([row.get("status") for row in rows]).value_counts().to_dict()
                print(f"[mapping] {index}/{total} {counts}", flush=True)
    write_mapping_table(out_dir, rows)
    return rows


def fetch_history(
    mapping: dict[str, Any],
    from_date: str,
    to_date: str,
    limiter: RateLimiter,
    retries: int,
) -> tuple[pd.DataFrame, int]:
    session = make_session()
    start = 0
    pages = 0
    rows: list[list[Any]] = []
    columns: list[str] | None = None
    url = (
        f"{ISS_ROOT}/history/engines/{mapping.get('engine', 'stock')}/markets/{mapping['market']}"
        f"/securities/{mapping['secid']}.json"
    )
    while True:
        payload = request_json(
            session,
            url,
            {
                "from": from_date,
                "till": to_date,
                "start": start,
                "limit": PAGE_LIMIT,
                "iss.meta": "off",
            },
            limiter,
            retries,
        )
        block = payload.get("history", {})
        page_rows = block.get("data", [])
        if columns is None:
            columns = block.get("columns", [])
        if not page_rows:
            break
        rows.extend(page_rows)
        pages += 1
        start += len(page_rows)
        cursor_rows = payload.get("history.cursor", {}).get("data", [])
        total = int(cursor_rows[0][1]) if cursor_rows and len(cursor_rows[0]) > 1 else None
        if (total is not None and start >= total) or len(page_rows) < PAGE_LIMIT:
            break
        if pages > 10000:
            raise RuntimeError("pagination exceeded 10,000 pages")
    frame = pd.DataFrame(rows, columns=columns or [])
    if not frame.empty:
        frame["REQUESTED_ISIN"] = mapping["requested_isin"]
        frame["SOURCE_SECID"] = mapping["secid"]
        frame["SOURCE_ENGINE"] = mapping.get("engine", "stock")
        frame["SOURCE_MARKET"] = mapping["market"]
        frame["FETCHED_AT_UTC"] = utc_now()
    return frame, pages


def history_status_path(out_dir: Path, isin: str) -> Path:
    return out_dir / "history_status" / f"{isin}.json"


def download_one_history(
    mapping: dict[str, Any],
    out_dir: Path,
    from_date: str,
    to_date: str,
    limiter: RateLimiter,
    retries: int,
    force: bool,
) -> dict[str, Any]:
    isin = mapping["requested_isin"]
    status_path = history_status_path(out_dir, isin)
    history_path = out_dir / "histories" / f"{isin}.parquet"
    if status_path.exists() and not force:
        try:
            cached = json.loads(status_path.read_text(encoding="utf-8"))
            cached_covers_range = (
                bool(cached.get("from_date"))
                and bool(cached.get("to_date"))
                and str(cached["from_date"]) <= from_date
                and str(cached["to_date"]) >= to_date
            )
            if (
                cached.get("status") == "ok"
                and cached_covers_range
                and Path(str(cached.get("file", ""))).exists()
            ):
                return cached | {"skipped": True}
            if cached.get("status") == "empty" and cached_covers_range:
                return cached | {"skipped": True}
        except Exception:
            pass
    try:
        frame, pages = fetch_history(mapping, from_date, to_date, limiter, retries)
        if frame.empty:
            result = {
                "requested_isin": isin,
                "status": "empty",
                "rows": 0,
                "pages": pages,
                "from_date": from_date,
                "to_date": to_date,
                "checked_at": utc_now(),
            }
        else:
            saved_path, file_format = atomic_history_frame(frame, out_dir, isin)
            result = {
                "requested_isin": isin,
                "status": "ok",
                "rows": len(frame),
                "pages": pages,
                "from_date": from_date,
                "to_date": to_date,
                "file": str(saved_path),
                "file_format": file_format,
                "checked_at": utc_now(),
            }
    except Exception as exc:
        result = {
            "requested_isin": isin,
            "status": "error",
            "rows": 0,
            "from_date": from_date,
            "to_date": to_date,
            "error": str(exc),
            "checked_at": utc_now(),
        }
    atomic_json(status_path, result)
    return result


def download_histories(
    mappings: list[dict[str, Any]],
    out_dir: Path,
    from_date: str,
    to_date: str,
    workers: int,
    rps: float,
    retries: int,
    force: bool,
) -> list[dict[str, Any]]:
    resolved = [row for row in mappings if row.get("status") == "resolved"]
    limiter = RateLimiter(rps)
    results: list[dict[str, Any]] = []
    total = len(resolved)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(workers, 1)) as executor:
        futures = {
            executor.submit(
                download_one_history,
                mapping,
                out_dir,
                from_date,
                to_date,
                limiter,
                retries,
                force,
            ): mapping["requested_isin"]
            for mapping in resolved
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if index % 25 == 0 or index == total:
                counts = pd.Series([row.get("status") for row in results]).value_counts().to_dict()
                rows = sum(int(row.get("rows", 0) or 0) for row in results)
                print(f"[history] {index}/{total} {counts} rows={rows:,}", flush=True)
    return results


def write_summary(out_dir: Path, universe: pd.DataFrame, mappings: list[dict[str, Any]], histories: list[dict[str, Any]]) -> None:
    summary = {
        "updated_at": utc_now(),
        "universe_isins": int(len(universe)),
        "mapping_status": pd.Series([row.get("status") for row in mappings]).value_counts().to_dict(),
        "history_status": pd.Series([row.get("status") for row in histories]).value_counts().to_dict(),
        "history_rows": int(sum(int(row.get("rows", 0) or 0) for row in histories)),
    }
    atomic_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    universe = load_universe(args.universe, args.limit)
    atomic_csv(universe, out_dir / "universe_used.csv")
    print(
        f"universe={len(universe)} from={args.from_date} to={args.to_date} "
        f"workers={args.workers} rps={args.rps}",
        flush=True,
    )

    mappings = []
    mapping_table = out_dir / "mapping.csv"
    if args.stage in {"mapping", "all"} or not mapping_table.exists():
        mappings = resolve_universe(
            universe, out_dir, args.workers, args.rps, args.retries, args.force
        )
    else:
        mappings = pd.read_csv(mapping_table, low_memory=False).fillna("").to_dict("records")

    histories: list[dict[str, Any]] = []
    if args.stage in {"history", "all"}:
        histories = download_histories(
            mappings,
            out_dir,
            args.from_date,
            args.to_date,
            args.workers,
            args.rps,
            args.retries,
            args.force,
        )
    write_summary(out_dir, universe, mappings, histories)


if __name__ == "__main__":
    main()
