"""Retry unresolved ISINs by scanning every currently available MOEX ISS market.

The regular downloader resolves ISINs through ISS's global securities search.
This companion scanner is a fallback for codes that search does not return: it
enumerates each ISS engine/market securities list and matches its ISIN column.
It is resumable at the output level and only changes the main mapping when
``--apply`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ISS_ROOT = "https://iss.moex.com/iss"


class RateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        self.interval = 1.0 / max(requests_per_second, 0.1)
        self.next_allowed = 0.0

    def wait(self) -> None:
        delay = self.next_allowed - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self.next_allowed = max(self.next_allowed, time.monotonic()) + self.interval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan all current MOEX ISS markets for unresolved ISINs."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--prefix",
        default="RU",
        help="Only retry unresolved ISINs having this prefix; use an empty value for all.",
    )
    parser.add_argument("--rps", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Update mapping_cache and mapping.csv for matches found by the scan.",
    )
    return parser.parse_args()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def records(payload: dict[str, Any], block_name: str) -> list[dict[str, Any]]:
    block = payload.get(block_name, {})
    columns = block.get("columns", [])
    return [dict(zip(columns, row)) for row in block.get("data", [])]


def get_value(row: dict[str, Any], key: str) -> Any:
    expected = key.casefold()
    for column, value in row.items():
        if str(column).casefold() == expected:
            return value
    return ""


def fetch(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    limiter: RateLimiter,
    timeout: float,
) -> dict[str, Any]:
    limiter.wait()
    response = session.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def load_cached_mappings(out_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in (out_dir / "mapping_cache").glob("*.json"):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return rows


def write_mapping_table(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "requested_isin", "status", "secid", "engine", "market", "shortname", "name",
        "type", "group", "is_traded", "primary_boardid", "marketprice_boardid",
        "matched_by", "error", "checked_at",
    ]
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    atomic_csv(frame[columns].sort_values("requested_isin"), out_dir / "mapping.csv")


def choose_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    def rank(candidate: dict[str, Any]) -> tuple[int, int, str, str]:
        return (
            int(str(candidate.get("status", "")).upper() != "A"),
            int(candidate.get("engine") != "stock"),
            str(candidate.get("engine", "")),
            str(candidate.get("market", "")),
        )

    return sorted(candidates, key=rank)[0]


def main() -> None:
    args = parse_args()
    mappings = load_cached_mappings(args.out_dir)
    targets = {
        str(row.get("requested_isin", "")).upper()
        for row in mappings
        if row.get("status") == "unresolved"
        and str(row.get("requested_isin", "")).upper().startswith(args.prefix.upper())
    }
    if not targets:
        print("No unresolved ISINs match the requested prefix.", flush=True)
        return

    session = requests.Session()
    session.headers.update({"User-Agent": "dissertation-moex-quotes/1.0"})
    limiter = RateLimiter(args.rps)
    engine_rows = records(
        fetch(session, f"{ISS_ROOT}/engines.json", {"iss.meta": "off"}, limiter, args.timeout),
        "engines",
    )
    engines = [str(get_value(row, "name")) for row in engine_rows if get_value(row, "name")]
    matches: dict[str, list[dict[str, Any]]] = {isin: [] for isin in targets}
    scan_errors: list[dict[str, str]] = []
    scanned_markets: list[dict[str, str]] = []
    params = {
        "iss.meta": "off",
        "securities.columns": "SECID,BOARDID,ISIN,SHORTNAME,SECNAME,STATUS",
    }

    for engine in engines:
        try:
            market_rows = records(
                fetch(
                    session,
                    f"{ISS_ROOT}/engines/{engine}/markets.json",
                    {"iss.meta": "off"},
                    limiter,
                    args.timeout,
                ),
                "markets",
            )
        except Exception as exc:
            scan_errors.append({"engine": engine, "market": "", "error": str(exc)})
            continue
        markets = [str(get_value(row, "name")) for row in market_rows if get_value(row, "name")]
        for market in markets:
            scanned_markets.append({"engine": engine, "market": market})
            try:
                security_rows = records(
                    fetch(
                        session,
                        f"{ISS_ROOT}/engines/{engine}/markets/{market}/securities.json",
                        params,
                        limiter,
                        args.timeout,
                    ),
                    "securities",
                )
            except Exception as exc:
                scan_errors.append({"engine": engine, "market": market, "error": str(exc)})
                continue
            for row in security_rows:
                isin = str(get_value(row, "ISIN")).strip().upper()
                if isin not in targets:
                    continue
                matches[isin].append(
                    {
                        "requested_isin": isin,
                        "secid": str(get_value(row, "SECID")),
                        "engine": engine,
                        "market": market,
                        "boardid": str(get_value(row, "BOARDID")),
                        "shortname": get_value(row, "SHORTNAME"),
                        "name": get_value(row, "SECNAME"),
                        "status": str(get_value(row, "STATUS")),
                    }
                )

    result_rows = []
    selected: dict[str, dict[str, Any]] = {}
    for isin in sorted(targets):
        candidates = matches[isin]
        if candidates:
            choice = choose_candidate(candidates)
            selected[isin] = choice
            result_rows.append(choice | {"scan_status": "found", "candidate_count": len(candidates)})
        else:
            result_rows.append({"requested_isin": isin, "scan_status": "not_found", "candidate_count": 0})
    result_frame = pd.DataFrame(result_rows)
    atomic_csv(result_frame, args.out_dir / "all_market_scan.csv")

    applied = 0
    if args.apply and selected:
        by_isin = {str(row.get("requested_isin", "")).upper(): row for row in mappings}
        for isin, candidate in selected.items():
            original = by_isin[isin]
            updated = original | {
                "status": "resolved",
                "secid": candidate["secid"],
                "engine": candidate["engine"],
                "market": candidate["market"],
                "shortname": candidate["shortname"],
                "name": candidate["name"],
                "primary_boardid": candidate["boardid"],
                "is_traded": candidate["status"],
                "matched_by": "all_market_scan",
                "checked_at": now(),
            }
            atomic_json(args.out_dir / "mapping_cache" / f"{isin}.json", updated)
            by_isin[isin] = updated
            applied += 1
        mappings = list(by_isin.values())
        write_mapping_table(args.out_dir, mappings)

    summary = {
        "checked_at": now(),
        "target_isins": len(targets),
        "scanned_markets": len(scanned_markets),
        "matched_isins": len(selected),
        "not_found_isins": len(targets) - len(selected),
        "candidate_markets": Counter(
            f"{row['engine']}/{row['market']}" for row in selected.values()
        ),
        "scan_errors": scan_errors,
        "applied_matches": applied,
    }
    atomic_json(args.out_dir / "all_market_scan_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
