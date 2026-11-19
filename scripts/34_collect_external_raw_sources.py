"""Collect public raw source files needed for the PIF/news study.

The script creates a new destination (or resumes only with --resume). It does
not modify existing project data and does not calculate analytical variables.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
import xml.etree.ElementTree as ET

import pandas as pd
import requests
from bs4 import BeautifulSoup


DEFAULT_ROOT = Path("./market_data/raw_external_2026-09-30")
HOLDINGS = Path("./funds_data/investfunds_fund_holdings/processed/fund_holdings_long.csv.gz")
FUND_CATALOG = Path("./funds_data/investfunds_fund_holdings/runs/2026_07_01/funds_catalog.csv")
MAC_UNIVERSE = Path(__file__).resolve().parents[1] / "tmp/moex_quote_universe_2008_2026.csv"
START_DATE = "2012-01-01"
END_DATE = "2026-08-31"
MOEX_ROOT = "https://iss.moex.com/iss"
INVESTFUNDS_URL = "https://investfunds.ru/funds/1566/"
RANKING_URL = "https://investfunds.ru/fund-rankings/fund-inflow/"


class Limiter:
    def __init__(self, rps: float) -> None:
        self.interval = 1.0 / max(rps, 0.05)
        self.next_at = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            allowed = max(now, self.next_at)
            self.next_at = allowed + self.interval
        if allowed > now:
            time.sleep(allowed - now)


def get(url: str, params: Any = None, retries: int = 4, timeout: int = 60) -> requests.Response:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers={"User-Agent": "PIF-research-raw-data-collection/1.0"},
            )
            if response.status_code == 429 or response.status_code >= 500:
                raise requests.HTTPError(f"HTTP {response.status_code}")
            response.raise_for_status()
            return response
        except Exception as exc:
            last = exc
            if attempt + 1 < retries:
                time.sleep(min(12, 1.5 ** attempt))
    raise RuntimeError(f"{url}: {last}")


def refuse_or_resume(root: Path, resume: bool) -> None:
    if root.exists() and not resume:
        raise FileExistsError(f"Output exists; pass --resume to continue without replacing files: {root}")
    root.mkdir(parents=True, exist_ok=True)


def save_bytes_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(content)


def write_csv_new(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        frame.to_csv(path, index=False, encoding="utf-8-sig")


def write_parquet_new(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        frame.to_parquet(path, index=False, compression="zstd")


def collect_cbr(root: Path, resume: bool) -> dict[str, Any]:
    out = root / "central_bank"
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}

    usd_xml = out / "USD_RUB_official_dynamic_2012-01-01_2026-08-31.xml"
    if not usd_xml.exists():
        response = get(
            "https://www.cbr.ru/scripts/XML_dynamic.asp",
            {"date_req1": "01/01/2012", "date_req2": "31/08/2026", "VAL_NM_RQ": "R01235"},
        )
        save_bytes_new(usd_xml, response.content)
    root_xml = ET.parse(usd_xml).getroot()
    usd_rows = []
    for row in root_xml.findall("Record"):
        usd_rows.append({
            "Date": row.get("Date"),
            "Id": row.get("Id"),
            "Nominal": row.findtext("Nominal"),
            "Value": row.findtext("Value"),
            "VunitRate": row.findtext("VunitRate"),
        })
    usd = pd.DataFrame(usd_rows)
    write_csv_new(usd, out / "USD_RUB_official_daily.csv")
    usd_dates = pd.to_datetime(usd["Date"], dayfirst=True, errors="coerce") if len(usd) else pd.Series(dtype="datetime64[ns]")
    report["USD_RUB"] = {
        "rows": len(usd),
        "min_date": str(usd_dates.min().date()) if len(usd_dates.dropna()) else None,
        "max_date": str(usd_dates.max().date()) if len(usd_dates.dropna()) else None,
    }

    ruonia_csv = out / "RUONIA_daily.csv"
    ruonia_xlsx = out / "RUONIA_official_dynamic_2012-01-01_2026-08-31.xlsx"
    if not ruonia_xlsx.exists():
        response = get(
            "https://cbr.ru/hd_base/ruonia/dynamics/",
            {"UniDbQuery.From": "01.01.2012", "UniDbQuery.To": "31.08.2026", "UniDbQuery.Posted": "True"},
        )
        html = BeautifulSoup(response.content, "html.parser")
        link = next((a.get("href") for a in html.select("a[href]") if "DownloadExcel" in a.get("href", "")), None)
        if not link:
            raise RuntimeError("Could not find the official RUONIA XLSX export in the Bank of Russia response")
        workbook = get(urljoin("https://cbr.ru", link))
        save_bytes_new(ruonia_xlsx, workbook.content)
    if not ruonia_csv.exists():
        try:
            frame = pd.read_excel(ruonia_xlsx)
        except Exception as exc:
            raise RuntimeError(f"RUONIA workbook saved, but Excel parser failed: {exc}") from exc
        frame.to_csv(ruonia_csv, index=False, encoding="utf-8-sig")
    frame = pd.read_csv(ruonia_csv, encoding="utf-8-sig")
    date_col = next((c for c in frame.columns if str(c).strip().upper() in {"DT", "DATE", "TRADEDATE"}), frame.columns[0])
    dates = pd.to_datetime(frame[date_col], dayfirst=True, errors="coerce").dropna()
    report["RUONIA"] = {"rows": len(frame), "min_date": str(dates.min().date()) if len(dates) else None, "max_date": str(dates.max().date()) if len(dates) else None, "columns": frame.columns.tolist()}
    return report


def collect_one_index(code: str, root: Path, resume: bool, limiter: Limiter) -> dict[str, Any]:
    folder = root / "moex" / "indices"
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / f"{code}_daily_history_{START_DATE}_{END_DATE}.csv"
    raw_path = folder / f"{code}_daily_history_{START_DATE}_{END_DATE}.jsonl.gz"
    if csv_path.exists() and raw_path.exists() and resume:
        frame = pd.read_csv(csv_path, low_memory=False)
        date_col = next((c for c in frame.columns if c.upper() == "TRADEDATE"), "TRADEDATE")
        return {"index": code, "status": "existing_in_supplement", "rows": len(frame), "min_date": frame[date_col].min() if len(frame) else None, "max_date": frame[date_col].max() if len(frame) else None}

    start = 0
    raw_pages: list[bytes] = []
    records: list[list[Any]] = []
    columns: list[str] | None = None
    while True:
        limiter.wait()
        response = get(
            f"{MOEX_ROOT}/history/engines/stock/markets/index/securities/{code}.json",
            {"from": START_DATE, "till": END_DATE, "start": start, "limit": 100, "iss.meta": "off"},
        )
        raw_pages.append(response.content)
        payload = response.json()
        block = payload.get("history", {})
        page = block.get("data", [])
        columns = columns or block.get("columns", [])
        if not page:
            break
        records.extend(page)
        start += len(page)
        cursor = payload.get("history.cursor", {}).get("data", [])
        total = int(cursor[0][1]) if cursor and len(cursor[0]) > 1 and cursor[0][1] is not None else None
        if (total is not None and start >= total) or len(page) < 100:
            break
    frame = pd.DataFrame(records, columns=columns or [])
    if "TRADEDATE" in frame.columns:
        frame = frame.sort_values("TRADEDATE", kind="stable").reset_index(drop=True)
    write_csv_new(frame, csv_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        with gzip.open(raw_path, "wb") as stream:
            for body in raw_pages:
                stream.write(body.replace(b"\n", b" ") + b"\n")
    return {"index": code, "status": "ok", "rows": len(frame), "min_date": frame["TRADEDATE"].min() if len(frame) and "TRADEDATE" in frame else None, "max_date": frame["TRADEDATE"].max() if len(frame) and "TRADEDATE" in frame else None, "columns": frame.columns.tolist()}


def collect_indices(root: Path, resume: bool, rps: float) -> list[dict[str, Any]]:
    codes = [
        "IMOEX", "IMOEX2", "RVI", "RGBI", "RGBITR",
        # Older broad corporate-bond tickers and newer active index families.
        "RUCBITR", "RUCBICP", "RUCBITR1Y", "RUCBTRNS", "RUCBCPNS",
        "RUCBTR3YNS", "RUCBTR5YNS",
    ]
    limiter = Limiter(rps)
    results = []
    for code in codes:
        try:
            result = collect_one_index(code, root, resume, limiter)
        except Exception as exc:
            result = {"index": code, "status": "error", "error": str(exc)}
        results.append(result)
        print(f"[MOEX index] {code}: {result.get('status')} rows={result.get('rows', 0)} {result.get('min_date')}..{result.get('max_date')}", flush=True)
    return results


def month_ends(start_year: int = 2012, end_year: int = 2026, end_month: int = 8) -> list[str]:
    dates = []
    for year in range(start_year, end_year + 1):
        last_month = end_month if year == end_year else 12
        for month in range(1, last_month + 1):
            day = pd.Period(f"{year}-{month:02d}").end_time.date()
            dates.append(day.isoformat())
    return dates


def parse_inflow_page(html: bytes, as_of: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.select("table")
    if len(tables) < 2:
        return []
    fund_rows = tables[-2].select("tr")[1:]
    flow_rows = tables[-1].select("tr")[1:]
    parsed = []
    for index, (fund_row, flow_row) in enumerate(zip(fund_rows, flow_rows)):
        fund_text = fund_row.get_text(" ", strip=True)
        anchor = fund_row.select_one('a[href^="/funds/"]')
        flow_cells = [cell.get_text(" ", strip=True) for cell in flow_row.select("td,th")]
        if len(flow_cells) < 6:
            continue
        fund_href = anchor.get("href", "") if anchor else ""
        fund_name = anchor.get_text(" ", strip=True) if anchor else fund_text
        company_display = fund_text.split(fund_name, 1)[-1].strip() if fund_name else ""
        match = re.search(r"/funds/(\d+)/", fund_href)
        rank = re.match(r"\s*(\d+)", fund_text)
        uk_anchor = flow_row.select_one('a[href^="/uk/"]')
        parsed.append({
            "ranking_date": as_of,
            "rank_position": int(rank.group(1)) if rank else None,
            "fund_id": int(match.group(1)) if match else None,
            "fund_name_display": fund_name,
            "management_company_display_source": company_display,
            "fund_page_url": urljoin("https://investfunds.ru", fund_href) if fund_href else None,
            "ranking_row_text": fund_text,
            "management_company": uk_anchor.get_text(" ", strip=True) if uk_anchor else None,
            "management_company_page_url": urljoin("https://investfunds.ru", uk_anchor.get("href", "")) if uk_anchor else None,
            "attracted_funds_rub_source_value": flow_cells[1],
            "fund_type": flow_cells[2],
            "investment_object": flow_cells[3],
            "specialization": flow_cells[4],
            "geography": flow_cells[5],
        })
    return parsed


def collect_inflow_ranking(root: Path, resume: bool, rps: float) -> dict[str, Any]:
    folder = root / "investfunds" / "monthly_flow_rankings_html_gz"
    folder.mkdir(parents=True, exist_ok=True)
    limiter = Limiter(rps)
    rows: list[dict[str, Any]] = []
    failures = []
    targets = month_ends()
    for i, as_of in enumerate(targets, start=1):
        raw_path = folder / f"fund_inflow_ranking_{as_of}.html.gz"
        try:
            if raw_path.exists() and resume:
                with gzip.open(raw_path, "rb") as stream:
                    raw = stream.read()
            else:
                limiter.wait()
                response = get(RANKING_URL, {"date": as_of, "pId": "1"}, timeout=90)
                raw = response.content
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                if not raw_path.exists():
                    with gzip.open(raw_path, "wb") as stream:
                        stream.write(raw)
            parsed = parse_inflow_page(raw, as_of)
            rows.extend(parsed)
            if not parsed:
                failures.append({"date": as_of, "error": "no fund rows parsed"})
        except Exception as exc:
            failures.append({"date": as_of, "error": str(exc)})
        if i % 12 == 0 or i == len(targets):
            print(f"[Investfunds inflows] {i}/{len(targets)} months; parsed rows={len(rows)}; failures={len(failures)}", flush=True)
    frame = pd.DataFrame(rows)
    write_csv_new(frame, root / "investfunds" / "public_monthly_fund_inflow_ranking_verified.csv")
    write_parquet_new(frame, root / "investfunds" / "public_monthly_fund_inflow_ranking_verified.parquet")
    write_csv_new(pd.DataFrame(failures), root / "investfunds" / "public_monthly_fund_inflow_ranking_fetch_failures_verified.csv")
    fund_set = frame.groupby("ranking_date", dropna=False)["fund_id"].nunique().reset_index(name="funds_in_public_ranking") if len(frame) else pd.DataFrame()
    write_csv_new(fund_set, root / "investfunds" / "public_monthly_ranking_counts_verified.csv")
    return {"months_requested": len(targets), "months_failed": len(failures), "records": len(frame), "fund_ids": int(frame["fund_id"].nunique()) if len(frame) else 0, "min_date": frame["ranking_date"].min() if len(frame) else None, "max_date": frame["ranking_date"].max() if len(frame) else None}


def load_funds(root: Path) -> pd.DataFrame:
    if not FUND_CATALOG.exists():
        raise FileNotFoundError(FUND_CATALOG)
    catalog = pd.read_csv(FUND_CATALOG, encoding="utf-8-sig", dtype={"fund_id": "int64"})
    catalog = catalog.drop_duplicates("fund_id").sort_values("fund_id")
    write_csv_new(catalog, root / "investfunds" / "fund_ids_from_holdings.csv")
    return catalog


def collect_chart_batch(ids: list[int], data_key: str, index: int, root: Path, resume: bool, limiter: Limiter) -> dict[str, Any]:
    folder = root / "investfunds" / "nav_unit_value_chart_raw"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{data_key}_batch_{index:04d}.json"
    if path.exists() and resume:
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        limiter.wait()
        params: list[tuple[str, str]] = [
            ("action", "chartData"), ("data_key", data_key),
            ("date_from", "01.01.2012"), ("date_to", "31.08.2026"),
            ("currencyId", "1"),
        ] + [("ids[]", str(fund_id)) for fund_id in ids]
        response = get(INVESTFUNDS_URL, params, timeout=75)
        payload = response.json()
        save_bytes_new(path, json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    out_rows = []
    errors = []
    series_list = payload if isinstance(payload, list) else []
    if len(series_list) != len(ids):
        # The public chart endpoint omits fund IDs from each returned series.
        # If it omits an empty/no-history series, response-order alignment is
        # ambiguous, so request this small batch one fund at a time.
        series_list = []
        for fund_id in ids:
            single_path = folder / f"{data_key}_single_{fund_id}.json"
            if single_path.exists() and resume:
                single = json.loads(single_path.read_text(encoding="utf-8"))
            else:
                limiter.wait()
                params = [
                    ("action", "chartData"), ("data_key", data_key),
                    ("date_from", "01.01.2012"), ("date_to", "31.08.2026"),
                    ("currencyId", "1"), ("ids[]", str(fund_id)),
                ]
                single_response = get(INVESTFUNDS_URL, params, timeout=75)
                single = single_response.json()
                save_bytes_new(single_path, json.dumps(single, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            if isinstance(single, list) and single:
                series_list.append((fund_id, single[0]))
            else:
                errors.append({"fund_id": fund_id, "data_key": data_key, "status": "series_missing_from_response"})
    else:
        series_list = list(zip(ids, series_list))
    returned_ids: set[int] = set()
    for fund_id, series in series_list:
        if fund_id is not None:
            returned_ids.add(int(fund_id))
        for point in series.get("data", []) or []:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                out_rows.append({"fund_id": fund_id, "data_key": data_key, "timestamp_ms_source": point[0], "value_source": point[1], "series_name_source": series.get("name"), "series_tooltip_source": series.get("tooltip")})
        if not series.get("data"):
            errors.append({"fund_id": fund_id, "data_key": data_key, "status": "no_points"})
    errors.extend({"fund_id": fund_id, "data_key": data_key, "status": "series_missing_from_response"} for fund_id in ids if fund_id not in returned_ids and not any(e.get("fund_id") == fund_id for e in errors))
    return {"rows": out_rows, "errors": errors}


def collect_fund_chart_data(root: Path, resume: bool, rps: float, workers: int) -> dict[str, Any]:
    catalog = load_funds(root)
    ids = catalog["fund_id"].astype(int).tolist()
    batches = [ids[i:i + 5] for i in range(0, len(ids), 5)]
    limiter = Limiter(rps)
    combined_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for data_key in ("pay", "sca"):
        results = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(collect_chart_batch, batch, data_key, i, root, resume, limiter): i for i, batch in enumerate(batches, start=1)}
            for n, future in enumerate(as_completed(futures), start=1):
                try:
                    results.append(future.result())
                except Exception as exc:
                    failures.append({"data_key": data_key, "batch": futures[future], "error": str(exc)})
                if n % 25 == 0 or n == len(futures):
                    print(f"[Investfunds {data_key}] {n}/{len(futures)} batches", flush=True)
        for result in results:
            combined_rows.extend(result["rows"])
            failures.extend(result["errors"])
    frame = pd.DataFrame(combined_rows)
    write_csv_new(frame, root / "investfunds" / "fund_nav_and_unit_value_public_chart_data_verified.csv")
    write_parquet_new(frame, root / "investfunds" / "fund_nav_and_unit_value_public_chart_data_verified.parquet")
    write_csv_new(pd.DataFrame(failures), root / "investfunds" / "fund_nav_and_unit_value_fetch_gaps_verified.csv")
    stats = {}
    if len(frame):
        for key, part in frame.groupby("data_key"):
            ts = pd.to_datetime(part["timestamp_ms_source"], unit="ms", utc=True, errors="coerce")
            stats[key] = {"rows": len(part), "fund_ids": int(part["fund_id"].nunique()), "min_timestamp_utc": str(ts.min()) if len(ts.dropna()) else None, "max_timestamp_utc": str(ts.max()) if len(ts.dropna()) else None}
    return {"funds_requested": len(ids), "requests_per_data_key": len(batches), "failures": len(failures), "series": stats}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--task", choices=["all", "cbr", "indices", "investfunds", "inflows", "nav"], default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rps", type=float, default=1.5, help="shared request rate per source service")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    refuse_or_resume(args.output_dir, args.resume)
    summary_path = args.output_dir / "collection_summary.json"
    if args.resume and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {"source_period": [START_DATE, END_DATE]}
    summary["last_updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()

    if args.task in {"all", "cbr"}:
        summary["central_bank"] = collect_cbr(args.output_dir, args.resume)
    if args.task in {"all", "indices"}:
        summary["moex_indices"] = collect_indices(args.output_dir, args.resume, args.rps)
    if args.task in {"all", "investfunds", "inflows"}:
        summary["investfunds_inflows"] = collect_inflow_ranking(args.output_dir, args.resume, args.rps)
    if args.task in {"all", "investfunds", "nav"}:
        summary["investfunds_nav_and_unit_value"] = collect_fund_chart_data(args.output_dir, args.resume, args.rps, args.workers)

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
