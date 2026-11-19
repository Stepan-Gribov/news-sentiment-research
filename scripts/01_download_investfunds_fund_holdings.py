from __future__ import annotations

import argparse
import csv
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import _bootstrap  # noqa: F401

import requests
from bs4 import BeautifulSoup

from dissertation_pipeline.paths import PATHS
from dissertation_pipeline.selenium_fast import apply_fast_chrome_options, configure_fast_driver, fast_chrome_prefs


BASE_URL = "https://pro.investfunds.ru/"
FUNDS_AJAX_URL = urljoin(BASE_URL, "main_data_ajax.php")
DEFAULT_OUT_ROOT = PATHS.funds_dir / "investfunds_fund_holdings"
REQUEST_TIMEOUT = 45
MIN_VALID_XLS_BYTES = 1024
DEFAULT_RUN_ID_DATE_FORMAT = "%Y_%m_%d"

# Set manually when you want to continue an existing run folder from code/VSCode,
# for example: MANUAL_RUN_ID = "2026_07_01".
MANUAL_RUN_ID: str | None = "2026_07_01"

MANIFEST_FIELDNAMES = [
    "fund_id",
    "fund_name",
    "date",
    "date_iso",
    "download_id",
    "xls_url",
    "xls_path",
    "existing_source_run",
    "html_path",
    "status",
    "error",
    "bytes",
]
DATES_FIELDNAMES = ["fund_id", "fund_name", "date", "date_iso"]


@dataclass(frozen=True)
class FundOption:
    fund_id: str
    fund_name: str


@dataclass(frozen=True)
class DateOption:
    fund_id: str
    fund_name: str
    date_text: str
    date_iso: str


@dataclass(frozen=True)
class ExistingDownload:
    fund_id: str
    date_iso: str
    xls_path: Path
    bytes: int
    source_run: str


def build_driver(chromedriver: Path | None, user_data_dir: Path, download_dir: Path):
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service

    user_data_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    options = Options()
    apply_fast_chrome_options(
        options,
        prefs=fast_chrome_prefs(
            {
                "download.default_directory": str(download_dir),
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "safebrowsing.enabled": True,
            }
        ),
    )
    options.add_argument(f"--user-data-dir={user_data_dir}")
    if chromedriver:
        driver = webdriver.Chrome(service=Service(str(chromedriver)), options=options)
    else:
        driver = webdriver.Chrome(options=options)
    configure_fast_driver(driver)
    return driver


def selenium_session(
    chromedriver: Path | None,
    user_data_dir: Path,
    download_dir: Path,
    login_wait_seconds: int,
) -> tuple[requests.Session, str]:
    driver = build_driver(chromedriver, user_data_dir, download_dir)
    try:
        driver.get(BASE_URL)
        print(f"Opened {BASE_URL}. Log in if needed; waiting {login_wait_seconds}s...", flush=True)
        time.sleep(login_wait_seconds)
        page_html = driver.page_source
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": driver.execute_script("return navigator.userAgent;"),
                "Referer": BASE_URL,
            }
        )
        for cookie in driver.get_cookies():
            session.cookies.set(cookie["name"], cookie["value"], path=cookie.get("path", "/"))
            domain = cookie.get("domain")
            if domain:
                session.cookies.set(cookie["name"], cookie["value"], domain=domain.lstrip("."), path=cookie.get("path", "/"))
        return session, page_html
    finally:
        driver.quit()


def request_text(session: requests.Session, url: str, params: dict | None = None, retries: int = 4, sleep_seconds: float = 1.5) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            response.encoding = response.encoding or "utf-8"
            return response.text
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            wait = sleep_seconds * attempt
            print(f"request failed attempt={attempt}/{retries} wait={wait:.1f}s url={url} params={params}: {exc}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"request failed after retries: {url}") from last_error


def request_bytes(session: requests.Session, url: str, retries: int = 4, sleep_seconds: float = 1.5) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.content
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            wait = sleep_seconds * attempt
            print(f"download failed attempt={attempt}/{retries} wait={wait:.1f}s url={url}: {exc}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"download failed after retries: {url}") from last_error


def parse_funds(page_html: str) -> list[FundOption]:
    soup = BeautifulSoup(page_html, "html.parser")
    select = soup.find("select", {"id": "prop_funds_combo"})
    if select is None:
        return []

    funds: list[FundOption] = []
    for option in select.find_all("option"):
        fund_id = (option.get("value") or "").strip()
        fund_name = option.get_text(strip=True)
        if not fund_id or fund_id == "0" or not fund_name or "выберите" in fund_name.casefold():
            continue
        funds.append(FundOption(fund_id=fund_id, fund_name=fund_name))
    return funds


def parse_dates(fund: FundOption, dates_html: str) -> list[DateOption]:
    soup = BeautifulSoup(dates_html, "html.parser")
    select = soup.find("select", {"id": "prop_dates_combo"})
    if select is None:
        return []

    dates: list[DateOption] = []
    for option in select.find_all("option"):
        date_text = (option.get("value") or option.get_text(strip=True) or "").strip()
        if not date_text or date_text == "0" or "выберите" in date_text.casefold():
            continue
        date_iso = datetime.strptime(date_text, "%d.%m.%Y").date().isoformat()
        dates.append(DateOption(fund.fund_id, fund.fund_name, date_text, date_iso))
    return dates


def parse_xls_link(result_html: str) -> tuple[str | None, str | None]:
    soup = BeautifulSoup(result_html, "html.parser")
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "/funds_property/download/xls/" in href:
            match = re.search(r"/xls/(\d+)/?", href)
            download_id = match.group(1) if match else None
            return urljoin(BASE_URL, href), download_id
    return None, None


def slugify(value: str, max_len: int = 80) -> str:
    value = value.replace("\u00a0", " ")
    value = re.sub(r"[^\wА-Яа-яЁё.-]+", "_", value, flags=re.UNICODE)
    value = re.sub(r"_+", "_", value).strip("_.")
    return value[:max_len] or "fund"


def date_in_range(date_iso: str, start_date: str | None, end_date: str | None) -> bool:
    if start_date and date_iso < start_date:
        return False
    if end_date and date_iso > end_date:
        return False
    return True


def is_valid_xls(path: Path, min_bytes: int = MIN_VALID_XLS_BYTES) -> bool:
    if not path.exists() or path.stat().st_size < min_bytes:
        return False
    try:
        head = path.read_bytes()[:8]
    except OSError:
        return False
    # Legacy XLS is CDFV2, XLSX is ZIP. Some InvestFunds files are .xls with CDFV2.
    return head.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1") or head.startswith(b"PK")


def build_existing_download_index(out_root: Path, min_bytes: int) -> dict[tuple[str, str], ExistingDownload]:
    """Find already downloaded fund-date XLS files across previous runs."""
    index: dict[tuple[str, str], ExistingDownload] = {}
    runs_dir = out_root / "runs"
    if not runs_dir.exists():
        return index

    for manifest_path in sorted(runs_dir.glob("*/manifest.csv")):
        source_run = manifest_path.parent.name
        try:
            with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fund_id = str(row.get("fund_id") or "").strip()
                    date_iso = str(row.get("date_iso") or "").strip()
                    xls_path_text = str(row.get("xls_path") or "").strip()
                    if not fund_id or not date_iso or not xls_path_text:
                        continue
                    xls_path = Path(xls_path_text)
                    if not is_valid_xls(xls_path, min_bytes=min_bytes):
                        continue
                    key = (fund_id, date_iso)
                    current = index.get(key)
                    size = xls_path.stat().st_size
                    if current is None or size > current.bytes:
                        index[key] = ExistingDownload(fund_id, date_iso, xls_path, size, source_run)
        except Exception as exc:  # noqa: BLE001
            print(f"skip unreadable manifest={manifest_path}: {exc}", flush=True)

    for xls_path in sorted(runs_dir.glob("*/xls/*/*.xls")):
        source_run = xls_path.parents[2].name
        if not is_valid_xls(xls_path, min_bytes=min_bytes):
            continue
        fund_id = xls_path.parent.name
        match = re.match(r"(\d{4}-\d{2}-\d{2})_", xls_path.name)
        if not fund_id or not match:
            continue
        date_iso = match.group(1)
        key = (fund_id, date_iso)
        size = xls_path.stat().st_size
        current = index.get(key)
        if current is None or size > current.bytes:
            index[key] = ExistingDownload(fund_id, date_iso, xls_path, size, source_run)

    return index


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except Exception as exc:  # noqa: BLE001
        print(f"skip unreadable csv={path}: {exc}", flush=True)
        return []


def row_key(row: dict[str, object]) -> tuple[str, str]:
    return str(row.get("fund_id") or "").strip(), str(row.get("date_iso") or "").strip()


def resolve_run_id(cli_run_id: str | None) -> str:
    if cli_run_id:
        return cli_run_id
    if MANUAL_RUN_ID:
        return MANUAL_RUN_ID
    return datetime.now().strftime(DEFAULT_RUN_ID_DATE_FORMAT)


def load_initial_session(args: argparse.Namespace, download_dir: Path) -> tuple[requests.Session, str]:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": BASE_URL})
    page = request_text(session, BASE_URL)
    if parse_funds(page):
        return session, page

    print("prop_funds_combo not found via plain requests; falling back to Selenium session cookies.", flush=True)
    return selenium_session(args.chromedriver, args.user_data_dir, download_dir, args.login_wait_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download InvestFunds PIF holdings by iterating funds and available report dates."
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--run-id",
        "--run-folder",
        dest="run_id",
        default=None,
        help=(
            "Name of folder under runs/. If omitted, uses MANUAL_RUN_ID from code, "
            "otherwise current date as YYYY_MM_DD."
        ),
    )
    parser.add_argument("--chromedriver", type=Path, default=None)
    parser.add_argument("--user-data-dir", type=Path, default=PATHS.tmp_dir / "investfunds_chrome_profile")
    parser.add_argument("--login-wait-seconds", type=int, default=5)
    parser.add_argument("--sample-funds", type=int, default=None)
    parser.add_argument("--sample-dates", type=int, default=None)
    parser.add_argument("--fund-id", action="append", default=None, help="Limit to one or more InvestFunds fund ids.")
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD inclusive.")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD inclusive.")
    parser.add_argument("--sleep-seconds", type=float, default=0.7)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    parser.add_argument("--min-valid-xls-bytes", type=int, default=MIN_VALID_XLS_BYTES)
    parser.add_argument("--start-fund-index", type=int, default=0, help="Optional 0-based start index for manual debugging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_id = resolve_run_id(args.run_id)
    run_dir = args.out_root / "runs" / args.run_id
    xls_dir = run_dir / "xls"
    html_dir = run_dir / "html"
    download_tmp_dir = run_dir / "_browser_downloads"
    for path in [xls_dir, html_dir, download_tmp_dir]:
        path.mkdir(parents=True, exist_ok=True)

    session, page_html = load_initial_session(args, download_tmp_dir)
    funds = parse_funds(page_html)
    if not funds:
        raise RuntimeError("Could not find prop_funds_combo on InvestFunds page. Check login/session.")

    if args.fund_id:
        requested = set(args.fund_id)
        funds = [fund for fund in funds if fund.fund_id in requested]
    if args.sample_funds:
        funds = funds[: args.sample_funds]

    write_csv(
        run_dir / "funds_catalog.csv",
        [{"fund_id": f.fund_id, "fund_name": f.fund_name} for f in funds],
        ["fund_id", "fund_name"],
    )

    print(f"funds selected: {len(funds):,}", flush=True)
    existing_downloads = (
        build_existing_download_index(args.out_root, args.min_valid_xls_bytes)
        if args.skip_existing
        else {}
    )
    print(f"already downloaded fund-date pairs indexed: {len(existing_downloads):,}", flush=True)
    manifest_path = run_dir / "manifest.csv"
    dates_catalog_path = run_dir / "dates_catalog.csv"
    manifest_by_key = {
        row_key(row): row
        for row in read_csv_rows(manifest_path)
        if all(row_key(row))
    }
    dates_by_key = {
        row_key(row): row
        for row in read_csv_rows(dates_catalog_path)
        if all(row_key(row))
    }
    print(
        f"current run catalog loaded: manifest_rows={len(manifest_by_key):,} date_rows={len(dates_by_key):,}",
        flush=True,
    )
    total_downloaded = 0
    total_skipped = 0
    total_errors = 0

    for fund_index, fund in enumerate(funds[args.start_fund_index :], start=args.start_fund_index + 1):
        try:
            dates_html = request_text(session, FUNDS_AJAX_URL, params={"data_type": 2, "f_id": fund.fund_id})
            dates = parse_dates(fund, dates_html)
            dates = [d for d in dates if date_in_range(d.date_iso, args.start_date, args.end_date)]
            if args.sample_dates:
                dates = dates[: args.sample_dates]
            for d in dates:
                dates_by_key[(d.fund_id, d.date_iso)] = {
                    "fund_id": d.fund_id,
                    "fund_name": d.fund_name,
                    "date": d.date_text,
                    "date_iso": d.date_iso,
                }
            print(f"[{fund_index}/{len(funds)}] fund_id={fund.fund_id} dates={len(dates)} name={fund.fund_name}", flush=True)

            for date_index, date_option in enumerate(dates, start=1):
                existing_key = (fund.fund_id, date_option.date_iso)
                row: dict[str, object] = {
                    "fund_id": fund.fund_id,
                    "fund_name": fund.fund_name,
                    "date": date_option.date_text,
                    "date_iso": date_option.date_iso,
                    "download_id": "",
                    "xls_url": "",
                    "xls_path": "",
                    "existing_source_run": "",
                    "html_path": "",
                    "status": "",
                    "error": "",
                    "bytes": 0,
                }
                try:
                    existing = existing_downloads.get(existing_key)
                    if args.skip_existing and existing is not None:
                        row["status"] = "exists_global"
                        row["xls_path"] = str(existing.xls_path)
                        row["existing_source_run"] = existing.source_run
                        row["bytes"] = existing.bytes
                        total_skipped += 1
                        print(
                            f"  [{date_index}/{len(dates)}] {date_option.date_iso} status=exists_global "
                            f"source_run={existing.source_run} bytes={existing.bytes}",
                            flush=True,
                        )
                        manifest_by_key[row_key(row)] = row
                        continue

                    result_html = request_text(
                        session,
                        FUNDS_AJAX_URL,
                        params={"data_type": 3, "f_id": fund.fund_id, "date": date_option.date_text},
                    )
                    fund_slug = slugify(fund.fund_name)
                    html_path = html_dir / fund.fund_id / f"{date_option.date_iso}_{fund_slug}.html"
                    html_path.parent.mkdir(parents=True, exist_ok=True)
                    html_path.write_text(result_html, encoding="utf-8")
                    row["html_path"] = str(html_path)

                    xls_url, download_id = parse_xls_link(result_html)
                    row["download_id"] = download_id or ""
                    row["xls_url"] = xls_url or ""
                    if not xls_url:
                        row["status"] = "no_xls_link"
                        total_skipped += 1
                    else:
                        xls_path = xls_dir / fund.fund_id / f"{date_option.date_iso}_{fund_slug}_{download_id or 'noid'}.xls"
                        row["xls_path"] = str(xls_path)
                        if args.skip_existing and is_valid_xls(xls_path, min_bytes=args.min_valid_xls_bytes):
                            row["status"] = "exists"
                            row["bytes"] = xls_path.stat().st_size
                            total_skipped += 1
                        else:
                            payload = request_bytes(session, xls_url)
                            xls_path.parent.mkdir(parents=True, exist_ok=True)
                            xls_path.write_bytes(payload)
                            if not is_valid_xls(xls_path, min_bytes=args.min_valid_xls_bytes):
                                raise RuntimeError(f"downloaded file is not a valid XLS or too small: {xls_path}")
                            row["status"] = "downloaded"
                            row["bytes"] = len(payload)
                            existing_downloads[existing_key] = ExistingDownload(
                                fund.fund_id,
                                date_option.date_iso,
                                xls_path,
                                len(payload),
                                args.run_id,
                            )
                            total_downloaded += 1
                    print(
                        f"  [{date_index}/{len(dates)}] {date_option.date_iso} status={row['status']} bytes={row['bytes']}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    row["status"] = "error"
                    row["error"] = repr(exc)
                    total_errors += 1
                    print(f"  [{date_index}/{len(dates)}] {date_option.date_iso} ERROR {exc}", flush=True)
                manifest_by_key[row_key(row)] = row
                time.sleep(args.sleep_seconds)
        except Exception as exc:  # noqa: BLE001
            total_errors += 1
            print(f"[{fund_index}/{len(funds)}] fund_id={fund.fund_id} ERROR {exc}", flush=True)
        finally:
            write_csv(
                manifest_path,
                list(manifest_by_key.values()),
                MANIFEST_FIELDNAMES,
            )
            write_csv(
                dates_catalog_path,
                list(dates_by_key.values()),
                DATES_FIELDNAMES,
            )

    print(
        f"done run_dir={run_dir} downloaded={total_downloaded} skipped={total_skipped} errors={total_errors}",
        flush=True,
    )


if __name__ == "__main__":
    main()
