from __future__ import annotations

import argparse
import socket
import subprocess
import random
import re
import time
import unicodedata
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
from pandas.errors import EmptyDataError
from bs4 import BeautifulSoup
from tqdm import tqdm

import _bootstrap  # noqa: F401

from dissertation_pipeline.paths import PATHS
from dissertation_pipeline.news.layout import links_run_dir
from dissertation_pipeline.news.resume import atomic_write_csv, clear_inprogress, mark_inprogress, should_skip_output
from dissertation_pipeline.selenium_fast import (
    apply_fast_chrome_options,
    configure_fast_driver,
    fast_chrome_args,
    find_chrome_executable,
)


OLD_FRANK_LINKS_DIR = PATHS.project_root.parent.parent / "data" / "news_links" / "links_frank"


class FrankBlockedError(RuntimeError):
    pass


def load_names(path: Path, sample_size: int | None) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen = set()
    names = [name for name in names if not (name in seen or seen.add(name))]
    return names[:sample_size] if sample_size else names


def build_driver():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    options = Options()
    apply_fast_chrome_options(options)
    options.add_argument("--window-size=1240,720")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(options=options)
    configure_fast_driver(driver)
    driver.set_page_load_timeout(20)
    try:
        driver.minimize_window()
    except Exception:
        pass
    return driver


def wait_for_port(port: int, host: str = "127.0.0.1", timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError(f"Chrome remote debugging port did not open: {host}:{port}")


def start_remote_chrome(port: int, profile_dir: Path) -> subprocess.Popen:
    chrome_path = find_chrome_executable()
    profile_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [
            str(chrome_path),
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "--window-size=1240,720",
            *fast_chrome_args(),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    wait_for_port(port)
    return proc


def attach_remote_driver(port: int):
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    options = Options()
    options.add_experimental_option("debuggerAddress", f"127.0.0.1:{port}")
    driver = webdriver.Chrome(options=options)
    configure_fast_driver(driver)
    return driver


def is_blocked_html(html: str, title: str = "", current_url: str = "") -> bool:
    low = " ".join([html or "", title or "", current_url or ""]).lower()
    return any(
        marker in low
        for marker in [
            "servicepipe",
            "id_spinner",
            "id_captcha_frame_div",
            "checkjs",
            "доступ запрещен",
            "frank media проверка",
            "/xpvnsulc/",
        ]
    )


def fetch_page_source_selenium(
    driver,
    url: str,
    wait_sec: float = 1.5,
    page_timeout: int = 20,
    verbose: bool = False,
) -> str:
    driver.set_page_load_timeout(page_timeout)
    driver.set_script_timeout(page_timeout)
    driver.get(url)
    time.sleep(wait_sec)
    html = driver.page_source or ""
    if verbose:
        print(
            "Frank Selenium:",
            f"title={driver.title!r}",
            f"url={driver.current_url[:160]!r}",
            f"html_len={len(html)}",
            flush=True,
        )
    if is_blocked_html(html, driver.title, driver.current_url):
        raise FrankBlockedError("Frank Media returned ServicePipe/403 challenge page")
    return html


def fetch_with_retries(driver, url: str, max_tries: int = 60, base_sleep: float = 2.0, verbose: bool = False) -> str:
    from selenium.common.exceptions import TimeoutException, WebDriverException

    last_error = None
    for attempt in range(1, max_tries + 1):
        try:
            return fetch_page_source_selenium(driver, url, verbose=verbose)
        except (FrankBlockedError, RuntimeError, TimeoutException, WebDriverException) as exc:
            last_error = exc
            msg = str(exc).lower()
            retryable = (
                isinstance(exc, FrankBlockedError)
                or "blocked" in msg
                or "challenge" in msg
                or "timeout" in msg
                or "disconnected" in msg
                or "unknown error" in msg
            )
            if not retryable:
                raise
            sleep_s = min(90.0, base_sleep * attempt + random.uniform(0.2, 1.0))
            print(f"Retry {attempt}/{max_tries} (blocked/challenge) -> sleeping {sleep_s:.1f}s : {url}")
            time.sleep(sleep_s)
    raise last_error


def max_page(html: str) -> int:
    soup = BeautifulSoup(html, "html.parser")
    nums = set()
    for a in soup.select("nav.pagination a.page, nav.pagination a[href]"):
        text = a.get_text(strip=True)
        if text.isdigit():
            nums.add(int(text))
        match = re.search(r"(?:\?|&)p=(\d+)", a.get("href", ""))
        if match:
            nums.add(int(match.group(1)))
    return max(nums) if nums else 1


def extract_news_items(search_query: str, html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for section in soup.find_all("section", class_=re.compile("item")):
        time_tag = section.find("time", class_=re.compile("date"))
        link_tag = section.find("a", class_=re.compile("link"))
        if not link_tag:
            continue
        title_tag = link_tag.find("h2", class_=re.compile("title"))
        rows.append(
            {
                "search_query": search_query,
                "link": link_tag.get("href", "").strip(),
                "header": title_tag.get_text(strip=True) if title_tag else "",
                "date": (time_tag.get("datetime", "").split("T")[0] if time_tag else ""),
            }
        )
    return pd.DataFrame(rows)


def collect_one_query(driver, search_query: str, date_from: str, date_to: str, verbose: bool = False) -> pd.DataFrame:
    parts = []
    query_params = {"s": search_query, "from": date_from, "to": date_to}
    first_url = "https://frankmedia.ru/search?" + urlencode({**query_params, "p": 1})
    first_html = fetch_with_retries(driver, first_url, verbose=verbose)
    for page in range(1, max_page(first_html) + 1):
        html = first_html if page == 1 else fetch_with_retries(driver, "https://frankmedia.ru/search?" + urlencode({**query_params, "p": page}), verbose=verbose)
        df = extract_news_items(search_query, html)
        if not df.empty:
            parts.append(df)
    if not parts:
        return pd.DataFrame(columns=["search_query", "link", "header", "date"])
    return pd.concat(parts, ignore_index=True).drop_duplicates(subset=["link"])


def month_windows(date_from: str, date_to: str) -> list[tuple[str, str, str]]:
    """Return inclusive calendar-month windows labelled YYYY-MM."""
    start = date.fromisoformat(date_from)
    end = date.fromisoformat(date_to)
    if start > end:
        raise ValueError("date-from must not be after date-to")
    windows: list[tuple[str, str, str]] = []
    cursor = start
    while cursor <= end:
        next_month = date(cursor.year + (cursor.month == 12), 1 if cursor.month == 12 else cursor.month + 1, 1)
        window_end = min(end, next_month - timedelta(days=1))
        windows.append((f"{cursor.year:04d}-{cursor.month:02d}", cursor.isoformat(), window_end.isoformat()))
        cursor = next_month
    return windows


def collect_all_publications(driver, out_dir: Path, date_from: str, date_to: str, *, force: bool, verbose: bool) -> None:
    """Collect every result returned by Frank's date-filtered site search.

    The empty search query is deliberate: it enumerates site publications rather
    than publications mentioning a company.  One file per month makes restarts
    and the subsequent page-download stage resumable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    statuses = []
    for label, window_start, window_end in month_windows(date_from, date_to):
        out_path = out_dir / f"{label}.csv"
        if should_skip_output(out_path, force=force):
            statuses.append({"window": label, "status": "skipped_existing", "rows": None})
            continue
        mark_inprogress(out_path)
        try:
            frame = collect_one_query(driver, "", window_start, window_end, verbose=verbose)
            if not frame.empty:
                frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype("string")
                frame = frame.sort_values(["date", "link"], ascending=[False, True])
            atomic_write_csv(frame, out_path)
            clear_inprogress(out_path)
            statuses.append({"window": label, "status": "ok", "rows": len(frame)})
            print(f"Frank full site {label}: {len(frame):,} publications", flush=True)
        except Exception as exc:
            statuses.append({"window": label, "status": "failed", "rows": None, "error": str(exc)})
            print(f"Frank full site failed for {label}: {exc}", flush=True)
            break
    pd.DataFrame(statuses).to_csv(out_dir / "_collection_status.csv", index=False)


def existing_output_has_rows(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        return len(pd.read_csv(path, encoding="utf-16", sep="\t", nrows=2)) > 0
    except Exception:
        return False


def normalize_restored_links(df: pd.DataFrame, search_query: str, date_from: str | None, date_to: str | None) -> pd.DataFrame:
    result = df.copy()
    if "search_query" not in result.columns:
        result.insert(0, "search_query", search_query)
    result = result.reindex(columns=["search_query", "link", "header", "date"])
    result["search_query"] = result["search_query"].fillna(search_query)
    parsed_dates = pd.to_datetime(result["date"], errors="coerce")
    if date_from:
        result = result[parsed_dates >= pd.Timestamp(date_from)]
        parsed_dates = pd.to_datetime(result["date"], errors="coerce")
    if date_to:
        result = result[parsed_dates <= pd.Timestamp(date_to)]
    return result.drop_duplicates(subset=["link"]).reset_index(drop=True)


def build_normalized_file_index(source_dir: Path) -> dict[str, Path]:
    return {unicodedata.normalize("NFC", path.stem).casefold(): path for path in source_dir.glob("*.csv")}


def restore_log_paths(out_dir: Path) -> tuple[Path, Path]:
    default_out_dir = PATHS.news_links_dir / "frank"
    if out_dir.resolve() == default_out_dir.resolve():
        PATHS.logs_dir.mkdir(parents=True, exist_ok=True)
        return PATHS.logs_dir / "frank_restore_status.csv", PATHS.logs_dir / "frank_restore_missing.txt"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "_restore_status.csv", out_dir / "_restore_missing.txt"


def restore_from_existing(source_dir: Path, out_dir: Path, names: list[str], date_from: str, date_to: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path, missing_path = restore_log_paths(out_dir)
    source_index = build_normalized_file_index(source_dir)
    restored = []
    missing = []
    for name in tqdm(names, desc="Restore Frank links"):
        src = source_index.get(unicodedata.normalize("NFC", name).casefold())
        dst = out_dir / f"{name}.csv"
        if not src or not src.exists():
            missing.append(name)
            continue
        try:
            df = pd.read_csv(src, encoding="utf-16", sep="\t")
        except EmptyDataError:
            df = pd.DataFrame(columns=["link", "header", "date"])
        df = normalize_restored_links(df, name, date_from, date_to)
        df.to_csv(dst, index=False, encoding="utf-16", sep="\t")
        restored.append({"name": name, "rows": len(df), "source": str(src), "output": str(dst)})

    status = pd.DataFrame(restored)
    status.to_csv(status_path, index=False)
    if missing:
        missing_path.write_text("\n".join(missing) + "\n", encoding="utf-8")
    else:
        if missing_path.exists():
            missing_path.unlink()
    print(f"Restored Frank files: {len(restored)}; missing: {len(missing)}")
    print(f"Status: {status_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Frank Media article links by company names.")
    parser.add_argument("--names-file", type=Path, default=PATHS.company_names_file)
    parser.add_argument("--date-from", default="2026-01-01")
    parser.add_argument("--date-to", default="2026-04-01")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument(
        "--all-publications",
        action="store_true",
        help="Enumerate all Frank Media search results by month, without company-name queries.",
    )
    parser.add_argument("--restore-from-existing", type=Path, nargs="?", const=OLD_FRANK_LINKS_DIR, default=None)
    parser.add_argument(
        "--mode",
        choices=["restore", "live"],
        default="live",
        help="Use live Selenium parsing, or restore already collected historical links.",
    )
    parser.add_argument("--preserve-existing-nonempty", action="store_true", default=True)
    parser.add_argument("--browser-mode", choices=["remote", "webdriver"], default="remote")
    parser.add_argument("--remote-port", type=int, default=9223)
    parser.add_argument("--chrome-profile-dir", type=Path, default=PATHS.tmp_dir / "frank_remote_chrome_profile")
    parser.add_argument("--keep-browser", action="store_true")
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument("--force", action="store_true", help="Alias for --replace-existing.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.force:
        args.replace_existing = True
    if args.out_dir is None:
        if args.mode == "restore":
            args.out_dir = PATHS.news_links_history_dir / "frank"
        else:
            media_dir = "frank_all_publications" if args.all_publications else "frank"
            args.out_dir = links_run_dir(media_dir, args.date_from, args.date_to)

    if args.all_publications and args.mode != "live":
        raise SystemExit("--all-publications is available only in live mode")
    names = load_names(args.names_file, args.sample_size)
    if args.restore_from_existing or args.mode == "restore":
        source_dir = args.restore_from_existing or OLD_FRANK_LINKS_DIR
        restore_from_existing(source_dir, args.out_dir, names, args.date_from, args.date_to)
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.all_publications:
        chrome_proc = None
        driver = None
        try:
            if args.browser_mode == "remote":
                chrome_proc = start_remote_chrome(args.remote_port, args.chrome_profile_dir)
                driver = attach_remote_driver(args.remote_port)
            else:
                driver = build_driver()
            collect_all_publications(
                driver,
                args.out_dir,
                args.date_from,
                args.date_to,
                force=args.replace_existing,
                verbose=args.verbose,
            )
        finally:
            if driver is not None:
                try:
                    driver.quit()
                except Exception:
                    pass
            if chrome_proc is not None and not args.keep_browser:
                chrome_proc.terminate()
        return

    statuses = []
    pending_names = []
    for name in names:
        out_path = args.out_dir / f"{name}.csv"
        if should_skip_output(out_path, force=args.replace_existing):
            statuses.append({"name": name, "status": "skipped_existing", "rows": None})
            continue
        pending_names.append(name)
    if not pending_names:
        print("Frank links: all outputs already exist; nothing to do.", flush=True)
        PATHS.logs_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(statuses).to_csv(PATHS.logs_dir / "frank_live_collection_status.csv", index=False)
        return

    chrome_proc = None
    driver = None
    try:
        if args.browser_mode == "remote":
            chrome_proc = start_remote_chrome(args.remote_port, args.chrome_profile_dir)
            driver = attach_remote_driver(args.remote_port)
        else:
            driver = build_driver()

        for name in tqdm(pending_names, desc="Frank companies"):
            out_path = args.out_dir / f"{name}.csv"
            try:
                mark_inprogress(out_path)
                df = collect_one_query(driver, name, args.date_from, args.date_to, verbose=args.verbose)
                if df.empty and args.preserve_existing_nonempty and existing_output_has_rows(out_path):
                    statuses.append({"name": name, "status": "empty_live_preserved_existing", "rows": None})
                    clear_inprogress(out_path)
                    continue
                new_rows = len(df)
                if out_path.exists() and not args.replace_existing:
                    try:
                        existing = pd.read_csv(out_path, encoding="utf-16", sep="\t")
                        existing = existing.reindex(columns=["search_query", "link", "header", "date"])
                        df = pd.concat([existing, df], ignore_index=True)
                        df = df.drop_duplicates(subset=["link"], keep="first").reset_index(drop=True)
                    except EmptyDataError:
                        pass
                if "date" in df.columns:
                    df["_sort_date"] = pd.to_datetime(df["date"], errors="coerce")
                    df = df.sort_values("_sort_date", ascending=False).drop(columns=["_sort_date"])
                atomic_write_csv(df, out_path)
                clear_inprogress(out_path)
                statuses.append({"name": name, "status": "ok", "new_rows": new_rows, "output_rows": len(df)})
            except Exception as exc:
                statuses.append({"name": name, "status": "failed", "rows": None, "error": str(exc)})
                print(f"Frank failed for {name}: {exc}")
                break
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass
        if chrome_proc is not None and not args.keep_browser:
            chrome_proc.terminate()

    if statuses:
        PATHS.logs_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(statuses).to_csv(PATHS.logs_dir / "frank_live_collection_status.csv", index=False)


if __name__ == "__main__":
    main()
