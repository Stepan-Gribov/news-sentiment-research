from __future__ import annotations

import argparse
import importlib.util
import os
import queue
import random
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

import _bootstrap  # noqa: F401

from dissertation_pipeline.paths import PATHS
from dissertation_pipeline.news.layout import pages_run_dir
from dissertation_pipeline.selenium_fast import configure_fast_driver, fast_chrome_args, find_chrome_executable


def ensure_parquet_engine() -> None:
    if importlib.util.find_spec("pyarrow") or importlib.util.find_spec("fastparquet"):
        return
    raise SystemExit(
        "Parquet support is missing. Install dependencies outside Google Drive, for example: "
        "python3 -m pip install -r requirements.txt"
    )


def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        status=6,
        backoff_factor=1.0,
        status_forcelist=(403, 408, 425, 429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=64, pool_maxsize=64)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
            "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        }
    )
    return session


def fetch(session: requests.Session, url: str, timeout: int = 25) -> tuple[str, str]:
    try:
        time.sleep(random.uniform(0.2, 0.8))
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        if not response.encoding:
            response.encoding = response.apparent_encoding
        return response.text, ""
    except Exception as exc:
        return "", str(exc)


def wait_for_port(port: int, host: str = "127.0.0.1", timeout: float = 20.0) -> None:
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
            "--window-size=1440,1600",
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
            "/exhkqyad",
            "/xpvnsulc",
        ]
    )


def fetch_selenium(driver, url: str, retries: int = 12, wait_sec: float = 2.0) -> tuple[str, str]:
    from selenium.common.exceptions import TimeoutException, WebDriverException

    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            driver.set_page_load_timeout(30)
            driver.set_script_timeout(30)
            driver.get(url)
            time.sleep(wait_sec)
            html = driver.page_source or ""
            if is_blocked_html(html, driver.title, driver.current_url):
                raise RuntimeError("servicepipe_challenge")
            return html, ""
        except (RuntimeError, TimeoutException, WebDriverException) as exc:
            last_error = str(exc)
            sleep_s = min(30.0, attempt * 2.0 + random.uniform(0.2, 1.0))
            print(f"Selenium retry {attempt}/{retries}: {last_error} -> sleeping {sleep_s:.1f}s")
            time.sleep(sleep_s)
    return "", last_error


def expected_rows(csv_path: Path, sample_size: int | None) -> int:
    try:
        rows = len(pd.read_csv(csv_path, encoding="utf-16", sep="\t", usecols=[0]))
    except Exception:
        rows = len(pd.read_csv(csv_path, encoding="utf-16", sep="\t"))
    return min(rows, sample_size) if sample_size else rows


def parquet_is_complete(path: Path, expected_row_count: int, min_success_rate: float) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        df = pd.read_parquet(path, columns=["page_source", "page_source_error"])
    except Exception:
        return False
    if len(df) != expected_row_count:
        return False
    if expected_row_count == 0:
        return True
    success_count = df["page_source"].fillna("").astype(str).str.len().gt(0).sum()
    success_rate = success_count / expected_row_count
    return success_rate >= min_success_rate


def atomic_write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, out_path)


def load_partial_if_compatible(df: pd.DataFrame, partial_path: Path, link_col: str) -> pd.DataFrame:
    if not partial_path.exists() or partial_path.stat().st_size == 0:
        return df
    try:
        partial = pd.read_parquet(partial_path)
    except Exception as exc:
        print(f"{partial_path.name}: unreadable partial ignored: {exc}", flush=True)
        return df
    if len(partial) != len(df) or link_col not in partial.columns:
        print(f"{partial_path.name}: incompatible partial ignored", flush=True)
        return df
    if not partial[link_col].astype(str).equals(df[link_col].astype(str)):
        print(f"{partial_path.name}: link order changed, partial ignored", flush=True)
        return df
    for col in ["page_source", "page_source_error"]:
        if col in partial.columns:
            df[col] = partial[col].fillna("").astype(str)
    restored = df["page_source"].fillna("").astype(str).str.len().gt(0).sum()
    print(f"{partial_path.name}: restored {restored:,} successful pages from partial checkpoint", flush=True)
    return df


def validate_download_result(df: pd.DataFrame, csv_path: Path, min_success_rate: float) -> None:
    if len(df) == 0:
        return
    success_count = df["page_source"].fillna("").astype(str).str.len().gt(0).sum()
    success_rate = success_count / len(df)
    error_count = df["page_source_error"].fillna("").astype(str).str.len().gt(0).sum()
    print(
        f"{csv_path.stem}: pages={len(df):,} success={success_count:,} "
        f"errors={error_count:,} success_rate={success_rate:.3f}",
        flush=True,
    )
    if success_rate < min_success_rate:
        raise RuntimeError(
            f"{csv_path}: success_rate={success_rate:.3f} below min_success_rate={min_success_rate:.3f}. "
            "Not writing final parquet; retry later."
        )


def pending_items(df: pd.DataFrame, link_col: str) -> list[tuple[int, object]]:
    return [
        (idx, url)
        for idx, url in df[link_col].items()
        if not str(df.at[idx, "page_source"] or "").strip()
    ]


def download_pending_round(
    df: pd.DataFrame,
    pending: list[tuple[int, object]],
    csv_path: Path,
    partial_path: Path,
    link_col: str,
    max_workers: int,
    checkpoint_every: int,
    round_number: int,
) -> tuple[int, int]:
    session = build_session()
    completed_since_checkpoint = 0
    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch, session, str(url).strip()): idx for idx, url in pending}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{csv_path.stem} r{round_number}"):
            idx = futures[future]
            html, error = future.result()
            df.at[idx, "page_source"] = html
            df.at[idx, "page_source_error"] = error
            if str(html or "").strip():
                success_count += 1
            completed_since_checkpoint += 1
            if checkpoint_every > 0 and completed_since_checkpoint >= checkpoint_every:
                atomic_write_parquet(df, partial_path)
                completed_since_checkpoint = 0
        if checkpoint_every > 0 and pending:
            atomic_write_parquet(df, partial_path)
    return len(pending), success_count


def process_file(
    csv_path: Path,
    out_dir: Path,
    link_col: str,
    max_workers: int,
    sample_size: int | None,
    min_success_rate: float,
    checkpoint_every: int,
    retry_failed_rounds: int,
    retry_failed_forever: bool,
    retry_failed_sleep_seconds: int,
) -> None:
    df = pd.read_csv(csv_path, encoding="utf-16", sep="\t")
    if sample_size:
        df = df.head(sample_size).copy()
    if link_col not in df.columns:
        raise ValueError(f"missing link column {link_col!r} in {csv_path}")

    df["page_source"] = ""
    df["page_source_error"] = ""
    out_path = out_dir / f"{csv_path.stem}.parquet"
    partial_path = out_dir / f".{csv_path.stem}.partial.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_partial_if_compatible(df, partial_path, link_col)

    round_number = 1
    while True:
        pending = pending_items(df, link_col)
        if not pending:
            print(f"{csv_path.stem}: all pages downloaded/restored", flush=True)
            break

        max_rounds = None if retry_failed_forever else 1 + max(0, retry_failed_rounds)
        if max_rounds is not None and round_number > max_rounds:
            print(
                f"{csv_path.stem}: retry rounds exhausted; remaining_empty={len(pending):,}",
                flush=True,
            )
            break

        print(
            f"{csv_path.stem}: round={round_number} pending={len(pending):,} "
            f"workers={max_workers}",
            flush=True,
        )
        attempted, round_success = download_pending_round(
            df=df,
            pending=pending,
            csv_path=csv_path,
            partial_path=partial_path,
            link_col=link_col,
            max_workers=max_workers,
            checkpoint_every=checkpoint_every,
            round_number=round_number,
        )
        print(
            f"{csv_path.stem}: round={round_number} attempted={attempted:,} "
            f"success={round_success:,}",
            flush=True,
        )

        remaining = pending_items(df, link_col)
        if not remaining:
            break
        if not retry_failed_forever and round_number >= 1 + max(0, retry_failed_rounds):
            round_number += 1
            continue
        if retry_failed_sleep_seconds > 0:
            print(
                f"{csv_path.stem}: sleeping {retry_failed_sleep_seconds}s before retrying "
                f"{len(remaining):,} failed/empty pages",
                flush=True,
            )
            time.sleep(retry_failed_sleep_seconds)
        round_number += 1

    validate_download_result(df, csv_path, min_success_rate)
    atomic_write_parquet(df, out_path)
    if partial_path.exists():
        partial_path.unlink()


def process_file_selenium(
    csv_path: Path,
    out_dir: Path,
    link_col: str,
    sample_size: int | None,
    drivers,
    min_success_rate: float,
    checkpoint_every: int,
    retry_failed_rounds: int,
    retry_failed_forever: bool,
    retry_failed_sleep_seconds: int,
) -> None:
    df = pd.read_csv(csv_path, encoding="utf-16", sep="\t")
    if sample_size:
        df = df.head(sample_size).copy()
    if link_col not in df.columns:
        raise ValueError(f"missing link column {link_col!r} in {csv_path}")

    df["page_source"] = ""
    df["page_source_error"] = ""
    out_path = out_dir / f"{csv_path.stem}.parquet"
    partial_path = out_dir / f".{csv_path.stem}.partial.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_partial_if_compatible(df, partial_path, link_col)

    round_number = 1
    while True:
        pending = pending_items(df, link_col)
        if not pending:
            print(f"{csv_path.stem}: all pages downloaded/restored", flush=True)
            break

        max_rounds = None if retry_failed_forever else 1 + max(0, retry_failed_rounds)
        if max_rounds is not None and round_number > max_rounds:
            print(
                f"{csv_path.stem}: selenium retry rounds exhausted; remaining_empty={len(pending):,}",
                flush=True,
            )
            break

        work_queue: queue.Queue[tuple[int, object]] = queue.Queue()
        for item in pending:
            work_queue.put(item)

        state = {"completed_since_checkpoint": 0, "round_success": 0}
        state_lock = threading.Lock()
        progress = tqdm(total=len(pending), desc=f"{csv_path.stem} r{round_number}")

        def worker(worker_id: int, driver) -> None:
            while True:
                try:
                    idx, url = work_queue.get_nowait()
                except queue.Empty:
                    return
                html, error = fetch_selenium(driver, str(url).strip())
                with state_lock:
                    df.at[idx, "page_source"] = html
                    df.at[idx, "page_source_error"] = error
                    if str(html or "").strip():
                        state["round_success"] += 1
                    state["completed_since_checkpoint"] += 1
                    progress.update(1)
                    if checkpoint_every > 0 and state["completed_since_checkpoint"] >= checkpoint_every:
                        atomic_write_parquet(df, partial_path)
                        state["completed_since_checkpoint"] = 0
                work_queue.task_done()

        try:
            with ThreadPoolExecutor(max_workers=len(drivers)) as executor:
                futures = [executor.submit(worker, worker_id, driver) for worker_id, driver in enumerate(drivers)]
                for future in as_completed(futures):
                    future.result()
        finally:
            progress.close()

        if checkpoint_every > 0:
            with state_lock:
                atomic_write_parquet(df, partial_path)

        print(
            f"{csv_path.stem}: selenium round={round_number} attempted={len(pending):,} "
            f"success={state['round_success']:,}",
            flush=True,
        )
        remaining = pending_items(df, link_col)
        if not remaining:
            break
        if not retry_failed_forever and round_number >= 1 + max(0, retry_failed_rounds):
            round_number += 1
            continue
        if retry_failed_sleep_seconds > 0:
            print(
                f"{csv_path.stem}: sleeping {retry_failed_sleep_seconds}s before selenium retry of "
                f"{len(remaining):,} failed/empty pages",
                flush=True,
            )
            time.sleep(retry_failed_sleep_seconds)
        round_number += 1

    validate_download_result(df, csv_path, min_success_rate)
    atomic_write_parquet(df, out_path)
    if partial_path.exists():
        partial_path.unlink()


def csv_has_rows(path: Path) -> bool:
    try:
        return len(pd.read_csv(path, encoding="utf-16", sep="\t", nrows=1)) > 0
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download page sources from link CSV files.")
    parser.add_argument("--media", required=True, choices=["rbc", "interfax", "kommersant", "prime", "frank"])
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--link-col", default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--sample-files", type=int, default=None)
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--nonempty-only", action="store_true")
    parser.add_argument("--allow-no-files", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-download even if a valid parquet already exists.")
    parser.add_argument(
        "--fast-skip-existing",
        action="store_true",
        help="Skip existing non-empty parquet files without reading them. Useful for large restart runs.",
    )
    parser.add_argument(
        "--min-success-rate",
        type=float,
        default=0.01,
        help="Do not mark a CSV as completed if successful page downloads are below this share.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=200,
        help="Save partial parquet after this many attempted pages inside a CSV. Set 0 to disable.",
    )
    parser.add_argument(
        "--retry-failed-rounds",
        type=int,
        default=3,
        help="After the first multithreaded pass, retry empty/failed pages this many extra rounds.",
    )
    parser.add_argument(
        "--retry-failed-forever",
        action="store_true",
        help="Keep retrying empty/failed pages inside each CSV until all are downloaded or the process is stopped.",
    )
    parser.add_argument(
        "--retry-failed-sleep-seconds",
        type=int,
        default=45,
        help="Pause between retry rounds for pages that failed or returned empty HTML.",
    )
    parser.add_argument("--no-selenium", action="store_true", help="Disable Selenium fallback for media that need it.")
    parser.add_argument("--selenium-workers", type=int, default=1, help="Number of parallel Selenium browser workers.")
    parser.add_argument("--remote-port", type=int, default=9240)
    parser.add_argument("--chrome-profile-dir", type=Path, default=PATHS.tmp_dir / "page_downloader_chrome_profile")
    parser.add_argument("--keep-browser", action="store_true")
    args = parser.parse_args()

    ensure_parquet_engine()

    link_cols = {"rbc": "fronturl", "interfax": "link", "kommersant": "url", "prime": "link", "frank": "link"}
    if args.run_id:
        input_dir = args.input_dir or (PATHS.news_links_runs_dir / args.run_id / args.media)
        out_dir = args.out_dir or pages_run_dir(args.media, args.run_id)
    else:
        input_dir = args.input_dir or (PATHS.news_links_history_dir / args.media)
        out_dir = args.out_dir or (PATHS.news_pages_history_dir / args.media)
    link_col = args.link_col or link_cols[args.media]

    # Collection-status and other underscore-prefixed files are metadata, not
    # article indexes.  They may use a different encoding and must never be
    # handed to the page downloader.
    files = sorted(path for path in input_dir.glob("*.csv") if not path.name.startswith("_"))
    if args.nonempty_only:
        files = [path for path in files if csv_has_rows(path)]
    if args.sample_files:
        files = files[: args.sample_files]
    if not files:
        if args.allow_no_files:
            print(f"No CSV files found in {input_dir}; skipped")
            return
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    use_selenium = args.media == "frank" and not args.no_selenium
    chrome_procs = []
    drivers = []
    try:
        if use_selenium:
            selenium_workers = max(1, args.selenium_workers)
            print(f"{args.media}: starting {selenium_workers} Selenium workers", flush=True)
            for worker_id in range(selenium_workers):
                port = args.remote_port + worker_id
                profile_dir = args.chrome_profile_dir / f"worker_{worker_id}"
                chrome_proc = start_remote_chrome(port, profile_dir)
                driver = attach_remote_driver(port)
                chrome_procs.append(chrome_proc)
                drivers.append(driver)
        for file_path in files:
            out_path = out_dir / f"{file_path.stem}.parquet"
            if not args.force and args.fast_skip_existing and out_path.exists() and out_path.stat().st_size > 0:
                print(f"{args.media}: {file_path.stem} [fast skip existing parquet]", flush=True)
                continue
            expected_row_count = expected_rows(file_path, args.sample_rows)
            if not args.force and parquet_is_complete(out_path, expected_row_count, args.min_success_rate):
                print(f"{args.media}: {file_path.stem} [skip existing parquet]", flush=True)
                continue
            if use_selenium:
                process_file_selenium(
                    file_path,
                    out_dir,
                    link_col,
                    args.sample_rows,
                    drivers,
                    args.min_success_rate,
                    args.checkpoint_every,
                    args.retry_failed_rounds,
                    args.retry_failed_forever,
                    args.retry_failed_sleep_seconds,
                )
            else:
                process_file(
                    file_path,
                    out_dir,
                    link_col,
                    args.max_workers,
                    args.sample_rows,
                    args.min_success_rate,
                    args.checkpoint_every,
                    args.retry_failed_rounds,
                    args.retry_failed_forever,
                    args.retry_failed_sleep_seconds,
                )
    finally:
        for driver in drivers:
            try:
                driver.quit()
            except Exception:
                pass
        if not args.keep_browser:
            for chrome_proc in chrome_procs:
                chrome_proc.terminate()


if __name__ == "__main__":
    main()
