import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from dissertation_pipeline.selenium_fast import apply_fast_chrome_options, configure_fast_driver


_thread_local = threading.local()


def get_with_retry(session, url, retries=3, timeout=10, backoff=1.5):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            time.sleep(random.uniform(0.8, 1.8))
            response = session.get(url, timeout=timeout)
            if response.status_code in (403, 429):
                raise requests.HTTPError(f"blocked_status_{response.status_code}", response=response)
            response.raise_for_status()
            return response
        except Exception as e:
            last_err = e
            if isinstance(e, requests.HTTPError) and "blocked_status" in str(e):
                raise
            sleep_s = min(30, backoff ** attempt + random.uniform(0.2, 1.0))
            print(f"Retry {attempt}/{retries}: {url}")
            time.sleep(sleep_s)
    raise last_err


def build_rbc_url(query, date_from, date_to, page):
    params = {
        "query": query,
        "dateFrom": date_from,
        "dateTo": date_to,
        "page": page,
    }
    return "https://www.rbc.ru/search/ajax/?" + urlencode(params)


def build_rbc_search_url(query):
    return "https://www.rbc.ru/search/?" + urlencode({"query": query})


def parse_rbc_items(items, search_query, window_from, window_to):
    rows = []

    for item in items:
        rows.append(
            {
                "search_query": search_query,
                "window_from": window_from,
                "window_to": window_to,
                "publish_date": item.get("publish_date"),
                "project": item.get("project"),
                "category": item.get("category"),
                "title": item.get("title"),
                "body": item.get("body"),
                "fronturl": item.get("fronturl"),
            }
        )

    return pd.DataFrame(rows)


def month_windows(date_from_str, date_to_str):
    date_format = "%Y-%m-%d" if "-" in date_from_str else "%d.%m.%Y"
    start_date = datetime.strptime(date_from_str, date_format)
    end_date = datetime.strptime(date_to_str, date_format)

    windows = []
    current = start_date

    while current < end_date:
        next_month = min(current + relativedelta(months=1), end_date)
        windows.append(
            (
                current.strftime("%d.%m.%Y"),
                next_month.strftime("%d.%m.%Y"),
            )
        )
        current = next_month

    return windows


def build_chrome_options():
    from selenium.webdriver.chrome.options import Options

    options = Options()
    apply_fast_chrome_options(options)
    options.add_argument("--window-size=1440,1600")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
    return options


def build_driver():
    from selenium import webdriver

    driver = webdriver.Chrome(options=build_chrome_options())
    configure_fast_driver(driver)
    return driver


def attach_remote_driver(port):
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    options = Options()
    options.add_experimental_option("debuggerAddress", f"127.0.0.1:{port}")
    driver = webdriver.Chrome(options=options)
    configure_fast_driver(driver)
    return driver


def warmup_rbc(driver, query, wait_sec=2.0):
    driver.set_page_load_timeout(30)
    driver.set_script_timeout(30)
    driver.get(build_rbc_search_url(query))
    time.sleep(wait_sec)


def parse_json_from_page_source(page_source):
    text = str(page_source or "").strip()
    if not text:
        raise ValueError("empty_rbc_response")
    if text.startswith("{"):
        return json.loads(text)
    # Chrome wraps application/json in <pre> for normal navigation.
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(text, "html.parser")
        pre = soup.find("pre")
        if pre:
            return json.loads(pre.get_text())
        body = soup.find("body")
        if body:
            body_text = body.get_text(strip=True)
            if body_text.startswith("{"):
                return json.loads(body_text)
    except Exception:
        pass
    raise ValueError("rbc_response_is_not_json")


def browser_fetch_json(driver, url, retries=5, base_sleep=1.5, verbose=False):
    from selenium.common.exceptions import TimeoutException, WebDriverException

    script = """
        const url = arguments[0];
        const done = arguments[arguments.length - 1];
        fetch(url, {
            credentials: "include",
            headers: {
                "Accept": "application/json, text/plain, */*",
                "X-Requested-With": "XMLHttpRequest"
            }
        })
        .then(async response => {
            const text = await response.text();
            done({ok: response.ok, status: response.status, text});
        })
        .catch(error => done({ok: false, status: 0, text: String(error)}));
    """
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            result = driver.execute_async_script(script, url)
            status = int(result.get("status") or 0)
            text = result.get("text") or ""
            if verbose:
                print(f"RBC browser fetch: status={status}, len={len(text)}, url={url}")
            if status in (403, 429):
                raise RuntimeError(f"blocked_status_{status}")
            if not result.get("ok"):
                raise RuntimeError(f"browser_fetch_failed_status_{status}: {text[:200]}")
            return json.loads(text)
        except (json.JSONDecodeError, RuntimeError, TimeoutException, WebDriverException) as exc:
            last_error = exc
            if attempt == 1:
                try:
                    driver.get(url)
                    time.sleep(1.0)
                    return parse_json_from_page_source(driver.page_source)
                except Exception as nav_exc:
                    last_error = nav_exc
            sleep_s = min(30.0, base_sleep * attempt + random.uniform(0.2, 1.0))
            print(f"RBC retry {attempt}/{retries}: {last_error} -> sleeping {sleep_s:.1f}s")
            time.sleep(sleep_s)
    raise last_error


def collect_rbc_one_window(session, query, window_from, window_to):
    all_parts = []
    page = 1

    while True:
        url = build_rbc_url(query, window_from, window_to, page)
        response = get_with_retry(session, url)
        data = response.json()

        items = data.get("items", [])
        more_exists = data.get("moreExists", False)

        if not items:
            break

        df_page = parse_rbc_items(items, query, window_from, window_to)
        if not df_page.empty:
            all_parts.append(df_page)

        print(
            f"{query}: {window_from} - {window_to}, "
            f"page {page}, items {len(items)}, moreExists={more_exists}"
        )

        if not more_exists:
            break

        if page >= 99:
            print(f"WARNING: page limit reached for {query} in {window_from} - {window_to}")
            break

        page += 1
        time.sleep(random.uniform(0.2, 0.7))

    if all_parts:
        return pd.concat(all_parts, ignore_index=True)

    return pd.DataFrame(
        columns=[
            "search_query",
            "window_from",
            "window_to",
            "publish_date",
            "project",
            "category",
            "title",
            "body",
            "fronturl",
        ]
    )


def collect_rbc_by_name(query, date_from, date_to):
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.rbc.ru/search/",
            "X-Requested-With": "XMLHttpRequest",
        }
    )

    windows = month_windows(date_from, date_to)
    all_parts = []

    for window_from, window_to in windows:
        try:
            df_window = collect_rbc_one_window(session, query, window_from, window_to)
            if not df_window.empty:
                all_parts.append(df_window)
        except Exception as e:
            print(f"FAILED window {query} {window_from} - {window_to}: {e}")

    if all_parts:
        result = pd.concat(all_parts, ignore_index=True)
        result.drop_duplicates(subset=["fronturl"], inplace=True, ignore_index=True)
    else:
        result = pd.DataFrame(
            columns=[
                "search_query",
                "window_from",
                "window_to",
                "publish_date",
                "project",
                "category",
                "title",
                "body",
                "fronturl",
            ]
        )

    return result


def collect_rbc_one_window_selenium(driver, query, window_from, window_to, verbose=False):
    all_parts = []
    page = 1

    while True:
        url = build_rbc_url(query, window_from, window_to, page)
        data = browser_fetch_json(driver, url, verbose=verbose)

        items = data.get("items", [])
        more_exists = data.get("moreExists", False)

        if not items:
            break

        df_page = parse_rbc_items(items, query, window_from, window_to)
        if not df_page.empty:
            all_parts.append(df_page)

        print(
            f"{query}: {window_from} - {window_to}, "
            f"page {page}, items {len(items)}, moreExists={more_exists}",
            flush=True,
        )

        if not more_exists:
            break

        if page >= 99:
            print(f"WARNING: page limit reached for {query} in {window_from} - {window_to}")
            break

        page += 1
        time.sleep(random.uniform(0.2, 0.7))

    if all_parts:
        return pd.concat(all_parts, ignore_index=True)

    return pd.DataFrame(
        columns=[
            "search_query",
            "window_from",
            "window_to",
            "publish_date",
            "project",
            "category",
            "title",
            "body",
            "fronturl",
        ]
    )


def collect_rbc_by_name_selenium(driver, query, date_from, date_to, verbose=False):
    warmup_rbc(driver, query)
    windows = month_windows(date_from, date_to)
    all_parts = []

    for window_from, window_to in windows:
        try:
            df_window = collect_rbc_one_window_selenium(driver, query, window_from, window_to, verbose=verbose)
            if not df_window.empty:
                all_parts.append(df_window)
        except Exception as e:
            print(f"FAILED Selenium window {query} {window_from} - {window_to}: {e}", flush=True)

    if all_parts:
        result = pd.concat(all_parts, ignore_index=True)
        result.drop_duplicates(subset=["fronturl"], inplace=True, ignore_index=True)
    else:
        result = pd.DataFrame(
            columns=[
                "search_query",
                "window_from",
                "window_to",
                "publish_date",
                "project",
                "category",
                "title",
                "body",
                "fronturl",
            ]
        )

    return result


def build_retry_strategy(total_retries=6, backoff_factor=1.0):
    return Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(403, 408, 425, 429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
        respect_retry_after_header=True,
    )


def build_page_session():
    session = requests.Session()
    retry = build_retry_strategy()
    adapter = HTTPAdapter(max_retries=retry, pool_connections=64, pool_maxsize=64)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
    )
    return session


def get_thread_session():
    if not hasattr(_thread_local, "session"):
        _thread_local.session = build_page_session()
    return _thread_local.session


def fetch_page_source(url, retries=8, timeout=25, backoff=1.8):
    last_err = None
    session = get_thread_session()

    for attempt in range(1, retries + 1):
        try:
            time.sleep(random.uniform(0.8, 1.8))
            response = session.get(url, timeout=timeout)
            status_code = response.status_code

            if status_code in (403, 429):
                raise requests.HTTPError(f"blocked_status_{status_code}", response=response)

            response.raise_for_status()

            if not response.encoding:
                response.encoding = response.apparent_encoding

            html = response.text
            if not isinstance(html, str) or not html.strip():
                raise ValueError("empty_page_source")

            return html
        except Exception as e:
            last_err = e
            sleep_s = min(45, backoff ** attempt + random.uniform(0.5, 2.0))
            time.sleep(sleep_s)

    raise last_err


def process_url(idx, raw_url):
    url = str(raw_url).strip()

    if not url or url.lower() == "nan":
        return idx, "", "empty_url"

    if "rbc.ru" not in url:
        return idx, "", "external_source"

    try:
        html = fetch_page_source(url)
        return idx, html, ""
    except Exception as e:
        return idx, "", str(e)


def save_parquet(df, output_path):
    try:
        import pyarrow  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Parquet output requires 'pyarrow'. Install it with 'python -m pip install pyarrow'."
        ) from e

    df.to_parquet(output_path, index=False)


def main1():
    stocks_names = (
        pd.read_csv(
            r"./local/My Drive\4. Disserartion research\calculations\code\data_loading\rus_stocks_names.txt",
            header=None,
        )
        .squeeze()
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    date_from = "01.01.2024"
    date_to = "01.04.2026"

    out_dir = Path(r"./local/Desktop\rbc_links")
    out_dir.mkdir(parents=True, exist_ok=True)

    for query in stocks_names:
        try:
            print(f"\nProcessing: {query}")
            result = collect_rbc_by_name(query, date_from, date_to)
            result.to_csv(
                out_dir / f"{query}.csv",
                index=False,
                encoding="utf-16",
                sep="\t",
            )
        except Exception as e:
            print(f"FAILED {query}: {e}")


def main2(max_workers=3):
    input_dir = Path(r"./local/My Drive\4. Disserartion research\calculations\data\news_links\links_rbc")
    output_dir = Path(r"./local/RBC_downloaded_news")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))[42:]
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {input_dir}")

    for file_path in csv_files:
        print(f"\nProcessing file: {file_path.name}")

        try:
            data = pd.read_csv(file_path, encoding="utf-16", sep="\t")
        except Exception as e:
            print(f"FAILED to read {file_path.name}: {e}")
            continue

        if data.empty:
            print(f"SKIPPED empty file: {file_path.name}")
            continue

        if "fronturl" not in data.columns:
            print(f"SKIPPED {file_path.name}: column 'fronturl' not found")
            continue

        data = data.drop_duplicates(subset=["fronturl"]).reset_index(drop=True)
        data["page_source"] = ""
        data["page_source_error"] = ""

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_url, idx, raw_url): idx
                for idx, raw_url in data["fronturl"].items()
            }

            with tqdm(total=len(futures), desc=file_path.stem, unit="url") as pbar:
                for future in as_completed(futures):
                    idx, html, error = future.result()
                    data.at[idx, "page_source"] = html
                    data.at[idx, "page_source_error"] = error
                    pbar.update(1)

        output_path = output_dir / f"{file_path.stem}.parquet"
        save_parquet(data, output_path)
        print(f"SAVED {output_path}")


if __name__ == "__main__":
    main2()
