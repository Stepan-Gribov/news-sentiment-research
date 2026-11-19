import os
import re
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from dateutil.relativedelta import relativedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from dissertation_pipeline.selenium_fast import apply_fast_chrome_options, configure_fast_driver
from selenium import webdriver
from selenium.common.exceptions import (
    SessionNotCreatedException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait


def get_with_retry(session, url, retries=3, timeout=10, backoff=1.5):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code in (403, 429):
                raise requests.HTTPError(f"blocked_status_{r.status_code}", response=r)
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            if isinstance(e, requests.HTTPError) and "blocked_status" in str(e):
                raise
            sleep_s = min(60, backoff ** attempt + random.uniform(0.2, 1.0))
            print(f"Retry {attempt}/{retries}: {url}")
            time.sleep(sleep_s)
    raise last_err


def build_url(search_query, dt_start, dt_end, offset):
    return (
        "https://1prime.ru/services/search/getmore/"
        f"?tags_limit=20"
        f"&date_from={dt_start}"
        f"&date_to={dt_end}"
        f"&sort=date"
        f"&query={search_query}"
        f"&offset={offset}"
    )


def parse_1prime_page(html, search_query):
    soup = BeautifulSoup(html, "html.parser")

    container = soup.find("div", class_="list-items-loaded")
    if container is None:
        return pd.DataFrame(), False

    # stop condition: explicit "nothing found"
    notfound = container.find("div", class_="list-item-notfound")
    if notfound:
        return pd.DataFrame(), False

    items = container.find_all("div", class_="list-item")
    rows = []

    for item in items:
        title_tag = item.find("a", class_="list-item__title")
        if not title_tag:
            continue

        href = title_tag.get("href", "").strip()
        title = title_tag.get_text(strip=True)

        date_tag = item.find("div", class_="list-item__date")
        timestamp = date_tag.get_text(strip=True) if date_tag else None

        full_link = href
        if href.startswith("/"):
            full_link = "https://1prime.ru" + href

        rows.append({
            "search_query": search_query,
            "name": title,
            "link": full_link,
            "category": "",       # endpoint sample does not show category
            "timestamp": timestamp
        })

    df = pd.DataFrame(rows)

    # determine whether there may be next page
    next_url = container.get("data-next-url", "")
    has_next = bool(next_url.strip()) and not df.empty

    return df, has_next


RU_MONTHS = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}


def parse_prime_timestamp(timestamp, window_start):
    if not isinstance(timestamp, str) or not timestamp.strip():
        return pd.NaT
    match = re.search(r"(\d{1,2})\s+([а-яё]+)(?:,\s*(\d{1,2}):(\d{2}))?", timestamp.lower())
    if not match:
        return pd.NaT
    day = int(match.group(1))
    month = RU_MONTHS.get(match.group(2))
    if not month:
        return pd.NaT
    year = datetime.strptime(window_start, "%Y-%m-%d").year
    hour = int(match.group(3) or 0)
    minute = int(match.group(4) or 0)
    return pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)


def collect_prime_window(name_, dt_start, dt_end):
    """
    Collect one date window. This is split out so long historical runs can
    parallelize monthly windows without sharing requests.Session across threads.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://1prime.ru/",
    })

    columns = ["search_query", "name", "link", "category", "timestamp", "date"]
    parts = []

    offset = 0
    seen_links = set()

    while True:
        url = build_url(name_, dt_start, dt_end, offset)
        r = get_with_retry(session, url)
        page_source = r.text

        data, has_next = parse_1prime_page(page_source, name_)

        if data.empty:
            break

        data = data[~data["link"].isin(seen_links)]
        if data.empty:
            break

        data["date"] = data["timestamp"].map(lambda value: parse_prime_timestamp(value, dt_start))
        seen_links.update(data["link"].tolist())
        parts.append(data)

        if not has_next:
            break

        offset += 20
        time.sleep(random.uniform(0.2, 0.5))

    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True).drop_duplicates(subset=["link"]).reindex(columns=columns)


def open_browser(name_, from_dates, to_dates, driver=None, max_workers=1):
    """
    Kept your function name for compatibility, but now uses requests instead
    of Selenium. `max_workers` parallelizes independent monthly windows.
    """
    columns = ["search_query", "name", "link", "category", "timestamp", "date"]
    windows = list(zip(from_dates, to_dates))
    if max_workers <= 1 or len(windows) <= 1:
        parts = [collect_prime_window(name_, start, end) for start, end in windows]
    else:
        parts = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(collect_prime_window, name_, start, end): (start, end)
                for start, end in windows
            }
            for future in as_completed(futures):
                start, end = futures[future]
                try:
                    parts.append(future.result())
                except Exception as exc:
                    print(f"Prime window failed for {name_}: {start} - {end}: {exc}", flush=True)

    parts = [part for part in parts if not part.empty]
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True).drop_duplicates(subset=["link"]).reindex(columns=columns)


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


def build_chrome_options():
    options = Options()
    apply_fast_chrome_options(options)
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1440,2200")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
    return options


def build_driver():
    chromedriver_path = Path(__file__).resolve().parent / "chromedriver-win64" / "chromedriver.exe"
    options = build_chrome_options()

    if chromedriver_path.exists():
        try:
            service = Service(executable_path=str(chromedriver_path))
            driver = webdriver.Chrome(service=service, options=options)
            configure_fast_driver(driver)
            return driver
        except SessionNotCreatedException:
            pass

    driver = webdriver.Chrome(options=options)
    configure_fast_driver(driver)
    return driver


def is_valid_prime_page_source(page_source):
    if not isinstance(page_source, str) or not page_source.strip():
        return False

    return (
        'class="article__body' in page_source
        or 'itemprop="articleBody"' in page_source
        or 'data-page-project="prime"' in page_source
    )


def fetch_page_source_selenium(driver, url, wait_timeout=20):
    driver.get(url)

    WebDriverWait(driver, wait_timeout).until(
        lambda d: d.execute_script("return document.readyState") in ("interactive", "complete")
    )

    try:
        WebDriverWait(driver, wait_timeout).until(
            lambda d: (
                len(d.find_elements(By.CSS_SELECTOR, ".article__body")) > 0
                or len(d.find_elements(By.CSS_SELECTOR, '[itemprop="articleBody"]')) > 0
            )
        )
    except TimeoutException:
        pass

    page_source = driver.page_source
    if not is_valid_prime_page_source(page_source):
        raise ValueError("invalid_or_blocked_page_source")

    return page_source


def save_parquet(df, output_path):
    try:
        import pyarrow  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Parquet output requires 'pyarrow'. Install it with 'python -m pip install pyarrow'."
        ) from e

    df.to_parquet(output_path, index=False)


def main():
    stocks_names = pd.read_csv(
        r'./local/My Drive\4. Disserartion research\calculations\code\data_loading\rus_stocks_names.txt',
        header=None
    ).squeeze().dropna().astype(str).unique().tolist()

    start_date = datetime.strptime("01.01.2024", "%d.%m.%Y")
    end_date = datetime.strptime("01.03.2026", "%d.%m.%Y")

    from_dates = []
    to_dates = []

    current_date = start_date
    while current_date < end_date:
        from_dates.append(current_date.strftime("%Y-%m-%d"))
        next_date = current_date + relativedelta(months=1)
        to_dates.append(next_date.strftime("%Y-%m-%d"))
        current_date = next_date

    out_dir = r'./local/Desktop/links_prime'
    os.makedirs(out_dir, exist_ok=True)

    for name_ in stocks_names:
        try:
            print(f"Processing: {name_}")
            res = open_browser(name_, from_dates, to_dates)

            # final dedup across all windows
            if not res.empty:
                res.drop_duplicates(subset=["link"], inplace=True, ignore_index=True)

            res.to_csv(
                os.path.join(out_dir, f"{name_}.csv"),
                index=False,
                encoding='utf-16',
                sep='\t'
            )

        except Exception as e:
            print(f"FAILED {name_}: {e}")


def main2():
    input_dir = Path(
        r"./local/My Drive\4. Disserartion research\calculations\data\news_links\links_prime"
    )
    output_dir = Path(r"./local/prime_downloaded_news")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {input_dir}")

    driver = build_driver()
    try:
        for file_path in csv_files[103:]:
            print(f"\nProcessing file: {file_path.name}")

            try:
                data = pd.read_csv(file_path, encoding="utf-16", sep="\t")
            except Exception as e:
                print(f"FAILED to read {file_path.name}: {e}")
                continue

            if data.empty:
                print(f"SKIPPED empty file: {file_path.name}")
                continue

            if "link" not in data.columns:
                print(f"SKIPPED {file_path.name}: column 'link' not found")
                continue

            data = data.drop_duplicates(subset=["link"]).reset_index(drop=True)
            data["page_source"] = ""
            data["page_source_error"] = ""

            with tqdm(total=len(data), desc=file_path.stem, unit="url") as pbar:
                for idx, raw_url in data["link"].items():
                    url = str(raw_url).strip()

                    if not url or url.lower() == "nan":
                        data.at[idx, "page_source_error"] = "empty_url"
                        pbar.update(1)
                        continue

                    if "1prime.ru" not in url:
                        data.at[idx, "page_source_error"] = "external_source"
                        pbar.update(1)
                        continue

                    page_source = ""
                    error = ""

                    for attempt in range(1, 4):
                        try:
                            time.sleep(random.uniform(1.0, 2.0))
                            page_source = fetch_page_source_selenium(driver, url)
                            error = ""
                            break
                        except (TimeoutException, WebDriverException, ValueError) as e:
                            error = f"attempt_{attempt}: {e}"
                            try:
                                driver.quit()
                            except Exception:
                                pass
                            driver = build_driver()
                            time.sleep(random.uniform(2.0, 4.0))

                    data.at[idx, "page_source"] = page_source
                    data.at[idx, "page_source_error"] = error
                    pbar.update(1)

            output_path = output_dir / f"{file_path.stem}.parquet"
            save_parquet(data, output_path)
            print(f"SAVED {output_path}")
    finally:
        try:
            driver.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main2()
