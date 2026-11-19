import os
import time
import random
import threading
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


_thread_local = threading.local()


def get_with_retry(session, url, retries=10, timeout=20, backoff=1.5):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            sleep_s = min(30, backoff ** attempt + random.uniform(0.2, 1.0))
            print(f"Retry {attempt}/{retries}: {url}")
            time.sleep(sleep_s)
    raise last_err


def build_search_url(query, from_, to_, page=1, sec=0):
    return (
        "https://www.interfax.ru/search/"
        f"?phrase={query}"
        f"&df={from_}"
        f"&dt={to_}"
        f"&sec={sec}"
        f"&p={page}"
    )


def parse_search_page(html):
    soup = BeautifulSoup(html, "html.parser")

    result_block = soup.find("div", class_="sPageResult")
    if result_block is None:
        return pd.DataFrame(), 1

    rows = []

    for item in result_block.find_all("div", recursive=False):
        time_tag = item.find("time")
        if not time_tag:
            continue

        category = ""
        title = ""
        news_link = ""

        # --- case 1: photo card ---
        title_block = item.find("div", class_="title")
        if title_block:
            inner_links = title_block.find_all("a", href=True)
            if len(inner_links) >= 2:
                category = inner_links[0].get_text(strip=True)
                news_link = inner_links[-1].get("href", "").strip()
                title = inner_links[-1].get_text(strip=True)

        # --- case 2: regular card ---
        if not title:
            links = item.find_all("a", href=True)
            if len(links) >= 2:
                category = links[0].get_text(strip=True)
                news_link = links[-1].get("href", "").strip()
                title = links[-1].get_text(strip=True)

        if not news_link.startswith("/"):
            continue

        rows.append({
            "category": category,
            "title": title,
            "link": "https://www.interfax.ru" + news_link,
            "date": time_tag.get("datetime", "").strip()
        })

    df = pd.DataFrame(rows)

    max_page = 1
    pag_block = soup.find("div", class_="PDSelect")
    if pag_block:
        for a in pag_block.find_all("a"):
            t = a.get_text(strip=True)
            if t.isdigit():
                max_page = max(max_page, int(t))

    return df, max_page


def collect_by_name(query, from_dates, to_dates):
    data = pd.DataFrame()

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://www.interfax.ru/search/"
    })

    for from_, to_ in zip(from_dates, to_dates):
        try:
            # first page
            url = build_search_url(query, from_, to_, page=1, sec=0)
            r = get_with_retry(session, url)
            df_page, max_page = parse_search_page(r.text)

            if not df_page.empty:
                data = pd.concat([data, df_page], ignore_index=True)

            # next pages
            for page in range(2, max_page + 1):
                url = build_search_url(query, from_, to_, page=page, sec=0)
                r = get_with_retry(session, url)
                df_page, _ = parse_search_page(r.text)

                if df_page.empty:
                    break

                data = pd.concat([data, df_page], ignore_index=True)
                time.sleep(random.uniform(0.2, 0.7))

        except Exception as e:
            print(f"FAILED window {from_} - {to_} for {query}: {e}")

    if not data.empty:
        data.drop_duplicates(subset=["link"], inplace=True, ignore_index=True)

    return data


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
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    })
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

    if "interfax.ru" not in url:
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


def main():
    start_date = datetime.strptime("01.01.2024", "%d.%m.%Y")
    end_date = datetime.strptime("01.03.2026", "%d.%m.%Y")

    from_dates = []
    to_dates = []

    current_date = start_date
    while current_date < end_date:
        from_dates.append(current_date.strftime("%d.%m.%Y"))
        next_date = current_date + relativedelta(months=1)
        to_dates.append(next_date.strftime("%d.%m.%Y"))
        current_date = next_date

    stocks_names = pd.read_csv(
        r'./local/Desktop/links_interfax/rus_stocks_names.txt',
        header=None
    ).squeeze().dropna().astype(str).unique().tolist()

    out_dir = r'./local/Desktop/links_interfax'
    os.makedirs(out_dir, exist_ok=True)

    for name in stocks_names[:]:
        print(f"Processing: {name}")
        data = collect_by_name(name, from_dates, to_dates)
        data.to_csv(
            os.path.join(out_dir, f"{name}.csv"),
            index=False,
            encoding='utf-16',
            sep='\t'
        )


def main2(max_workers=3):
    input_dir = Path(
        r"./local/My Drive\4. Disserartion research\calculations\data\news_links\links_interfax"
    )
    output_dir = Path(r"./local/interfax_downloaded_news")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))
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

        if "link" not in data.columns:
            print(f"SKIPPED {file_path.name}: column 'link' not found")
            continue

        data = data.drop_duplicates(subset=["link"]).reset_index(drop=True)
        data["page_source"] = ""
        data["page_source_error"] = ""

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_url, idx, raw_url): idx
                for idx, raw_url in data["link"].items()
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
