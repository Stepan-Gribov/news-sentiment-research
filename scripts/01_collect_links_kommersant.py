from __future__ import annotations

import argparse
import random
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

import _bootstrap  # noqa: F401

from dissertation_pipeline.paths import PATHS
from dissertation_pipeline.news.layout import links_run_dir
from dissertation_pipeline.news.resume import atomic_write_csv, clear_inprogress, mark_inprogress, should_skip_output


def load_names(path: Path, sample_size: int | None) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen = set()
    names = [name for name in names if not (name in seen or seen.add(name))]
    return names[:sample_size] if sample_size else names


def build_url(search_query: str, date_start: str, date_end: str, page: int) -> str:
    return (
        "https://www.kommersant.ru/search/results"
        f"?search_query={search_query}&sort_type=0&search_full=1&tag=86"
        f"&tag_type=4&tag_extra=0&time_range=2"
        f"&dateStart={date_start}&dateEnd={date_end}&page={page}"
    )


def get_with_retry(session: requests.Session, url: str, retries: int = 8, timeout: int = 15) -> requests.Response:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except Exception as exc:
            last_error = exc
            time.sleep(min(30, 1.5**attempt + random.uniform(0.2, 1.0)))
    raise last_error


def parse_max_page(html: str) -> int:
    soup = BeautifulSoup(html, "html.parser")
    return max(
        (int(a.get_text(strip=True)) for a in soup.select("a.ui-paginator__link") if a.get_text(strip=True).isdigit()),
        default=1,
    )


def parse_links(search_query: str, html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for item in soup.select(".uho__item, .rubric_lenta__item"):
        link_tag = item.select_one("h2.uho__name a.uho__link")
        date_tag = item.select_one("p.uho__tag")
        if not link_tag:
            continue
        href = link_tag.get("href", "").strip().split("?", 1)[0]
        title = link_tag.get_text(strip=True)
        raw_date = date_tag.get_text(" ", strip=True) if date_tag else ""
        match = re.search(r"\b\d{2}\.\d{2}\.\d{4}\b", raw_date)
        timestamp = match.group(0) if match else ""
        if href and title:
            rows.append(
                {
                    "search_query": search_query,
                    "timestamp": timestamp,
                    "url": "https://www.kommersant.ru" + href,
                    "title": title,
                }
            )
    return pd.DataFrame(rows)


def collect_one_query(search_query: str, date_from: str, date_to: str) -> pd.DataFrame:
    start_bound = pd.Timestamp(date_from)
    end_bound = pd.Timestamp(date_to)
    starts = []
    cur = start_bound
    while cur < end_bound:
        starts.append(cur)
        cur = min(cur + pd.offsets.MonthBegin(1), end_bound)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 AppleWebKit/537.36 Chrome/122 Safari/537.36"})
    parts = []
    for start in starts:
        end = min(start + pd.offsets.MonthBegin(1), end_bound)
        date_start = start.strftime("%Y-%m-%d")
        date_end = end.strftime("%Y-%m-%d")
        first = get_with_retry(session, build_url(search_query, date_start, date_end, 1)).text
        max_page = parse_max_page(first)
        for page in range(1, max_page + 1):
            html = first if page == 1 else get_with_retry(session, build_url(search_query, date_start, date_end, page)).text
            df = parse_links(search_query, html)
            if not df.empty:
                parts.append(df)
        time.sleep(random.uniform(0.8, 1.8))
    if not parts:
        return pd.DataFrame(columns=["search_query", "timestamp", "url", "title"])
    result = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["url"])
    parsed_dates = pd.to_datetime(result["timestamp"], format="%d.%m.%Y", errors="coerce")
    result = result[(parsed_dates >= start_bound) & (parsed_dates <= end_bound)].copy()
    return result.reindex(columns=["search_query", "timestamp", "url", "title"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Kommersant article links by company names.")
    parser.add_argument("--names-file", type=Path, default=PATHS.company_names_file)
    parser.add_argument("--date-from", default="2024-01-01")
    parser.add_argument("--date-to", default="2026-04-01")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1, help="Parallel company workers.")
    parser.add_argument("--force", action="store_true", help="Recollect even if a valid output CSV already exists.")
    args = parser.parse_args()

    args.out_dir = args.out_dir or links_run_dir("kommersant", args.date_from, args.date_to)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    names = load_names(args.names_file, args.sample_size)

    def collect_one(name: str) -> None:
        out_path = args.out_dir / f"{name}.csv"
        if should_skip_output(out_path, force=args.force):
            print(f"Kommersant links: {name} [skip existing]", flush=True)
            return
        print(f"Kommersant links: {name}", flush=True)
        mark_inprogress(out_path)
        try:
            df = collect_one_query(name, args.date_from, args.date_to)
            atomic_write_csv(df, out_path)
        finally:
            clear_inprogress(out_path)

    workers = max(1, args.workers)
    print(f"Kommersant company workers={workers}", flush=True)
    if workers == 1:
        for name in tqdm(names, desc="Kommersant companies"):
            collect_one(name)
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(collect_one, name): name for name in names}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Kommersant companies"):
            name = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"Kommersant links failed: {name}: {exc}", flush=True)


if __name__ == "__main__":
    main()
