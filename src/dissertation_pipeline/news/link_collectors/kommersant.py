import os
import re
import time
import random
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from bs4 import BeautifulSoup


INPUT_DIR = Path(r"./local/My Drive\4. Disserartion research\calculations\data\news_links\links_kommersant")
OUTPUT_DIR = Path(r"./local/kommersant_content")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- thread-local session ----
_thread_local = threading.local()


def get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        })
        _thread_local.session = session
    return _thread_local.session


def get_with_retry(url: str, retries: int = 8, timeout: int = 25) -> str:
    last_error = None
    session = get_session()

    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()

            if not response.encoding:
                response.encoding = response.apparent_encoding

            html = response.text
            if not html or not html.strip():
                raise ValueError("Empty page source")

            return html

        except Exception as e:
            last_error = e
            sleep_s = min(20, 1.5 ** attempt + random.uniform(0.2, 1.0))
            print(f"Retry {attempt}/{retries}: {url}")
            time.sleep(sleep_s)

    raise last_error


def extract_kommersant_article_text(page_source: str) -> str:
    if page_source is None or not isinstance(page_source, str) or not page_source.strip():
        return ""

    soup = BeautifulSoup(page_source, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()

    text_chunks = []

    paragraphs = soup.find_all("p", class_=re.compile(r"\bdoc__text\b"))
    for p in paragraphs:
        txt = p.get_text(" ", strip=True)
        txt = re.sub(r"\s+", " ", txt).strip()
        if txt:
            text_chunks.append(txt)

    if not text_chunks:
        article = (
            soup.find("article")
            or soup.find("div", class_=re.compile(r"(doc|article|content)", re.I))
            or soup.find("main")
        )

        if article:
            for p in article.find_all("p"):
                txt = p.get_text(" ", strip=True)
                txt = re.sub(r"\s+", " ", txt).strip()
                if txt:
                    text_chunks.append(txt)

    return " ".join(text_chunks).strip()


def process_one_row(idx: int, url: str):
    url = str(url).strip()

    if not url or url.lower() == "nan":
        return idx, "", "", "Empty URL"

    try:
        html = get_with_retry(url)
        article_text = extract_kommersant_article_text(html)
        return idx, html, article_text, ""
    except Exception as e:
        return idx, "", "", str(e)


def process_one_file(csv_path: Path, output_path: Path, max_workers: int = 12) -> None:
    df = pd.read_excel(csv_path)

    if "url" not in df.columns:
        raise ValueError(f"Column 'url' not found in {csv_path.name}")

    df["page_source"] = ""
    df["article_text"] = ""
    df["error"] = ""

    total = len(df)
    done = 0

    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, url in df["url"].items():
            futures[executor.submit(process_one_row, idx, url)] = idx

        for future in as_completed(futures):
            idx, html, article_text, error = future.result()

            df.at[idx, "page_source"] = html
            df.at[idx, "article_text"] = article_text
            df.at[idx, "error"] = error

            done += 1
            print(f"{csv_path.name} | {done}/{total}", end="\r", flush=True)

    print()
    df.to_csv(output_path, index=False, encoding="utf-16", sep="\t")


def main():
    xlsx_files = list(sorted(INPUT_DIR.glob("*.xlsx")))[123:]

    if not xlsx_files:
        raise RuntimeError(f"No XLSX files found in {INPUT_DIR}")

    for num, xlsx_path in enumerate(xlsx_files, start=1):
        try:
            output_path = OUTPUT_DIR / f"{xlsx_path.stem}.csv"
            print(f"\n[{num}/{len(xlsx_files)}] Processing: {xlsx_path.name}")
            process_one_file(xlsx_path, output_path, max_workers=12)
            print(f"[{num}/{len(xlsx_files)}] Saved: {output_path.name}")
        except Exception as e:
            print(f"[{num}/{len(xlsx_files)}] FAILED: {xlsx_path.name} -> {e}")


if __name__ == "__main__":
    main()