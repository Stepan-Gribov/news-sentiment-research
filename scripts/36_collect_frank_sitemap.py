from __future__ import annotations

import argparse
import importlib.util
import re
import time
from pathlib import Path

import pandas as pd


COLLECTOR_PATH = Path(__file__).with_name("01_collect_links_frank.py")
SPEC = importlib.util.spec_from_file_location("frank_collector", COLLECTOR_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {COLLECTOR_PATH}")
frank = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(frank)

SITEMAP_INDEX_URL = "https://frankmedia.ru/sitemap_index.xml"
POST_SITEMAP_RE = re.compile(r"^https://frankmedia\.ru/post-sitemap(?:\d+)?\.xml$", re.I)
SITEMAP_ROW_RE = re.compile(
    r"^(?P<link>https?://frankmedia\.ru/\d+(?:\?[^\s]+)?)\t\d+\t(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2} [+-]\d{2}:\d{2})$",
    re.M,
)


def fetch_rendered_text(driver, url: str, wait_seconds: float) -> str:
    driver.set_page_load_timeout(30)
    driver.set_script_timeout(30)
    driver.get(url)
    time.sleep(wait_seconds)
    html = driver.page_source or ""
    if frank.is_blocked_html(html, driver.title, driver.current_url):
        raise RuntimeError(f"ServicePipe challenge for {url}")
    return str(driver.execute_script("return document.documentElement.innerText || ''"))


def discover_post_sitemaps(driver, wait_seconds: float) -> list[str]:
    fetch_rendered_text(driver, SITEMAP_INDEX_URL, wait_seconds)
    urls = [element.get_attribute("href") for element in driver.find_elements("css selector", "a[href]")]
    return sorted({url for url in urls if url and POST_SITEMAP_RE.fullmatch(url)})


def parse_sitemap_rows(rendered_text: str, sitemap_url: str, min_date: str | None, max_date: str | None) -> pd.DataFrame:
    rows = []
    for match in SITEMAP_ROW_RE.finditer(rendered_text):
        timestamp = match.group("timestamp")
        publication_date = timestamp[:10]
        if min_date and publication_date < min_date:
            continue
        if max_date and publication_date > max_date:
            continue
        rows.append(
            {
                "link": match.group("link"),
                "date": publication_date,
                "sitemap_lastmod": timestamp,
                "sitemap_url": sitemap_url,
            }
        )
    return pd.DataFrame(rows, columns=["link", "date", "sitemap_lastmod", "sitemap_url"]).drop_duplicates("link")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a resumable all-publications Frank Media index from official Yoast sitemaps.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/frank_full_archive/sitemap_links"))
    parser.add_argument("--min-date", default="2018-01-01")
    parser.add_argument("--max-date", default=None)
    parser.add_argument("--max-sitemaps", type=int, default=None, help="Cap sitemap count for a smoke test.")
    parser.add_argument("--remote-port", type=int, default=9223)
    parser.add_argument("--chrome-profile-dir", type=Path, default=Path("data/frank_full_archive/_sitemap_chrome"))
    parser.add_argument("--wait-seconds", type=float, default=1.5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.wait_seconds < 0.5:
        raise SystemExit("wait-seconds must be at least 0.5")
    if args.max_sitemaps is not None and args.max_sitemaps < 1:
        raise SystemExit("max-sitemaps must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    chrome = frank.start_remote_chrome(args.remote_port, args.chrome_profile_dir)
    driver = frank.attach_remote_driver(args.remote_port)
    statuses = []
    try:
        sitemaps = discover_post_sitemaps(driver, args.wait_seconds)
        if args.max_sitemaps:
            sitemaps = sitemaps[: args.max_sitemaps]
        if not sitemaps:
            raise RuntimeError("No post sitemaps found in sitemap index")
        print(f"Frank post sitemaps: {len(sitemaps):,}", flush=True)
        for number, sitemap_url in enumerate(sitemaps, start=1):
            out_path = args.out_dir / f"post-sitemap-{number:04d}.csv"
            if frank.should_skip_output(out_path, force=args.force):
                statuses.append({"sitemap": sitemap_url, "status": "skipped_existing", "rows": None})
                continue
            frank.mark_inprogress(out_path)
            try:
                rendered_text = fetch_rendered_text(driver, sitemap_url, args.wait_seconds)
                frame = parse_sitemap_rows(rendered_text, sitemap_url, args.min_date, args.max_date)
                if frame.empty:
                    raise RuntimeError("No article rows matched the rendered sitemap")
                frank.atomic_write_csv(frame, out_path)
                frank.clear_inprogress(out_path)
                statuses.append({"sitemap": sitemap_url, "status": "ok", "rows": len(frame)})
                print(f"{out_path.name}: {len(frame):,} publications", flush=True)
            except Exception as exc:
                statuses.append({"sitemap": sitemap_url, "status": "failed", "rows": None, "error": str(exc)})
                print(f"{sitemap_url}: {exc}", flush=True)
                break
    finally:
        pd.DataFrame(statuses).to_csv(args.out_dir / "_collection_status.csv", index=False)
        try:
            driver.quit()
        finally:
            chrome.terminate()


if __name__ == "__main__":
    main()
