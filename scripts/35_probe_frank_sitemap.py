from __future__ import annotations

import importlib.util
import re
import time
from pathlib import Path
from xml.etree import ElementTree


SCRIPT = Path(__file__).with_name("01_collect_links_frank.py")
SPEC = importlib.util.spec_from_file_location("frank_collector", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {SCRIPT}")
frank = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(frank)

URLS = [
    "https://frankmedia.ru/robots.txt",
    "https://frankmedia.ru/sitemap.xml",
    "https://frankmedia.ru/sitemap_index.xml",
    "https://frankmedia.ru/wp-sitemap.xml",
    "https://frankmedia.ru/post-sitemap.xml",
]


def main() -> None:
    port = 9226
    profile = Path("data/frank_full_archive/_sitemap_probe_chrome")
    chrome = frank.start_remote_chrome(port, profile)
    driver = frank.attach_remote_driver(port)
    try:
        for url in URLS:
            driver.get(url)
            time.sleep(3)
            html = driver.page_source or ""
            document_text = driver.execute_script("return document.documentElement.innerText || ''")
            links = [element.get_attribute("href") for element in driver.find_elements("css selector", "a[href]")]
            sitemap_links = [link for link in links if link and "sitemap" in link]
            article_urls = re.findall(r"https?://frankmedia\.ru/\d+(?:\?[^\s]+)?", document_text)
            lastmods = re.findall(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}", document_text)
            first_article_offset = document_text.find("https://frankmedia.ru/")
            article_preview = document_text[first_article_offset:first_article_offset + 280] if first_article_offset >= 0 else ""
            blocked = frank.is_blocked_html(html, driver.title, driver.current_url)
            loc_count = 0
            if not blocked:
                try:
                    root = ElementTree.fromstring(document_text.lstrip("\ufeff"))
                    loc_count = sum(1 for node in root.iter() if node.tag.endswith("loc"))
                except ElementTree.ParseError:
                    pass
            print(
                f"url={url} blocked={blocked} final_url={driver.current_url} "
                f"bytes={len(html)} xml_locations={loc_count} sitemap_links={sitemap_links[:12]!r} "
                f"article_urls={len(article_urls)} lastmods={len(lastmods)} sample_articles={article_urls[:3]!r} "
                f"article_preview={article_preview!r}",
                flush=True,
            )
    finally:
        try:
            driver.quit()
        finally:
            chrome.terminate()


if __name__ == "__main__":
    main()
