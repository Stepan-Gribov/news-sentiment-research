from __future__ import annotations

import atexit
import gzip
import html as html_lib
import json
import multiprocessing as mp
import random
import re
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from bs4 import BeautifulSoup

from dissertation_pipeline.news.interfax_archive import (
    HttpClient,
    LocalRateLimiter,
    SharedRateLimiter,
    chunked,
    normalize_text,
    parse_sitemap_locations,
)


SITEMAP_INDEX_URL = "https://www.rbc.ru/sitemap_index.xml"
SITEMAP_RE = re.compile(r"sitemap_(\d+)\.xml$", re.I)
MATERIAL_ID_RE = re.compile(r"^[0-9a-f]{24}$", re.I)
DATED_PATH_RE = re.compile(r"/(\d{2})/(\d{2})/(\d{4})/[0-9a-f]{24}$", re.I)
NON_ARTICLE_SEGMENTS = {
    "companies",
    "person",
    "photoreport",
    "photoslider",
    "story",
    "tags",
    "video",
}
RETRYABLE_BROWSER_ERRORS = {"qrator_challenge", "browser_timeout", "webdriver_error"}


OUTPUT_COLUMNS = [
    "media",
    "material_id",
    "requested_material_id",
    "doc_id",
    "url",
    "link",
    "final_url",
    "http_status",
    "parse_status",
    "timestamp",
    "published_at",
    "publication_date",
    "date_source",
    "modified_at",
    "title",
    "subtitle",
    "description",
    "authors",
    "category",
    "rubric",
    "section",
    "url_section",
    "region",
    "tags",
    "tickers",
    "companies",
    "material_type",
    "project_nick",
    "is_paywall",
    "access_level",
    "article_text_extracted",
    "text_length",
    "response_bytes",
    "sitemap_group",
    "sitemap_lastmod",
    "browser_attempts",
    "fetched_at_utc",
    "error",
    "extraction_version",
]


def extract_material_identity(url: str) -> tuple[str, str, str] | None:
    parsed = urlparse(str(url).strip())
    if parsed.netloc.lower() not in {"rbc.ru", "www.rbc.ru"}:
        return None
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 2 or not MATERIAL_ID_RE.fullmatch(parts[-1]):
        return None
    lowered = [part.lower() for part in parts[:-1]]
    if any(part in NON_ARTICLE_SEGMENTS for part in lowered):
        return None
    has_date = bool(DATED_PATH_RE.search(parsed.path.rstrip("/")))
    has_news_marker = any(part in {"news", "freenews", "rbcfreenews"} for part in lowered)
    if not (has_date or has_news_marker):
        return None
    date_match = DATED_PATH_RE.search(parsed.path.rstrip("/"))
    publication_date = ""
    if date_match:
        publication_date = f"{date_match.group(3)}-{date_match.group(2)}-{date_match.group(1)}"
    return parts[0].lower(), parts[-1].lower(), publication_date


def sitemap_group(url: str) -> tuple[str, int] | None:
    match = SITEMAP_RE.search(urlparse(url).path.rsplit("/", 1)[-1])
    if not match:
        return None
    number = int(match.group(1))
    return f"map-{number:04d}", number


def normalize_datetime(value: object) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    try:
        return parsedate_to_datetime(text).isoformat()
    except (TypeError, ValueError, OverflowError):
        pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return text


def markdown_to_text(value: object) -> str:
    text = html_lib.unescape(str(value or "")).replace("\r\n", "\n")
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"</?[^>]+>", " ", text)
    text = re.sub(r"(?m)^\s{0,3}(?:#{1,6}|>|[-*+]\s+)\s*", "", text)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    parts = [normalize_text(part.strip(" *_")) for part in text.split("\n")]
    return "\n".join(part for part in parts if part)


def _named_values(values: object) -> list[str]:
    if not isinstance(values, list):
        values = [values] if values else []
    result: list[str] = []
    for value in values:
        if isinstance(value, dict):
            candidate = value.get("name") or value.get("title") or value.get("ticker") or value.get("code")
        else:
            candidate = value
        text = normalize_text(candidate)
        if text and text not in result:
            result.append(text)
    return result


def extract_next_material(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    node = soup.select_one("#__NEXT_DATA__")
    if not node:
        return {}
    parsed = json.loads(node.get_text())
    material = parsed.get("props", {}).get("pageProps", {}).get("material", {})
    return material if isinstance(material, dict) else {}


def _meta_content(soup: BeautifulSoup, *, property_name: str | None = None, name: str | None = None) -> str:
    selector = f'meta[property="{property_name}"]' if property_name else f'meta[name="{name}"]'
    node = soup.select_one(selector)
    return normalize_text(node.get("content", "")) if node else ""


def _html_article_text(soup: BeautifulSoup) -> str:
    for selector in ("[itemprop='articleBody']", ".article__text", ".article__content", "article"):
        node = soup.select_one(selector)
        if not node:
            continue
        copy = BeautifulSoup(str(node), "html.parser")
        for unwanted in copy.select("script, style, noscript, iframe, svg, .banner, .advert, .adfox"):
            unwanted.decompose()
        text = normalize_text(copy.get_text(" ", strip=True))
        if text:
            return text
    return ""


def parse_html_article(html: str, requested_url: str, final_url: str) -> dict[str, object]:
    """Fallback for older RBC pages which no longer expose ``__NEXT_DATA__``."""
    soup = BeautifulSoup(html, "html.parser")
    requested_identity = extract_material_identity(requested_url)
    requested_id = requested_identity[1] if requested_identity else ""
    canonical_node = soup.select_one('link[rel="canonical"]')
    canonical_url = normalize_text(canonical_node.get("href", "")) if canonical_node else final_url
    canonical_identity = extract_material_identity(canonical_url) or requested_identity
    material_id = canonical_identity[1] if canonical_identity else requested_id
    url_section = canonical_identity[0] if canonical_identity else (requested_identity[0] if requested_identity else "")
    h1 = soup.select_one("h1")
    title = _meta_content(soup, property_name="og:title") or normalize_text(h1.get_text(" ", strip=True) if h1 else "")
    description = _meta_content(soup, name="description") or _meta_content(soup, property_name="og:description")
    published_at = normalize_datetime(
        _meta_content(soup, property_name="article:published_time")
        or _meta_content(soup, name="date")
    )
    url_date = canonical_identity[2] if canonical_identity else (requested_identity[2] if requested_identity else "")
    publication_date = published_at[:10] if published_at else url_date
    body = _html_article_text(soup)
    if description and description not in body:
        body = f"{description}\n{body}" if body else description
    has_evidence = bool(material_id and (title or publication_date or body))
    status = "ok" if has_evidence else "not_article"
    if status == "ok" and not body:
        status = "metadata_only"
    if final_url.rstrip("/") != requested_url.rstrip("/"):
        status = f"redirected_{status}"
    return {
        **{column: "" for column in OUTPUT_COLUMNS},
        "media": "rbc",
        "material_id": material_id,
        "requested_material_id": requested_id,
        "doc_id": material_id,
        "url": canonical_url or final_url or requested_url,
        "link": canonical_url or final_url or requested_url,
        "parse_status": status,
        "timestamp": published_at or publication_date,
        "published_at": published_at or publication_date,
        "publication_date": publication_date,
        "date_source": "page_metadata" if published_at else "url" if url_date else "",
        "title": title,
        "description": description,
        "category": _meta_content(soup, property_name="article:section") or url_section,
        "rubric": _meta_content(soup, property_name="article:section"),
        "section": _meta_content(soup, property_name="article:section"),
        "url_section": url_section,
        "tags": "; ".join(
            value for value in (_meta_content(soup, property_name="article:tag"),) if value
        ),
        "article_text_extracted": body,
        "text_length": len(body),
        # Keep this nullable so a fallback row can share a Parquet part with
        # rows parsed from the Next.js payload, where the column is boolean.
        "is_paywall": None,
        "extraction_version": "rbc_archive_html_fallback_v1",
    }


def parse_material(material: dict, requested_url: str) -> dict[str, object]:
    requested_identity = extract_material_identity(requested_url)
    requested_id = requested_identity[1] if requested_identity else ""
    canonical_url = normalize_text(material.get("fronturl") or material.get("url") or requested_url)
    canonical_identity = extract_material_identity(canonical_url) or requested_identity
    material_id = normalize_text(material.get("id")) or (canonical_identity[1] if canonical_identity else requested_id)
    url_section = canonical_identity[0] if canonical_identity else (requested_identity[0] if requested_identity else "")

    published_at = normalize_datetime(material.get("publishDate") or material.get("urlDate"))
    modified_at = normalize_datetime(material.get("modifDate"))
    url_date = canonical_identity[2] if canonical_identity else (requested_identity[2] if requested_identity else "")
    publication_date = published_at[:10] if published_at else url_date
    date_source = "page_metadata" if published_at else "url" if url_date else ""

    title = normalize_text(material.get("title") or material.get("seoTitle"))
    subtitle = normalize_text(material.get("subtitle") or material.get("anonsText"))
    description = normalize_text(
        material.get("metaDescription") or material.get("seoDescription") or subtitle
    )
    body = markdown_to_text(material.get("bodyMd")) or normalize_text(material.get("bodyText"))
    if description and description not in body and body not in description:
        body = f"{description}\n{body}" if body else description

    category_node = material.get("category") if isinstance(material.get("category"), dict) else {}
    category = normalize_text(category_node.get("name") or category_node.get("title"))
    categories = _named_values(material.get("categories"))
    tags = _named_values(material.get("tags")) + _named_values(material.get("masterTags"))
    tags = list(dict.fromkeys(tags))
    authors = _named_values(material.get("author")) + _named_values(material.get("authorInvolved"))
    author_name = normalize_text(material.get("authorName"))
    if author_name:
        authors.insert(0, author_name)
    authors = list(dict.fromkeys(authors))

    companies = _named_values(material.get("companyCards"))
    tickers = _named_values(material.get("tickers"))
    region_node = material.get("region") if isinstance(material.get("region"), dict) else {}
    region = normalize_text(region_node.get("name") or material.get("regionNick"))

    has_evidence = bool(material_id and (title or published_at or body))
    parse_status = "ok" if has_evidence else "not_article"
    if parse_status == "ok" and not body:
        parse_status = "metadata_only"
    if requested_id and material_id and material_id != requested_id:
        parse_status = f"redirected_{parse_status}"

    return {
        "media": "rbc",
        "material_id": material_id,
        "requested_material_id": requested_id,
        "doc_id": material_id,
        "url": canonical_url,
        "link": canonical_url,
        "parse_status": parse_status,
        "timestamp": published_at or publication_date,
        "published_at": published_at or publication_date,
        "publication_date": publication_date,
        "date_source": date_source,
        "modified_at": modified_at,
        "title": title,
        "subtitle": subtitle,
        "description": description,
        "authors": "; ".join(authors),
        "category": category or "; ".join(categories) or url_section,
        "rubric": "; ".join(categories) or category,
        "section": category,
        "url_section": url_section,
        "region": region,
        "tags": "; ".join(tags),
        "tickers": "; ".join(tickers),
        "companies": "; ".join(companies),
        "material_type": normalize_text(material.get("type")),
        "project_nick": normalize_text(material.get("projectNick")),
        "is_paywall": bool(material.get("isPaywall")),
        "access_level": normalize_text(material.get("accessLevel")),
        "article_text_extracted": body,
        "text_length": len(body),
        "extraction_version": "rbc_archive_v1",
    }


def inventory_row(candidate: dict[str, object]) -> dict[str, object]:
    publication_date = str(candidate.get("url_date", ""))
    return {
        **{column: "" for column in OUTPUT_COLUMNS},
        "media": "rbc",
        "material_id": candidate["material_id"],
        "requested_material_id": candidate["material_id"],
        "doc_id": candidate["material_id"],
        "url": candidate["url"],
        "link": candidate["url"],
        "final_url": candidate["url"],
        "parse_status": "inventory_only",
        "timestamp": publication_date,
        "published_at": publication_date,
        "publication_date": publication_date,
        "date_source": "url" if publication_date else "",
        "url_section": candidate.get("url_section", ""),
        # Content rows use a boolean here.  Keep inventory/error rows nullable
        # rather than mixing an empty string with booleans in Parquet parts.
        "is_paywall": None,
        "text_length": 0,
        "sitemap_group": candidate.get("sitemap_group", ""),
        "sitemap_lastmod": candidate.get("sitemap_lastmod", ""),
        "extraction_version": "rbc_inventory_v1",
    }


def discover_sitemap_urls(
    client: HttpClient,
    *,
    min_map: int,
    max_map: int | None,
    newest_first: bool,
    index_url: str = SITEMAP_INDEX_URL,
) -> list[tuple[str, str]]:
    result = client.get(index_url)
    if result.status != 200:
        raise RuntimeError(f"Cannot download RBC sitemap index: {result.status} {result.error}")
    selected: list[tuple[int, str, str]] = []
    for row in parse_sitemap_locations(result.text):
        group = sitemap_group(row["loc"])
        if not group:
            continue
        group_name, number = group
        if number >= min_map and (max_map is None or number <= max_map):
            selected.append((number, group_name, row["loc"]))
    selected.sort(key=lambda item: item[0], reverse=newest_first)
    return [(group, url) for _, group, url in selected]


def discover_group_candidates(client: HttpClient, group: str, url: str) -> list[dict[str, object]]:
    result = client.get(url)
    if result.status != 200:
        raise RuntimeError(f"Cannot download RBC sitemap {group}: {result.status} {result.error}")
    by_id: dict[str, dict[str, object]] = {}
    for row in parse_sitemap_locations(result.text):
        identity = extract_material_identity(row["loc"])
        if identity is None:
            continue
        section, material_id, publication_date = identity
        by_id[material_id] = {
            "material_id": material_id,
            "url": row["loc"],
            "url_section": section,
            "url_date": publication_date,
            "sitemap_group": group,
            "sitemap_lastmod": row.get("lastmod", ""),
        }
    return [by_id[key] for key in sorted(by_id)]


_BROWSER_DRIVER = None
_BROWSER_LIMITER = None
_BROWSER_TIMEOUT = 30.0
_BROWSER_RETRIES = 3
_BROWSER_PROFILE_DIR: Path | None = None


def _close_browser() -> None:
    global _BROWSER_DRIVER, _BROWSER_PROFILE_DIR
    if _BROWSER_DRIVER is not None:
        try:
            _BROWSER_DRIVER.quit()
        except Exception:
            pass
        _BROWSER_DRIVER = None
    if _BROWSER_PROFILE_DIR is not None:
        shutil.rmtree(_BROWSER_PROFILE_DIR, ignore_errors=True)
        _BROWSER_PROFILE_DIR = None


def _build_browser():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    global _BROWSER_PROFILE_DIR
    _BROWSER_PROFILE_DIR = Path(tempfile.mkdtemp(prefix="rbc-chrome-"))
    options = Options()
    options.page_load_strategy = "none"
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-background-networking")
    options.add_argument("--disable-renderer-backgrounding")
    # A separate temporary profile prevents a detached scheduled task and a
    # concurrent local Chrome session from contending for the same profile.
    options.add_argument(f"--user-data-dir={_BROWSER_PROFILE_DIR}")
    options.add_argument("--remote-debugging-port=0")
    options.add_argument("--disable-features=RendererCodeIntegrity")
    options.add_argument("--no-default-browser-check")
    options.add_argument("--no-first-run")
    options.add_argument("--blink-settings=imagesEnabled=false")
    options.add_argument("--window-size=1440,1200")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(15)
    try:
        driver.execute_cdp_cmd("Network.enable", {})
        driver.execute_cdp_cmd(
            "Network.setBlockedURLs",
            {
                "urls": [
                    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp", "*.svg", "*.ico",
                    "*.woff", "*.woff2", "*.ttf", "*.mp4", "*adfox.ru*", "*mc.yandex.ru*",
                    "*googletagmanager.com*", "*google-analytics.com*",
                ]
            },
        )
    except Exception:
        pass
    return driver


def initialize_browser_worker(
    requests_per_second: float,
    timeout: float,
    retries: int,
    next_allowed,
    blocked_until,
    rate_lock,
) -> None:
    global _BROWSER_DRIVER, _BROWSER_LIMITER, _BROWSER_TIMEOUT, _BROWSER_RETRIES
    _BROWSER_LIMITER = SharedRateLimiter(requests_per_second, next_allowed, blocked_until, rate_lock)
    _BROWSER_TIMEOUT = timeout
    _BROWSER_RETRIES = max(1, retries)
    # Selenium Manager uses one shared cache. Serialize browser construction
    # so concurrently spawned Windows workers do not race while touching it.
    with rate_lock:
        _BROWSER_DRIVER = _build_browser()
    atexit.register(_close_browser)


def _browser_candidate(candidate: dict[str, object]) -> dict[str, object]:
    global _BROWSER_DRIVER
    requested_url = str(candidate["url"])
    requested_id = str(candidate["material_id"])
    last_error = "browser_timeout"
    last_html = ""
    final_url = requested_url

    for attempt in range(1, _BROWSER_RETRIES + 1):
        try:
            assert _BROWSER_LIMITER is not None
            _BROWSER_LIMITER.wait()
            if _BROWSER_DRIVER is None:
                _BROWSER_DRIVER = _build_browser()
            _BROWSER_DRIVER.get(requested_url)
            deadline = time.monotonic() + _BROWSER_TIMEOUT
            while time.monotonic() < deadline:
                time.sleep(0.25)
                last_html = _BROWSER_DRIVER.page_source
                final_url = _BROWSER_DRIVER.current_url
                try:
                    material = extract_next_material(last_html)
                except (json.JSONDecodeError, TypeError):
                    material = {}
                if material:
                    parsed = parse_material(material, requested_url)
                    return {
                        **{column: "" for column in OUTPUT_COLUMNS},
                        **parsed,
                        "final_url": final_url,
                        "http_status": 200,
                        "response_bytes": len(last_html.encode("utf-8", errors="ignore")),
                        "sitemap_group": candidate.get("sitemap_group", ""),
                        "sitemap_lastmod": candidate.get("sitemap_lastmod", ""),
                        "browser_attempts": attempt,
                        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                        "error": "",
                    }
                fallback = parse_html_article(last_html, requested_url, final_url)
                if fallback["parse_status"] not in {"not_article", "redirected_not_article"}:
                    return {
                        **fallback,
                        "final_url": final_url,
                        "http_status": 200,
                        "response_bytes": len(last_html.encode("utf-8", errors="ignore")),
                        "sitemap_group": candidate.get("sitemap_group", ""),
                        "sitemap_lastmod": candidate.get("sitemap_lastmod", ""),
                        "browser_attempts": attempt,
                        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                        "error": "",
                    }
                title = normalize_text(_BROWSER_DRIVER.title).lower()
                if "страница не найдена" in title or title.startswith("404"):
                    return {
                        **{column: "" for column in OUTPUT_COLUMNS},
                        **inventory_row(candidate),
                        "final_url": final_url,
                        "http_status": 404,
                        "parse_status": "not_found",
                        "response_bytes": len(last_html.encode("utf-8", errors="ignore")),
                        "browser_attempts": attempt,
                        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                        "extraction_version": "rbc_archive_v1",
                    }
            last_error = "qrator_challenge" if "/__qrator/" in last_html else "browser_timeout"
            if last_error == "qrator_challenge":
                _BROWSER_LIMITER.cooldown(15.0 * attempt)
        except Exception as exc:
            last_error = f"webdriver_error: {type(exc).__name__}: {exc}"
            _close_browser()
        if attempt < _BROWSER_RETRIES:
            time.sleep(min(30.0, 2.0 ** (attempt - 1) + random.uniform(0.2, 0.8)))

    return {
        **{column: "" for column in OUTPUT_COLUMNS},
        **inventory_row(candidate),
        "final_url": final_url,
        "http_status": 401 if "qrator" in last_error else 0,
        "parse_status": "request_error",
        "response_bytes": len(last_html.encode("utf-8", errors="ignore")),
        "browser_attempts": _BROWSER_RETRIES,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "error": last_error,
        "extraction_version": "rbc_archive_v1",
    }


def create_browser_pool(
    *,
    processes: int,
    requests_per_second: float,
    timeout: float,
    retries: int,
) -> ProcessPoolExecutor:
    context = mp.get_context("spawn")
    next_allowed = context.Value("d", 0.0)
    blocked_until = context.Value("d", 0.0)
    rate_lock = context.Lock()
    return ProcessPoolExecutor(
        max_workers=processes,
        mp_context=context,
        initializer=initialize_browser_worker,
        initargs=(requests_per_second, timeout, retries, next_allowed, blocked_until, rate_lock),
    )


def atomic_write_rows(rows: list[dict[str, object]], path: Path, output_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    if output_format == "parquet":
        pd.DataFrame(rows).reindex(columns=OUTPUT_COLUMNS).to_parquet(temp, index=False, compression="zstd")
    elif output_format == "jsonl.gz":
        with gzip.open(temp, "wt", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps({column: row.get(column, "") for column in OUTPUT_COLUMNS}, ensure_ascii=False) + "\n")
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    temp.replace(path)


def part_is_complete(path: Path, expected_ids: list[str], output_format: str) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        if output_format == "parquet":
            rows = pd.read_parquet(path, columns=["requested_material_id", "parse_status"]).to_dict("records")
        else:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
    except Exception:
        return False
    actual_ids = [str(row.get("requested_material_id", "")) for row in rows]
    return actual_ids == expected_ids and not any(row.get("parse_status") == "request_error" for row in rows)


def write_inventory_part(
    candidates: list[dict[str, object]],
    path: Path,
    *,
    force: bool,
    output_format: str,
) -> dict[str, int]:
    expected_ids = [str(item["material_id"]) for item in candidates]
    if not force and part_is_complete(path, expected_ids, output_format):
        return {"skipped": len(candidates), "inventory_only": 0, "ok": 0, "metadata_only": 0, "not_found": 0, "errors": 0}
    atomic_write_rows([inventory_row(item) for item in candidates], path, output_format)
    return {"skipped": 0, "inventory_only": len(candidates), "ok": 0, "metadata_only": 0, "not_found": 0, "errors": 0}


def download_content_part(
    executor: ProcessPoolExecutor,
    candidates: list[dict[str, object]],
    path: Path,
    *,
    force: bool,
    output_format: str,
) -> dict[str, int]:
    expected_ids = [str(item["material_id"]) for item in candidates]
    if not force and part_is_complete(path, expected_ids, output_format):
        return {"skipped": len(candidates), "inventory_only": 0, "ok": 0, "metadata_only": 0, "not_found": 0, "errors": 0}
    futures = {executor.submit(_browser_candidate, item): str(item["material_id"]) for item in candidates}
    rows = [future.result() for future in as_completed(futures)]
    position = {material_id: index for index, material_id in enumerate(expected_ids)}
    rows.sort(key=lambda row: position[str(row["requested_material_id"])])
    # Persist terminal per-article failures with the rest of the part instead of
    # retrying the whole part forever.  part_is_complete intentionally treats a
    # part containing request_error as incomplete, so a later resumed run will
    # retry it while the current full-archive run can continue making progress.
    atomic_write_rows(rows, path, output_format)
    statuses = pd.Series([str(row.get("parse_status", "")) for row in rows])
    return {
        "skipped": 0,
        "inventory_only": 0,
        "ok": int(statuses.isin(["ok", "redirected_ok"]).sum()),
        "metadata_only": int(statuses.str.contains("metadata_only", regex=False).sum()),
        "not_found": int(statuses.eq("not_found").sum()),
        "errors": int(statuses.str.contains("error", regex=False).sum()),
    }
