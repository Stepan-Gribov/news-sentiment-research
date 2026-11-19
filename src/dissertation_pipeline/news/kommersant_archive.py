from __future__ import annotations

import html as html_lib
import gzip
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree

import pandas as pd
import requests
from bs4 import BeautifulSoup


SITEMAP_INDEX_URL = "https://www.kommersant.ru/sitemaps/all_docs_sitemap.xml"
DOC_URL_TEMPLATE = "https://www.kommersant.ru/doc/{doc_id}"
DOC_ID_RE = re.compile(r"https?://(?:www\.)?kommersant\.ru/doc/(\d+)(?:[/?#]|$)", re.I)
MONTH_RE = re.compile(r"sitemap-(\d{4})-(\d{2})\.xml$")
RETRYABLE_HTTP_STATUSES = {403, 408, 425, 429, 500, 502, 503, 504}
TERMINAL_HTTP_STATUSES = {404, 410}


OUTPUT_COLUMNS = [
    "media",
    "doc_id",
    "requested_doc_id",
    "url",
    "link",
    "final_url",
    "http_status",
    "parse_status",
    "timestamp",
    "published_at",
    "modified_at",
    "title",
    "subtitle",
    "description",
    "authors",
    "category",
    "rubric",
    "section",
    "publishing",
    "region",
    "themes",
    "article_text_extracted",
    "text_length",
    "response_bytes",
    "sitemap_month",
    "sitemap_lastmod",
    "fetched_at_utc",
    "error",
    "extraction_version",
]


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = html_lib.unescape(str(value)).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def join_unique(values: Iterable[object], separator: str = "; ") -> str:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = normalize_text(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return separator.join(result)


def extract_doc_id(url: str) -> int | None:
    match = DOC_ID_RE.search(str(url))
    return int(match.group(1)) if match else None


def _meta_content(soup: BeautifulSoup, *, property_name: str | None = None, name: str | None = None) -> str:
    selector = f'meta[property="{property_name}"]' if property_name else f'meta[name="{name}"]'
    node = soup.select_one(selector)
    return normalize_text(node.get("content", "")) if node else ""


def _news_article_json(soup: BeautifulSoup) -> dict:
    def find_news_article(value: object) -> dict | None:
        if isinstance(value, dict):
            kind = value.get("@type")
            kinds = kind if isinstance(kind, list) else [kind]
            if any(item in {"Article", "NewsArticle", "ReportageNewsArticle"} for item in kinds):
                return value
            for child in value.values():
                found = find_news_article(child)
                if found:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = find_news_article(child)
                if found:
                    return found
        return None

    for node in soup.select('script[type="application/ld+json"]'):
        try:
            parsed = json.loads(node.get_text(strip=True))
        except (TypeError, json.JSONDecodeError):
            continue
        found = find_news_article(parsed)
        if found:
            return found
    return {}


def _article_text(soup: BeautifulSoup) -> str:
    # New pages expose clean paragraphs. Pull quotes and the author footer are
    # intentionally excluded because they duplicate body text or metadata.
    nodes = [
        node
        for node in soup.select(".article_text_wrapper .doc__text")
        if "document_authors" not in node.get("class", [])
    ]
    if nodes:
        return "\n".join(
            part for part in (normalize_text(node.get_text(" ", strip=True)) for node in nodes) if part
        )

    # The 1990s archive uses FONT/BR markup instead of paragraphs.
    wrapper = soup.select_one(".article_text_wrapper") or soup.select_one(".doc__body")
    if not wrapper:
        return ""
    for node in wrapper.select("script, style, noscript, template, svg, iframe, .document_authors"):
        node.decompose()
    parts = [normalize_text(part) for part in wrapper.get_text("\n").splitlines()]
    return "\n".join(part for part in parts if part)


def _authors(article_node, soup: BeautifulSoup, article_json: dict) -> str:
    values: list[object] = []
    if article_node:
        raw_authors = article_node.get("data-article-authors", "")
        try:
            parsed_authors = json.loads(raw_authors) if raw_authors else []
        except json.JSONDecodeError:
            parsed_authors = []
        if isinstance(parsed_authors, list):
            for author in parsed_authors:
                values.append(author.get("name", "") if isinstance(author, dict) else author)
        values.append(article_node.get("data-analytics-authors", ""))
    values.extend(node.get_text(" ", strip=True) for node in soup.select(".document_authors"))

    # JSON-LD often names the publisher ("Коммерсантъ") as author. Keep it only
    # when no personal byline exists, so the field is never silently invented.
    personal = join_unique(values)
    if personal:
        return personal
    json_authors = article_json.get("author", [])
    if not isinstance(json_authors, list):
        json_authors = [json_authors]
    return join_unique(
        author.get("name", "") if isinstance(author, dict) else author for author in json_authors
    )


def parse_article_html(html: str, requested_url: str) -> dict[str, object]:
    soup = BeautifulSoup(html, "html.parser")
    article_node = soup.select_one("article.js-article") or soup.select_one("article.doc")
    article_json = _news_article_json(soup)

    canonical_node = soup.select_one('link[rel="canonical"]')
    canonical_url = normalize_text(canonical_node.get("href", "")) if canonical_node else requested_url
    requested_doc_id = extract_doc_id(requested_url)
    canonical_doc_id = extract_doc_id(canonical_url) or requested_doc_id

    attrs = article_node.attrs if article_node else {}
    title_node = soup.select_one("h1.doc_header__name") or soup.select_one("h1")
    title = normalize_text(
        attrs.get("data-article-title")
        or article_json.get("headline")
        or _meta_content(soup, property_name="og:title")
        or (title_node.get_text(" ", strip=True) if title_node else "")
    )
    subtitle = normalize_text(
        article_json.get("alternativeHeadline")
        or attrs.get("data-article-description")
        or _meta_content(soup, property_name="og:description")
    )
    description = normalize_text(
        article_json.get("description")
        or _meta_content(soup, name="description")
        or subtitle
    )
    published_at = normalize_text(
        article_json.get("datePublished")
        or _meta_content(soup, property_name="article:published_time")
        or attrs.get("data-analytics-date")
    )
    modified_at = normalize_text(
        article_json.get("dateModified") or _meta_content(soup, property_name="article:modified_time")
    )
    rubric = normalize_text(attrs.get("data-article-rubric-name") or attrs.get("data-analytics-rubrics"))
    section = normalize_text(attrs.get("data-article-section-name"))
    categories = normalize_text(attrs.get("data-article-categories"))
    text = _article_text(soup)

    has_article_evidence = bool(article_node or article_json or published_at or text)
    parse_status = "ok" if has_article_evidence else "not_article"
    if parse_status == "ok" and not text:
        parse_status = "metadata_only"
    if canonical_doc_id != requested_doc_id:
        parse_status = f"redirected_{parse_status}"

    return {
        "media": "kommersant",
        "doc_id": canonical_doc_id,
        "requested_doc_id": requested_doc_id,
        "url": canonical_url,
        "link": canonical_url,
        "parse_status": parse_status,
        "timestamp": published_at,
        "published_at": published_at,
        "modified_at": modified_at,
        "title": title,
        "subtitle": subtitle,
        "description": description,
        "authors": _authors(article_node, soup, article_json),
        "category": categories or rubric or section,
        "rubric": rubric,
        "section": section,
        "publishing": normalize_text(
            attrs.get("data-article-publishing-name") or attrs.get("data-analytics-publishing")
        ),
        "region": normalize_text(attrs.get("data-article-region-name") or attrs.get("data-analytics-region")),
        "themes": normalize_text(attrs.get("data-analytics-themes")),
        "article_text_extracted": text,
        "text_length": len(text),
        "extraction_version": "kommersant_archive_v1",
    }


class RateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self.interval = 1.0 / requests_per_second
        self._next_allowed = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + self.interval
        if delay:
            time.sleep(delay)


@dataclass
class HttpResult:
    status: int
    final_url: str
    text: str
    response_bytes: int
    error: str


class HttpClient:
    def __init__(
        self,
        *,
        requests_per_second: float,
        retries: int,
        timeout: float,
        user_agent: str,
    ) -> None:
        self.limiter = RateLimiter(requests_per_second)
        self.retries = max(1, retries)
        self.timeout = timeout
        self.user_agent = user_agent
        self._local = threading.local()

    def _session(self) -> requests.Session:
        if not hasattr(self._local, "session"):
            session = requests.Session()
            session.headers.update(
                {
                    "User-Agent": self.user_agent,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.5",
                    "Accept-Encoding": "gzip, deflate",
                }
            )
            self._local.session = session
        return self._local.session

    def get(self, url: str) -> HttpResult:
        last_error = ""
        last_status = 0
        last_url = url
        for attempt in range(1, self.retries + 1):
            self.limiter.wait()
            try:
                response = self._session().get(url, timeout=self.timeout, allow_redirects=True)
                last_status = response.status_code
                last_url = response.url
                if response.status_code in TERMINAL_HTTP_STATUSES:
                    return HttpResult(response.status_code, response.url, "", len(response.content), "")
                if response.status_code == 200:
                    response.encoding = response.encoding or response.apparent_encoding or "utf-8"
                    return HttpResult(200, response.url, response.text, len(response.content), "")
                last_error = f"HTTP {response.status_code}"
                if response.status_code not in RETRYABLE_HTTP_STATUSES:
                    return HttpResult(response.status_code, response.url, "", len(response.content), last_error)
                retry_after = response.headers.get("Retry-After", "")
                try:
                    delay = min(300.0, float(retry_after))
                except ValueError:
                    delay = min(60.0, 2.0 ** (attempt - 1) + random.uniform(0.1, 0.8))
                time.sleep(delay)
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.retries:
                    time.sleep(min(60.0, 2.0 ** (attempt - 1) + random.uniform(0.1, 0.8)))
        return HttpResult(last_status, last_url, "", 0, last_error or "request failed")


def parse_sitemap_locations(xml_text: str) -> list[dict[str, str]]:
    root = ElementTree.fromstring(xml_text.lstrip("\ufeff"))
    rows: list[dict[str, str]] = []
    for node in root:
        values = {child.tag.rsplit("}", 1)[-1]: normalize_text(child.text) for child in node}
        if values.get("loc"):
            rows.append(values)
    return rows


def sitemap_month(url: str) -> str | None:
    match = MONTH_RE.search(url)
    return f"{match.group(1)}-{match.group(2)}" if match else None


def discover_sitemap_urls(
    client: HttpClient,
    *,
    start_month: str,
    end_month: str,
    index_url: str = SITEMAP_INDEX_URL,
) -> list[tuple[str, str]]:
    result = client.get(index_url)
    if result.status != 200:
        raise RuntimeError(f"Cannot download sitemap index: {result.status} {result.error}")
    selected: list[tuple[str, str]] = []
    for row in parse_sitemap_locations(result.text):
        month = sitemap_month(row["loc"])
        if month and start_month <= month <= end_month:
            selected.append((month, row["loc"]))
    return sorted(selected)


def discover_month_candidates(client: HttpClient, month: str, url: str) -> list[dict[str, object]]:
    result = client.get(url)
    if result.status != 200:
        raise RuntimeError(f"Cannot download sitemap {month}: {result.status} {result.error}")
    by_id: dict[int, dict[str, object]] = {}
    for row in parse_sitemap_locations(result.text):
        doc_id = extract_doc_id(row["loc"])
        if doc_id is None:
            continue
        by_id[doc_id] = {
            "doc_id": doc_id,
            "url": DOC_URL_TEMPLATE.format(doc_id=doc_id),
            "sitemap_month": month,
            "sitemap_lastmod": row.get("lastmod", ""),
        }
    return [by_id[key] for key in sorted(by_id)]


def download_candidate(client: HttpClient, candidate: dict[str, object]) -> dict[str, object]:
    requested_url = str(candidate["url"])
    result = client.get(requested_url)
    common = {
        "requested_doc_id": int(candidate["doc_id"]),
        "doc_id": int(candidate["doc_id"]),
        "url": requested_url,
        "link": requested_url,
        "final_url": result.final_url,
        "http_status": result.status,
        "response_bytes": result.response_bytes,
        "sitemap_month": candidate.get("sitemap_month", ""),
        "sitemap_lastmod": candidate.get("sitemap_lastmod", ""),
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "error": result.error,
    }
    if result.status in TERMINAL_HTTP_STATUSES:
        return {
            **{column: "" for column in OUTPUT_COLUMNS},
            **common,
            "media": "kommersant",
            "parse_status": "not_found",
            "text_length": 0,
            "extraction_version": "kommersant_archive_v1",
        }
    if result.status != 200 or not result.text:
        redirected_outside_kommersant = (
            result.final_url != requested_url and extract_doc_id(result.final_url) is None
        )
        retryable_failure = result.status == 0 or result.status in RETRYABLE_HTTP_STATUSES
        return {
            **{column: "" for column in OUTPUT_COLUMNS},
            **common,
            "media": "kommersant",
            "parse_status": (
                "request_error"
                if retryable_failure
                else "external_redirect"
                if redirected_outside_kommersant
                else "http_error"
            ),
            "text_length": 0,
            "extraction_version": "kommersant_archive_v1",
        }
    try:
        parsed = parse_article_html(result.text, requested_url)
        return {**common, **parsed}
    except Exception as exc:  # Keep the response-level fact without storing HTML.
        return {
            **{column: "" for column in OUTPUT_COLUMNS},
            **common,
            "media": "kommersant",
            "parse_status": "parse_error",
            "text_length": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "extraction_version": "kommersant_archive_v1",
        }


def atomic_write_rows(rows: list[dict[str, object]], path: Path, output_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    if output_format == "parquet":
        frame = pd.DataFrame(rows).reindex(columns=OUTPUT_COLUMNS)
        frame.to_parquet(temp, index=False, compression="zstd")
    elif output_format == "jsonl.gz":
        with gzip.open(temp, "wt", encoding="utf-8") as handle:
            for row in rows:
                normalized = {column: row.get(column, "") for column in OUTPUT_COLUMNS}
                handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    temp.replace(path)


def part_is_complete(path: Path, expected_ids: list[int], output_format: str) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        if output_format == "parquet":
            frame = pd.read_parquet(path, columns=["requested_doc_id", "parse_status"])
            rows = frame.to_dict("records")
        else:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
    except Exception:
        return False
    try:
        actual_ids = [int(row["requested_doc_id"]) for row in rows]
    except (KeyError, TypeError, ValueError):
        return False
    if actual_ids != expected_ids:
        return False
    return not any(row.get("parse_status") == "request_error" for row in rows)


def chunked(values: list[dict[str, object]], size: int) -> Iterable[list[dict[str, object]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def download_part(
    client: HttpClient,
    candidates: list[dict[str, object]],
    path: Path,
    *,
    workers: int,
    force: bool,
    output_format: str,
) -> dict[str, int]:
    expected_ids = [int(item["doc_id"]) for item in candidates]
    if not force and part_is_complete(path, expected_ids, output_format):
        return {"skipped": len(candidates), "ok": 0, "metadata_only": 0, "not_found": 0, "errors": 0}

    rows: list[dict[str, object]] = []
    if workers == 1:
        rows = [download_candidate(client, item) for item in candidates]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_id = {executor.submit(download_candidate, client, item): int(item["doc_id"]) for item in candidates}
            for future in as_completed(future_to_id):
                rows.append(future.result())
        rows.sort(key=lambda row: int(row["requested_doc_id"]))

    # A transient failure leaves the part unwritten. On restart only this small
    # chunk is retried; successfully written chunks remain untouched.
    transient = [row for row in rows if row.get("parse_status") == "request_error"]
    if transient:
        ids = ", ".join(str(row["requested_doc_id"]) for row in transient[:8])
        raise RuntimeError(f"Transient request errors in {path.name}: {ids}")

    atomic_write_rows(rows, path, output_format)
    statuses = pd.Series([str(row.get("parse_status", "")) for row in rows])
    return {
        "skipped": 0,
        "ok": int(statuses.isin(["ok", "redirected_ok"]).sum()),
        "metadata_only": int(statuses.str.contains("metadata_only", regex=False).sum()),
        "not_found": int(statuses.eq("not_found").sum()),
        "errors": int(statuses.str.contains("error", regex=False).sum()),
    }
