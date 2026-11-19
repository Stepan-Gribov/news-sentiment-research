from __future__ import annotations

import gzip
import html as html_lib
import json
import multiprocessing as mp
import random
import re
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse
from xml.etree import ElementTree

import pandas as pd
import requests
from bs4 import BeautifulSoup


SITEMAP_INDEX_URL = "https://www.interfax.ru/SEO_SiteMapIndex.xml"
NUMBERED_SITEMAP_RE = re.compile(r"SEO_SiteMap_(\d+)\.xml$", re.I)
LATEST_SITEMAP_RE = re.compile(r"SEO_SiteMapLastChanges\.xml$", re.I)
ARTICLE_URL_RE = re.compile(
    r"^https?://(?:www\.)?interfax\.ru/(?P<section>[^/?#]+)/(?P<article_id>\d+)(?:[/?#]|$)",
    re.I,
)
ARTICLE_SECTIONS = {
    "business",
    "confederationscup2017",
    "culture",
    "digital",
    "euro2016",
    "interview",
    "moscow",
    "olymp2016",
    "olymp2018",
    "russia",
    "wc2018",
    "world",
}
RETRYABLE_HTTP_STATUSES = {403, 408, 425, 429, 500, 502, 503, 504}
TERMINAL_HTTP_STATUSES = {404, 410}


OUTPUT_COLUMNS = [
    "media",
    "article_id",
    "requested_article_id",
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
    "article_text_extracted",
    "text_length",
    "response_bytes",
    "sitemap_group",
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


def extract_article_identity(url: str) -> tuple[str, int] | None:
    match = ARTICLE_URL_RE.match(str(url).strip())
    if not match:
        return None
    section = match.group("section").lower()
    if section not in ARTICLE_SECTIONS:
        return None
    return section, int(match.group("article_id"))


def normalize_datetime(value: object) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return text


def _meta_contents(
    soup: BeautifulSoup,
    *,
    property_name: str | None = None,
    name: str | None = None,
) -> list[str]:
    selector = f'meta[property="{property_name}"]' if property_name else f'meta[name="{name}"]'
    return [normalize_text(node.get("content", "")) for node in soup.select(selector)]


def _meta_content(
    soup: BeautifulSoup,
    *,
    property_name: str | None = None,
    name: str | None = None,
) -> str:
    return next((value for value in _meta_contents(soup, property_name=property_name, name=name) if value), "")


def _find_article_json(value: object) -> dict | None:
    if isinstance(value, dict):
        kind = value.get("@type")
        kinds = kind if isinstance(kind, list) else [kind]
        if any(item in {"Article", "NewsArticle", "ReportageNewsArticle"} for item in kinds):
            return value
        for child in value.values():
            found = _find_article_json(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_article_json(child)
            if found:
                return found
    return None


def _news_article_json(soup: BeautifulSoup) -> dict:
    for node in soup.select('script[type="application/ld+json"]'):
        try:
            parsed = json.loads(node.get_text(strip=True))
        except (TypeError, json.JSONDecodeError):
            continue
        found = _find_article_json(parsed)
        if found:
            return found
    return {}


def _article_text(soup: BeautifulSoup) -> str:
    body = soup.select_one('article[itemprop="articleBody"]')
    if body:
        parts = [normalize_text(node.get_text(" ", strip=True)) for node in body.select("p")]
        parts = [part for part in parts if part]
        if parts:
            return "\n".join(parts)
        body_copy = BeautifulSoup(str(body), "html.parser")
        for node in body_copy.select("h1, script, style, noscript, template, svg, iframe, [itemprop='author'], [itemprop='publisher']"):
            node.decompose()
        return normalize_text(body_copy.get_text(" ", strip=True))
    return ""


def _author_names(article_json: dict, soup: BeautifulSoup) -> str:
    values: list[object] = []
    authors = article_json.get("author", [])
    if not isinstance(authors, list):
        authors = [authors]
    values.extend(author.get("name", "") if isinstance(author, dict) else author for author in authors)
    values.extend(
        node.get("content", "")
        for node in soup.select('[itemprop="author"] [itemprop="name"], meta[itemprop="author"]')
    )
    return join_unique(values)


def parse_article_html(html: str, requested_url: str) -> dict[str, object]:
    soup = BeautifulSoup(html, "html.parser")
    article_json = _news_article_json(soup)
    requested_identity = extract_article_identity(requested_url)

    canonical_node = soup.select_one('link[rel="canonical"]')
    canonical_url = normalize_text(canonical_node.get("href", "")) if canonical_node else ""
    canonical_url = canonical_url or _meta_content(soup, property_name="og:url") or requested_url
    canonical_identity = extract_article_identity(canonical_url) or requested_identity
    requested_id = requested_identity[1] if requested_identity else None
    article_id = canonical_identity[1] if canonical_identity else requested_id
    url_section = canonical_identity[0] if canonical_identity else (requested_identity[0] if requested_identity else "")

    h1 = soup.select_one('h1[itemprop="headline"]') or soup.select_one("h1")
    title = normalize_text(
        article_json.get("headline")
        or _meta_content(soup, property_name="og:title")
        or (h1.get_text(" ", strip=True) if h1 else "")
    )
    description = normalize_text(
        article_json.get("description")
        or _meta_content(soup, name="description")
        or _meta_content(soup, property_name="og:description")
    )
    published_at = normalize_datetime(
        article_json.get("datePublished")
        or _meta_content(soup, property_name="article:published_time")
    )
    modified_at = normalize_datetime(
        article_json.get("dateModified")
        or _meta_content(soup, property_name="article:modified_time")
    )
    section = normalize_text(
        article_json.get("articleSection")
        or _meta_content(soup, property_name="article:section")
    )
    tags = join_unique(_meta_contents(soup, property_name="article:tag"))
    text = _article_text(soup)
    publication_date = published_at[:10] if published_at else ""

    has_article_evidence = bool(article_json or published_at or text or (title and article_id))
    parse_status = "ok" if has_article_evidence else "not_article"
    if parse_status == "ok" and not text:
        parse_status = "metadata_only"
    if requested_id is not None and article_id != requested_id:
        parse_status = f"redirected_{parse_status}"

    return {
        "media": "interfax",
        "article_id": article_id,
        "requested_article_id": requested_id,
        "doc_id": article_id,
        "url": canonical_url,
        "link": canonical_url,
        "parse_status": parse_status,
        "timestamp": published_at,
        "published_at": published_at,
        "publication_date": publication_date,
        "date_source": "page_metadata" if published_at else "",
        "modified_at": modified_at,
        "title": title,
        "subtitle": description,
        "description": description,
        "authors": _author_names(article_json, soup),
        "category": section or url_section,
        "rubric": section,
        "section": section,
        "url_section": url_section,
        "region": "",
        "tags": tags,
        "article_text_extracted": text,
        "text_length": len(text),
        "extraction_version": "interfax_archive_v1",
    }


class LocalRateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self.interval = 1.0 / requests_per_second
        self._next_allowed = 0.0
        self._blocked_until = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                target = max(self._next_allowed, self._blocked_until)
                if now >= target:
                    self._next_allowed = now + self.interval
                    return
                delay = target - now
            time.sleep(delay)

    def cooldown(self, seconds: float) -> None:
        with self._lock:
            self._blocked_until = max(self._blocked_until, time.monotonic() + max(0.0, seconds))


class SharedRateLimiter:
    def __init__(self, requests_per_second: float, next_allowed, blocked_until, lock) -> None:
        self.interval = 1.0 / requests_per_second
        self.next_allowed = next_allowed
        self.blocked_until = blocked_until
        self.lock = lock

    def wait(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                target = max(self.next_allowed.value, self.blocked_until.value)
                if now >= target:
                    self.next_allowed.value = now + self.interval
                    return
                delay = target - now
            time.sleep(delay)

    def cooldown(self, seconds: float) -> None:
        with self.lock:
            self.blocked_until.value = max(
                self.blocked_until.value,
                time.monotonic() + max(0.0, seconds),
            )


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
        limiter,
        retries: int,
        timeout: float,
        user_agent: str,
    ) -> None:
        self.limiter = limiter
        self.retries = max(1, retries)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
            }
        )

    def get(self, url: str) -> HttpResult:
        last_error = ""
        last_status = 0
        last_url = url
        for attempt in range(1, self.retries + 1):
            self.limiter.wait()
            try:
                response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
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
                if response.status_code == 429:
                    delay = max(20.0, delay)
                    self.limiter.cooldown(delay)
                elif response.status_code == 403:
                    delay = max(15.0, delay)
                    self.limiter.cooldown(delay)
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


def sitemap_group(url: str) -> tuple[str, int | None] | None:
    path = urlparse(url).path.rsplit("/", 1)[-1]
    match = NUMBERED_SITEMAP_RE.search(path)
    if match:
        number = int(match.group(1))
        return f"map-{number:03d}", number
    if LATEST_SITEMAP_RE.search(path):
        return "latest", None
    return None


def discover_sitemap_urls(
    client: HttpClient,
    *,
    min_map: int,
    max_map: int | None,
    include_latest: bool,
    newest_first: bool,
    index_url: str = SITEMAP_INDEX_URL,
) -> list[tuple[str, str]]:
    result = client.get(index_url)
    if result.status != 200:
        raise RuntimeError(f"Cannot download sitemap index: {result.status} {result.error}")
    numbered: list[tuple[int, str, str]] = []
    latest: list[tuple[str, str]] = []
    for row in parse_sitemap_locations(result.text):
        group = sitemap_group(row["loc"])
        if not group:
            continue
        group_name, number = group
        if number is None:
            if include_latest:
                latest.append((group_name, row["loc"]))
        elif number >= min_map and (max_map is None or number <= max_map):
            numbered.append((number, group_name, row["loc"]))
    numbered.sort(key=lambda item: item[0], reverse=not newest_first)
    groups = [(name, url) for _, name, url in numbered]
    return (latest + groups) if newest_first else (groups + latest)


def discover_group_candidates(client: HttpClient, group: str, url: str) -> list[dict[str, object]]:
    result = client.get(url)
    if result.status != 200:
        raise RuntimeError(f"Cannot download sitemap {group}: {result.status} {result.error}")
    by_id: dict[int, dict[str, object]] = {}
    for row in parse_sitemap_locations(result.text):
        identity = extract_article_identity(row["loc"])
        if identity is None:
            continue
        section, article_id = identity
        by_id[article_id] = {
            "article_id": article_id,
            "url": row["loc"],
            "url_section": section,
            "sitemap_group": group,
            "sitemap_lastmod": row.get("lastmod", ""),
        }
    return [by_id[key] for key in sorted(by_id)]


def download_candidate(client: HttpClient, candidate: dict[str, object]) -> dict[str, object]:
    requested_url = str(candidate["url"])
    requested_id = int(candidate["article_id"])
    result = client.get(requested_url)
    common = {
        "requested_article_id": requested_id,
        "article_id": requested_id,
        "doc_id": requested_id,
        "url": requested_url,
        "link": requested_url,
        "final_url": result.final_url,
        "http_status": result.status,
        "url_section": candidate.get("url_section", ""),
        "response_bytes": result.response_bytes,
        "sitemap_group": candidate.get("sitemap_group", ""),
        "sitemap_lastmod": candidate.get("sitemap_lastmod", ""),
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "error": result.error,
    }
    if result.status in TERMINAL_HTTP_STATUSES:
        return {
            **{column: "" for column in OUTPUT_COLUMNS},
            **common,
            "media": "interfax",
            "parse_status": "not_found",
            "text_length": 0,
            "extraction_version": "interfax_archive_v1",
        }
    if result.status != 200 or not result.text:
        retryable_failure = result.status == 0 or result.status in RETRYABLE_HTTP_STATUSES
        return {
            **{column: "" for column in OUTPUT_COLUMNS},
            **common,
            "media": "interfax",
            "parse_status": "request_error" if retryable_failure else "http_error",
            "text_length": 0,
            "extraction_version": "interfax_archive_v1",
        }
    try:
        return {**common, **parse_article_html(result.text, requested_url)}
    except Exception as exc:
        return {
            **{column: "" for column in OUTPUT_COLUMNS},
            **common,
            "media": "interfax",
            "parse_status": "parse_error",
            "text_length": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "extraction_version": "interfax_archive_v1",
        }


_WORKER_CLIENT: HttpClient | None = None


def initialize_worker(
    requests_per_second: float,
    retries: int,
    timeout: float,
    user_agent: str,
    next_allowed,
    blocked_until,
    rate_lock,
) -> None:
    global _WORKER_CLIENT
    limiter = SharedRateLimiter(requests_per_second, next_allowed, blocked_until, rate_lock)
    _WORKER_CLIENT = HttpClient(
        limiter=limiter,
        retries=retries,
        timeout=timeout,
        user_agent=user_agent,
    )


def download_candidate_in_worker(candidate: dict[str, object]) -> dict[str, object]:
    if _WORKER_CLIENT is None:
        raise RuntimeError("Interfax process worker was not initialized")
    return download_candidate(_WORKER_CLIENT, candidate)


def create_process_pool(
    *,
    processes: int,
    requests_per_second: float,
    retries: int,
    timeout: float,
    user_agent: str,
) -> ProcessPoolExecutor:
    context = mp.get_context("spawn")
    next_allowed = context.Value("d", 0.0)
    blocked_until = context.Value("d", 0.0)
    rate_lock = context.Lock()
    # Shared synchronization primitives stay alive through the executor's
    # initargs and enforce one aggregate request rate across all processes.
    return ProcessPoolExecutor(
        max_workers=processes,
        mp_context=context,
        initializer=initialize_worker,
        initargs=(
            requests_per_second,
            retries,
            timeout,
            user_agent,
            next_allowed,
            blocked_until,
            rate_lock,
        ),
    )


def atomic_write_rows(rows: list[dict[str, object]], path: Path, output_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    if output_format == "parquet":
        pd.DataFrame(rows).reindex(columns=OUTPUT_COLUMNS).to_parquet(
            temp,
            index=False,
            compression="zstd",
        )
    elif output_format == "jsonl.gz":
        with gzip.open(temp, "wt", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        {column: row.get(column, "") for column in OUTPUT_COLUMNS},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    temp.replace(path)


def part_is_complete(path: Path, expected_ids: list[int], output_format: str) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        if output_format == "parquet":
            rows = pd.read_parquet(path, columns=["requested_article_id", "parse_status"]).to_dict("records")
        else:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
    except Exception:
        return False
    try:
        actual_ids = [int(row["requested_article_id"]) for row in rows]
    except (KeyError, TypeError, ValueError):
        return False
    return actual_ids == expected_ids and not any(row.get("parse_status") == "request_error" for row in rows)


def chunked(values: list[dict[str, object]], size: int) -> Iterable[list[dict[str, object]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def download_part(
    executor: ProcessPoolExecutor,
    candidates: list[dict[str, object]],
    path: Path,
    *,
    force: bool,
    output_format: str,
) -> dict[str, int]:
    expected_ids = [int(item["article_id"]) for item in candidates]
    if not force and part_is_complete(path, expected_ids, output_format):
        return {"skipped": len(candidates), "ok": 0, "metadata_only": 0, "not_found": 0, "errors": 0}

    rows: list[dict[str, object]] = []
    futures = {executor.submit(download_candidate_in_worker, item): int(item["article_id"]) for item in candidates}
    for future in as_completed(futures):
        rows.append(future.result())
    position = {article_id: index for index, article_id in enumerate(expected_ids)}
    rows.sort(key=lambda row: position[int(row["requested_article_id"])])

    transient = [row for row in rows if row.get("parse_status") == "request_error"]
    if transient:
        ids = ", ".join(str(row["requested_article_id"]) for row in transient[:8])
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
