from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


YANDEX_SEARCH_URL = "https://searchapi.api.cloud.yandex.net/v2/web/search"
YANDEX_RESULT_CAP = 250


@dataclass(frozen=True)
class QueryProfile:
    key: str
    expression: str
    relevance_terms: tuple[str, ...]


DEFAULT_QUERY_PROFILES = {
    "corporate_news": QueryProfile(
        key="corporate_news",
        expression=(
            '("корпоративные новости" | "пресс-релиз" | "совет директоров" | '
            "дивиденды | акционеры | стратегия | инвестиции | сделка | контракт)"
        ),
        relevance_terms=(
            "корпоративные новости",
            "пресс-релиз",
            "совет директоров",
            "дивиденд",
            "акционер",
            "стратег",
            "инвестиц",
            "сделк",
            "контракт",
        ),
    ),
    "disclosure": QueryProfile(
        key="disclosure",
        expression=(
            '("раскрытие информации" | "существенный факт" | отчетность | эмитент | '
            "МСФО | РСБУ | облигации | рейтинг | дивиденды)"
        ),
        relevance_terms=(
            "раскрытие информации",
            "существенный факт",
            "отчетност",
            "эмитент",
            "мсфо",
            "рсбу",
            "облигац",
            "рейтинг",
            "дивиденд",
        ),
    ),
}


@dataclass(frozen=True)
class DateWindow:
    start: date
    end: date
    depth: int = 0

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1


@dataclass
class SearchPage:
    rows: list[dict]
    total_found: int | None
    raw_xml: bytes
    cache_path: Path | None = None


def parse_iso_date(value: str) -> date:
    return date.fromisoformat(value)


def monthly_windows(date_from: str | date, date_to_exclusive: str | date) -> list[DateWindow]:
    start = parse_iso_date(date_from) if isinstance(date_from, str) else date_from
    stop = parse_iso_date(date_to_exclusive) if isinstance(date_to_exclusive, str) else date_to_exclusive
    if start >= stop:
        raise ValueError(f"date_from must be before exclusive date_to: {start} >= {stop}")

    windows: list[DateWindow] = []
    current = start
    while current < stop:
        if current.month == 12:
            next_month = date(current.year + 1, 1, 1)
        else:
            next_month = date(current.year, current.month + 1, 1)
        next_start = min(next_month, stop)
        windows.append(DateWindow(current, next_start - timedelta(days=1)))
        current = next_start
    return windows


def split_window(window: DateWindow) -> tuple[DateWindow, DateWindow]:
    if window.days <= 1:
        raise ValueError("a one-day window cannot be split")
    left_days = window.days // 2
    left_end = window.start + timedelta(days=left_days - 1)
    right_start = left_end + timedelta(days=1)
    return (
        DateWindow(window.start, left_end, window.depth + 1),
        DateWindow(right_start, window.end, window.depth + 1),
    )


def build_query(company: str, profile: QueryProfile, window: DateWindow) -> str:
    company = company.strip()
    if not company:
        raise ValueError("company must not be empty")
    date_filter = f"date:{window.start:%Y%m%d}..{window.end:%Y%m%d}"
    query = f"{company} {profile.expression} {date_filter}"
    if len(query) > 400:
        raise ValueError(f"Yandex query exceeds 400 characters: {len(query)}")
    return query


def _element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def _parse_int(text: str | None) -> int | None:
    if not text:
        return None
    digits = re.sub(r"[^0-9]", "", text)
    return int(digits) if digits else None


def parse_search_xml(raw_xml: bytes, *, page: int, groups_on_page: int) -> SearchPage:
    try:
        root = ET.fromstring(raw_xml)
    except ET.ParseError as exc:
        raise ValueError(f"invalid Yandex Search XML: {exc}") from exc

    error = root.find(".//error")
    if error is not None:
        code = error.attrib.get("code", "unknown")
        raise RuntimeError(f"Yandex Search XML error {code}: {_element_text(error)}")

    total_found = None
    found_nodes = root.findall(".//found") + root.findall(".//found-docs")
    for node in found_nodes:
        value = _parse_int(_element_text(node))
        if value is None:
            continue
        if node.attrib.get("priority") == "all":
            total_found = value
            break
        if total_found is None:
            total_found = value

    rows: list[dict] = []
    for index, doc in enumerate(root.findall(".//doc"), start=1):
        link = _element_text(doc.find("./url"))
        if not link:
            continue
        passages = [
            _element_text(passage)
            for passage in doc.findall("./passages/passage")
            if _element_text(passage)
        ]
        rows.append(
            {
                "rank": page * groups_on_page + index,
                "link": link,
                "domain": _element_text(doc.find("./domain")) or urlparse(link).netloc,
                "title": _element_text(doc.find("./title")),
                "snippet": " ".join(passages),
                "engine_date": _element_text(doc.find("./modtime")),
                "mime_type": _element_text(doc.find("./mime-type")),
                "saved_copy_url": _element_text(doc.find("./saved-copy-url")),
            }
        )
    return SearchPage(rows=rows, total_found=total_found, raw_xml=raw_xml)


def _atomic_write_bytes(data: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp_path.write_bytes(data)
    tmp_path.replace(path)


class YandexSearchClient:
    def __init__(
        self,
        *,
        api_key: str,
        folder_id: str,
        groups_on_page: int = 100,
        timeout_seconds: int = 45,
        request_delay_seconds: float = 0.2,
        force: bool = False,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key must not be empty")
        if not folder_id.strip():
            raise ValueError("folder_id must not be empty")
        if not 1 <= groups_on_page <= 100:
            raise ValueError("groups_on_page must be between 1 and 100 for XML output")

        self.api_key = api_key
        self.folder_id = folder_id
        self.groups_on_page = groups_on_page
        self.timeout_seconds = timeout_seconds
        self.request_delay_seconds = request_delay_seconds
        self.force = force

        retry = Retry(
            total=4,
            connect=4,
            read=4,
            status=4,
            backoff_factor=1.0,
            status_forcelist=(408, 425, 429, 500, 502, 503, 504),
            allowed_methods=("POST",),
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.headers.update(
            {
                "Authorization": f"Api-Key {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    def request_body(self, query_text: str, page: int) -> dict:
        return {
            "query": {
                "searchType": "SEARCH_TYPE_RU",
                "queryText": query_text,
                "familyMode": "FAMILY_MODE_MODERATE",
                "page": str(page),
                "fixTypoMode": "FIX_TYPO_MODE_OFF",
            },
            "sortSpec": {
                "sortMode": "SORT_MODE_BY_RELEVANCE",
                "sortOrder": "SORT_ORDER_DESC",
            },
            "groupSpec": {
                "groupMode": "GROUP_MODE_FLAT",
                "groupsOnPage": str(self.groups_on_page),
                "docsInGroup": "1",
            },
            "maxPassages": "3",
            "region": "225",
            "l10N": "LOCALIZATION_RU",
            "folderId": self.folder_id,
            "responseFormat": "FORMAT_XML",
        }

    def search_page(self, query_text: str, page: int, cache_path: Path) -> SearchPage:
        if not self.force and cache_path.exists() and cache_path.stat().st_size > 0:
            try:
                parsed = parse_search_xml(
                    cache_path.read_bytes(),
                    page=page,
                    groups_on_page=self.groups_on_page,
                )
                parsed.cache_path = cache_path
                return parsed
            except Exception as exc:
                print(f"Ignoring unreadable Yandex cache {cache_path.name}: {exc}", flush=True)

        response = self.session.post(
            YANDEX_SEARCH_URL,
            json=self.request_body(query_text, page),
            timeout=self.timeout_seconds,
        )
        if not response.ok:
            body = response.text[:1000]
            raise RuntimeError(f"Yandex Search HTTP {response.status_code}: {body}")
        payload = response.json()
        raw_data = payload.get("rawData")
        if not isinstance(raw_data, str) or not raw_data:
            raise RuntimeError(f"Yandex Search response has no rawData: {json.dumps(payload)[:1000]}")
        try:
            raw_xml = base64.b64decode(raw_data, validate=True)
        except Exception as exc:
            raise RuntimeError("Yandex Search rawData is not valid Base64") from exc

        parsed = parse_search_xml(raw_xml, page=page, groups_on_page=self.groups_on_page)
        _atomic_write_bytes(raw_xml, cache_path)
        parsed.cache_path = cache_path
        if self.request_delay_seconds > 0:
            time.sleep(self.request_delay_seconds)
        return parsed


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower().replace("ё", "е")).strip()


def score_relevance(row: dict, company: str, profile: QueryProfile) -> tuple[int, list[str]]:
    title = _normalize(row.get("title", ""))
    snippet = _normalize(row.get("snippet", ""))
    domain = _normalize(row.get("domain", ""))
    company_token = _normalize(company)

    score = 0
    if company_token and company_token in title:
        score += 3
    elif company_token and company_token in snippet:
        score += 1

    matched: list[str] = []
    for raw_term in profile.relevance_terms:
        term = _normalize(raw_term)
        if term in title:
            score += 2
            matched.append(raw_term)
        elif term in snippet:
            score += 1
            matched.append(raw_term)

    if domain.endswith("gazprom.ru") or domain.endswith("e-disclosure.ru"):
        score += 2
    return score, matched


class YandexWindowCollector:
    def __init__(
        self,
        client: YandexSearchClient,
        *,
        raw_dir: Path,
        min_window_days: int = 1,
    ) -> None:
        if min_window_days < 1:
            raise ValueError("min_window_days must be at least 1")
        self.client = client
        self.raw_dir = raw_dir
        self.min_window_days = min_window_days

    def _cache_path(self, profile: QueryProfile, window: DateWindow, query: str, page: int) -> Path:
        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
        return (
            self.raw_dir
            / profile.key
            / (
                f"{window.start:%Y%m%d}_{window.end:%Y%m%d}"
                f"_d{window.depth}_{query_hash}_p{page}.xml"
            )
        )

    def collect_window(
        self,
        *,
        company: str,
        profile: QueryProfile,
        window: DateWindow,
    ) -> tuple[list[dict], dict]:
        query = build_query(company, profile, window)
        max_pages = math.ceil(YANDEX_RESULT_CAP / self.client.groups_on_page)
        rows: list[dict] = []
        total_found_values: list[int] = []
        last_page_size = 0
        pages_requested = 0

        for page_number in range(max_pages):
            cache_path = self._cache_path(profile, window, query, page_number)
            page = self.client.search_page(query, page_number, cache_path)
            pages_requested += 1
            last_page_size = len(page.rows)
            if page.total_found is not None:
                total_found_values.append(page.total_found)

            for row in page.rows:
                relevance_score, matched_terms = score_relevance(row, company, profile)
                row.update(
                    {
                        "company": company,
                        "query_profile": profile.key,
                        "search_query": query,
                        "window_from": window.start.isoformat(),
                        "window_to": window.end.isoformat(),
                        "partition_depth": window.depth,
                        "page": page_number,
                        "raw_response_path": str(page.cache_path or cache_path),
                        "local_relevance_score": relevance_score,
                        "matched_terms": "; ".join(matched_terms),
                    }
                )
                rows.append(row)

            if last_page_size < self.client.groups_on_page:
                break
            unique_count = len({row["link"] for row in rows})
            total_found = max(total_found_values) if total_found_values else None
            if total_found is not None and unique_count >= min(total_found, YANDEX_RESULT_CAP):
                break

        unique_count = len({row["link"] for row in rows})
        total_found = max(total_found_values) if total_found_values else None
        saturated = bool(
            unique_count >= YANDEX_RESULT_CAP
            or (total_found is not None and total_found > YANDEX_RESULT_CAP)
            or (pages_requested == max_pages and last_page_size >= self.client.groups_on_page)
        )
        diagnostic = {
            "company": company,
            "query_profile": profile.key,
            "search_query": query,
            "window_from": window.start.isoformat(),
            "window_to": window.end.isoformat(),
            "partition_depth": window.depth,
            "window_days": window.days,
            "pages_requested": pages_requested,
            "rows_returned": len(rows),
            "unique_links": unique_count,
            "engine_total_found": total_found,
            "saturated": saturated,
            "split": saturated and window.days > self.min_window_days,
            "leaf_saturated": saturated and window.days <= self.min_window_days,
        }
        print(
            f"Yandex {profile.key}: {window.start}..{window.end} "
            f"pages={pages_requested} links={unique_count} total={total_found} "
            f"saturated={saturated}",
            flush=True,
        )
        return rows, diagnostic

    def collect_adaptive(
        self,
        *,
        company: str,
        profile: QueryProfile,
        window: DateWindow,
    ) -> tuple[list[dict], list[dict]]:
        rows, diagnostic = self.collect_window(company=company, profile=profile, window=window)
        diagnostics = [diagnostic]
        if not diagnostic["split"]:
            return rows, diagnostics

        left, right = split_window(window)
        for child in (left, right):
            child_rows, child_diagnostics = self.collect_adaptive(
                company=company,
                profile=profile,
                window=child,
            )
            rows.extend(child_rows)
            diagnostics.extend(child_diagnostics)
        return rows, diagnostics

    def collect(
        self,
        *,
        company: str,
        date_from: str,
        date_to_exclusive: str,
        profiles: Iterable[QueryProfile],
    ) -> tuple[list[dict], list[dict]]:
        all_rows: list[dict] = []
        all_diagnostics: list[dict] = []
        windows = monthly_windows(date_from, date_to_exclusive)
        for profile in profiles:
            for window in windows:
                rows, diagnostics = self.collect_adaptive(
                    company=company,
                    profile=profile,
                    window=window,
                )
                all_rows.extend(rows)
                all_diagnostics.extend(diagnostics)
        return all_rows, all_diagnostics
