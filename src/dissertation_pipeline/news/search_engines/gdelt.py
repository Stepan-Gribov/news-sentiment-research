from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

from dissertation_pipeline.news.search_engines.yandex import (
    DateWindow,
    monthly_windows,
    split_window,
)


GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_RESULT_CAP = 250


@dataclass(frozen=True)
class GdeltQueryProfile:
    key: str
    expression: str
    relevance_terms: tuple[str, ...]


DEFAULT_GDELT_PROFILES = {
    "corporate_news": GdeltQueryProfile(
        key="corporate_news",
        expression=(
            '("corporate news" OR "press release" OR "board of directors" OR dividend OR '
            "shareholder OR strategy OR investment OR deal OR contract)"
        ),
        relevance_terms=(
            "корпоратив",
            "пресс-релиз",
            "совет директор",
            "дивиденд",
            "акционер",
            "стратег",
            "инвестиц",
            "сделк",
            "контракт",
        ),
    ),
    "disclosure": GdeltQueryProfile(
        key="disclosure",
        expression=(
            '("information disclosure" OR "material fact" OR reporting OR issuer OR IFRS OR '
            "RAS OR bond OR rating OR dividend)"
        ),
        relevance_terms=(
            "раскрытие информац",
            "существенн",
            "факт",
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


@dataclass
class GdeltResponse:
    rows: list[dict]
    raw_json: bytes
    cache_path: Path | None = None


def build_gdelt_query(
    search_term: str,
    profile: GdeltQueryProfile,
    *,
    source_language: str = "russian",
) -> str:
    search_term = search_term.strip()
    if not search_term:
        raise ValueError("search_term must not be empty")
    return f"{search_term} {profile.expression} sourcelang:{source_language}"


def parse_gdelt_json(raw_json: bytes) -> GdeltResponse:
    try:
        payload = json.loads(raw_json.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid GDELT JSON response: {exc}") from exc
    articles = payload.get("articles", [])
    if not isinstance(articles, list):
        raise ValueError("GDELT JSON response has no article list")

    rows: list[dict] = []
    for rank, article in enumerate(articles, start=1):
        link = str(article.get("url") or "").strip()
        if not link:
            continue
        rows.append(
            {
                "rank": rank,
                "link": link,
                "mobile_link": str(article.get("url_mobile") or "").strip(),
                "domain": str(article.get("domain") or "").strip(),
                "title": str(article.get("title") or "").strip(),
                "engine_date": str(article.get("seendate") or "").strip(),
                "social_image": str(article.get("socialimage") or "").strip(),
                "language": str(article.get("language") or "").strip(),
                "source_country": str(article.get("sourcecountry") or "").strip(),
            }
        )
    return GdeltResponse(rows=rows, raw_json=raw_json)


def _atomic_write_bytes(data: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp_path.write_bytes(data)
    tmp_path.replace(path)


class GdeltDocClient:
    def __init__(
        self,
        *,
        request_interval_seconds: float = 10.0,
        timeout_seconds: int = 120,
        max_retries: int = 8,
        force: bool = False,
    ) -> None:
        if request_interval_seconds < 5:
            raise ValueError("GDELT requires at least five seconds between requests")
        self.request_interval_seconds = request_interval_seconds
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.force = force
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "dissertation-news-research/0.1",
                "Accept": "application/json",
            }
        )
        self._last_request_at: float | None = None

    def _wait_for_rate_limit(self) -> None:
        if self._last_request_at is None:
            return
        remaining = self.request_interval_seconds - (time.monotonic() - self._last_request_at)
        if remaining > 0:
            time.sleep(remaining)

    def search(
        self,
        *,
        query: str,
        window: DateWindow,
        cache_path: Path,
    ) -> GdeltResponse:
        if not self.force and cache_path.exists() and cache_path.stat().st_size > 0:
            try:
                result = parse_gdelt_json(cache_path.read_bytes())
                result.cache_path = cache_path
                return result
            except Exception as exc:
                print(f"Ignoring unreadable GDELT cache {cache_path.name}: {exc}", flush=True)

        params = {
            "query": query,
            "mode": "artlist",
            "maxrecords": str(GDELT_RESULT_CAP),
            "format": "json",
            "sort": "datedesc",
            "startdatetime": f"{window.start:%Y%m%d}000000",
            "enddatetime": f"{window.end:%Y%m%d}235959",
        }
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self._wait_for_rate_limit()
            try:
                response = self.session.get(
                    GDELT_DOC_URL,
                    params=params,
                    timeout=self.timeout_seconds,
                )
                self._last_request_at = time.monotonic()
                if response.status_code == 429:
                    raise RuntimeError("GDELT rate limit (HTTP 429)")
                response.raise_for_status()
                result = parse_gdelt_json(response.content)
                _atomic_write_bytes(response.content, cache_path)
                result.cache_path = cache_path
                return result
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                if "HTTP 429" in str(exc):
                    backoff = min(180.0, 30.0 * attempt)
                else:
                    backoff = max(self.request_interval_seconds, attempt * 5.0)
                print(
                    f"GDELT attempt {attempt}/{self.max_retries} failed: {exc}; "
                    f"retrying after {backoff:.1f}s",
                    flush=True,
                )
                time.sleep(backoff)
        raise RuntimeError(f"GDELT request failed after {self.max_retries} attempts: {last_error}")


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower().replace("ё", "е")).strip()


def score_gdelt_relevance(
    row: dict,
    *,
    company: str,
    profile: GdeltQueryProfile,
) -> tuple[int, list[str]]:
    title = _normalize(row.get("title", ""))
    domain = _normalize(row.get("domain", ""))
    company_stem = _normalize(company)

    score = 3 if company_stem and company_stem in title else 0
    matched: list[str] = []
    for raw_term in profile.relevance_terms:
        term = _normalize(raw_term)
        if term in title:
            score += 2
            matched.append(raw_term)
    if domain.endswith("gazprom.ru") or domain.endswith("e-disclosure.ru"):
        score += 2
    return score, matched


class GdeltWindowCollector:
    def __init__(
        self,
        client: GdeltDocClient,
        *,
        raw_dir: Path,
        min_window_days: int = 1,
    ) -> None:
        if min_window_days < 1:
            raise ValueError("min_window_days must be at least 1")
        self.client = client
        self.raw_dir = raw_dir
        self.min_window_days = min_window_days

    def _cache_path(self, profile: GdeltQueryProfile, window: DateWindow, query: str) -> Path:
        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
        return (
            self.raw_dir
            / profile.key
            / (
                f"{window.start:%Y%m%d}_{window.end:%Y%m%d}"
                f"_d{window.depth}_{query_hash}.json"
            )
        )

    def collect_window(
        self,
        *,
        company: str,
        search_term: str,
        profile: GdeltQueryProfile,
        window: DateWindow,
        source_language: str,
    ) -> tuple[list[dict], dict]:
        query = build_gdelt_query(search_term, profile, source_language=source_language)
        cache_path = self._cache_path(profile, window, query)
        response = self.client.search(query=query, window=window, cache_path=cache_path)
        rows = response.rows
        for row in rows:
            score, matched_terms = score_gdelt_relevance(row, company=company, profile=profile)
            row.update(
                {
                    "company": company,
                    "query_profile": profile.key,
                    "search_query": query,
                    "window_from": window.start.isoformat(),
                    "window_to": window.end.isoformat(),
                    "partition_depth": window.depth,
                    "raw_response_path": str(response.cache_path or cache_path),
                    "local_relevance_score": score,
                    "matched_terms": "; ".join(matched_terms),
                }
            )

        unique_count = len({row["link"] for row in rows})
        saturated = len(rows) >= GDELT_RESULT_CAP
        diagnostic = {
            "company": company,
            "query_profile": profile.key,
            "search_query": query,
            "window_from": window.start.isoformat(),
            "window_to": window.end.isoformat(),
            "partition_depth": window.depth,
            "window_days": window.days,
            "rows_returned": len(rows),
            "unique_links": unique_count,
            "saturated": saturated,
            "split": saturated and window.days > self.min_window_days,
            "leaf_saturated": saturated and window.days <= self.min_window_days,
        }
        print(
            f"GDELT {profile.key}: {window.start}..{window.end} "
            f"links={unique_count} saturated={saturated}",
            flush=True,
        )
        return rows, diagnostic

    def collect_adaptive(
        self,
        *,
        company: str,
        search_term: str,
        profile: GdeltQueryProfile,
        window: DateWindow,
        source_language: str,
    ) -> tuple[list[dict], list[dict]]:
        rows, diagnostic = self.collect_window(
            company=company,
            search_term=search_term,
            profile=profile,
            window=window,
            source_language=source_language,
        )
        diagnostics = [diagnostic]
        if not diagnostic["split"]:
            return rows, diagnostics
        for child in split_window(window):
            child_rows, child_diagnostics = self.collect_adaptive(
                company=company,
                search_term=search_term,
                profile=profile,
                window=child,
                source_language=source_language,
            )
            rows.extend(child_rows)
            diagnostics.extend(child_diagnostics)
        return rows, diagnostics

    def collect(
        self,
        *,
        company: str,
        search_term: str,
        date_from: str,
        date_to_exclusive: str,
        profiles: Iterable[GdeltQueryProfile],
        source_language: str = "russian",
    ) -> tuple[list[dict], list[dict]]:
        all_rows: list[dict] = []
        all_diagnostics: list[dict] = []
        windows = monthly_windows(date_from, date_to_exclusive)
        for profile in profiles:
            for window in windows:
                rows, diagnostics = self.collect_adaptive(
                    company=company,
                    search_term=search_term,
                    profile=profile,
                    window=window,
                    source_language=source_language,
                )
                all_rows.extend(rows)
                all_diagnostics.extend(diagnostics)
        return all_rows, all_diagnostics
