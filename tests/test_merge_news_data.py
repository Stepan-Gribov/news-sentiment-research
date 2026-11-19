from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

spec = importlib.util.spec_from_file_location("merge_news_data", SCRIPTS_DIR / "08_merge_news_data.py")
merge_news_data = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(merge_news_data)


def test_parse_dates_supports_all_historical_formats() -> None:
    values = pd.Series(
        [
            "2026-01-17",
            "2012-05-22T17:28",
            "2024-01-11T11:21:58+03:00",
            "7 сентября 2006, 12:00",
            "12.05.2026",
            "3 января, 14:06",
            "09:47",
        ]
    )

    parsed = merge_news_data.parse_dates(values)

    assert parsed.iloc[:5].notna().all()
    assert parsed.iloc[0] == pd.Timestamp("2026-01-17")
    assert parsed.iloc[1] == pd.Timestamp("2012-05-22 17:28:00")
    assert parsed.iloc[2] == pd.Timestamp("2024-01-11 08:21:58")
    assert parsed.iloc[3] == pd.Timestamp("2006-09-07 12:00:00")
    assert parsed.iloc[4] == pd.Timestamp("2026-05-12")
    assert pd.isna(parsed.iloc[5])
    assert pd.isna(parsed.iloc[6])


def test_dates_are_recovered_from_validated_url_patterns() -> None:
    rbc_date, rbc_source, rbc_precision = merge_news_data.infer_date_from_url(
        "https://chr.rbc.ru/chr/01/06/2021/60b6198e9a79472e77715e89", "rbc"
    )
    prime_date, prime_source, prime_precision = merge_news_data.infer_date_from_url(
        "https://1prime.ru/20250103/vtb-854013196.html", "prime"
    )

    assert rbc_date == pd.Timestamp("2021-06-01")
    assert rbc_source == "url_path"
    assert rbc_precision == "day"
    assert prime_date == pd.Timestamp("2025-01-03")
    assert prime_source == "url_path"
    assert prime_precision == "day"


def test_interfax_company_is_recovered_from_source_file() -> None:
    frame = pd.DataFrame(
        {
            "media": ["interfax"],
            "link": ["https://www.interfax.ru/123"],
            "company": [pd.NA],
            "date": ["2012-05-22T17:28"],
            "title": ["Title"],
            "category": ["Экономика"],
            "text": ["Article"],
            "source_file": ["Газпром"],
        }
    )

    result = merge_news_data.normalize_unified_articles(frame, Path("unified.parquet"))

    assert result.loc[0, "company"] == "Газпром"
    assert result.loc[0, "search_query"] == "Газпром"
    assert result.loc[0, "company_source"] == "article_source_file"


def test_link_enrichment_fills_missing_date_and_category() -> None:
    articles = pd.DataFrame(
        {
            "media": ["prime"],
            "company": ["Газпром"],
            "search_query": ["Газпром"],
            "title": ["Title"],
            "link": ["https://1prime.ru/123/"],
            "date": [pd.NaT],
            "category": [pd.NA],
            "text": ["Article"],
            "source_file": ["Газпром"],
        }
    )
    links = pd.DataFrame(
        {
            "media": ["prime"],
            "company": ["Газпром"],
            "search_query": ["Газпром"],
            "link": ["https://1prime.ru/123"],
            "date": [pd.Timestamp("2024-02-01 14:30")],
            "title": ["Title"],
            "category": ["Компании"],
            "project": [pd.NA],
            "query_profile": [pd.NA],
            "source_file": ["Газпром.csv"],
        }
    )

    result = merge_news_data.enrich_articles_from_links(articles, links)

    assert result.loc[0, "date"] == pd.Timestamp("2024-02-01 14:30")
    assert result.loc[0, "date_source"] == "link_metadata"
    assert result.loc[0, "category"] == "Компании"


def test_company_map_preserves_many_to_many_search_associations() -> None:
    articles = pd.DataFrame(
        {
            "media": ["rbc"],
            "company": ["Газпром"],
            "search_query": ["Газпром"],
            "title": ["Joint story"],
            "link": ["https://rbc.ru/story"],
            "date": [pd.Timestamp("2024-01-01")],
            "category": ["Бизнес"],
            "project": ["RBC"],
            "query_profile": [pd.NA],
            "source_file": ["Газпром"],
        }
    )
    links = pd.DataFrame(
        {
            "media": ["rbc", "rbc"],
            "company": ["Газпром", "НОВАТЭК"],
            "search_query": ["Газпром", "НОВАТЭК"],
            "link": ["https://rbc.ru/story", "https://rbc.ru/story"],
            "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
            "title": ["Joint story", "Joint story"],
            "category": ["Бизнес", "Бизнес"],
            "project": ["RBC", "RBC"],
            "query_profile": [pd.NA, pd.NA],
            "source_file": ["Газпром.csv", "НОВАТЭК.csv"],
        }
    )

    result = merge_news_data.build_article_company_map(articles, links)

    assert set(result["company"]) == {"Газпром", "НОВАТЭК"}
    assert len(result) == 2
