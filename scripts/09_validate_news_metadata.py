from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401

from dissertation_pipeline.paths import PATHS


MISSING_TEXT = {"", "nan", "none", "nat", "<na>"}


def nonempty(series: pd.Series) -> pd.Series:
    return series.notna() & ~series.astype("string").str.strip().str.lower().isin(MISSING_TEXT)


def article_summary(frame: pd.DataFrame) -> dict:
    media_rows = []
    for media, part in frame.groupby("media", dropna=False):
        valid_dates = part.loc[part["date"].notna(), "date"]
        media_rows.append(
            {
                "media": str(media),
                "rows": int(len(part)),
                "company_rows": int(nonempty(part["company"]).sum()),
                "date_rows": int(part["date"].notna().sum()),
                "category_rows": int(nonempty(part["category"]).sum()),
                "text_rows": int(nonempty(part["text"]).sum()),
                "date_min": valid_dates.min().isoformat() if not valid_dates.empty else None,
                "date_max": valid_dates.max().isoformat() if not valid_dates.empty else None,
            }
        )
    return {
        "rows": int(len(frame)),
        "unique_media_link": int(frame[["media", "link"]].drop_duplicates().shape[0]),
        "company_rows": int(nonempty(frame["company"]).sum()),
        "date_rows": int(frame["date"].notna().sum()),
        "category_rows": int(nonempty(frame["category"]).sum()),
        "text_rows": int(nonempty(frame["text"]).sum()),
        "date_source_counts": {
            str(key): int(value) for key, value in frame["date_source"].fillna("missing").value_counts().items()
        },
        "by_media": media_rows,
    }


def map_summary(frame: pd.DataFrame) -> dict:
    return {
        "rows": int(len(frame)),
        "unique_article_keys": int(frame[["media", "link"]].drop_duplicates().shape[0]),
        "unique_company_labels": int(frame.loc[nonempty(frame["company"]), "company"].nunique()),
        "by_media": [
            {
                "media": str(media),
                "rows": int(len(part)),
                "articles": int(part["link"].nunique()),
                "company_labels": int(part["company"].nunique()),
            }
            for media, part in frame.groupby("media", dropna=False)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit final news metadata and article-company coverage.")
    parser.add_argument("--articles", type=Path, default=PATHS.news_merged_dir / "articles_total.parquet")
    parser.add_argument("--company-map", type=Path, default=PATHS.news_merged_dir / "article_company_map.parquet")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    article_columns = ["media", "link", "company", "date", "date_source", "category", "text"]
    map_columns = ["media", "link", "company"]
    articles = pd.read_parquet(args.articles, columns=article_columns)
    company_map = pd.read_parquet(args.company_map, columns=map_columns)
    report = {
        "articles_path": str(args.articles),
        "company_map_path": str(args.company_map),
        "articles": article_summary(articles),
        "article_company_map": map_summary(company_map),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"saved audit: {args.output}")


if __name__ == "__main__":
    main()
