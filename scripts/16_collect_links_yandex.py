from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401

from dissertation_pipeline.news.layout import links_run_dir
from dissertation_pipeline.news.resume import atomic_write_csv
from dissertation_pipeline.news.search_engines.yandex import (
    DEFAULT_QUERY_PROFILES,
    YandexSearchClient,
    YandexWindowCollector,
    build_query,
    monthly_windows,
)


def env_first(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def safe_filename(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-zА-Яа-яЁё._-]+", "_", value.strip())
    return value.strip("._") or "company"


def select_profiles(raw: str):
    keys = [value.strip() for value in raw.split(",") if value.strip()]
    unknown = [key for key in keys if key not in DEFAULT_QUERY_PROFILES]
    if unknown:
        raise SystemExit(
            f"Unknown query profiles: {', '.join(unknown)}. "
            f"Available: {', '.join(DEFAULT_QUERY_PROFILES)}"
        )
    return [DEFAULT_QUERY_PROFILES[key] for key in keys]


def build_link_table(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            columns=[
                "company",
                "link",
                "domain",
                "title",
                "snippet",
                "engine_date",
                "local_relevance_score",
                "matched_profiles",
                "matched_terms",
                "discovery_count",
            ]
        )

    records_df = pd.DataFrame(records)
    profiles_by_link = records_df.groupby("link")["query_profile"].apply(
        lambda values: "; ".join(sorted(set(values.astype(str))))
    )
    count_by_link = records_df.groupby("link").size()
    terms_by_link = records_df.groupby("link")["matched_terms"].apply(
        lambda values: "; ".join(
            sorted(
                {
                    term.strip()
                    for value in values.fillna("").astype(str)
                    for term in value.split(";")
                    if term.strip()
                }
            )
        )
    )

    best = records_df.sort_values(
        ["local_relevance_score", "partition_depth", "rank"],
        ascending=[False, False, True],
    ).drop_duplicates(subset=["link"], keep="first")
    best["matched_profiles"] = best["link"].map(profiles_by_link)
    best["matched_terms"] = best["link"].map(terms_by_link)
    best["discovery_count"] = best["link"].map(count_by_link)
    return best.sort_values(
        ["local_relevance_score", "engine_date", "rank"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def build_query_results_table(records: list[dict]) -> pd.DataFrame:
    columns = [
        "company",
        "query_profile",
        "search_query",
        "window_from",
        "window_to",
        "partition_depth",
        "page",
        "rank",
        "link",
        "domain",
        "title",
        "snippet",
        "engine_date",
        "mime_type",
        "saved_copy_url",
        "local_relevance_score",
        "matched_terms",
        "raw_response_path",
    ]
    return pd.DataFrame(records).reindex(columns=columns)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect corporate-news and disclosure links through Yandex Search API."
    )
    parser.add_argument("--company", default="Газпром")
    parser.add_argument("--date-from", default="2026-01-01")
    parser.add_argument(
        "--date-to",
        default="2026-06-01",
        help="Exclusive end date, matching the other datamining scripts.",
    )
    parser.add_argument("--profiles", default="corporate_news,disclosure")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--groups-on-page", type=int, default=100)
    parser.add_argument("--min-window-days", type=int, default=1)
    parser.add_argument("--min-relevance-score", type=int, default=3)
    parser.add_argument("--request-delay", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Ignore cached raw XML responses.")
    args = parser.parse_args()

    profiles = select_profiles(args.profiles)
    output_dir = args.out_dir or links_run_dir("yandex", args.date_from, args.date_to)
    raw_dir = output_dir / "raw" / safe_filename(args.company)

    if args.dry_run:
        print(f"company={args.company}")
        print(f"date interval=[{args.date_from}, {args.date_to})")
        print(f"profiles={','.join(profile.key for profile in profiles)}")
        for profile in profiles:
            for window in monthly_windows(args.date_from, args.date_to):
                print(f"{profile.key}: {build_query(args.company, profile, window)}")
        print(f"output={output_dir}")
        return

    api_key = env_first("DISS_YANDEX_SEARCH_API_KEY", "YANDEX_API_KEY")
    folder_id = env_first("DISS_YANDEX_FOLDER_ID", "YANDEX_FOLDER_ID")
    if not api_key or not folder_id:
        raise SystemExit(
            "Yandex credentials are missing. Set DISS_YANDEX_SEARCH_API_KEY and "
            "DISS_YANDEX_FOLDER_ID (YANDEX_API_KEY/YANDEX_FOLDER_ID are also accepted)."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    client = YandexSearchClient(
        api_key=api_key,
        folder_id=folder_id,
        groups_on_page=args.groups_on_page,
        request_delay_seconds=args.request_delay,
        force=args.force,
    )
    collector = YandexWindowCollector(
        client,
        raw_dir=raw_dir,
        min_window_days=args.min_window_days,
    )
    records, diagnostics = collector.collect(
        company=args.company,
        date_from=args.date_from,
        date_to_exclusive=args.date_to,
        profiles=profiles,
    )

    company_file = safe_filename(args.company)
    records_df = build_query_results_table(records)
    links_df = build_link_table(records)
    relevant_df = links_df[
        links_df["local_relevance_score"] >= args.min_relevance_score
    ].reset_index(drop=True)
    diagnostics_df = pd.DataFrame(diagnostics)

    atomic_write_csv(records_df, output_dir / f"{company_file}_query_results.csv")
    atomic_write_csv(links_df, output_dir / f"{company_file}_all_links.csv")
    atomic_write_csv(relevant_df, output_dir / f"{company_file}_relevant_links.csv")
    atomic_write_csv(diagnostics_df, output_dir / f"{company_file}_windows.csv")

    print(f"raw query rows={len(records_df):,}")
    print(f"unique links={len(links_df):,}")
    print(
        f"relevant links (score >= {args.min_relevance_score})={len(relevant_df):,}"
    )
    print(f"output={output_dir}")


if __name__ == "__main__":
    main()
