from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401

from dissertation_pipeline.news.layout import links_run_dir
from dissertation_pipeline.news.resume import atomic_write_csv
from dissertation_pipeline.news.search_engines.gdelt import (
    DEFAULT_GDELT_PROFILES,
    GdeltDocClient,
    GdeltWindowCollector,
    build_gdelt_query,
)
from dissertation_pipeline.news.search_engines.yandex import monthly_windows


def safe_filename(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-zА-Яа-яЁё._-]+", "_", value.strip())
    return value.strip("._") or "company"


def select_profiles(raw: str):
    keys = [value.strip() for value in raw.split(",") if value.strip()]
    unknown = [key for key in keys if key not in DEFAULT_GDELT_PROFILES]
    if unknown:
        raise SystemExit(
            f"Unknown profiles: {', '.join(unknown)}. "
            f"Available: {', '.join(DEFAULT_GDELT_PROFILES)}"
        )
    return [DEFAULT_GDELT_PROFILES[key] for key in keys]


def build_query_results_table(records: list[dict]) -> pd.DataFrame:
    columns = [
        "company",
        "query_profile",
        "search_query",
        "window_from",
        "window_to",
        "partition_depth",
        "rank",
        "link",
        "mobile_link",
        "domain",
        "title",
        "engine_date",
        "social_image",
        "language",
        "source_country",
        "local_relevance_score",
        "matched_terms",
        "raw_response_path",
    ]
    return pd.DataFrame(records).reindex(columns=columns)


def build_link_table(records: list[dict]) -> pd.DataFrame:
    records_df = build_query_results_table(records)
    if records_df.empty:
        return records_df.assign(matched_profiles="", discovery_count=0)

    profiles_by_link = records_df.groupby("link")["query_profile"].apply(
        lambda values: "; ".join(sorted(set(values.astype(str))))
    )
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
    count_by_link = records_df.groupby("link").size()
    best = records_df.sort_values(
        ["local_relevance_score", "partition_depth", "rank"],
        ascending=[False, False, True],
    ).drop_duplicates("link")
    best["matched_profiles"] = best["link"].map(profiles_by_link)
    best["matched_terms"] = best["link"].map(terms_by_link)
    best["discovery_count"] = best["link"].map(count_by_link)
    return best.sort_values(
        ["local_relevance_score", "engine_date", "rank"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Free GDELT corporate-news link pilot.")
    parser.add_argument("--company", default="Газпром")
    parser.add_argument(
        "--search-term",
        default="gazprom",
        help="English/transliterated term searched in GDELT's translated full text.",
    )
    parser.add_argument("--date-from", default="2026-01-01")
    parser.add_argument("--date-to", default="2026-06-01", help="Exclusive end date.")
    parser.add_argument("--profiles", default="corporate_news,disclosure")
    parser.add_argument("--source-language", default="russian")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--min-window-days", type=int, default=1)
    parser.add_argument("--min-relevance-score", type=int, default=3)
    parser.add_argument("--request-interval", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Ignore cached JSON responses.")
    args = parser.parse_args()

    profiles = select_profiles(args.profiles)
    output_dir = args.out_dir or links_run_dir("gdelt", args.date_from, args.date_to)
    raw_dir = output_dir / "raw" / safe_filename(args.company)

    if args.dry_run:
        print(f"company={args.company}")
        print(f"date interval=[{args.date_from}, {args.date_to})")
        for profile in profiles:
            query = build_gdelt_query(
                args.search_term,
                profile,
                source_language=args.source_language,
            )
            for window in monthly_windows(args.date_from, args.date_to):
                print(f"{profile.key}: {window.start}..{window.end}: {query}")
        print(f"output={output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    client = GdeltDocClient(
        request_interval_seconds=args.request_interval,
        force=args.force,
    )
    collector = GdeltWindowCollector(
        client,
        raw_dir=raw_dir,
        min_window_days=args.min_window_days,
    )
    records, diagnostics = collector.collect(
        company=args.company,
        search_term=args.search_term,
        date_from=args.date_from,
        date_to_exclusive=args.date_to,
        profiles=profiles,
        source_language=args.source_language,
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
    print(f"relevant links (score >= {args.min_relevance_score})={len(relevant_df):,}")
    print(f"output={output_dir}")


if __name__ == "__main__":
    main()
