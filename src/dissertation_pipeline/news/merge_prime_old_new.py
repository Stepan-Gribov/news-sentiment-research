import csv
import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


OLD_DEFAULT = Path(
    r"./local/Thesis\funds_from laptop 3-26\Data\Processed_data\1prime_news_database\1prime_downloaded_news_processed"
)
NEW_DEFAULT = Path(r"./local/NEWS_PAGES_PARQ\prime_downloaded_news")
OUTPUT_DEFAULT = Path(r"./local/NEWS_PAGES_PARQ\prime_merged_full.parquet")


OLD_COLUMNS = [
    "source_file",
    "title",
    "date",
    "link",
    "source",
    "key",
    "news_content",
    "translation",
    "sentiment",
    "prob",
]


def set_max_csv_field_size_limit():
    max_size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_size)
            return
        except OverflowError:
            max_size //= 10


set_max_csv_field_size_limit()


def normalize_link(value) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().rstrip("/").lower()


def read_one_old_csv(csv_path: Path) -> pd.DataFrame:
    attempts = [
        {"sep": "^", "engine": "python", "encoding": "utf-16"},
        {"sep": "^", "engine": "python", "encoding": "utf-8"},
        {"sep": "^", "engine": "python", "encoding": "utf-8-sig"},
    ]

    last_error = None
    for params in attempts:
        try:
            df = pd.read_csv(csv_path, **params)
            break
        except Exception as e:
            last_error = e
    else:
        raise last_error

    if list(df.columns) == OLD_COLUMNS:
        return df

    if len(df.columns) == 1:
        col = df.columns[0]
        split_df = df[col].str.split("^", expand=True)
        split_df.columns = OLD_COLUMNS[: split_df.shape[1]]
        for missing_col in OLD_COLUMNS:
            if missing_col not in split_df.columns:
                split_df[missing_col] = None
        return split_df[OLD_COLUMNS]

    renamed = df.copy()
    renamed.columns = [str(c).strip() for c in renamed.columns]
    for missing_col in OLD_COLUMNS:
        if missing_col not in renamed.columns:
            renamed[missing_col] = None
    return renamed[OLD_COLUMNS]


def collect_old_dataset(old_dir: Path) -> pd.DataFrame:
    csv_files = sorted(old_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No old CSV files found in {old_dir}")

    parts = []
    for idx, csv_path in enumerate(csv_files, start=1):
        print(f"[OLD {idx}/{len(csv_files)}] {csv_path.name}")
        parts.append(read_one_old_csv(csv_path))

    old_df = pd.concat(parts, ignore_index=True)
    old_df["dataset_origin"] = "old"
    old_df["company_hint"] = old_df["key"]
    old_df["article_text_extracted"] = old_df["news_content"]
    old_df["search_query"] = None
    old_df["category"] = None
    old_df["page_source_error"] = None
    return old_df[
        [
            "source_file",
            "company_hint",
            "title",
            "link",
            "date",
            "category",
            "news_content",
            "article_text_extracted",
            "translation",
            "sentiment",
            "prob",
            "source",
            "key",
            "search_query",
            "dataset_origin",
            "page_source_error",
        ]
    ]


def read_one_new_parquet(parquet_path: Path) -> pd.DataFrame:
    columns = ["search_query", "name", "link", "category", "timestamp", "article_text_extracted", "page_source_error"]
    df = pd.read_parquet(parquet_path, columns=columns).copy()

    df["source_file"] = parquet_path.stem
    df["company_hint"] = df["search_query"]
    df["title"] = df["name"]
    df["date"] = df["timestamp"]
    df["news_content"] = df["article_text_extracted"]
    df["translation"] = None
    df["sentiment"] = None
    df["prob"] = None
    df["source"] = None
    df["key"] = None
    df["dataset_origin"] = "new"

    return df[
        [
            "source_file",
            "company_hint",
            "title",
            "link",
            "date",
            "category",
            "news_content",
            "article_text_extracted",
            "translation",
            "sentiment",
            "prob",
            "source",
            "key",
            "search_query",
            "dataset_origin",
            "page_source_error",
        ]
    ]


def collect_new_dataset(new_dir: Path) -> pd.DataFrame:
    parquet_files = sorted(new_dir.glob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No new parquet files found in {new_dir}")

    parts = []
    for idx, parquet_path in enumerate(parquet_files, start=1):
        print(f"[NEW {idx}/{len(parquet_files)}] {parquet_path.name}")
        parts.append(read_one_new_parquet(parquet_path))

    return pd.concat(parts, ignore_index=True)


def merge_datasets(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.concat([old_df, new_df], ignore_index=True)

    merged["link_norm"] = merged["link"].map(normalize_link)
    merged["has_article_text"] = merged["article_text_extracted"].astype(str).str.strip().ne("")
    merged["has_translation"] = merged["translation"].astype(str).str.strip().ne("")
    merged["origin_priority"] = merged["dataset_origin"].map({"old": 0, "new": 1}).fillna(9)

    merged = merged.sort_values(
        by=["link_norm", "origin_priority", "has_article_text", "has_translation"],
        ascending=[True, True, False, False],
        kind="stable",
    )
    merged = merged.drop_duplicates(subset=["link_norm"], keep="first").reset_index(drop=True)
    merged = merged.drop(columns=["link_norm", "has_article_text", "has_translation", "origin_priority"])
    return merged


def prepare_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    string_columns = [
        "source_file",
        "company_hint",
        "title",
        "link",
        "date",
        "category",
        "news_content",
        "article_text_extracted",
        "translation",
        "sentiment",
        "source",
        "key",
        "search_query",
        "dataset_origin",
        "page_source_error",
    ]

    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].where(df[col].notna(), None)
            df[col] = df[col].astype("string")

    return df


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--old-dir", type=Path, default=OLD_DEFAULT)
    parser.add_argument("--new-dir", type=Path, default=NEW_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    return parser.parse_args()


def main():
    args = parse_args()

    old_df = collect_old_dataset(args.old_dir)
    print(f"Old rows: {len(old_df)}")

    new_df = collect_new_dataset(args.new_dir)
    print(f"New rows: {len(new_df)}")

    merged_df = merge_datasets(old_df, new_df)
    merged_df = prepare_for_parquet(merged_df)
    print(f"Merged rows after dedup: {len(merged_df)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(args.output, index=False)
    print(f"Saved merged parquet: {args.output}")


if __name__ == "__main__":
    main()
