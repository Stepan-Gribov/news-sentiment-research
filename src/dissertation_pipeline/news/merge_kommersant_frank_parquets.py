from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT_DIR = Path(
    r"./local/My Drive\4. Disserartion research\calculations\data\news_links\NEWS_PAGES_PARQ"
)

SOURCE_CONFIG = {
    "kommersant_downloaded_news": {
        "output_name": "kommersant_merged_full.parquet",
        "drop_columns": {"page_source"},
        "dedupe_key": "url",
    },
    "frank_downloaded_news": {
        "output_name": "frank_merged_full.parquet",
        "drop_columns": {"content"},
        "dedupe_key": "link",
    },
}


def clean_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def is_missing(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().eq("")


def safe_read_parquet(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        print(f"SKIP broken parquet: {path.name} -> {exc}")
        return None


def extract_first_regex(text: object, patterns: list[str]) -> str:
    html_text = clean_value(text)
    if not html_text:
        return ""

    for pattern in patterns:
        match = re.search(pattern, html_text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return clean_value(match.group(1))
    return ""


def fill_kommersant_metadata(df: pd.DataFrame) -> pd.DataFrame:
    if "category" not in df.columns:
        df["category"] = ""
    if "date" not in df.columns:
        df["date"] = ""

    if "timestamp" in df.columns:
        missing_date = is_missing(df["date"])
        df.loc[missing_date, "date"] = df.loc[missing_date, "timestamp"].fillna("").astype(str)

    if "page_source" not in df.columns:
        return df

    need_category = is_missing(df["category"])
    if need_category.any():
        category_patterns = [
            r'<meta[^>]+property=["\']article:section["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+name=["\']article:section["\'][^>]+content=["\']([^"\']+)["\']',
            r'class=["\'][^"\']*doc_header__subheader[^"\']*["\'][^>]*>([^<]+)<',
            r'class=["\'][^"\']*doc_header__rubric[^"\']*["\'][^>]*>([^<]+)<',
        ]
        df.loc[need_category, "category"] = (
            df.loc[need_category, "page_source"].map(lambda value: extract_first_regex(value, category_patterns))
        )

    return df


def fill_frank_metadata(df: pd.DataFrame) -> pd.DataFrame:
    if "category" not in df.columns:
        df["category"] = ""
    if "date" not in df.columns:
        df["date"] = ""
    return df


def normalize_company_frame(folder_name: str, parquet_path: Path, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["company"] = parquet_path.stem
    df["source_media"] = folder_name.replace("_downloaded_news", "")

    if folder_name == "kommersant_downloaded_news":
        df = fill_kommersant_metadata(df)
    elif folder_name == "frank_downloaded_news":
        df = fill_frank_metadata(df)

    return df


def prepare_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
            df[column] = df[column].fillna("").astype(str)
    return df


def iter_source_parquets(source_dir: Path) -> Iterable[Path]:
    for path in sorted(source_dir.glob("*.parquet")):
        if path.name.endswith(".tmp.parquet"):
            continue
        yield path


def merge_source(folder_name: str) -> None:
    config = SOURCE_CONFIG[folder_name]
    source_dir = ROOT_DIR / folder_name
    output_path = ROOT_DIR / config["output_name"]

    if not source_dir.exists():
        print(f"SKIP missing folder: {source_dir}")
        return

    frames: list[pd.DataFrame] = []
    parquet_files = list(iter_source_parquets(source_dir))
    total_files = len(parquet_files)
    print(f"\n[{folder_name}] Found {total_files} parquet files")

    for index, parquet_path in enumerate(parquet_files, start=1):
        print(f"[{folder_name}] [{index}/{total_files}] {parquet_path.name}")
        df = safe_read_parquet(parquet_path)
        if df is None or df.empty:
            continue

        df = normalize_company_frame(folder_name, parquet_path, df)
        drop_columns = [column for column in config["drop_columns"] if column in df.columns]
        if drop_columns:
            df = df.drop(columns=drop_columns)
        frames.append(df)

    if not frames:
        print(f"[{folder_name}] No readable data found")
        return

    merged = pd.concat(frames, ignore_index=True, sort=False)
    dedupe_key = config["dedupe_key"]
    if dedupe_key in merged.columns:
        merged[dedupe_key] = merged[dedupe_key].fillna("").astype(str).str.strip()
        merged = merged.loc[merged[dedupe_key].ne("")].copy()
        merged = merged.drop_duplicates(subset=[dedupe_key], keep="first")

    merged = prepare_for_parquet(merged)
    merged.to_parquet(output_path, index=False)
    print(f"[{folder_name}] Saved {len(merged):,} rows -> {output_path}")


def main() -> None:
    for folder_name in SOURCE_CONFIG:
        merge_source(folder_name)


if __name__ == "__main__":
    main()
