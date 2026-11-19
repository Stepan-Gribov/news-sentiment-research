from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from pandas.errors import EmptyDataError
from tqdm import tqdm

import _bootstrap  # noqa: F401

from dissertation_pipeline.paths import PATHS


MEDIA = ["rbc", "interfax", "kommersant", "prime", "frank"]

LINK_COLS = {
    "rbc": ["fronturl", "link"],
    "interfax": ["link"],
    "kommersant": ["url", "link"],
    "prime": ["link"],
    "frank": ["link"],
}

DATE_COLS = ["date", "publish_date", "timestamp"]
TITLE_COLS = ["title", "header", "name"]
TEXT_COLS = ["article_text_extracted", "article_text", "content_txt", "news_content", "text", "body"]
ARTICLE_READ_COLS = set(
    [
        "media",
        "source_media",
        "dataset_origin",
        "source",
        "company",
        "company_hint",
        "query",
        "search_query",
        "link",
        "url",
        "fronturl",
        "category",
        "topic",
        "section",
        "project",
        "query_profile",
        *DATE_COLS,
        *TITLE_COLS,
        *TEXT_COLS,
    ]
)
LINK_METADATA_COLS = [
    "media",
    "company",
    "search_query",
    "link",
    "date",
    "date_source",
    "date_precision",
    "title",
    "category",
    "project",
    "query_profile",
    "source_file",
]

RUSSIAN_MONTHS = {
    "января": "01",
    "февраля": "02",
    "марта": "03",
    "апреля": "04",
    "мая": "05",
    "июня": "06",
    "июля": "07",
    "августа": "08",
    "сентября": "09",
    "октября": "10",
    "ноября": "11",
    "декабря": "12",
}

DATE_PATH_DMY_RE = re.compile(r"/(0?[1-9]|[12]\d|3[01])/(0?[1-9]|1[0-2])/((?:19|20)\d{2})(?:/|$)")
DATE_PATH_COMPACT_YMD_RE = re.compile(r"/((?:19|20)\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?:/|$)")
RBC_FREENEWS_OBJECT_ID_RE = re.compile(r"/freenews/([0-9a-f]{24})(?:[/?#]|$)", flags=re.IGNORECASE)


def is_blank(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    return series.isna() | text.isin(["", "nan", "none", "nat", "<na>"])


def coalesce_columns(df: pd.DataFrame, cols: list[str], fallback: object = pd.NA) -> pd.Series:
    result = pd.Series(pd.NA, index=df.index, dtype="object")
    for col in cols:
        if col not in df.columns:
            continue
        values = df[col]
        take = is_blank(result) & ~is_blank(values)
        result.loc[take] = values.loc[take]
    if fallback is not pd.NA:
        result = result.mask(is_blank(result), fallback)
    return result


def normalize_company_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return unicodedata.normalize("NFC", str(value)).strip()


def normalize_link_series(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.rstrip("/").str.lower()


def infer_date_from_url(link: object, media: object) -> tuple[pd.Timestamp | pd.NaT, str | None, str | None]:
    """Recover dates only from URL patterns whose semantics were validated on known rows."""
    url = str(link).strip()
    media_name = str(media).strip().lower()
    if not url:
        return pd.NaT, None, None

    if media_name == "rbc":
        match = DATE_PATH_DMY_RE.search(url)
        if match:
            try:
                return (
                    pd.Timestamp(year=int(match.group(3)), month=int(match.group(2)), day=int(match.group(1))),
                    "url_path",
                    "day",
                )
            except ValueError:
                pass

        # Regional RBC freenews URLs use a Mongo-style object id.  On rows with
        # known timestamps this decoder agrees within 24 hours for about 98%.
        match = RBC_FREENEWS_OBJECT_ID_RE.search(url)
        if match:
            try:
                timestamp = pd.to_datetime(int(match.group(1)[:8], 16), unit="s", utc=True).tz_localize(None)
                return timestamp, "rbc_freenews_object_id", "inferred_timestamp"
            except (ValueError, OverflowError):
                pass

    if media_name == "prime":
        match = DATE_PATH_COMPACT_YMD_RE.search(url)
        if match:
            try:
                return (
                    pd.Timestamp(year=int(match.group(1)), month=int(match.group(2)), day=int(match.group(3))),
                    "url_path",
                    "day",
                )
            except ValueError:
                pass

    return pd.NaT, None, None


def fill_dates_from_urls(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "date_source" not in result.columns:
        result["date_source"] = pd.NA
    if "date_precision" not in result.columns:
        result["date_precision"] = pd.NA
    present = result["date"].notna()
    result.loc[present & is_blank(result["date_precision"]), "date_precision"] = "timestamp_or_day"

    missing_indices = result.index[result["date"].isna()]
    for index in missing_indices:
        inferred_date, source, precision = infer_date_from_url(result.at[index, "link"], result.at[index, "media"])
        if pd.notna(inferred_date):
            result.at[index, "date"] = inferred_date
            result.at[index, "date_source"] = source
            result.at[index, "date_precision"] = precision
    return result


def parse_dates(series: pd.Series) -> pd.Series:
    """Parse all date formats used by the five news collectors.

    Pandas 2.x infers one strict format for a mixed Series.  The historical merge
    concatenates ISO timestamps, date-only values and Russian Prime timestamps,
    so the old implementation silently converted entire media sources to NaT.
    """
    text = series.astype("string").str.strip().str.lower()
    text = text.mask(text.isin(["", "nan", "none", "nat", "<na>"]))
    russian_month_pattern = "|".join(RUSSIAN_MONTHS)
    has_russian_month = text.str.contains(russian_month_pattern, regex=True, na=False)
    has_explicit_year = text.str.contains(r"\b(?:19|20)\d{2}\b", regex=True, na=False)
    time_only = text.str.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", na=False)
    # Never let dateutil silently attach the current year/date.  Prime URLs carry
    # the publication day explicitly and are handled by fill_dates_from_urls().
    text = text.mask((has_russian_month & ~has_explicit_year) | time_only)
    for month_name, month_number in RUSSIAN_MONTHS.items():
        text = text.str.replace(
            rf"(?i)(\d{{1,2}})\s+{month_name}\s+(\d{{4}})",
            rf"\2-{month_number}-\1",
            regex=True,
        )
    # Kommersant run files use Russian day-first dates such as 12.05.2026.
    text = text.str.replace(
        r"\b(0?[1-9]|[12]\d|3[01])[./](0?[1-9]|1[0-2])[./]((?:19|20)\d{2})\b",
        r"\3-\2-\1",
        regex=True,
    )
    text = text.str.replace(",", " ", regex=False).str.replace(r"\s+", " ", regex=True)
    try:
        parsed = pd.to_datetime(text, errors="coerce", utc=True, format="mixed")
    except TypeError:  # pandas < 2.0
        parsed = pd.to_datetime(text, errors="coerce", utc=True)
    return parsed.dt.tz_convert(None)


def first_existing(df: pd.DataFrame, cols: list[str]) -> str | None:
    return next((col for col in cols if col in df.columns), None)


def read_link_csv(path: Path, media: str) -> pd.DataFrame:
    try:
        if path.suffix.lower() == ".xlsx":
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path, encoding="utf-16", sep="\t")
    except EmptyDataError:
        return pd.DataFrame(columns=LINK_METADATA_COLS)
    link = coalesce_columns(df, LINK_COLS[media])
    if is_blank(link).all():
        return pd.DataFrame(columns=LINK_METADATA_COLS)
    company = coalesce_columns(df, ["company", "company_hint", "query", "search_query"], path.stem)
    search_query = coalesce_columns(df, ["search_query", "query", "company", "company_hint"], path.stem)
    out = pd.DataFrame(
        {
            "media": media,
            "company": company.map(normalize_company_value),
            "search_query": search_query.map(normalize_company_value),
            "link": link.astype("string").str.strip(),
            "date": coalesce_columns(df, DATE_COLS),
            "title": coalesce_columns(df, TITLE_COLS),
            "category": coalesce_columns(df, ["category", "topic", "section"]),
            "project": coalesce_columns(df, ["project"]),
            "query_profile": coalesce_columns(df, ["query_profile"]),
            "source_file": str(path),
        }
    )
    out = out[~is_blank(out["link"])]
    return out


def is_smoke_run_file(path: Path) -> bool:
    parts = path.parts
    if "runs" not in parts:
        return False
    idx = parts.index("runs")
    return idx + 1 < len(parts) and parts[idx + 1].startswith("smoke_")


def iter_link_files(include_smoke_runs: bool = False) -> list[tuple[Path, str]]:
    roots = [PATHS.news_links_history_dir, PATHS.news_links_runs_dir]
    files: list[tuple[Path, str]] = []
    for root in roots:
        for media in MEDIA:
            files.extend((path, media) for path in sorted((root).glob(f"**/{media}/*.csv")))
            files.extend((path, media) for path in sorted((root).glob(f"**/{media}/*.xlsx")))
    if not include_smoke_runs:
        files = [(path, media) for path, media in files if not is_smoke_run_file(path)]
    return files


def collect_link_records(sample_files: int | None = None, include_smoke_runs: bool = False) -> pd.DataFrame:
    files = iter_link_files(include_smoke_runs=include_smoke_runs)
    if sample_files:
        files = files[:sample_files]
    parts = []
    for path, media in tqdm(files, desc="merge links"):
        df = read_link_csv(path, media)
        if not df.empty:
            parts.append(df)
    if not parts:
        return pd.DataFrame(columns=LINK_METADATA_COLS)
    result = pd.concat(parts, ignore_index=True)
    result["date"] = parse_dates(result["date"])
    result["date_source"] = pd.NA
    result.loc[result["date"].notna(), "date_source"] = "link_metadata"
    result["date_precision"] = pd.NA
    result = fill_dates_from_urls(result)
    result["_link_norm"] = normalize_link_series(result["link"])
    return result


def merge_links(
    sample_files: int | None = None,
    include_smoke_runs: bool = False,
    records: pd.DataFrame | None = None,
) -> pd.DataFrame:
    result = records.copy() if records is not None else collect_link_records(sample_files, include_smoke_runs)
    if result.empty:
        return pd.DataFrame(columns=LINK_METADATA_COLS)
    if "_link_norm" not in result.columns:
        result["_link_norm"] = normalize_link_series(result["link"])
    metadata_cols = ["company", "search_query", "date", "title", "category", "project", "query_profile"]
    result["_metadata_score"] = sum((~is_blank(result[col])).astype(int) for col in metadata_cols)
    result = result.sort_values(["media", "_link_norm", "_metadata_score"], ascending=[True, True, False])
    result = result.drop_duplicates(subset=["media", "_link_norm"], keep="first")
    result = result.drop(columns=["_link_norm", "_metadata_score"])
    result = result.sort_values(["media", "date"], ascending=[True, False]).reset_index(drop=True)
    return result


def infer_media(path: Path) -> str | None:
    low = str(path).lower()
    if "rbc" in low or "рбк" in low:
        return "rbc"
    for media in ["interfax", "kommersant", "prime", "frank"]:
        if media in low:
            return media
    return None


def read_article_parquet(path: Path) -> pd.DataFrame:
    """Read only analysis columns and never load raw HTML/page_source blobs."""
    available = pq.ParquetFile(path).schema_arrow.names
    columns = [col for col in available if col in ARTICLE_READ_COLS]
    return pd.read_parquet(path, columns=columns)


def normalize_articles(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    media = infer_media(path)
    link = coalesce_columns(df, ["link", "url", "fronturl"])
    if not media or is_blank(link).all():
        return pd.DataFrame(columns=["media", "company", "title", "link", "date", "category", "text", "source_file"])
    company = coalesce_columns(df, ["company", "company_hint", "query", "search_query"], path.stem)
    search_query = coalesce_columns(df, ["search_query", "query", "company", "company_hint"], path.stem)
    out = pd.DataFrame(
        {
            "media": media,
            "company": company.map(normalize_company_value),
            "search_query": search_query.map(normalize_company_value),
            "title": coalesce_columns(df, TITLE_COLS),
            "link": link.astype("string").str.strip(),
            "date": coalesce_columns(df, DATE_COLS),
            "category": coalesce_columns(df, ["category", "topic", "section"]),
            "project": coalesce_columns(df, ["project"]),
            "query_profile": coalesce_columns(df, ["query_profile"]),
            "text": coalesce_columns(df, TEXT_COLS),
            "source_file": str(path),
        }
    )
    out = out[~is_blank(out["link"])]
    return out


def iter_article_files(
    include_history: bool = True,
    include_runs: bool = True,
    include_smoke_runs: bool = False,
) -> list[Path]:
    files = []
    roots = []
    if include_history:
        roots.append(PATHS.news_pages_history_dir)
    if include_runs:
        roots.append(PATHS.news_pages_runs_dir)
    for root in roots:
        files.extend(sorted(root.glob("**/*.parquet")))
    if not include_smoke_runs:
        files = [path for path in files if not is_smoke_run_file(path)]
    return files


def merge_articles(sample_files: int | None = None, include_smoke_runs: bool = False) -> pd.DataFrame:
    files = iter_article_files(include_smoke_runs=include_smoke_runs)
    if sample_files:
        files = files[:sample_files]
    parts = []
    for path in tqdm(files, desc="merge articles"):
        try:
            df = read_article_parquet(path)
        except Exception as exc:
            print(f"skip unreadable parquet: {path} ({exc})")
            continue
        normalized = normalize_articles(df, path)
        if not normalized.empty:
            parts.append(normalized)
    if not parts:
        return pd.DataFrame(columns=["media", "company", "title", "link", "date", "category", "text", "source_file"])
    result = pd.concat(parts, ignore_index=True)
    result["date"] = parse_dates(result["date"])
    result["_text_len"] = result["text"].fillna("").astype(str).str.len()
    result = result.sort_values(["media", "link", "_text_len"], ascending=[True, True, False])
    result = result.drop_duplicates(subset=["media", "link"], keep="first")
    result = result.drop(columns=["_text_len"]).sort_values(["media", "date"], ascending=[True, False])
    return result.reset_index(drop=True)


def normalize_unified_articles(df: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    link_col = first_existing(df, ["link", "url", "fronturl"])
    if not link_col:
        return pd.DataFrame(columns=["media", "company", "title", "link", "date", "category", "text", "source_file"])
    result = df.copy()
    result["link"] = result[link_col].astype(str).str.strip()
    if "source_file" not in result.columns:
        result["source_file"] = str(source_path)
    if "media" not in result.columns:
        result["media"] = "unknown"
    for col in ["company", "title", "date", "category", "text"]:
        if col not in result.columns:
            result[col] = pd.NA
    company_was_present = ~is_blank(result["company"])
    if "company_source" not in result.columns:
        result["company_source"] = pd.NA
    result.loc[company_was_present & is_blank(result["company_source"]), "company_source"] = "article_metadata"

    # Historical Interfax files used the query/company name as source_file.
    # merge_all_news.py did not include source_file among company aliases.
    if "source_file" in result.columns:
        missing_interfax_company = result["media"].eq("interfax") & is_blank(result["company"])
        result.loc[missing_interfax_company, "company"] = result.loc[
            missing_interfax_company, "source_file"
        ].map(lambda value: normalize_company_value(Path(str(value)).stem))
        result.loc[missing_interfax_company, "company_source"] = "article_source_file"

    if "search_query" not in result.columns:
        result["search_query"] = pd.NA
    result["search_query"] = result["search_query"].mask(is_blank(result["search_query"]), result["company"])
    for col in ["project", "query_profile"]:
        if col not in result.columns:
            result[col] = pd.NA
    result = result[result["link"].notna() & (result["link"] != "") & (result["link"] != "nan")]
    return result


def deduplicate_articles(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    result = df.copy()
    result["date"] = parse_dates(result["date"])
    result["_text_len"] = result["text"].fillna("").astype(str).str.len()
    metadata_cols = ["company", "search_query", "date", "title", "category"]
    result["_metadata_score"] = sum((~is_blank(result[col])).astype(int) for col in metadata_cols)
    result["_link_norm"] = normalize_link_series(result["link"])
    result = result.sort_values(
        ["media", "_link_norm", "_text_len", "_metadata_score"],
        ascending=[True, True, False, False],
    )
    result = result.drop_duplicates(subset=["media", "_link_norm"], keep="first")
    result = result.drop(columns=["_text_len", "_metadata_score", "_link_norm"]).sort_values(
        ["media", "date"], ascending=[True, False]
    )
    return result.reset_index(drop=True)


def enrich_articles_from_links(articles: pd.DataFrame, link_records: pd.DataFrame) -> pd.DataFrame:
    """Fill missing article metadata from the original search/link results."""
    if articles.empty:
        return articles
    result = articles.copy()
    result["date"] = parse_dates(result["date"])
    result["_link_norm"] = normalize_link_series(result["link"])

    if "date_source" not in result.columns:
        result["date_source"] = pd.NA
    result.loc[result["date"].notna() & is_blank(result["date_source"]), "date_source"] = "article_metadata"
    if "company_source" not in result.columns:
        result["company_source"] = pd.NA
    result.loc[~is_blank(result["company"]) & is_blank(result["company_source"]), "company_source"] = "article_metadata"

    if link_records.empty:
        return fill_dates_from_urls(result.drop(columns=["_link_norm"]))

    links = link_records.copy()
    if "_link_norm" not in links.columns:
        links["_link_norm"] = normalize_link_series(links["link"])
    metadata_cols = ["company", "search_query", "date", "title", "category", "project", "query_profile"]
    links["_metadata_score"] = sum((~is_blank(links[col])).astype(int) for col in metadata_cols)
    links = links.sort_values(["media", "_link_norm", "_metadata_score"], ascending=[True, True, False])
    links = links.drop_duplicates(["media", "_link_norm"], keep="first")
    link_enrichment_cols = [*metadata_cols, "date_source", "date_precision"]
    for col in ["date_source", "date_precision"]:
        if col not in links.columns:
            links[col] = pd.NA
    links = links[["media", "_link_norm", *link_enrichment_cols]].rename(
        columns={col: f"{col}__link" for col in link_enrichment_cols}
    )
    result = result.merge(links, how="left", on=["media", "_link_norm"], validate="many_to_one")

    for col in metadata_cols:
        if col not in result.columns:
            result[col] = pd.NA
        link_col = f"{col}__link"
        missing = is_blank(result[col]) if col != "date" else result[col].isna()
        available = ~is_blank(result[link_col]) if col != "date" else result[link_col].notna()
        fill = missing & available
        result.loc[fill, col] = result.loc[fill, link_col]
        if col == "company":
            result.loc[fill, "company_source"] = "link_search_query"
        elif col == "date":
            link_source = result.loc[fill, "date_source__link"].fillna("link_metadata")
            link_precision = result.loc[fill, "date_precision__link"].fillna("timestamp_or_day")
            result.loc[fill, "date_source"] = link_source
            result.loc[fill, "date_precision"] = link_precision
        result = result.drop(columns=[link_col])

    result = result.drop(columns=["date_source__link", "date_precision__link"])

    return fill_dates_from_urls(result.drop(columns=["_link_norm"]))


def build_article_company_map(articles: pd.DataFrame, link_records: pd.DataFrame) -> pd.DataFrame:
    """Build a many-to-many article-to-query/company bridge for issuer-level research."""
    columns = [
        "media",
        "link",
        "company",
        "search_query",
        "date",
        "date_source",
        "date_precision",
        "title",
        "category",
        "project",
        "query_profile",
        "source_file",
        "association_source",
    ]
    article_keys = articles[["media", "link"]].copy()
    article_keys["_link_norm"] = normalize_link_series(article_keys["link"])
    article_keys = article_keys.drop_duplicates(["media", "_link_norm"])

    article_map = articles.copy()
    for col in columns[:-1]:
        if col not in article_map.columns:
            article_map[col] = pd.NA
    article_map = article_map[columns[:-1]]
    article_map["association_source"] = "article_metadata"

    link_map = link_records.copy()
    for col in columns[:-1]:
        if col not in link_map.columns:
            link_map[col] = pd.NA
    if "_link_norm" not in link_map.columns:
        link_map["_link_norm"] = normalize_link_series(link_map["link"])
    link_map = link_map.merge(
        article_keys[["media", "_link_norm"]], how="inner", on=["media", "_link_norm"], validate="many_to_one"
    )
    link_map = link_map[columns[:-1]]
    link_map["association_source"] = "link_search_result"

    result = pd.concat([link_map, article_map], ignore_index=True, sort=False)
    result = result[~is_blank(result["company"])].copy()
    result["company"] = result["company"].map(normalize_company_value)
    result["search_query"] = result["search_query"].mask(is_blank(result["search_query"]), result["company"])
    result["date"] = parse_dates(result["date"])
    result["_link_norm"] = normalize_link_series(result["link"])
    result["_company_norm"] = result["company"].str.lower()
    metadata_cols = ["search_query", "date", "title", "category", "project", "query_profile"]
    result["_metadata_score"] = sum((~is_blank(result[col])).astype(int) for col in metadata_cols)
    result["_source_priority"] = result["association_source"].eq("link_search_result").astype(int)
    result = result.sort_values(
        ["media", "_link_norm", "_company_norm", "_metadata_score", "_source_priority"],
        ascending=[True, True, True, False, False],
    )
    result = result.drop_duplicates(["media", "_link_norm", "_company_norm"], keep="first")
    result = result.drop(columns=["_link_norm", "_company_norm", "_metadata_score", "_source_priority"])
    return result.sort_values(["media", "company", "date"], ascending=[True, True, False]).reset_index(drop=True)


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    df.to_parquet(temp_path, index=False)
    temp_path.replace(path)


def print_metadata_coverage(name: str, df: pd.DataFrame) -> None:
    print(f"{name}: rows={len(df):,}")
    for media, part in df.groupby("media", dropna=False):
        company_count = int((~is_blank(part["company"])).sum())
        date_count = int(part["date"].notna().sum())
        category_count = int((~is_blank(part["category"])).sum())
        print(
            f"  {media}: rows={len(part):,}, company={company_count:,}, "
            f"date={date_count:,}, category={category_count:,}"
        )


def merge_articles_from_unified_and_runs(
    sample_files: int | None = None,
    include_smoke_runs: bool = False,
    include_run_articles: bool = True,
) -> pd.DataFrame:
    """Use the already merged historical dataset and append only new run files.

    Rebuilding history from all per-company page parquet files is possible but slow
    and disk-heavy because the historical Kommersant files are very large.
    """
    parts = []
    unified_path = PATHS.news_merged_dir / "all_news_merged_unified.parquet"
    if unified_path.exists():
        base = pd.read_parquet(unified_path)
        normalized_base = normalize_unified_articles(base, unified_path)
        if not normalized_base.empty:
            parts.append(normalized_base)
        print(f"historical unified articles: rows={len(normalized_base)}, path={unified_path}")
    else:
        print(f"historical unified articles not found: {unified_path}")

    run_files = (
        iter_article_files(
            include_history=False,
            include_runs=True,
            include_smoke_runs=include_smoke_runs,
        )
        if include_run_articles
        else []
    )
    if sample_files:
        run_files = run_files[:sample_files]
    for path in tqdm(run_files, desc="merge run articles"):
        try:
            df = read_article_parquet(path)
        except Exception as exc:
            print(f"skip unreadable parquet: {path} ({exc})")
            continue
        normalized = normalize_articles(df, path)
        if not normalized.empty:
            parts.append(normalized)

    if not parts:
        return pd.DataFrame(columns=["media", "company", "title", "link", "date", "category", "text", "source_file"])
    return deduplicate_articles(pd.concat(parts, ignore_index=True, sort=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge news_data history and run folders into deduplicated total parquet files.")
    parser.add_argument("--skip-articles", action="store_true")
    parser.add_argument("--skip-company-map", action="store_true")
    parser.add_argument(
        "--skip-run-articles",
        action="store_true",
        help="Repair/enrich the historical unified corpus without appending article page run files.",
    )
    parser.add_argument("--include-smoke-runs", action="store_true")
    parser.add_argument(
        "--rebuild-articles-from-pages",
        action="store_true",
        help="Slow full rebuild from every historical/run page parquet instead of using all_news_merged_unified.parquet + run files.",
    )
    parser.add_argument("--sample-files", type=int, default=None)
    args = parser.parse_args()

    PATHS.news_merged_dir.mkdir(parents=True, exist_ok=True)

    link_records = collect_link_records(
        sample_files=args.sample_files,
        include_smoke_runs=args.include_smoke_runs,
    )
    links = merge_links(
        sample_files=args.sample_files,
        include_smoke_runs=args.include_smoke_runs,
        records=link_records,
    )
    links_path = PATHS.news_merged_dir / "links_total.parquet"
    write_parquet_atomic(links, links_path)
    print(f"links_total: rows={len(links)}, path={links_path}")

    if not args.skip_articles:
        if args.rebuild_articles_from_pages:
            articles = merge_articles(sample_files=args.sample_files, include_smoke_runs=args.include_smoke_runs)
        else:
            articles = merge_articles_from_unified_and_runs(
                sample_files=args.sample_files,
                include_smoke_runs=args.include_smoke_runs,
                include_run_articles=not args.skip_run_articles,
            )
        articles = enrich_articles_from_links(articles, link_records)
        articles = deduplicate_articles(articles)
        articles_path = PATHS.news_merged_dir / "articles_total.parquet"
        write_parquet_atomic(articles, articles_path)
        print(f"articles_total: rows={len(articles)}, path={articles_path}")
        print_metadata_coverage("articles_total metadata coverage", articles)

        if not args.skip_company_map:
            company_map = build_article_company_map(articles, link_records)
            company_map_path = PATHS.news_merged_dir / "article_company_map.parquet"
            write_parquet_atomic(company_map, company_map_path)
            print(f"article_company_map: rows={len(company_map)}, path={company_map_path}")


if __name__ == "__main__":
    main()
