#!/usr/bin/env python3
"""Restore missing RBC publication dates without modifying the source parquet.

The historical merge can contain RBC articles whose extracted page has a valid
``publish_date`` but whose merged date is empty.  This script first restores
only those direct page dates.  For remaining RBC URLs, it uses the timestamp
embedded in a Mongo/BSON ObjectId *only* when the URL belongs to rbc.ru and the
decoded value is plausible.  The latter is explicitly marked as inferred in
``date_source`` and ``date_precision``.

The original articles_total.parquet is never overwritten.  A JSON audit is
written next to the new parquet so downstream analysis can distinguish direct
metadata from URL-derived timestamps and retain a count of unresolved rows.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import _bootstrap  # noqa: F401

from dissertation_pipeline.paths import PATHS


OBJECT_ID_RE = re.compile(r"(?<![0-9a-f])([0-9a-f]{24})(?![0-9a-f])", re.IGNORECASE)
PLACEHOLDER_VALUES = {"", "nan", "none", "nat", "<na>"}
RAW_LINK_COLUMNS = ("fronturl", "link", "url")
RAW_DATE_COLUMNS = ("publish_date", "date", "timestamp")


def normalize_link(value: object) -> str:
    """Canonical URL key used by the merged and raw-page layers."""

    if value is None or pd.isna(value):
        return ""
    return str(value).strip().rstrip("/").lower()


def is_rbc_url(link: object) -> bool:
    try:
        hostname = urlparse(str(link).strip()).hostname
    except ValueError:
        return False
    if not hostname:
        return False
    hostname = hostname.lower()
    return hostname == "rbc.ru" or hostname.endswith(".rbc.ru")


def parse_date_values(values: Iterable[object]) -> list[datetime | None]:
    """Parse mixed raw-page date formats as naive UTC datetimes."""

    series = pd.Series(list(values), dtype="string").str.strip()
    series = series.mask(series.str.lower().isin(PLACEHOLDER_VALUES))
    try:
        parsed = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    except (TypeError, ValueError):  # pandas < 2.0 or a non-mixed input
        parsed = pd.to_datetime(series, errors="coerce", utc=True)

    # Older pandas accepts ``format='mixed'`` as a literal format rather than
    # its modern mixed-format sentinel and quietly returns NaT for every row.
    # Reparse the unresolved values one-by-one: raw metadata batches can mix
    # `...UTC` and `...+03:00`, while scalar parsing is format-independent.
    unresolved = parsed.isna() & series.notna()
    if unresolved.any():
        for position in series.index[unresolved]:
            parsed.at[position] = pd.to_datetime(
                series.at[position], errors="coerce", utc=True
            )

    result: list[datetime | None] = []
    for value in parsed:
        if pd.isna(value):
            result.append(None)
            continue
        result.append(value.to_pydatetime().replace(tzinfo=None))
    return result


def choose_first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    existing = set(columns)
    return next((name for name in candidates if name in existing), None)


def candidate_page_files(raw_root: Path, source_files: Iterable[object]) -> list[Path]:
    """Prefer explicitly referenced sources, then scan all retained RBC pages."""

    ordered: dict[Path, None] = {}
    for raw_path in source_files:
        text = str(raw_path).strip()
        if not text.startswith("/"):
            continue
        path = Path(text)
        if path.is_file() and path.suffix.lower() == ".parquet":
            ordered[path] = None

    if raw_root.is_dir():
        for path in sorted(raw_root.rglob("*.parquet")):
            if path.parent.name.lower() == "rbc":
                ordered[path] = None
    return list(ordered)


def collect_raw_page_dates(
    files: Iterable[Path],
    wanted_links: set[str],
) -> tuple[dict[str, datetime], dict[str, str], dict[str, int]]:
    """Find direct publish dates in saved RBC page parquet files.

    Only URL/date columns are read.  The function deliberately never creates a
    date from title text, file names, crawl timestamps, or company-level hints.
    """

    dates: dict[str, datetime] = {}
    source_for_link: dict[str, str] = {}
    stats: Counter[str] = Counter()

    for file_path in files:
        stats["files_considered"] += 1
        try:
            parquet_file = pq.ParquetFile(file_path)
            names = parquet_file.schema_arrow.names
            link_column = choose_first_existing(names, RAW_LINK_COLUMNS)
            date_column = choose_first_existing(names, RAW_DATE_COLUMNS)
            if not link_column or not date_column:
                stats["files_without_link_or_date"] += 1
                continue

            stats["files_read"] += 1
            for batch in parquet_file.iter_batches(
                columns=[link_column, date_column], batch_size=25_000
            ):
                # Parquet readers may return requested fields in their physical
                # schema order (publish_date precedes fronturl in RBC files),
                # so never rely on the order supplied in ``columns``.
                links = [
                    normalize_link(value)
                    for value in batch.column(batch.schema.get_field_index(link_column)).to_pylist()
                ]
                raw_dates = parse_date_values(
                    batch.column(batch.schema.get_field_index(date_column)).to_pylist()
                )
                stats["raw_rows_scanned"] += len(links)

                for link, date_value in zip(links, raw_dates):
                    if not link or link not in wanted_links or date_value is None:
                        continue
                    existing = dates.get(link)
                    if existing is None:
                        dates[link] = date_value
                        source_for_link[link] = str(file_path)
                        stats["matched_links"] += 1
                    elif existing != date_value:
                        # Retain the first direct metadata value and make the
                        # discrepancy visible in the audit rather than guessing.
                        stats["conflicting_direct_dates"] += 1
        except Exception:
            stats["unreadable_files"] += 1

    return dates, source_for_link, dict(stats)


def decode_object_id_date(link: object, upper_bound: datetime) -> datetime | None:
    """Decode the four-byte timestamp in a valid-looking RBC Mongo ObjectId."""

    if not is_rbc_url(link):
        return None
    match = OBJECT_ID_RE.search(str(link))
    if not match:
        return None
    try:
        value = datetime.fromtimestamp(int(match.group(1)[:8], 16), tz=timezone.utc).replace(
            tzinfo=None
        )
    except (OverflowError, OSError, ValueError):
        return None
    if not datetime(2000, 1, 1) <= value <= upper_bound:
        return None
    return value


def output_batch(
    batch: pa.RecordBatch,
    schema: pa.Schema,
    raw_dates: dict[str, datetime],
    raw_sources: dict[str, str],
    upper_bound: datetime,
    recovery_counts: Counter[str],
    remaining_hosts: Counter[str],
) -> pa.RecordBatch:
    """Replace missing RBC dates in one batch while preserving the full schema."""

    required = ("media", "link", "date", "date_source", "date_precision")
    missing_fields = [field for field in required if field not in schema.names]
    if missing_fields:
        raise ValueError(f"Input parquet lacks required fields: {missing_fields}")

    positions = {name: schema.get_field_index(name) for name in required}
    media_values = batch.column(positions["media"]).to_pylist()
    link_values = batch.column(positions["link"]).to_pylist()
    date_values = batch.column(positions["date"]).to_pylist()
    source_values = batch.column(positions["date_source"]).to_pylist()
    precision_values = batch.column(positions["date_precision"]).to_pylist()
    changed = False

    for row_number, (media, link, date_value) in enumerate(
        zip(media_values, link_values, date_values)
    ):
        if str(media).strip().lower() != "rbc" or date_value is not None:
            continue

        normalized = normalize_link(link)
        direct_date = raw_dates.get(normalized)
        if direct_date is not None:
            date_values[row_number] = direct_date
            source_values[row_number] = "rbc_saved_page_publish_date"
            precision_values[row_number] = "timestamp_or_day"
            recovery_counts["direct_page_metadata"] += 1
            changed = True
            continue

        object_id_date = decode_object_id_date(link, upper_bound)
        if object_id_date is not None:
            date_values[row_number] = object_id_date
            source_values[row_number] = "rbc_object_id_timestamp"
            precision_values[row_number] = "inferred_timestamp"
            recovery_counts["object_id_timestamp"] += 1
            changed = True
            continue

        try:
            hostname = urlparse(str(link)).hostname or "<no-host>"
        except ValueError:
            hostname = "<invalid-url>"
        remaining_hosts[hostname.lower()] += 1

    if not changed:
        return batch

    arrays = list(batch.columns)
    arrays[positions["date"]] = pa.array(date_values, type=schema.field("date").type)
    arrays[positions["date_source"]] = pa.array(
        source_values, type=schema.field("date_source").type
    )
    arrays[positions["date_precision"]] = pa.array(
        precision_values, type=schema.field("date_precision").type
    )
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PATHS.news_merged_dir / "articles_total.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PATHS.news_merged_dir / "articles_total_rbc_dates_restored.parquet",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=PATHS.news_merged_dir / "articles_total_rbc_dates_restored.audit.json",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=PATHS.news_data_dir / "article_pages",
        help="Root containing retained raw article-page parquet files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5_000,
        help="Rows rewritten at a time; keep this modest because text is large.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output/audit created by this script.",
    )
    parser.add_argument(
        "--referenced-files-only",
        action="store_true",
        help="Skip the full retained RBC-page scan (faster, lower coverage).",
    )
    args = parser.parse_args()

    input_path = args.input.resolve(strict=True)
    output_path = args.output.expanduser().resolve()
    audit_path = args.audit.expanduser().resolve()
    if input_path == output_path:
        raise SystemExit("Refusing to overwrite the source parquet; choose a new --output path.")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    for path in (output_path, audit_path):
        if path.exists() and not args.overwrite:
            raise SystemExit(f"Output already exists: {path}. Use --overwrite to replace it.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    source = pq.ParquetFile(input_path)
    schema = source.schema_arrow
    required_fields = {"media", "link", "date", "date_source", "date_precision", "source_file"}
    absent_fields = sorted(required_fields - set(schema.names))
    if absent_fields:
        raise SystemExit(f"Input parquet lacks required fields: {absent_fields}")

    index_table = pq.read_table(input_path, columns=["media", "link", "date", "source_file"])
    index_frame = index_table.to_pandas()
    missing_rbc = index_frame.loc[
        (index_frame["media"].astype("string").str.lower() == "rbc")
        & index_frame["date"].isna()
    ].copy()
    wanted_links = {normalize_link(value) for value in missing_rbc["link"]}
    wanted_links.discard("")
    referenced = candidate_page_files(Path("/does-not-exist"), missing_rbc["source_file"])
    if args.referenced_files_only:
        raw_files = referenced
    else:
        raw_files = candidate_page_files(args.raw_root, missing_rbc["source_file"])

    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "rows": source.metadata.num_rows,
                "rbc_missing_dates_before": len(missing_rbc),
                "unique_missing_rbc_links": len(wanted_links),
                "referenced_raw_files": len(referenced),
                "raw_files_to_scan": len(raw_files),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    raw_dates, raw_sources, raw_scan_stats = collect_raw_page_dates(raw_files, wanted_links)
    print(
        json.dumps(
            {"raw_direct_dates_found": len(raw_dates), "raw_scan": raw_scan_stats},
            ensure_ascii=False,
        ),
        flush=True,
    )

    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    if partial_path.exists():
        if not args.overwrite:
            raise SystemExit(
                f"Partial output exists: {partial_path}. Inspect it or rerun with --overwrite."
            )
        partial_path.unlink()

    recovery_counts: Counter[str] = Counter()
    remaining_hosts: Counter[str] = Counter()
    upper_bound = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=1)
    with pq.ParquetWriter(partial_path, schema=schema, compression="zstd") as writer:
        for batch in source.iter_batches(batch_size=args.batch_size):
            writer.write_batch(
                output_batch(
                    batch,
                    schema,
                    raw_dates,
                    raw_sources,
                    upper_bound,
                    recovery_counts,
                    remaining_hosts,
                )
            )

    os.replace(partial_path, output_path)
    output_dates = pq.read_table(output_path, columns=["media", "date"]).to_pandas()
    rbc_missing_after = int(
        (
            (output_dates["media"].astype("string").str.lower() == "rbc")
            & output_dates["date"].isna()
        ).sum()
    )
    audit: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "original_rows": int(source.metadata.num_rows),
        "rbc_missing_dates_before": int(len(missing_rbc)),
        "rbc_missing_dates_after": rbc_missing_after,
        "recovered_rows": dict(sorted(recovery_counts.items())),
        "direct_raw_dates_available_for_unique_links": len(raw_dates),
        "raw_source_paths_for_unique_links": len(raw_sources),
        "referenced_raw_files": len(referenced),
        "raw_files_scanned": len(raw_files),
        "raw_page_scan": raw_scan_stats,
        "unresolved_by_host": dict(remaining_hosts.most_common(20)),
        "method_notes": [
            "Direct page publish_date takes precedence over a URL-derived timestamp.",
            "rbc_object_id_timestamp is inferred from a Mongo/BSON ObjectId and remains explicitly marked.",
            "Rows still missing were left missing; no date was guessed from title, company, crawl time, or filename.",
        ],
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
