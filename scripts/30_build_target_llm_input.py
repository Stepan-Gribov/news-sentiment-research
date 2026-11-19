#!/usr/bin/env python3
"""Build reproducible article×company input for target-level LLM sentiment.

The script reads article text from the date-restored article parquet and target
companies from ``article_company_map.parquet``.  It keeps one record for each
unique ``media + link + target_company`` pair, rather than copying a generic
article-level label to every company.

Only an explicitly bounded leading text excerpt is sent to the local model.  A
title and the number of omitted characters remain in the record, so the exact
input specification is auditable and compatible with a finite context window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.parquet as pq

import _bootstrap  # noqa: F401

from dissertation_pipeline.paths import PATHS


INPUT_VERSION = "target-llm-input-v1"
WHITESPACE_RE = re.compile(r"\s+")
EMPTY_VALUES = {"", "nan", "none", "nat", "<na>"}


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return WHITESPACE_RE.sub(" ", str(value)).strip()


def normalize_media(value: object) -> str:
    return clean_text(value).lower()


def normalize_link(value: object) -> str:
    return clean_text(value).rstrip("/").lower()


def article_key(media: str, normalized_link: str) -> str:
    return f"{media}\x1f{normalized_link}"


def pair_id(media: str, normalized_link: str, company: str) -> str:
    payload = f"{INPUT_VERSION}\x1f{media}\x1f{normalized_link}\x1f{company}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def iso_date(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed.isoformat()


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = set(columns)
    return next((candidate for candidate in candidates if candidate in available), None)


def load_targets(map_path: Path) -> tuple[dict[str, set[str]], dict[str, int]]:
    """Return a de-duplicated target-company set for every article key."""

    parquet = pq.ParquetFile(map_path)
    media_column = first_existing(parquet.schema_arrow.names, ("media", "source_media"))
    link_column = first_existing(parquet.schema_arrow.names, ("link", "url", "fronturl"))
    company_column = first_existing(parquet.schema_arrow.names, ("company", "target_company"))
    if not all((media_column, link_column, company_column)):
        raise ValueError(
            "Company map needs media, link and company fields; found "
            f"{parquet.schema_arrow.names}"
        )

    targets: dict[str, set[str]] = {}
    stats: Counter[str] = Counter()
    for batch in parquet.iter_batches(
        columns=[media_column, link_column, company_column], batch_size=100_000
    ):
        media_values = batch.column(batch.schema.get_field_index(media_column)).to_pylist()
        link_values = batch.column(batch.schema.get_field_index(link_column)).to_pylist()
        company_values = batch.column(batch.schema.get_field_index(company_column)).to_pylist()
        stats["company_map_rows"] += len(media_values)
        for media_value, link_value, company_value in zip(
            media_values, link_values, company_values
        ):
            media = normalize_media(media_value)
            link = normalize_link(link_value)
            company = clean_text(company_value)
            if not media or not link or not company or company.lower() in EMPTY_VALUES:
                stats["company_map_rows_invalid"] += 1
                continue
            key = article_key(media, link)
            company_set = targets.setdefault(key, set())
            before = len(company_set)
            company_set.add(company)
            if len(company_set) == before:
                stats["duplicate_article_company_rows"] += 1

    stats["mapped_articles"] = len(targets)
    stats["unique_article_company_pairs"] = sum(len(values) for values in targets.values())
    stats["unique_target_companies"] = len(
        {company for values in targets.values() for company in values}
    )
    return targets, dict(stats)


def fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--articles",
        type=Path,
        default=PATHS.news_merged_dir / "articles_total_rbc_dates_restored.parquet",
    )
    parser.add_argument(
        "--company-map",
        type=Path,
        default=PATHS.news_merged_dir / "article_company_map.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PATHS.news_data_dir / "llm_inputs" / "target_llm_input_v1.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PATHS.news_data_dir / "llm_inputs" / "target_llm_input_v1.manifest.json",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=4_800,
        help="Deterministic leading body excerpt; title is stored separately.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5_000,
        help="Article rows read at once; article text can be large.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing generated input and manifest.",
    )
    args = parser.parse_args()

    if args.max_text_chars <= 0 or args.batch_size <= 0:
        raise SystemExit("--max-text-chars and --batch-size must be positive")

    articles_path = args.articles.expanduser().resolve(strict=True)
    company_map_path = args.company_map.expanduser().resolve(strict=True)
    output_path = args.output.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    if output_path == articles_path:
        raise SystemExit("Refusing to write JSONL over the source article parquet.")
    for path in (output_path, manifest_path):
        if path.exists() and not args.overwrite:
            raise SystemExit(f"Output exists: {path}. Use --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    targets, map_stats = load_targets(company_map_path)
    print(json.dumps({"company_map": map_stats}, ensure_ascii=False), flush=True)

    article_parquet = pq.ParquetFile(articles_path)
    required = ("media", "link", "text", "title", "date", "date_source", "date_precision")
    absent = [name for name in required if name not in article_parquet.schema_arrow.names]
    if absent:
        raise SystemExit(f"Article parquet lacks required fields: {absent}")

    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    if partial_path.exists():
        if not args.overwrite:
            raise SystemExit(
                f"Partial input exists: {partial_path}. Inspect it or rerun with --overwrite."
            )
        partial_path.unlink()

    stats: Counter[str] = Counter()
    targets_by_media: Counter[str] = Counter()
    with partial_path.open("w", encoding="utf-8") as destination:
        for batch in article_parquet.iter_batches(columns=list(required), batch_size=args.batch_size):
            values = {
                name: batch.column(batch.schema.get_field_index(name)).to_pylist() for name in required
            }
            for row_number in range(batch.num_rows):
                stats["articles_seen"] += 1
                media = normalize_media(values["media"][row_number])
                normalized_link = normalize_link(values["link"][row_number])
                companies = targets.get(article_key(media, normalized_link))
                if not companies:
                    stats["articles_without_company_map"] += 1
                    continue

                body = clean_text(values["text"][row_number])
                if not body:
                    stats["articles_without_text"] += 1
                    continue
                title = clean_text(values["title"][row_number])
                original_chars = len(body)
                excerpt = body[: args.max_text_chars].rstrip()
                truncated = original_chars > len(excerpt)
                stats["eligible_articles"] += 1
                if truncated:
                    stats["articles_with_truncated_text"] += 1

                for company in sorted(companies):
                    record = {
                        "id": pair_id(media, normalized_link, company),
                        "media": media,
                        "link": clean_text(values["link"][row_number]),
                        "target_company": company,
                        "published_at": iso_date(values["date"][row_number]),
                        "date_source": clean_text(values["date_source"][row_number]) or None,
                        "date_precision": clean_text(values["date_precision"][row_number]) or None,
                        "title": title,
                        "text": excerpt,
                        "text_chars_original": original_chars,
                        "text_truncated": truncated,
                    }
                    destination.write(
                        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                    stats["target_pairs_written"] += 1
                    targets_by_media[media] += 1

    os.replace(partial_path, output_path)
    manifest: dict[str, Any] = {
        "input_version": INPUT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "articles": fingerprint(articles_path),
        "company_map": fingerprint(company_map_path),
        "output": fingerprint(output_path),
        "max_text_chars": args.max_text_chars,
        "article_batch_size": args.batch_size,
        "statistics": dict(sorted(stats.items())),
        "pairs_by_media": dict(sorted(targets_by_media.items())),
        "company_map_statistics": map_stats,
        "notes": [
            "Each line is one unique article × target_company pair.",
            "Text is a deterministic leading excerpt; title remains separate.",
            "Date metadata is read from the restored article parquet, not the older mapping layer.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
