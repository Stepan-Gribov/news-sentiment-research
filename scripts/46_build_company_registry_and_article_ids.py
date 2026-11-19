#!/usr/bin/env python3
"""Create a company-ID registry and collapse article-company pairs to one row/article."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import openpyxl
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MERGED_DIR = Path.home() / "dev/dissertation_supp/news_data/merged"
DEFAULT_ARTICLE_COMPANY_MAP = MERGED_DIR / "article_company_map_algorithmic_v3.parquet"
DEFAULT_COMPANIES = max(
    PROJECT_ROOT.glob("outputs/*/*classification*.xlsx"),
    key=lambda path: path.stat().st_mtime,
    default=None,
)
DEFAULT_COMPANY_REGISTRY = MERGED_DIR / "company_registry_v1.parquet"
DEFAULT_ARTICLE_IDS = MERGED_DIR / "article_company_ids_v1.parquet"
SHEET_NAME = "По компаниям"

COMPANY_SCHEMA = pa.schema(
    [
        ("company_id", pa.string()),
        ("company_name", pa.string()),
        ("isins", pa.string()),
        ("industry", pa.string()),
        ("product", pa.string()),
    ]
)
ARTICLE_SCHEMA = pa.schema(
    [
        ("article_id", pa.string()),
        ("date", pa.timestamp("ns")),
        ("media", pa.string()),
        ("title", pa.string()),
        ("company_ids", pa.string()),
        ("company_count", pa.int32()),
        ("company_match_signals_json", pa.string()),
        ("article_url", pa.string()),
    ]
)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return unicodedata.normalize("NFKC", str(value)).casefold().replace("ё", "е").strip()


def company_id_for(name: str, isin_values: Any) -> str:
    """Keep IDs byte-for-byte compatible with article_company_map_algorithmic_v3."""
    identity = f"{normalize_text(name)}\t{normalize_text(isin_values)}".encode("utf-8")
    return "c_" + hashlib.blake2b(identity, digest_size=12).hexdigest()


def display_name_rank(name: str) -> tuple[int, float, int, str]:
    letters = [character for character in name if character.isalpha()]
    all_caps = bool(letters) and all(character.isupper() for character in letters)
    uppercase_share = sum(character.isupper() for character in letters) / max(len(letters), 1)
    return (int(all_caps), uppercase_share, len(name), name)


def merge_description_values(values: list[str]) -> str:
    merged: dict[str, str] = {}
    for value in values:
        for term in value.split(";"):
            term = term.strip()
            if term:
                merged.setdefault(normalize_text(term), term)
    return "; ".join(merged[key] for key in sorted(merged))


def load_company_registry(path: Path) -> list[dict[str, str]]:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    if SHEET_NAME not in workbook.sheetnames:
        raise ValueError(f"В книге нет листа «{SHEET_NAME}»: {path}")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in workbook[SHEET_NAME].iter_rows(min_row=2, max_col=5, values_only=True):
        name = str(row[0] or "").strip()
        if not name:
            continue
        isins = str(row[2] or "").strip()
        record = {
            "company_id": company_id_for(name, row[2]),
            "company_name": name,
            "isins": isins,
            "industry": str(row[3] or "").strip(),
            "product": str(row[4] or "").strip(),
        }
        grouped[record["company_id"]].append(record)

    workbook.close()
    registry = []
    for company_id, records in grouped.items():
        names = sorted({record["company_name"] for record in records}, key=display_name_rank)
        registry.append(
            {
                "company_id": company_id,
                "company_name": names[0],
                "isins": records[0]["isins"],
                "industry": merge_description_values([record["industry"] for record in records]),
                "product": merge_description_values([record["product"] for record in records]),
            }
        )
    return sorted(registry, key=lambda record: record["company_id"])


def merge_article_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    companies: dict[str, dict[str, Any]] = {}
    for row in rows:
        for field in ("date", "media", "title", "article_url"):
            if row[field] != first[field]:
                raise ValueError(f"Conflicting {field} for article {first['article_id']}")
        company_id = row["company_id"]
        entry = companies.setdefault(company_id, {"signals": set()})
        entry["signals"].update(filter(None, str(row["match_signal"] or "").split(";")))

    ordered_ids = sorted(companies)
    details = []
    for company_id in ordered_ids:
        entry = companies[company_id]
        ordered_signals = [signal for signal in ("name", "industry", "product") if signal in entry["signals"]]
        details.append(
            {
                "company_id": company_id,
                "match_signal": ";".join(ordered_signals),
            }
        )

    return {
        "article_id": first["article_id"],
        "date": first["date"],
        "media": first["media"],
        "title": first["title"],
        "company_ids": ",".join(ordered_ids),
        "company_count": len(ordered_ids),
        "company_match_signals_json": json.dumps(details, ensure_ascii=False, separators=(",", ":")),
        "article_url": first["article_url"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--article-company-map", type=Path, default=DEFAULT_ARTICLE_COMPANY_MAP)
    parser.add_argument("--companies", type=Path, default=DEFAULT_COMPANIES)
    parser.add_argument("--company-registry-input", type=Path, help="Use existing canonical IDs and all description fields from a Parquet registry; do not recreate IDs from names/ISINs")
    parser.add_argument("--company-registry", type=Path, default=DEFAULT_COMPANY_REGISTRY)
    parser.add_argument("--article-ids-output", type=Path, default=DEFAULT_ARTICLE_IDS)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = args.article_company_map.expanduser().resolve()
    company_source = args.company_registry_input or args.companies
    if company_source is None:
        raise ValueError("Provide --company-registry-input or --companies")
    company_path = company_source.expanduser().resolve()
    registry_path = args.company_registry.expanduser().resolve()
    article_output_path = args.article_ids_output.expanduser().resolve()
    if {source_path, company_path} & {registry_path, article_output_path}:
        raise ValueError("Output paths must not overwrite either input")
    for path in (source_path, company_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.batch_size < 1:
        raise ValueError("--batch-size должен быть положительным")
    audit_path = article_output_path.with_suffix(article_output_path.suffix + ".audit.json")
    existing = [path for path in (registry_path, article_output_path, audit_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Уже существуют выходные файлы (для замены укажи --overwrite): " + ", ".join(map(str, existing)))

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    article_output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.company_registry_input:
        registry_table = pq.read_table(company_path)
        if "company_name" not in registry_table.column_names and "company" in registry_table.column_names:
            registry_table = registry_table.rename_columns(["company_name" if name == "company" else name for name in registry_table.column_names])
        for required in ("company_id", "company_name"):
            if required not in registry_table.column_names:
                raise ValueError(f"Canonical registry missing {required}")
        companies = registry_table.to_pylist()
        if len({r["company_id"] for r in companies}) != len(companies) or any(not r["company_id"] for r in companies):
            raise ValueError("Canonical registry has duplicate or missing IDs")
        if any(r.get("status", "active") != "active" for r in companies):
            raise ValueError("Use only the active canonical registry")
    else:
        companies = load_company_registry(company_path)
        registry_table = pa.Table.from_pylist(companies, schema=COMPANY_SCHEMA)
    company_ids = {record["company_id"] for record in companies}
    source = pq.ParquetFile(source_path)
    started = time.monotonic()
    seen_article_ids: set[str] = set()
    mapped_company_ids: set[str] = set()
    mapping_pairs = 0
    output_rows = 0
    output_pairs = 0
    next_progress = 1_000_000
    buffer: list[dict[str, Any]] = []
    current_article_rows: list[dict[str, Any]] = []
    current_article_id: str | None = None
    writer: pq.ParquetWriter | None = None

    with tempfile.TemporaryDirectory(prefix=".company_id_aggregate_", dir=article_output_path.parent) as tmp_dir:
        staged_registry = Path(tmp_dir) / registry_path.name
        staged_articles = Path(tmp_dir) / article_output_path.name
        pq.write_table(registry_table, staged_registry, compression="snappy", use_dictionary=["company_id", "company_name"])
        writer = pq.ParquetWriter(staged_articles, ARTICLE_SCHEMA, compression="snappy", use_dictionary=["media"])

        def flush_article() -> None:
            nonlocal output_rows, output_pairs, buffer, current_article_rows
            if not current_article_rows:
                return
            aggregated = merge_article_rows(current_article_rows)
            buffer.append(aggregated)
            output_rows += 1
            output_pairs += aggregated["company_count"]
            current_article_rows = []
            if len(buffer) >= 5_000:
                writer.write_table(pa.Table.from_pylist(buffer, schema=ARTICLE_SCHEMA))
                buffer = []

        for batch in source.iter_batches(
            columns=["article_id", "date", "media", "title", "company_id", "match_signal", "article_url"],
            batch_size=args.batch_size,
        ):
            data = batch.to_pydict()
            for i in range(batch.num_rows):
                row = {
                    "article_id": data["article_id"][i],
                    "date": data["date"][i],
                    "media": data["media"][i],
                    "title": data["title"][i],
                    "company_id": data["company_id"][i],
                    "match_signal": data["match_signal"][i],
                    "article_url": data["article_url"][i],
                }
                article_id = row["article_id"]
                if not article_id:
                    raise ValueError("В исходной карте встретилась строка без article_id")
                if article_id != current_article_id:
                    flush_article()
                    if article_id in seen_article_ids:
                        raise ValueError(f"Статья {article_id} встретилась не подряд; небезопасно схлопывать поток")
                    if article_id:
                        seen_article_ids.add(article_id)
                    current_article_id = article_id
                    current_article_rows = [row]
                else:
                    current_article_rows.append(row)
                mapped_company_ids.add(row["company_id"])
                mapping_pairs += 1
            if mapping_pairs >= next_progress:
                print(json.dumps({"processed_pairs": mapping_pairs, "total_pairs": source.metadata.num_rows, "articles_output": output_rows, "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
                next_progress += 1_000_000

        flush_article()
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=ARTICLE_SCHEMA))
        writer.close()
        writer = None

        unknown_ids = mapped_company_ids - company_ids
        if unknown_ids:
            raise ValueError(f"В карте есть {len(unknown_ids)} company_id, отсутствующих в справочнике: {sorted(unknown_ids)[:5]}")
        if args.company_registry_input and output_pairs != mapping_pairs:
            raise ValueError("Input pairs were not unique; use the cleaned article/company mapping")
        if mapping_pairs != source.metadata.num_rows:
            raise ValueError("Not all source pairs were processed")
        if pq.ParquetFile(staged_articles).metadata.num_rows != output_rows:
            raise ValueError("Written article count mismatch")

        os.replace(staged_registry, registry_path)
        os.replace(staged_articles, article_output_path)

    elapsed = time.monotonic() - started
    audit = {
        "article_company_source": str(source_path),
        "company_source": str(company_path),
        "company_registry": str(registry_path),
        "article_ids_output": str(article_output_path),
        "companies_in_registry": len(companies),
        "companies_matched": len(mapped_company_ids),
        "article_company_pairs_in_source": mapping_pairs,
        "article_company_pairs_in_output": output_pairs,
        "duplicate_source_pairs_collapsed": mapping_pairs - output_pairs,
        "unique_articles_output": output_rows,
        "rows_per_article": "one",
        "company_ids_delimiter": ",",
        "unmatched_companies_in_registry": len(company_ids - mapped_company_ids),
        "company_id_rule": "Existing canonical IDs preserved verbatim from registry input" if args.company_registry_input else "BLAKE2b-96(normalized company name + source ISIN field); IDs align with article_company_map_algorithmic_v3",
        "registry_description_fields": registry_table.column_names,
        "elapsed_seconds": round(elapsed, 2),
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**audit, "audit": str(audit_path)}, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
