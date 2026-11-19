from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


SOURCES = {
    "Коммерсантъ": ("kommersant_archive", "doc_id"),
    "ПРАЙМ": ("prime_archive", "article_id"),
    "Интерфакс": ("interfax_archive", "article_id"),
    "РБК": ("rbc_archive_content", "material_id"),
}
DATE_COLUMNS = ("published_at", "publication_date", "timestamp")


def parse_dates(values: list[object]) -> pd.Series:
    series = pd.Series(values, dtype="object")
    try:
        parsed = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    except TypeError:
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed[(parsed >= "1980-01-01") & (parsed < "2100-01-01")]


def scan_source(name: str, root: str, id_column: str) -> dict:
    files = sorted(Path(root).rglob("*.parquet"))
    rows = 0
    text_rows = 0
    dated_rows = 0
    ids: set[str] = set()
    text_ids: set[str] = set()
    dated_ids: set[str] = set()
    statuses: Counter[str] = Counter()
    min_date = None
    max_date = None
    errors: list[str] = []

    for index, file in enumerate(files, 1):
        try:
            parquet = pq.ParquetFile(file)
            names = set(parquet.schema.names)
            columns = [c for c in (id_column, "article_text_extracted", "parse_status", *DATE_COLUMNS) if c in names]
            for batch in parquet.iter_batches(columns=columns, batch_size=65536):
                data = batch.to_pydict()
                batch_rows = batch.num_rows
                rows += batch_rows

                raw_ids = data.get(id_column, [None] * batch_rows)
                normalized_ids = [str(v).strip() if v is not None else "" for v in raw_ids]
                ids.update(v for v in normalized_ids if v)

                texts = data.get("article_text_extracted", [None] * batch_rows)
                has_text = [isinstance(v, str) and bool(v.strip()) for v in texts]
                text_rows += sum(has_text)
                text_ids.update(v for v, ok in zip(normalized_ids, has_text) if v and ok)

                for value in data.get("parse_status", [None] * batch_rows):
                    statuses[str(value) if value is not None else "null"] += 1

                selected_dates: list[object] = [None] * batch_rows
                for column in DATE_COLUMNS:
                    values = data.get(column)
                    if values is None:
                        continue
                    for pos, value in enumerate(values):
                        if selected_dates[pos] in (None, "") and value not in (None, ""):
                            selected_dates[pos] = value
                parsed = parse_dates(selected_dates)
                dated_rows += int(parsed.notna().sum())
                if not parsed.empty:
                    current_min = parsed.min()
                    current_max = parsed.max()
                    min_date = current_min if min_date is None or current_min < min_date else min_date
                    max_date = current_max if max_date is None or current_max > max_date else max_date
                    for pos in parsed.index:
                        value = normalized_ids[int(pos)]
                        if value:
                            dated_ids.add(value)
        except Exception as exc:  # keep the audit running and report every unreadable shard
            errors.append(f"{file}: {type(exc).__name__}: {exc}")
        if index % 1000 == 0 or index == len(files):
            print(f"{name}: {index}/{len(files)} files", flush=True)

    return {
        "media": name,
        "root": root,
        "parquet_files": len(files),
        "rows": rows,
        "unique_articles": len(ids),
        "rows_with_text": text_rows,
        "unique_articles_with_text": len(text_ids),
        "rows_with_date": dated_rows,
        "unique_articles_with_date": len(dated_ids),
        "coverage_start": min_date.date().isoformat() if min_date is not None else None,
        "coverage_end": max_date.date().isoformat() if max_date is not None else None,
        "parse_status": dict(statuses.most_common()),
        "read_errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data_root = Path(args.data_root)

    results = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(scan_source, name, str(data_root / folder), id_column): name
            for name, (folder, id_column) in SOURCES.items()
        }
        for future in as_completed(futures):
            results.append(future.result())

    order = {name: pos for pos, name in enumerate(SOURCES)}
    results.sort(key=lambda row: order[row["media"]])
    payload = {
        "sources": results,
        "totals": {
            "rows": sum(row["rows"] for row in results),
            "unique_articles": sum(row["unique_articles"] for row in results),
            "rows_with_text": sum(row["rows_with_text"] for row in results),
            "unique_articles_with_text": sum(row["unique_articles_with_text"] for row in results),
            "parquet_files": sum(row["parquet_files"] for row in results),
            "read_errors": sum(len(row["read_errors"]) for row in results),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
