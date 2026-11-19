#!/usr/bin/env python3
"""Stream the remote news-search SQLite article table into mapper Parquet format.

Only the five columns needed by 45_map_articles_to_companies.py are exported;
the source SQLite database is opened read-only and is never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA = pa.schema(
    [
        ("media", pa.string()),
        ("title", pa.string()),
        ("link", pa.string()),
        ("date", pa.timestamp("ns")),
        ("text", pa.string()),
    ]
)


def parse_date(value: Any) -> datetime | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw or raw.casefold() in {"nat", "nan", "none", "null"}:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%Y/%m/%d"):
            try:
                parsed = datetime.strptime(raw[:10], fmt)
                break
            except ValueError:
                continue
        else:
            return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True, help="news_search/news.sqlite3")
    parser.add_argument("--output", type=Path, required=True, help="New Parquet file; existing files are not replaced")
    parser.add_argument("--batch-rows", type=int, default=20_000)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for a smoke test")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.database.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Выход уже существует, ничего не перезаписываю: {output}")
    if args.batch_rows < 1 or args.progress_every < 1:
        raise ValueError("--batch-rows и --progress-every должны быть положительными")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit должен быть положительным")

    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}_", suffix=".parquet.tmp", dir=output.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    connection: sqlite3.Connection | None = None
    writer: pq.ParquetWriter | None = None
    started = time.monotonic()
    total_rows = 0
    missing_dates = 0
    missing_text = 0
    source_counts: Counter[str] = Counter()
    minimum_date: datetime | None = None
    maximum_date: datetime | None = None

    try:
        uri = source.as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        query = "SELECT source, title, url, published_at, text FROM articles ORDER BY id"
        if args.limit is not None:
            query += f" LIMIT {int(args.limit)}"
        cursor = connection.execute(query)
        writer = pq.ParquetWriter(
            temporary,
            SCHEMA,
            compression="snappy",
            use_dictionary=["media"],
            write_statistics=True,
        )

        while True:
            rows = cursor.fetchmany(args.batch_rows)
            if not rows:
                break
            columns: dict[str, list[Any]] = {name: [] for name in SCHEMA.names}
            for source_name, title, link, published, body in rows:
                media = str(source_name or "")
                article_date = parse_date(published)
                source_counts[media] += 1
                missing_dates += article_date is None
                missing_text += not bool(body)
                if article_date is not None:
                    minimum_date = article_date if minimum_date is None else min(minimum_date, article_date)
                    maximum_date = article_date if maximum_date is None else max(maximum_date, article_date)
                columns["media"].append(media)
                columns["title"].append(str(title or ""))
                columns["link"].append(str(link or ""))
                columns["date"].append(article_date)
                columns["text"].append(str(body or ""))

            table = pa.Table.from_pydict(columns, schema=SCHEMA)
            writer.write_table(table, row_group_size=args.batch_rows)
            total_rows += len(rows)
            if total_rows % args.progress_every == 0:
                elapsed = max(time.monotonic() - started, 0.001)
                print(
                    f"Экспортировано {total_rows:,} строк; скорость {total_rows / elapsed:,.0f} строк/с",
                    flush=True,
                )

        writer.close()
        writer = None
        connection.close()
        connection = None
        os.replace(temporary, output)
        elapsed = time.monotonic() - started
        audit = {
            "source_sqlite": str(source),
            "output_parquet": str(output),
            "rows_exported": total_rows,
            "sources": dict(sorted(source_counts.items())),
            "min_date": minimum_date.isoformat(sep=" ") if minimum_date else None,
            "max_date": maximum_date.isoformat(sep=" ") if maximum_date else None,
            "missing_dates": missing_dates,
            "empty_text_rows": missing_text,
            "elapsed_seconds": round(elapsed, 2),
            "rows_per_second": round(total_rows / max(elapsed, 0.001), 2),
            "compression": "snappy",
        }
        audit_path = output.with_suffix(output.suffix + ".audit.json")
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
        return 0
    finally:
        if writer is not None:
            writer.close()
        if connection is not None:
            connection.close()
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
