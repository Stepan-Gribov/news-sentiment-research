#!/usr/bin/env python3
"""Read-only audit: count and flag empty or short article texts."""

import argparse
import csv
import json
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq


SOURCES = {
    "interfax": ("interfax_archive", "article_id"),
    "kommersant": ("kommersant_archive", "doc_id"),
    "prime": ("prime_archive", "article_id"),
    "rbc": ("rbc_archive_content", "material_id"),
}
TEXT = "article_text_extracted"
FIELDS = ("url", "link", "published_at", "publication_date", "title", "headline", "parse_status", "status")
CSV_FIELDS = ("source", "article_id", "url", "published_at", "title", "parse_status",
              "word_count", "short_text_flag", "flag_reason", "archive_file", "row_in_file")


def pick(names, *options):
    return next((name for name in options if name in names), None)


def scan(root, source, id_field, minimum, writer):
    result = {"source": source, "archive_root": str(root), "files": 0, "rows": 0,
              "empty_texts": 0, "short_texts": 0, "files_without_text_column": 0,
              "unidentifiable_rows": 0, "read_errors": []}
    if not root.is_dir():
        result["read_errors"].append(f"Archive folder not found: {root}")
        return result

    for path in sorted(root.rglob("*.parquet")):
        result["files"] += 1
        if result["files"] % 1_000 == 0:
            print(f"{source}: scanned {result['files']:,} files", flush=True)
        try:
            parquet = pq.ParquetFile(path)
            names = set(parquet.schema_arrow.names)
            has_text = TEXT in names
            result["files_without_text_column"] += int(not has_text)
            columns = [name for name in (TEXT, id_field, *FIELDS) if name in names]
            if not columns:
                count = parquet.metadata.num_rows
                result["rows"] += count
                result["empty_texts"] += count
                result["short_texts"] += count
                result["unidentifiable_rows"] += count
                continue

            offset = 0
            for batch in parquet.iter_batches(columns=columns, batch_size=65_536):
                data, count = batch.to_pydict(), batch.num_rows
                result["rows"] += count
                urls = data.get("url", data.get("link", [""] * count))
                dates = data.get("published_at", data.get("publication_date", [""] * count))
                titles = data.get("title", data.get("headline", [""] * count))
                statuses = data.get("parse_status", data.get("status", [""] * count))
                ids = data.get(id_field, [""] * count)

                for i, value in enumerate(data.get(TEXT, [None] * count)):
                    text = value if isinstance(value, str) else ""
                    words = len(text.split())
                    if not text.strip():
                        result["empty_texts"] += 1
                    if words >= minimum:
                        continue
                    result["short_texts"] += 1
                    writer.writerow({
                        "source": source,
                        "article_id": ids[i] if ids[i] is not None else "",
                        "url": urls[i] if urls[i] is not None else "",
                        "published_at": dates[i] if dates[i] is not None else "",
                        "title": titles[i] if titles[i] is not None else "",
                        "parse_status": statuses[i] if statuses[i] is not None else "",
                        "word_count": words,
                        "short_text_flag": "true",
                        "flag_reason": ("missing_text_column" if not has_text else
                                        "empty_text" if not text.strip() else "below_word_threshold"),
                        "archive_file": path.relative_to(root.parent).as_posix(),
                        "row_in_file": offset + i,
                    })
                offset += count
        except Exception as exc:
            result["read_errors"].append(f"{path}: {type(exc).__name__}: {exc}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--min-words", type=int, default=40,
                        help="Flag texts below this word count; never deletes them.")
    args = parser.parse_args()
    if args.min_words < 1:
        parser.error("--min-words must be positive")
    if not args.output_dir.is_dir():
        parser.error(f"Output folder does not exist: {args.output_dir}")

    stamp = date.today().isoformat()
    candidates = args.output_dir / f"news_short_text_candidates_{stamp}.csv"
    summary_file = args.output_dir / f"news_short_text_audit_{stamp}.json"
    results = []
    with candidates.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for source, (folder, id_field) in SOURCES.items():
            result = scan(args.data_root / folder, source, id_field, args.min_words, writer)
            results.append(result)
            print(f"{source}: {result['rows']:,} rows; {result['empty_texts']:,} empty; "
                  f"{result['short_texts']:,} under {args.min_words} words; "
                  f"{len(result['read_errors'])} read errors", flush=True)

    summary_file.write_text(json.dumps({"minimum_words": args.min_words, "sources": results,
                                       "candidates_csv": str(candidates.resolve()),
                                       "read_only": True}, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(f"Saved: {candidates}\nSaved: {summary_file}")


if __name__ == "__main__":
    main()
