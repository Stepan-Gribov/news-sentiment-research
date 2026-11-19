import csv
import gc
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


INPUT_DIR = Path(r"./local/My Drive\4. Disserartion research\calculations\data\news_links\1.links_kommersant_content")
OUTPUT_DIR = Path(r"./local/NEWS_PAGES_PARQ\kommersant_downloaded_news")
CHUNK_SIZE = 200


def set_max_csv_field_size_limit():
    max_size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_size)
            return
        except OverflowError:
            max_size //= 10


set_max_csv_field_size_limit()


def get_read_attempts(csv_path: Path):
    with open(csv_path, "rb") as f:
        bom = f.read(4)

    if bom.startswith(b"\xff\xfe") or bom.startswith(b"\xfe\xff"):
        return [
            {"encoding": "utf-16", "sep": "\t", "chunksize": CHUNK_SIZE},
            {"encoding": "utf-16", "sep": "\t", "chunksize": CHUNK_SIZE, "engine": "python"},
        ]

    return [
        {"encoding": "utf-8", "sep": ",", "chunksize": CHUNK_SIZE},
        {"encoding": "utf-8-sig", "sep": ",", "chunksize": CHUNK_SIZE},
        {"encoding": "utf-16", "sep": "\t", "chunksize": CHUNK_SIZE},
        {"encoding": "utf-16", "sep": "\t", "chunksize": CHUNK_SIZE, "engine": "python"},
    ]


def iter_csv_chunks(csv_path: Path):
    last_error = None

    for params in get_read_attempts(csv_path):
        try:
            yield from pd.read_csv(csv_path, **params)
            return
        except Exception as e:
            last_error = e

    raise last_error


def normalize_chunk(chunk: pd.DataFrame, reference_columns=None) -> pd.DataFrame:
    chunk = chunk.copy()
    chunk = chunk.fillna("")

    for col in chunk.columns:
        chunk[col] = chunk[col].astype(str)

    if reference_columns is not None:
        for col in reference_columns:
            if col not in chunk.columns:
                chunk[col] = ""
        chunk = chunk.loc[:, reference_columns]

    return chunk


def convert_csv_to_parquet(csv_path: Path, output_path: Path):
    temp_path = output_path.with_suffix(".tmp.parquet")
    if temp_path.exists():
        temp_path.unlink()

    writer = None
    reference_columns = None

    try:
        for chunk_idx, chunk in enumerate(iter_csv_chunks(csv_path), start=1):
            if reference_columns is None:
                reference_columns = list(chunk.columns)

            chunk = normalize_chunk(chunk, reference_columns=reference_columns)
            table = pa.Table.from_pandas(chunk, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(temp_path, table.schema)
            else:
                table = pa.Table.from_pandas(
                    chunk,
                    schema=writer.schema,
                    preserve_index=False,
                    safe=False,
                )

            writer.write_table(table)

            del chunk
            del table
            gc.collect()

        if writer is None:
            raise RuntimeError("CSV produced no rows")

        writer.close()
        temp_path.replace(output_path)
    except Exception:
        if writer is not None:
            writer.close()
        if temp_path.exists():
            temp_path.unlink()
        raise


def is_completed_parquet(output_path: Path) -> bool:
    return output_path.exists() and output_path.is_file() and output_path.stat().st_size > 0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(INPUT_DIR.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {INPUT_DIR}")

    for idx, csv_path in enumerate(csv_files, start=1):
        print(f"[{idx}/{len(csv_files)}] Processing: {csv_path.name}")

        output_path = OUTPUT_DIR / f"{csv_path.stem}.parquet"
        if is_completed_parquet(output_path):
            print(f"[{idx}/{len(csv_files)}] Skipped existing: {output_path.name}")
            continue

        try:
            convert_csv_to_parquet(csv_path, output_path)
            print(f"[{idx}/{len(csv_files)}] Saved: {output_path}")
        except Exception as e:
            print(f"[{idx}/{len(csv_files)}] FAILED: {csv_path.name} -> {e}")
            continue


if __name__ == "__main__":
    main()
