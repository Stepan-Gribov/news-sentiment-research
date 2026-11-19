from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from dissertation_pipeline.paths import PATHS


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract article text from downloaded page parquet files.")
    parser.add_argument("--media", required=True, choices=["rbc", "interfax", "kommersant", "prime", "frank"])
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--sample-files", type=int, default=None)
    parser.add_argument("--allow-no-files", action="store_true")
    args = parser.parse_args()

    from dissertation_pipeline.news.extract_news_text_from_parquets import process_one_file

    if args.run_id:
        input_dir = args.input_dir or (PATHS.news_pages_runs_dir / args.run_id / args.media)
    else:
        input_dir = args.input_dir or (PATHS.news_pages_history_dir / args.media)
    files = sorted(input_dir.glob("*.parquet"))
    if args.sample_files:
        files = files[: args.sample_files]
    if not files:
        if args.allow_no_files:
            print(f"No parquet files found in {input_dir}; skipped")
            return
        raise FileNotFoundError(f"No parquet files found in {input_dir}")
    for file_path in files:
        process_one_file(file_path, args.media)


if __name__ == "__main__":
    main()
