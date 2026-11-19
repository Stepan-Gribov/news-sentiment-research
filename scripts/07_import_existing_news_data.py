from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401

from dissertation_pipeline.paths import PATHS


DEFAULT_SOURCE = PATHS.project_root.parent.parent / "data" / "news_links"

LINK_DIRS = {
    "rbc": "links_rbc",
    "interfax": "links_interfax",
    "kommersant": "links_kommersant",
    "prime": "links_prime",
    "frank": "links_frank",
}

PAGE_DIRS = {
    "rbc": "NEWS_PAGES_PARQ/RBC_downloaded_news",
    "interfax": "NEWS_PAGES_PARQ/interfax_downloaded_news",
    "kommersant": "NEWS_PAGES_PARQ/kommersant_downloaded_news",
    "prime": "NEWS_PAGES_PARQ/prime_downloaded_news",
    "frank": "NEWS_PAGES_PARQ/frank_downloaded_news",
}

MERGED_FILES = [
    "NEWS_PAGES_PARQ/all_news_merged_unified.parquet",
    "NEWS_PAGES_PARQ/all_news_merged_unified.csv",
    "NEWS_PAGES_PARQ/rbc_merged_full.parquet",
    "NEWS_PAGES_PARQ/interfax_merged_full.parquet",
    "NEWS_PAGES_PARQ/kommersant_merged_full.parquet",
    "NEWS_PAGES_PARQ/prime_merged_full.parquet",
    "NEWS_PAGES_PARQ/frank_merged_full.parquet",
]


def copy_file(src: Path, dst: Path, force: bool, storage: str) -> bool:
    if not src.exists() or not src.is_file() or src.name == ".DS_Store":
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force and dst.stat().st_size == src.stat().st_size:
        return False
    if dst.exists():
        dst.unlink()
    if storage == "hardlink":
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)
    return True


def copy_tree_files(src_dir: Path, dst_dir: Path, pattern: str, force: bool, storage: str) -> tuple[int, int]:
    copied = 0
    seen = 0
    if not src_dir.exists():
        return seen, copied
    for src in sorted(src_dir.glob(pattern)):
        if not src.is_file() or src.name == ".DS_Store":
            continue
        seen += 1
        copied += int(copy_file(src, dst_dir / src.name, force, storage))
    return seen, copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Import existing news data into the simplified supplement layout.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--storage", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--skip-pages", action="store_true", help="Only import link CSVs and merged total files.")
    parser.add_argument("--media", choices=sorted(LINK_DIRS), nargs="*")
    args = parser.parse_args()

    media_list = args.media or sorted(LINK_DIRS)
    PATHS.ensure_base_dirs()

    total_seen = 0
    total_copied = 0
    for media in media_list:
        seen, copied = (0, 0)
        for pattern in ("*.csv", "*.xlsx"):
            pattern_seen, pattern_copied = copy_tree_files(
                args.source / LINK_DIRS[media],
                PATHS.news_links_history_dir / media,
                pattern,
                args.force,
                args.storage,
            )
            seen += pattern_seen
            copied += pattern_copied
        total_seen += seen
        total_copied += copied
        print(f"links/{media}: seen={seen}, copied={copied}")

    if not args.skip_pages:
        for media in media_list:
            seen, copied = copy_tree_files(
                args.source / PAGE_DIRS[media],
                PATHS.news_pages_history_dir / media,
                "*.parquet",
                args.force,
                args.storage,
            )
            total_seen += seen
            total_copied += copied
            print(f"article_pages/{media}: seen={seen}, copied={copied}")

    for rel in MERGED_FILES:
        src = args.source / rel
        copied = copy_file(src, PATHS.news_merged_dir / src.name, args.force, args.storage)
        if src.exists():
            total_seen += 1
            total_copied += int(copied)
        print(f"merged/{src.name}: exists={src.exists()}, copied={copied}")

    print(f"done: seen={total_seen}, copied={total_copied}")
    print(f"news_data={PATHS.news_data_dir}")


if __name__ == "__main__":
    main()
