from __future__ import annotations

import runpy
from pathlib import Path

import _bootstrap  # noqa: F401


HERE = Path(__file__).resolve().parent


def main() -> None:
    for script_name in [
        "01_collect_news_links.py",
        "02_download_news_pages.py",
        "03_extract_news_texts.py",
        "04_merge_news_data.py",
    ]:
        print(f"\n\n######## {script_name} ########", flush=True)
        runpy.run_path(str(HERE / script_name), run_name="__main__")


if __name__ == "__main__":
    main()
