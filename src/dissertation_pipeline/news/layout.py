from __future__ import annotations

import re
from pathlib import Path

from dissertation_pipeline.paths import PATHS


def safe_run_id(date_from: str, date_to: str) -> str:
    raw = f"{date_from}_to_{date_to}"
    return re.sub(r"[^0-9A-Za-zА-Яа-я_-]+", "-", raw).strip("-")


def links_run_dir(media: str, date_from: str, date_to: str) -> Path:
    return PATHS.news_links_runs_dir / safe_run_id(date_from, date_to) / media


def pages_run_dir(media: str, run_id: str) -> Path:
    return PATHS.news_pages_runs_dir / run_id / media
