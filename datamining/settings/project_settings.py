from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dissertation_pipeline.paths import PATHS


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in {None, ""} else default


def env_list(name: str, default: list[str]) -> list[str]:
    value = os.environ.get(name)
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


PROJECT_ROOT = PATHS.project_root
SUPP_ROOT = PATHS.supp_root
PYTHON_EXECUTABLE = Path(os.environ.get("DISS_PYTHON_EXECUTABLE", sys.executable))

# Change to "full" for production refresh. "smoke" writes to a separate run folder.
RUN_MODE = os.environ.get("DISS_RUN_MODE", "full")

DATE_FROM_ISO = os.environ.get("DISS_DATE_FROM_ISO", "2026-01-01")
DATE_TO_ISO = os.environ.get("DISS_DATE_TO_ISO", "2026-06-01")
DATE_FROM_DMY = os.environ.get("DISS_DATE_FROM_DMY", "01.01.2026")
DATE_TO_DMY = os.environ.get("DISS_DATE_TO_DMY", "01.06.2026")

BASE_RUN_ID = f"{DATE_FROM_ISO}_to_{DATE_TO_ISO}"
RUN_ID = BASE_RUN_ID if RUN_MODE == "full" else f"smoke_{BASE_RUN_ID}"

MEDIA = env_list("DISS_MEDIA", ["rbc"])
# ["interfax", "prime", "kommersant", "frank"]

# Smoke limits. Full mode ignores LINK_SAMPLE_SIZE and PAGE_SAMPLE_* unless explicitly set.
SMOKE_COMPANIES = ["Газпром", "ВТБ"]
SMOKE_USE_COMPANY_FILE = env_bool("DISS_SMOKE_USE_COMPANY_FILE", True)
LINK_SAMPLE_SIZE = env_int("DISS_LINK_SAMPLE_SIZE", 5)
PAGE_SAMPLE_FILES_PER_MEDIA = env_int("DISS_PAGE_SAMPLE_FILES_PER_MEDIA", 1)
PAGE_SAMPLE_ROWS_PER_FILE = env_int("DISS_PAGE_SAMPLE_ROWS_PER_FILE", 3)
PAGE_MAX_WORKERS = env_int("DISS_PAGE_MAX_WORKERS", 8)
PAGE_SELENIUM_WORKERS = env_int("DISS_PAGE_SELENIUM_WORKERS", 3)
LINK_MEDIA_WORKERS = env_int("DISS_LINK_MEDIA_WORKERS", min(5, max(1, len(MEDIA))))
LINK_COMPANY_WORKERS = env_int("DISS_LINK_COMPANY_WORKERS", 8)
LINK_WINDOW_WORKERS = env_int("DISS_LINK_WINDOW_WORKERS", 4)

COMPANY_NAMES_FILE = PROJECT_ROOT / "config" / "company_lists" / "top500_securities_news_search_names.txt"
EXCLUDE_LATIN_COMPANY_NAMES = env_bool("DISS_EXCLUDE_LATIN_COMPANY_NAMES", True)

# Link collection restart policy. Completed company CSV files are skipped on retry,
# so restarting the media script does not duplicate already collected outputs.
LINK_COLLECTION_RETRY_FOREVER = env_bool("DISS_LINK_COLLECTION_RETRY_FOREVER", False)
LINK_COLLECTION_MAX_RETRIES = env_int("DISS_LINK_COLLECTION_MAX_RETRIES", 10)
LINK_COLLECTION_RETRY_SLEEP_SECONDS = env_int("DISS_LINK_COLLECTION_RETRY_SLEEP_SECONDS", 60)

# Page downloading restart policy. Existing valid parquet files are skipped.
PAGE_DOWNLOAD_RETRY_FOREVER = env_bool("DISS_PAGE_DOWNLOAD_RETRY_FOREVER", False)
PAGE_DOWNLOAD_MAX_RETRIES = env_int("DISS_PAGE_DOWNLOAD_MAX_RETRIES", 10)
PAGE_DOWNLOAD_RETRY_SLEEP_SECONDS = env_int("DISS_PAGE_DOWNLOAD_RETRY_SLEEP_SECONDS", 60)
PAGE_DOWNLOAD_MIN_SUCCESS_RATE = float(os.environ.get("DISS_PAGE_DOWNLOAD_MIN_SUCCESS_RATE", "0.01"))
PAGE_DOWNLOAD_CHECKPOINT_EVERY = env_int("DISS_PAGE_DOWNLOAD_CHECKPOINT_EVERY", 200)
PAGE_DOWNLOAD_FAST_SKIP_EXISTING = env_bool("DISS_PAGE_DOWNLOAD_FAST_SKIP_EXISTING", True)
PAGE_DOWNLOAD_FAILED_RETRY_FOREVER = env_bool("DISS_PAGE_DOWNLOAD_FAILED_RETRY_FOREVER", False)
PAGE_DOWNLOAD_FAILED_RETRY_ROUNDS = env_int("DISS_PAGE_DOWNLOAD_FAILED_RETRY_ROUNDS", 3)
PAGE_DOWNLOAD_FAILED_RETRY_SLEEP_SECONDS = env_int("DISS_PAGE_DOWNLOAD_FAILED_RETRY_SLEEP_SECONDS", 45)

NEWS_DATA_DIR = SUPP_ROOT / "news_data"
LINKS_RUN_DIR = NEWS_DATA_DIR / "links" / "runs" / RUN_ID
ARTICLE_PAGES_RUN_DIR = NEWS_DATA_DIR / "article_pages" / "runs" / RUN_ID
MERGED_DIR = NEWS_DATA_DIR / "merged"
LOG_DIR = SUPP_ROOT / "logs" / "datamining" / RUN_ID
TMP_DIR = SUPP_ROOT / "_tmp" / "datamining" / RUN_ID
SMOKE_COMPANY_NAMES_FILE = TMP_DIR / "smoke_company_names.txt"

LLM_OUTPUT_DIR = SUPP_ROOT / "llm_outputs"
FUNDS_DATA_DIR = SUPP_ROOT / "funds_data"
MARKET_DATA_DIR = SUPP_ROOT / "market_data"
ANALYSIS_OUTPUT_DIR = SUPP_ROOT / "analysis_outputs"


@dataclass(frozen=True)
class MediaCommand:
    media: str
    date_from: str
    date_to: str
    extra_args: tuple[str, ...] = ()


LINK_COLLECTION = {
    "rbc": MediaCommand("rbc", DATE_FROM_ISO, DATE_TO_ISO),
    "interfax": MediaCommand("interfax", DATE_FROM_DMY, DATE_TO_DMY),
    "prime": MediaCommand("prime", DATE_FROM_ISO, DATE_TO_ISO),
    "kommersant": MediaCommand("kommersant", DATE_FROM_ISO, DATE_TO_ISO),
    "frank": MediaCommand("frank", DATE_FROM_ISO, DATE_TO_ISO),
}


def ensure_dirs() -> None:
    for path in [
        LINKS_RUN_DIR,
        ARTICLE_PAGES_RUN_DIR,
        MERGED_DIR,
        LOG_DIR,
        TMP_DIR,
        FUNDS_DATA_DIR,
        ANALYSIS_OUTPUT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    for media in MEDIA:
        (LINKS_RUN_DIR / media).mkdir(parents=True, exist_ok=True)
        (ARTICLE_PAGES_RUN_DIR / media).mkdir(parents=True, exist_ok=True)
    if RUN_MODE == "smoke":
        if SMOKE_USE_COMPANY_FILE and COMPANY_NAMES_FILE.exists():
            names = [
                line.strip()
                for line in COMPANY_NAMES_FILE.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ][:LINK_SAMPLE_SIZE]
        else:
            names = SMOKE_COMPANIES[:LINK_SAMPLE_SIZE]
        SMOKE_COMPANY_NAMES_FILE.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")


def python_script(name: str) -> Path:
    return PROJECT_ROOT / "scripts" / name
