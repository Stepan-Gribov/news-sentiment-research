from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import _bootstrap  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_media(value: str) -> str:
    return ",".join(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    today = date.today()
    default_from = today.replace(day=1)

    parser = argparse.ArgumentParser(
        description="Run a small restart-safe news-link smoke test using the current top-500 company list."
    )
    parser.add_argument("--media", default="prime", help="Comma-separated media list, e.g. prime,rbc,interfax.")
    parser.add_argument("--sample-size", type=int, default=2, help="First N names from top500 list.")
    parser.add_argument("--date-from", default=default_from.isoformat())
    parser.add_argument("--date-to", default=today.isoformat())
    parser.add_argument("--retry-forever", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    env["DISS_RUN_MODE"] = "smoke"
    env["DISS_SMOKE_USE_COMPANY_FILE"] = "1"
    env["DISS_LINK_SAMPLE_SIZE"] = str(args.sample_size)
    env["DISS_MEDIA"] = parse_media(args.media)
    env["DISS_DATE_FROM_ISO"] = args.date_from
    env["DISS_DATE_TO_ISO"] = args.date_to
    env["DISS_DATE_FROM_DMY"] = date.fromisoformat(args.date_from).strftime("%d.%m.%Y")
    env["DISS_DATE_TO_DMY"] = date.fromisoformat(args.date_to).strftime("%d.%m.%Y")
    env["DISS_PYTHON_EXECUTABLE"] = sys.executable
    if args.retry_forever:
        env["DISS_LINK_COLLECTION_RETRY_FOREVER"] = "1"

    script = PROJECT_ROOT / "datamining" / "scripts" / "01_collect_news_links.py"
    cmd = [sys.executable, str(script)]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
