from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dateutil.relativedelta import relativedelta

import _bootstrap  # noqa: F401

from dissertation_pipeline.paths import PATHS
from dissertation_pipeline.news.layout import links_run_dir
from dissertation_pipeline.news.resume import atomic_write_csv, clear_inprogress, mark_inprogress, should_skip_output


def month_windows(date_from: str, date_to: str) -> tuple[list[str], list[str]]:
    start = datetime.strptime(date_from, "%d.%m.%Y")
    end = datetime.strptime(date_to, "%d.%m.%Y")
    from_dates, to_dates = [], []
    cur = start
    while cur < end:
        nxt = min(cur + relativedelta(months=1), end)
        from_dates.append(cur.strftime("%d.%m.%Y"))
        to_dates.append(nxt.strftime("%d.%m.%Y"))
        cur = nxt
    return from_dates, to_dates


def load_names(path: Path, sample_size: int | None) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return names[:sample_size] if sample_size else names


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Interfax article links by company names.")
    parser.add_argument("--names-file", type=Path, default=PATHS.company_names_file)
    parser.add_argument("--date-from", default="01.01.2024")
    parser.add_argument("--date-to", default="01.04.2026")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1, help="Parallel company workers.")
    parser.add_argument("--force", action="store_true", help="Recollect even if a valid output CSV already exists.")
    args = parser.parse_args()

    from dissertation_pipeline.news.link_collectors.interfax import collect_by_name

    args.out_dir = args.out_dir or links_run_dir("interfax", args.date_from, args.date_to)
    from_dates, to_dates = month_windows(args.date_from, args.date_to)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    names = load_names(args.names_file, args.sample_size)

    def collect_one(name: str) -> None:
        out_path = args.out_dir / f"{name}.csv"
        if should_skip_output(out_path, force=args.force):
            print(f"Interfax links: {name} [skip existing]", flush=True)
            return
        print(f"Interfax links: {name}", flush=True)
        mark_inprogress(out_path)
        try:
            df = collect_by_name(name, from_dates, to_dates)
            atomic_write_csv(df, out_path)
        finally:
            clear_inprogress(out_path)

    workers = max(1, args.workers)
    print(f"Interfax company workers={workers}", flush=True)
    if workers == 1:
        for name in names:
            collect_one(name)
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(collect_one, name): name for name in names}
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"Interfax links failed: {name}: {exc}", flush=True)


if __name__ == "__main__":
    main()
