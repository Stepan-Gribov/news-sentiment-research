from __future__ import annotations

import argparse
import socket
import subprocess
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from dissertation_pipeline.paths import PATHS
from dissertation_pipeline.news.layout import links_run_dir
from dissertation_pipeline.news.resume import atomic_write_csv, clear_inprogress, mark_inprogress, should_skip_output
from dissertation_pipeline.selenium_fast import configure_fast_driver, fast_chrome_args, find_chrome_executable


def load_names(path: Path, sample_size: int | None) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return names[:sample_size] if sample_size else names


def wait_for_port(port: int, host: str = "127.0.0.1", timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError(f"Chrome remote debugging port did not open: {host}:{port}")


def start_remote_chrome(port: int, profile_dir: Path) -> subprocess.Popen:
    chrome_path = find_chrome_executable()
    profile_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [
            str(chrome_path),
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "--window-size=1440,1600",
            *fast_chrome_args(),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    wait_for_port(port)
    return proc


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect RBC article links by company names.")
    parser.add_argument("--names-file", type=Path, default=PATHS.company_names_file)
    parser.add_argument("--date-from", default="2024-01-01")
    parser.add_argument("--date-to", default="2026-04-01")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--mode", choices=["selenium", "requests"], default="selenium")
    parser.add_argument("--browser-mode", choices=["remote", "webdriver"], default="remote")
    parser.add_argument("--remote-port", type=int, default=9224)
    parser.add_argument("--chrome-profile-dir", type=Path, default=PATHS.tmp_dir / "rbc_remote_chrome_profile")
    parser.add_argument("--keep-browser", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recollect even if a valid output CSV already exists.")
    args = parser.parse_args()

    from dissertation_pipeline.news.link_collectors.rbc import (
        attach_remote_driver,
        build_driver,
        collect_rbc_by_name,
        collect_rbc_by_name_selenium,
    )

    args.out_dir = args.out_dir or links_run_dir("rbc", args.date_from, args.date_to)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    names = load_names(args.names_file, args.sample_size)
    pending_names = []
    for name in names:
        out_path = args.out_dir / f"{name}.csv"
        if should_skip_output(out_path, force=args.force):
            print(f"RBC links: {name} [skip existing]", flush=True)
            continue
        pending_names.append(name)
    if not pending_names:
        print("RBC links: all outputs already exist; nothing to do.", flush=True)
        return

    if args.mode == "requests":
        for name in pending_names:
            out_path = args.out_dir / f"{name}.csv"
            print(f"RBC links: {name}", flush=True)
            mark_inprogress(out_path)
            df = collect_rbc_by_name(name, args.date_from, args.date_to)
            atomic_write_csv(df, out_path)
            clear_inprogress(out_path)
        return

    chrome_proc = None
    driver = None
    try:
        if args.browser_mode == "remote":
            chrome_proc = start_remote_chrome(args.remote_port, args.chrome_profile_dir)
            driver = attach_remote_driver(args.remote_port)
        else:
            driver = build_driver()
        configure_fast_driver(driver)

        for name in pending_names:
            out_path = args.out_dir / f"{name}.csv"
            print(f"RBC links: {name}", flush=True)
            mark_inprogress(out_path)
            df = collect_rbc_by_name_selenium(driver, name, args.date_from, args.date_to, verbose=args.verbose)
            atomic_write_csv(df, out_path)
            clear_inprogress(out_path)
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass
        if chrome_proc is not None and not args.keep_browser:
            chrome_proc.terminate()


if __name__ == "__main__":
    main()
