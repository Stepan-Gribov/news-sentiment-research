from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401

from dissertation_pipeline.paths import PATHS


RUN_ID = "2012-03-31_to_2026-06-30_top500"
DATE_FROM_ISO = "2012-03-31"
DATE_TO_ISO = "2026-06-30"
DATE_FROM_DMY = "31.03.2012"
DATE_TO_DMY = "30.06.2026"
MEDIA = ["rbc", "interfax", "prime", "kommersant", "frank"]


@dataclass(frozen=True)
class JobSpec:
    media: str
    command: list[str]
    log_path: Path
    out_dir: Path


def count_csv(path: Path) -> int:
    return sum(1 for _ in path.glob("*.csv")) if path.exists() else 0


def log_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def tail_log(path: Path, n_lines: int = 5) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n_lines:])


def build_jobs(run_id: str, names_file: Path, links_root: Path, logs_root: Path, tmp_root: Path) -> list[JobSpec]:
    project = PATHS.project_root
    return [
        JobSpec(
            media="rbc",
            command=[
                sys.executable,
                "-u",
                str(project / "scripts" / "01_collect_links_rbc.py"),
                "--names-file",
                str(names_file),
                "--date-from",
                DATE_FROM_ISO,
                "--date-to",
                DATE_TO_ISO,
                "--out-dir",
                str(links_root / "rbc"),
                "--mode",
                "selenium",
                "--browser-mode",
                "remote",
                "--remote-port",
                "9241",
                "--chrome-profile-dir",
                str(tmp_root / "rbc_chrome_profile"),
            ],
            log_path=logs_root / "rbc.log",
            out_dir=links_root / "rbc",
        ),
        JobSpec(
            media="interfax",
            command=[
                sys.executable,
                "-u",
                str(project / "scripts" / "01_collect_links_interfax.py"),
                "--names-file",
                str(names_file),
                "--date-from",
                DATE_FROM_DMY,
                "--date-to",
                DATE_TO_DMY,
                "--out-dir",
                str(links_root / "interfax"),
            ],
            log_path=logs_root / "interfax.log",
            out_dir=links_root / "interfax",
        ),
        JobSpec(
            media="prime",
            command=[
                sys.executable,
                "-u",
                str(project / "scripts" / "01_collect_links_prime.py"),
                "--names-file",
                str(names_file),
                "--date-from",
                DATE_FROM_ISO,
                "--date-to",
                DATE_TO_ISO,
                "--out-dir",
                str(links_root / "prime"),
            ],
            log_path=logs_root / "prime.log",
            out_dir=links_root / "prime",
        ),
        JobSpec(
            media="kommersant",
            command=[
                sys.executable,
                "-u",
                str(project / "scripts" / "01_collect_links_kommersant.py"),
                "--names-file",
                str(names_file),
                "--date-from",
                DATE_FROM_ISO,
                "--date-to",
                DATE_TO_ISO,
                "--out-dir",
                str(links_root / "kommersant"),
            ],
            log_path=logs_root / "kommersant.log",
            out_dir=links_root / "kommersant",
        ),
        JobSpec(
            media="frank",
            command=[
                sys.executable,
                "-u",
                str(project / "scripts" / "01_collect_links_frank.py"),
                "--names-file",
                str(names_file),
                "--date-from",
                DATE_FROM_ISO,
                "--date-to",
                DATE_TO_ISO,
                "--out-dir",
                str(links_root / "frank"),
                "--remote-port",
                "9242",
                "--chrome-profile-dir",
                str(tmp_root / "frank_chrome_profile"),
            ],
            log_path=logs_root / "frank.log",
            out_dir=links_root / "frank",
        ),
    ]


def write_run_info(logs_root: Path, links_root: Path, names_file: Path, run_id: str) -> None:
    names_count = sum(1 for line in names_file.read_text(encoding="utf-8").splitlines() if line.strip())
    text = "\n".join(
        [
            f"run_id={run_id}",
            f"started_at={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"names_file={names_file}",
            f"names_count={names_count}",
            f"date_from_iso={DATE_FROM_ISO}",
            f"date_to_iso={DATE_TO_ISO}",
            f"date_from_dmy={DATE_FROM_DMY}",
            f"date_to_dmy={DATE_TO_DMY}",
            f"links_dir={links_root}",
            f"logs_dir={logs_root}",
            "",
        ]
    )
    (logs_root / "run_info.txt").write_text(text, encoding="utf-8")


def start_jobs(jobs: list[JobSpec], logs_root: Path) -> dict[str, subprocess.Popen]:
    processes: dict[str, subprocess.Popen] = {}
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    for job in jobs:
        job.out_dir.mkdir(parents=True, exist_ok=True)
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = job.log_path.open("w", encoding="utf-8")
        log_fh.write(f"started {job.media} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_fh.write("command: " + " ".join(job.command) + "\n")
        log_fh.flush()
        proc = subprocess.Popen(
            job.command,
            cwd=PATHS.project_root,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
        (logs_root / f"{job.media}.pid").write_text(str(proc.pid) + "\n", encoding="utf-8")
        processes[job.media] = proc
        print(f"started {job.media}: pid={proc.pid} log={job.log_path}", flush=True)
    return processes


def terminate_processes(processes: dict[str, subprocess.Popen]) -> None:
    for media, proc in processes.items():
        if proc.poll() is None:
            print(f"terminating {media} pid={proc.pid}", flush=True)
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                proc.terminate()


def monitor(jobs: list[JobSpec], processes: dict[str, subprocess.Popen], interval_sec: int) -> int:
    started = time.time()
    try:
        while True:
            print("\nSTATUS", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"elapsed_min={(time.time() - started) / 60:.1f}", flush=True)
            all_done = True
            for job in jobs:
                proc = processes[job.media]
                code = proc.poll()
                alive = code is None
                all_done = all_done and not alive
                state = "alive" if alive else f"exit={code}"
                print(
                    f"{job.media:10s} {state:10s} files={count_csv(job.out_dir):4d} log_mb={log_size(job.log_path) / 1024 / 1024:.2f}",
                    flush=True,
                )
            if all_done:
                break
            time.sleep(interval_sec)
    except KeyboardInterrupt:
        terminate_processes(processes)
        raise

    failed = 0
    for job in jobs:
        code = processes[job.media].poll()
        if code != 0:
            failed += 1
            print(f"\nFAILED {job.media} exit={code}", flush=True)
            print(tail_log(job.log_path, 20), flush=True)
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full top-500 news link collection for all media.")
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--interval-sec", type=int, default=60)
    parser.add_argument("--media", nargs="*", default=MEDIA, choices=MEDIA)
    args = parser.parse_args()

    names_file = PATHS.company_names_file
    links_root = PATHS.news_links_runs_dir / args.run_id
    logs_root = PATHS.logs_dir / f"news_links_{args.run_id}"
    tmp_root = PATHS.tmp_dir / f"news_links_{args.run_id}"

    for path in [links_root, logs_root, tmp_root]:
        path.mkdir(parents=True, exist_ok=True)

    write_run_info(logs_root, links_root, names_file, args.run_id)
    jobs = [job for job in build_jobs(args.run_id, names_file, links_root, logs_root, tmp_root) if job.media in set(args.media)]
    if not jobs:
        raise SystemExit("No media selected.")
    processes = start_jobs(jobs, logs_root)
    failed = monitor(jobs, processes, args.interval_sec)
    if failed:
        raise SystemExit(failed)


if __name__ == "__main__":
    main()
