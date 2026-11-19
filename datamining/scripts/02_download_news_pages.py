from __future__ import annotations

import subprocess
import time

import _bootstrap  # noqa: F401
from settings import project_settings as cfg


def run_media(media: str) -> None:
    cmd = [
        str(cfg.PYTHON_EXECUTABLE),
        str(cfg.python_script("02_download_pages_from_links.py")),
        "--media",
        media,
        "--run-id",
        cfg.RUN_ID,
        "--max-workers",
        str(cfg.PAGE_MAX_WORKERS),
        "--selenium-workers",
        str(cfg.PAGE_SELENIUM_WORKERS),
        "--nonempty-only",
        "--allow-no-files",
        "--min-success-rate",
        str(cfg.PAGE_DOWNLOAD_MIN_SUCCESS_RATE),
        "--checkpoint-every",
        str(cfg.PAGE_DOWNLOAD_CHECKPOINT_EVERY),
        "--retry-failed-rounds",
        str(cfg.PAGE_DOWNLOAD_FAILED_RETRY_ROUNDS),
        "--retry-failed-sleep-seconds",
        str(cfg.PAGE_DOWNLOAD_FAILED_RETRY_SLEEP_SECONDS),
    ]
    if cfg.PAGE_DOWNLOAD_FAST_SKIP_EXISTING:
        cmd += ["--fast-skip-existing"]
    if cfg.PAGE_DOWNLOAD_FAILED_RETRY_FOREVER:
        cmd += ["--retry-failed-forever"]
    if cfg.RUN_MODE == "smoke":
        cmd += [
            "--sample-files",
            str(cfg.PAGE_SAMPLE_FILES_PER_MEDIA),
            "--sample-rows",
            str(cfg.PAGE_SAMPLE_ROWS_PER_FILE),
        ]

    print("\n== download pages:", media, "==", flush=True)
    print(" ".join(cmd), flush=True)

    retry_forever = bool(getattr(cfg, "PAGE_DOWNLOAD_RETRY_FOREVER", False))
    max_retries = int(getattr(cfg, "PAGE_DOWNLOAD_MAX_RETRIES", 0))
    sleep_seconds = int(getattr(cfg, "PAGE_DOWNLOAD_RETRY_SLEEP_SECONDS", 60))
    failed_attempts = 0

    while True:
        attempt_number = failed_attempts + 1
        print(
            f"media={media}: page download attempt {attempt_number}"
            + (" (retry forever)" if retry_forever else f" of {max_retries + 1}"),
            flush=True,
        )
        try:
            subprocess.run(cmd, cwd=cfg.PROJECT_ROOT, check=True)
            print(f"media={media}: page download completed", flush=True)
            return
        except KeyboardInterrupt:
            raise
        except subprocess.CalledProcessError as exc:
            failed_attempts += 1
            can_retry = retry_forever or failed_attempts <= max_retries
            print(
                f"media={media}: page download failed with exit code {exc.returncode}; "
                f"failed_attempts={failed_attempts}; can_retry={can_retry}",
                flush=True,
            )
            if not can_retry:
                raise
            print(f"media={media}: sleeping {sleep_seconds}s before retry", flush=True)
            time.sleep(sleep_seconds)


def main() -> None:
    cfg.ensure_dirs()
    print(f"RUN_MODE={cfg.RUN_MODE}")
    print(f"RUN_ID={cfg.RUN_ID}")
    print(f"page parquet output={cfg.ARTICLE_PAGES_RUN_DIR}")

    for media in cfg.MEDIA:
        run_media(media)


if __name__ == "__main__":
    main()
