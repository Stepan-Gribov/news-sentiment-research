from __future__ import annotations

import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import _bootstrap  # noqa: F401
from settings import project_settings as cfg


SCRIPT_BY_MEDIA = {
    "rbc": "01_collect_links_rbc.py",
    "interfax": "01_collect_links_interfax.py",
    "prime": "01_collect_links_prime.py",
    "kommersant": "01_collect_links_kommersant.py",
    "frank": "01_collect_links_frank.py",
}


LATIN_RE = re.compile(r"[A-Za-z]")


def filtered_names_file(names_file: Path) -> Path:
    if not getattr(cfg, "EXCLUDE_LATIN_COMPANY_NAMES", False):
        return names_file

    names = [
        line.strip()
        for line in names_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    filtered = [name for name in names if not LATIN_RE.search(name)]
    out_path = cfg.TMP_DIR / f"{names_file.stem}_no_latin.txt"
    out_path.write_text("\n".join(filtered) + ("\n" if filtered else ""), encoding="utf-8")
    print(
        f"company names filter: exclude latin letters, kept {len(filtered):,} of {len(names):,}; file={out_path}",
        flush=True,
    )
    return out_path


def run_media(media: str, names_file: Path) -> None:
    spec = cfg.LINK_COLLECTION[media]
    out_dir = cfg.LINKS_RUN_DIR / media
    cmd = [
        str(cfg.PYTHON_EXECUTABLE),
        str(cfg.python_script(SCRIPT_BY_MEDIA[media])),
        "--names-file",
        str(names_file),
        "--date-from",
        spec.date_from,
        "--date-to",
        spec.date_to,
        "--out-dir",
        str(out_dir),
    ]
    if media in {"rbc", "frank"}:
        cmd += [
            "--remote-port",
            "9234" if media == "rbc" else "9235",
            "--chrome-profile-dir",
            str(cfg.TMP_DIR / f"{media}_chrome_profile"),
        ]
    if media in {"prime", "interfax", "kommersant"}:
        cmd += ["--workers", str(getattr(cfg, "LINK_COMPANY_WORKERS", 1))]
    if media in {"prime"}:
        cmd += ["--window-workers", str(getattr(cfg, "LINK_WINDOW_WORKERS", 1))]
    cmd += list(spec.extra_args)

    print("\n== collect links:", media, "==", flush=True)
    print(" ".join(cmd), flush=True)

    retry_forever = bool(getattr(cfg, "LINK_COLLECTION_RETRY_FOREVER", False))
    max_retries = int(getattr(cfg, "LINK_COLLECTION_MAX_RETRIES", 0))
    sleep_seconds = int(getattr(cfg, "LINK_COLLECTION_RETRY_SLEEP_SECONDS", 60))
    failed_attempts = 0

    while True:
        attempt_number = failed_attempts + 1
        print(
            f"media={media}: attempt {attempt_number}"
            + (" (retry forever)" if retry_forever else f" of {max_retries + 1}"),
            flush=True,
        )
        try:
            subprocess.run(cmd, cwd=cfg.PROJECT_ROOT, check=True)
            print(f"media={media}: completed", flush=True)
            return
        except KeyboardInterrupt:
            raise
        except subprocess.CalledProcessError as exc:
            failed_attempts += 1
            can_retry = retry_forever or failed_attempts <= max_retries
            print(
                f"media={media}: failed with exit code {exc.returncode}; "
                f"failed_attempts={failed_attempts}; can_retry={can_retry}",
                flush=True,
            )
            if not can_retry:
                raise
            print(f"media={media}: sleeping {sleep_seconds}s before retry", flush=True)
            time.sleep(sleep_seconds)


def main() -> None:
    cfg.ensure_dirs()
    source_names_file = cfg.SMOKE_COMPANY_NAMES_FILE if cfg.RUN_MODE == "smoke" else cfg.COMPANY_NAMES_FILE
    names_file = filtered_names_file(source_names_file)
    print(f"RUN_MODE={cfg.RUN_MODE}")
    print(f"RUN_ID={cfg.RUN_ID}")
    print(f"links output={cfg.LINKS_RUN_DIR}")
    workers = max(1, min(int(getattr(cfg, "LINK_MEDIA_WORKERS", 1)), len(cfg.MEDIA)))
    print(f"media workers={workers}")
    if workers == 1:
        for media in cfg.MEDIA:
            run_media(media, names_file)
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_media, media, names_file): media for media in cfg.MEDIA}
        for future in as_completed(futures):
            media = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"media={media}: failed in parallel runner: {exc}", flush=True)
                raise


if __name__ == "__main__":
    main()
