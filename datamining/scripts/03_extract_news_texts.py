from __future__ import annotations

import subprocess
import sys

import _bootstrap  # noqa: F401
from settings import project_settings as cfg


def main() -> None:
    cfg.ensure_dirs()
    print(f"RUN_MODE={cfg.RUN_MODE}")
    print(f"RUN_ID={cfg.RUN_ID}")
    print(f"text extraction input={cfg.ARTICLE_PAGES_RUN_DIR}")

    for media in cfg.MEDIA:
        cmd = [
            sys.executable,
            str(cfg.python_script("03_extract_news_texts.py")),
            "--media",
            media,
            "--run-id",
            cfg.RUN_ID,
            "--allow-no-files",
        ]
        if cfg.RUN_MODE == "smoke":
            cmd += ["--sample-files", str(cfg.PAGE_SAMPLE_FILES_PER_MEDIA)]
        print("\n== extract texts:", media, "==", flush=True)
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=cfg.PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
