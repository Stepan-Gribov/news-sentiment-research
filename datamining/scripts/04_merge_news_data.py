from __future__ import annotations

import subprocess
import sys

import _bootstrap  # noqa: F401
from settings import project_settings as cfg


def main() -> None:
    cfg.ensure_dirs()
    cmd = [sys.executable, str(cfg.python_script("08_merge_news_data.py"))]
    if cfg.RUN_MODE == "smoke":
        cmd.append("--skip-articles")
    print("== merge news data ==", flush=True)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cfg.PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
