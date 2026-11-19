from __future__ import annotations

import subprocess

import _bootstrap  # noqa: F401
from settings import project_settings as cfg


def main() -> None:
    cfg.ensure_dirs()
    out_dir = cfg.LINKS_RUN_DIR / "yandex"
    cmd = [
        str(cfg.PYTHON_EXECUTABLE),
        str(cfg.PROJECT_ROOT / "scripts" / "16_collect_links_yandex.py"),
        "--company",
        "Газпром",
        "--date-from",
        cfg.DATE_FROM_ISO,
        "--date-to",
        cfg.DATE_TO_ISO,
        "--profiles",
        "corporate_news,disclosure",
        "--out-dir",
        str(out_dir),
    ]
    print("== Yandex Gazprom pilot ==", flush=True)
    print(f"date interval=[{cfg.DATE_FROM_ISO}, {cfg.DATE_TO_ISO})", flush=True)
    print(f"output={out_dir}", flush=True)
    subprocess.run(cmd, cwd=cfg.PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
