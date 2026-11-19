from __future__ import annotations

import _bootstrap  # noqa: F401

from dissertation_pipeline.paths import PATHS


def main() -> None:
    PATHS.ensure_base_dirs()
    print(f"project_root={PATHS.project_root}")
    print(f"supp_root={PATHS.supp_root}")
    print("base directories are ready")


if __name__ == "__main__":
    main()

