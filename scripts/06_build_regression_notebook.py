from __future__ import annotations

import argparse
import runpy
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "src/dissertation_pipeline/analysis/build_regressions_updated_old_style.py"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the updated old-style regression notebook.")
    parser.parse_known_args()
    runpy.run_path(str(SCRIPT), run_name="__main__")
