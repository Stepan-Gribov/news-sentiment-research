from __future__ import annotations

import argparse
import runpy
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "src/dissertation_pipeline/market/build_updated_market_metrics.py"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build updated MOEX portfolio metrics for regressions.")
    parser.parse_known_args()
    runpy.run_path(str(SCRIPT), run_name="__main__")
