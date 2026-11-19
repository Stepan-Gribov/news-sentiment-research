from __future__ import annotations

import argparse
import runpy
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "src/dissertation_pipeline/llm/rebuild_news_panel_from_all_news_parts100.py"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rebuild article/monthly news panel from all_news.jsonl and LLM parts(100)."
    )
    parser.parse_known_args()
    runpy.run_path(str(SCRIPT), run_name="__main__")
