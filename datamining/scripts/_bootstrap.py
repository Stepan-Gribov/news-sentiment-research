from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DATAMINING_DIR = PROJECT_ROOT / "datamining"

for path in [PROJECT_ROOT, SRC_DIR, DATAMINING_DIR]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
