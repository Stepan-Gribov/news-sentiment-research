from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def inprogress_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.inprogress")


def csv_output_is_valid(path: Path, *, encoding: str = "utf-16", sep: str = "\t") -> bool:
    """A valid CSV means the company was already attempted and can be skipped."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        df = pd.read_csv(path, encoding=encoding, sep=sep, nrows=5)
    except Exception:
        return False
    return len(df.columns) > 0


def should_skip_output(path: Path, *, force: bool = False, encoding: str = "utf-16", sep: str = "\t") -> bool:
    if force:
        return False
    if inprogress_path(path).exists():
        return False
    return csv_output_is_valid(path, encoding=encoding, sep=sep)


def mark_inprogress(path: Path) -> Path:
    marker = inprogress_path(path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("in_progress\n", encoding="utf-8")
    return marker


def clear_inprogress(path: Path) -> None:
    marker = inprogress_path(path)
    if marker.exists():
        marker.unlink()


def atomic_write_csv(
    df: pd.DataFrame,
    path: Path,
    *,
    encoding: str = "utf-16",
    sep: str = "\t",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    df.to_csv(tmp_path, index=False, encoding=encoding, sep=sep)
    tmp_path.replace(path)
