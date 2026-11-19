from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_tsv_utf16(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", encoding="utf-16", **kwargs)


def write_tsv_utf16(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", encoding="utf-16", index=False, **kwargs)

