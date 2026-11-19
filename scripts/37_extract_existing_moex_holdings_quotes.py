"""Copy exact-holdings MOEX quote rows into the new raw-data package.

Source files are left untouched. Existing analytical columns such as returns
and volatility are excluded; official quote/security fields are retained.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("./")
PACKAGE = DATA_ROOT / "market_data/raw_external_2026-09-30"
QUOTES_SOURCE = DATA_ROOT / "analysis_outputs/quotes_data"
GAP_ROOT = PACKAGE / "moex/securities/daily_quote_gap_2026-07-01_2026-08-31"
EXCLUDED_DERIVED_COLUMNS = {"ret", "return", "returns", "volatility", "ret_volatility", "yield_volatility", "log_return", "log_ret"}


def save_new(frame: pd.DataFrame, stem: Path) -> None:
    csv_path = stem.with_suffix(".csv")
    parquet_path = stem.with_suffix(".parquet")
    if csv_path.exists() and parquet_path.exists():
        return
    if csv_path.exists() or parquet_path.exists():
        raise FileExistsError(f"Only one representation exists; refusing to replace either: {csv_path}, {parquet_path}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    frame.to_parquet(parquet_path, index=False, compression="zstd")


def raw_columns(frame: pd.DataFrame) -> pd.DataFrame:
    drop = [column for column in frame.columns if str(column).strip().lower() in EXCLUDED_DERIVED_COLUMNS]
    return frame.drop(columns=drop)


def main() -> None:
    universe_path = PACKAGE / "moex/securities/holdings_exact_isin_universe.csv"
    universe = pd.read_csv(universe_path, dtype=str).fillna("")
    isins = set(universe["requested_isin"].str.strip().str.upper())
    mapping_path = GAP_ROOT / "mapping.csv"
    mapping = pd.read_csv(mapping_path, dtype=str).fillna("")
    resolved_map = mapping[mapping["status"].eq("resolved")][["requested_isin", "secid"]].copy()
    resolved_map["secid"] = resolved_map["secid"].str.strip().str.upper()
    resolved_map["requested_isin"] = resolved_map["requested_isin"].str.strip().str.upper()
    if resolved_map["secid"].duplicated().any():
        raise ValueError("MOEX mapping has multiple holdings ISINs for the same SECID; refusing ambiguous local quote match")
    manifest = []

    all_instr_path = QUOTES_SOURCE / "All_instr_df_append_2024_10_2025_11.csv"
    all_instr = pd.read_csv(all_instr_path, encoding="utf-16", sep="\t", low_memory=False, dtype={"ISIN": str})
    all_instr["SECID"] = all_instr["SECID"].fillna("").str.strip().str.upper()
    all_instr = all_instr.merge(
        resolved_map.rename(columns={"requested_isin": "HOLDINGS_ISIN_BY_MOEX_MAPPING", "secid": "SECID"}),
        on="SECID",
        how="inner",
        validate="many_to_one",
    )
    all_instr = raw_columns(all_instr)
    all_instr_out = PACKAGE / "moex/securities/existing_local_All_instr_exact_holdings_quotes_2024-10_2025-11"
    save_new(all_instr, all_instr_out)
    manifest.append({"dataset": "existing_local_All_instr_exact_holdings_quotes", "source_file": str(all_instr_path), "rows": len(all_instr), "unique_isins": int(all_instr["HOLDINGS_ISIN_BY_MOEX_MAPPING"].nunique())})

    nonshare_path = QUOTES_SOURCE / "moex_nonshare_history.csv"
    nonshare = pd.read_csv(nonshare_path, low_memory=False, dtype={"REQUESTED_ISIN": str, "ISIN": str})
    if "REQUESTED_ISIN" in nonshare:
        nonshare["REQUESTED_ISIN"] = nonshare["REQUESTED_ISIN"].fillna("").str.strip().str.upper()
        by_isin = nonshare[nonshare["REQUESTED_ISIN"].isin(isins)].copy()
    else:
        by_isin = nonshare.iloc[0:0].copy()
    if len(by_isin):
        nonshare = raw_columns(by_isin)
        nonshare_out = PACKAGE / "moex/securities/existing_local_nonshare_exact_holdings_quotes_2024-10_2025-12"
        save_new(nonshare, nonshare_out)
        matched_ids = int(nonshare["REQUESTED_ISIN"].nunique())
    else:
        matched_ids = 0
    manifest.append({"dataset": "existing_local_nonshare_exact_holdings_quotes", "source_file": str(nonshare_path), "rows": len(by_isin), "unique_isins": matched_ids})

    history_files = sorted((GAP_ROOT / "histories").glob("*.parquet"))
    if not history_files:
        raise FileNotFoundError(f"No downloaded MOEX top-up histories found under {GAP_ROOT / 'histories'}")
    histories = pd.concat((pd.read_parquet(path) for path in history_files), ignore_index=True, sort=False)
    gap_out = GAP_ROOT / "quotes_exact_holdings_2026-07-01_2026-08-31"
    save_new(histories, gap_out)
    manifest.append({"dataset": "moex_quotes_exact_holdings_top_up", "source_file": str(GAP_ROOT / "histories"), "rows": len(histories), "unique_isins": int(histories["REQUESTED_ISIN"].nunique())})

    full_map = universe.merge(mapping, on="requested_isin", how="left", suffixes=("", "_moex"))
    unresolved = full_map[full_map["status"].ne("resolved")].copy()
    unresolved_path = PACKAGE / "moex/securities/unresolved_moex_identifiers.csv"
    if not unresolved_path.exists():
        unresolved.to_csv(unresolved_path, index=False, encoding="utf-8-sig")

    status_rows = []
    status_dir = GAP_ROOT / "history_status"
    for path in sorted(status_dir.glob("*.json")):
        status_rows.append(json.loads(path.read_text(encoding="utf-8")))
    statuses = pd.DataFrame(status_rows)
    if len(statuses):
        statuses = mapping.merge(statuses, on="requested_isin", how="inner", suffixes=("_mapping", "_history"))
    statuses_path = PACKAGE / "moex/securities/moex_history_status_2026-07-01_2026-08-31.csv"
    if not statuses_path.exists():
        statuses.to_csv(statuses_path, index=False, encoding="utf-8-sig")

    no_history = statuses[statuses["status_history"].isin(["empty", "error"])].copy()
    no_history_path = PACKAGE / "moex/securities/unresolved_moex_histories_2026-07-01_2026-08-31.csv"
    if not no_history_path.exists():
        no_history.to_csv(no_history_path, index=False, encoding="utf-8-sig")

    summary = {
        "exact_holdings_isins": int(universe["requested_isin"].nunique()),
        "unresolved_mappings": int(len(unresolved)),
        "resolved_mappings": int((full_map["status"] == "resolved").sum()),
        "history_status": statuses["status_history"].value_counts().to_dict() if len(statuses) else {},
        "history_rows": int(statuses["rows"].fillna(0).astype(int).sum()) if "rows" in statuses else 0,
        "raw_quote_extracts": manifest,
        "derived_columns_excluded": sorted(EXCLUDED_DERIVED_COLUMNS),
    }
    summary_path = PACKAGE / "moex/securities/exact_holdings_quote_extraction_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to replace existing output: {summary_path}")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
