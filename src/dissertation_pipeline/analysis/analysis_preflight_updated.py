from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    project_dir = Path.cwd()
    parser = argparse.ArgumentParser(
        description=(
            "Fail-fast readiness check for repeating the original regression "
            "analysis on updated data through end-2025."
        )
    )
    parser.add_argument(
        "--y-file",
        type=Path,
        default=project_dir / "Y_interpolated_updated.csv",
    )
    parser.add_argument(
        "--portfolio-metrics-file",
        type=Path,
        default=project_dir / "portfolio_metrics" / "portfolio_metrics_updated.csv",
    )
    parser.add_argument(
        "--append-metrics-file",
        type=Path,
        default=project_dir / "portfolio_metrics" / "portfolio_metrics_append_2024_12_2025_11.csv",
    )
    parser.add_argument(
        "--fund-returns-file",
        type=Path,
        default=project_dir / "portfolio_metrics" / "fund_returns_append_available.csv",
    )
    parser.add_argument(
        "--controls-file",
        type=Path,
        default=Path("./local/Desktop/my_data/Master thesis/funds/Data/Controls/controls.xlsx"),
    )
    parser.add_argument(
        "--quote-resolution-file",
        type=Path,
        default=project_dir / "quotes_data" / "moex_resolution_report_append_2024_10_2025_11.csv",
    )
    parser.add_argument(
        "--required-end-date",
        default="2025-11-30",
        help="Expected last month-end for the updated analysis sample.",
    )
    return parser.parse_args()


def require_exists(path: Path, label: str, problems: list[str]) -> None:
    if not path.exists():
        problems.append(f"{label}: file missing -> {path}")


def summarize_y(y_file: Path, required_end: pd.Timestamp, problems: list[str]) -> dict:
    y = pd.read_csv(y_file, sep="\t", encoding="utf-16", usecols=["date", "fund_key"], low_memory=False)
    y["date"] = pd.to_datetime(y["date"])
    append = y[y["date"] > pd.Timestamp("2024-11-30")][["date", "fund_key"]].drop_duplicates()

    y_max = y["date"].max()
    if y_max < required_end:
        problems.append(f"Y coverage ends at {y_max.date()}, expected at least {required_end.date()}")

    return {
        "y_full_rows": int(len(y)),
        "y_append_pairs": int(len(append)),
        "y_max_date": y_max,
        "append_pairs": append,
    }


def summarize_controls(controls_file: Path, required_end: pd.Timestamp, problems: list[str]) -> dict:
    controls = pd.read_excel(controls_file)
    expected_cols = {"date", "Flow", "Imoex", "Ruonia", "USD_RUB", "RVI"}
    missing_cols = sorted(expected_cols - set(controls.columns))
    if missing_cols:
        problems.append(f"controls: missing columns {missing_cols}")

    if "date" in controls.columns:
        controls["date"] = pd.to_datetime(controls["date"], errors="coerce")
        controls_max = controls["date"].max()
        if controls_max < required_end:
            problems.append(
                f"controls: end at {controls_max.date()}, expected at least {required_end.date()}"
            )
    else:
        controls_max = pd.NaT

    return {
        "controls_rows": int(len(controls)),
        "controls_max_date": controls_max,
    }


def summarize_metrics(
    append_pairs: pd.DataFrame,
    append_metrics_file: Path,
    problems: list[str],
) -> dict:
    metrics = pd.read_csv(append_metrics_file)
    metrics["date"] = pd.to_datetime(metrics["date"], errors="coerce")
    merged = append_pairs.merge(metrics, on=["date", "fund_key"], how="left")
    pv_nonnull = int(merged["portfolio_var"].notna().sum()) if "portfolio_var" in merged.columns else 0
    liq_nonnull = int(merged["fund_liquidity"].notna().sum()) if "fund_liquidity" in merged.columns else 0
    total = int(len(merged))

    if pv_nonnull == 0:
        problems.append("portfolio_var: zero non-null observations on append period")
    if liq_nonnull == 0:
        problems.append("fund_liquidity: zero non-null observations on append period")

    return {
        "append_metric_pairs": total,
        "append_portfolio_var_nonnull": pv_nonnull,
        "append_fund_liquidity_nonnull": liq_nonnull,
    }


def summarize_returns(
    append_pairs: pd.DataFrame,
    fund_returns_file: Path,
    problems: list[str],
) -> dict:
    returns = pd.read_csv(fund_returns_file)
    returns["date"] = pd.to_datetime(returns["date"], errors="coerce")
    merged = append_pairs.merge(
        returns[["date", "fund_key", "fund_ret", "excess_return"]],
        on=["date", "fund_key"],
        how="left",
    )
    fund_ret_nonnull = int(merged["fund_ret"].notna().sum())
    excess_nonnull = int(merged["excess_return"].notna().sum())

    if fund_ret_nonnull == 0:
        problems.append("fund_ret: zero non-null observations on append period")
    if excess_nonnull == 0:
        problems.append("excess_return: zero non-null observations on append period")

    return {
        "append_fund_ret_nonnull": fund_ret_nonnull,
        "append_excess_return_nonnull": excess_nonnull,
    }


def summarize_quote_resolution(path: Path) -> dict:
    df = pd.read_csv(path)
    resolved = int((df["status"] == "resolved").sum()) if "status" in df.columns else 0
    unresolved = int((df["status"] != "resolved").sum()) if "status" in df.columns else 0
    return {
        "quote_requested_ids": int(len(df)),
        "quote_resolved_ids": resolved,
        "quote_unresolved_ids": unresolved,
    }


def main() -> None:
    args = parse_args()
    required_end = pd.Timestamp(args.required_end_date)
    problems: list[str] = []

    require_exists(args.y_file, "Y", problems)
    require_exists(args.portfolio_metrics_file, "portfolio_metrics_updated", problems)
    require_exists(args.append_metrics_file, "append_metrics", problems)
    require_exists(args.fund_returns_file, "fund_returns_append", problems)
    require_exists(args.controls_file, "controls", problems)
    require_exists(args.quote_resolution_file, "quote_resolution", problems)

    if problems:
        print("PRECHECK FAILED")
        for item in problems:
            print("-", item)
        raise SystemExit(1)

    summary: dict[str, object] = {}
    y_summary = summarize_y(args.y_file, required_end, problems)
    summary.update({k: v for k, v in y_summary.items() if k != "append_pairs"})

    controls_summary = summarize_controls(args.controls_file, required_end, problems)
    summary.update(controls_summary)

    metrics_summary = summarize_metrics(y_summary["append_pairs"], args.append_metrics_file, problems)
    summary.update(metrics_summary)

    returns_summary = summarize_returns(y_summary["append_pairs"], args.fund_returns_file, problems)
    summary.update(returns_summary)

    quote_summary = summarize_quote_resolution(args.quote_resolution_file)
    summary.update(quote_summary)

    print("UPDATED ANALYSIS PREFLIGHT")
    for key, value in summary.items():
        print(f"{key}: {value}")

    if problems:
        print("\nBLOCKERS")
        for item in problems:
            print("-", item)
        raise SystemExit(2)

    print("\nREADY")


if __name__ == "__main__":
    main()
