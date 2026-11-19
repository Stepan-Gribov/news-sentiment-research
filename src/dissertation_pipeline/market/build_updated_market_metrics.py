from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from moex_quotes_downloader import (
    build_combined_quotes,
    download_histories,
    ensure_parent,
    fetch_index_history,
    load_local_reference,
    resolve_universe,
)


LEGACY_FUNDS_ROOT = Path("./local/Desktop/my_data/Master thesis/funds")
LEGACY_SECURITIES_EXPORT = (
    LEGACY_FUNDS_ROOT / "Code" / "Data loading" / "Export_ru_securities-list_20250203.csv"
)
LEGACY_PORTFOLIO_VAR_PATH = LEGACY_FUNDS_ROOT / "Code" / "Analisys" / "portfolio_variances.xlsx"
LEGACY_FUND_PRICES_PATH = LEGACY_FUNDS_ROOT / "Data" / "Y" / "funds_01012015_17052025.csv"
LEGACY_CONTROLS_PATH = LEGACY_FUNDS_ROOT / "Data" / "Controls" / "controls.xlsx"


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"


def _progress_line(stage: str, completed: int, total: int, start_time: float) -> None:
    elapsed = max(pd.Timestamp.now().timestamp() - start_time, 1e-9)
    rate = completed / elapsed
    remaining = max(total - completed, 0)
    eta = remaining / rate if rate > 0 else float("inf")
    share = completed / total if total else 1.0
    print(
        f"[{stage}] {completed}/{total} ({share:.1%}) | "
        f"elapsed={_format_duration(elapsed)} | eta={_format_duration(eta)}"
    )


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    quotes_dir = script_dir / "quotes_data"
    metrics_dir = script_dir / "portfolio_metrics"
    controls_dir = script_dir / "controls_data"
    legacy_liquidity_default = script_dir / "llm_core_replication_outputs" / "fund_liquidity.pkl"

    parser = argparse.ArgumentParser(
        description=(
            "Extend the old thesis market-data block with updated quotes and "
            "portfolio metrics for the new Y_interpolated_updated.csv tail."
        )
    )
    parser.add_argument(
        "--y-file",
        type=Path,
        default=script_dir / "Y_interpolated_updated.csv",
        help="Updated Y file in UTF-16 TSV format.",
    )
    parser.add_argument(
        "--append-after",
        default="2024-11-30",
        help="Build new metrics only for dates strictly greater than this cutoff.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=90,
        help="Lookback window used in the original notebook.",
    )
    parser.add_argument(
        "--quotes-from-date",
        default=None,
        help="Optional manual quote start date. Default is min(new_date) - lookback + 1.",
    )
    parser.add_argument(
        "--quotes-to-date",
        default=None,
        help="Optional manual quote end date. Default is max(new_date).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel workers for MOEX history downloads.",
    )
    parser.add_argument(
        "--sample-fund-dates",
        type=int,
        default=None,
        help="Run only the first N unique fund_key x date portfolios for a smoke test.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse an existing local quotes file instead of downloading again.",
    )
    parser.add_argument(
        "--skip-controls",
        action="store_true",
        help="Do not download local IMOEX and RVI tails.",
    )
    parser.add_argument(
        "--securities-export",
        type=Path,
        default=LEGACY_SECURITIES_EXPORT,
        help="Local MOEX securities export for ISIN -> SECID resolution.",
    )
    parser.add_argument(
        "--quotes-out",
        type=Path,
        default=quotes_dir / "All_instr_df_append_2024_10_2025_11.csv",
        help="Local append-only quotes table.",
    )
    parser.add_argument(
        "--nonshare-quotes-path",
        type=Path,
        default=quotes_dir / "moex_nonshare_history.csv",
        help="Additional non-share MOEX histories keyed by REQUESTED_ISIN.",
    )
    parser.add_argument(
        "--quotes-dict-out",
        type=Path,
        default=quotes_dir / "all_instr_quotes_dict_append_2024_10_2025_11.pkl",
        help="Local append-only per-SECID raw history dict.",
    )
    parser.add_argument(
        "--mapping-out",
        type=Path,
        default=quotes_dir / "moex_resolution_report_append_2024_10_2025_11.csv",
        help="Resolution diagnostics for the append quote universe.",
    )
    parser.add_argument(
        "--imoex-out",
        type=Path,
        default=controls_dir / "imoex_df_append_2024_10_2025_11.xlsx",
        help="Append-period IMOEX history.",
    )
    parser.add_argument(
        "--rvi-out",
        type=Path,
        default=controls_dir / "rvi_df_append_2024_10_2025_11.xlsx",
        help="Append-period RVI history.",
    )
    parser.add_argument(
        "--legacy-portfolio-var",
        type=Path,
        default=LEGACY_PORTFOLIO_VAR_PATH,
        help="Old portfolio variance file from the original project.",
    )
    parser.add_argument(
        "--legacy-fund-liquidity",
        type=Path,
        default=legacy_liquidity_default,
        help="Existing local fund liquidity cache for the old period.",
    )
    parser.add_argument(
        "--append-metrics-out",
        type=Path,
        default=metrics_dir / "portfolio_metrics_append_2024_12_2025_11.csv",
        help="Append-only portfolio metrics.",
    )
    parser.add_argument(
        "--full-metrics-out",
        type=Path,
        default=metrics_dir / "portfolio_metrics_updated.csv",
        help="Old + append portfolio metrics merged without duplicates.",
    )
    parser.add_argument(
        "--append-fund-ret-out",
        type=Path,
        default=metrics_dir / "fund_returns_append_available.csv",
        help="Append-period fund returns where the old PIF price source still exists.",
    )
    parser.add_argument(
        "--fund-prices-path",
        type=Path,
        default=LEGACY_FUND_PRICES_PATH,
        help="Daily fund unit prices used in the old notebook.",
    )
    parser.add_argument(
        "--controls-path",
        type=Path,
        default=LEGACY_CONTROLS_PATH,
        help="Controls file containing imoex_ret from the old project.",
    )
    return parser.parse_args()


def load_y(y_file: Path) -> pd.DataFrame:
    panel = pd.read_csv(y_file, sep="\t", encoding="utf-16", low_memory=False)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["ISIN конечного актива"] = (
        panel["ISIN конечного актива"].astype(str).where(panel["ISIN конечного актива"].notna(), "")
    )
    panel["ISIN конечного актива"] = panel["ISIN конечного актива"].str.strip()
    panel["Доля, %"] = pd.to_numeric(panel["Доля, %"], errors="coerce")
    return panel


def apply_sample_limit(panel_append: pd.DataFrame, sample_fund_dates: Optional[int]) -> pd.DataFrame:
    if sample_fund_dates is None:
        return panel_append

    pairs = (
        panel_append[["date", "fund_key"]]
        .drop_duplicates()
        .sort_values(["date", "fund_key"])
        .head(sample_fund_dates)
    )
    limited = panel_append.merge(pairs, on=["date", "fund_key"], how="inner")
    print(
        f"Sample mode enabled: {pairs.shape[0]} fund-date pairs, "
        f"{limited['fund_key'].nunique()} funds, {len(limited)} rows"
    )
    return limited


def add_sample_suffix(path: Path, sample_fund_dates: Optional[int]) -> Path:
    if sample_fund_dates is None:
        return path
    suffix = f"_sample_{sample_fund_dates}"
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def apply_sample_output_paths(args: argparse.Namespace) -> None:
    if args.sample_fund_dates is None:
        return
    args.quotes_out = add_sample_suffix(args.quotes_out, args.sample_fund_dates)
    args.quotes_dict_out = add_sample_suffix(args.quotes_dict_out, args.sample_fund_dates)
    args.mapping_out = add_sample_suffix(args.mapping_out, args.sample_fund_dates)
    args.imoex_out = add_sample_suffix(args.imoex_out, args.sample_fund_dates)
    args.rvi_out = add_sample_suffix(args.rvi_out, args.sample_fund_dates)
    args.append_metrics_out = add_sample_suffix(args.append_metrics_out, args.sample_fund_dates)
    args.full_metrics_out = add_sample_suffix(args.full_metrics_out, args.sample_fund_dates)
    args.append_fund_ret_out = add_sample_suffix(args.append_fund_ret_out, args.sample_fund_dates)


def compute_quote_window(
    panel_append: pd.DataFrame,
    lookback_days: int,
    from_date: Optional[str],
    to_date: Optional[str],
) -> tuple[str, str]:
    append_min = panel_append["date"].min()
    append_max = panel_append["date"].max()
    inferred_from = (append_min - pd.Timedelta(days=lookback_days - 1)).strftime("%Y-%m-%d")
    inferred_to = append_max.strftime("%Y-%m-%d")
    return from_date or inferred_from, to_date or inferred_to


def load_quotes(quotes_path: Path) -> pd.DataFrame:
    quotes = pd.read_csv(quotes_path, sep="\t", encoding="utf-16", low_memory=False)
    quotes["TRADEDATE"] = pd.to_datetime(quotes["TRADEDATE"], errors="coerce")
    if "ISIN" not in quotes.columns:
        quotes["ISIN"] = pd.NA
    if "REQUESTED_SECID" in quotes.columns:
        quotes["REQUESTED_SECID"] = quotes["REQUESTED_SECID"].astype(str).str.strip()
        quotes.loc[quotes["REQUESTED_SECID"].isin(["", "nan"]), "REQUESTED_SECID"] = pd.NA
    quotes["ISIN"] = quotes["ISIN"].astype("string")
    if "REQUESTED_SECID" in quotes.columns:
        quotes["ISIN"] = quotes["ISIN"].fillna(quotes["REQUESTED_SECID"])
    quotes["ISIN"] = quotes["ISIN"].fillna(quotes["SECID"])
    quotes["ISIN"] = quotes["ISIN"].astype(str).str.strip()
    quotes["ret"] = pd.to_numeric(quotes["ret"], errors="coerce")
    quotes["VALUE"] = pd.to_numeric(quotes["VALUE"], errors="coerce")
    return quotes


def load_nonshare_quotes(quotes_path: Path) -> pd.DataFrame:
    quotes = pd.read_csv(quotes_path, low_memory=False)
    quotes["TRADEDATE"] = pd.to_datetime(quotes["TRADEDATE"], errors="coerce")
    quotes["REQUESTED_ISIN"] = quotes["REQUESTED_ISIN"].astype(str).str.strip()
    quotes = quotes[quotes["REQUESTED_ISIN"].ne("") & quotes["REQUESTED_ISIN"].ne("nan")].copy()
    quotes["ISIN"] = quotes["REQUESTED_ISIN"]

    for price_col in ["CLOSE", "LEGALCLOSEPRICE", "WAPRICE"]:
        if price_col in quotes.columns:
            quotes[price_col] = pd.to_numeric(quotes[price_col], errors="coerce")
    quotes["VALUE"] = pd.to_numeric(quotes["VALUE"], errors="coerce")

    close_proxy = None
    for candidate in ["CLOSE", "LEGALCLOSEPRICE", "WAPRICE"]:
        if candidate in quotes.columns:
            if close_proxy is None:
                close_proxy = quotes[candidate].copy()
            else:
                close_proxy = close_proxy.fillna(quotes[candidate])
    if close_proxy is None:
        quotes["ret"] = np.nan
    else:
        quotes["close_proxy"] = close_proxy
        quotes = quotes.sort_values(["ISIN", "TRADEDATE"])
        quotes["ret"] = quotes.groupby("ISIN")["close_proxy"].pct_change()

    keep = [col for col in ["ISIN", "TRADEDATE", "ret", "VALUE"] if col in quotes.columns]
    return quotes.loc[:, keep]


def combine_quote_sources(primary_quotes: pd.DataFrame, nonshare_quotes_path: Path) -> pd.DataFrame:
    quotes_parts = [primary_quotes.copy()]
    if nonshare_quotes_path.exists():
        nonshare = load_nonshare_quotes(nonshare_quotes_path)
        if not nonshare.empty:
            quotes_parts.append(nonshare)
            print(
                f"Loaded non-share quotes from {nonshare_quotes_path}: "
                f"{len(nonshare)} rows, {nonshare['ISIN'].nunique()} identifiers"
            )
    else:
        print(f"Non-share quotes file not found: {nonshare_quotes_path}")

    quotes = pd.concat(quotes_parts, ignore_index=True, sort=False)
    quotes = quotes.dropna(subset=["ISIN", "TRADEDATE"]).copy()
    quotes["ISIN"] = quotes["ISIN"].astype(str).str.strip()
    quotes = quotes[quotes["ISIN"].ne("") & quotes["ISIN"].ne("nan")]
    quotes = quotes.sort_values(["ISIN", "TRADEDATE"])
    quotes = quotes.drop_duplicates(subset=["ISIN", "TRADEDATE"], keep="last")
    return quotes


def build_append_quotes(
    panel_append: pd.DataFrame,
    args: argparse.Namespace,
    quotes_from: str,
    quotes_to: str,
) -> pd.DataFrame:
    if args.skip_download:
        if not args.quotes_out.exists():
            raise FileNotFoundError(f"--skip-download was passed, but {args.quotes_out} does not exist.")
        print(f"Reusing existing append quotes from {args.quotes_out}")
        return load_quotes(args.quotes_out)

    universe = (
        panel_append["ISIN конечного актива"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    universe = universe[universe.ne("")]
    universe = pd.Index(universe.unique()).tolist()
    print(f"Append-period unique identifiers: {len(universe)}")

    local_ref = load_local_reference(args.securities_export)
    resolved_df = resolve_universe(universe, local_ref, workers=args.workers)

    history_dict, download_stats = download_histories(
        resolved_df=resolved_df,
        from_date=quotes_from,
        to_date=quotes_to,
        workers=args.workers,
    )

    if not download_stats.empty:
        resolved_df = resolved_df.merge(download_stats, on=["secid", "market"], how="left")
    else:
        resolved_df["history_rows"] = pd.NA
        resolved_df["download_status"] = pd.NA
        resolved_df["download_error"] = pd.NA

    quotes_df = build_combined_quotes(history_dict)

    ensure_parent(args.mapping_out)
    resolved_df.to_csv(args.mapping_out, index=False)
    print(f"Resolution report saved to {args.mapping_out}")

    if quotes_df.empty:
        raise RuntimeError("No append-period quote history was downloaded.")

    ensure_parent(args.quotes_out)
    quotes_df.to_csv(args.quotes_out, index=False, encoding="utf-16", sep="\t")
    print(f"Append quotes saved to {args.quotes_out} with shape {quotes_df.shape}")

    ensure_parent(args.quotes_dict_out)
    with args.quotes_dict_out.open("wb") as handle:
        pickle.dump(history_dict, handle)
    print(f"Raw history dict saved to {args.quotes_dict_out}")

    if not args.skip_controls:
        imoex_df = fetch_index_history("IMOEX", from_date=quotes_from, to_date=quotes_to)
        rvi_df = fetch_index_history("RVI", from_date=quotes_from, to_date=quotes_to)
        ensure_parent(args.imoex_out)
        ensure_parent(args.rvi_out)
        imoex_df.to_excel(args.imoex_out, index=False)
        rvi_df.to_excel(args.rvi_out, index=False)
        print(f"Append IMOEX saved to {args.imoex_out}")
        print(f"Append RVI saved to {args.rvi_out}")

    return quotes_df


def build_portfolios(panel_append: pd.DataFrame) -> pd.DataFrame:
    portfolios = (
        panel_append[["date", "fund_key", "ISIN конечного актива", "Доля, %"]]
        .dropna(subset=["ISIN конечного актива", "Доля, %"])
        .sort_values(["date", "fund_key", "Доля, %"], ascending=[True, True, False])
    )
    return portfolios


def compute_append_metrics(
    portfolios: pd.DataFrame,
    quotes: pd.DataFrame,
    lookback_days: int,
) -> pd.DataFrame:
    quotes = quotes.copy()
    quotes["Amihud_illiq"] = quotes["ret"].abs() / quotes["VALUE"]
    quotes.loc[quotes["VALUE"] == 0, "Amihud_illiq"] = np.nan

    ret_index = (
        quotes.loc[:, ["ISIN", "TRADEDATE", "ret"]]
        .dropna(subset=["ISIN", "TRADEDATE"])
        .set_index(["ISIN", "TRADEDATE"])
        .sort_index()["ret"]
    )
    illiq_index = (
        quotes.loc[:, ["ISIN", "TRADEDATE", "Amihud_illiq"]]
        .dropna(subset=["ISIN", "TRADEDATE"])
        .set_index(["ISIN", "TRADEDATE"])
        .sort_index()["Amihud_illiq"]
    )
    available_ret_isins = set(ret_index.index.get_level_values(0))
    available_illiq_isins = set(illiq_index.index.get_level_values(0))

    rows = []
    grouped = portfolios.groupby(["date", "fund_key"], sort=True)
    total_groups = grouped.ngroups
    start_time = pd.Timestamp.now().timestamp()

    for idx, ((dt, fund_key), grp) in enumerate(grouped, start=1):
        uniq = grp.drop_duplicates("ISIN конечного актива")
        weights = pd.Series(
            uniq["Доля, %"].values,
            index=uniq["ISIN конечного актива"].astype(str),
            name="weights",
        )
        start = pd.Timestamp(dt) - pd.Timedelta(days=lookback_days - 1)

        ret_data = {}
        for isin in weights.index:
            if isin not in available_ret_isins:
                continue
            try:
                ret_ser = ret_index.loc[isin].loc[start:dt]
            except KeyError:
                continue
            if not ret_ser.empty:
                ret_data[isin] = ret_ser

        if ret_data:
            ret_matrix = pd.DataFrame(ret_data).dropna(axis=1, how="any")
            if not ret_matrix.empty and ret_matrix.shape[1] > 0 and len(ret_matrix) > 1:
                w_adj = weights.loc[ret_matrix.columns]
                cov = ret_matrix.cov()
                portfolio_var = float(w_adj.T @ cov @ w_adj)
            else:
                portfolio_var = np.nan
        else:
            portfolio_var = np.nan

        illiq_means = {}
        for isin in weights.index:
            if isin not in available_illiq_isins:
                continue
            try:
                illiq_ser = illiq_index.loc[isin].loc[start:dt]
            except KeyError:
                continue
            if not illiq_ser.empty:
                illiq_means[isin] = illiq_ser.mean()

        if illiq_means:
            w_adj_liq = weights.loc[list(illiq_means.keys())]
            fund_liquidity = float((w_adj_liq * pd.Series(illiq_means)).sum())
        else:
            fund_liquidity = np.nan

        rows.append(
            {
                "date": pd.Timestamp(dt),
                "fund_key": fund_key,
                "portfolio_var": portfolio_var,
                "fund_liquidity": fund_liquidity,
            }
        )

        if idx % 50 == 0 or idx == total_groups:
            _progress_line("metrics", idx, total_groups, start_time)

    return pd.DataFrame(rows)


def load_legacy_metric(path: Path, value_col: str) -> pd.DataFrame:
    if not path.exists():
        print(f"Legacy metric file not found: {path}")
        return pd.DataFrame(columns=["date", "fund_key", value_col])

    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path, usecols=["date", "fund_key", value_col])
    else:
        df = pd.read_pickle(path)
        keep = ["date", "fund_key", value_col]
        df = df.loc[:, [col for col in keep if col in df.columns]]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date", "fund_key"])


def merge_metrics(
    legacy_portfolio_var: pd.DataFrame,
    legacy_fund_liquidity: pd.DataFrame,
    append_metrics: pd.DataFrame,
) -> pd.DataFrame:
    legacy = legacy_portfolio_var.merge(
        legacy_fund_liquidity,
        on=["date", "fund_key"],
        how="outer",
    )
    combined = pd.concat([legacy, append_metrics], ignore_index=True)
    combined = combined.sort_values(["date", "fund_key"])
    combined = combined.drop_duplicates(subset=["date", "fund_key"], keep="last")
    return combined


def build_append_fund_returns(
    panel_append: pd.DataFrame,
    fund_prices_path: Path,
    controls_path: Path,
) -> pd.DataFrame:
    if not fund_prices_path.exists():
        print(f"Fund price source not found: {fund_prices_path}")
        return pd.DataFrame(columns=["date", "fund_key", "fund_ret", "excess_return"])

    pif_prices = pd.read_csv(
        fund_prices_path,
        sep=";",
        encoding="cp1251",
        low_memory=False,
    ).dropna(subset=["Фонд"])
    pif_prices["Дата"] = pd.to_datetime(pif_prices["Дата"], format="%d.%m.%Y", errors="coerce")
    pif_prices["Пай"] = pd.to_numeric(pif_prices["Пай"], errors="coerce")
    pif_prices = pif_prices.sort_values(["Фонд", "Дата"], ascending=[False, True])

    fund_panel = panel_append[["date", "Название фонда", "fund_key"]].drop_duplicates()
    fund_panel = fund_panel.merge(
        pif_prices[["Дата", "Фонд", "Пай", "СЧА"]],
        left_on=["date", "Название фонда"],
        right_on=["Дата", "Фонд"],
        how="left",
    )

    agg_prices = fund_panel.groupby(["fund_key", "date"], as_index=False)["Пай"].mean()
    agg_prices = agg_prices.sort_values(["fund_key", "date"])
    agg_prices["fund_ret"] = agg_prices.groupby("fund_key")["Пай"].pct_change()

    returns = agg_prices.copy()
    returns["imoex_ret"] = np.nan

    if controls_path.exists():
        controls = pd.read_excel(controls_path)
        controls["date"] = pd.to_datetime(controls["date"], errors="coerce")
        if "imoex_ret" not in controls.columns and "Imoex" in controls.columns:
            controls = controls.sort_values("date")
            controls["imoex_ret"] = controls["Imoex"].div(controls["Imoex"].shift(1)) - 1
        if "imoex_ret" in controls.columns:
            returns = returns.merge(
                controls[["date", "imoex_ret"]],
                on="date",
                how="left",
                suffixes=("", "_ctrl"),
            )
            if "imoex_ret_ctrl" in returns.columns:
                returns["imoex_ret"] = returns["imoex_ret_ctrl"]
                returns = returns.drop(columns=["imoex_ret_ctrl"])
    else:
        print(f"Controls file not found: {controls_path}")

    returns["excess_return"] = returns["fund_ret"] - returns["imoex_ret"]
    return returns


def main() -> None:
    args = parse_args()
    apply_sample_output_paths(args)

    panel = load_y(args.y_file)
    append_cutoff = pd.Timestamp(args.append_after)
    panel_append = panel[panel["date"] > append_cutoff].copy()
    if panel_append.empty:
        raise RuntimeError(f"No rows found in {args.y_file} after {append_cutoff.date()}")
    panel_append = apply_sample_limit(panel_append, args.sample_fund_dates)

    print(
        "Append panel:",
        len(panel_append),
        "rows,",
        panel_append["fund_key"].nunique(),
        "funds,",
        panel_append["date"].min().date(),
        "to",
        panel_append["date"].max().date(),
    )

    quotes_from, quotes_to = compute_quote_window(
        panel_append=panel_append,
        lookback_days=args.lookback_days,
        from_date=args.quotes_from_date,
        to_date=args.quotes_to_date,
    )
    print(f"Quote download window: {quotes_from} .. {quotes_to}")

    quotes_df = build_append_quotes(
        panel_append=panel_append,
        args=args,
        quotes_from=quotes_from,
        quotes_to=quotes_to,
    )
    quotes_df = combine_quote_sources(quotes_df, args.nonshare_quotes_path)
    print(
        f"Combined quote pool: {len(quotes_df)} rows, "
        f"{quotes_df['ISIN'].nunique()} identifiers"
    )
    portfolios = build_portfolios(panel_append)
    append_metrics = compute_append_metrics(
        portfolios=portfolios,
        quotes=quotes_df,
        lookback_days=args.lookback_days,
    )
    append_metrics = append_metrics.sort_values(["date", "fund_key"])

    legacy_portfolio_var = load_legacy_metric(args.legacy_portfolio_var, "portfolio_var")
    legacy_fund_liquidity = load_legacy_metric(args.legacy_fund_liquidity, "fund_liquidity")
    combined_metrics = merge_metrics(
        legacy_portfolio_var=legacy_portfolio_var,
        legacy_fund_liquidity=legacy_fund_liquidity,
        append_metrics=append_metrics,
    )

    ensure_parent(args.append_metrics_out)
    append_metrics.to_csv(args.append_metrics_out, index=False)
    print(f"Append-only metrics saved to {args.append_metrics_out}")

    ensure_parent(args.full_metrics_out)
    combined_metrics.to_csv(args.full_metrics_out, index=False)
    print(f"Full updated metrics saved to {args.full_metrics_out}")
    print(
        "Full metrics coverage:",
        combined_metrics["date"].min().date(),
        "to",
        combined_metrics["date"].max().date(),
        f"({len(combined_metrics)} rows)",
    )

    append_fund_returns = build_append_fund_returns(
        panel_append=panel_append,
        fund_prices_path=args.fund_prices_path,
        controls_path=args.controls_path,
    )
    ensure_parent(args.append_fund_ret_out)
    append_fund_returns.to_csv(args.append_fund_ret_out, index=False)
    if append_fund_returns.empty:
        print("No append-period fund returns were built from the old PIF price source.")
    else:
        print(f"Append fund returns saved to {args.append_fund_ret_out}")
        print(
            "Fund return coverage:",
            append_fund_returns["date"].min().date(),
            "to",
            append_fund_returns["date"].max().date(),
            f"({len(append_fund_returns)} rows)",
        )


if __name__ == "__main__":
    main()
