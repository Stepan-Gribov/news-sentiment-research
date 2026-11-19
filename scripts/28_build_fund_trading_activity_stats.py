#!/usr/bin/env python3
"""Build descriptive statistics for mutual-fund trading activity proxies.

The source is the actual (non-interpolated) InvestFunds holdings snapshots.
The script produces compact CSV/JSON tables suitable for an editable report
workbook.  It deliberately labels weight/value changes as proxies: snapshots
do not identify individual transactions.
"""

from __future__ import annotations

import bisect
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPP_ROOT = Path(
    os.environ.get(
        "DISS_SUPP_ROOT",
        "./",
    )
)
Y_PATH = SUPP_ROOT / "funds_data/current/Y_interpolated_updated.csv"
NEWS_PATH = SUPP_ROOT / "news_data/merged/all_news_merged_unified.csv"
QUOTE_PATH = SUPP_ROOT / "analysis_outputs/quotes_data/All_instr_df_append_2024_10_2025_11.csv"
RESOLUTION_PATH = SUPP_ROOT / "analysis_outputs/quotes_data/moex_resolution_report_append_2024_10_2025_11.csv"
LONG_HOLDINGS_PATH = SUPP_ROOT / "funds_data/investfunds_fund_holdings/processed/fund_holdings_long.csv.gz"
OUT_DIR = PROJECT_ROOT / "outputs/fund_trading_activity_2026-09-08"

ACTUAL_FLAG = "0"
WEIGHT_CHANGE_THRESHOLD_PP = 0.05
SAMPLE_START = pd.Timestamp("2024-10-31")
SAMPLE_BASELINE = pd.Timestamp("2024-09-30")
SAMPLE_END = pd.Timestamp("2025-11-30")


def norm_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return " ".join(str(value).strip().casefold().replace("ё", "е").split())


def clean_string(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def load_news_companies() -> set[str]:
    companies: set[str] = set()
    for chunk in pd.read_csv(
        NEWS_PATH,
        usecols=["company"],
        dtype="string",
        encoding="utf-8",
        chunksize=250_000,
        low_memory=False,
    ):
        companies.update(norm_text(x) for x in chunk["company"].dropna().tolist())
    companies.discard("")
    return companies


def load_fund_type_map() -> dict[str, dict[str, str]]:
    """Map the Y fund key to metadata from the processed InvestFunds data."""
    result: dict[str, dict[str, str]] = {}
    if not LONG_HOLDINGS_PATH.exists():
        return result
    columns = [
        "fund_name_xls",
        "registration_number",
        "management_company",
        "fund_type_category",
    ]
    for chunk in pd.read_csv(
        LONG_HOLDINGS_PATH,
        usecols=columns,
        dtype="string",
        encoding="utf-8",
        compression="gzip",
        chunksize=200_000,
        low_memory=False,
    ):
        chunk = chunk.fillna("")
        keys = clean_string(chunk["fund_name_xls"]) + "__" + clean_string(chunk["registration_number"])
        for key, management_company, fund_type in zip(
            keys.tolist(),
            chunk["management_company"].tolist(),
            chunk["fund_type_category"].tolist(),
        ):
            if not key or key in result:
                continue
            result[key] = {
                "management_company": str(management_company or "").strip(),
                "fund_type": str(fund_type or "").strip(),
            }
    return result


def load_quote_data() -> tuple[dict[str, str], dict[str, tuple[list[pd.Timestamp], list[float]]], dict[str, object]]:
    """Return ISIN -> SECID, quote histories, and coverage metadata."""
    resolution = pd.read_csv(RESOLUTION_PATH, dtype="string", low_memory=False).fillna("")
    resolution["market"] = resolution["market"].str.lower()
    resolution["status"] = resolution["status"].str.lower()
    resolution = resolution[
        (resolution["market"] == "shares")
        & (resolution["status"] == "resolved")
        & (resolution["requested_id"].str.strip() != "")
        & (resolution["secid"].str.strip() != "")
    ]
    isin_to_secid = dict(zip(resolution["requested_id"].str.strip(), resolution["secid"].str.strip()))

    quotes = pd.read_csv(
        QUOTE_PATH,
        usecols=["SECID", "TRADEDATE", "CLOSE"],
        dtype={"SECID": "string", "TRADEDATE": "string"},
        encoding="utf-16",
        sep="\t",
        low_memory=False,
    )
    quotes["SECID"] = clean_string(quotes["SECID"])
    quotes["TRADEDATE"] = pd.to_datetime(quotes["TRADEDATE"], errors="coerce")
    quotes["CLOSE"] = pd.to_numeric(quotes["CLOSE"], errors="coerce")
    quotes = quotes[
        (quotes["SECID"] != "")
        & quotes["TRADEDATE"].notna()
        & quotes["CLOSE"].notna()
        & (quotes["CLOSE"] > 0)
    ].drop_duplicates(["SECID", "TRADEDATE"], keep="last")

    histories: dict[str, tuple[list[pd.Timestamp], list[float]]] = {}
    for secid, group in quotes.sort_values(["SECID", "TRADEDATE"]).groupby("SECID", sort=False):
        histories[str(secid)] = (
            group["TRADEDATE"].tolist(),
            group["CLOSE"].astype(float).tolist(),
        )
    quote_secids = set(histories)
    isin_to_secid = {isin: secid for isin, secid in isin_to_secid.items() if secid in quote_secids}
    metadata = {
        "resolution_share_rows": int(len(resolution)),
        "resolved_share_isins_with_local_quotes": int(len(isin_to_secid)),
        "quote_secids": int(len(quote_secids)),
        "quote_rows_with_close": int(len(quotes)),
        "quote_first_date": str(quotes["TRADEDATE"].min().date()) if len(quotes) else "",
        "quote_last_date": str(quotes["TRADEDATE"].max().date()) if len(quotes) else "",
    }
    return isin_to_secid, histories, metadata


def load_actual_holdings() -> pd.DataFrame:
    usecols = [
        "date",
        "fund_key",
        "Название фонда",
        "Название УК",
        "Класс активов",
        "Company",
        "ISIN конечного актива",
        "Сумма, руб.",
        "Доля, %",
        "total_assets",
        "is_interpolated",
    ]
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        Y_PATH,
        usecols=usecols,
        dtype="string",
        encoding="utf-16",
        sep="\t",
        chunksize=250_000,
        low_memory=False,
    ):
        chunk = chunk.fillna("")
        chunk = chunk[chunk["is_interpolated"].astype(str).str.strip() == ACTUAL_FLAG].copy()
        if chunk.empty:
            continue
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        chunk["weight_pct"] = pd.to_numeric(chunk["Доля, %"], errors="coerce").fillna(0.0)
        chunk["amount_rub"] = pd.to_numeric(chunk["Сумма, руб."], errors="coerce").fillna(0.0)
        chunk["total_assets_num"] = pd.to_numeric(chunk["total_assets"], errors="coerce")
        chunk["fund_key"] = clean_string(chunk["fund_key"])
        chunk["Company"] = clean_string(chunk["Company"])
        chunk["isin"] = clean_string(chunk["ISIN конечного актива"])
        chunk["norm_company"] = chunk["Company"].map(norm_text)
        chunk = chunk[chunk["date"].notna() & (chunk["fund_key"] != "")].copy()
        if not chunk.empty:
            frames.append(
                chunk[
                    [
                        "date",
                        "fund_key",
                        "Название фонда",
                        "Название УК",
                        "Класс активов",
                        "Company",
                        "norm_company",
                        "isin",
                        "weight_pct",
                        "amount_rub",
                        "total_assets_num",
                    ]
                ]
            )
    if not frames:
        raise RuntimeError(f"No actual holdings loaded from {Y_PATH}")
    return pd.concat(frames, ignore_index=True)


def add_fund_metadata(df: pd.DataFrame, fund_type_map: dict[str, dict[str, str]]) -> pd.DataFrame:
    mapped = df["fund_key"].map(fund_type_map)
    df["fund_type"] = mapped.map(lambda x: x.get("fund_type", "") if isinstance(x, dict) else "")
    df["management_company"] = mapped.map(
        lambda x: x.get("management_company", "") if isinstance(x, dict) else ""
    )
    df["fund_type"] = df["fund_type"].replace("", "Не определен")
    return df


def aggregate_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate duplicate security lines to one fund/date/position."""
    result = df.copy()
    result["position_id"] = result["isin"]
    missing = result["position_id"].eq("")
    result.loc[missing, "position_id"] = "NAME:" + result.loc[missing, "norm_company"]
    result = result[result["position_id"] != "NAME:"].copy()
    group_cols = ["fund_key", "date", "position_id"]
    agg = (
        result.groupby(group_cols, as_index=False, sort=False)
        .agg(
            weight_pct=("weight_pct", "sum"),
            amount_rub=("amount_rub", "sum"),
            total_assets=("total_assets_num", "median"),
            fund_name=("Название фонда", "first"),
            management_company=("Название УК", "first"),
            fund_type=("fund_type", "first"),
            asset_class=("Класс активов", "first"),
            company=("Company", "first"),
            norm_company=("norm_company", "first"),
            isin=("isin", "first"),
        )
    )
    return agg


def build_portfolio_totals(df: pd.DataFrame) -> pd.DataFrame:
    totals = (
        df.groupby(["fund_key", "date"], as_index=False, sort=False)
        .agg(
            portfolio_weight_pct=("weight_pct", "sum"),
            portfolio_assets=("total_assets_num", "median"),
            fund_name=("Название фонда", "first"),
            management_company=("Название УК", "first"),
            fund_type=("fund_type", "first"),
        )
    )
    return totals


def nearest_quote(
    secid: str,
    date: pd.Timestamp,
    histories: dict[str, tuple[list[pd.Timestamp], list[float]]],
) -> float | None:
    history = histories.get(secid)
    if history is None:
        return None
    dates, prices = history
    idx = bisect.bisect_right(dates, date) - 1
    if idx < 0:
        return None
    return float(prices[idx])


def interval_metrics(
    prev: pd.DataFrame,
    curr: pd.DataFrame,
    portfolio_prev: pd.Series | None,
    portfolio_curr: pd.Series | None,
    sample_name: str,
    isin_to_secid: dict[str, str],
    histories: dict[str, tuple[list[pd.Timestamp], list[float]]],
    use_quotes: bool,
) -> dict[str, object]:
    merged = prev[["position_id", "weight_pct", "amount_rub", "isin"]].rename(
        columns={"weight_pct": "weight_prev", "amount_rub": "amount_prev", "isin": "isin_prev"}
    ).merge(
        curr[["position_id", "weight_pct", "amount_rub", "isin"]].rename(
            columns={"weight_pct": "weight_curr", "amount_rub": "amount_curr", "isin": "isin_curr"}
        ),
        on="position_id",
        how="outer",
    )
    for col in ["weight_prev", "weight_curr", "amount_prev", "amount_curr"]:
        merged[col] = merged[col].fillna(0.0)
    merged["isin"] = (
        merged["isin_curr"].astype("string").replace("", pd.NA).fillna(merged["isin_prev"]).fillna("").astype(str)
    )
    merged["active"] = (merged["weight_prev"] > 0) | (merged["weight_curr"] > 0)
    merged = merged[merged["active"]].copy()
    changed = (merged["weight_curr"] - merged["weight_prev"]).abs() > WEIGHT_CHANGE_THRESHOLD_PP
    entries = (merged["weight_prev"] <= WEIGHT_CHANGE_THRESHOLD_PP) & (merged["weight_curr"] > WEIGHT_CHANGE_THRESHOLD_PP)
    exits = (merged["weight_curr"] <= WEIGHT_CHANGE_THRESHOLD_PP) & (merged["weight_prev"] > WEIGHT_CHANGE_THRESHOLD_PP)
    continuing = (merged["weight_prev"] > 0) & (merged["weight_curr"] > 0)
    weight_abs_change_pp = float((merged["weight_curr"] - merged["weight_prev"]).abs().sum())
    portfolio_prev_weight = float(portfolio_prev["portfolio_weight_pct"]) if portfolio_prev is not None else np.nan
    portfolio_curr_weight = float(portfolio_curr["portfolio_weight_pct"]) if portfolio_curr is not None else np.nan
    portfolio_denom = nanmean_or_nan([portfolio_prev_weight, portfolio_curr_weight])
    if not np.isfinite(portfolio_denom) or portfolio_denom <= 0:
        portfolio_denom = 100.0
    sleeve_prev = float(merged["weight_prev"].sum())
    sleeve_curr = float(merged["weight_curr"].sum())
    sleeve_denom = nanmean_or_nan([sleeve_prev, sleeve_curr])

    record: dict[str, object] = {
        "sample": sample_name,
        "fund_key": str(prev["fund_key"].iloc[0]),
        "fund_name": str(prev["fund_name"].iloc[0]),
        "management_company": str(prev["management_company"].iloc[0]),
        "fund_type": str(prev["fund_type"].iloc[0]),
        "date_prev": prev["date"].iloc[0],
        "date_curr": curr["date"].iloc[0],
        "period": curr["date"].iloc[0].strftime("%Y-%m"),
        "year": int(curr["date"].iloc[0].year),
        "interval_days": int((curr["date"].iloc[0] - prev["date"].iloc[0]).days),
        "positions_union": int(len(merged)),
        "positions_current": int((merged["weight_curr"] > WEIGHT_CHANGE_THRESHOLD_PP).sum()),
        "changed_positions": int(changed.sum()),
        "entry_positions": int(entries.sum()),
        "exit_positions": int(exits.sum()),
        "changed_positions_share_pct": float(changed.mean() * 100) if len(merged) else np.nan,
        "entry_positions_share_pct": float(entries.mean() * 100) if len(merged) else np.nan,
        "exit_positions_share_pct": float(exits.mean() * 100) if len(merged) else np.nan,
        "weight_abs_change_pp": weight_abs_change_pp,
        "weight_turnover_proxy_pct": float(weight_abs_change_pp / 2.0 / portfolio_denom * 100.0),
        "sleeve_weight_prev_pct": sleeve_prev,
        "sleeve_weight_curr_pct": sleeve_curr,
        "sleeve_turnover_proxy_pct": float(weight_abs_change_pp / 2.0 / sleeve_denom * 100.0)
        if np.isfinite(sleeve_denom) and sleeve_denom > 0
        else np.nan,
        "portfolio_assets_prev": finite_float(portfolio_prev["portfolio_assets"])
        if portfolio_prev is not None
        else np.nan,
        "portfolio_assets_curr": finite_float(portfolio_curr["portfolio_assets"])
        if portfolio_curr is not None
        else np.nan,
        "quote_adj_turnover_proxy_pct": np.nan,
        "quote_adj_positions": 0,
        "quote_adj_continuing_share_pct": np.nan,
    }

    if use_quotes:
        prev_date = prev["date"].iloc[0]
        curr_date = curr["date"].iloc[0]
        active_values: list[float] = []
        continuing_with_quotes = 0
        for row in merged[continuing].itertuples(index=False):
            isin = str(row.isin or "")
            secid = isin_to_secid.get(isin)
            if not secid:
                continue
            price_prev = nearest_quote(secid, prev_date, histories)
            price_curr = nearest_quote(secid, curr_date, histories)
            if price_prev is None or price_curr is None:
                continue
            expected_curr = float(row.amount_prev) * price_curr / price_prev
            active_values.append(float(row.amount_curr) - expected_curr)
            continuing_with_quotes += 1
        assets_denom = nanmean_or_nan(
            [record["portfolio_assets_prev"], record["portfolio_assets_curr"]]
        )
        if active_values and np.isfinite(assets_denom) and assets_denom > 0:
            record["quote_adj_turnover_proxy_pct"] = float(
                np.abs(np.asarray(active_values)).sum() / 2.0 / assets_denom * 100.0
            )
        record["quote_adj_positions"] = int(continuing_with_quotes)
        cont_count = int(continuing.sum())
        record["quote_adj_continuing_share_pct"] = (
            float(continuing_with_quotes / cont_count * 100.0) if cont_count else np.nan
        )
    return record


def build_intervals(
    positions: pd.DataFrame,
    portfolio_totals: pd.DataFrame,
    sample_name: str,
    date_min: pd.Timestamp,
    date_max: pd.Timestamp,
    isin_to_secid: dict[str, str],
    histories: dict[str, tuple[list[pd.Timestamp], list[float]]],
    use_quotes: bool,
) -> pd.DataFrame:
    positions = positions[(positions["date"] >= date_min) & (positions["date"] <= date_max)].copy()
    totals_lookup = portfolio_totals.set_index(["fund_key", "date"])
    records: list[dict[str, object]] = []
    for fund_key, fund_group in positions.groupby("fund_key", sort=False):
        by_date = {date: group for date, group in fund_group.groupby("date", sort=True)}
        dates = sorted(by_date)
        for prev_date, curr_date in zip(dates[:-1], dates[1:]):
            prev_group = by_date[prev_date]
            curr_group = by_date[curr_date]
            prev_total = totals_lookup.loc[(fund_key, prev_date)] if (fund_key, prev_date) in totals_lookup.index else None
            curr_total = totals_lookup.loc[(fund_key, curr_date)] if (fund_key, curr_date) in totals_lookup.index else None
            records.append(
                interval_metrics(
                    prev_group,
                    curr_group,
                    prev_total,
                    curr_total,
                    sample_name,
                    isin_to_secid,
                    histories,
                    use_quotes,
                )
            )
    return pd.DataFrame.from_records(records)


def safe_quantile(series: pd.Series, q: float) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna()
    return float(series.quantile(q)) if len(series) else np.nan


def finite_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def nanmean_or_nan(values: Iterable[object]) -> float:
    numbers = np.asarray([finite_float(value) for value in values], dtype=float)
    finite = numbers[np.isfinite(numbers)]
    return float(finite.mean()) if len(finite) else np.nan


def summarize_intervals(intervals: pd.DataFrame, sample_name: str) -> pd.DataFrame:
    if intervals.empty:
        return pd.DataFrame()
    grouped = intervals.groupby("period", sort=True)
    rows = []
    for period, group in grouped:
        rows.append(
            {
                "sample": sample_name,
                "period": period,
                "year": int(group["year"].iloc[0]),
                "funds_with_interval": int(group["fund_key"].nunique()),
                "interval_observations": int(len(group)),
                "median_turnover_proxy_pct": float(group["weight_turnover_proxy_pct"].median()),
                "mean_turnover_proxy_pct": float(group["weight_turnover_proxy_pct"].mean()),
                "p75_turnover_proxy_pct": safe_quantile(group["weight_turnover_proxy_pct"], 0.75),
                "median_changed_positions_share_pct": float(group["changed_positions_share_pct"].median()),
                "median_entry_exit_count": float((group["entry_positions"] + group["exit_positions"]).median()),
                "median_positions_current": float(group["positions_current"].median()),
                "median_sleeve_turnover_proxy_pct": float(group["sleeve_turnover_proxy_pct"].median()),
                "median_quote_adj_turnover_proxy_pct": float(group["quote_adj_turnover_proxy_pct"].median()),
                "quote_adj_intervals": int(group["quote_adj_turnover_proxy_pct"].notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_by_year(intervals: pd.DataFrame, sample_name: str) -> pd.DataFrame:
    if intervals.empty:
        return pd.DataFrame()
    rows = []
    for year, group in intervals.groupby("year", sort=True):
        rows.append(
            {
                "sample": sample_name,
                "year": int(year),
                "funds_with_interval": int(group["fund_key"].nunique()),
                "interval_observations": int(len(group)),
                "median_turnover_proxy_pct": float(group["weight_turnover_proxy_pct"].median()),
                "mean_turnover_proxy_pct": float(group["weight_turnover_proxy_pct"].mean()),
                "p75_turnover_proxy_pct": safe_quantile(group["weight_turnover_proxy_pct"], 0.75),
                "median_changed_positions_share_pct": float(group["changed_positions_share_pct"].median()),
                "median_entry_exit_count": float((group["entry_positions"] + group["exit_positions"]).median()),
                "median_positions_current": float(group["positions_current"].median()),
                "median_sleeve_turnover_proxy_pct": float(group["sleeve_turnover_proxy_pct"].median()),
                "median_quote_adj_turnover_proxy_pct": float(group["quote_adj_turnover_proxy_pct"].median()),
                "quote_adj_intervals": int(group["quote_adj_turnover_proxy_pct"].notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_funds(intervals: pd.DataFrame, sample_name: str) -> pd.DataFrame:
    if intervals.empty:
        return pd.DataFrame()
    rows = []
    for (fund_key, fund_name, management_company, fund_type), group in intervals.groupby(
        ["fund_key", "fund_name", "management_company", "fund_type"], sort=False
    ):
        rows.append(
            {
                "sample": sample_name,
                "fund_key": fund_key,
                "fund_name": fund_name,
                "management_company": management_company,
                "fund_type": fund_type,
                "intervals": int(len(group)),
                "first_period": str(group["period"].min()),
                "last_period": str(group["period"].max()),
                "median_turnover_proxy_pct": float(group["weight_turnover_proxy_pct"].median()),
                "mean_turnover_proxy_pct": float(group["weight_turnover_proxy_pct"].mean()),
                "p75_turnover_proxy_pct": safe_quantile(group["weight_turnover_proxy_pct"], 0.75),
                "median_changed_positions_share_pct": float(group["changed_positions_share_pct"].median()),
                "median_entry_positions": float(group["entry_positions"].median()),
                "median_exit_positions": float(group["exit_positions"].median()),
                "median_positions_current": float(group["positions_current"].median()),
                "median_sleeve_weight_pct": float(
                    pd.concat([group["sleeve_weight_prev_pct"], group["sleeve_weight_curr_pct"]]).median()
                ),
                "median_sleeve_turnover_proxy_pct": float(group["sleeve_turnover_proxy_pct"].median()),
                "median_quote_adj_turnover_proxy_pct": float(group["quote_adj_turnover_proxy_pct"].median()),
                "quote_adj_intervals": int(group["quote_adj_turnover_proxy_pct"].notna().sum()),
                "quote_adj_coverage_median_pct": float(group["quote_adj_continuing_share_pct"].median()),
            }
        )
    return pd.DataFrame(rows)


def summarize_types(intervals: pd.DataFrame, sample_name: str) -> pd.DataFrame:
    if intervals.empty:
        return pd.DataFrame()
    rows = []
    for (fund_type,), group in intervals.groupby(["fund_type"], sort=False):
        rows.append(
            {
                "sample": sample_name,
                "fund_type": fund_type,
                "funds_with_interval": int(group["fund_key"].nunique()),
                "interval_observations": int(len(group)),
                "median_turnover_proxy_pct": float(group["weight_turnover_proxy_pct"].median()),
                "mean_turnover_proxy_pct": float(group["weight_turnover_proxy_pct"].mean()),
                "p75_turnover_proxy_pct": safe_quantile(group["weight_turnover_proxy_pct"], 0.75),
                "median_changed_positions_share_pct": float(group["changed_positions_share_pct"].median()),
                "median_entry_exit_count": float((group["entry_positions"] + group["exit_positions"]).median()),
                "median_positions_current": float(group["positions_current"].median()),
                "median_sleeve_turnover_proxy_pct": float(group["sleeve_turnover_proxy_pct"].median()),
                "median_quote_adj_turnover_proxy_pct": float(group["quote_adj_turnover_proxy_pct"].median()),
                "quote_adj_intervals": int(group["quote_adj_turnover_proxy_pct"].notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, name: str) -> None:
    out = OUT_DIR / name
    serial = df.copy()
    for col in serial.columns:
        if pd.api.types.is_datetime64_any_dtype(serial[col]):
            serial[col] = serial[col].dt.strftime("%Y-%m-%d")
    serial.to_csv(out, index=False, encoding="utf-8-sig")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in [Y_PATH, NEWS_PATH, QUOTE_PATH, RESOLUTION_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)

    news_companies = load_news_companies()
    fund_type_map = load_fund_type_map()
    isin_to_secid, histories, quote_metadata = load_quote_data()
    holdings = load_actual_holdings()
    holdings = add_fund_metadata(holdings, fund_type_map)
    holdings["is_stock"] = holdings["Класс активов"].eq("stocks")
    holdings["has_news"] = holdings["norm_company"].isin(news_companies)
    holdings["has_quote"] = holdings["isin"].isin(isin_to_secid)

    portfolio_totals = build_portfolio_totals(holdings)
    stock = aggregate_positions(holdings[holdings["is_stock"]].copy())
    stock_news_quote = aggregate_positions(
        holdings[holdings["is_stock"] & holdings["has_news"] & holdings["has_quote"]].copy()
    )

    # Broad historical sample: all actual stock holdings from the first
    # InvestFunds snapshot through the latest one.
    stock_intervals = build_intervals(
        stock,
        portfolio_totals,
        "Все акции",
        stock["date"].min(),
        stock["date"].max(),
        isin_to_secid,
        histories,
        False,
    )

    # Narrow sample: actual snapshots inside the period covered by the local
    # quote append and exact company-name news join. Include September as the
    # weight-only baseline for the first October observation.
    matched_intervals = build_intervals(
        stock_news_quote,
        portfolio_totals,
        "Новости + котировки",
        SAMPLE_BASELINE,
        SAMPLE_END,
        isin_to_secid,
        histories,
        True,
    )
    matched_intervals = matched_intervals[matched_intervals["date_curr"] >= SAMPLE_START].copy()

    # Separate quote-adjusted rows so the workbook can explicitly show their
    # narrower continuing-position coverage.
    interval_stats = pd.concat(
        [
            summarize_intervals(stock_intervals, "Все акции"),
            summarize_intervals(matched_intervals, "Новости + котировки"),
        ],
        ignore_index=True,
    )
    annual_stats = pd.concat(
        [
            summarize_by_year(stock_intervals, "Все акции"),
            summarize_by_year(matched_intervals, "Новости + котировки"),
        ],
        ignore_index=True,
    )
    fund_stats = pd.concat(
        [
            summarize_funds(stock_intervals, "Все акции"),
            summarize_funds(matched_intervals, "Новости + котировки"),
        ],
        ignore_index=True,
    )
    type_stats = pd.concat(
        [
            summarize_types(stock_intervals, "Все акции"),
            summarize_types(matched_intervals, "Новости + котировки"),
        ],
        ignore_index=True,
    )

    sample_period = holdings[(holdings["date"] >= SAMPLE_START) & (holdings["date"] <= SAMPLE_END)]
    sample_period_stock = sample_period[sample_period["is_stock"]]
    matched_period = sample_period_stock[sample_period_stock["has_news"] & sample_period_stock["has_quote"]]
    coverage_rows = [
        {
            "metric": "Actual holdings rows in quote/news window",
            "value": int(len(sample_period)),
            "unit": "rows",
            "details": "2024-10-31 to 2025-11-30; is_interpolated=0",
        },
        {
            "metric": "Funds in all actual holdings window",
            "value": int(sample_period["fund_key"].nunique()),
            "unit": "funds",
            "details": "all asset classes",
        },
        {
            "metric": "Stock funds in actual holdings window",
            "value": int(sample_period_stock["fund_key"].nunique()),
            "unit": "funds",
            "details": "Класс активов = stocks",
        },
        {
            "metric": "Stock rows with news company match",
            "value": int(sample_period_stock[sample_period_stock["has_news"]].shape[0]),
            "unit": "rows",
            "details": f"exact normalized company join; news labels={len(news_companies):,}",
        },
        {
            "metric": "Stock rows with quote-backed ISIN",
            "value": int(sample_period_stock[sample_period_stock["has_quote"]].shape[0]),
            "unit": "rows",
            "details": f"shares mapping + local quote history; ISINs={len(isin_to_secid):,}",
        },
        {
            "metric": "Stock rows in news + quote sample",
            "value": int(sample_period_stock[sample_period_stock["has_news"] & sample_period_stock["has_quote"]].shape[0]),
            "unit": "rows",
            "details": "used for the narrow sample",
        },
        {
            "metric": "Funds in news + quote sample",
            "value": int(sample_period_stock[sample_period_stock["has_news"] & sample_period_stock["has_quote"]]["fund_key"].nunique()),
            "unit": "funds",
            "details": "funds with matched stock positions",
        },
        {
            "metric": "Companies in news + quote sample",
            "value": int(sample_period_stock[sample_period_stock["has_news"] & sample_period_stock["has_quote"]]["norm_company"].nunique()),
            "unit": "companies",
            "details": "exact normalized company labels",
        },
        {
            "metric": "ISINs in news + quote sample",
            "value": int(sample_period_stock[sample_period_stock["has_news"] & sample_period_stock["has_quote"]]["isin"].nunique()),
            "unit": "ISINs",
            "details": "resolved to a local MOEX share history",
        },
        {
            "metric": "Actual report dates in quote/news window",
            "value": int(sample_period["date"].nunique()),
            "unit": "dates",
            "details": "month-end snapshots",
        },
    ]
    coverage = pd.DataFrame(coverage_rows)

    # Compact overall table for the report's first sheet.
    overall_rows = []
    for sample_name, intervals in [
        ("Все акции", stock_intervals),
        ("Новости + котировки", matched_intervals),
    ]:
        overall_rows.append(
            {
                "sample": sample_name,
                "period": (
                    f"{intervals['date_prev'].min().strftime('%Y-%m')} – {intervals['date_curr'].max().strftime('%Y-%m')}"
                    if not intervals.empty
                    else ""
                ),
                "funds_with_interval": int(intervals["fund_key"].nunique()) if not intervals.empty else 0,
                "interval_observations": int(len(intervals)),
                "median_turnover_proxy_pct": float(intervals["weight_turnover_proxy_pct"].median()) if not intervals.empty else np.nan,
                "mean_turnover_proxy_pct": float(intervals["weight_turnover_proxy_pct"].mean()) if not intervals.empty else np.nan,
                "p75_turnover_proxy_pct": safe_quantile(intervals["weight_turnover_proxy_pct"], 0.75),
                "median_changed_positions_share_pct": float(intervals["changed_positions_share_pct"].median()) if not intervals.empty else np.nan,
                "median_entry_exit_count": float((intervals["entry_positions"] + intervals["exit_positions"]).median()) if not intervals.empty else np.nan,
                "median_positions_current": float(intervals["positions_current"].median()) if not intervals.empty else np.nan,
                "median_quote_adj_turnover_proxy_pct": float(intervals["quote_adj_turnover_proxy_pct"].median()) if not intervals.empty else np.nan,
                "quote_adj_intervals": int(intervals["quote_adj_turnover_proxy_pct"].notna().sum()) if not intervals.empty else 0,
            }
        )
    overall = pd.DataFrame(overall_rows)

    write_csv(overall, "overall_summary.csv")
    write_csv(coverage, "coverage.csv")
    write_csv(interval_stats, "interval_stats.csv")
    write_csv(annual_stats, "annual_stats.csv")
    write_csv(fund_stats.sort_values(["sample", "median_turnover_proxy_pct"], ascending=[True, False]), "fund_stats.csv")
    write_csv(type_stats.sort_values(["sample", "median_turnover_proxy_pct"], ascending=[True, False]), "type_stats.csv")

    # Small lookup table of matched securities for reproducibility/audit.
    matched_securities = (
        sample_period_stock[sample_period_stock["has_news"] & sample_period_stock["has_quote"]]
        .groupby(["isin", "norm_company"], as_index=False)
        .agg(
            company=("Company", "first"),
            funds=("fund_key", "nunique"),
            observations=("fund_key", "size"),
            first_date=("date", "min"),
            last_date=("date", "max"),
        )
        .sort_values(["funds", "observations"], ascending=False)
    )
    write_csv(matched_securities, "matched_securities.csv")

    metadata = {
        "source_y": str(Y_PATH),
        "source_news": str(NEWS_PATH),
        "source_quotes": str(QUOTE_PATH),
        "source_resolution": str(RESOLUTION_PATH),
        "source_fund_metadata": str(LONG_HOLDINGS_PATH),
        "actual_flag": ACTUAL_FLAG,
        "weight_change_threshold_pp": WEIGHT_CHANGE_THRESHOLD_PP,
        "broad_sample": {
            "name": "Все акции",
            "date_first": str(stock["date"].min().date()),
            "date_last": str(stock["date"].max().date()),
            "funds": int(stock["fund_key"].nunique()),
            "positions": int(stock["position_id"].nunique()),
            "intervals": int(len(stock_intervals)),
        },
        "matched_sample": {
            "name": "Новости + котировки",
            "date_first": str(SAMPLE_START.date()),
            "date_baseline": str(SAMPLE_BASELINE.date()),
            "date_last": str(SAMPLE_END.date()),
            "funds": int(matched_period["fund_key"].nunique()),
            "companies": int(matched_period["norm_company"].nunique()),
            "isins": int(matched_period["isin"].nunique()),
            "intervals": int(len(matched_intervals)),
        },
        "news_company_labels": int(len(news_companies)),
        "fund_metadata_keys": int(len(fund_type_map)),
        "quote": quote_metadata,
    }
    with (OUT_DIR / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, default=str)

    print(json.dumps(metadata, ensure_ascii=False, indent=2, default=str))
    print(f"Wrote analysis tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
