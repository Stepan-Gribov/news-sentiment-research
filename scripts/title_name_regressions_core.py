"""Strict, source-independent transformations for the title/name Gemma study.

Inputs use an explicit normalized contract. No parser guesses, historical LLM
corpora, interpolated holdings, or default zero asset returns are accepted here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


class IntegrityError(ValueError):
    pass


class NotEstimable(ValueError):
    pass


def require_columns(frame, names, label):
    absent = sorted(set(names) - set(frame.columns))
    if absent:
        raise IntegrityError(f"{label}: missing columns {absent}")


def unique_keys(frame, keys, label):
    require_columns(frame, keys, label)
    if frame[keys].isna().any(axis=None):
        raise IntegrityError(f"{label}: null key in {keys}")
    duplicate = frame.duplicated(keys, keep=False)
    if duplicate.any():
        raise IntegrityError(f"{label}: {int(duplicate.sum())} rows have duplicate keys {keys}")


def safe_many_to_one(left, right, on, *, how="left", label="join"):
    unique_keys(right, on, label + " right")
    before = len(left)
    result = left.merge(right, on=on, how=how, validate="many_to_one")
    if how == "left" and len(result) != before:
        raise IntegrityError(f"{label}: a left join changed row count")
    return result


def validate_issuer_bridge(bridge):
    require_columns(bridge, ["isin", "company_id"], "issuer bridge")
    if bridge[["isin", "company_id"]].isna().any(axis=None):
        raise IntegrityError("issuer bridge: missing identifier")
    pairs = bridge[["isin", "company_id"]].drop_duplicates()
    ambiguous = pairs.groupby("isin", observed=True).company_id.nunique().gt(1)
    if ambiguous.any():
        raise IntegrityError(f"issuer bridge: {int(ambiguous.sum())} ISINs map to multiple company IDs")
    return pairs.reset_index(drop=True)


def report_intervals(reports):
    """Adjacent actual holdings dates; publication dates are not substituted."""
    required = ["fund_id", "report_date", "report_id", "full_portfolio"]
    require_columns(reports, required, "reports")
    out = reports[required].copy()
    out["report_date"] = pd.to_datetime(out.report_date, errors="raise")
    unique_keys(out, ["fund_id", "report_date"], "reports; revisions require an explicit policy")
    if out.full_portfolio.isna().any():
        raise IntegrityError("reports: portfolio completeness must be explicit")
    out = out.sort_values(["fund_id", "report_date"])
    grouped = out.groupby("fund_id", observed=True)
    out["end_date"] = grouped.report_date.shift(-1)
    out["closing_report_id"] = grouped.report_id.shift(-1)
    out["closing_full_portfolio"] = grouped.full_portfolio.shift(-1)
    out = out.loc[out.end_date.notna()].rename(columns={
        "report_date": "start_date", "report_id": "opening_report_id",
        "full_portfolio": "opening_full_portfolio"})
    if not out.end_date.gt(out.start_date).all():
        raise IntegrityError("report intervals must have positive length")
    out["interval_id"] = ["fi_" + hashlib.blake2b(
        f"{f}\t{a.isoformat()}\t{b.isoformat()}".encode(), digest_size=16).hexdigest()
        for f, a, b in zip(out.fund_id, out.start_date, out.end_date)]
    out["month"] = out.end_date.dt.to_period("M").astype(str)
    unique_keys(out, ["interval_id"], "intervals")
    return out.reset_index(drop=True)


def interval_total_return(start_price, end_price, split_factor,
                          cash_per_opening_share, *, coverage_complete):
    """Terminal distribution wealth and splits are per opening share.

    For baseline reinvested returns, cash_per_opening_share must already include
    the verified reinvestment wealth through the endpoint. This helper does not
    turn an incomplete cash-dividend table into a total-return series.
    """
    if not coverage_complete:
        return np.nan
    values = np.asarray([start_price, end_price, split_factor, cash_per_opening_share], dtype=float)
    if not np.isfinite(values).all() or start_price <= 0 or end_price < 0 or split_factor <= 0:
        return np.nan
    return (end_price * split_factor + cash_per_opening_share) / start_price - 1


def build_weight_panel(holdings, intervals, returns, *, weight_tolerance=1e-8):
    """Union endpoints and revalue the entire opening portfolio, including residual.

    holdings: fund_id, report_date, security_id, company_id (nullable), weight,
              is_leaf, asset_type. Weights are fractions of a common portfolio
              denominator; cash must be present as a position or explicit residual.
    returns: interval_id, security_id, total_return, return_coverage_complete.
    Non-leaf categories are excluded explicitly; duplicated leaf identifiers fail.
    """
    require_columns(holdings, ["fund_id", "report_date", "security_id", "company_id",
                              "weight", "is_leaf", "asset_type"], "holdings")
    if holdings.is_leaf.isna().any():
        raise IntegrityError("holdings: leaf/category classification is missing")
    h = holdings.loc[holdings.is_leaf.astype(bool)].copy()
    h["report_date"] = pd.to_datetime(h.report_date, errors="raise")
    h["weight"] = pd.to_numeric(h.weight, errors="raise")
    if h.weight.dropna().lt(0).any():
        raise NotEstimable("Short/negative asset weights require a separate frozen portfolio rule")
    unique_keys(h, ["fund_id", "report_date", "security_id"], "normalized leaf holdings")
    unique_keys(intervals, ["interval_id"], "intervals")
    require_columns(returns, ["interval_id", "security_id", "total_return", "return_coverage_complete"], "returns")
    unique_keys(returns, ["interval_id", "security_id"], "returns")
    base = intervals.copy()
    for col in ["start_date", "end_date"]:
        base[col] = pd.to_datetime(base[col], errors="raise")
    if not base.opening_full_portfolio.isin([True, False]).all() or not base.closing_full_portfolio.isin([True, False]).all():
        raise IntegrityError("intervals: full-portfolio flags must be known booleans")
    endpoint_frames = []
    for side, date in [("open", "start_date"), ("close", "end_date")]:
        left = base[["interval_id", "fund_id", date]]
        unique_keys(left, ["fund_id", date], f"{side} report boundary")
        part = left.merge(h, left_on=["fund_id", date], right_on=["fund_id", "report_date"],
                          how="inner", validate="one_to_many")
        part = part[["interval_id", "security_id", "weight", "company_id", "asset_type"]].rename(
            columns={x: x + "_" + side for x in ["weight", "company_id", "asset_type"]})
        part["present_" + side] = True
        endpoint_frames.append(part)
    out = endpoint_frames[0].merge(endpoint_frames[1], on=["interval_id", "security_id"],
                                   how="outer", validate="one_to_one")
    out = safe_many_to_one(out, base, ["interval_id"], label="interval endpoints")
    for side, complete in [("open", "opening_full_portfolio"), ("close", "closing_full_portfolio")]:
        out["present_" + side] = out["present_" + side].fillna(False).astype(bool)
        absent_known = ~out["present_" + side] & out[complete].astype(bool)
        # A known absence in a complete report is zero; a missing reported weight is not.
        out.loc[absent_known, "weight_" + side] = 0.0
    disagree = (out.company_id_open.notna() & out.company_id_close.notna()
                & out.company_id_open.ne(out.company_id_close))
    if disagree.any():
        raise IntegrityError("An unchanged security has conflicting canonical company IDs")
    out["company_id"] = out.company_id_open.combine_first(out.company_id_close)
    out["asset_type"] = out.asset_type_open.combine_first(out.asset_type_close)

    residuals = []
    for iid, g in out.groupby("interval_id", sort=False, observed=True):
        residual = {}
        for side, complete in [("open", "opening_full_portfolio"), ("close", "closing_full_portfolio")]:
            w = g["weight_" + side]
            known = bool(g[complete].iloc[0]) and w.notna().all()
            total = w.sum(min_count=1)
            residual[side] = max(0.0, 1 - total) if known and total <= 1 + weight_tolerance else np.nan
        if any(np.isfinite(x) and x > weight_tolerance for x in residual.values()):
            row = {k: g[k].iloc[0] for k in base.columns}
            row.update(security_id="__RESIDUAL__", company_id=None, asset_type="residual",
                       weight_open=residual["open"], weight_close=residual["close"],
                       present_open=False, present_close=False)
            residuals.append(row)
    if residuals:
        out = pd.concat([out, pd.DataFrame(residuals)], ignore_index=True)
    out = safe_many_to_one(out, returns, ["interval_id", "security_id"], label="interval total returns")
    out["raw_delta_weight"] = out.weight_close - out.weight_open
    out["full_entry"] = out.weight_open.eq(0) & out.weight_close.gt(0)
    out["full_exit"] = out.weight_open.gt(0) & out.weight_close.eq(0)
    out["return_known"] = (out.return_coverage_complete.eq(True)
                            & np.isfinite(out.total_return) & out.total_return.ge(-1))
    no_open_exposure = out.weight_open.eq(0)
    out["opening_growth_value"] = np.where(no_open_exposure, 0.0,
        np.where(out.return_known, out.weight_open * (1 + out.total_return), np.nan))
    out["weight_buy_and_hold"] = np.nan
    out["AT"] = np.nan
    out["at_gate_reason"] = ""
    out["portfolio_return_weight_coverage"] = np.nan
    out["portfolio_bh_denominator"] = np.nan
    for iid, idx in out.groupby("interval_id", sort=False, observed=True).groups.items():
        g = out.loc[idx]
        w = g.weight_open
        denom_w = w.sum(min_count=1)
        if w.notna().all() and denom_w > 0:
            covered = w.where(g.return_known | w.eq(0), 0).sum()
            out.loc[idx, "portfolio_return_weight_coverage"] = covered / denom_w
        reason = None
        if not bool(g.opening_full_portfolio.iloc[0]): reason = "opening_report_incomplete"
        elif not bool(g.closing_full_portfolio.iloc[0]): reason = "closing_report_incomplete"
        elif g[["weight_open", "weight_close"]].isna().any(axis=None): reason = "missing_weight"
        elif abs(w.sum() - 1) > weight_tolerance or abs(g.weight_close.sum() - 1) > weight_tolerance:
            reason = "portfolio_denominator_not_reconciled"
        elif g.opening_growth_value.isna().any(): reason = "missing_opening_asset_total_return"
        else:
            denominator = g.opening_growth_value.sum()
            if not np.isfinite(denominator) or denominator <= 0:
                reason = "nonpositive_bh_denominator"
        if reason:
            out.loc[idx, "at_gate_reason"] = reason
            continue
        out.loc[idx, "portfolio_bh_denominator"] = denominator
        out.loc[idx, "weight_buy_and_hold"] = g.opening_growth_value / denominator
        out.loc[idx, "AT"] = g.weight_close - g.opening_growth_value / denominator
        out.loc[idx, "at_gate_reason"] = "eligible"
    unique_keys(out, ["interval_id", "security_id"], "portfolio panel")
    return out


def aggregate_news_windows(targets, candidates, *, allow_partial=False, minimum_valid_fraction=1.0):
    """Prefix-sum counts for company windows (news_start, news_end].

    event_date must be a pre-normalized date under the frozen date policy.
    candidates includes every expected mapping pair, even if LLM output is missing.
    """
    require_columns(targets, ["interval_id", "company_id", "news_start", "news_end"], "news targets")
    unique_keys(targets, ["interval_id", "company_id"], "news targets")
    require_columns(candidates, ["pair_id", "article_id", "company_id", "event_date",
                                  "attempted", "valid", "relevance", "sentiment_label"], "news candidates")
    unique_keys(candidates, ["pair_id"], "news candidates")
    unique_keys(candidates, ["article_id", "company_id"], "article/company mapping")
    c = candidates.copy()
    c["event_date"] = pd.to_datetime(c.event_date, errors="coerce")
    if getattr(c.event_date.dtype, "tz", None) is not None:
        raise IntegrityError("Normalize source timezone explicitly before news aggregation")
    bad_valid = c.valid.eq(True) & (~c.attempted.eq(True) | ~c.relevance.isin(["relevant", "irrelevant"]))
    relevant = c.valid.eq(True) & c.relevance.eq("relevant")
    if bad_valid.any() or (relevant & ~c.sentiment_label.isin(["positive", "negative", "neutral"])).any():
        raise IntegrityError("LLM valid flag disagrees with its relevance/sentiment fields")
    c["_expected"] = 1
    c["_attempted"] = c.attempted.eq(True).astype(int)
    c["_valid"] = c.valid.eq(True).astype(int)
    c["_relevant"] = relevant.astype(int)
    for label in ["positive", "negative", "neutral"]:
        c["_" + label] = (relevant & c.sentiment_label.eq(label)).astype(int)
    count_fields = ["expected", "attempted", "valid", "relevant", "positive", "negative", "neutral"]
    out = targets.copy()
    out["news_start"] = pd.to_datetime(out.news_start, errors="raise")
    out["news_end"] = pd.to_datetime(out.news_end, errors="raise")
    if not out.news_end.gt(out.news_start).all():
        raise IntegrityError("News exposure windows must have positive length")
    for field in count_fields:
        out["n_" + field] = 0
    dated = c.loc[c.event_date.notna()].sort_values("event_date")
    grouped = {key: g for key, g in dated.groupby("company_id", observed=True)}
    for company, indices in out.groupby("company_id", observed=True).groups.items():
        if company not in grouped:
            continue
        g = grouped[company]
        days = g.event_date.to_numpy(dtype="datetime64[ns]")
        left = np.searchsorted(days, out.loc[indices, "news_start"].to_numpy(dtype="datetime64[ns]"), side="right")
        right = np.searchsorted(days, out.loc[indices, "news_end"].to_numpy(dtype="datetime64[ns]"), side="right")
        values = g[["_" + x for x in count_fields]].to_numpy(dtype=np.int64)
        prefix = np.vstack([np.zeros((1, values.shape[1]), dtype=np.int64), values.cumsum(axis=0)])
        out.loc[indices, ["n_" + x for x in count_fields]] = prefix[right] - prefix[left]
    out["n_errors_or_unattempted"] = out.n_expected - out.n_valid
    out["valid_fraction"] = out.n_valid.div(out.n_expected.replace(0, np.nan))
    out["tone_observed"] = (out.n_positive - out.n_negative).div(out.n_relevant.replace(0, np.nan))
    complete = out.n_valid.eq(out.n_expected)
    eligible_coverage = complete | (allow_partial & out.valid_fraction.ge(minimum_valid_fraction))
    out["Tone"] = out.tone_observed.where(eligible_coverage)
    out["news_status"] = np.select(
        [out.n_expected.eq(0), ~complete, out.n_relevant.eq(0)],
        ["no_title_candidates", "partial_or_error_coverage", "no_relevant_news"],
        default="complete_relevant_news")
    audit = {"undated_pairs": int(c.event_date.isna().sum()), "expected_pairs": len(c),
             "dated_pairs": len(dated), "event_window": "(news_start, news_end]"}
    return out, audit


def attach_exact_month_lags(panel, controls, keys, columns, *, panel_month="month", control_month="month", lag=1):
    """A missing calendar month stays missing instead of shifting to a prior row."""
    if lag < 1:
        raise IntegrityError("Baseline controls must be lagged by at least one calendar month")
    require_columns(panel, keys + [panel_month], "lag target")
    require_columns(controls, keys + [control_month] + columns, "lag controls")
    right = controls[keys + [control_month] + columns].copy()
    right["_control_month"] = pd.PeriodIndex(right[control_month], freq="M").astype(str)
    right = right.drop(columns=[control_month])
    unique_keys(right, keys + ["_control_month"], "calendar controls")
    left = panel.copy()
    left["_control_month"] = (pd.PeriodIndex(left[panel_month], freq="M") - lag).astype(str)
    right = right.rename(columns={x: f"{x}_lag{lag}" for x in columns})
    return safe_many_to_one(left, right, keys + ["_control_month"], label="exact month lag")


def nts_by_interval(panel, *, signal="S", activity="AT", zero_tolerance=1e-12,
                    minimum_companies=2,minimum_trade_coverage=.5):
    """Input must include all corporate positions, not only news-covered rows."""
    require_columns(panel, ["fund_id", "interval_id","company_id", signal, activity], "NTS")
    rows = []
    for (fund, interval), g in panel.groupby(["fund_id", "interval_id"], observed=True):
        eligible = g[signal].notna() & g[activity].notna() & g[signal].abs().gt(zero_tolerance)
        a = g.loc[eligible, activity].to_numpy(float)
        s = g.loc[eligible, signal].to_numpy(float)
        denominator = np.dot(np.abs(a), np.abs(s))
        all_mass=float(g[activity].abs().sum(min_count=1))
        coverage=float(np.abs(a).sum()/all_mass) if all_mass>zero_tolerance else np.nan
        companies=g.loc[eligible,"company_id"].nunique()
        reason="defined"
        if g[activity].isna().any():reason="corporate_AT_coverage_incomplete"
        elif denominator<=zero_tolerance:reason="zero_signal_or_zero_rebalancing"
        elif companies<minimum_companies:reason="insufficient_companies"
        elif not np.isfinite(coverage) or coverage<minimum_trade_coverage:reason="low_active_trade_coverage"
        value = float(np.dot(a, s) / denominator) if reason=="defined" else np.nan
        if np.isfinite(value) and abs(value) > 1 + 1e-10:
            raise IntegrityError("NTS outside [-1, 1]")
        rows.append(dict(fund_id=fund, interval_id=interval, NTS=value,
                         nts_denominator=denominator, nts_eligible_positions=int(eligible.sum()),
                         nts_eligible_companies=companies,nts_active_trade_coverage=coverage,
                         nts_status=reason))
    return pd.DataFrame(rows)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def freeze_json(path, payload):
    """Create once; an incompatible rerun must use a new output directory."""
    path = Path(path)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8-sig"))
        if old != payload:
            raise IntegrityError("The frozen specification/input fingerprint differs; use a new run")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
