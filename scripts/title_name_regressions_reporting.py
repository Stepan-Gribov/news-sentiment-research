"""Pre-estimation coverage, exclusion audits, and compact regression tables."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from title_name_regressions_core import aggregate_news_windows


def primary_reasons(frame,rules,accepted):
    out=pd.Series(accepted,index=frame.index,dtype=object)
    available=pd.Series(True,index=frame.index)
    for reason,flag in rules:
        selected=available&flag.fillna(False)
        out.loc[selected]=reason
        available&=~selected
    return out


def pre_estimation_audits(output,spec):
    """No coefficient fitting; every expected item has exactly one main reason."""
    output=Path(output)
    candidates=pd.read_parquet(output/"news_candidates.parquet")
    clusters=pd.read_parquet(output/"event_cluster_audit.parquet",columns=["pair_id","is_representative"])
    candidates=candidates.merge(clusters,on="pair_id",how="left",validate="one_to_one")
    start=pd.Timestamp(spec["time_and_scope"]["signal_history_start"])
    prepared=json.loads((output/"preparation.json").read_text(encoding="utf-8"))
    end=pd.Period(prepared["last_baseline_month"],freq="M").end_time.normalize()
    rules=[("explicit_inferred_date",candidates.date_inferred.eq(True)),
        ("missing_date",candidates.event_date.isna()),
        ("outside_signal_history",~candidates.event_date.between(start,end)),
        ("event_duplicate",candidates.is_representative.eq(False)),
        ("input_error",candidates.pair_state.eq("input_error")),
        ("classification_error",candidates.pair_state.eq("classification_error")),
        ("irrelevant",candidates.pair_state.eq("irrelevant"))]
    candidates["primary_exclusion"]=primary_reasons(candidates,rules,"relevant_event_with_date")
    columns=["pair_id","article_id","company_id","source","event_date","pair_state",
             "date_provenance_unknown","is_representative","primary_exclusion"]
    candidates[columns].to_parquet(output/"news_pair_exclusion_audit.parquet",index=False,compression="zstd")
    candidates["year"]=pd.to_datetime(candidates.event_date).dt.year.astype("Int64")
    candidates["label_error"]=~candidates.valid
    quality=candidates.groupby(["company_id","source","year"],dropna=False,observed=True).agg(
        expected_pairs=("pair_id","size"),label_errors=("label_error","sum"),valid=("valid","sum"),
        unknown_date_provenance=("date_provenance_unknown","sum")).reset_index()
    quality["error_fraction"]=quality.label_errors/quality.expected_pairs
    quality.to_csv(output/"news_quality_by_company_source_year.csv",index=False)

    reports=pd.read_parquet(output/"report_quality.parquet")
    intervals=pd.read_parquet(output/"all_true_intervals.parquet")
    tolerance=spec["holdings_and_weights"]["reconciliation_tolerance_fraction"]
    rounding=spec["holdings_and_weights"]["rounding_gap_max_fraction"]
    reports["primary_exclusion"]=primary_reasons(reports,[
        ("missing_date",reports.report_date.isna()),("ambiguous_fund_identity",~reports.fund_identity_verified),
        ("gross_assets_missing_or_nonpositive",~reports.gross_thousand.gt(0)),
        ("NAV_missing_or_nonpositive",~reports.NAV.gt(0)),
        ("liabilities_missing",reports.obligations_thousand.isna()),
        ("NAV_reconciliation_failed",~reports.nav_reconciliation_fraction.le(tolerance)),
        ("missing_asset_amount",reports.any_missing_amount.fillna(True)),
        ("negative_asset_amount",reports.any_negative.fillna(True)),
        ("individual_weight_above_one",reports.any_weight_gt1.fillna(True)),
        ("category_conflict",reports.category_conflict),
        ("unallocated_amount_above_tolerance",~reports.unallocated_fraction.abs().le(rounding))],"reconciled_report")
    reports[["report_id","fund_id","report_date","primary_exclusion"]].to_csv(output/"report_exclusion_audit.csv",index=False)
    if not reports.primary_exclusion.eq("reconciled_report").eq(reports.full_portfolio).all():
        raise ValueError("Report exclusion flow does not reconcile with its actual eligibility gate")
    intervals["primary_exclusion"]=primary_reasons(intervals,[
        ("not_consecutive_actual_calendar_EOM",~intervals.calendar_eom_consecutive),
        ("outside_research_period",~intervals.month.between(spec["time_and_scope"]["baseline_closing_month_start"],spec["time_and_scope"]["baseline_closing_month_end_cap"])),
        ("opening_report_quality_failed",~intervals.opening_full_portfolio.astype(bool)),
        ("closing_report_quality_failed",~intervals.closing_full_portfolio.astype(bool)),
        ("explicit_passive_mandate",intervals.opening_passive_explicit|intervals.closing_passive_explicit)],"preliminary_eligible_interval")
    intervals[["interval_id","fund_id","start_date","end_date","primary_exclusion"]].to_csv(output/"interval_exclusion_audit.csv",index=False)
    if not intervals.primary_exclusion.eq("preliminary_eligible_interval").eq(intervals.preliminary_eligible).all():
        raise ValueError("Interval exclusion flow does not reconcile with its actual eligibility gate")
    inventory=pd.read_parquet(output/"holdings_inventory.parquet")
    targets=inventory.loc[inventory.target_security_candidate.eq(True)&inventory.company_id.notna(),["report_id","company_id"]].drop_duplicates()
    windows=[]
    for side in ["opening_report_id","closing_report_id"]:
        windows.append(intervals[["interval_id",side]].merge(targets,left_on=side,right_on="report_id",how="inner",validate="one_to_many")[["interval_id","company_id"]])
    windows=pd.concat(windows,ignore_index=True).drop_duplicates()
    if len(windows):
        windows=windows.merge(intervals[["interval_id","start_date","end_date","calendar_eom_consecutive"]],on="interval_id",how="left",validate="many_to_one")
        windows=windows.rename(columns={"start_date":"news_start","end_date":"news_end"})
        representatives=pd.read_parquet(output/"event_representatives.parquet")
        windows,_=aggregate_news_windows(windows,representatives)
        windows["S_assigned"]=False  # Non-EOM exposures never borrow calendar-month S.
    windows.to_parquet(output/"true_interval_news_counts.parquet",index=False,compression="zstd")
    rows=[]
    for unit,frame in [("news_pairs",candidates),("fund_reports",reports),("actual_intervals",intervals)]:
        for reason,n in frame.primary_exclusion.value_counts(dropna=False).items():
            rows.append(dict(unit=unit,primary_reason=reason,n=int(n),denominator=len(frame)))
    pd.DataFrame(rows).to_csv(output/"primary_exclusion_flow.csv",index=False)
    return dict(primary_exclusion_counts=rows,true_interval_company_windows=len(windows),
                counts_are_mutually_exclusive_within_unit=True,
                supplementary_windows="actual (d0,d1] counts only, no S/model attached to non-EOM windows")


def write_unavailable_nts_and_alpha(output,panel,spec):
    output=Path(output)
    columns=["fund_id","interval_id","month"]
    intervals=pd.read_parquet(output/"all_true_intervals.parquet")
    nts=intervals.loc[intervals.preliminary_eligible,columns].drop_duplicates().copy()
    nts["NTS"]=np.nan
    nts["nts_status"]="NOT_ESTIMABLE_NO_VERIFIED_WHOLE_PORTFOLIO_AT"
    nts.to_parquet(output/"fund_month_NTS.parquet",index=False,compression="zstd")
    path=Path(spec["resources"]["holdings_root"])/"fund_returns_append_available.csv"
    source=pd.read_csv(path,dtype={"fund_key":str})
    source["date"]=pd.to_datetime(source.date,errors="coerce")
    source["valid_return"]=pd.to_numeric(source.fund_ret,errors="coerce").notna()
    coverage=source.groupby("fund_key",dropna=False,observed=True).agg(
        available_rows=("date","size"),first_date=("date","min"),last_date=("date","max"),
        nonmissing_simple_fund_returns=("valid_return","sum")).reset_index()
    coverage["fund_id_link_status"]="NOT_VERIFIED"
    coverage["historical_factor_and_RF_coverage"]="NOT_VERIFIED"
    coverage["minimum_prior_complete_months_required"]=36
    coverage["eligible_alpha_h1"]=False;coverage["eligible_alpha_h2"]=False;coverage["eligible_alpha_h3"]=False
    coverage.to_csv(output/"alpha_history_coverage.csv",index=False)
    return dict(nts_fund_intervals=len(nts),defined_NTS=0,registered_fund_return_keys=len(coverage),
                available_rows=len(source),first_registered_return_date=str(source.date.min().date()),
                last_registered_return_date=str(source.date.max().date()),alpha_horizons={"1":0,"2":0,"3":0},
                note="No return or missing label was replaced with zero; no factor alpha was synthesized")


def write_regression_tables(output,coefficients,models,gates):
    text=["# First title/name regression package","",
          "Exploratory DW only: changes in reported gross-asset weights include valuation effects. They are not active-trading AT and are not publication-baseline H1/H2 tests.","",
          "| Baseline | Status | Reason |","|---|---|---|"]
    for name in ["H1","H2"]:text.append(f"| {name} | {gates[name]['status']} | {gates[name]['reason']} |")
    for row in gates["H3"]:text.append(f"| H3 h={row['horizon']} | {row['status']} | {row['reason']} |")
    text += ["","## Exploratory coefficients","",
             "SEs condition on the generated signal and fixed LLM labels. Two-way fund/company clusters unless a model is explicitly three-way; no significance stars.","",
             "| Model | Term | Effect bps | 95% CI bps | SE bps | p | N | Funds | Companies | Within R² |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    def number(x):return f"{x:.6g}" if x is not None and np.isfinite(x) else "NA"
    for row in coefficients:
        if row["term"] not in ["S_positive","S_negative"]:continue
        text.append("| "+" | ".join([row["model"],row["term"],number(row["effect_bps"]),
            f"[{number(row['ci95_low_bps'])}, {number(row['ci95_high_bps'])}]",number(row["se"]*10000),
            number(row["p"]),str(row["N"]),str(row["funds"]),str(row["companies"]),number(row["within_r2"])])+" |")
    text += ["","## Model gates and descriptive H2 restrictions",""]
    for model in models:
        if model["status"]=="NOT_ESTIMABLE":text.append(f"- {model['model']}: NOT_ESTIMABLE — {model['reason']}")
        else:
            h2=model["H2_descriptive"]
            text.append(f"- {model['model']}: {model['status']}; fragile clusters={model['fragile_clusters']}; H2 p_sum={number(h2.get('sum_restriction',{}).get('p'))}, p_diff={number(h2.get('difference_restriction',{}).get('p'))}, p_abs={number(h2.get('p_abs'))}.")
    text += ["","Static issuer links and unknown active mandates are provisional. Date provenance is unknown in the canonical archive. No causal, real-time tradability, LLM accuracy or scientific-hypothesis-completion claim is made.",""]
    (Path(output)/"regression_tables.md").write_text("\n".join(text),encoding="utf-8")
