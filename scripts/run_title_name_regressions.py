"""Reproducible, gated title/name Gemma panel and first regression package.

Heavy files are written only to a new analysis_outputs directory on research-host.
The accepted AT/alpha baselines remain NOT_ESTIMABLE when their inputs fail.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from title_name_regressions_core import (IntegrityError,NotEstimable,build_weight_panel,
    freeze_json,sha256_file,unique_keys)
from title_name_regressions_data import normalize_holdings,catalog_source_description
from title_name_regressions_estimation import (fit_panel_fe,fit_signal,signal_control_plan,
    h1_directional_test,h2_absolute_reaction_test)
from title_name_regressions_news import (input_index,read_labels,collapse_events,monthly_states,pair_identifier)
from title_name_regressions_reporting import (pre_estimation_audits,write_unavailable_nts_and_alpha,
    write_regression_tables)
from title_name_regressions_integrity import (verified_llm_binding,preparation_identity,
    validate_preparation,seal_preparation)
from title_name_regressions_crosswalk import load_approved_w2,maturity_interval_flags


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [clean(x) for x in value]
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(np.bool_,)):return bool(value)
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    if isinstance(value,(pd.Timestamp,pd.Period,Path,datetime)):return str(value)
    return value


def save(path,payload):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(clean(payload),ensure_ascii=True,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(temporary,path)


def parquet(path,frame):
    path=Path(path)
    temporary=path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary,index=False,compression="zstd")
    os.replace(temporary,path)


@contextmanager
def exclusive(path):
    with Path(path).open("a+b") as handle:
        if handle.tell()==0:handle.write(b"0");handle.flush()
        handle.seek(0)
        if os.name=="nt":
            import msvcrt
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            handle.seek(0)
            if os.name=="nt":msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(handle,fcntl.LOCK_UN)


def announce(output,stage,**items):
    payload=dict(stage=stage,updated_at_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**items)
    save(output/"checkpoint.json",payload)
    print(json.dumps(clean(payload),ensure_ascii=True),flush=True)


def source_stamp(path,*,digest=True):
    path=Path(path);s=path.stat()
    result=dict(path=str(path),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    if digest:result["sha256"]=sha256_file(path)
    return result


def prepare(spec,output,audit=None):
    previous=output/"preparation.json"
    state=None
    if previous.exists():
        state=json.loads(previous.read_text(encoding="utf-8"))
        if not state.get("preparation_provenance"):
            raise IntegrityError("Legacy preparation lacks verified transformation provenance; preserve it and use a fresh versioned output directory")
    if audit is None:raise IntegrityError("Final audited classifier identity is required before preparation")
    resources=spec["resources"]
    w2=load_approved_w2(output/"config/crosswalk_input.json")
    run=Path(resources["llm_run"])
    catalog_description=catalog_source_description(resources["catalog"])
    active_registry=next(row["path"] for row in catalog_description["sources"] if row["source_id"]=="refs.companies")
    manifest=json.loads((run/"input_manifest.json").read_text(encoding="utf-8-sig"))
    run_manifest=json.loads((run/"run_manifest.json").read_text(encoding="utf-8-sig"))
    binding=verified_llm_binding(spec,audit,manifest,run_manifest,input_manifest_path=run/"input_manifest.json",
        run_manifest_path=run/"run_manifest.json")
    paths=[resources["title_mapping"],resources["title_registry"],resources["llm_database"],
           run/"input_manifest.json",run/"run_manifest.json",run/"input.jsonl",
           Path(resources["holdings_root"])/"fund_holdings_long.csv.gz",
           Path(resources["holdings_root"])/"fund_holdings_wide.csv.gz",
           Path(resources["holdings_root"])/"holdings_security_name_isin_mapping.csv",
           active_registry,Path(resources["quote_root"])/"mapping.csv"]+w2["dependencies"]
    identity=preparation_identity(output/"config/spec.json",output/"config/audit.json",paths,
        catalog_description,binding,Path(__file__).parent)
    if state is not None:
        validate_preparation(output,state,identity)
        announce(output,"prepared_cache_content_and_code_verified",preparation_fingerprint=identity["fingerprint"])
        return state
    before=identity["sources"]
    save(output/"catalog_sources.json",catalog_description)
    save(output/"w2_input_metadata.json",w2["metadata"])
    announce(output,"indexing_frozen_news",expected_pairs=manifest["total"])
    # The frozen document adds explanatory prose after the IANA zone name.
    zone=spec["time_and_scope"]["timezone"].split(" with ",1)[0]
    index=input_index(run/"input.jsonl",manifest,zone)
    mapping=pd.read_parquet(resources["title_mapping"],columns=["article_id","company_id"])
    unique_keys(mapping,["article_id","company_id"],"title/name mapping")
    expected={pair_identifier(a,c) for a,c in zip(mapping.article_id,mapping.company_id)}
    if expected!=set(index.pair_id):raise IntegrityError("Mapping and frozen input pair sets differ")
    candidates=read_labels(resources["llm_database"],index,expected_logical_sha256=binding["logical_results_sha256"],
        expected_model=binding["model_identifier"],expected_prompt_version=binding["prompt_version"])
    parquet(output/"news_candidates.parquet",candidates)
    announce(output,"deduplicating_events",candidate_pairs=len(candidates))
    # Explicitly inferred rows are removed; absent provenance remains flagged unknown.
    date_pool=candidates.loc[~candidates.date_inferred.eq(True)].copy()
    representatives,clusters=collapse_events(date_pool,run/"input.jsonl")
    parquet(output/"event_cluster_audit.parquet",clusters)
    parquet(output/"event_representatives.parquet",representatives)
    announce(output,"normalizing_exact_holdings",event_representatives=len(representatives))
    inventory,reports,intervals,conflicts,holding_audit,market=normalize_holdings(resources["catalog"],spec,crosswalk=w2)
    parquet(output/"w2_edge_selection_audit.parquet",w2["selection_audit"])
    parquet(output/"scheduled_corporate_events.parquet",w2["scheduled_maturities"])
    w2["temporal_exclusion_audit"].to_csv(output/"temporal_exclusion_audit.csv",index=False)
    parquet(output/"holdings_inventory.parquet",inventory)
    parquet(output/"report_quality.parquet",reports)
    parquet(output/"all_true_intervals.parquet",intervals)
    conflicts.to_csv(output/"issuer_security_conflicts.csv",index=False)
    intervals[[x for x in intervals if "portfolio" in x or "unallocated" in x or "reconciliation" in x or x in ["interval_id","fund_id","start_date","end_date","preliminary_eligible"]]].to_csv(
        output/"portfolio_return_coverage.csv",index=False)
    eligible=intervals.loc[intervals.preliminary_eligible]
    dated=intervals.loc[intervals.calendar_eom_consecutive & intervals.month.le(spec["time_and_scope"]["baseline_closing_month_end_cap"])]
    last_month=str(eligible.month.max()) if len(eligible) else str(dated.month.max()) if len(dated) else spec["time_and_scope"]["baseline_closing_month_end_cap"]
    monthly,_=monthly_states(representatives,index.company_id.unique(),spec["time_and_scope"]["signal_history_start"],last_month)
    monthly["log_news_count"]=np.log1p(monthly.n_relevant)
    parquet(output/"company_month_states.parquet",monthly)
    state_audit=monthly.groupby(["month","news_status"],observed=True).agg(company_months=("company_id","size"),
        expected=("n_expected","sum"),relevant=("n_relevant","sum"),errors=("n_errors_or_unattempted","sum")).reset_index()
    state_audit.to_csv(output/"news_state_audit.csv",index=False)
    after_identity=preparation_identity(output/"config/spec.json",output/"config/audit.json",paths,
        catalog_source_description(resources["catalog"]),binding,Path(__file__).parent)
    if identity!=after_identity:raise IntegrityError("Preparation dependencies changed during execution")
    payload=dict(spec_sha256=sha256_file(output/"config/spec.json"),sources=before,
        input_sha256=manifest["input_sha256"],input_bytes=manifest["input_bytes"],
        llm_counts=candidates.pair_state.value_counts().to_dict(),
        pair_count=len(candidates),unique_articles=candidates.article_id.nunique(),companies=candidates.company_id.nunique(),
        inferred_pairs_excluded=int(candidates.date_inferred.eq(True).sum()),date_provenance_unknown=int(candidates.date_provenance_unknown.sum()),
        missing_date_pairs=int(candidates.event_date.isna().sum()),events=len(representatives),
        duplicate_members_removed=len(clusters)-len(representatives),conflicting_label_events=clusters.loc[clusters.conflicting_member_labels,"cluster_id"].nunique(),
        last_baseline_month=last_month,holdings=holding_audit,
        scheduled_maturity_fact_rows=len(w2["scheduled_maturities"]),
        R4_conflicting_report_rows=len(w2["temporal_exclusion_audit"]),
        verified_llm_identity=binding,w2_input=w2["metadata"],preparation_provenance=seal_preparation(output,identity),
        monthly_news_state_counts=monthly.news_status.value_counts().to_dict())
    save(previous,payload)
    announce(output,"prepared",events=len(representatives),eligible_intervals=len(eligible))
    return payload


def make_portfolio_panel(output):
    inventory=pd.read_parquet(output/"holdings_inventory.parquet")
    intervals=pd.read_parquet(output/"all_true_intervals.parquet")
    intervals=intervals.loc[intervals.preliminary_eligible].copy()
    if not len(intervals):return pd.DataFrame(),dict(status="NO_RECONCILED_CONSECUTIVE_EOM_INTERVALS")
    endpoint_ids=set(intervals.opening_report_id)|set(intervals.closing_report_id)
    inventory=inventory.loc[inventory.report_id.isin(endpoint_ids)].copy()
    returns=pd.DataFrame(dict(interval_id=pd.Series(dtype=str),security_id=pd.Series(dtype=str),
        total_return=pd.Series(dtype=float),return_coverage_complete=pd.Series(dtype=bool)))
    panel=build_weight_panel(inventory,intervals,returns)
    # Attributes are checked at their actual endpoint, not inferred from a name.
    attrs=[]
    for boundary in ["opening_report_id","closing_report_id"]:
        a=intervals[["interval_id",boundary]].merge(inventory[["report_id","security_id","target_security_candidate","residence_documented_by_disclosure"]],
             left_on=boundary,right_on="report_id",how="inner",validate="one_to_many")
        attrs.append(a[["interval_id","security_id","target_security_candidate","residence_documented_by_disclosure"]])
    combined=pd.concat(attrs,ignore_index=True)
    combined[["target_security_candidate","residence_documented_by_disclosure"]]=combined[["target_security_candidate","residence_documented_by_disclosure"]].fillna(False)
    flags=combined.groupby(["interval_id","security_id"],observed=True).agg(
        target_security_candidate=("target_security_candidate","all"),
        residence_documented_by_disclosure=("residence_documented_by_disclosure","all")).reset_index()
    panel=panel.merge(flags,on=["interval_id","security_id"],how="left",validate="one_to_one")
    all_rows=len(panel)
    coverage=panel.groupby("interval_id",observed=True).agg(
        fund_id=("fund_id","first"),start_date=("start_date","first"),end_date=("end_date","first"),
        union_positions=("security_id","size"),return_weight_coverage=("portfolio_return_weight_coverage","first"),
        at_gate_reason=("at_gate_reason","first"),source_target_positions=("target_security_candidate","sum"))
    coverage.reset_index().to_csv(output/"portfolio_return_coverage.csv",index=False)
    target=panel.target_security_candidate.eq(True)&panel.company_id.notna()&panel.residence_documented_by_disclosure.eq(True)
    events=output/"scheduled_corporate_events.parquet"
    scheduled=maturity_interval_flags(panel,pd.read_parquet(events)) if events.exists() else pd.Series(False,index=panel.index)
    panel["scheduled_maturity_in_interval"]=scheduled
    panel.loc[scheduled,["fund_id","interval_id","security_id","start_date","end_date","full_exit"]].to_csv(output/"scheduled_maturity_interval_audit.csv",index=False)
    target_before_action_screen=int(target.sum())
    target &= ~scheduled
    diagnostic=dict(all_union_inventory_rows=all_rows,static_unambiguous_target_rows=int(panel.target_security_candidate.eq(True).sum()),
        target_rows_with_documented_Russian_category=int(target.sum()),
        target_rows_before_scheduled_maturity_screen=target_before_action_screen,
        scheduled_maturity_target_intervals_excluded=target_before_action_screen-int(target.sum()),
        action_screen_scope="Available MATDATE scheduled facts; not actual payment history or proof of absence of other actions",
        rejected_residence_or_target_metadata=int((~target).sum()),
        total_return_AT_status="NOT_ESTIMABLE; verified instrument/cash/residual total returns unavailable",
        temporal_issuer_links="static provisional; no historical-identity assertion")
    panel=panel.loc[target].copy()
    panel["DW"]=panel.raw_delta_weight
    panel["opening_weight"]=panel.weight_open
    panel["opening_log_NAV"]=np.log(panel.opening_NAV.where(panel.opening_NAV.gt(0)))
    panel["fund_security_id"]=panel.fund_id.astype(str)+"|"+panel.security_id.astype(str)
    panel["asset_class"]=panel.asset_type
    panel["measurement_status"]="EXPLORATORY_DW_STATIC_LINK_PROVISIONAL_ACTIVE_STATUS"
    panel["historical_identity_verified"]=False
    panel["corporate_action_history_verified"]=False
    panel["residence_evidence"]="CONTEMPORANEOUS_EXPLICIT_RUSSIAN_DISCLOSURE_CATEGORY"
    unique_keys(panel,["fund_id","security_id","start_date","end_date"],"final union target panel")
    return panel,diagnostic


def make_signal_variants(output,spec,prepared,selected_controls):
    candidates=pd.read_parquet(output/"news_candidates.parquet")
    representatives=pd.read_parquet(output/"event_representatives.parquet")
    base=pd.read_parquet(output/"company_month_states.parquet")
    for name in spec["signal_model"]["candidate_lag_company_controls"]:base[name]=np.nan
    variants={"main":base}
    configurations={"no_event_collapse":dict(data=candidates),
                    "coverage95":dict(data=representatives,allow_partial=True,minimum_valid_fraction=.95),
                    "exclude_endpoint_days":dict(data=representatives,boundary_exclusion=True)}
    raw_date=representatives.copy()
    raw_date["event_date"]=pd.to_datetime(raw_date.published_at,errors="coerce",utc=True).dt.tz_localize(None).dt.normalize()
    configurations["stored_utc_calendar"]=dict(data=raw_date)
    for name,options in configurations.items():
        data=options.pop("data")
        data=data.loc[~data.date_inferred.eq(True)].copy()
        m,_=monthly_states(data,candidates.company_id.unique(),spec["time_and_scope"]["signal_history_start"],prepared["last_baseline_month"],**options)
        m["log_news_count"]=np.log1p(m.n_relevant)
        for control in selected_controls:m[control]=np.nan
        variants[name]=m
    diagnostics={};computed={}
    for name,monthly in variants.items():
        announce(output,"signal_fit",variant=name,company_months=len(monthly))
        try:
            fitted,details=fit_signal(monthly,selected_controls,
                minimum_company_months=spec["signal_model"]["minimum_company_usable_months"],
                minimum_companies_month=spec["signal_model"]["minimum_companies_per_month"])
            details["status"]="ESTIMATED_SIGNAL"
        except NotEstimable as error:
            fitted=monthly.copy();fitted["S"]=np.nan
            details=dict(status="NOT_ESTIMABLE",reason=str(error),n_final=0)
        computed[name]=fitted;diagnostics[name]=details
    try:
        no_company,details=fit_signal(base,(),
            minimum_company_months=spec["signal_model"]["minimum_company_usable_months"],
            minimum_companies_month=spec["signal_model"]["minimum_companies_per_month"])
        details["status"]="ESTIMATED_SIGNAL"
    except NotEstimable as error:
        no_company=base.copy();no_company["S"]=np.nan
        details=dict(status="NOT_ESTIMABLE",reason=str(error),n_final=0)
    computed["no_company_controls"]=no_company;diagnostics["no_company_controls"]=details
    parquet(output/"company_month_signal.parquet",computed["main"])
    for name,monthly in computed.items():
        if name!="main":parquet(output/("company_month_signal_"+name+".parquet"),monthly)
    save(output/"signal_variant_status.json",diagnostics)
    return computed,diagnostics


def independent_small_check(frame,regressors):
    """Dense dummy OLS plus independently grouped cluster scores on a small cut."""
    eligible=frame.dropna(subset=["DW"]+regressors)
    funds=eligible.fund_id.value_counts().head(6).index
    companies=eligible.loc[eligible.fund_id.isin(funds)].company_id.value_counts().head(6).index
    cut=eligible.loc[eligible.fund_id.isin(funds)&eligible.company_id.isin(companies)].sort_values(["month","fund_id","security_id"]).tail(1500)
    result=fit_panel_fe(cut,"DW",regressors,allow_absorbed=regressors[2:],allow_collinear=regressors[2:])
    d=result.sample
    dummy=pd.get_dummies(d[["fund_security_id","month"]].astype(str),drop_first=True,dtype=float)
    x=np.column_stack([np.ones(len(d)),d[result.names].to_numpy(float),dummy.to_numpy(float)])
    y=d.DW.to_numpy(float)
    inverse=np.linalg.pinv(x)
    beta=inverse@y
    error=y-x@beta
    bread=inverse@inverse.T
    rank=np.linalg.matrix_rank(x)
    score=x*error[:,None]
    covariance=np.zeros((x.shape[1],x.shape[1]))
    for values,sign in [(d.fund_id,1),(d.company_id,1),(pd.MultiIndex.from_frame(d[["fund_id","company_id"]]),-1)]:
        codes,levels=pd.factorize(values,sort=False)
        sums=np.zeros((len(levels),x.shape[1]))
        for index in range(len(d)):sums[codes[index]]+=score[index]
        meat=sums.T@sums
        covariance+=sign*(len(levels)/(len(levels)-1))*((len(d)-1)/(len(d)-rank))*(bread@meat@bread)
    where=np.arange(1,1+len(result.names))
    discrepancy=float(np.max(np.abs(beta[where]-result.beta)))
    cov_difference=float(np.max(np.abs(covariance[np.ix_(where,where)]-result.covariance)))
    if discrepancy>1e-7 or cov_difference>1e-8:
        raise IntegrityError("Independent small OLS/cluster check disagrees with absorbed model")
    return dict(status="PASS",nobs=len(d),funds=d.fund_id.nunique(),companies=d.company_id.nunique(),
                max_absolute_beta_difference=discrepancy,max_absolute_covariance_difference=cov_difference,
                independent_design="explicit dummy matrix/pseudoinverse and looped fund/company/intersection scores")


def estimate(spec,audit,output,prepared,spec_path,audit_path):
    if not audit.get("audit_execution_complete") or audit.get("status")!="AUDIT_COMPLETE" or audit.get("qa_refinement_in_progress"):
        announce(output,"WAITING_FOR_FINAL_AUDIT",audit_status=audit.get("status"),qa_refinement=audit.get("qa_refinement_in_progress"))
        return None
    if audit["llm"].get("integrity_failures") or audit["llm"].get("status")!="READY":
        raise IntegrityError("Independent LLM key-integrity gate did not pass")
    review_gate_path=output/"config/crosswalk_review_gate.json"
    if not review_gate_path.exists():
        announce(output,"WAITING_FOR_R4_CROSSWALK_REVIEW",w2_version=prepared["w2_input"]["version"])
        return None
    review_gate=json.loads(review_gate_path.read_text(encoding="utf-8-sig"))
    if review_gate.get("allow_exploratory_DW") is not True or review_gate.get("scope")!="EXPLORATORY_DW_ONLY":
        announce(output,"R4_CROSSWALK_SCOPE_NOT_APPROVED",gate=review_gate)
        return None
    if review_gate["w2_version"]!=prepared["w2_input"]["version"] or review_gate["spec_sha256"]!=sha256_file(spec_path):
        raise IntegrityError("Crosswalk review refers to a different W2 version or R1 specification")
    review_path=Path(review_gate["review"]["path"])
    if sha256_file(review_path)!=review_gate["review"]["sha256"]:
        raise IntegrityError("Pinned independent crosswalk review changed")
    observed_gates={k:audit["market"].get(k) for k in ["total_return_ready","cash_residual_ready","alpha_ready"]}
    if any(v is not False for v in observed_gates.values()):
        raise IntegrityError("Audit return gates changed; this first input-gated runner must not silently ignore new verified inputs")
    announce(output,"pre_estimation_coverage_audits")
    exclusion_audit=pre_estimation_audits(output,spec)
    # Fund returns are inspected only for coverage; they are not a validated alpha input.
    prepared["sources"]=list(prepared["sources"])+[source_stamp(Path(spec["resources"]["holdings_root"])/"fund_returns_append_available.csv"),
        source_stamp(review_gate_path),source_stamp(review_path)]
    monthly=pd.read_parquet(output/"company_month_states.parquet")
    for name in spec["signal_model"]["candidate_lag_company_controls"]:monthly[name]=np.nan
    plan=signal_control_plan(monthly,spec["signal_model"]["candidate_lag_company_controls"])
    plan["unavailable_input_reason"]="Registered CLOSE is unverified for splits; current shares do not establish historical cap. No split-adjusted price/market-cap source was approved by the audit."
    # Gates and the complete selected nuisance list are frozen before any actual fitted slope.
    code_paths=[Path(__file__)]+sorted(Path(__file__).parent.glob("title_name_regressions_*.py"))
    frozen=dict(spec_id=spec["spec_id"],spec_sha256=sha256_file(spec_path),audit_sha256=sha256_file(audit_path),
        code_hashes={p.name:sha256_file(p) for p in code_paths},source_hashes=prepared["sources"],
        preparation_fingerprint=prepared["preparation_provenance"]["identity"]["fingerprint"],
        preparation_manifest_sha256=sha256_file(output/"preparation.json"),
        verified_llm_identity=prepared["verified_llm_identity"],
        w2_input=prepared["w2_input"],
        crosswalk_independent_review=review_gate,
        signal_control_plan=plan,
        audit_observed_market_gates=observed_gates,
        software_versions={name:importlib.metadata.version(name) for name in ["pandas","numpy","scipy","statsmodels","pyarrow","duckdb"]},
        python=platform.python_version(),
        gates=dict(pair_keys="PASS",dates="PROVISIONAL_UTC_AND_UNKNOWN_PROVENANCE",total_returns=False,
            full_H12_controls=False,baseline_AT=False,baseline_H3_alpha=False),
        implementation_rules=dict(missing_text_duplicate_policy="singleton; absence is not duplicate evidence",
            exploratory_universe="unique static issuer link, documented Russian asset category, reconciled exact EOM reports; active status unknown is provisional",
            security_weights="reported instrument value / gross assets, never mixed share_pct denominators",
            unpriced_residual="preserved, never zero return; formal AT disabled",
            statistical_interpretation="exploratory DW is valuation-contaminated and does not test baseline H1/H2"))
    freeze_json(output/"run_manifest.json",clean(frozen))
    announce(output,"specification_and_gates_frozen",spec_sha256=frozen["spec_sha256"],audit_sha256=frozen["audit_sha256"])
    save(output/"signal_control_availability.json",plan)
    signals,signal_diagnostics=make_signal_variants(output,spec,prepared,plan["selected"])
    panel,panel_diagnostics=make_portfolio_panel(output)
    models=[];coefficient_rows=[];sample_flow=[];independent=None
    controls=spec["controls"]["reduced_preliminary_H12"]
    if len(panel):
        main=signals["main"][["company_id","month","S","Tone","lag_log_news_count","n_relevant","news_status","valid_fraction"]]
        panel=panel.merge(main,on=["company_id","month"],how="left",validate="many_to_one")
        parquet(output/"fund_security_panel.parquet",panel)
        variants=[]
        for name,monthly in signals.items():
            value=monthly[["company_id","month","S","lag_log_news_count","valid_fraction"]].rename(
                columns={"S":"signal_variant","lag_log_news_count":"variant_lag_log_news_count","valid_fraction":"variant_valid_fraction"})
            variant=panel.drop(columns=["signal_variant"],errors="ignore").merge(value,on=["company_id","month"],how="left",validate="many_to_one")
            variant["main_lag_log_news_count"]=variant.lag_log_news_count
            variant["lag_log_news_count"]=variant.variant_lag_log_news_count
            if name=="coverage95":variant["current_error_share"]=1-variant.variant_valid_fraction
            variants.append(("EXPLORATORY_DW_S_"+name,variant,"signal_variant",controls+(["current_error_share"] if name=="coverage95" else [])))
            if name not in ["main","no_company_controls"]:
                common=variant.S.notna()&variant.signal_variant.notna()&variant[controls+["main_lag_log_news_count"]].notna().all(axis=1)
                alt=variant.loc[common].copy()
                selected_common=controls+(["current_error_share"] if name=="coverage95" else [])
                variants.append(("EXPLORATORY_DW_S_"+name+"_COMMON",alt,"signal_variant",selected_common))
                main_common=alt.copy();main_common["lag_log_news_count"]=main_common.main_lag_log_news_count
                variants.append(("EXPLORATORY_DW_MAIN_COMMON_"+name,main_common,"S",selected_common))
        variants.append(("EXPLORATORY_DW_RAW_TONE",panel,"Tone",controls))
        common_tone=panel.loc[panel.S.notna()&panel.Tone.notna()].copy()
        variants.append(("EXPLORATORY_DW_RAW_TONE_COMMON",common_tone,"Tone",controls))
        variants.append(("EXPLORATORY_DW_MAIN_COMMON_RAW_TONE",common_tone,"S",controls))
        for shift,label in [(-1,"LAG"),(1,"LEAD_PLACEBO")]:
            m=signals["main"][["company_id","month","S"]].copy()
            m["month"]=(pd.PeriodIndex(m.month,freq="M")-shift).astype(str)
            m=m.rename(columns={"S":"shifted_signal"})
            v=panel.merge(m,on=["company_id","month"],how="left",validate="many_to_one")
            common=v.S.notna()&v.shifted_signal.notna()
            variants.append(("EXPLORATORY_DW_"+label+"_COMMON",v.loc[common],"shifted_signal",controls))
            variants.append(("EXPLORATORY_DW_MAIN_COMMON_"+label,v.loc[common],"S",controls))
        for asset in ["equity","bond"]:
            variants.append(("EXPLORATORY_DW_"+asset.upper(),panel.loc[panel.asset_class.eq(asset)],"S",controls))
        variants += [("EXPLORATORY_DW_PRE_2022_02",panel.loc[panel.month.lt("2022-02")],"S",controls),
                     ("EXPLORATORY_DW_POST_2022_02",panel.loc[panel.month.gt("2022-02")],"S",controls),
                     ("EXPLORATORY_DW_TRANSITION_2022_02",panel.loc[panel.month.eq("2022-02")],"S",controls),
                     ("EXPLORATORY_DW_MAIN_THREE_WAY_CLUSTERS",panel,"S",controls)]
        # Percentiles are calculated on each control's unique natural unit, not fund-expanded news rows.
        winsor=panel.copy();winsor_cutpoints={}
        units={"opening_weight":["fund_id","security_id","start_date"],
               "lag_log_news_count":["company_id","month"],"opening_log_NAV":["fund_id","start_date"],
               "opening_HHI":["fund_id","start_date"]}
        for control in controls:
            natural=panel[units[control]+[control]].drop_duplicates()
            if natural.duplicated(units[control]).any():raise IntegrityError("Control differs at its declared natural unit")
            q=natural[control].quantile([.01,.99])
            winsor_cutpoints[control]=dict(p01=q.loc[.01],p99=q.loc[.99],natural_rows=len(natural))
            winsor[control]=winsor[control].clip(q.loc[.01],q.loc[.99])
        save(output/"winsor_control_cutpoints.json",winsor_cutpoints)
        variants.append(("EXPLORATORY_DW_WINSOR_CONTROLS_1_99",winsor,"S",controls))
        for label,frame,signal,selected in variants:
            announce(output,"estimating_exploratory_DW",model=label,candidate_rows=len(frame))
            d=frame.copy();d["S_positive"]=d[signal].clip(lower=0);d["S_negative"]=(-d[signal]).clip(lower=0)
            regressors=["S_positive","S_negative"]+list(selected)
            try:
                result=fit_panel_fe(d,"DW",regressors,allow_absorbed=selected,allow_collinear=selected,
                    extra_cluster="month" if label.endswith("THREE_WAY_CLUSTERS") else None)
                details=dict(model=label,status="ESTIMATED_EXPLORATORY_NOT_BASELINE_H1_H2",outcome="DW",signal=signal,
                    diagnostics=result.diagnostics,H1_directional_descriptive=h1_directional_test(result),
                    H2_descriptive=h2_absolute_reaction_test(result),fragile_clusters=result.diagnostics["inference_df"]+1<30)
                for row in result.table():
                    row.update(model=label,outcome="DW",effect_bps=row["beta"]*10000,ci95_low_bps=row["ci95_low"]*10000,
                        ci95_high_bps=row["ci95_high"]*10000,N=result.diagnostics["nobs"],funds=result.diagnostics["funds"],
                        companies=result.diagnostics["companies"],within_r2=result.diagnostics["within_r2"],
                        securities=result.diagnostics["securities"],months=result.diagnostics["time_effects"],
                        fund_clusters=result.diagnostics["fund_clusters"],company_clusters=result.diagnostics["company_clusters"],
                        period_start=result.diagnostics["min_period"],period_end=result.diagnostics["max_period"])
                    coefficient_rows.append(row)
                if independent is None and label=="EXPLORATORY_DW_S_main":
                    try:independent=independent_small_check(d,regressors)
                    except NotEstimable as error:independent=dict(status="NOT_ESTIMABLE_SMALL_CUT",reason=str(error))
                save(output/(label+"_diagnostics.json"),details)
            except NotEstimable as error:
                details=dict(model=label,status="NOT_ESTIMABLE",reason=str(error),candidate_rows=len(d))
            models.append(details)
        sample_flow=[dict(stage="union_target_rows_after_report_security_scope",n=len(panel)),
                     dict(stage="defined_main_S",n=int(panel.S.notna().sum())),
                     dict(stage="full_entries_before_signal_complete_case",n=int(panel.full_entry.sum())),
                     dict(stage="full_exits_before_signal_complete_case",n=int(panel.full_exit.sum()))]
    else:
        parquet(output/"fund_security_panel.parquet",panel)
        models.append(dict(model="EXPLORATORY_DW_S_main",status="NOT_ESTIMABLE",reason="No reconciled exact EOM target panel"))
    coefficients=pd.DataFrame(coefficient_rows) if coefficient_rows else pd.DataFrame(columns=[
        "model","term","beta","se","ci95_low","ci95_high","p","N","funds","companies","effect_bps"])
    coefficients.to_csv(output/"regression_results.csv",index=False)
    pd.DataFrame(prepared.get("llm_counts",{}).items(),columns=["pair_state","n"]).to_csv(output/"llm_state_counts.csv",index=False)
    flow=[dict(stage="mapping_pairs",n=prepared["pair_count"]),dict(stage="event_representatives",n=prepared["events"]),
          dict(stage="deduplicated_members_removed",n=prepared["duplicate_members_removed"]),
          dict(stage="exact_eligible_intervals",n=prepared["holdings"]["preliminary_eligible_intervals"])]+sample_flow
    pd.DataFrame(flow).to_csv(output/"sample_flow.csv",index=False)
    gates=dict(H1=dict(status="NOT_ESTIMABLE_BASELINE",reason="Verified full-portfolio total returns/corporate actions/cash and full controls are missing"),
        H2=dict(status="NOT_ESTIMABLE_BASELINE",reason="No baseline total-return AT; DW asymmetry is exploratory only"),
        H3=[dict(horizon=h,status="NOT_ESTIMABLE",reason="AT/NTS gate plus verified 36-month past fund/factor/RF history fails; registered returns only Dec2024-Nov2025") for h in [1,2,3]],
        AT_PRICE=dict(status="NOT_ESTIMABLE",reason="All-position split-adjustment/identifier-change and cash price coverage not verified"),
        DAR=dict(status="NOT_ESTIMABLE",reason="Future target total returns and suitable verified benchmarks unavailable"))
    save(output/"gate_status.json",gates)
    alpha_coverage=write_unavailable_nts_and_alpha(output,panel,spec)
    write_regression_tables(output,coefficient_rows,models,gates)
    save(output/"diagnostics.json",dict(signal=signal_diagnostics,panel=panel_diagnostics,models=models,
        independent_small_calculation=independent,exclusions=exclusion_audit,alpha_coverage=alpha_coverage,
        blocked_sensitivities={"return_coverage_and_corporate_actions":"No verified AT", "high_flow_exclusion":"Distribution-adjusted flow unavailable", "H3_past_only_signal":"No AT/NTS or alpha baseline"}))
    n_estimated=sum(x["status"].startswith("ESTIMATED") for x in models)
    summary=dict(status="PRELIMINARY_PACKAGE_COMPLETED" if n_estimated else "INPUT_GATED_PACKAGE_COMPLETED_NO_ESTIMATES",spec_id=spec["spec_id"],spec_sha256=frozen["spec_sha256"],
        audit_sha256=frozen["audit_sha256"],code_hashes=frozen["code_hashes"],source_hashes=frozen["source_hashes"],
        preparation_fingerprint=frozen["preparation_fingerprint"],
        preparation_manifest_sha256=frozen["preparation_manifest_sha256"],
        verified_llm_identity=frozen["verified_llm_identity"],
        w2_input=frozen["w2_input"],
        crosswalk_independent_review=frozen["crosswalk_independent_review"],
        hypotheses=gates,news_counts=prepared,signal=signal_diagnostics,panel=panel_diagnostics,
        models=models,coefficients=coefficient_rows,sample_flow=flow,independent_small_calculation=independent,
        exclusions=exclusion_audit,alpha_history_coverage=alpha_coverage,software_versions=frozen["software_versions"],
        reproducible_command=f'"./local/research-data-service/venv/Scripts/python.exe" -u "{output}/code/run_title_name_regressions.py" --spec "{output}/config/spec.json" --audit "{output}/config/audit.json" --crosswalk-input "{output}/config/crosswalk_input.json" --output "{output}" --stage all',
        output_root=str(output),
        limitations=["Exploratory DW is valuation-contaminated, not observed active rebalancing", "Static issuer bridge lacks historical validity",
            "Unknown active mandates remain provisional; explicit passive mandates excluded", "Unknown date provenance and UTC ingestion convention",
            "Generated-regressor and LLM-label uncertainty are not resolved by conditional cluster SEs", "No completed independent human validation of this sample",
            "No claim of causality or a tradable publicly disclosed holdings strategy"],
        generated_at_utc=datetime.now(timezone.utc).isoformat())
    save(output/"summary.json",summary)
    announce(output,"completed",models=len(models),estimated=sum(x["status"].startswith("ESTIMATED") for x in models))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec",type=Path,required=True);p.add_argument("--audit",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--stage",choices=["prepare","estimate","all"],default="all")
    p.add_argument("--crosswalk-input",type=Path,required=True)
    args=p.parse_args()
    spec=json.loads(args.spec.read_text(encoding="utf-8-sig"))
    if spec["status"]!="FROZEN_RULES_WITH_INPUT_GATES":raise IntegrityError("Unrecognized unfrozen specification")
    output=args.output.resolve()
    immutable=Path(spec["resources"]["immutable_root"]).resolve()
    if output==immutable or immutable in output.parents or "GoogleDrive" in str(output):
        raise IntegrityError("Heavy regression outputs must be outside immutable inputs and Drive")
    if os.name!="nt":raise IntegrityError("Actual data execution belongs on research-host; use synthetic tests on the Mac")
    # A completed directory is a historical package, never a destination for new code.
    if (output/"summary.json").exists():
        raise IntegrityError("Completed package is immutable; use a new versioned output directory for any rerun")
    if (output/"run_manifest.json").exists():
        old=json.loads((output/"run_manifest.json").read_text(encoding="utf-8"))
        current=[Path(__file__)]+sorted(Path(__file__).parent.glob("title_name_regressions_*.py"))
        if old.get("code_hashes")!={p.name:sha256_file(p) for p in current} or old.get("spec_sha256")!=sha256_file(args.spec) or old.get("audit_sha256")!=sha256_file(args.audit):
            raise IntegrityError("Unfinished frozen package belongs to different code/specification/audit; use a new version")
    output.mkdir(parents=True,exist_ok=True)
    awake=False
    with exclusive(output/".regression_writer.lock"):
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState.argtypes=[ctypes.c_uint32]
            ctypes.windll.kernel32.SetThreadExecutionState.restype=ctypes.c_uint32
            awake=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000003))
            # Deployment retains a byte-identical source snapshot at this path.
            if args.spec.resolve()!=output/"config/spec.json":
                raise IntegrityError("Use the immutable per-run config/spec.json deployment copy")
            if args.audit.resolve()!=output/"config/audit.json":
                raise IntegrityError("Use the exact per-run config/audit.json bound to preparation provenance")
            if args.crosswalk_input.resolve()!=output/"config/crosswalk_input.json":
                raise IntegrityError("Use the pinned per-run config/crosswalk_input.json")
            audit=json.loads(args.audit.read_text(encoding="utf-8-sig"))
            prepared=prepare(spec,output,audit)
            if args.stage in ["estimate","all"]:
                estimate(spec,audit,output,prepared,args.spec,args.audit)
        except Exception as error:
            announce(output,"failed",error=repr(error))
            raise
        finally:
            if awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=="__main__":main()
