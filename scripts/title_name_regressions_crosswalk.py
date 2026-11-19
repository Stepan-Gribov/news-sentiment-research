"""Strict W2 input contract; current identifier facts are not historical validity."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from title_name_regressions_core import IntegrityError,require_columns,unique_keys,sha256_file
from title_name_regressions_data import valid_isin


def registration_key(value):
    if value is None or (not isinstance(value,(str,list,dict)) and pd.isna(value)):return ""
    value=unicodedata.normalize("NFKC",str(value)).upper().strip()
    value=re.sub(r"[‐‑‒–—−]","-",value)
    value=re.sub(r"\s+","",value)
    return "" if value in {"","NAN","NONE","NULL","NAT","<NA>"} else value


def structured_list(value,label):
    if isinstance(value,list):return value
    if value is None or (not isinstance(value,str) and pd.isna(value)):return []
    if isinstance(value,str):
        try:result=json.loads(value)
        except (ValueError,TypeError) as error:raise IntegrityError("Invalid W2 JSON field: "+label) from error
        if isinstance(result,list):return result
    raise IntegrityError("W2 structured field is not a list: "+label)


def select_exact_edges(crosswalk,canonical_issuers,*,version,blocked_codes=(),require_unique_company=True):
    required=["registration_code_normalized","isin","mapping_status","company_join_status",
        "company_ids_static","source_evidence","security_types","historical_baseline_eligible",
        "identity_valid_from","identity_valid_to"]
    require_columns(crosswalk,required,"W2 crosswalk")
    require_columns(canonical_issuers,["isin","company_id"],"canonical issuer edges")
    frame=crosswalk.copy()
    keys=frame.registration_code_normalized.map(registration_key)
    if keys.eq("").any() or not keys.eq(frame.registration_code_normalized).all():
        raise IntegrityError("W2 keys are not in the declared typography-only normal form")
    unique_keys(frame,["registration_code_normalized"],"W2 registration-code edges")
    if frame.historical_baseline_eligible.fillna(False).astype(bool).any() or frame[["identity_valid_from","identity_valid_to"]].notna().any(axis=None):
        raise IntegrityError("This reviewed W2 interface cannot silently accept a new historical-validity claim")
    bridge=canonical_issuers.dropna().drop_duplicates(["isin","company_id"])
    issuers=bridge.groupby("isin",observed=True).company_id.agg(lambda x:sorted(set(x))).to_dict()
    blocked={registration_key(x) for x in blocked_codes}
    accepted=[];audit=[]
    for row in frame.to_dict("records"):
        code=row["registration_code_normalized"];isin=row["isin"]
        companies=structured_list(row["company_ids_static"],"company_ids_static")
        evidence=structured_list(row["source_evidence"],"source_evidence")
        types=structured_list(row["security_types"],"security_types")
        reason=None;company_reason=None
        if row["mapping_status"]!="EXACT_CURRENT_CONFIRMED":reason="W2_IDENTIFIER_CONFLICT_OR_UNCONFIRMED"
        elif code in blocked:reason="UNRESOLVED_TEMPORAL_OR_IDENTITY_REVIEW"
        elif not valid_isin(isin):reason="INVALID_ISIN"
        elif not evidence:reason="MISSING_OFFICIAL_SOURCE_EVIDENCE"
        else:
            for source in evidence:
                if not isinstance(source,dict) or urlparse(str(source.get("source_url",""))).hostname!="iss.moex.com" or not re.fullmatch(r"[a-f0-9]{64}",str(source.get("source_sha256",""))):
                    reason="INVALID_OFFICIAL_SOURCE_EVIDENCE";break
        if row["company_join_status"]!="UNIQUE_STATIC" or len(companies)!=1:company_reason="W2_COMPANY_AMBIGUOUS_OR_UNMAPPED"
        elif len(issuers.get(isin,[]))!=1:company_reason="CANONICAL_COMPANY_AMBIGUOUS_OR_UNMAPPED"
        elif companies!=issuers[isin]:company_reason="W2_CANONICAL_COMPANY_CONTRADICTION"
        if require_unique_company and not reason:reason=company_reason
        audit.append(dict(registration_code_normalized=code,isin=isin,reason=reason or "ACCEPTED_CURRENT_STATIC_ONLY",
            w2_version=version,historical_baseline_eligible=False))
        if reason:continue
        accepted.append(dict(registration_code_normalized=code,isin=isin,company_id=companies[0] if not company_reason else None,
            current_security_type=types[0] if len(types)==1 else None,
            security_type_ambiguous=len(types)>1,w2_version=version,
            historical_identity_verified=False,identity_evidence_level="EXACT_CURRENT_STATIC_ONLY"))
    columns=["registration_code_normalized","isin","company_id","current_security_type","security_type_ambiguous",
             "w2_version","historical_identity_verified","identity_evidence_level"]
    result=pd.DataFrame(accepted,columns=columns)
    return result,pd.DataFrame(audit)


def merged_registration_map(existing_pairs,new_edges):
    """Conflicting existing/new exact facts block a code; neither source wins."""
    old=existing_pairs[["security_code","isin"]].rename(columns={"security_code":"registration_code_normalized"}).copy()
    old["registration_code_normalized"]=old.registration_code_normalized.map(registration_key)
    added=new_edges[["registration_code_normalized","isin"]]
    combined=pd.concat([old,added],ignore_index=True).dropna().drop_duplicates()
    combined=combined.loc[combined.registration_code_normalized.ne("")&combined["isin"].map(valid_isin)]
    counts=combined.groupby("registration_code_normalized",observed=True)["isin"].nunique()
    conflicts=set(counts.loc[counts.gt(1)].index)
    clean=combined.loc[~combined.registration_code_normalized.isin(conflicts)]
    return clean.set_index("registration_code_normalized")["isin"].to_dict(),conflicts


def affected_report_dates(holdings,exclusions):
    """Exclude verified conflicting code/date observations, not the whole code."""
    if not len(exclusions):return pd.Series(False,index=holdings.index)
    require_columns(exclusions,["registration_code_normalized","report_date"],"R4 dated exclusions")
    keys=pd.MultiIndex.from_arrays([holdings.registration_match_key,
        pd.to_datetime(holdings.report_date,errors="coerce").dt.normalize()])
    bad=pd.MultiIndex.from_arrays([exclusions.registration_code_normalized.map(registration_key),
        pd.to_datetime(exclusions.report_date,errors="raise").dt.normalize()])
    return pd.Series(keys.isin(bad),index=holdings.index)


def scheduled_maturity_facts(crosswalk):
    """MATDATE is a scheduled screen, never proof of a payment or cessation.

    ISSUEDATE is trading start; BUYBACKDATE is a yield horizon; EARLYREPAYMENT
    may be a boolean. None is promoted to an actual redemption/validity date.
    """
    rows=[]
    for row in crosswalk.to_dict("records"):
        if row["mapping_status"]!="EXACT_CURRENT_CONFIRMED" or not valid_isin(row["isin"]):continue
        for fact in structured_list(row.get("dated_source_facts"),"dated_source_facts"):
            if fact.get("field")!="MATDATE":continue
            title=str(fact.get("source_title","")).lower()
            if "погашени" not in title and not title.startswith("market snapshot maturity field"):continue
            value=str(fact.get("value", ""))
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}",value):continue
            date=pd.to_datetime(value,errors="coerce")
            if pd.isna(date):continue
            rows.append(dict(security_id="isin:"+row["isin"],scheduled_maturity_date=date,
                source_url=fact.get("source_url"),source_sha256=fact.get("source_sha256"),source_title=fact.get("source_title"),
                interpretation="SCHEDULED_NOT_CONFIRMED_ACTUAL_REDEMPTION"))
    columns=["security_id","scheduled_maturity_date","source_url","source_sha256","source_title","interpretation"]
    return pd.DataFrame(rows,columns=columns).drop_duplicates()


def maturity_interval_flags(panel,facts):
    flags=pd.Series(False,index=panel.index)
    if not len(panel) or not len(facts):return flags
    dates=facts[["security_id","scheduled_maturity_date"]].drop_duplicates()
    rows=panel[["interval_id","security_id","start_date","end_date"]].copy()
    rows["_row_index"]=panel.index
    matched=rows.merge(dates,on="security_id",how="inner",validate="many_to_many")
    affected=matched.scheduled_maturity_date.gt(matched.start_date)&matched.scheduled_maturity_date.le(matched.end_date)
    flags.loc[matched.loc[affected,"_row_index"].unique()]=True
    return flags


def load_approved_w2(config_path):
    """Verify a pinned handoff, release and every released artifact before use."""
    config_path=Path(config_path)
    config=json.loads(config_path.read_text(encoding="utf-8-sig"))
    if config.get("ready_for_handoff") is not True or config.get("scope")!="EXPLORATORY_CURRENT_STATIC_ONLY":
        raise IntegrityError("W2 input is not approved for the explicit current/static exploratory scope")
    dependencies=[str(config_path)]
    def checked(entry):
        path=Path(entry["path"])
        if sha256_file(path)!=entry["sha256"]:raise IntegrityError("Pinned W2 artifact changed: "+str(path))
        dependencies.append(str(path))
        return path
    handoff=json.loads(checked(config["handoff"]).read_text(encoding="utf-8-sig"))
    if handoff.get("ready_for_handoff") is not True:raise IntegrityError("Worker handoff is not ready")
    release=json.loads(checked(config["release"]).read_text(encoding="utf-8-sig"))
    if release.get("status")!="PASS":raise IntegrityError("W2 release verification failed")
    summary=json.loads(checked(release["summary"]).read_text(encoding="utf-8-sig"))
    qa=json.loads(checked(release["qa"]).read_text(encoding="utf-8-sig"))
    if summary.get("version")!=config["w2_version"] or summary.get("status")!="COMPLETE_WITH_LIMITS" or not summary.get("collection_complete"):
        raise IntegrityError("Unexpected/incomplete W2 version or collection status")
    if qa.get("status")!="PASS" or not qa.get("checks") or not all(c.get("passed") is True for c in qa["checks"]):
        raise IntegrityError("Independent W2 QA is incomplete or failed")
    if qa.get("summary_sha256")!=release["summary"]["sha256"]:
        raise IntegrityError("W2 QA and release refer to different summaries")
    paths={}
    for item in release["artifacts"]:
        path=checked(item)
        paths[path.name]=path
    if not {"crosswalk.parquet","temporal_review_queue.json","conflicts.json"}.issubset(paths):
        raise IntegrityError("W2 release lacks required mapping/conflict/temporal artifacts")
    if Path(release["summary"]["path"]).parent.name!=config["w2_version"]:
        raise IntegrityError("W2 version directory differs from pinned version")
    # R4 independently identified the genuinely conflicting holding dates.
    # Source registration dates are NOT turned into invented validity intervals.
    review=json.loads(checked(config["r4_review"]).read_text(encoding="utf-8-sig"))
    if review.get("status")!="COMPLETE_ACCEPTED_FOR_PROVISIONAL_DW_WITH_FINDINGS" or review.get("ready_for_handoff") is not True:
        raise IntegrityError("Final R4 does not approve the provisional scope")
    if review["crosswalk_release"]["version"]!=config["w2_version"] or review["specification"]["sha256"]!=config["spec_sha256"]:
        raise IntegrityError("R4 refers to a different W2 or R1")
    exclusions=[]
    for entry in review["temporal_queue_holdings_check"]["records"]:
        for date in entry["conflicting_dates"]:
            exclusions.append(dict(registration_code_normalized=registration_key(entry["code"]),report_date=date,
                reason="R4_CONFIRMED_CONFLICTING_REPORT_DATE",isin=entry["isin"]))
    exclusions=pd.DataFrame(exclusions,columns=["registration_code_normalized","report_date","reason","isin"]).drop_duplicates()
    crosswalk=pd.read_parquet(paths["crosswalk.parquet"])
    metadata=dict(version=summary["version"],status=summary["status"],scope=config["scope"],
        release_sha256=config["release"]["sha256"],summary_sha256=release["summary"]["sha256"],
        verification_sha256=release["qa"]["sha256"],independent_checks=len(qa["checks"]),
        source_hashes_verified_by_independent_QA=qa.get("public_sources_verified"),
        temporal_gate=summary["temporal_gate"],counts=summary["counts"],blocked_review_codes=[],
        affected_code_dates=exclusions.to_dict("records"),r4_review_sha256=config["r4_review"]["sha256"],
        temporal_policy="Actual R4-confirmed code/report dates only; no all-period code ban or invented validity interval",
        historical_identity_verified=False)
    return dict(frame=crosswalk,metadata=metadata,dependencies=sorted(set(dependencies)),blocked_codes=[],
        temporal_exclusions=exclusions,scheduled_maturities=scheduled_maturity_facts(crosswalk))
