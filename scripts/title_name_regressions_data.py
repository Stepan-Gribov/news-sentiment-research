"""Read canonical DuckDB pointers and normalize exact holdings conservatively."""
from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

from title_name_regressions_core import IntegrityError, report_intervals, unique_keys


def normalize_code(value):
    if pd.isna(value):return None
    result=str(value).strip().upper()
    return result or None


def valid_isin(value):
    if not isinstance(value,str) or not re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}[0-9]",value):return False
    digits="".join(str(ord(x)-55) if x.isalpha() else x for x in value)
    total=0
    for i,char in enumerate(reversed(digits)):
        value=int(char)*(2 if i%2 else 1)
        total+=value//10+value%10
    return total%10==0


def read_holdings_sources(catalog):
    import duckdb
    c=duckdb.connect(str(catalog),read_only=True,config={"threads":2,"memory_limit":"2GB"})
    try:
        report=c.execute("SELECT report_id,fund_id,report_date,registration_number,fund_name_xls,"
            "fund_type_category,try_cast(net_asset_value_rub AS DOUBLE) AS NAV,"
            "try_cast(total_assets_thousand_rub AS DOUBLE) AS gross_thousand,"
            "try_cast(total_obligations_thousand_rub AS DOUBLE) AS obligations_thousand FROM funds.reports").fetchdf()
        leaves=c.execute("SELECT * FROM funds.holdings").fetchdf()
        totals=c.execute("SELECT report_id,row_type,parent_category,parent_subcategory,item_name,"
            "try_cast(amount_thousand_rub AS DOUBLE) AS amount_thousand_rub FROM funds.category_totals "
            "WHERE row_type IN ('asset_category','asset_subcategory') AND table_type='asset'").fetchdf()
        securities=c.execute("SELECT isin,security_code FROM refs.securities").fetchdf()
        bridge=c.execute("SELECT company_id,isin FROM refs.company_securities").fetchdf()
        market=c.execute("SELECT requested_isin,secid,type,primary_boardid FROM refs.moex_mapping WHERE status='resolved'").fetchdf()
        return report,leaves,totals,securities,bridge,market
    finally:c.close()


def catalog_source_description(catalog):
    import duckdb
    c=duckdb.connect(str(catalog),read_only=True,config={"threads":1,"memory_limit":"512MB"})
    try:
        names=["funds.reports","funds.holdings","funds.category_totals","refs.company_securities",
               "refs.companies","refs.securities","refs.moex_mapping"]
        views=c.execute("SELECT schema_name,view_name,sql FROM duckdb_views() "
                        "WHERE NOT internal").fetchdf()
        views=views.loc[(views.schema_name+"."+views.view_name).isin(names)]
        return dict(sources=c.execute("SELECT * FROM meta.sources").fetchdf().to_dict("records"),
                    view_definitions=views.to_dict("records"))
    finally:c.close()


def normalize_holdings(catalog,spec,*,crosswalk=None):
    report,leaf,totals,securities,bridge,market=read_holdings_sources(catalog)
    diagnostic={"source_reports":len(report),"source_leaf_rows":len(leaf),"category_rows":len(totals)}
    report["report_date"]=pd.to_datetime(report.report_date,errors="coerce")
    # Dated complete-revision conflicts cannot be arbitrarily resolved here.
    duplicate=report.duplicated(["fund_id","report_date"],keep=False)
    if duplicate.any():
        raise IntegrityError("Duplicate fund/report-date sources require an audited identical-statement/correction policy")
    unique_keys(report,["report_id"],"source reports")
    registration=report.groupby("fund_id",observed=True).registration_number.nunique()
    ambiguous_funds=set(registration.loc[registration.gt(1)].index)
    reg_funds=report[["fund_id","registration_number"]].dropna().drop_duplicates().groupby("registration_number",observed=True).fund_id.nunique()
    conflicting_reg=set(reg_funds.loc[reg_funds.gt(1)].index)
    report["fund_identity_verified"]=report.registration_number.notna()&~report.fund_id.isin(ambiguous_funds)&~report.registration_number.isin(conflicting_reg)
    report["passive_explicit"]= (report.fund_name_xls.fillna("")+" "+report.fund_type_category.fillna("")).str.contains(r"индекс|index",case=False,regex=True)
    report["active_management_status"]="unknown_provisional"
    report.loc[report.passive_explicit,"active_management_status"]="explicit_passive_excluded"
    diagnostic["fund_identity_ambiguous_ids"]=len(ambiguous_funds)
    diagnostic["conflicting_registration_numbers"]=len(conflicting_reg)
    diagnostic["unknown_active_management_reports"]=int((~report.passive_explicit).sum())

    leaf["amount_thousand_rub"]=pd.to_numeric(leaf.amount_thousand_rub,errors="coerce")
    leaf["share_pct"]=pd.to_numeric(leaf.share_pct,errors="coerce")
    for frame in [leaf,totals]:
        for col in ["parent_category","parent_subcategory"]:frame[col]=frame[col].fillna("")
    leaf["registration_number_security"]=leaf.registration_number_security.map(normalize_code)
    securities["isin"]=securities["isin"].map(normalize_code)
    securities["security_code"]=securities.security_code.map(normalize_code)
    pairs=securities.loc[securities["isin"].map(valid_isin),["security_code","isin"]].dropna().drop_duplicates()
    code_n=pairs.groupby("security_code",observed=True)["isin"].nunique()
    bad_codes=set(code_n.loc[code_n.gt(1)].index)
    code_map=pairs.loc[~pairs.security_code.isin(bad_codes)].set_index("security_code")["isin"].to_dict()
    w2_target_keys=set();w2_identifier_keys=set();w2_type_pairs=pd.DataFrame(columns=["requested_isin","type"])
    leaf["registration_match_key"]=leaf.registration_number_security
    leaf["w2_static_link_used"]=False
    if crosswalk is not None:
        from title_name_regressions_crosswalk import registration_key,select_exact_edges,merged_registration_map,affected_report_dates
        canonical=bridge.copy();canonical["isin"]=canonical["isin"].map(normalize_code)
        version=crosswalk["metadata"]["version"]
        identifier_edges,id_audit=select_exact_edges(crosswalk["frame"],canonical,version=version,
            require_unique_company=False)
        target_edges,target_audit=select_exact_edges(crosswalk["frame"],canonical,version=version)
        code_map,bad_codes=merged_registration_map(pairs,identifier_edges)
        target_edges=target_edges.loc[~target_edges.registration_code_normalized.isin(bad_codes)]
        w2_target_keys=set(target_edges.registration_code_normalized)
        w2_identifier_keys=set(identifier_edges.registration_code_normalized)-bad_codes
        leaf["registration_match_key"]=leaf.registration_number_security.map(registration_key)
        leaf["w2_static_link_used"]=leaf.registration_match_key.isin(w2_identifier_keys)&~leaf.registration_match_key.map(valid_isin)
        unique_types=identifier_edges.loc[~identifier_edges.security_type_ambiguous&identifier_edges.current_security_type.notna(),["isin","current_security_type"]]
        w2_type_pairs=unique_types.rename(columns={"isin":"requested_isin","current_security_type":"type"}).drop_duplicates()
        target_audit.loc[target_audit.registration_code_normalized.isin(bad_codes),"reason"]="EXISTING_W2_ISIN_CONTRADICTION"
        crosswalk["selection_audit"]=target_audit
        diagnostic["w2_overlay"]=dict(**crosswalk["metadata"],identifier_edges=len(identifier_edges),
            target_company_edges=len(target_edges),selection_counts=target_audit.reason.value_counts().to_dict(),
            existing_overlay_conflicting_codes=sorted(bad_codes))
    leaf["isin"]=leaf.registration_match_key.map(lambda x:x if valid_isin(x) else code_map.get(x))
    bridge["isin"]=bridge["isin"].map(normalize_code)
    bridge=bridge.dropna().drop_duplicates(["isin","company_id"])
    issuer_n=bridge.groupby("isin",observed=True).company_id.nunique()
    bad_isins=set(issuer_n.loc[issuer_n.gt(1)].index)
    conflicts=bridge.loc[bridge["isin"].isin(bad_isins)].copy()
    company_map=bridge.loc[~bridge["isin"].isin(bad_isins)].set_index("isin").company_id.to_dict()
    leaf["company_id"]=leaf["isin"].map(company_map)
    leaf["issuer_bridge_ambiguous"]=leaf["isin"].isin(bad_isins)
    leaf["security_id"]=["isin:"+str(i) if pd.notna(i) else "reg:"+str(code) if pd.notna(code) else
        "unidentified:"+hashlib.blake2b(f"{cat}\t{sub}\t{name}".encode(),digest_size=12).hexdigest()
        for i,code,cat,sub,name in zip(leaf["isin"],leaf.registration_number_security,leaf.parent_category,leaf.parent_subcategory,leaf.item_name)]
    type_pairs=pd.concat([market[["requested_isin","type"]],w2_type_pairs],ignore_index=True).dropna().drop_duplicates()
    conflicting_types=set(type_pairs.groupby("requested_isin",observed=True).type.nunique().loc[lambda x:x.gt(1)].index)
    market_type=type_pairs.loc[~type_pairs.requested_isin.isin(conflicting_types)].set_index("requested_isin").type.to_dict()
    leaf["market_type"]=leaf["isin"].map(market_type)
    eq=leaf.parent_category.isin(["Акции","Акции российских эмитентов"])
    bond=leaf.parent_category.isin(["Облигации","Облигации российских эмитентов"])
    leaf["asset_class"]=np.select([eq,bond],["equity","bond"],default="other")
    leaf["residence_documented_by_disclosure"]=leaf.parent_category.isin(["Акции российских эмитентов","Облигации российских эмитентов"])
    leaf["target_security_candidate"]=(leaf.asset_class.isin(["equity","bond"])
        &leaf.company_id.notna()&~leaf.issuer_bridge_ambiguous
        &~leaf.market_type.isin(["ofz_bond","subfederal_bond","municipal_bond","state_bond","ifi_bond","etf_ppif","public_ppif"]))
    class_conflict=(eq&leaf.market_type.notna()&~leaf.market_type.isin(["common_share","preferred_share"])) | (
        bond&leaf.market_type.notna()&~leaf.market_type.isin(["corporate_bond","exchange_bond","euro_bond"]))
    leaf.loc[class_conflict,"target_security_candidate"]=False
    leaf.loc[leaf["isin"].isin(conflicting_types),"target_security_candidate"]=False
    if crosswalk is not None:
        leaf.loc[leaf.w2_static_link_used&~leaf.registration_match_key.isin(w2_target_keys),"target_security_candidate"]=False
        temporal=affected_report_dates(leaf,crosswalk.get("temporal_exclusions",pd.DataFrame()))
        leaf.loc[temporal,"target_security_candidate"]=False
        crosswalk["temporal_exclusion_audit"]=leaf.loc[temporal,["report_id","fund_id","report_date","registration_match_key","isin","amount_thousand_rub"]].copy()
        diagnostic["w2_temporal_conflicting_leaf_rows"]=int(temporal.sum())
        diagnostic["w2_temporal_conflicting_leaf_rows_2018_onward"]=int((temporal&pd.to_datetime(leaf.report_date).ge("2018-01-01")).sum())
    diagnostic["conflicting_source_security_type_isins"]=len(conflicting_types)

    # Aggregate actual reported lots; source categories never enter as target securities.
    identity_cols=["report_id","security_id"]
    for col in ["company_id","isin"]:
        if leaf.groupby(identity_cols,observed=True)[col].nunique().gt(1).any():
            raise IntegrityError(f"Holding lots disagree on {col}")
    class_count=leaf.groupby(identity_cols,observed=True).asset_class.transform("nunique")
    class_ambiguous=class_count.gt(1)
    diagnostic["lots_with_conflicting_source_asset_class"]=int(class_ambiguous.sum())
    diagnostic["report_security_groups_with_conflicting_source_asset_class"]=int(leaf.loc[class_ambiguous,identity_cols].drop_duplicates().shape[0])
    # Preserve every reported lot/value in the portfolio denominator. A conflicting
    # category does not justify guessing its target class or aborting other funds.
    leaf.loc[class_ambiguous,"asset_class"]="other"
    leaf.loc[class_ambiguous,"target_security_candidate"]=False
    leaf.loc[class_ambiguous,"residence_documented_by_disclosure"]=False
    leaf["missing_amount"]=leaf.amount_thousand_rub.isna()
    grouped=leaf.groupby(identity_cols,sort=False,observed=True,dropna=False)
    inventory=grouped.agg(amount_thousand_rub=("amount_thousand_rub","sum"),
        amount_nonmissing=("amount_thousand_rub","count"),missing_amount=("missing_amount","any"),
        source_lots=("amount_thousand_rub","size"),company_id=("company_id","first"),isin=("isin","first"),
        asset_class=("asset_class","first"),market_type=("market_type","first"),
        target_security_candidate=("target_security_candidate","all"),
        w2_static_link_used=("w2_static_link_used","any"),
        residence_documented_by_disclosure=("residence_documented_by_disclosure","all")).reset_index()
    inventory.loc[inventory.amount_nonmissing.eq(0),"amount_thousand_rub"]=np.nan
    inventory=inventory.drop(columns="amount_nonmissing")
    inventory["bucket_kind"]="instrument_leaf"
    leaf_sum=leaf.groupby(["report_id","parent_category","parent_subcategory"],dropna=False,observed=True).amount_thousand_rub.sum(min_count=1).rename("leaf_amount").reset_index()
    category_sum=leaf.groupby(["report_id","parent_category"],dropna=False,observed=True).amount_thousand_rub.sum(min_count=1).rename("leaf_amount").reset_index()
    total_keys=["report_id","parent_category","parent_subcategory"]
    category_conflicts=set()
    pieces=[]
    for row_type in ["asset_subcategory","asset_category"]:
        t=totals.loc[totals.row_type.eq(row_type)].copy()
        if row_type=="asset_category":t["parent_subcategory"]=""
        conflicts_n=t.groupby(total_keys,dropna=False,observed=True).amount_thousand_rub.nunique()
        category_conflicts.update(key[0] for key,n in conflicts_n.items() if n>1)
        t=t.drop_duplicates(total_keys+["amount_thousand_rub"])
        t=t.loc[~t.report_id.isin(category_conflicts)]
        unique_keys(t,total_keys,"nonoverlapping category metadata")
        t=t.merge(leaf_sum if row_type=="asset_subcategory" else category_sum,
            on=total_keys if row_type=="asset_subcategory" else total_keys[:2],how="left",validate="one_to_one")
        t["leaf_amount"]=t.leaf_amount.fillna(0)
        t["residual_amount"]=t.amount_thousand_rub-t.leaf_amount
        if row_type=="asset_category" and pieces:
            prior=pd.concat(pieces).groupby(["report_id","parent_category"],observed=True).residual_amount.sum().rename("sub_residual").reset_index()
            t=t.merge(prior,on=["report_id","parent_category"],how="left",validate="one_to_one")
            t["residual_amount"]-=t.sub_residual.fillna(0)
        negative=t.loc[t.residual_amount.lt(-.011),"report_id"]
        category_conflicts.update(negative)
        pieces.append(t.loc[t.residual_amount.gt(.000001),total_keys+["residual_amount"]])
    residual=pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame(columns=total_keys+["residual_amount"])
    if len(residual):
        residual["security_id"]=["documented_bucket:"+hashlib.blake2b(f"{a}\t{b}".encode(),digest_size=12).hexdigest() for a,b in zip(residual.parent_category,residual.parent_subcategory)]
        residual=residual.rename(columns={"residual_amount":"amount_thousand_rub"})
        residual["bucket_kind"]="documented_residual_bucket"
        residual["company_id"]=None;residual["isin"]=None;residual["asset_class"]="other"
        residual["target_security_candidate"]=False;residual["source_lots"]=0;residual["missing_amount"]=False
        inventory=pd.concat([inventory,residual[[x for x in inventory.columns if x in residual]]],ignore_index=True)
    inventory=inventory.merge(report[["report_id","fund_id","report_date","gross_thousand"]],on="report_id",how="left",validate="many_to_one")
    inventory["weight"]=inventory.amount_thousand_rub/inventory.gross_thousand
    sums=inventory.groupby("report_id",observed=True).agg(inventory_thousand=("amount_thousand_rub","sum"),
        any_missing_amount=("missing_amount","any"),any_negative=("weight",lambda x:x.lt(0).any()),
        any_weight_gt1=("weight",lambda x:x.gt(1.001).any()))
    report=report.merge(sums,on="report_id",how="left",validate="one_to_one")
    report["unallocated_fraction"]=(report.gross_thousand-report.inventory_thousand)/report.gross_thousand
    report["nav_reconciliation_fraction"]=(report.gross_thousand*1000-report.obligations_thousand*1000-report.NAV).abs()/(report.gross_thousand*1000)
    tolerance=spec["holdings_and_weights"]["reconciliation_tolerance_fraction"]
    rounding=spec["holdings_and_weights"]["rounding_gap_max_fraction"]
    report["category_conflict"]=report.report_id.isin(category_conflicts)
    report["full_portfolio"]=(report.gross_thousand.gt(0)&report.NAV.gt(0)&report.obligations_thousand.notna()
        &report.nav_reconciliation_fraction.le(tolerance)&report.unallocated_fraction.abs().le(rounding)
        &~report.category_conflict&~report.any_missing_amount.fillna(True)&~report.any_negative.fillna(True)
        &~report.any_weight_gt1.fillna(True)&report.fund_identity_verified&report.report_date.notna())
    # Preserve even small positive unidentified amounts as unpriced, not cash.
    gap=report.loc[report.unallocated_fraction.gt(1e-12)].copy()
    if len(gap):
        extra=gap[["report_id","fund_id","report_date","gross_thousand"]].copy()
        extra["security_id"]="__UNALLOCATED__";extra["company_id"]=None;extra["isin"]=None
        extra["asset_class"]="other";extra["target_security_candidate"]=False;extra["bucket_kind"]="unallocated_residual"
        extra["weight"]=gap.unallocated_fraction.to_numpy();extra["amount_thousand_rub"]=gap.gross_thousand.to_numpy()*gap.unallocated_fraction.to_numpy()
        inventory=pd.concat([inventory,extra],ignore_index=True)
    inventory["is_leaf"]=True
    inventory["asset_type"]=inventory.asset_class
    hhi=inventory.assign(squared=inventory.weight**2).groupby("report_id",observed=True).squared.sum(min_count=1)
    report["HHI"]=report.report_id.map(hhi)
    report["HHI_has_documented_or_unallocated_buckets"]=report.report_id.isin(inventory.loc[inventory.bucket_kind.ne("instrument_leaf"),"report_id"])
    intervals=report_intervals(report[["fund_id","report_date","report_id","full_portfolio"]])
    for boundary,prefix in [("opening_report_id","opening"),("closing_report_id","closing")]:
        fields=report[["report_id","NAV","HHI","HHI_has_documented_or_unallocated_buckets","passive_explicit","active_management_status","unallocated_fraction","nav_reconciliation_fraction"]].rename(
            columns={"report_id":boundary,**{x:prefix+"_"+x for x in ["NAV","HHI","HHI_has_documented_or_unallocated_buckets","passive_explicit","active_management_status","unallocated_fraction","nav_reconciliation_fraction"]}})
        intervals=intervals.merge(fields,on=boundary,how="left",validate="many_to_one")
    intervals["calendar_eom_consecutive"]=(intervals.start_date.dt.is_month_end&intervals.end_date.dt.is_month_end
        &(intervals.end_date.dt.year*12+intervals.end_date.dt.month-intervals.start_date.dt.year*12-intervals.start_date.dt.month).eq(1))
    eligible=(intervals.calendar_eom_consecutive &intervals.month.ge(spec["time_and_scope"]["baseline_closing_month_start"])
        &intervals.month.le(spec["time_and_scope"]["baseline_closing_month_end_cap"])
        &intervals.opening_full_portfolio.astype(bool)&intervals.closing_full_portfolio.astype(bool)
        &~intervals.opening_passive_explicit&~intervals.closing_passive_explicit)
    intervals["preliminary_eligible"]=eligible
    diagnostic.update(normalized_inventory_rows=len(inventory),reports_reconciled=int(report.full_portfolio.sum()),
        report_category_conflicts=len(category_conflicts),unambiguous_issuer_isins=len(company_map),ambiguous_issuer_isins=len(bad_isins),
        registration_code_to_multiple_isins=len(bad_codes),adjacent_actual_intervals=len(intervals),
        consecutive_eom_intervals=int(intervals.calendar_eom_consecutive.sum()),preliminary_eligible_intervals=int(eligible.sum()),
        normalization="amount_thousand_rub / total_assets_thousand_rub; share_pct not used as mixed weights",
        documented_residual_buckets=int(inventory.bucket_kind.eq("documented_residual_bucket").sum()),
        historical_issuer_validity="not established; static crosswalk only, no baseline temporal-identity claim")
    return inventory,report,intervals,conflicts,diagnostic,market
