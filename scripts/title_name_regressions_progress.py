"""One-shot status for the derived regression run; never mutates it."""
from pathlib import Path
import argparse
import json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=Path("./analysis_outputs/title_name_regressions_20261026"))
    p.add_argument("--preflight",action="store_true")
    args=p.parse_args();root=args.output
    state={"output_root":str(root)}
    for name in ["checkpoint.json","preparation.json","summary.json"]:
        path=root/name
        if not path.exists():continue
        value=json.loads(path.read_text(encoding="utf-8-sig"))
        if name=="preparation.json":
            value={k:value.get(k) for k in ["pair_count","events","missing_date_pairs","llm_counts","holdings"]}
        if name=="summary.json":
            value={k:value.get(k) for k in ["status","hypotheses","generated_at_utc"]}
        state[name]=value
    logs=sorted(root.glob("*.stderr.log"),key=lambda x:x.stat().st_mtime,reverse=True)
    if logs:
        path=logs[0]
        with path.open("rb") as stream:
            stream.seek(max(0,path.stat().st_size-7000))
            state["last_error_log"]={"path":str(path),"tail":stream.read().decode("utf-8",errors="replace")}
    if args.preflight and (root/"preparation.json").exists():
        import pandas as pd
        inventory=pd.read_parquet(root/"holdings_inventory.parquet",columns=["report_id","security_id","company_id","target_security_candidate","residence_documented_by_disclosure"])
        intervals=pd.read_parquet(root/"all_true_intervals.parquet")
        intervals=intervals.loc[intervals.preliminary_eligible]
        selected=inventory.target_security_candidate.eq(True)&inventory.residence_documented_by_disclosure.eq(True)
        candidate=inventory.loc[selected]
        targets=[]
        for side in ["opening_report_id","closing_report_id"]:
            targets.append(intervals[["fund_id","interval_id",side]].merge(candidate,left_on=side,right_on="report_id",how="inner",validate="one_to_many"))
        union=pd.concat(targets,ignore_index=True).drop_duplicates(["interval_id","security_id"])
        state["no_coefficient_panel_preflight"]=dict(target_endpoint_rows=len(candidate),
            eligible_union_target_rows=len(union),funds=union.fund_id.nunique(),companies=union.company_id.nunique(),
            securities=union.security_id.nunique(),note="Eligibility/key counts only; no coefficients fitted")
    print(json.dumps(state,ensure_ascii=True,default=str,allow_nan=False))


if __name__=="__main__":main()
