#!/usr/bin/env python3
"""W2: evidence-preserving exact registration-code -> ISIN enrichment.

Run on research-host. Immutable resources and the live LLM run are read only. Raw
public ISS responses, checkpoints and versioned outputs live in analysis_outputs.
No fuzzy name join, security successor inference or historical backdating.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sqlite3
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(".")
RESOURCE=ROOT/"databaseresources"
DEFAULT_OUTPUT=ROOT/"analysis_outputs/title_name_security_crosswalk_20261026"
ISIN_RE=re.compile(r"\b[A-Z]{2}[A-Z0-9]{9}[0-9]\b")
EQUITY={"акции","акции российских эмитентов","акции иностранных эмитентов"}
PRIORITY=["1-01-00077-A","1-02-00028-A","10301481B","1-02-00122-A"]
EMPTY={"","NAN","NONE","NULL","NAT","<NA>"}
BASE="https://iss.moex.com"
DATE_FIELDS={"REGISTRY_DATE","ISSUEDATE","MATDATE","BUYBACKDATE","EARLYREPAYMENT","ISSUESTARTDATE","ISSUEENDDATE"}


def now():
    return datetime.now(timezone.utc).isoformat()


def normalized(value):
    """Typography only. Never remove suffixes or replace Cyrillic lookalikes."""
    value=unicodedata.normalize("NFKC",str(value or "")).upper().strip()
    value=re.sub(r"[‐‑‒–—−]","-",value)
    value=re.sub(r"\s+","",value)
    return "" if value in EMPTY else value


def direct_isins(value):
    return set(ISIN_RE.findall(normalized(value)))


def valid_isin(value):
    if not re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}[0-9]",value or ""):
        return False
    digits="".join(str(ord(c)-55) if c.isalpha() else c for c in value)
    total=0
    for i,c in enumerate(reversed(digits)):
        n=int(c)*(2 if i%2 else 1)
        total+=n//10+n%10
    return total%10==0


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def stamp(path):
    p=Path(path);s=p.stat()
    return {"path":str(p),"bytes":s.st_size,"mtime_ns":s.st_mtime_ns,"sha256":sha(p)}


def read_csv(path):
    csv.field_size_limit(64*1024*1024)
    opener=gzip.open if str(path).endswith(".gz") else open
    with opener(path,"rt",encoding="utf-8-sig",newline="") as handle:
        yield from csv.DictReader(handle)


def table(obj,name):
    block=obj.get(name,{})
    return [dict(zip(block.get("columns",[]),row)) for row in block.get("data",[])]


def json_write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
    temporary.replace(path)


def csv_write(path,records,fields=None):
    records=list(records)
    fields=fields or sorted({k for row in records for k in row})
    with Path(path).open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in records:
            writer.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(list,dict,set)) else v for k,v in row.items()})


@contextmanager
def exclusive_lock(output):
    path=output/"builder.lock"
    with path.open("a+b") as handle:
        handle.seek(0,2)
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


class ISSCache:
    def __init__(self,root,delay=.65,offline=False):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.delay=delay;self.offline=offline;self.last=0.;self.requests=0;self.hits=0;self.errors=[]

    def get(self,url):
        if urllib.parse.urlparse(url).hostname!="iss.moex.com":raise ValueError("Only public MOEX ISS requests are supported")
        key=hashlib.sha256(url.encode()).hexdigest()
        raw_path=self.root/(key+".json");meta_path=self.root/(key+".meta.json")
        if raw_path.is_file() and meta_path.is_file():
            meta=json.loads(meta_path.read_text(encoding="utf-8"))
            if meta["url"]!=url or sha(raw_path)!=meta["sha256"]:raise RuntimeError("Cache identity/hash mismatch")
            self.hits+=1
            return json.loads(raw_path.read_text(encoding="utf-8-sig")),meta
        if self.offline:raise FileNotFoundError("Uncached public source: "+url)
        last_error=None
        for attempt in range(3):
            time.sleep(max(0,self.delay-(time.monotonic()-self.last)))
            self.last=time.monotonic();self.requests+=1
            try:
                req=urllib.request.Request(url,headers={"User-Agent":"Dissertation-research-identifier-audit/1.0","Accept":"application/json"})
                with urllib.request.urlopen(req,timeout=25) as response:
                    raw=response.read();status=response.status
                obj=json.loads(raw)
                meta={"url":url,"observed_at_utc":now(),"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw),"http_status":status,"raw_path":str(raw_path)}
                raw_path.write_bytes(raw);json_write(meta_path,meta)
                return obj,meta
            except (OSError,ValueError,urllib.error.URLError) as exc:
                last_error=f"{type(exc).__name__}: {exc}"
                if isinstance(exc,urllib.error.HTTPError) and exc.code in (400,401,403,404):break
                if attempt<2:time.sleep(2**(attempt+1))
        self.errors.append({"url":url,"error":last_error,"observed_at_utc":now()})
        raise RuntimeError(last_error)


def iss_url(path,**query):
    return BASE+path+"?"+urllib.parse.urlencode(query)


def read_inventory(resource):
    path=resource/"funds/mac-20261008/fund_holdings_long.csv.gz"
    codes={};total=Counter();raw_variants=Counter()
    for row in read_csv(path):
        total["all_rows"]+=1
        if row.get("table_type")!="asset" or row.get("row_type")!="asset_item":continue
        total["leaf_asset_rows"]+=1
        raw=str(row.get("registration_number_security") or "").strip();key=normalized(raw)
        category=str(row.get("parent_category") or "").strip()
        day=str(row.get("report_date") or "")[:10]
        if key not in codes:
            codes[key]={"code":key,"rows":0,"equity_rows":0,"raw_codes":set(),"categories":Counter(),"years":Counter(),"funds":set(),"dates":set(),"examples":set()}
        item=codes[key];item["rows"]+=1;item["equity_rows"]+=category.casefold() in EQUITY
        item["raw_codes"].add(raw);item["categories"][category]+=1;item["years"][day[:4] or "missing"]+=1
        item["funds"].add(str(row.get("fund_id") or ""));item["dates"].add(day)
        if len(item["examples"])<5:item["examples"].add(str(row.get("item_name") or ""))
        raw_variants[(raw,key)]+=1
    old=defaultdict(set);old_path=resource/"funds/mac-20261008/holdings_security_name_isin_mapping.csv"
    for row in read_csv(old_path):
        key=normalized(row.get("security_code"));isins=direct_isins(row.get("isin"))
        if key:old[key].update(isins)
    manifest_path=resource/"manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    candidates=[x for x in manifest.get("links",[]) if re.search(r"regnum|securit|metadata|issuer|emitent|isin",str(x.get("resource","")),re.I)]
    inventory={"holdings":stamp(path),"old_dictionary":stamp(old_path),"resource_manifest":stamp(manifest_path),"registered_reference_candidates":candidates,
               "counts":dict(total),"unique_nonempty_codes":len(set(codes)-{""}),"true_old_registration_bridges":sum(bool(v and k not in v) for k,v in old.items()),
               "known_reference_schema_limit":"refs.securities is projected from holdings_security_name_isin_mapping; refs.moex_mapping lacks registration number. W1 already checked these registered definitions."}
    return codes,old,inventory,raw_variants


def registry(resource):
    import pyarrow.parquet as pq
    path=resource/"mapping/active-20261008-safe-otkritie/cleaned_single_industry/company_registry_active.parquet"
    by_isin=defaultdict(set);names={}
    for row in pq.read_table(path,columns=["company_id","company","isins"]).to_pylist():
        names[row["company_id"]]=row["company"]
        for isin in ISIN_RE.findall(str(row["isins"] or "").upper()):by_isin[isin].add(row["company_id"])
    return by_isin,names,stamp(path)


def relevant_companies():
    db=ROOT/"news_data/llm_runs/gemma_title_name_no_gaz_20261008_01/results.sqlite3"
    con=sqlite3.connect(db.resolve().as_uri()+"?mode=ro",uri=True,timeout=20)
    try:
        con.execute("PRAGMA query_only=ON");con.execute("BEGIN")
        query="SELECT DISTINCT json_extract(result_json,'$.company_id') FROM results WHERE valid=1 AND json_extract(result_json,'$.relevance')='relevant'"
        return {r[0] for r in con.execute(query)}
    finally:con.close()


def add_candidates(store,obj,meta,codes,block="securities",source_kind="iss_security_catalog"):
    n=0
    for raw in table(obj,block):
        row={k.lower():v for k,v in raw.items()}
        key=normalized(row.get("regnumber"));isin=normalized(row.get("isin"))
        if not key or key not in codes or not isin:continue
        record={"registration_code":key,"source_regnumber":row.get("regnumber"),"isin":isin,"isin_checksum_valid":valid_isin(isin),
                "secid":str(row.get("secid") or ""),"issuer_source_id":str(row.get("emitent_id") or row.get("emitter_id") or ""),
                "issuer_legal_name":row.get("emitent_title"),"issuer_inn":row.get("emitent_inn"),"issuer_okpo":row.get("emitent_okpo"),
                "security_type":row.get("type"),"security_group":row.get("group"),"source_kind":source_kind,
                "source_url":meta["url"],"observed_at_utc":meta["observed_at_utc"],"raw_path":meta["raw_path"],"source_sha256":meta["sha256"],
                "source_fields":row}
        signature=(key,isin,record["secid"],meta["url"])
        store[signature]=record;n+=1
    return n


def consolidate(evidence,details,by_isin,codes,changeovers):
    grouped=defaultdict(list)
    for row in evidence.values():grouped[row["registration_code"]].append(row)
    output=[];conflicts=[]
    for key,items in sorted(grouped.items()):
        isins={r["isin"] for r in items};issuer_ids={r["issuer_source_id"] for r in items if r["issuer_source_id"]}
        detail_facts=[];detail_sources=[];detail_observations=[];detail_mismatch=False;issuer_mismatch=False
        card_issuer_ids=set()
        for item in items:
            detail=details.get(item["secid"])
            if not detail:continue
            vals={str(r["name"]):r for r in table(detail[0],"description")}
            observed_reg=normalized(vals.get("REGNUMBER",{}).get("value"));observed_isin=normalized(vals.get("ISIN",{}).get("value"))
            observed_issuer=str(vals.get("EMITTER_ID",{}).get("value") or "")
            observation={"secid":item["secid"],"registration_code_observed":observed_reg,"isin_observed":observed_isin,
                         "issuer_source_id_observed":observed_issuer,"source":detail[1]}
            if (observed_reg and observed_reg!=key) or (observed_isin and observed_isin!=item["isin"]):
                detail_mismatch=True;observation["comparison"]="CONFLICTING_IDENTITY";detail_observations.append(observation);continue
            if not observed_reg or not observed_isin:
                observation["comparison"]="IDENTITY_FIELDS_MISSING";detail_observations.append(observation)
                continue
            if observed_issuer:card_issuer_ids.add(observed_issuer)
            if observed_issuer and item["issuer_source_id"] and observed_issuer!=item["issuer_source_id"]:
                issuer_mismatch=True;observation["comparison"]="CONFLICTING_ISSUER"
            else:observation["comparison"]="EXACT_IDENTITY_CONFIRMED"
            detail_observations.append(observation)
            detail_sources.append(detail[1])
            for name,row in vals.items():
                if name in DATE_FIELDS and row.get("value"):
                    detail_facts.append({"field":name,"source_title":row.get("title"),"value":row["value"],"source_url":detail[1]["url"],"observed_at_utc":detail[1]["observed_at_utc"],"source_sha256":detail[1]["sha256"]})
        for item in items:
            for field in ("matdate",):
                if item["source_fields"].get(field):detail_facts.append({"field":field.upper(),"source_title":"Market snapshot maturity field, scheduled date; actual redemption not established","value":item["source_fields"][field],"source_url":item["source_url"],"source_sha256":item["source_sha256"]})
        reasons=[]
        if len(isins)!=1:reasons.append("multiple_isins_for_exact_registration_code")
        issuer_ids.update(card_issuer_ids)
        if len(issuer_ids)>1:reasons.append("multiple_legal_issuer_source_ids")
        if not all(r["isin_checksum_valid"] for r in items):reasons.append("invalid_isin_checksum")
        if detail_mismatch:reasons.append("catalog_vs_security_card_identity_mismatch")
        if issuer_mismatch:reasons.append("catalog_vs_security_card_issuer_mismatch")
        isin=next(iter(isins)) if len(isins)==1 else None
        companies=sorted(by_isin.get(isin,set())) if isin else []
        secids={r["secid"] for r in items}
        changes=[c for c in changeovers if str(c.get("old_secid")) in secids or str(c.get("new_secid")) in secids]
        unique_sources={r["source_url"]:{k:r[k] for k in ("source_url","observed_at_utc","raw_path","source_sha256")} for r in items}
        unique_dates={json.dumps(r,sort_keys=True,ensure_ascii=False):r for r in detail_facts}
        observed_dates=sorted(d for d in codes[key]["dates"] if re.fullmatch(r"\d{4}-\d{2}-\d{2}",d))
        temporal_issues=[]
        for fact in unique_dates.values():
            value=str(fact["value"])
            if fact["field"]=="REGISTRY_DATE" and observed_dates and re.fullmatch(r"\d{4}-\d{2}-\d{2}",value) and value>observed_dates[0]:
                temporal_issues.append({"issue":"source_registration_date_after_first_observed_holding","first_observed_holding":observed_dates[0],"source_fact":fact,
                                        "interpretation":"Historical review required; neither the holdings date nor the current source date establishes a complete identity interval."})
        record={"registration_code_normalized":key,"registration_code_originals":sorted(codes[key]["raw_codes"]),
                "source_regnumbers":sorted({str(r["source_regnumber"]) for r in items}),"isin":isin,"candidate_isins":sorted(isins),
                "mapping_status":"CONFLICT" if reasons else "EXACT_CURRENT_CONFIRMED","conflict_reasons":reasons,
                "matching_rule":"Exact normalized registration number; NFKC/case/whitespace/dash only; no name or confusable-letter matching",
                "issuer_source_ids":sorted(issuer_ids),"issuer_legal_names":sorted({str(r["issuer_legal_name"]) for r in items if r["issuer_legal_name"]}),
                "issuer_inns":sorted({str(r["issuer_inn"]) for r in items if r["issuer_inn"]}),"security_ids":sorted(secids),
                "security_types":sorted({str(r["security_type"]) for r in items if r["security_type"]}),
                "company_ids_static":companies,"company_join_status":"UNIQUE_STATIC" if len(companies)==1 else "AMBIGUOUS" if companies else "UNMAPPED",
                "observed_at_utc_min":min(r["observed_at_utc"] for r in items),"observed_at_utc_max":max(r["observed_at_utc"] for r in items),
                "source_evidence":list(unique_sources.values()),"security_card_evidence":list({d['url']:d for d in detail_sources}.values()),
                "security_card_observations":list({json.dumps(d,sort_keys=True,ensure_ascii=False):d for d in detail_observations}.values()),
                "dated_source_facts":list(unique_dates.values()),"technical_trading_code_changes":changes,
                "temporal_review_issues":temporal_issues,
                "temporal_evidence_level":"OFFICIAL_DATED_FACTS_NO_IDENTITY_INTERVAL" if detail_facts else "CURRENT_METADATA_ONLY",
                "identity_valid_from":None,"identity_valid_to":None,"historical_baseline_eligible":False,
                "temporal_limitation":"Source registration/trading/maturity dates and ticker changes do not prove a complete dated issuer-registration-ISIN history. Holdings first/last dates are never used as validity limits.",
                "holdings_rows":codes[key]["rows"],"strict_equity_holdings_rows":codes[key]["equity_rows"],"first_observed_holding":min(codes[key]["dates"]),"last_observed_holding":max(codes[key]["dates"])}
        output.append(record)
        if reasons or len(companies)>1:conflicts.append(record)
    return output,conflicts


def coverage(resource,old,crosswalk,by_isin,relevant):
    extended={k:set(v) for k,v in old.items()};blocked=set()
    for r in crosswalk:
        code=r["registration_code_normalized"]
        if r["mapping_status"]!="EXACT_CURRENT_CONFIRMED":blocked.add(code);continue
        extended.setdefault(code,set()).add(r["isin"])
    stats={label:{"counts":Counter(),"companies":set(),"relevant_companies":set(),"securities":set(),"codes":set(),"funds":set(),"years":defaultdict(Counter),"year_entities":defaultdict(lambda:{k:set() for k in ("companies","relevant_companies","securities","codes","funds")}),"equity_companies":set()} for label in ("before","after")}
    for row in read_csv(resource/"funds/mac-20261008/fund_holdings_long.csv.gz"):
        if row.get("table_type")!="asset" or row.get("row_type")!="asset_item":continue
        code=normalized(row.get("registration_number_security"));direct=direct_isins(code)
        equity=str(row.get("parent_category") or "").strip().casefold() in EQUITY
        year=str(row.get("report_date") or "")[:4] or "missing"
        for label,lookup in (("before",old),("after",extended)):
            s=stats[label];s["counts"]["asset_item_rows"]+=1;s["years"][year]["asset_item_rows"]+=1
            if label=="after" and code in blocked:
                s["counts"]["excluded_conflict_rows"]+=1;continue
            candidates=direct or lookup.get(code,set())
            if len(candidates)!=1:
                s["counts"]["unmapped_security_rows" if not candidates else "ambiguous_security_rows"]+=1;continue
            isin=next(iter(candidates));s["counts"]["single_isin_rows"]+=1;s["years"][year]["single_isin_rows"]+=1
            s["codes"].add(code);s["securities"].add(isin)
            s["year_entities"][year]["codes"].add(code);s["year_entities"][year]["securities"].add(isin)
            if equity:s["counts"]["equity_single_isin_rows"]+=1
            companies=by_isin.get(isin,set())
            if len(companies)!=1:
                s["counts"]["unmapped_company_rows" if not companies else "ambiguous_company_rows"]+=1;continue
            cid=next(iter(companies));s["companies"].add(cid);s["funds"].add(str(row.get("fund_id")))
            s["year_entities"][year]["companies"].add(cid);s["year_entities"][year]["funds"].add(str(row.get("fund_id")))
            s["counts"]["unique_static_company_rows"]+=1;s["years"][year]["unique_static_company_rows"]+=1
            if cid in relevant:
                s["counts"]["valid_relevant_company_rows"]+=1;s["years"][year]["valid_relevant_company_rows"]+=1
                s["relevant_companies"].add(cid);s["year_entities"][year]["relevant_companies"].add(cid)
            if equity:s["counts"]["equity_unique_company_rows"]+=1;s["equity_companies"].add(cid)
    result={}
    for label,s in stats.items():
        by_year={}
        for y,c in sorted(s["years"].items()):
            e=s["year_entities"][y]
            by_year[y]={**dict(c),"unique_companies":len(e["companies"]),"unique_relevant_companies":len(e["relevant_companies"]),"unique_securities":len(e["securities"]),"unique_codes":len(e["codes"]),"funds_with_unique_company":len(e["funds"])}
        result[label]={"counts":dict(s["counts"]),"unique_companies":len(s["companies"]),"unique_relevant_companies":len(s["relevant_companies"]),"unique_securities":len(s["securities"]),"unique_codes":len(s["codes"]),"funds_with_unique_company":len(s["funds"]),"unique_equity_companies":len(s["equity_companies"]),"company_ids":sorted(s["companies"]),"relevant_company_ids":sorted(s["relevant_companies"]),"by_year":by_year}
    result["interpretation"]="One count per asset_item; no join to individual articles. AFTER is static exploratory coverage, not proof of historical validity, Russian issuer residence, active fund status or final regression eligibility."
    return result


def build(args):
    out=args.output_root.resolve();resource=args.resource_root.resolve()
    if out==resource or resource in out.parents or "llm_runs" in out.parts:raise ValueError("Output must be outside protected inputs/live run")
    out.mkdir(parents=True,exist_ok=True)
    with exclusive_lock(out):
        version=datetime.now(timezone.utc).strftime("v%Y%m%dT%H%M%SZ")
        version_dir=out/version;version_dir.mkdir()
        progress={"status":"RUNNING","pid":os.getpid(),"version":version,"output_directory":str(version_dir),"started_at_utc":now()}
        def checkpoint(stage,**extra):
            progress.update(stage=stage,updated_at_utc=now(),**extra);json_write(out/"progress.json",progress)
            print(json.dumps(progress,ensure_ascii=True),flush=True)
        checkpoint("inventory")
        codes,old,inventory,variants=read_inventory(resource)
        by_isin,company_names,registry_info=registry(resource);relevant=relevant_companies()
        json_write(version_dir/"source_inventory.json",inventory)
        csv_write(version_dir/"holdings_code_inventory.csv",[{"raw_code":raw,"normalized_code":code,"rows":n} for (raw,code),n in variants.items()])
        cache=ISSCache(out/"cache",args.delay,args.offline)
        evidence={};details={};failures=[];sources=[];changeovers=[]
        def request(url,label):
            try:
                obj,meta=cache.get(url);sources.append(meta);return obj,meta
            except Exception as exc:
                failures.append({"stage":label,"url":url,"error":str(exc)})
                json_write(version_dir/"source_failures_checkpoint.json",failures)
                return None,None
        # Highest-frequency examples first, followed by a complete paginated catalog.
        for key in PRIORITY:
            url=iss_url("/iss/securities.json",**{"iss.meta":"off","q":key,"limit":100})
            obj,meta=request(url,"priority_exact_code")
            if obj:add_candidates(evidence,obj,meta,codes)
        checkpoint("catalog",candidate_rows=len(evidence),requests=cache.requests)
        catalog_complete=False;catalog_rows=0;catalog_seen=set()
        for page in range(args.max_pages):
            url=iss_url("/iss/securities.json",**{"iss.meta":"off","engine":"stock","limit":100,"start":page*100})
            obj,meta=request(url,"stock_catalog")
            if obj is None:break
            records=table(obj,"securities")
            if not records:catalog_complete=True;break
            fingerprint=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
            if fingerprint in catalog_seen:
                failures.append({"stage":"stock_catalog","error":"Repeated page detected; pagination stopped"});break
            catalog_seen.add(fingerprint);catalog_rows+=len(records);add_candidates(evidence,obj,meta,codes)
            if page%20==0:checkpoint("catalog",catalog_pages=page+1,catalog_rows=catalog_rows,candidate_rows=len(evidence),requests=cache.requests)
        # Current market tables supply exact identity rows and scheduled bond dates.
        for market in ("shares","bonds"):
            url=iss_url(f"/iss/engines/stock/markets/{market}/securities.json",**{"iss.only":"securities","securities.columns":"SECID,REGNUMBER,ISIN,MATDATE,SECNAME,SECTYPE,FACEUNIT"})
            obj,meta=request(url,"market_metadata")
            if obj:add_candidates(evidence,obj,meta,codes,source_kind="iss_market_snapshot")
        url=iss_url("/iss/history/engines/stock/markets/shares/securities/changeover.json",**{"iss.meta":"off"})
        obj,meta=request(url,"technical_ticker_changes")
        if obj:changeovers=[dict(row,source_url=meta["url"],source_sha256=meta["sha256"],observed_at_utc=meta["observed_at_utc"]) for row in table(obj,"changeover")]
        matched={r["registration_code"] for r in evidence.values()}
        targets=sorted([k for k in codes if k and not direct_isins(k) and k not in matched and re.fullmatch(r"[0-9A-ZА-ЯЁ-]{5,90}",k)],key=lambda k:(-codes[k]["rows"],k))
        search_targets=targets[:args.search_limit] if args.search_limit else targets
        checkpoint("unmatched_exact_queries",catalog_complete=catalog_complete,catalog_rows=catalog_rows,unmatched_queries=len(search_targets),candidate_rows=len(evidence),requests=cache.requests)
        for i,key in enumerate(search_targets):
            # q performs candidate retrieval; acceptance still requires the exact regnumber.
            start=0
            while True:
                url=iss_url("/iss/securities.json",**{"iss.meta":"off","q":key,"limit":100,"start":start})
                obj,meta=request(url,"unmatched_exact_query")
                if obj is None:break
                records=table(obj,"securities");add_candidates(evidence,obj,meta,codes)
                if len(records)<100:break
                start+=100
                if start>=1000:
                    failures.append({"stage":"unmatched_exact_query","code":key,"error":"Candidate-query pagination guard reached"});break
            if i%50==0:checkpoint("unmatched_exact_queries",queries_completed=i+1,queries_total=len(search_targets),candidate_rows=len(evidence),requests=cache.requests)
        checkpoint("unmatched_queries_finished",queries_completed=len(search_targets),queries_total=len(search_targets),candidate_rows=len(evidence),requests=cache.requests)
        # Save exact matches before optional richer card retrieval.
        prelim,conflicts=consolidate(evidence,details,by_isin,codes,changeovers)
        json_write(version_dir/"crosswalk_checkpoint.json",prelim)
        selected_secids={r["secid"] for r in evidence.values() if r["secid"]}
        scores={s:sum(codes[k]["rows"] for k in {r["registration_code"] for r in evidence.values() if r["secid"]==s}) for s in selected_secids}
        ordered=sorted(selected_secids,key=lambda s:(-scores[s],s))
        if args.detail_limit:ordered=ordered[:args.detail_limit]
        checkpoint("security_cards",cards_total=len(ordered),distinct_matched_codes=len(prelim),requests=cache.requests)
        for i,secid in enumerate(ordered):
            url=iss_url("/iss/securities/"+urllib.parse.quote(secid,safe="")+".json",**{"iss.meta":"off"})
            obj,meta=request(url,"security_card")
            if obj:details[secid]=(obj,meta)
            if i%50==0:checkpoint("security_cards",cards_completed=i+1,cards_total=len(ordered),requests=cache.requests)
        crosswalk,conflicts=consolidate(evidence,details,by_isin,codes,changeovers)
        checkpoint("coverage",distinct_matched_codes=len(crosswalk),cards_completed=len(ordered),cards_retrieved=len(details),requests=cache.requests)
        cov=coverage(resource,old,crosswalk,by_isin,relevant)
        unresolved=[{"registration_code_normalized":k,"raw_codes":sorted(v["raw_codes"]),"holdings_rows":v["rows"],"strict_equity_rows":v["equity_rows"],"examples":sorted(v["examples"]),"reason":"no_exact_public_match" if k in search_targets else "direct_isin_or_blank_or_not_queried"} for k,v in codes.items() if k not in {r['registration_code_normalized'] for r in crosswalk} and not direct_isins(k)]
        json_write(version_dir/"crosswalk.json",crosswalk);csv_write(version_dir/"crosswalk.csv",crosswalk)
        json_write(version_dir/"conflicts.json",conflicts);csv_write(version_dir/"unresolved_codes.csv",sorted(unresolved,key=lambda r:-r["holdings_rows"]))
        json_write(version_dir/"temporal_review_queue.json",[r for r in crosswalk if r["temporal_review_issues"]])
        json_write(version_dir/"evidence_index.json",list(evidence.values()));json_write(version_dir/"source_requests.json",list({r['url']:r for r in sources}.values()))
        json_write(version_dir/"technical_ticker_changes.json",changeovers);json_write(version_dir/"coverage.json",cov)
        # Typed derivative for the regression executor. Structured evidence remains in JSON.
        import pyarrow as pa
        import pyarrow.parquet as pq
        flat=[{k:(json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(dict,list)) else v) for k,v in row.items()} for row in crosswalk]
        if flat:pq.write_table(pa.Table.from_pylist(flat),version_dir/"crosswalk.parquet")
        accepted=[r for r in crosswalk if r['mapping_status']=='EXACT_CURRENT_CONFIRMED']
        collection_complete=catalog_complete and len(search_targets)==len(targets) and len(ordered)==len(selected_secids) and not failures
        report={"task":"W2 exact registration-code-to-ISIN enrichment","status":"PARTIAL_SOURCE_ERRORS" if failures else "COMPLETE_WITH_LIMITS" if collection_complete else "PARTIAL_SELECTION_LIMITS",
            "collection_complete":collection_complete,
            "observed_at_utc":now(),"version":version,"output_directory":str(version_dir),"builder_source":stamp(__file__),"source_inventory":inventory,"registry":registry_info,
            "normalization":"Unicode NFKC, uppercase, whitespace removal, dash variants only; no suffix stripping, name matching or Cyrillic-to-Latin conversion.",
            "catalog":{"complete":catalog_complete,"rows":catalog_rows,"exact_search_targets":len(targets),"exact_searches_attempted":len(search_targets),"cards_available":len(selected_secids),"cards_attempted":len(ordered),"cards_retrieved":len(details)},
            "counts":{"registration_codes_with_evidence":len(crosswalk),"exact_unambiguous_current_codes":len(accepted),"identity_conflict_codes":sum(r['mapping_status']=='CONFLICT' for r in crosswalk),"ambiguous_static_company_codes":sum(r['company_join_status']=='AMBIGUOUS' for r in crosswalk),"dated_fact_codes":sum(bool(r['dated_source_facts']) for r in accepted),"temporal_registration_timing_issue_codes":sum(bool(r['temporal_review_issues']) for r in crosswalk),"historically_validated_full_interval_codes":0},
            "coverage":cov,"priority_codes":[r for r in crosswalk if r['registration_code_normalized'] in PRIORITY],
            "failures":failures,"network":{"requests":cache.requests,"cache_hits":cache.hits,"minimum_interval_seconds":args.delay,"retries":3},
            "temporal_gate":"Current exact identifier links and source-labelled dated facts are usable as explicit exploratory evidence. No full historical issuer-identity interval has been asserted. Ticker change dates are not registration-number change dates; maturity dates are not proof of actual redemption.",
            "preservation":{"canonical_inputs_written":False,"live_llm_written":False,"W1_modified":False,"regression_code_or_spec_modified":False},
            "artifacts":[stamp(p) for p in sorted(version_dir.iterdir()) if p.is_file()]}
        json_write(version_dir/"summary.json",report);json_write(out/"latest_summary.json",report)
        checkpoint("finished",status=report["status"],summary_path=str(version_dir/"summary.json"),counts=report["counts"],coverage_before=cov["before"]["counts"],coverage_after=cov["after"]["counts"])
        return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resource-root",type=Path,default=RESOURCE);p.add_argument("--output-root",type=Path,default=DEFAULT_OUTPUT)
    p.add_argument("--delay",type=float,default=.65);p.add_argument("--offline",action="store_true")
    p.add_argument("--max-pages",type=int,default=1000)
    p.add_argument("--detail-limit",type=int,default=0,help="0 means all matching security cards")
    p.add_argument("--search-limit",type=int,default=0,help="0 means all syntactically plausible unmatched codes")
    args=p.parse_args()
    if args.delay<.5:raise ValueError("Minimum request interval is 0.5 seconds")
    build(args)


if __name__=="__main__":main()
