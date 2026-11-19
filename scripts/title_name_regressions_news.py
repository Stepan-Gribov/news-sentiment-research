"""Frozen-input news indexing, conservative event duplicates, monthly states."""
from __future__ import annotations

from collections import deque
import hashlib
import json
import re
import sqlite3
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from title_name_regressions_core import IntegrityError, aggregate_news_windows, unique_keys
from title_name_regressions_integrity import APPROVED_RESULT_MODEL,APPROVED_PROMPT_VERSION


def pair_identifier(article,company):
    return "p_"+hashlib.blake2b(f"{article}\t{company}".encode(),digest_size=16).hexdigest()


def input_index(path,manifest,timezone="Europe/Moscow"):
    digest=hashlib.sha256()
    rows=[]
    with Path(path).open("rb") as source:
        offset=0
        for line in source:
            digest.update(line)
            row=json.loads(line)
            if row.get("pair_id")!=pair_identifier(row["article_id"],row["company_id"]):
                raise IntegrityError("Frozen input contains an inconsistent pair ID")
            if row.get("match_signal")!="name" or row.get("match_scope")!="title":
                raise IntegrityError("Frozen input is not strictly title/name mapping")
            if row["company_id"]=="c_90995d77be411a67f2ef2b72":
                raise IntegrityError("Excluded GAZ ID found in frozen input")
            rows.append({k:row.get(k) for k in ["pair_id","article_id","company_id","source","published_at",
                                               "title","article_url","input_issue","text_sha256"]}
                        | dict(input_offset=offset,input_line_bytes=len(line),
                               date_source=row.get("date_source"),date_inferred=row.get("date_inferred",False)))
            offset+=len(line)
    if digest.hexdigest()!=manifest["input_sha256"] or offset!=manifest["input_bytes"]:
        raise IntegrityError("Frozen input content does not match its manifest")
    out=pd.DataFrame(rows)
    unique_keys(out,["pair_id"],"frozen input")
    unique_keys(out,["article_id","company_id"],"frozen input article/company")
    if len(out)!=manifest["total"]:
        raise IntegrityError("Frozen input row count differs from manifest")
    out["event_timestamp"]=pd.to_datetime(out.published_at,errors="coerce",utc=True).dt.tz_convert(timezone)
    out["event_date"]=out.event_timestamp.dt.tz_localize(None).dt.normalize()
    out["date_provenance_unknown"]=out.date_source.isna()
    return out


def read_labels(path,index,*,expected_logical_sha256=None,expected_model=APPROVED_RESULT_MODEL,
                expected_prompt_version=APPROVED_PROMPT_VERSION):
    c=sqlite3.connect(Path(path).as_uri()+"?mode=ro",uri=True,timeout=30)
    c.execute("PRAGMA query_only=ON")
    c.execute("BEGIN")
    rows=[];logical=hashlib.sha256()
    try:
        for pid,task,valid,body in c.execute("SELECT pair_id,task_index,valid,result_json FROM results ORDER BY task_index"):
            logical.update(json.dumps([pid,task,valid,body],ensure_ascii=False,separators=(",",":")).encode()+b"\n")
            r=json.loads(body)
            if r.get("pair_id")!=pid or bool(r.get("valid"))!=bool(valid):
                raise IntegrityError("SQLite ledger fields disagree with JSON payload")
            if valid or not r.get("input_issue"):
                if r.get("prompt_version")!=expected_prompt_version or r.get("model")!=expected_model:
                    raise IntegrityError("Result prompt/model identity differs from the approved audited classifier")
            rows.append(dict(pair_id=pid,task_index=task,attempted=True,valid=bool(valid),
                relevance=r.get("relevance") if valid else None,
                sentiment_label=r.get("sentiment_label") if valid else None,
                output_article_id=r.get("article_id"),output_company_id=r.get("company_id"),
                output_text_sha256=r.get("text_sha256"),output_prompt_version=r.get("prompt_version"),
                output_model=r.get("model"),
                output_input_issue=r.get("input_issue"),text_truncated=r.get("text_truncated")))
    finally:
        c.rollback()
        c.close()
    if expected_logical_sha256 is not None and logical.hexdigest()!=expected_logical_sha256:
        raise IntegrityError("Classifications differ from the exact WAL-aware logical W1 snapshot; a new audit is required")
    labels=pd.DataFrame(rows)
    unique_keys(labels,["pair_id"],"LLM ledger")
    unique_keys(labels,["task_index"],"LLM task index")
    if set(labels.pair_id)!=set(index.pair_id):
        raise IntegrityError("LLM/frozen-input pair sets are not identical")
    out=index.merge(labels,on="pair_id",validate="one_to_one")
    if (out.article_id.ne(out.output_article_id)|out.company_id.ne(out.output_company_id)).any():
        raise IntegrityError("LLM article/company metadata disagrees with frozen input")
    bad_text=out.output_text_sha256.notna() & out.text_sha256.ne(out.output_text_sha256)
    if bad_text.any():
        raise IntegrityError("LLM full text hash differs from frozen input")
    if (out.valid & ~out.relevance.isin(["relevant","irrelevant"])).any():
        raise IntegrityError("Valid LLM output has invalid relevance")
    relevant=out.valid & out.relevance.eq("relevant")
    if (relevant & ~out.sentiment_label.isin(["positive","neutral","negative"])).any():
        raise IntegrityError("Relevant valid output has invalid sentiment")
    if (out.valid & out.relevance.eq("irrelevant") & out.sentiment_label.notna()).any():
        raise IntegrityError("Irrelevant output has non-null sentiment")
    out["pair_state"]=np.select(
        [~out.valid & out.input_issue.notna(),~out.valid,out.relevance.eq("irrelevant")],
        ["input_error","classification_error","irrelevant"],
        default="relevant_")
    out.loc[relevant,"pair_state"]="relevant_"+out.loc[relevant,"sentiment_label"]
    out.attrs["logical_results_sha256"]=logical.hexdigest()
    return out


def normalize_words(text):
    normalized=unicodedata.normalize("NFKC",str(text or "")).lower().replace("ё","е")
    return re.findall(r"\w+",normalized,flags=re.UNICODE)


def _similarity(left,right):
    if not left or not right: return 0.0
    shared=len(left.intersection(right))
    return shared/(len(left)+len(right)-shared)


def collapse_events(candidates,input_path):
    """72-hour representative clusters without label-dependent replacement.

    Empty unavailable texts are singletons: equal absence is not evidence of an
    identical archived event. Original input bytes/text are never changed.
    """
    ordered=candidates.sort_values(["company_id","event_timestamp","article_id"],kind="mergesort",na_position="last")
    membership=[]
    with Path(input_path).open("rb") as source:
        for company,group in ordered.groupby("company_id",sort=False,observed=True):
            active=deque()
            for row in group.itertuples(index=False):
                source.seek(int(row.input_offset))
                raw=json.loads(source.readline())
                if raw["pair_id"]!=row.pair_id:
                    raise IntegrityError("Frozen input offset no longer identifies the expected pair")
                words=normalize_words(raw.get("text"))
                title=set(normalize_words(row.title))
                text_hash=hashlib.sha256(" ".join(words).encode()).hexdigest() if words else None
                stamp=row.event_timestamp
                if pd.notna(stamp):
                    while active and (stamp-active[0]["timestamp"]).total_seconds()>72*3600:
                        active.popleft()
                shingle=None
                matches=[]
                if words and pd.notna(stamp):
                    for representative in active:
                        if text_hash==representative["text_hash"]:
                            matches.append((1.0,representative["timestamp"],representative["article_id"],representative,"exact",1.0))
                            continue
                        if len(words)<50 or len(representative["words"])<50: continue
                        title_similarity=_similarity(title,representative["title"])
                        if title_similarity<.8: continue
                        if shingle is None: shingle=set(zip(words,words[1:],words[2:],words[3:],words[4:]))
                        if representative["shingles"] is None:
                            w=representative["words"]
                            representative["shingles"]=set(zip(w,w[1:],w[2:],w[3:],w[4:]))
                        other=representative["shingles"]
                        if min(len(shingle),len(other))/max(len(shingle),len(other),1)<.9: continue
                        text_similarity=_similarity(shingle,other)
                        if text_similarity>=.9:
                            matches.append((text_similarity,representative["timestamp"],representative["article_id"],representative,"near",title_similarity))
                if matches:
                    best=sorted(matches,key=lambda x:(-x[0],x[1],x[2]))[0]
                    representative=best[3]
                    membership.append(dict(pair_id=row.pair_id,cluster_id=representative["cluster_id"],
                        representative_pair_id=representative["pair_id"],representative_article_id=representative["article_id"],
                        is_representative=False,duplicate_method=best[4],text_similarity=best[0],title_similarity=best[5]))
                else:
                    cluster="ev_"+hashlib.blake2b(f"{company}\t{row.article_id}".encode(),digest_size=16).hexdigest()
                    membership.append(dict(pair_id=row.pair_id,cluster_id=cluster,representative_pair_id=row.pair_id,
                        representative_article_id=row.article_id,is_representative=True,
                        duplicate_method="representative" if words else "missing_text_singleton",
                        text_similarity=None,title_similarity=None))
                    if pd.notna(stamp):
                        active.append(dict(cluster_id=cluster,pair_id=row.pair_id,article_id=row.article_id,
                            timestamp=stamp,text_hash=text_hash,words=words,title=title,shingles=None))
    audit=candidates.merge(pd.DataFrame(membership),on="pair_id",validate="one_to_one")
    conflicts=audit.groupby("cluster_id",observed=True).pair_state.nunique().gt(1)
    audit["conflicting_member_labels"]=audit.cluster_id.map(conflicts)
    representatives=audit.loc[audit.is_representative].copy()
    return representatives,audit


def monthly_states(candidates,companies,start,end,*,allow_partial=False,minimum_valid_fraction=1.0,boundary_exclusion=False):
    months=pd.period_range(start,end,freq="M")
    grid=pd.MultiIndex.from_product([sorted(set(companies)),months.astype(str)],names=["company_id","month"]).to_frame(index=False)
    grid["interval_id"]=grid.month
    month_index=pd.PeriodIndex(grid.month,freq="M")
    grid["news_start"]=(month_index-1).to_timestamp(how="end").normalize()
    grid["news_end"]=month_index.to_timestamp(how="end").normalize()
    c=candidates.copy()
    if boundary_exclusion:
        dates=pd.to_datetime(c.event_date)
        c=c.loc[~dates.dt.is_month_end].copy()
    result,audit=aggregate_news_windows(grid,c,allow_partial=allow_partial,minimum_valid_fraction=minimum_valid_fraction)
    dated=c.loc[c.event_date.notna()].copy()
    dated["month"]=pd.to_datetime(dated.event_date).dt.to_period("M").astype(str)
    relevant=dated.valid.eq(True)&dated.relevance.eq("relevant")
    counts=dated.loc[relevant].groupby(["company_id","month","source"],observed=True).size().unstack("source",fill_value=0)
    for source in ["interfax","rbc","prime","kommersant"]:
        values=counts[source] if source in counts else pd.Series(dtype=float)
        source_frame=values.rename("source_count_"+source).reset_index()
        if not source_frame.empty:
            result=result.merge(source_frame,on=["company_id","month"],how="left",validate="one_to_one")
        else:
            result["source_count_"+source]=0
        result["source_count_"+source]=result["source_count_"+source].fillna(0).astype(int)
        result["source_share_"+source]=result["source_count_"+source].div(result.n_relevant.replace(0,np.nan))
    observed=result.n_valid.eq(result.n_expected)
    if allow_partial:
        observed |= result.valid_fraction.ge(minimum_valid_fraction)
    result["log_news_count_observed"]=np.log1p(result.n_relevant).where(observed)
    previous=result[["company_id","month","Tone","log_news_count_observed"]].copy()
    previous["month"]=(pd.PeriodIndex(previous.month,freq="M")+1).astype(str)
    previous=previous.rename(columns={"Tone":"lag_Tone","log_news_count_observed":"lag_log_news_count"})
    result=result.merge(previous,on=["company_id","month"],how="left",validate="one_to_one")
    return result,audit
