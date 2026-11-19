#!/usr/bin/env python3
"""Independent W1 audit. Read immutable sources and live SQLite in read-only mode.

No model, regression, service operation, source rewrite, or bulk copy is made.
Only a compact JSON report is written, outside databaseresources and the run.
Run on research-host with its existing pyarrow/duckdb environment. --probe reports
schemas first. Error states always remain distinct from valid neutral/no-news.
"""
from __future__ import annotations

import argparse
import ast
import calendar
import csv
import gzip
import hashlib
import json
import math
import re
import sqlite3
import sys
import time
import threading
from contextlib import closing
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(".")
RESOURCE = ROOT / "databaseresources"
RUN = ROOT / "news_data/llm_runs/gemma_title_name_no_gaz_20261008_01"
CATALOG = Path("./local/research-data-service/research.duckdb")
GAZ = "c_90995d77be411a67f2ef2b72"
ALLOWED_MEDIA = {"kommersant", "prime", "interfax", "rbc"}
ISIN_RE = re.compile(r"\b[A-Z]{2}[A-Z0-9]{9}[0-9]\b")
MISSING = {"", "none", "null", "nan", "nat", "<na>"}
EQUITY_CATEGORIES = {"акции", "акции российских эмитентов", "акции иностранных эмитентов"}
SQL = {
    "sqlite_schema": "SELECT name, sql FROM sqlite_master WHERE type IN ('table','view')",
    "sqlite_results": "SELECT pair_id,task_index,valid,result_json,elapsed_seconds FROM results ORDER BY task_index",
    "sqlite_counts": "SELECT count(*) n,count(DISTINCT pair_id) pair_ids,count(DISTINCT task_index) task_indexes,min(task_index) min_index,max(task_index) max_index,sum(valid=1) valid FROM results",
    "catalog_sources": "SELECT * FROM meta.sources",
    "catalog_views": "SELECT schema_name,view_name,sql FROM duckdb_views() WHERE schema_name IN ('llm','funds','refs','market')",
    "catalog_tables": "SELECT table_schema,table_name,column_name,data_type FROM information_schema.columns WHERE table_schema IN ('llm','funds','refs','market') ORDER BY 1,2,ordinal_position",
    "catalog_live": "SELECT count(*) n,count(*) FILTER (WHERE valid) valid,max(task_index) last_task FROM llm.sentiment_results_live",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def stable_pair_id(article_id, company_id):
    return "p_" + hashlib.blake2b(f"{article_id}\t{company_id}".encode(), digest_size=16).hexdigest()


def text(value):
    return "" if value is None else str(value).strip()


def code(value):
    value = text(value).upper()
    return "" if value.lower() in MISSING else re.sub(r"\s+", "", value)


def parse_date(value):
    value = text(value)
    if value.lower() in MISSING:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def date_key(value):
    value = parse_date(value)
    return value.isoformat() if value else None


def timestamp_key(value):
    """Compare stored times without inferring or converting an absent timezone."""
    value = text(value)
    if value.lower() in MISSING:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return None


def number(value):
    try:
        result = float(str(value).replace(" ", "").replace(",", "."))
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def file_info(path, digest=False):
    path = Path(path)
    out = {"path": str(path), "exists": path.is_file()}
    if out["exists"]:
        stat = path.stat()
        out.update(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        if digest:
            h = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    h.update(chunk)
            out["sha256"] = h.hexdigest()
    return out


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path, columns=None):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq
        parquet = pq.ParquetFile(path)
        selected = [c for c in columns if c in parquet.schema_arrow.names] if columns else None
        for batch in parquet.iter_batches(columns=selected, batch_size=20000):
            yield from batch.to_pylist()
    elif path.name.lower().endswith((".csv", ".csv.gz")):
        csv.field_size_limit(64 * 1024 * 1024)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                yield {k: row.get(k) for k in columns} if columns else row
    else:
        raise ValueError(f"Unsupported table type: {path.name}")


def schema(path):
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        p = pq.ParquetFile(path)
        return {"rows": p.metadata.num_rows, "columns": p.schema_arrow.names,
                "types": {f.name: str(f.type) for f in p.schema_arrow}}
    first = next(rows(path), {})
    return {"columns": list(first), "rows": None}


def sample(values, n=20):
    return sorted(values, key=str)[:n]


def read_only_sqlite(path):
    connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=20)
    connection.execute("PRAGMA query_only=ON")
    connection.execute("BEGIN")
    return connection


def outcome(row, stored_valid):
    if stored_valid not in (0, 1) or not isinstance(row.get("valid"), bool) or bool(stored_valid) != row.get("valid"):
        return "schema_error"
    if row.get("input_issue"):
        return "schema_error" if stored_valid else "input_error"
    if not bool(stored_valid):
        return "generation_error"
    relevance, label = row.get("relevance"), row.get("sentiment_label")
    if relevance not in {"relevant", "irrelevant"}:
        return "schema_error"
    if relevance == "relevant" and label not in {"positive", "neutral", "negative"}:
        return "schema_error"
    if relevance == "irrelevant" and label is not None:
        return "schema_error"
    return "valid_relevant" if relevance == "relevant" else "valid_irrelevant"


def distribution(groups):
    return {name: dict(sorted(counter.items(), key=lambda x: str(x[0]))) for name, counter in groups.items()}


def catalog_probe(path, include_live_count=False):
    result = {"path": str(path), "access": "read_only", "status": "BLOCKED"}
    try:
        import duckdb
        con = duckdb.connect(str(path), read_only=True)
        try:
            for label, key in (("sources", "catalog_sources"), ("views", "catalog_views"), ("schema", "catalog_tables")):
                try:
                    cur = con.execute(SQL[key])
                    names = [d[0] for d in cur.description]
                    result[label] = [dict(zip(names, row)) for row in cur.fetchall()]
                except Exception as exc:
                    result[label + "_error"] = f"{type(exc).__name__}: {exc}"
            if include_live_count:
                try:
                    cur = con.execute(SQL["catalog_live"])
                    result["live_counts"] = dict(zip([d[0] for d in cur.description], cur.fetchone()))
                except Exception as exc:
                    result["live_query_error"] = f"{type(exc).__name__}: {exc}"
            else:
                result["live_count"] = "Not requested: direct SQLite audit precedes any expensive live-view scan."
            result["status"] = "READY" if "sources" in result and "schema" in result else "PARTIAL"
        finally:
            con.close()
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def inventory(resource):
    paths = [resource / "news/articles.parquet",
             resource / "mapping/title-name-20261008-no-gaz/article_company_map_title_name.parquet",
             resource / "mapping/title-name-20261008-no-gaz/company_registry.parquet",
             resource / "mapping/active-20261008-safe-otkritie/cleaned_single_industry/company_registry_active.parquet"]
    for folder in (resource / "funds/mac-20261008", resource / "market/moex_quotes_2008_2026"):
        if folder.is_dir():
            paths.extend(sorted(p for p in folder.iterdir() if p.is_file() and p.name.endswith((".csv", ".csv.gz", ".parquet", ".json"))))
    result = []
    for p in dict.fromkeys(paths):
        info = file_info(p)
        if p.is_file() and p.name.endswith((".csv", ".csv.gz", ".parquet")):
            try:
                info.update(schema(p))
            except Exception as exc:
                info["schema_error"] = f"{type(exc).__name__}: {exc}"
        result.append(info)
    manifest = resource / "manifest.json"
    meta = file_info(manifest, digest=True)
    if manifest.is_file():
        value = read_json(manifest)
        meta["top_level_keys"] = list(value) if isinstance(value, dict) else None
        candidates = set()
        def visit(item):
            if isinstance(item, dict):
                for key, val in item.items():
                    if isinstance(val, str):
                        if re.search(r"security|securities|isin|dividend|coupon|redemption|factor|total.?return|fund.?price|unit.?price|risk.?free|nav", val, re.I) and re.search(r"\.(csv|parquet|json|xlsx?)(\.gz)?$", val, re.I):
                            candidates.add(val)
                    elif isinstance(val, (list, dict)):
                        visit(val)
            elif isinstance(item, list):
                for val in item:
                    visit(val)
        visit(value)
        meta["relevant_paths"] = sorted(candidates)
    return result, meta


def audit_llm(resource, run, project_root):
    map_path = resource / "mapping/title-name-20261008-no-gaz/article_company_map_title_name.parquet"
    fields = ["article_id", "company_id", "media", "date", "article_url", "match_signal", "match_scope"]
    expected, map_counts, map_groups = {}, Counter(), defaultdict(Counter)
    for row in rows(map_path, fields):
        aid, cid = text(row["article_id"]), text(row["company_id"])
        pid = stable_pair_id(aid, cid)
        map_counts["rows"] += 1
        map_counts["duplicate_pair_ids"] += pid in expected
        map_counts["missing_article_or_company_id"] += not aid or not cid
        map_counts["gaz_pairs"] += cid == GAZ
        map_counts["non_name_signal"] += row.get("match_signal") != "name"
        map_counts["non_title_scope_if_present"] += row.get("match_scope") not in (None, "title")
        map_counts["missing_date"] += parse_date(row.get("date")) is None
        year = parse_date(row.get("date"))
        map_groups["source"][text(row.get("media"))] += 1
        map_groups["year"][str(year.year) if year else "missing"] += 1
        map_groups["company_id"][cid] += 1
        expected[pid] = (aid, cid, text(row.get("media")), timestamp_key(row.get("date")))
    expected_articles = {row[0] for row in expected.values()}
    expected_companies = {row[1] for row in expected.values()}
    report = {"mapping": {**file_info(map_path, True), "schema": schema(map_path),
                          "counts": dict(map_counts), "unique_articles": len(expected_articles),
                          "unique_companies": len(expected_companies), "distributions": distribution(map_groups)},
              "sample_flow": [], "read_started_at_utc": utc_now()}
    for name in ("input_manifest.json", "run_manifest.json", "progress.json", "summary.json", "engine_config.json", "supervisor_status.json"):
        path = run / name
        if path.is_file():
            report[name] = {"file": file_info(path, True), "content": read_json(path)}
    report["markers"] = {name: file_info(run / name) for name in ("INPUT_SUCCESS", "SUCCESS", "COMPLETE_WITH_ERRORS")}
    runner_candidates = [run / "scripts/68_run_gemma_title_full.py", project_root / "scripts/68_run_gemma_title_full.py"]
    runner = next((p for p in runner_candidates if p.is_file()), runner_candidates[0])
    report["runner"] = file_info(runner, True)
    report["runner"]["checked_paths"] = [str(p) for p in runner_candidates]
    if runner.exists():
        source = runner.read_text(encoding="utf-8-sig")
        pair_node = next((node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "pair_id"), None)
        report["runner"]["pair_id_source"] = ast.get_source_segment(source, pair_node) if pair_node else None
    counts, groups, db_index, outcomes = Counter(), defaultdict(Counter), {}, {}
    seen_index, result_articles, result_companies, relevant_companies = set(), set(), set(), set()
    dated_relevant_companies, relevant_articles = set(), set()
    errors, record_hash = defaultdict(list), hashlib.sha256()
    db_path = run / "results.sqlite3"
    report["sqlite"] = {"before": file_info(db_path), "wal_before": file_info(Path(str(db_path) + "-wal")),
                        "mode": "ro; query_only=ON; consistent read transaction; WAL-aware; no immutable=1"}
    with closing(read_only_sqlite(db_path)) as con:
        report["sqlite"]["schema"] = [dict(name=n, sql=s) for n, s in con.execute(SQL["sqlite_schema"])]
        cur = con.execute(SQL["sqlite_counts"])
        report["sqlite"]["aggregate"] = dict(zip([d[0] for d in cur.description], cur.fetchone()))
        for pid, index, valid, body, elapsed in con.execute(SQL["sqlite_results"]):
            counts["results"] += 1
            counts["duplicate_task_index"] += index in seen_index
            counts["duplicate_pair_id"] += pid in db_index
            seen_index.add(index)
            record_hash.update(json.dumps([pid, index, valid, body], ensure_ascii=False, separators=(",", ":")).encode() + b"\n")
            try:
                row = json.loads(body)
                if not isinstance(row, dict):
                    raise ValueError("Result JSON is not an object")
            except (ValueError, TypeError):
                counts["malformed_result_json"] += 1
                db_index[pid] = (index, None, None, None, None, None, None, None)
                continue
            aid, cid = text(row.get("article_id")), text(row.get("company_id"))
            result_articles.add(aid)
            result_companies.add(cid)
            counts["pair_hash_mismatch"] += stable_pair_id(aid, cid) != pid
            counts["json_pair_id_mismatch"] += row.get("pair_id") != pid
            counts["valid_column_json_mismatch"] += bool(valid) != row.get("valid")
            counts["non_title_scope"] += row.get("match_scope") != "title"
            counts["non_name_signal"] += row.get("match_signal") != "name"
            counts["gaz_pairs"] += cid == GAZ
            counts["disallowed_source"] += row.get("source") not in ALLOWED_MEDIA
            expected_row = expected.get(pid)
            if expected_row:
                counts["mapping_identity_mismatch"] += (aid, cid, text(row.get("source"))) != expected_row[:3]
                counts["mapping_date_mismatch"] += timestamp_key(row.get("published_at")) != expected_row[3]
            status = outcome(row, valid)
            counts[status] += 1
            outcomes[pid] = status
            if status == "valid_relevant":
                relevant_companies.add(cid)
                relevant_articles.add(aid)
            if status not in ("valid_relevant", "valid_irrelevant") and len(errors[status]) < 20:
                errors[status].append(pid)
            published = parse_date(row.get("published_at"))
            if status == "valid_relevant" and published:
                counts["valid_relevant_with_date"] += 1
                dated_relevant_companies.add(cid)
            counts["missing_or_invalid_date"] += published is None
            counts["date_before_2000"] += bool(published and published.year < 2000)
            counts["date_after_audit_day"] += bool(published and published > date.today())
            counts["truncated_true"] += row.get("text_truncated") is True
            counts["truncation_field_absent"] += "text_truncated" not in row
            counts["used_retry"] += row.get("used_retry") is True
            counts["rationale_longer_than_12_words"] += len(text(row.get("short_rationale")).split()) > 12
            if not row.get("input_issue"):
                counts["model_mismatch"] += row.get("model") != "gemma4_e4b_q4qat"
                counts["prompt_version_mismatch"] += row.get("prompt_version") != report.get("run_manifest.json",{}).get("content",{}).get("prompt_version")
            groups["source"][text(row.get("source"))] += 1
            groups["year"][str(published.year) if published else "missing"] += 1
            groups["company_id"][cid] += 1
            groups["outcome"][status] += 1
            groups["relevance"][str(row.get("relevance"))] += 1
            groups["polarity_valid_relevant"][row.get("sentiment_label") if status == "valid_relevant" else "not_eligible"] += 1
            groups["input_issue"][str(row.get("input_issue"))] += 1
            groups["outcome_by_source"][f"{text(row.get('source'))}|{status}"] += 1
            groups["outcome_by_year"][f"{published.year if published else 'missing'}|{status}"] += 1
            groups["outcome_by_company"][f"{cid}|{status}"] += 1
            groups["relevant_polarity_by_source"][f"{text(row.get('source'))}|{row.get('sentiment_label') if status == 'valid_relevant' else 'not_eligible'}"] += 1
            groups["model"][str(row.get("model"))] += 1
            groups["prompt_version"][str(row.get("prompt_version"))] += 1
            db_index[pid] = (index, aid, cid, row.get("input_issue"), row.get("text_sha256"), row.get("input_sha256"), row.get("input_characters"), row.get("text_truncated"))
    report["sqlite"].update(after=file_info(db_path), wal_after=file_info(Path(str(db_path) + "-wal")),
                            logical_results_sha256=record_hash.hexdigest())
    missing, extra = set(expected) - set(db_index), set(db_index) - set(expected)
    counts["missing_result_pair_ids"] = len(missing)
    counts["extra_result_pair_ids"] = len(extra)
    counts["task_index_outside_input_range"] = sum(not isinstance(i, int) or not 1 <= i <= len(expected) for i in seen_index)
    counts["task_index_missing_in_expected_range"] = len(set(range(1, len(expected) + 1)) - seen_index)
    input_seen, input_hash, input_counts = set(), hashlib.sha256(), Counter()
    input_path = run / "input.jsonl"
    with input_path.open("rb") as handle:
        for index, raw in enumerate(handle, 1):
            input_hash.update(raw)
            if not raw.strip():
                input_counts["blank_lines"] += 1
                continue
            input_counts["rows"] += 1
            row = json.loads(raw)
            pid, aid, cid = row.get("pair_id"), row.get("article_id"), row.get("company_id")
            input_counts["duplicate_pair_ids"] += pid in input_seen
            input_seen.add(pid)
            input_counts["pair_hash_mismatch"] += stable_pair_id(aid, cid) != pid
            input_counts["non_title_scope"] += row.get("match_scope") != "title"
            input_counts["non_name_signal"] += row.get("match_signal") != "name"
            input_counts["gaz_pairs"] += cid == GAZ
            input_counts["text_sha_mismatch"] += hashlib.sha256(str(row.get("text") or "").encode()).hexdigest() != row.get("text_sha256")
            input_counts["input_errors"] += bool(row.get("input_issue"))
            e = expected.get(pid)
            input_counts["mapping_metadata_mismatch"] += bool(e and (aid, cid, text(row.get("source")), timestamp_key(row.get("published_at"))) != e)
            if pid in db_index:
                input_counts["result_task_index_mismatch"] += index != db_index[pid][0]
                input_counts["result_metadata_mismatch"] += (aid, cid, row.get("input_issue"), row.get("text_sha256")) != db_index[pid][1:5]
                if not row.get("input_issue"):
                    used_hash, used_chars, truncated = db_index[pid][5:]
                    full_text = str(row.get("text") or "")
                    valid_length = isinstance(used_chars, int) and 0 <= used_chars <= len(full_text)
                    input_counts["result_prefix_length_invalid"] += not valid_length
                    if valid_length:
                        input_counts["result_prefix_hash_mismatch"] += hashlib.sha256(full_text[:used_chars].encode()).hexdigest() != used_hash
                        input_counts["result_truncation_flag_mismatch"] += truncated != (used_chars < len(full_text))
    input_counts["missing_mapping_ids"] = len(set(expected) - input_seen)
    input_counts["extra_mapping_ids"] = len(input_seen - set(expected))
    input_digest = input_hash.hexdigest()
    input_manifest = report.get("input_manifest.json", {}).get("content", {})
    run_manifest = report.get("run_manifest.json", {}).get("content", {})
    input_counts["input_manifest_hash_mismatch"] = input_digest != input_manifest.get("input_sha256")
    input_counts["run_manifest_hash_mismatch"] = input_digest != run_manifest.get("input_sha256")
    runner_ok = report["runner"].get("sha256") == run_manifest.get("runner_script_sha256")
    comparison_candidates = [runner.parent / "43_compare_sentiment_models.py", project_root / "scripts/43_compare_sentiment_models.py"]
    comparison = next((p for p in comparison_candidates if p.is_file()), comparison_candidates[0])
    report["comparison_script"] = file_info(comparison, True)
    comparison_ok = report["comparison_script"].get("sha256") == run_manifest.get("comparison_script_sha256")
    if comparison.is_file():
        assignments = {}
        for node in ast.parse(comparison.read_text(encoding="utf-8-sig")).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {"SYSTEM_PROMPT", "PROMPT_VERSION"}:
                        try:
                            assignments[target.id] = ast.literal_eval(node.value)
                        except ValueError:
                            pass
        report["comparison_script"]["prompt_version_matches_manifest"] = assignments.get("PROMPT_VERSION") == run_manifest.get("prompt_version")
        prompt = assignments.get("SYSTEM_PROMPT")
        report["comparison_script"]["prompt_hash_matches_manifest"] = bool(isinstance(prompt, str) and hashlib.sha256(prompt.encode()).hexdigest() == run_manifest.get("prompt_sha256"))
    model_path = run_manifest.get("model",{}).get("path")
    if model_path:
        report["model_file"] = file_info(model_path)
        report["model_file"]["frozen_size_mtime_match"] = all(report["model_file"].get(k) == run_manifest["model"].get(k) for k in ("bytes","mtime_ns"))
        report["model_file"]["content_hash_verified"] = False
    bad = [key for key in counts if any(w in key for w in ("mismatch", "duplicate", "extra_", "missing_result", "malformed", "non_title", "non_name", "gaz_", "disallowed", "schema_error", "outside_")) and counts[key]]
    bad += ["input:" + key for key in input_counts if any(w in key for w in ("mismatch", "duplicate", "extra_", "missing_mapping", "non_title", "non_name", "gaz_", "blank_lines", "invalid")) and input_counts[key]]
    bad += ["mapping:" + key for key in map_counts if key not in ("rows", "missing_date") and map_counts[key]]
    report.update(counts=dict(counts), distributions=distribution(groups),
                  input={**file_info(input_path), "sha256": input_digest, "counts": dict(input_counts)},
                  unique_result_articles=len(result_articles), unique_result_companies=len(result_companies),
                  eligible_relevant_companies=len(relevant_companies),
                  valid_relevant_articles=len(relevant_articles), valid_relevant_dated_companies=len(dated_relevant_companies),
                  eligibility_definition="Valid relevant schema only; this is not the final time-valid regression sample. Missing dates remain excluded from dated counts.",
                  missing_id_examples=sample(missing), extra_id_examples=sample(extra), error_examples=dict(errors),
                  runner_hash_matches_manifest=runner_ok, comparison_hash_matches_manifest=comparison_ok, integrity_failures=bad,
                  status="READY" if not bad and runner_ok and comparison_ok and not missing else "BLOCKED",
                  classification_quality="Human validation is a separate dependency; schema validity is not label accuracy.",
                  date_limitation="Missing dates and pre-2000 dates are reported; no inferred date is silently repaired.",
                  name_match_policy="title/name candidate with aliases and conservative declensions; not literal exact-name proof.")
    report["sample_flow"] = [{"stage": name, "n": value} for name, value in (
        ("title_name_mapping_pairs", len(expected)), ("mapping_articles", len(expected_articles)),
        ("mapping_companies", len(expected_companies)), ("input_pairs", input_counts["rows"]),
        ("stored_results", counts["results"]), ("valid_relevant", counts["valid_relevant"]),
        ("valid_irrelevant", counts["valid_irrelevant"]), ("input_errors", counts["input_error"]),
        ("generation_errors", counts["generation_error"]), ("schema_errors", counts["schema_error"]),
        ("missing_results", len(missing)), ("extra_results", len(extra)))]
    return report, relevant_companies, expected_companies


def registry_bridge(path):
    isin_companies, company_isins, names = defaultdict(set), defaultdict(set), {}
    temporal_fields, duplicates = set(), Counter()
    for row in rows(path):
        cid = text(row.get("company_id"))
        duplicates[cid] += 1
        names[cid] = row.get("company_name") or row.get("company")
        temporal_fields.update(k for k in row if re.search(r"valid_from|valid_to|effective|start_date|end_date", k, re.I))
        for isin in ISIN_RE.findall(str(row.get("isins") or row.get("isin") or "").upper()):
            company_isins[cid].add(isin)
            isin_companies[isin].add(cid)
    result = {**file_info(path, True), "companies": len(names), "duplicate_company_ids": sum(n-1 for n in duplicates.values()),
              "company_isin_edges": sum(map(len, company_isins.values())), "unique_isins": len(isin_companies),
              "company_ids": sorted(names), "companies_without_isins": len(set(names) - set(company_isins)),
              "companies_multiple_isins": sum(len(v)>1 for v in company_isins.values()),
              "isins_multiple_companies": sum(len(v)>1 for v in isin_companies.values()),
              "conflict_examples": [{"isin": k, "company_ids": sorted(v)} for k,v in sorted(isin_companies.items()) if len(v)>1][:30],
              "temporal_fields": sorted(temporal_fields),
              "temporal_validity": "Not established merely by a current registry or historical aliases."}
    return result, isin_companies, company_isins


def is_equity_leaf(row):
    category = text(row.get("parent_category")).casefold()
    return row.get("table_type") == "asset" and row.get("row_type") == "asset_item" and category in EQUITY_CATEGORIES


def fund_intervals(reports):
    gaps, per_fund = Counter(), defaultdict(set)
    for fund, day in reports:
        if day:
            per_fund[fund].add(day)
    total, exact_eom, exact_eom_since_2018 = 0, 0, 0
    for dates in per_fund.values():
        ordered = sorted(dates)
        for before, after in zip(ordered, ordered[1:]):
            months = (after.year - before.year)*12 + after.month-before.month
            gaps[str(months)] += 1
            total += 1
            if months == 1 and all(d.day == calendar.monthrange(d.year,d.month)[1] for d in (before,after)):
                exact_eom += 1
                exact_eom_since_2018 += date(2018,1,1) <= after <= date(2026,5,31)
    return {"all_adjacent_disclosure_intervals": total, "month_gap_distribution": dict(sorted(gaps.items(), key=lambda x:int(x[0]))),
            "consecutive_calendar_month_intervals": gaps["1"], "same_month_intervals": gaps["0"],
            "consecutive_exact_calendar_month_end_intervals": exact_eom,
            "exact_calendar_month_end_closing_2018_01_to_2026_05": exact_eom_since_2018,
            "exact_month_end_window_note": "Count of consecutive genuine report endpoints only; no missing report interpolation, active-status/return/quality gate applied.",
            "multi_month_gap_intervals": sum(n for k,n in gaps.items() if int(k)>1),
            "missing_calendar_months_between_reports": sum((int(k)-1)*n for k,n in gaps.items() if int(k)>1),
            "funds_with_two_or_more_reports": sum(len(v)>1 for v in per_fund.values())}


def audit_holdings(resource, isin_companies, relevant_companies, dictionaries):
    funds = resource / "funds/mac-20261008"
    path = funds / "fund_holdings_long.csv.gz"
    code_isins, dictionary_reports = defaultdict(set), []
    for dictionary in dictionaries:
        fields = schema(dictionary)["columns"]
        lower = {field.lower(): field for field in fields}
        isin_col = next((lower[x] for x in ("isin", "isin_code", "isin_codes") if x in lower), None)
        reg_col = next((lower[x] for x in ("regnumber", "registration_number_security", "security_code", "registration_number") if x in lower), None)
        info = {**file_info(dictionary, True), "columns": fields, "isin_column": isin_col, "registration_column": reg_col}
        if isin_col and reg_col:
            n, dc, scopes = 0, Counter(), defaultdict(Counter)
            for row in rows(dictionary):
                reg = code(row.get(reg_col))
                isins = ISIN_RE.findall(text(row.get(isin_col)).upper())
                dc["rows_with_isin"] += bool(isins)
                dc["rows_missing_isin"] += not isins
                dc["registration_code_equals_isin"] += reg in isins
                dc["true_registration_code_to_isin_rows"] += bool(reg and isins and reg not in isins)
                scopes[text(row.get("security_scope"))]["with_isin" if isins else "without_isin"] += 1
                if reg and isins:
                    code_isins[reg].update(isins)
                n += 1
            info["rows"] = n
            info["coverage"] = dict(dc)
            info["coverage_by_security_scope"] = distribution(scopes)
        dictionary_reports.append(info)
    counts, categories, row_types = Counter(), Counter(), Counter()
    all_link_counts, all_link_types, all_link_companies, all_link_relevant = Counter(),Counter(),set(),set()
    corporate_candidates, corporate_relevant = set(),set()
    market_types = defaultdict(set)
    market_mapping = resource / "market/moex_quotes_2008_2026/mapping.csv"
    if market_mapping.is_file():
        for row in rows(market_mapping):
            market_types[code(row.get("requested_isin"))].add(text(row.get("type")))
    code_row_types, ambiguous_categories, unmapped_code_counts, code_examples = Counter(), Counter(), Counter(), {}
    reports, stock_reports, funds_seen, security_codes, equity_codes, matched_companies, relevant_holdings_companies = set(), set(), set(), set(), set(), set(), set()
    source_reports, identified_isins = set(), set()
    identity_rows, report_sources = Counter(), defaultdict(set)
    temporal_dates = []
    leaf_weights, cash_weights, leaf_missing_weight, leaf_values = defaultdict(float), defaultdict(float), Counter(), defaultdict(float)
    nav_seen, nav_values = set(), {}
    for row in rows(path):
        counts["all_rows"] += 1
        source_reports.add(text(row.get("source_path")))
        row_types[text(row.get("row_type"))] += 1
        counts["missing_fund_id_rows"] += not text(row.get("fund_id"))
        if code(row.get("registration_number_security")):
            code_row_types[text(row.get("row_type"))] += 1
        category = text(row.get("parent_category"))
        fund, day = text(row.get("fund_id")), parse_date(row.get("report_date"))
        funds_seen.add(fund)
        counts["missing_report_date_rows"] += day is None
        if day:
            reports.add((fund,day))
            report_sources[(fund,day)].add(text(row.get("source_path")))
            temporal_dates.append(day)
            nav = number(row.get("net_asset_value_rub"))
            if nav is not None:
                nav_seen.add((fund,day))
                nav_values[(fund,day)] = nav
        if row.get("table_type") != "asset" or row.get("row_type") != "asset_item":
            continue
        counts["leaf_asset_rows"] += 1
        categories[category] += 1
        row_weight, row_value = number(row.get("share_pct")), number(row.get("amount_thousand_rub"))
        if day:
            if row_weight is not None:
                leaf_weights[(fund,day)] += row_weight
                if re.search(r"денежн|cash|депозит|deposit|расчетн", category, re.I):
                    cash_weights[(fund,day)] += row_weight
            else:
                leaf_missing_weight[(fund,day)] += 1
            if row_value is not None:
                leaf_values[(fund,day)] += row_value
        reg = code(row.get("registration_number_security"))
        if reg:
            security_codes.add(reg)
        direct_isins = set(ISIN_RE.findall(reg))
        candidates = direct_isins if direct_isins else code_isins.get(reg,set())
        all_link_counts["asset_item_rows"] += 1
        if len(candidates) != 1:
            all_link_counts["unmapped_security" if not candidates else "ambiguous_isin"] += 1
        else:
            resolved_isin = next(iter(candidates))
            all_link_counts["direct_isin_rows" if direct_isins else "registration_code_resolved_rows"] += 1
            resolved_companies = isin_companies.get(resolved_isin,set())
            resolved_types = market_types.get(resolved_isin,set())
            if len(resolved_companies) != 1:
                all_link_counts["unmapped_company" if not resolved_companies else "ambiguous_company"] += 1
            else:
                resolved_company = next(iter(resolved_companies))
                all_link_counts["unique_company_rows"] += 1
                all_link_companies.add(resolved_company)
                all_link_types["|".join(sorted(resolved_types)) or "unverified_type"] += 1
                if resolved_company in relevant_companies:
                    all_link_relevant.add(resolved_company)
                    all_link_counts["valid_relevant_company_rows"] += 1
                if len(resolved_types)==1 and next(iter(resolved_types)) in {"common_share","preferred_share","depositary_receipt","corporate_bond","exchange_bond","euro_bond"}:
                    all_link_counts["current_corporate_type_candidate_rows"] += 1
                    corporate_candidates.add(resolved_company)
                    if resolved_company in relevant_companies:
                        corporate_relevant.add(resolved_company)
                        all_link_counts["current_corporate_type_with_relevant_news_rows"] += 1
        equity = is_equity_leaf(row)
        if not equity:
            if re.search(r"акци|equity|shares|stocks",category,re.I):
                counts["equity_word_in_nonstandard_parent_category_excluded"] += 1
                ambiguous_categories[category] += 1
            continue
        counts["equity_leaf_rows"] += 1
        stock_reports.add((fund,day))
        equity_codes.add(reg)
        identity_rows[(fund,day,reg,text(row.get("item_name")))] += 1
        counts["equity_missing_code"] += not reg
        weight, value = number(row.get("share_pct")), number(row.get("amount_thousand_rub"))
        counts["equity_missing_weight"] += weight is None
        counts["equity_missing_value"] += value is None
        counts["equity_negative_weight"] += bool(weight is not None and weight < 0)
        counts["equity_weight_gt100"] += bool(weight is not None and weight > 100)
        if len(candidates) != 1:
            counts["equity_unmapped_security" if not candidates else "equity_ambiguous_isin"] += 1
            if not candidates:
                unmapped_code_counts[reg] += 1
                if reg not in code_examples:
                    code_examples[reg] = text(row.get("item_name"))
            continue
        isin = next(iter(candidates))
        identified_isins.add(isin)
        companies = isin_companies.get(isin,set())
        counts["equity_single_isin_rows"] += 1
        if len(companies) != 1:
            counts["equity_unmapped_company" if not companies else "equity_ambiguous_company"] += 1
            continue
        cid = next(iter(companies))
        matched_companies.add(cid)
        counts["equity_unambiguous_company_rows"] += 1
        if cid in relevant_companies:
            relevant_holdings_companies.add(cid)
            counts["equity_rows_with_eligible_relevant_company"] += 1
    result = {**file_info(path, True), "status": "PARTIAL", "counts": dict(counts), "row_type_counts": dict(row_types),
              "row_types_with_security_codes": dict(code_row_types),
              "strict_equity_categories": sorted(EQUITY_CATEGORIES),
              "nonstandard_equity_like_categories_excluded": dict(ambiguous_categories),
              "top_unmapped_equity_codes": [{"code":k,"rows":v,"example_item":code_examples[k]} for k,v in unmapped_code_counts.most_common(25)],
              "all_asset_leaf_identity_bridge": {"counts":dict(all_link_counts),"unique_company_rows_by_current_market_type":dict(all_link_types),
                  "companies":len(all_link_companies),"valid_relevant_companies":len(all_link_relevant),
                  "current_corporate_type_candidate_companies":len(corporate_candidates),"current_corporate_type_candidate_relevant_companies":len(corporate_relevant),
                  "current_corporate_type_candidate_relevant_company_ids":sorted(corporate_relevant),
                  "scope_note":"Current R1 specification includes corporate equities and corporate bonds. This supplementary bridge covers all leaves; current type alone does not verify Russian issuer residence, time validity or portfolio eligibility. Equity subset counts remain a separate diagnostic."},
              "leaf_asset_categories": dict(categories), "funds": len(funds_seen), "unique_fund_exact_date": len(reports),
              "distinct_source_reports_in_long": len(source_reports), "equity_isins_identified": sorted(identified_isins),
              "unique_equity_fund_exact_date": len(stock_reports), "distinct_security_codes_all_leaf": len(security_codes),
              "distinct_equity_security_codes": len(equity_codes - {""}), "earliest_report_date": min(temporal_dates).isoformat() if temporal_dates else None,
              "latest_report_date": max(temporal_dates).isoformat() if temporal_dates else None,
              "fund_date_with_multiple_source_reports": sum(len(v)>1 for v in report_sources.values()),
              "duplicate_equity_fund_date_code_name_rows": sum(n-1 for n in identity_rows.values() if n>1),
              "report_dates_with_nav": len(nav_seen), "nav_is_fund_total_not_unit_price": True,
              "whole_portfolio_leaf_coverage": {"fund_dates_with_leaf_weights": len(leaf_weights),
                  "fund_dates_with_missing_leaf_weight": len(leaf_missing_weight),
                  "fund_dates_leaf_weight_sum_outside_99_101pct": sum(not 99<=v<=101 for v in leaf_weights.values()),
                  "fund_dates_with_cash_or_deposit_leaf": len(cash_weights),
                  "note": "Leaf percentages may use total assets rather than NAV; apparent residual is not automatically cash and cash return is not assigned zero."},
              "intervals_all_reports": fund_intervals(reports), "intervals_equity_reports": fund_intervals(stock_reports),
              "dictionary_sources": dictionary_reports, "registration_code_to_multiple_isins": sum(len(v)>1 for v in code_isins.values()),
              "companies_identified_in_equity_holdings": len(matched_companies),
              "eligible_relevant_companies_in_holdings": len(relevant_holdings_companies),
              "eligible_relevant_company_ids_in_holdings": sorted(relevant_holdings_companies),
              "intersection_policy": "Sets of companies and one count per leaf holding row; no news-row join/multiplication.",
              "temporal_identity_validity": "BLOCKED until dated issuer/security mapping is verified; observed first/last holding dates are not validity periods.",
              "at_readiness": "Raw delta weight is not AT; complete passive portfolio total-return denominator, cash/residual and entry/exit rules required."}
    wide = funds / "fund_holdings_wide.csv.gz"
    if wide.is_file():
        wc, wide_reports, wide_funds = Counter(), set(), set()
        for row in rows(wide):
            wc["rows"] += 1
            fund, day = text(row.get("fund_id")), parse_date(row.get("report_date"))
            wide_funds.add(fund)
            wide_reports.add((fund, day))
            wc["missing_report_date"] += day is None
            nav, assets, liabilities = (number(row.get(k)) for k in ("net_asset_value_rub", "total_assets_thousand_rub", "total_obligations_thousand_rub"))
            wc["missing_nav"] += nav is None
            wc["missing_total_assets"] += assets is None
            wc["missing_total_obligations"] += liabilities is None
            if all(x is not None for x in (nav,assets,liabilities)):
                wc["nav_versus_assets_minus_obligations_gt_1pct"] += abs(nav-(assets-liabilities)*1000) > max(1000,abs(nav)*0.01)
        result["wide_reports"] = {**file_info(wide, True), "counts": dict(wc), "funds": len(wide_funds), "unique_fund_exact_date": len(wide_reports),
                                  "fund_dates_missing_long": len(wide_reports-reports), "fund_dates_missing_wide": len(reports-wide_reports),
                                  "limitation": "Totals are checks of disclosure metadata; no unit-price return or cash return is inferred."}
    return result


def verify_live_view_sample(catalog, run, limit=8):
    """Bounded equality check, never a full live-view aggregation."""
    result = {"status": "BLOCKED", "sample_limit": limit, "full_view_count_requested": False}
    query = f"SELECT pair_id,task_index,valid,result_json FROM llm.sentiment_results_live LIMIT {int(limit)}"
    result["sql"] = query
    try:
        import duckdb
        with closing(duckdb.connect(str(catalog), read_only=True)) as con:
            con.execute("SET autoinstall_known_extensions=false")
            timer = threading.Timer(20, con.interrupt)
            timer.start()
            try:
                observed = con.execute(query).fetchall()
            finally:
                timer.cancel()
        mismatches = []
        with closing(read_only_sqlite(run / "results.sqlite3")) as con:
            for pid, index, valid, body in observed:
                direct = con.execute("SELECT task_index,valid,result_json FROM results WHERE pair_id=?",(pid,)).fetchone()
                if direct != (index,int(valid),body):
                    mismatches.append(pid)
        result.update(rows_compared=len(observed), mismatches=mismatches,
                      status="READY" if observed and not mismatches else "BLOCKED",
                      scope="Bounded row equality plus catalog view definition; not an independent full view census.")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def audit_return_inputs(resource, holdings):
    """Check available price/return evidence without deriving unverified outcomes."""
    out = {"status": "PARTIAL"}
    market_mapping = resource / "market/moex_quotes_2008_2026/mapping.csv"
    types, statuses, per_isin = Counter(), Counter(), defaultdict(set)
    for row in rows(market_mapping):
        types[text(row.get("type"))] += 1
        statuses[text(row.get("status"))] += 1
        per_isin[text(row.get("requested_isin"))].add(text(row.get("type")))
    equity_isins = set(holdings.get("equity_isins_identified", []))
    equity_types = Counter()
    for isin in equity_isins:
        equity_types["|".join(sorted(per_isin.get(isin,set()))) or "unmapped"] += 1
    out["security_asset_classes"] = {**file_info(market_mapping,True), "all_mapping_type_counts": dict(types), "all_status_counts": dict(statuses),
        "identified_equity_holdings_isins_by_type": dict(equity_types),
        "identified_equity_isins_missing_market_mapping": sorted(equity_isins-set(per_isin)),
        "isin_multiple_market_types": sum(len(v)>1 for v in per_isin.values()),
        "limitation": "MOEX current types supplement leaf category; no historical identity period or automatic alias repair is inferred."}
    fund_returns = resource / "funds/mac-20261008/fund_returns_append_available.csv"
    counts, keys, dates, returns_by_fund = Counter(),set(),[],defaultdict(list)
    for row in rows(fund_returns):
        counts["rows"] += 1
        fund, day = text(row.get("fund_key")),parse_date(row.get("date"))
        key = (fund,day)
        counts["duplicate_fund_date"] += key in keys
        keys.add(key)
        counts["missing_date"] += day is None
        if day:
            dates.append(day)
        unit,ret,market,excess = (number(row.get(k)) for k in ("Пай","fund_ret","imoex_ret","excess_return"))
        for label,value in (("unit_price",unit),("fund_return",ret),("imoex_return",market),("excess_return",excess)):
            counts["missing_"+label] += value is None
        counts["nonpositive_unit_price"] += bool(unit is not None and unit<=0)
        if ret is not None and market is not None and excess is not None:
            counts["excess_not_fund_minus_imoex"] += abs(excess-(ret-market))>1e-8
        if day:
            returns_by_fund[fund].append((day,unit,ret))
    for series in returns_by_fund.values():
        series.sort()
        for previous,current in zip(series,series[1:]):
            if previous[1] and previous[1]>0 and current[1] is not None and current[2] is not None:
                counts["unit_return_comparable_intervals"] += 1
                counts["fund_return_not_equal_simple_unit_price_change"] += abs(current[1]/previous[1]-1-current[2])>1e-8
    out["available_fund_returns"] = {**file_info(fund_returns,True), "counts": dict(counts), "fund_keys": len({x[0] for x in keys}),
        "earliest_date": min(dates).isoformat() if dates else None, "latest_date": max(dates).isoformat() if dates else None,
        "intervals": fund_intervals(keys),
        "provenance_and_fund_key_to_holdings_id": "Not established by matching a name/key alone.",
        "alpha_status": "BLOCKED: excess_return=fund_ret-imoex_ret is not a validated factor alpha; risk-free and factor definitions/window need separate evidence."}
    return out


def audit_market(resource, manifest_info):
    folder = resource / "market/moex_quotes_2008_2026/histories"
    files = sorted(folder.glob("*.parquet"))
    result = {"path": str(folder), "files": len(files), "status": "PARTIAL", "date_column": None}
    if files:
        import duckdb
        con = duckdb.connect(":memory:")
        relation = "read_parquet('" + str(folder).replace("\\", "/").replace("'", "''") + "/*.parquet',union_by_name=true)"
        fields = [r[0] for r in con.execute("DESCRIBE SELECT * FROM " + relation).fetchall()]
        result["columns"] = fields
        dates = next((x for x in ("TRADEDATE", "tradedate", "date") if x in fields), None)
        isin = next((x for x in ("REQUESTED_ISIN", "requested_isin", "ISIN", "isin") if x in fields), None)
        query = "SELECT count(*) n"
        if dates:
            query += f',min(try_cast("{dates}" AS DATE)) first_date,max(try_cast("{dates}" AS DATE)) last_date,count(*) FILTER(WHERE try_cast("{dates}" AS DATE) IS NULL) missing_dates'
        if isin:
            query += f',count(DISTINCT "{isin}") isins'
        cur = con.execute(query + " FROM " + relation)
        result.update(dict(zip([d[0] for d in cur.description],cur.fetchone())))
        result["sql"] = query + " FROM " + relation
        result["date_column"] = dates
        if dates and isin:
            result["duplicate_isin_dates_may_be_multi_board"] = con.execute(f'SELECT coalesce(sum(n-1),0) FROM (SELECT count(*) n FROM {relation} GROUP BY "{isin}","{dates}" HAVING count(*)>1)').fetchone()[0]
        result["return_or_payment_fields"] = [x for x in fields if re.search(r"dividend|coupon|redemption|total.?return|adj|accrued|yield",x,re.I)]
        con.close()
    result["candidate_payment_or_factor_resources"] = manifest_info.get("relevant_paths", [])
    result["total_return_ready"] = False
    result["cash_residual_ready"] = False
    result["alpha_ready"] = False
    result["limitation"] = "Quotes/yield fields are not total returns. Resource names alone do not verify payments, cash/residual returns, fund unit prices, factors, or alpha windows. No missing return is set to zero."
    return result


def write_report(path, report, resource, run):
    path = Path(path).resolve()
    for protected in (Path(resource).resolve(),Path(run).resolve()):
        if path == protected or protected in path.parents:
            raise ValueError("Report must be outside immutable resources and live run")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite audit report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-root", type=Path, default=RESOURCE)
    parser.add_argument("--run-root", type=Path, default=RUN)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--security-dictionary", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--catalog-live-count", action="store_true", help="Optional potentially expensive catalog live scan, after direct source audit")
    parser.add_argument("--live-view-sample", action="store_true", help="Optional bounded equality check of eight view rows after direct audit")
    args = parser.parse_args()
    started = time.monotonic()
    report = {"version": "title-name-regression-input-audit-v1", "observed_at_utc": utc_now(),
              "mode": "schema_probe" if args.probe else "full_read_only_audit", "sql": SQL,
              "resource_root": str(args.resource_root), "run_root": str(args.run_root),
              "historical_expectations_not_observations": {"mapping_pairs": 341799, "articles": 308470, "companies": 610,
                  "fund_reports": 43678, "report_lines_all_types": 1693870, "quote_rows": 4918463}, "errors": {}}
    try:
        report["inventory"],report["resource_manifest"] = inventory(args.resource_root)
    except Exception as exc:
        report["errors"]["inventory"] = f"{type(exc).__name__}: {exc}"
    report["catalog"] = catalog_probe(args.catalog)
    if not args.probe:
        relevant_companies, expected_companies, isin_companies = set(), set(), {}
        try:
            report["llm"], relevant_companies, expected_companies = audit_llm(args.resource_root,args.run_root,args.project_root)
            print(json.dumps({"section":"llm","status":report["llm"]["status"],"sample_flow":report["llm"]["sample_flow"],"integrity_failures":report["llm"]["integrity_failures"]}),flush=True)
            live = report["catalog"].get("live_counts")
            if live:
                actual = report["llm"]["sqlite"]["aggregate"]
                report["catalog"]["sqlite_counts_match"] = (live["n"],live["valid"],live["last_task"]) == (actual["n"],actual["valid"],actual["max_index"])
        except Exception as exc:
            report["errors"]["llm"] = f"{type(exc).__name__}: {exc}"
        try:
            registry = args.resource_root / "mapping/active-20261008-safe-otkritie/cleaned_single_industry/company_registry_active.parquet"
            report["registry"],isin_companies,company_isins = registry_bridge(registry)
            report["registry"]["mapped_company_ids_missing_registry"] = sorted(expected_companies - set(report["registry"]["company_ids"]))
            report["registry"]["mapped_company_ids_without_isin"] = sorted((expected_companies & set(report["registry"]["company_ids"])) - set(company_isins))
        except Exception as exc:
            report["errors"]["registry"] = f"{type(exc).__name__}: {exc}"
        try:
            report["holdings"] = audit_holdings(args.resource_root,isin_companies,relevant_companies,args.security_dictionary)
            if "llm" not in report:
                report["holdings"]["eligible_relevant_companies_in_holdings"] = None
                report["holdings"]["eligible_relevant_company_ids_in_holdings"] = None
                report["holdings"]["counts"]["equity_rows_with_eligible_relevant_company"] = None
                report["holdings"]["intersection_error"] = "Gemma audit unavailable; zero is not a measured intersection."
        except Exception as exc:
            report["errors"]["holdings"] = f"{type(exc).__name__}: {exc}"
        try:
            report["market"] = audit_market(args.resource_root,report.get("resource_manifest",{}))
        except Exception as exc:
            report["errors"]["market"] = f"{type(exc).__name__}: {exc}"
        try:
            report["return_inputs"] = audit_return_inputs(args.resource_root,report.get("holdings",{}))
        except Exception as exc:
            report["errors"]["return_inputs"] = f"{type(exc).__name__}: {exc}"
    if args.catalog_live_count and not args.probe:
        report["catalog"] = catalog_probe(args.catalog, include_live_count=True)
    if args.live_view_sample and "llm" in report:
        report["live_view_sample"] = verify_live_view_sample(args.catalog,args.run_root)
    report["elapsed_seconds"] = time.monotonic() - started
    report["audit_execution_complete"] = not args.probe and not report["errors"]
    report["status"] = "PROBE_COMPLETE" if args.probe and not report["errors"] else "AUDIT_COMPLETE" if report["audit_execution_complete"] else "EXECUTION_BLOCKED"
    report["regression_readiness"] = {h: {"status": "BLOCKED", "reason": reason} for h,reason in (
        ("H1", "Requires verified security identities in time, complete passive total returns and cash/residual treatment."),
        ("H2", "Requires H1 trading measure and the accepted news exposure/missingness protocol."),
        ("H3", "Requires H1/H2 plus genuine fund unit returns, factors and a frozen forward-alpha window."))}
    if not args.probe:
        report["dependency_readiness"] = [
            {"dependency":"frozen_title_name_inputs_and_pair_ids", "status":report.get("llm",{}).get("status","BLOCKED"), "n":report.get("llm",{}).get("counts",{}).get("results")},
            {"dependency":"valid_relevant_results", "status":"READY" if report.get("llm",{}).get("status")=="READY" else "BLOCKED", "n":report.get("llm",{}).get("counts",{}).get("valid_relevant"), "condition":"Exclude every error/irrelevant result; this count precedes date and portfolio eligibility."},
            {"dependency":"human_label_validation", "status":"BLOCKED", "reason":"Schema/hash validation is not evidence of classification accuracy."},
            {"dependency":"publication_dates_and_provenance", "status":"BLOCKED", "reason":"Missing dates excluded; canonical archive has no date_source/date_precision fields; inference/metadata provenance is not reconstructed."},
            {"dependency":"leaf_equity_holdings", "status":"READY" if "holdings" in report else "BLOCKED", "n":report.get("holdings",{}).get("counts",{}).get("equity_leaf_rows"), "condition":"Leaf category only; exact dates and duplicates reported separately."},
            {"dependency":"time_valid_company_security_bridge", "status":"BLOCKED", "reason":"Ambiguous ISIN links excluded; current static mappings lack validity intervals."},
            {"dependency":"unambiguous_static_news_holdings_intersection", "status":"READY" if "holdings" in report and "llm" in report else "BLOCKED", "n":report.get("holdings",{}).get("eligible_relevant_companies_in_holdings"), "condition":"Company-set overlap only; not time-specific regression observations."},
            {"dependency":"market_quote_inventory", "status":"READY" if "market" in report else "BLOCKED", "n":report.get("market",{}).get("n"), "condition":"Duplicates across boards require an explicit selection rule."},
            {"dependency":"security_total_returns_and_corporate_actions", "status":"BLOCKED", "reason":"Quotes/coupon-rate fields do not establish payment dates, splits, dividends or reinvestment."},
            {"dependency":"whole_portfolio_cash_residual_and_AT", "status":"BLOCKED", "reason":"Full passive denominator and cash/residual returns not established; raw weight changes are not AT."},
            {"dependency":"fund_unit_returns_factors_forward_alpha", "status":"BLOCKED", "reason":"Historical return file coverage/provenance needs linkage; simple market excess return is not factor alpha."},
        ]
    write_report(args.output,report,args.resource_root,args.run_root)
    print(json.dumps({"status":report["status"],"output":str(args.output),"errors":report["errors"],"elapsed_seconds":report["elapsed_seconds"]},ensure_ascii=True))


if __name__ == "__main__":
    main()
