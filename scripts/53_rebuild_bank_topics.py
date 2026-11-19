"""Versioned bank-topic rebuild; old archives and active map stay intact until success."""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_mapper(directory):
    spec = importlib.util.spec_from_file_location("mapper_bank", directory / "45_map_articles_to_companies.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preflight(args, mapper):
    active = pq.read_table(args.previous_clean / "company_registry_active.parquet").to_pylist()
    banks = [r for r in active if mapper.is_bank_company(r)]
    input_file = pq.ParquetFile(args.articles)
    counts = dict(bank=0, credit=0, deposit=0, any_bank_topic=0)
    raw_patterns = {
        "bank": r"банк(?:и|а|ов|ам|ами|ах|ом|е)?",
        "credit": r"кредит(?:а|у|ом|е|ы|ов|ам|ами|ах)?",
        "deposit": r"вклад(?:а|у|ом|е|ы|ов|ам|ами|ах)?",
    }
    processed = 0
    for batch in input_file.iter_batches(columns=["title", "media"], batch_size=100_000):
        titles = pc.utf8_lower(pc.fill_null(batch.column(0), ""))
        media = pc.utf8_lower(pc.fill_null(batch.column(1), ""))
        included = pc.not_equal(media, "frank")
        masks = []
        for key, pattern in raw_patterns.items():
            mask = pc.and_(included, pc.match_substring_regex(titles, r"(^|[^a-zа-я0-9])(?:" + pattern + r")([^a-zа-я0-9]|$)"))
            counts[key] += pc.sum(pc.cast(mask, pa.int64())).as_py() or 0
            masks.append(mask)
        counts["any_bank_topic"] += pc.sum(pc.cast(pc.or_(pc.or_(masks[0], masks[1]), masks[2]), pa.int64())).as_py() or 0
        processed += batch.num_rows
    old_raw = args.previous_clean.parent / "article_company_map.parquet"
    old_meta = pq.ParquetFile(old_raw).metadata
    old_bytes_per_pair = old_raw.stat().st_size / old_meta.num_rows
    # Conservative uncompressed relationship count, compressed-storage estimate.
    upper_pairs = old_meta.num_rows + counts["any_bank_topic"] * len(banks)
    estimated_peak_bytes = upper_pairs * old_bytes_per_pair * 3.2
    report = dict(policy=mapper.BANK_KEYWORD_POLICY_VERSION,source_articles=processed,active_banks=len(banks),
                  bank_companies=[dict(company_id=r["company_id"],company=r["company"]) for r in banks],
                  headline_counts=counts,upper_bound_raw_pairs=upper_pairs,old_bytes_per_pair=old_bytes_per_pair,
                  estimated_peak_additional_gb=round(estimated_peak_bytes/1024**3,2),
                  free_gb=round(shutil.disk_usage(args.run_dir).free/1024**3,2),search_scope="title",
                  note="Storage estimate, not a guarantee. Shared banking topics create candidates for every active bank; old map is preserved.")
    save_json(args.run_dir / "preflight.json", report)
    print(json.dumps({k:v for k,v in report.items() if k!="bank_companies"},ensure_ascii=True), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--articles", type=Path, required=True)
    parser.add_argument("--companies", type=Path, required=True)
    parser.add_argument("--previous-clean", type=Path, required=True)
    parser.add_argument("--active-config", type=Path, required=True)
    parser.add_argument("--blacklist", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    mapper = load_mapper(here)
    report = preflight(args, mapper)
    if args.preflight_only:
        return
    if report["estimated_peak_additional_gb"] + 5 > report["free_gb"]:
        raise RuntimeError("Insufficient safe disk headroom for a versioned rebuild; old data unchanged")
    if not (args.previous_clean / "SUCCESS").is_file():
        raise RuntimeError("Previous cleaned mapping not complete")
    started = time.monotonic()
    source_before = {str(p): (p.stat().st_size,p.stat().st_mtime_ns) for p in [args.articles,args.companies,args.previous_clean/"article_company_map_clean.parquet"]}
    initial_active = json.loads(args.active_config.read_text(encoding="utf-8"))
    progress = args.run_dir / "progress.json"
    def stage(name, command):
        save_json(progress,dict(status="running",stage=name,started_at_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=round(time.monotonic()-started,1)))
        print(json.dumps(dict(stage=name,command=command),ensure_ascii=True),flush=True)
        with (args.run_dir/(name+".log")).open("a",encoding="utf-8") as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    raw = args.run_dir / "article_company_map.parquet"
    clean = args.run_dir / "cleaned_bank_topics"
    compact = args.run_dir / "compact_bank_topics"
    try:
        stage("map",[sys.executable,str(here/"45_map_articles_to_companies.py"),"--articles",str(args.articles),"--companies",str(args.companies),"--output",str(raw),"--workers",str(args.workers)])
        stage("clean",[sys.executable,str(here/"50_clean_company_mapping.py"),"--companies",str(args.companies),"--mapping",str(raw),"--blacklist",str(args.blacklist),"--output-dir",str(clean)])
        stage("compact",[sys.executable,str(here/"46_build_company_registry_and_article_ids.py"),"--article-company-map",str(clean/"article_company_map_clean.parquet"),"--company-registry-input",str(clean/"company_registry_active.parquet"),"--company-registry",str(compact/"company_registry.parquet"),"--article-ids-output",str(compact/"article_company_ids.parquet")])
        audit = json.loads((clean/"cleanup_audit.json").read_text(encoding="utf-8"))
        compact_audit = json.loads((compact/"article_company_ids.parquet.audit.json").read_text(encoding="utf-8"))
        raw_audit = json.loads(raw.with_suffix(".parquet.audit.json").read_text(encoding="utf-8"))
        if raw_audit["article_rows_read"] != pq.ParquetFile(args.articles).metadata.num_rows or raw_audit.get("bank_keyword_policy") != mapper.BANK_KEYWORD_POLICY_VERSION:
            raise RuntimeError("Incomplete archive coverage or wrong keyword policy")
        def counts_from(directory):
            with (directory/"company_match_counts_sorted_clean.csv").open(encoding="utf-8-sig",newline="") as handle:
                return {r["company_id"]:r for r in csv.DictReader(handle)}
        old_counts,new_counts = counts_from(args.previous_clean),counts_from(clean)
        if old_counts.keys() != new_counts.keys():
            raise RuntimeError("Canonical company IDs changed during bank-only rebuild")
        bank_ids = {r["company_id"] for r in report["bank_companies"]}
        for cid,previous in old_counts.items():
            current = new_counts[cid]
            if int(current["with_name"]) != int(previous["with_name"]):
                raise RuntimeError("Name-match coverage changed for "+cid)
            if cid not in bank_ids and int(current["matches"]) != int(previous["matches"]):
                raise RuntimeError("Non-bank match coverage changed for "+cid)
            if cid in bank_ids and int(current["matches"]) < int(previous["matches"]):
                raise RuntimeError("Existing bank candidates lost for "+cid)
        save_json(args.run_dir/"bank_only_reconciliation.json",dict(companies=len(new_counts),bank_companies=len(bank_ids),nonbank_counts_unchanged=True,name_match_counts_unchanged=True,bank_counts_not_reduced=True,full_archive_rows=raw_audit["article_rows_read"]))
        if not (clean/"SUCCESS").is_file():
            raise RuntimeError("Missing clean SUCCESS")
        if any((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)!=stamp for p,stamp in source_before.items()):
            raise RuntimeError("An input changed during processing")
        if json.loads(args.active_config.read_text(encoding="utf-8")) != initial_active:
            raise RuntimeError("Another job changed active map; manual promotion required")
        updated = dict(initial_active)
        updated.update(version="20261006_bank_topics_01",run_root=str(args.run_dir).replace("\\","/"),active_directory=str(clean).replace("\\","/"),compact_mapping_directory=str(compact).replace("\\","/"),bank_keyword_policy=mapper.BANK_KEYWORD_POLICY_VERSION,bank_keyword_policy_search_scope="title",previous_active_directory=initial_active["active_directory"])
        save_json(args.run_dir/"previous_active_config.json",initial_active)
        save_json(args.run_dir/"new_active_config.json",updated)
        save_json(args.active_config,updated)
        save_json(progress,dict(status="completed",stage="completed",active_companies=audit["active_companies"],clean_pairs=audit["clean_pairs"],elapsed_seconds=round(time.monotonic()-started,1),compact_audit=compact_audit))
        (args.run_dir/"SUCCESS").write_text("Bank topic rebuild, cleanup and compact mapping completed; old sources preserved.\n",encoding="utf-8")
    except Exception as exc:
        save_json(progress,dict(status="failed",error=str(exc),elapsed_seconds=round(time.monotonic()-started,1)))
        raise


if __name__ == "__main__":
    main()
