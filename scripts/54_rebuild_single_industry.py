"""Full versioned rebuild. Activate only after unchanged IDs/products/names checks."""
from __future__ import annotations
import argparse
import collections
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


def save(path, value):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    os.replace(temporary,path)


def main():
    parser=argparse.ArgumentParser()
    for key in ("run-dir","articles","companies","previous-clean","active-config","blacklist","industry-overrides"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--workers",type=int,default=6)
    parser.add_argument("--version",default=None)
    parser.add_argument("--fix-bare-otkritie",action="store_true")
    args=parser.parse_args()
    here=Path(__file__).resolve().parent
    started=time.monotonic()
    progress=args.run_dir/"progress.json"
    original=json.loads(args.active_config.read_text(encoding="utf-8"))
    policy=json.loads(args.industry_overrides.read_text(encoding="utf-8"))
    sources=[args.articles,args.companies,args.previous_clean/"article_company_map_clean.parquet",args.previous_clean/"company_registry_active.parquet",args.industry_overrides,args.blacklist]
    stamps={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in sources}
    def status(stage,**extra):
        save(progress,dict(status="running",stage=stage,started_at_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=round(time.monotonic()-started,1),**extra))
    def stage(name,command):
        if shutil.disk_usage(args.run_dir).free<5*1024**3:
            raise RuntimeError("Disk safety reserve exhausted; old map remains active")
        status(name)
        print(json.dumps(dict(stage=name,command=command)),flush=True)
        with (args.run_dir/(name+".log")).open("a",encoding="utf-8") as handle:
            subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,check=True)
    try:
        status("preflight")
        if not (args.previous_clean/"SUCCESS").is_file():
            raise RuntimeError("Previous map is incomplete")
        spec=importlib.util.spec_from_file_location("mapper",here/"45_map_articles_to_companies.py")
        mapper=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mapper)
        mapper._WORKER_STEMMER=mapper.SnowballStemmer("russian")
        before=mapper.load_company_rows(args.companies)
        after=mapper.load_company_rows(args.companies,args.industry_overrides)
        before_patterns=mapper.build_pattern_records(before,15)
        after_patterns=mapper.build_pattern_records(after,15)
        affected_ids=set()
        if args.fix_bare_otkritie:
            registry=pq.read_table(args.previous_clean/"company_registry_active.parquet").to_pylist()
            affected=[r for r in registry if r["company"]=="Открытие Холдинг"]
            if len(affected)!=1:
                raise RuntimeError("Expected exactly one protected Otkritie Holding issuer")
            affected_ids={r["company_id"] for r in affected}
            allowed_original_ids=set(affected[0]["old_company_ids"].split("; "))
            previous_spec=importlib.util.spec_from_file_location("previous_mapper",args.previous_clean.parent/"scripts/45_map_articles_to_companies.py")
            previous_mapper=importlib.util.module_from_spec(previous_spec)
            previous_spec.loader.exec_module(previous_mapper)
            previous_mapper._WORKER_STEMMER=previous_mapper.SnowballStemmer("russian")
            previous_companies=previous_mapper.load_company_rows(args.companies,args.industry_overrides)
            before_patterns=previous_mapper.build_pattern_records(previous_companies,15)
        def refs(records,signal):
            return {(tuple(p["tokens"]),tuple(r)) for p in records for r in p["refs"] if r[2]==signal}
        for signal in (("industry","product") if args.fix_bare_otkritie else ("name","product")):
            if refs(before_patterns,signal)!=refs(after_patterns,signal):
                raise RuntimeError("Existing "+signal+" rules changed")
        if args.fix_bare_otkritie:
            removed=refs(before_patterns,"name")-refs(after_patterns,"name")
            added=refs(after_patterns,"name")-refs(before_patterns,"name")
            if added or not removed or any(ref[0] not in allowed_original_ids or tuple(ref[6])!=("открытие",) for _,ref in removed):
                raise RuntimeError("Name changes extend beyond the approved bare Otkritie alias")
            save(args.run_dir/"name_pattern_reconciliation.json",dict(removed_bare_alias_refs=len(removed),no_added_name_patterns=True,all_other_name_patterns_unchanged=True,industry_product_patterns_unchanged=True,affected_company_ids=sorted(affected_ids)))
        # Coarse upper bound: first industry word stem only, remaining words
        # ignored. Whole article texts are not scanned during this estimate.
        prefix_owners=collections.Counter()
        for c in after:
            if c.get("primary_industry_explicit") and c["industry"]:
                prefix=mapper.stem_tokens(mapper.tokenize(c["industry"]))[0]
                prefix_owners[prefix]+=1
        file=pq.ParquetFile(args.articles)
        additional_upper=0
        batches=[] if args.fix_bare_otkritie else file.iter_batches(columns=["title"],batch_size=100_000)
        for batch in batches:
            titles=pc.utf8_lower(pc.fill_null(batch.column(0),""))
            titles=pc.replace_substring(titles,pattern="ё",replacement="е")
            for prefix,owners in prefix_owners.items():
                matches=pc.match_substring_regex(titles,r"(^|[^а-яa-z0-9])"+prefix)
                additional_upper+=(pc.sum(pc.cast(matches,pa.int64())).as_py() or 0)*owners
        old_raw=args.previous_clean.parent/"article_company_map.parquet"
        old_rows=pq.ParquetFile(old_raw).metadata.num_rows
        estimated_gb=(old_rows+additional_upper)*old_raw.stat().st_size/max(old_rows,1)*3.5/1024**3
        free_gb=shutil.disk_usage(args.run_dir).free/1024**3
        preflight=dict(source_rows=file.metadata.num_rows,primary_industry_companies=sum(bool(v) for v in policy["canonical_industries"].values()),blank_industry_companies=sum(not v for v in policy["canonical_industries"].values()),estimated_peak_additional_gb=round(estimated_gb,2),free_gb=round(free_gb,2),new_industry_coarse_upper_pairs=additional_upper,name_product_pattern_sets_unchanged=not args.fix_bare_otkritie,industry_product_pattern_sets_unchanged=args.fix_bare_otkritie,only_bare_otkritie_name_removed=args.fix_bare_otkritie,search_scope="title",estimate_note="Name-alias removal cannot add pairs; otherwise use a coarse title-prefix projection. Compressed storage projection is not a guarantee")
        save(args.run_dir/"preflight.json",preflight)
        print(json.dumps(preflight),flush=True)
        if estimated_gb+5>free_gb:
            raise RuntimeError("Insufficient disk headroom for versioned rebuild")
        raw=args.run_dir/"article_company_map.parquet"
        clean=args.run_dir/"cleaned_single_industry"
        compact=args.run_dir/"compact_single_industry"
        stage("map",[sys.executable,str(here/"45_map_articles_to_companies.py"),"--articles",str(args.articles),"--companies",str(args.companies),"--industry-overrides",str(args.industry_overrides),"--output",str(raw),"--workers",str(args.workers)])
        stage("clean",[sys.executable,str(here/"50_clean_company_mapping.py"),"--companies",str(args.companies),"--industry-overrides",str(args.industry_overrides),"--mapping",str(raw),"--blacklist",str(args.blacklist),"--output-dir",str(clean)])
        stage("compact",[sys.executable,str(here/"46_build_company_registry_and_article_ids.py"),"--article-company-map",str(clean/"article_company_map_clean.parquet"),"--company-registry-input",str(clean/"company_registry_active.parquet"),"--company-registry",str(compact/"company_registry.parquet"),"--article-ids-output",str(compact/"article_company_ids.parquet")])
        status("verify")
        old_registry={r["company_id"]:r for r in pq.read_table(args.previous_clean/"company_registry_active.parquet").to_pylist()}
        new_registry={r["company_id"]:r for r in pq.read_table(clean/"company_registry_active.parquet").to_pylist()}
        if old_registry.keys()!=new_registry.keys():
            raise RuntimeError("Canonical company IDs changed")
        for cid,row in old_registry.items():
            new=new_registry[cid]
            if any(row[k]!=new[k] for k in row if k!="industry"):
                raise RuntimeError("A non-industry database field changed: "+cid)
            if new["industry"]!=policy["canonical_industries"][cid]:
                raise RuntimeError("Wrong primary industry: "+cid)
        def counts(directory):
            with (directory/"company_match_counts_sorted_clean.csv").open(encoding="utf-8-sig",newline="") as f:
                return {r["company_id"]:r for r in csv.DictReader(f)}
        old_counts,new_counts=counts(args.previous_clean),counts(clean)
        if old_counts.keys()!=new_counts.keys():
            raise RuntimeError("Canonical company IDs changed in counts")
        for cid in old_counts:
            if cid in affected_ids:
                if int(new_counts[cid]["with_name"])>=int(old_counts[cid]["with_name"]) or int(new_counts[cid]["matches"])>int(old_counts[cid]["matches"]):
                    raise RuntimeError("Bare-name removal did not reduce Otkritie name matches")
            elif args.fix_bare_otkritie:
                if any(old_counts[cid][key]!=new_counts[cid][key] for key in old_counts[cid] if key!="rank"):
                    raise RuntimeError("Another issuer's coverage changed: "+cid)
            elif old_counts[cid]["with_name"]!=new_counts[cid]["with_name"]:
                raise RuntimeError("Name-match coverage changed")
        raw_audit=json.loads(raw.with_suffix(".parquet.audit.json").read_text(encoding="utf-8"))
        clean_audit=json.loads((clean/"cleanup_audit.json").read_text(encoding="utf-8"))
        compact_audit=json.loads((compact/"article_company_ids.parquet.audit.json").read_text(encoding="utf-8"))
        if raw_audit["article_rows_read"]!=file.metadata.num_rows or not raw_audit["primary_industry_explicit"]:
            raise RuntimeError("Archive coverage or single-industry policy failed")
        if args.fix_bare_otkritie and raw_audit.get("name_alias_policy")!="no-bare-otkritie-20261008-v1":
            raise RuntimeError("Name-alias fix not applied by the mapper")
        if not (clean/"SUCCESS").is_file():
            raise RuntimeError("Cleanup not completed")
        if any((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)!=stamp for p,stamp in stamps.items()):
            raise RuntimeError("An input changed")
        if json.loads(args.active_config.read_text(encoding="utf-8"))!=original:
            raise RuntimeError("Another task activated a map; manual promotion required")
        # Compatibility paths required by the existing sentiment sampler.
        archive_link=args.run_dir/"articles_all_from_news_sqlite.parquet"
        if not archive_link.exists():
            os.link(args.articles,archive_link)
        dependency_link=args.run_dir/"python_packages_20260930"
        if not dependency_link.exists():
            target=args.articles.parent/"python_packages_20260930"
            if os.name=="nt":
                subprocess.run(["cmd","/c","mklink","/J",str(dependency_link),str(target)],check=True)
            else:
                dependency_link.symlink_to(target,target_is_directory=True)
        shutil.copy2(here/"45_map_articles_to_companies.py",args.run_dir/"45_map_articles_to_companies.py")
        updated=dict(original)
        updated.update(version=args.version or policy["version"],run_root=str(args.run_dir).replace("\\","/"),active_directory=str(clean).replace("\\","/"),compact_mapping_directory=str(compact).replace("\\","/"),single_industry_policy=str(args.industry_overrides).replace("\\","/"),industry_search_scope="title",previous_active_directory=original["active_directory"])
        if args.fix_bare_otkritie:
            updated["name_alias_policy"]="no-bare-otkritie-20261008-v1"
        save(args.run_dir/"previous_active_config.json",original)
        save(args.run_dir/"new_active_config.json",updated)
        save(args.run_dir/"single_industry_reconciliation.json",dict(company_ids_unchanged=True,all_non_industry_fields_unchanged=True,name_match_counts_unchanged=not args.fix_bare_otkritie,name_product_pattern_sets_unchanged=not args.fix_bare_otkritie,unaffected_company_counts_unchanged=args.fix_bare_otkritie,industry_product_pattern_sets_unchanged=args.fix_bare_otkritie,affected_company_counts=[dict(company_id=cid,before=old_counts[cid],after=new_counts[cid]) for cid in affected_ids],source_rows=file.metadata.num_rows,companies=len(new_registry)))
        save(args.active_config,updated)
        save(progress,dict(status="completed",stage="completed",source_rows=file.metadata.num_rows,active_companies=len(new_registry),clean_pairs=clean_audit["clean_pairs"],compact_audit=compact_audit,elapsed_seconds=round(time.monotonic()-started,1)))
        (args.run_dir/"SUCCESS").write_text("Single primary industry mapping completed; old data preserved.\n",encoding="utf-8")
    except Exception as error:
        save(progress,dict(status="failed",stage="failed",error=str(error),elapsed_seconds=round(time.monotonic()-started,1)))
        raise

if __name__=="__main__":
    main()
