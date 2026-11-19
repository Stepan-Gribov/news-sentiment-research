"""Read-only audit of Otkritie name candidates; never modifies source datasets."""
import collections
import importlib.util
import json
import pathlib
import random
import sys
import time
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0,"./news_data/company_mapping_runs/run_20260930_01/python_packages_20260930")
root=pathlib.Path("./news_data/company_mapping_runs/run_20261006_single_industry_01")
old=root.parent/"run_20260930_01"
spec=importlib.util.spec_from_file_location("mapper",root/"scripts/45_map_articles_to_companies.py")
mapper=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mapper)
mapper._WORKER_STEMMER=mapper.SnowballStemmer("russian")
registry=pq.read_table(root/"cleaned_single_industry/company_registry_active.parquet").to_pylist()
company=next(r for r in registry if r["company"]=="Открытие Холдинг")
original_ids=set(company["old_company_ids"].split("; "))
source_companies=[r for r in mapper.load_company_rows(old/"fund_asset_ru_isin_by_company_classification_template.xlsx") if r["company_id"] in original_ids]
records=mapper.build_pattern_records(source_companies,15)
strict=[]
bare=[]
for p in records:
    references=[r for r in p["refs"] if r[2]=="name"]
    bare.extend([r for r in references if tuple(r[6])==("открытие",)])
    kept=[r for r in references if tuple(r[6])!=("открытие",)]
    if kept:
        strict.append(dict(p,refs=kept))
mapper._WORKER_TRIE,mapper._WORKER_FAIL,mapper._WORKER_OUTPUTS,mapper._WORKER_REFS=mapper.build_automaton(strict)
rng=random.Random(20261007)
sample=[]
name_count=0
term_counts=collections.Counter()
started=time.monotonic()
file=pq.ParquetFile(root/"cleaned_single_industry/article_company_map_clean.parquet")
for batch in file.iter_batches(columns=["company_id","match_signal","matched_terms","title","article_url","article_id","date","media"],batch_size=100_000):
    table=pa.Table.from_batches([batch])
    table=table.filter(pc.equal(table["company_id"],company["company_id"]))
    table=table.filter(pc.match_substring_regex(table["match_signal"],r"(^|;)name(;|$)"))
    for row in table.to_pylist():
        name_count+=1
        term_counts.update(json.loads(row["matched_terms"])["name"])
        if len(sample)<64:
            sample.append(row)
        else:
            index=rng.randrange(name_count)
            if index<len(sample):
                sample[index]=row
print(json.dumps(dict(stage="sample_collected",name_matches=name_count,sample_size=len(sample),elapsed_seconds=round(time.monotonic()-started,1)),ensure_ascii=True),flush=True)
pending={r["article_url"]:r for r in sample}
archive=pq.ParquetFile(old/"articles_all_from_news_sqlite.parquet")
found=[]
for group in range(archive.metadata.num_row_groups):
    links=archive.read_row_group(group,columns=["link"])
    indices=pc.indices_nonzero(pc.is_in(links["link"],value_set=pa.array(list(pending),type=pa.string())))
    if len(indices):
        articles=archive.read_row_group(group,columns=["link","title","text","date","media"]).take(indices)
        for article in articles.to_pylist():
            target=pending.pop(article["link"],None)
            if target is None:
                continue
            evidence={}
            for field in ("title","text"):
                surfaces=mapper.tokenize(article[field])
                mapper.add_pattern_hits(mapper.scan_tokens(mapper.stem_tokens(surfaces)),evidence,field=="title",surfaces)
            has_explicit=bool(evidence)
            text=str(article["text"] or "")
            normalized=mapper.normalize_text(text)
            opening=__import__("re").search(r"\bоткрыт[а-я]*\b",normalized)
            snippet=normalized[max(0,opening.start()-120):opening.end()+180] if opening else ""
            found.append(dict(article_id=target["article_id"],date=str(article["date"]),media=article["media"],title=article["title"],url=article["link"],explicit_multiword_name=has_explicit,opening_context=snippet))
    if not pending:
        break
report=dict(company=company["company"],name_matches=name_count,sample_method="Uniform reservoir sample of name-match pairs; seed 20261007; n=64",sample_size=len(sample),texts_found=len(found),explicit_multiword_name=sum(r["explicit_multiword_name"] for r in found),no_explicit_multiword_name=sum(not r["explicit_multiword_name"] for r in found),bare_name_patterns=[dict(company_id=r[0],label=r[3],actual_alias=list(r[6])) for r in bare],name_evidence_labels=dict(term_counts),examples_without_explicit_name=[r for r in found if not r["explicit_multiword_name"]][:8],examples_with_explicit_name=[r for r in found if r["explicit_multiword_name"]][:3],elapsed_seconds=round(time.monotonic()-started,1),limitation="Absence of the original multiword name patterns is not by itself proof of irrelevance. All 64 texts were checked, not all 645305 matches. Labels store full issuer names rather than the actual triggering alias.")
print(json.dumps(report,ensure_ascii=True),flush=True)
