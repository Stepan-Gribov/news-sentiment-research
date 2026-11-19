"""Additional title/name-only mapping; never replaces the active full mapping.

Scan every source title using the active canonical registry and its name policy.
Keep stable article/company IDs; do not read full texts except missing-URL IDs.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as futures
import csv
import importlib.util
import json
import itertools
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

MATCHER = None
TITLE_NAME_EXCLUDED_NAMES = {"газ"}


def load_matcher(path):
    spec = importlib.util.spec_from_file_location("title_name_matcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def name_patterns(mapper, registry):
    mapper._WORKER_STEMMER = mapper.SnowballStemmer("russian")
    mapper._stem_token.cache_clear()
    rows = []
    names = {row["company_id"]: row["company"] for row in registry}
    for row in registry:
        if mapper.normalize_text(row["company"]).strip() in TITLE_NAME_EXCLUDED_NAMES:
            continue
        for alias in str(row["aliases"]).split("; "):
            if alias.strip():
                rows.append(dict(company_id=row["company_id"], name=alias, industry="", product=""))
    output = []
    for pattern in mapper.build_pattern_records(rows, 15):
        refs = {tuple([ref[0], names[ref[0]], *ref[2:]]) for ref in pattern["refs"] if ref[2] == "name"}
        if refs:
            output.append(dict(tokens=pattern["tokens"], refs=sorted(refs)))
    # Snowball handles most forms but stems Газпром/Газпрома differently.
    # Add conservative declensions for consonant-final Cyrillic name tokens.
    # Whole-token matching prevents Газпром from matching Газпромбанк.
    expanded = collections.defaultdict(set)
    for pattern in output:
        for ref in pattern["refs"]:
            surface = ref[6]
            choices = []
            for token in surface:
                alternatives = {token}
                if len(token) >= 4 and mapper.re.fullmatch(r"[а-я]+", token) and token[-1] in "бвгджзклмнпрстфхцчшщ":
                    alternatives.update(token + suffix for suffix in ("а", "у", "ом", "е", "ы", "ов", "ам", "ами", "ах", "я", "ю", "ем", "и"))
                choices.append({mapper.stem_tokens([form])[0] for form in alternatives})
            for tokens in itertools.product(*choices):
                expanded[tokens].add(ref)
    return [dict(tokens=tokens, refs=sorted(refs)) for tokens, refs in expanded.items()]


def worker_init(mapper_path, patterns):
    global MATCHER
    MATCHER = load_matcher(Path(mapper_path))
    MATCHER._WORKER_STEMMER = MATCHER.SnowballStemmer("russian")
    MATCHER._stem_token.cache_clear()
    MATCHER._WORKER_TRIE, MATCHER._WORKER_FAIL, MATCHER._WORKER_OUTPUTS, MATCHER._WORKER_REFS = MATCHER.build_automaton(patterns)


def title_hits(title):
    surface = MATCHER.tokenize(title)
    hits = {}
    MATCHER.add_pattern_hits(MATCHER.scan_tokens(MATCHER.stem_tokens(surface)), hits, True, surface)
    return hits


def select_rows(rows):
    output = []
    cached_title = None
    hits = {}
    for row in rows:
        title = row["title"] or ""
        if title != cached_title:
            cached_title = title
            hits = title_hits(title)
        if not hits:
            continue
        article_id = MATCHER.make_article_id(row["media"], row["link"], row["date"], title, row.get("text", ""))
        for cid, match in sorted(hits.items()):
            output.append(dict(article_id=article_id, date=row["date"], media=row["media"], title=title,
                company_id=cid, company=match["company"], match_signal="name",
                matched_terms=json.dumps({"name": sorted(match["terms"]["name"])}, ensure_ascii=False, separators=(",", ":")),
                article_url=row["link"]))
    return output


def process_part(task):
    index, rows, destination = task
    selected = select_rows(rows)
    path = Path(destination) / f"part_{index:06d}.parquet"
    pq.write_table(pa.Table.from_pylist(selected, schema=MATCHER.OUTPUT_SCHEMA), path, compression="snappy")
    counts = collections.Counter(row["company_id"] for row in selected)
    media = collections.Counter(row["media"] for row in selected)
    dates = [row["date"] for row in selected if row["date"] is not None]
    return dict(index=index, part=str(path), checked=len(rows), pairs=len(selected), counts=dict(counts),
                media=dict(media), missing_dates=sum(row["date"] is None for row in selected),
                min_date=min(dates).isoformat() if dates else None, max_date=max(dates).isoformat() if dates else None)


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--articles", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=20_000)
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1:
        raise ValueError("Workers and batch size must be positive")
    config = json.loads(args.active_config.read_text(encoding="utf-8"))
    source_dir = Path(config["active_directory"])
    if not (source_dir / config["completion_marker"]).is_file():
        raise ValueError("Full mapping is not complete")
    source = source_dir / config["article_company_map"]
    registry_path = source_dir / config["company_registry"]
    mapper_path = Path(config["run_root"]) / "scripts/45_map_articles_to_companies.py"
    compact_script = Path(__file__).with_name("46_build_company_registry_and_article_ids.py")
    protected = [args.active_config, source, registry_path, mapper_path, args.articles]
    stamps = {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in protected}
    if args.output_dir.resolve() == source_dir.resolve() or args.output_dir.resolve() in source_dir.resolve().parents:
        raise ValueError("Additional output must be separate from the active map")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reserved = ["article_company_map_title_name.parquet", "article_company_ids.parquet", "company_registry.parquet", "SUCCESS", "progress.json"]
    if any((args.output_dir / name).exists() for name in reserved):
        raise FileExistsError("Output run already exists; refusing to overwrite")
    if shutil.disk_usage(args.output_dir).free < 2 * 1024**3:
        raise RuntimeError("Less than 2 GiB disk headroom")
    started = time.monotonic()
    progress = args.output_dir / "progress.json"
    scanned = checked = pairs = missing_dates = 0
    counts = collections.Counter()
    media = collections.Counter()
    minima, maxima = [], []
    parts = {}
    parquet = pq.ParquetFile(args.articles)
    mapper = load_matcher(mapper_path)
    registry = pq.read_table(registry_path).to_pylist()
    if len({row["company_id"] for row in registry}) != len(registry) or any(row.get("status", "active") != "active" for row in registry):
        raise ValueError("Registry is not a unique active company list")
    excluded_registry = [row for row in registry if mapper.normalize_text(row["company"]).strip() in TITLE_NAME_EXCLUDED_NAMES]
    registry = [row for row in registry if mapper.normalize_text(row["company"]).strip() not in TITLE_NAME_EXCLUDED_NAMES]
    patterns = name_patterns(mapper, registry)
    print(json.dumps(dict(stage="initialized", source_articles=parquet.metadata.num_rows, company_count=len(registry), name_patterns=len(patterns))), flush=True)
    def status(stage, **extra):
        save_json(progress, dict(status="running", stage=stage, source_articles=parquet.metadata.num_rows,
                                scanned_articles=scanned, checked_titles=checked, title_name_pairs=pairs,
                                elapsed_seconds=round(time.monotonic()-started, 1), **extra))
    status("title_scan")
    staging = args.output_dir / "parts"
    staging.mkdir(exist_ok=False)
    def consume(pending):
        nonlocal checked, pairs, missing_dates
        done, _ = futures.wait(pending, return_when=futures.FIRST_COMPLETED)
        for future in done:
            pending.remove(future)
            item = future.result()
            parts[item["index"]] = item["part"]
            checked += item["checked"]
            pairs += item["pairs"]
            counts.update(item["counts"])
            media.update(item["media"])
            missing_dates += item["missing_dates"]
            if item["min_date"]:
                minima.append(item["min_date"])
                maxima.append(item["max_date"])
        status("title_scan")
    try:
        with futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                                        initializer=worker_init, initargs=(str(mapper_path), patterns)) as pool:
            pending = set()
            index = 0
            fallback_rows = 0
            for group in range(parquet.metadata.num_row_groups):
                table = parquet.read_row_group(group, columns=["media", "title", "link", "date"])
                rows = table.to_pylist()
                missing_links = [i for i, row in enumerate(rows) if not str(row["link"] or "").strip()]
                if missing_links:
                    bodies = parquet.read_row_group(group, columns=["text"])["text"].to_pylist()
                    for offset in missing_links:
                        rows[offset]["text"] = bodies[offset]
                    fallback_rows += len(missing_links)
                for start in range(0, len(rows), args.batch_size):
                    chunk = rows[start:start+args.batch_size]
                    scanned += len(chunk)
                    pending.add(pool.submit(process_part, (index, chunk, str(staging))))
                    index += 1
                    if len(pending) >= 2 * args.workers:
                        consume(pending)
            while pending:
                consume(pending)
        status("merge")
        output = args.output_dir / "article_company_map_title_name.parquet"
        temporary = output.with_suffix(".partial.parquet")
        tables = [pq.read_table(parts[index]) for index in sorted(parts)]
        table = pa.concat_tables(tables) if tables else pa.Table.from_pylist([], schema=mapper.OUTPUT_SCHEMA)
        table = table.sort_by([("article_id", "ascending"), ("company_id", "ascending")])
        if table.num_rows > 1:
            same_article = pc.equal(table["article_id"].slice(1), table["article_id"].slice(0, table.num_rows-1))
            same_company = pc.equal(table["company_id"].slice(1), table["company_id"].slice(0, table.num_rows-1))
            keep = pc.invert(pc.and_(same_article, same_company)).combine_chunks()
            table = table.filter(pa.concat_arrays([pa.array([True]), keep]))
        duplicate_pairs = pairs - table.num_rows
        pairs = table.num_rows
        counts = collections.Counter(table["company_id"].to_pylist())
        media = collections.Counter(table["media"].to_pylist())
        missing_dates = table["date"].null_count
        min_date = pc.min(table["date"]).as_py()
        max_date = pc.max(table["date"]).as_py()
        pq.write_table(table, temporary, compression="snappy", row_group_size=100_000)
        del table, tables
        if pq.ParquetFile(temporary).metadata.num_rows != pairs:
            raise ValueError("Pair count mismatch")
        os.replace(temporary, output)
        status("compact")
        scoped_registry = staging / "scoped_registry.parquet"
        pq.write_table(pa.Table.from_pylist(registry, schema=pq.read_schema(registry_path)), scoped_registry, compression="snappy")
        with (args.output_dir / "compact.log").open("x", encoding="utf-8") as log:
            subprocess.run([sys.executable, str(compact_script), "--article-company-map", str(output),
                            "--company-registry-input", str(scoped_registry), "--company-registry", str(args.output_dir / "company_registry.parquet"),
                            "--article-ids-output", str(args.output_dir / "article_company_ids.parquet")], stdout=log, stderr=subprocess.STDOUT, check=True)
        compact = json.loads((args.output_dir / "article_company_ids.parquet.audit.json").read_text(encoding="utf-8"))
        if compact["article_company_pairs_in_output"] != pairs or compact["duplicate_source_pairs_collapsed"] != 0:
            raise ValueError("Compact reconciliation failed")
        if any((Path(path).stat().st_size, Path(path).stat().st_mtime_ns) != stamp for path, stamp in stamps.items()):
            raise ValueError("A protected input changed; additional map not marked complete")
        names = {row["company_id"]: row["company"] for row in registry}
        with (args.output_dir / "company_match_counts_title_name.csv").open("x", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["company_id", "company", "matches"])
            writer.writeheader()
            writer.writerows(dict(company_id=cid, company=names[cid], matches=count) for cid, count in counts.most_common())
        audit = dict(status="completed", policy="title-name-only-v1", source_mapping_version=config["version"],
                     source_articles=parquet.metadata.num_rows, scanned_articles=scanned, checked_titles=checked,
                     title_name_pairs=pairs, unique_articles=compact["unique_articles_output"], companies_matched=len(counts),
                     media_pairs=dict(media), missing_date_pairs=missing_dates, min_date=min_date.isoformat() if min_date else None,
                     max_date=max_date.isoformat() if max_date else None, source_unchanged=True, active_mapping_not_replaced=True,
                     ids_preserved=True, duplicate_pairs=0, duplicate_input_pairs_collapsed=duplicate_pairs,
                     missing_url_text_fallback_rows=fallback_rows, match_scope="title", signals=["name"], workers=args.workers,
                     name_inflection_policy="existing-names-plus-conservative-consonant-declensions-v1",
                     excluded_companies=[dict(company_id=row["company_id"], company=row["company"], reason="user_excluded_ambiguous_GAZ_20261008") for row in excluded_registry],
                     elapsed_seconds=round(time.monotonic()-started, 1))
        save_json(args.output_dir / "audit.json", audit)
        save_json(args.output_dir / "mapping_config.json", dict(kind="additional_title_name", source_mapping_version=config["version"],
                  directory=str(args.output_dir), article_company_map=output.name, compact_article_company_map="article_company_ids.parquet",
                  company_registry="company_registry.parquet", completion_marker="SUCCESS", active_mapping_not_replaced=True))
        (args.output_dir / "README.md").write_text(
            "# Additional company-name matches in news titles\n\n"
            "Scans all source news TITLES using allowed names/aliases of the existing canonical registry. Industry and product signals are excluded. Case, ё/е and punctuation are normalized. Existing stemming is supplemented with conservative declensions of consonant-final name words (Газпром/Газпрома), with whole-token boundaries. Ambiguous names and excluded entities remain excluded.\n\n"
            "Article IDs use the same frozen source URL/hash function. Canonical company IDs are unchanged. The active full map, registry, source news and active configuration are not modified. Texts are read only for missing-URL fallback IDs, never matched. Because declension coverage is improved, this additional map is NOT asserted to be a strict subset of the old name matches.\n\n"
            "Files: article_company_map_title_name.parquet (one unique article/company pair); article_company_ids.parquet (one article, comma-separated company IDs); company_registry.parquet (same canonical company IDs and descriptions); company_match_counts_title_name.csv; audit.json; progress.json; mapping_config.json. Use results only with SUCCESS.\n",
            encoding="utf-8")
        save_json(progress, {**audit, "stage": "completed"})
        (args.output_dir / "SUCCESS").write_text("Additional title/name map reconciled; existing data preserved.\n", encoding="utf-8")
        # These are exclusively this run's regenerated scratch parts; final
        # verified Parquet outputs contain every surviving row.
        try:
            for path in parts.values():
                Path(path).unlink()
            scoped_registry.unlink()
            staging.rmdir()
        except OSError as cleanup_error:
            print(json.dumps({"scratch_cleanup_warning": str(cleanup_error)}), flush=True)
        print(json.dumps(audit), flush=True)
    except Exception as error:
        save_json(progress, dict(status="failed", error=str(error), elapsed_seconds=round(time.monotonic()-started, 1)))
        raise


if __name__ == "__main__":
    main()
