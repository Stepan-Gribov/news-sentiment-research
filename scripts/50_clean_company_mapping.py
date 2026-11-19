#!/usr/bin/env python3
"""Versioned, non-destructive issuer cleanup and unique article/issuer candidates.

Inputs are never modified. A SUCCESS marker is written only after reconciliation.
Unrelated shared ISINs are reported, never automatically merged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import openpyxl
import pyarrow as pa
import pyarrow.parquet as pq


SIGNALS = ("name", "industry", "product")
PROTECTED = ("Магнит", "Полюс", "Открытие Холдинг")
PLACES = set("Башкортостан|Беларусь|Волгоград|Казань|Казахстан|Карелия|Краснодар|Москва|Новосибирск|Омск|Россия|Санкт-Петербург|Томск|Уфа|Чувашия|ХМАО|Нижний Новгород|Марий Эл".casefold().split("|"))
GOVERNMENT = re.compile(r"^(администрация|министерство|минфин|департамент финансов|республика\b)|\b(область|край|республика)$|автономный округ|^(ненецкий|ямало-ненецкий) ао$")
EXCHANGE = re.compile(r"\bбиржа\b|\bexchange\b")
LEGAL = re.compile(r"\b(пао|оао|зао|ао|ооо)\b")
HISTORY = re.compile(r"\((?:бывш\.?|ранее|прежнее название)\s*([^)]*)\)", re.I)
LEGAL_PARENTHESES = re.compile(r"\((?:пао|оао|зао|ао|ооо|государственная компания)\)", re.I)
STATUS_PARENTHESES = re.compile(r"\(не\s*сущ\.?[^)]*\)", re.I)
VERIFIED_RENAMES = [
    (["Новабев Групп", "Белуга Групп", "Синергия"], "https://novabev.com/about/our-background/"),
    (["ЯКОВЛЕВ", "ИРКУТ"], "https://yakovlev.ru/press-centre/releases/yakovlev-zavershil-korporativnye-protsedury-po-pereimenovaniyu/"),
    (["ПИК-Специализированный застройщик", "Группа Компаний ПИК"], "https://db-estate.media.pik-service.ru/different/2022/06/06/godovoj_otchet_gk_pik_2021_proekt_dlya_sd_WMgvWq1PkiFs8S5B.pdf"),
    (["Первая Грузовая Компания", "Первая Грузовая Компания АО"], "https://cbr.ru/registries/rcb/ecb/orginfo/?id=11050134"),
]
# Only spelling/abbreviation aliases; different legal issuers, parents and subsidiaries
# are NOT merged by a shared brand or shared ISIN alone.
SPELLING_GROUPS = [
    ["ВТБ", "Банк ВТБ"], ["РОСБАНК", "АКБ РОСБАНК"],
    ["БФА Банк", "Банк БФА"], ["БАНК УРАЛСИБ", "УРАЛСИБ"],
    ["НЛМК", "Новолипецкий металлургический комбинат"],
    ["ИНТЕР РАО", "ИНТЕР РАО ЕЭС"], ["ИНТЕРРАО", "ИНТЕР РАО ЕЭС"],
    ["МОЭСК", "Московская объединенная электросетевая компания"],
    ["Группа ЛСР", "ГруппаЛСР"], ["ЛенСпецСМУ", "ССМО ЛенСпецСМУ"],
    ["ОК РУСАЛ", "РУСАЛ", "РУСАЛ Плс"],
    ["ФК ОТКРЫТИЕ", "Открытие Холдинг"],
    ["ПКБ", "Первое коллекторское бюро", "ПКО ПКБ"],
    ["ЦППК", "Центральная ППК"],
    ["Металлоинвест", "ХК Металлоинвест"],
    ["Росинтер Ресторантс", "Росинтер Ресторантс Холдинг"],
    ["ГК Сегежа", "Сегежа Групп"],
    ["Компания М.видео", "М.видео"],
    ["Positive Technologies", "Позитив Текнолоджиз"],
    ["En+ Group", "ЭН+ ГРУП"],
    ["Т Плюс", "Волжская ТГК (Т Плюс)"],
]


def norm(value):
    return unicodedata.normalize("NFKC", str(value or "")).casefold().replace("ё", "е")


def base(name):
    value = HISTORY.sub("", norm(name))
    value = STATUS_PARENTHESES.sub("", value)
    value = LEGAL_PARENTHESES.sub("", value)
    value = LEGAL.sub("", value)
    return re.sub(r"[^a-zа-я0-9]+", "", value)


def legal_type(name):
    tokens = LEGAL.findall(norm(name))
    return "llc" if "ооо" in tokens else ("stock" if tokens else "")


def excluded(name):
    name = norm(name).strip()
    if EXCHANGE.search(name):
        return "exchange_excluded_by_user"
    if name in PLACES or GOVERNMENT.search(name):
        return "sovereign_regional_or_municipal_issuer_not_company"
    return ""


def split_terms(values):
    return sorted({t.strip() for value in values for t in str(value or "").split(";") if t.strip()}, key=norm)


def save_csv(path, rows, fields=None):
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_registry(workbook_path, blacklist_path):
    book = openpyxl.load_workbook(workbook_path, read_only=True, data_only=True)
    records = {}
    source_rows = 0
    for row in book["По компаниям"].iter_rows(min_row=2, max_col=10, values_only=True):
        name = str(row[0] or "").strip()
        if not name:
            continue
        source_rows += 1
        cid = "c_" + hashlib.blake2b(f"{norm(name)}\t{norm(row[2])}".encode(), digest_size=12).hexdigest()
        item = records.setdefault(cid, dict(company_id=cid, company=name, isins=set(), industry=[], product=[], geography=[], demand_segments=[], competitors=[]))
        item["isins"].update(re.findall(r"\b[A-Z]{2}[A-Z0-9]{10}\b", str(row[2] or "")))
        for field, index in [("industry", 3), ("product", 4), ("geography", 5), ("demand_segments", 6), ("competitors", 7)]:
            item[field].append(str(row[index] or "").strip())
    book.close()
    parent = {cid: cid for cid in records}

    def root(cid):
        while parent[cid] != cid:
            parent[cid] = parent[parent[cid]]
            cid = parent[cid]
        return cid

    merges = []

    def join(a, b, reason):
        if root(a) == root(b):
            return
        parent[root(b)] = root(a)
        merges.append(dict(left_id=a, left_name=records[a]["company"], right_id=b, right_name=records[b]["company"], method=reason))

    grouped = defaultdict(list)
    for cid, record in records.items():
        grouped[base(record["company"])].append(cid)
    for ids in grouped.values():
        mixed_types = {legal_type(records[cid]["company"]) for cid in ids} >= {"llc", "stock"}
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                ta, tb = legal_type(records[a]["company"]), legal_type(records[b]["company"])
                # A bank brand can describe a different LLC vs JSC: do not collapse it.
                if {ta, tb} == {"llc", "stock"}:
                    continue
                if mixed_types and ta != tb:
                    continue
                if ta != tb and not (records[a]["isins"] & records[b]["isins"]):
                    continue
                join(a, b, "normalized_spelling_or_historical_annotation")

    for group in SPELLING_GROUPS:
        ids = [cid for cid, r in records.items() if base(r["company"]) in {base(n) for n in group}]
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if records[a]["isins"] & records[b]["isins"]:
                    join(a, b, "explicit_spelling_alias_and_shared_isin")

    for names, url in VERIFIED_RENAMES:
        ids = [cid for cid, r in records.items() if norm(r["company"]) in {norm(n) for n in names}]
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if records[a]["isins"] & records[b]["isins"] or base(records[a]["company"]) == base(records[b]["company"]):
                    join(a, b, "primary_source_confirmed_alias: " + url)

    # An explicit historical name in the source is usable only when it points to
    # one unambiguous remaining issuer cluster with the same security identifier.
    for cid, r in records.items():
        for old_name in HISTORY.findall(r["company"]):
            candidates = [other for other, rr in records.items()
                          if other != cid and base(rr["company"]) == base(old_name)
                          and rr["isins"] & r["isins"]]
            if len({root(other) for other in candidates}) == 1:
                for other in candidates:
                    if {legal_type(r["company"]), legal_type(records[other]["company"])} != {"llc", "stock"}:
                        join(cid, other, "source_explicit_historical_name_and_shared_isin")

    with blacklist_path.open(encoding="utf-8-sig") as handle:
        rules = list(csv.DictReader(handle))
    entity_exclusions = [r for r in rules if r["action"] == "exclude_all_signals" and r["rule_scope"] == "entity"]
    clusters = defaultdict(list)
    for cid in records:
        clusters[root(cid)].append(cid)
    canonical, crosswalk = {}, []
    for ids in clusters.values():
        preferred = min(ids, key=lambda cid: (base(records[cid]["company"]) not in {base(n) for n in PROTECTED}, norm(records[cid]["company"]) not in {norm(names[0]) for names, _ in VERIFIED_RENAMES}, bool(HISTORY.search(records[cid]["company"]) or STATUS_PARENTHESES.search(records[cid]["company"])), bool(LEGAL.search(norm(records[cid]["company"]))), len(records[cid]["company"]), records[cid]["company"]))
        name = records[preferred]["company"]
        reason = next((excluded(records[cid]["company"]) for cid in ids if excluded(records[cid]["company"])), "")
        if not reason:
            reason = next((r["reason"] or "explicit_entity_blacklist" for r in entity_exclusions
                           if r["company_id"] in ids or (r["company_name"] and any(
                               norm(records[cid]["company"]) == norm(r["company_name"]) for cid in ids))), "")
        entry = dict(company_id=preferred, company=name, isins="; ".join(sorted(set().union(*(records[cid]["isins"] for cid in ids)))), aliases="; ".join(sorted({records[cid]["company"] for cid in ids}, key=norm)), old_company_ids="; ".join(sorted(ids)))
        for field in ("industry", "product", "geography", "demand_segments", "competitors"):
            entry[field] = "; ".join(split_terms([v for cid in ids for v in records[cid][field]]))
        entry["status"] = "excluded" if reason else "active"
        entry["exclusion_reason"] = reason
        canonical[preferred] = entry
        for cid in ids:
            crosswalk.append(dict(old_company_id=cid, old_company=records[cid]["company"], company_id=preferred, company=name, status=entry["status"], exclusion_reason=reason))
            if reason:
                rules.append(dict(rule_scope="entity", company_id=cid, company_name=records[cid]["company"], match_term="", action="exclude_all_signals", reason=reason, keep_match_if=""))

    cross = {r["old_company_id"]: r for r in crosswalk}
    by_isin = defaultdict(set)
    for cid, r in records.items():
        for isin in r["isins"]:
            by_isin[isin].add(cross[cid]["company_id"])
    collisions = [dict(isin=isin, company_ids="; ".join(sorted(ids)), companies="; ".join(canonical[c]["company"] for c in sorted(ids)), status="review_not_auto_merged") for isin, ids in sorted(by_isin.items()) if len(ids) > 1 and any(canonical[c]["status"] == "active" for c in ids)]
    # Explicit user exceptions must survive any later policy changes.
    for protected in PROTECTED:
        assert any(base(e["company"]) == base(protected) and e["status"] == "active" for e in canonical.values()), protected
    return source_rows, records, canonical, crosswalk, merges, collisions, rules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--companies", type=Path, required=True)
    parser.add_argument("--industry-overrides", type=Path)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--blacklist", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--registry-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_rows, old, canonical, crosswalk, merges, collisions, rules = build_registry(args.companies, args.blacklist)
    if args.industry_overrides:
        policy = json.loads(args.industry_overrides.read_text(encoding="utf-8"))
        overrides = policy["canonical_industries"]
        # The reviewed canonical database is authoritative for existing fields.
        # Python set iteration previously made case-only alias ordering random.
        # Keep the prior alias order verbatim rather than changing descriptions.
        registry_path = args.industry_overrides.with_name("company_registry_single_industry.json")
        expected = {r["company_id"]: r for r in json.loads(registry_path.read_text(encoding="utf-8"))}
        active_ids = {cid for cid, row in canonical.items() if row["status"] == "active"}
        if active_ids != set(overrides) or active_ids != set(expected):
            raise ValueError("Single-industry registry IDs differ from active companies")
        for cid, industry in overrides.items():
            if any(delimiter in industry for delimiter in (";", ",", "|", "\n")):
                raise ValueError("More than one primary industry")
            canonical[cid]["industry"] = industry
            if set(canonical[cid]["aliases"].split("; ")) != set(expected[cid]["aliases"].split("; ")):
                raise ValueError("Canonical name aliases changed: " + cid)
            canonical[cid]["aliases"] = expected[cid]["aliases"]
            if canonical[cid] != expected[cid]:
                raise ValueError("A non-industry company field changed: " + cid)
    active = sorted((r for r in canonical.values() if r["status"] == "active"), key=lambda r: norm(r["company"]))
    excluded_rows = [r for r in canonical.values() if r["status"] == "excluded"]
    save_csv(args.output_dir / "company_registry_active.csv", active)
    save_csv(args.output_dir / "company_registry_all.csv", list(canonical.values()))
    pq.write_table(pa.Table.from_pylist(active), args.output_dir / "company_registry_active.parquet", compression="snappy")
    save_csv(args.output_dir / "company_alias_crosswalk.csv", crosswalk)
    save_csv(args.output_dir / "company_exclusions.csv", excluded_rows)
    save_csv(args.output_dir / "company_merge_log.csv", merges, ["left_id", "left_name", "right_id", "right_name", "method"])
    save_csv(args.output_dir / "isin_identity_conflicts_review.csv", collisions, ["isin", "company_ids", "companies", "status"])
    save_csv(args.output_dir / "company_mapping_blacklist_active.csv", rules)
    initial = dict(source_workbook_rows=source_rows, old_company_ids=len(old), active_companies=len(active), excluded_canonical_entities=len(excluded_rows), excluded_original_ids=sum(r["status"] == "excluded" for r in crosswalk), aliases_collapsed=len(old) - len(canonical), isin_conflicts_for_review=len(collisions))
    print(json.dumps(initial), flush=True)
    if args.registry_only:
        return
    started = time.monotonic()
    source = pq.ParquetFile(args.mapping)
    source_stat = args.mapping.stat()
    cross = {r["old_company_id"]: r for r in crosswalk}
    suppress_canonical_ids = {cross[r["company_id"]]["company_id"] for r in rules if r["action"] == "suppress_bare_name_signal"}
    suppress_ids = {cid for cid, r in cross.items() if r["company_id"] in suppress_canonical_ids}
    # The old export labels full names rather than actual aliases. The practice
    # full phrase therefore MUST NOT be removed by its hypothetical bare alias.
    generic = {norm(r["match_term"]) for r in rules if r["rule_scope"] == "generic_keyword"}
    counts = {r["company_id"]: dict(company=r["company"], company_id=r["company_id"], matches=0, with_name=0, industry_product_only=0, kommersant=0, prime=0, interfax=0, rbc=0, missing_date_matches=0, min_date=None, max_date=None) for r in active}
    stats = Counter()
    seen_articles = set()
    current_id = None
    current = {}
    buffer = []
    partial = args.output_dir / "article_company_map_clean.partial.parquet"
    writer = pq.ParquetWriter(partial, source.schema_arrow, compression="snappy", use_dictionary=["company_id", "company", "match_signal", "media"])

    @lru_cache(maxsize=100_000)
    def cleaned_terms(cid, text):
        terms = json.loads(text or "{}")
        if not isinstance(terms, dict):
            raise ValueError("Unexpected matched_terms format")
        terms = {signal: {str(t) for t in terms.get(signal, []) if norm(t) not in generic} for signal in SIGNALS}
        if cid in suppress_ids:
            terms["name"] = set()
        return {k: v for k, v in terms.items() if v}

    def flush_article():
        if current_id is None:
            return
        if current_id in seen_articles:
            raise ValueError(f"Non-contiguous repeated article ID: {current_id}")
        seen_articles.add(current_id)
        for cid, entry in current.items():
            row, terms = entry
            row["match_signal"] = ";".join(k for k in SIGNALS if terms.get(k))
            row["matched_terms"] = json.dumps({k: sorted(terms[k], key=norm) for k in SIGNALS if terms.get(k)}, ensure_ascii=False, separators=(",", ":"))
            buffer.append(row)
            stats["clean_pairs"] += 1
            c = counts[cid]
            c["matches"] += 1
            c["with_name" if terms.get("name") else "industry_product_only"] += 1
            if row["media"] in c:
                c[row["media"]] += 1
            if row["date"] is None:
                c["missing_date_matches"] += 1
            else:
                c["min_date"] = min(c["min_date"], row["date"]) if c["min_date"] else row["date"]
                c["max_date"] = max(c["max_date"], row["date"]) if c["max_date"] else row["date"]
        if len(buffer) >= 40_000:
            writer.write_table(pa.Table.from_pylist(buffer, schema=source.schema_arrow))
            buffer.clear()

    try:
        for batch in source.iter_batches(batch_size=100_000):
            for row in batch.to_pylist():
                stats["raw_pairs"] += 1
                aid = row["article_id"]
                if aid != current_id:
                    flush_article()
                    current_id, current = aid, {}
                cid = row["company_id"]
                policy = cross[cid]
                if policy["status"] == "excluded":
                    stats["excluded_entity_pairs"] += 1
                    continue
                terms = cleaned_terms(cid, row["matched_terms"])
                if not terms:
                    stats["empty_after_ambiguous_or_generic_filter"] += 1
                    continue
                if cid in suppress_ids and "name" in row["match_signal"].split(";"):
                    stats["ambiguous_name_signals_removed"] += 1
                new_id = policy["company_id"]
                if new_id in current:
                    first, accumulated = current[new_id]
                    for field in ("date", "media", "title", "article_url"):
                        if first[field] != row[field]:
                            raise ValueError(f"Conflicting metadata for {aid}: {field}")
                    for key, values in terms.items():
                        accumulated.setdefault(key, set()).update(values)
                    stats["duplicate_pairs_collapsed"] += 1
                else:
                    row["company_id"] = new_id
                    row["company"] = policy["company"]
                    current[new_id] = (row, {k: set(v) for k, v in terms.items()})
            elapsed = time.monotonic() - started
            print(json.dumps(dict(processed=stats["raw_pairs"], total=source.metadata.num_rows, clean_pairs=stats["clean_pairs"], elapsed_seconds=round(elapsed, 1))), flush=True)
        flush_article()
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=source.schema_arrow))
    finally:
        writer.close()
    assert stats["raw_pairs"] == source.metadata.num_rows
    assert stats["raw_pairs"] == sum(stats[k] for k in ("clean_pairs", "excluded_entity_pairs", "empty_after_ambiguous_or_generic_filter", "duplicate_pairs_collapsed"))
    assert sum(c["matches"] for c in counts.values()) == stats["clean_pairs"]
    assert source_stat.st_size == args.mapping.stat().st_size and source_stat.st_mtime_ns == args.mapping.stat().st_mtime_ns
    assert pq.ParquetFile(partial).metadata.num_rows == stats["clean_pairs"]
    partial.rename(args.output_dir / "article_company_map_clean.parquet")
    rows = sorted(counts.values(), key=lambda c: (-c["matches"], norm(c["company"])))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
        for key in ("min_date", "max_date"):
            row[key] = row[key].isoformat() if row[key] else ""
    save_csv(args.output_dir / "company_match_counts_sorted_clean.csv", rows, ["rank", "company", "company_id", "matches", "with_name", "industry_product_only", "kommersant", "prime", "interfax", "rbc", "missing_date_matches", "min_date", "max_date"])
    report = dict(**initial, **stats, source_mapping=str(args.mapping), clean_mapping=str(args.output_dir / "article_company_map_clean.parquet"), elapsed_seconds=round(time.monotonic() - started, 1), unique_pair_key=["article_id", "company_id"], source_unchanged=True, protected_companies=["Открытие Холдинг", "Магнит", "Полюс"], top_20=rows[:20], limitation="The name blacklist removes bare ambiguous name evidence; thematic industry/product candidates remain. Candidates are not validated corporate sentiment labels. Remaining shared-ISIN conflicts are not merged without identity evidence.")
    previous_audit = args.mapping.parent / "cleanup_audit.json"
    if previous_audit.exists():
        previous = json.loads(previous_audit.read_text(encoding="utf-8"))
        if Path(previous["clean_mapping"]).resolve() != args.mapping.resolve() or previous["clean_pairs"] != stats["raw_pairs"]:
            raise ValueError("Source lineage audit does not match the input map")
        report["previous_cleanup_audit"] = str(previous_audit)
        report["original_raw_pairs"] = previous.get("original_raw_pairs", previous["raw_pairs"])
        for key in ("duplicate_pairs_collapsed", "excluded_entity_pairs", "empty_after_ambiguous_or_generic_filter"):
            report["total_" + key] = previous.get("total_" + key, previous.get(key, 0)) + stats[key]
        assert report["original_raw_pairs"] == report["clean_pairs"] + sum(report["total_" + key] for key in ("duplicate_pairs_collapsed", "excluded_entity_pairs", "empty_after_ambiguous_or_generic_filter"))
    (args.output_dir / "cleanup_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "SUCCESS").write_text("Unique article/canonical-company candidates reconciled; original inputs unchanged.\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
