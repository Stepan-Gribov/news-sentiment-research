#!/usr/bin/env python3
"""Freeze a blinded, event-grouped human-validation sample from C04a.

The script never calls an LLM.  It reads the immutable C04a JSONL in streaming
passes, assigns event groups before splitting, draws a fixed stratified sample,
and writes two independent annotation forms with empty human-label fields.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SAMPLE_VERSION = "target-human-validation-c04b-v1"
EVENT_PROXY_VERSION = "normalized-visible-body-prefix-1200-v1"
DEFAULT_SEED = 20260916
SPLIT_PROBABILITIES = {"development": 0.4, "heldout": 0.6}
RBC_OBJECT_ID_SOURCES = {"rbc_freenews_object_id", "rbc_object_id_timestamp"}
RISK_ORDER = (
    "rbc_object_id_date",
    "ambiguous_short_target",
    "multi_target_article",
    "truncated",
    "target_literal_absent",
    "standard",
)
DEFAULT_RISK_QUOTAS = {
    "development": {
        "rbc_object_id_date": 10,
        "ambiguous_short_target": 10,
        "multi_target_article": 15,
        "truncated": 20,
        "target_literal_absent": 25,
        "standard": 40,
    },
    "heldout": {
        "rbc_object_id_date": 15,
        "ambiguous_short_target": 15,
        "multi_target_article": 25,
        "truncated": 30,
        "target_literal_absent": 35,
        "standard": 60,
    },
}
FORM_FIELDS = (
    "row_id",
    "annotator_id",
    "source",
    "published_at",
    "target_company",
    "title",
    "text",
    "relevance_label",
    "relevance_unclear_reason",
    "polarity_label",
    "polarity_unclear_reason",
    "confidence",
    "notes",
)
EMPTY_ANNOTATION_FIELDS = (
    "relevance_label",
    "relevance_unclear_reason",
    "polarity_label",
    "polarity_unclear_reason",
    "confidence",
    "notes",
)
SELECTION_FIELDS = (
    "row_id",
    "pair_id",
    "input_line_number",
    "split",
    "event_group_id",
    "media",
    "link",
    "published_at",
    "target_company",
    "period",
    "length_bin",
    "visible_text_chars",
    "text_chars_original",
    "truncated",
    "target_literal_absent",
    "multi_target_article",
    "rbc_object_id_date",
    "ambiguous_short_target",
    "risk_class",
    "sampling_stratum_id",
    "stratum_population",
    "stratum_sample_size",
    "split_assignment_probability",
    "within_stratum_probability",
    "overall_inclusion_probability",
    "design_weight",
    "sampling_reason",
)
ADJUDICATION_FIELDS = (
    "row_id",
    "pair_id",
    "split",
    "annotator_a_relevance",
    "annotator_b_relevance",
    "adjudicated_relevance",
    "relevance_adjudication_note",
    "annotator_a_polarity",
    "annotator_b_polarity",
    "adjudicated_polarity",
    "polarity_adjudication_note",
    "adjudicator_notes",
)
SPACE_RE = re.compile(r"\s+")
ALNUM_RE = re.compile(r"[^\w]+", flags=re.UNICODE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path, digest: str | None = None) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": digest or sha256_file(path),
    }


def normalized_link(record: dict[str, Any]) -> str:
    return SPACE_RE.sub(" ", str(record.get("link") or "")).strip().rstrip("/").lower()


def article_key(record: dict[str, Any]) -> str:
    return f"{str(record.get('media') or '').strip().lower()}\x1f{normalized_link(record)}"


def visible_normalized(value: Any) -> str:
    text = ALNUM_RE.sub(" ", str(value or "").casefold())
    return SPACE_RE.sub(" ", text).strip()


def event_group_id(record: dict[str, Any]) -> str:
    body = visible_normalized(record.get("text"))
    if not body:
        body = visible_normalized(record.get("title"))
    payload = f"{EVENT_PROXY_VERSION}\x1f{body[:1200]}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def target_literal_absent(record: dict[str, Any]) -> bool:
    target = SPACE_RE.sub(" ", str(record.get("target_company") or "")).strip().casefold()
    visible = f"{record.get('title') or ''} {record.get('text') or ''}".casefold()
    return bool(target) and target not in visible


def ambiguous_short_target(record: dict[str, Any]) -> bool:
    normalized = visible_normalized(record.get("target_company"))
    tokens = normalized.split()
    return len(tokens) == 1 and len(tokens[0]) <= 5


def rbc_object_id_date(record: dict[str, Any]) -> bool:
    return (
        str(record.get("media") or "").strip().lower() == "rbc"
        and record.get("date_source") in RBC_OBJECT_ID_SOURCES
    )


def period(record: dict[str, Any]) -> str:
    value = str(record.get("published_at") or "")
    if len(value) < 4 or not value[:4].isdigit():
        raise ValueError(f"Missing/invalid published_at for {record.get('id')}")
    year = int(value[:4])
    if year <= 2015:
        return "early_through_2015"
    if year <= 2021:
        return "middle_2016_2021"
    return "late_2022_onward"


def length_bin(record: dict[str, Any]) -> str:
    size = len(str(record.get("text") or ""))
    if size <= 800:
        return "short_le_800"
    if size <= 2400:
        return "medium_801_2400"
    return "long_gt_2400"


def stable_int(*parts: Any) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def assigned_split(seed: int, event_id: str) -> str:
    unit = stable_int(SAMPLE_VERSION, seed, "split", event_id) / float(2**256)
    return "development" if unit < SPLIT_PROBABILITIES["development"] else "heldout"


def primary_risk(flags: dict[str, bool]) -> str:
    for name in RISK_ORDER[:-1]:
        if flags[name]:
            return name
    return "standard"


def first_pass(input_path: Path) -> tuple[str, int, dict[bytes, int], dict[bytes, bytes]]:
    digest = hashlib.sha256()
    article_counts: dict[bytes, int] = {}
    article_events: dict[bytes, bytes] = {}
    rows = 0
    with input_path.open("rb") as source:
        for line_number, raw in enumerate(source, start=1):
            digest.update(raw)
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Malformed input at line {line_number}: {exc}") from exc
            pair_id = str(record.get("id") or "")
            if not pair_id or not record.get("text"):
                raise ValueError(f"C04a contract missing id/text at line {line_number}")
            akey = hashlib.blake2b(article_key(record).encode("utf-8"), digest_size=16).digest()
            eid = bytes.fromhex(event_group_id(record))
            previous = article_events.setdefault(akey, eid)
            if previous != eid:
                raise ValueError(
                    "One normalized media+URL maps to multiple event proxies; "
                    f"line={line_number} id={pair_id}"
                )
            article_counts[akey] = article_counts.get(akey, 0) + 1
            rows += 1
    return digest.hexdigest(), rows, article_counts, article_events


def candidate_from_record(
    record: dict[str, Any], line_number: int, seed: int, article_counts: dict[bytes, int]
) -> tuple[tuple[str, str, str, str, str], dict[str, Any]]:
    eid = event_group_id(record)
    akey = hashlib.blake2b(article_key(record).encode("utf-8"), digest_size=16).digest()
    flags = {
        "rbc_object_id_date": rbc_object_id_date(record),
        "ambiguous_short_target": ambiguous_short_target(record),
        "multi_target_article": article_counts[akey] > 1,
        "truncated": bool(record.get("text_truncated")),
        "target_literal_absent": target_literal_absent(record),
    }
    split = assigned_split(seed, eid)
    media = str(record.get("media") or "").strip().lower()
    per = period(record)
    length = length_bin(record)
    risk = primary_risk(flags)
    cell = (split, risk, media, per, length)
    candidate = {
        "pair_id": str(record["id"]),
        "input_line_number": line_number,
        "event_group_id": eid,
        "split": split,
        "media": media,
        "link": record.get("link"),
        "published_at": record.get("published_at"),
        "target_company": record.get("target_company"),
        "period": per,
        "length_bin": length,
        "visible_text_chars": len(str(record.get("text") or "")),
        "text_chars_original": record.get("text_chars_original"),
        **flags,
        "risk_class": risk,
        "priority": stable_int(SAMPLE_VERSION, seed, "pair-priority", record["id"]),
    }
    return cell, candidate


def balanced_cell_weights(cells: list[tuple[str, str, str]]) -> dict[tuple[str, str, str], float]:
    sources = sorted({cell[0] for cell in cells})
    periods_by_source = {
        source: sorted({cell[1] for cell in cells if cell[0] == source}) for source in sources
    }
    lengths_by_source_period = {
        (source, per): sorted(
            {cell[2] for cell in cells if cell[0] == source and cell[1] == per}
        )
        for source in sources
        for per in periods_by_source[source]
    }
    weights: dict[tuple[str, str, str], float] = {}
    for source, per, length in cells:
        weights[(source, per, length)] = (
            1.0
            / len(sources)
            / len(periods_by_source[source])
            / len(lengths_by_source_period[(source, per)])
        )
    return weights


def allocate_cells(
    populations: dict[tuple[str, str, str], int], quota: int
) -> dict[tuple[str, str, str], int]:
    if sum(populations.values()) < quota:
        raise ValueError(f"Risk stratum has {sum(populations.values())} rows for quota {quota}")
    cells = sorted(populations)
    weights = balanced_cell_weights(cells)
    allocation = {cell: 0 for cell in cells}
    for _ in range(quota):
        available = [cell for cell in cells if allocation[cell] < populations[cell]]
        if not available:
            raise ValueError("No capacity left while allocating sample")
        chosen = max(
            available,
            key=lambda cell: (
                weights[cell] / (allocation[cell] + 1),
                -allocation[cell],
                tuple(cell),
            ),
        )
        allocation[chosen] += 1
    return {cell: count for cell, count in allocation.items() if count}


def second_pass(
    input_path: Path,
    seed: int,
    article_counts: dict[bytes, int],
    risk_quotas: dict[str, dict[str, int]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str, str, str], int], dict[str, Any]]:
    cell_counts: Counter[tuple[str, str, str, str, str]] = Counter()
    reservoirs: dict[
        tuple[str, str, str, str, str], list[tuple[int, str, dict[str, Any]]]
    ] = defaultdict(list)
    flag_population: Counter[str] = Counter()
    source_year_population: Counter[str] = Counter()
    with input_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            cell, candidate = candidate_from_record(record, line_number, seed, article_counts)
            cell_counts[cell] += 1
            for flag in (
                "rbc_object_id_date",
                "ambiguous_short_target",
                "multi_target_article",
                "truncated",
                "target_literal_absent",
            ):
                flag_population[flag] += bool(candidate[flag])
            source_year_population[
                f"{candidate['media']}\x1f{str(candidate['published_at'])[:4]}"
            ] += 1
            capacity = risk_quotas[cell[0]][cell[1]]
            heap = reservoirs[cell]
            entry = (-candidate["priority"], candidate["pair_id"], candidate)
            if len(heap) < capacity:
                heapq.heappush(heap, entry)
            elif candidate["priority"] < -heap[0][0]:
                heapq.heapreplace(heap, entry)

    allocations: dict[tuple[str, str, str, str, str], int] = {}
    for split, risk_targets in risk_quotas.items():
        for risk, quota in risk_targets.items():
            populations = {
                (media, per, length): count
                for (cell_split, cell_risk, media, per, length), count in cell_counts.items()
                if cell_split == split and cell_risk == risk
            }
            partial = allocate_cells(populations, quota)
            for (media, per, length), count in partial.items():
                allocations[(split, risk, media, per, length)] = count

    selected: list[dict[str, Any]] = []
    for cell, sample_size in sorted(allocations.items()):
        population = cell_counts[cell]
        candidates = sorted((entry[2] for entry in reservoirs[cell]), key=lambda row: row["priority"])
        if len(candidates) < sample_size:
            raise ValueError(f"Reservoir too small for {cell}: {len(candidates)} < {sample_size}")
        split_probability = SPLIT_PROBABILITIES[cell[0]]
        within_probability = sample_size / population
        overall_probability = split_probability * within_probability
        stratum_text = "|".join(cell)
        stratum_id = hashlib.sha256(
            f"{SAMPLE_VERSION}\x1f{stratum_text}".encode("utf-8")
        ).hexdigest()[:20]
        for candidate in candidates[:sample_size]:
            candidate = dict(candidate)
            candidate.update(
                {
                    "sampling_stratum_id": stratum_id,
                    "stratum_population": population,
                    "stratum_sample_size": sample_size,
                    "split_assignment_probability": split_probability,
                    "within_stratum_probability": within_probability,
                    "overall_inclusion_probability": overall_probability,
                    "design_weight": 1.0 / overall_probability,
                    "sampling_reason": "balanced source×period×length within mutually exclusive risk class",
                }
            )
            selected.append(candidate)

    population_summary = {
        "rows": int(sum(cell_counts.values())),
        "flags": dict(sorted(flag_population.items())),
        "source_year_pairs": [
            {"media": key.split("\x1f")[0], "year": key.split("\x1f")[1], "pairs": value}
            for key, value in sorted(source_year_population.items())
        ],
    }
    return selected, dict(cell_counts), population_summary


def attach_full_records(input_path: Path, selected: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    wanted = {row["pair_id"] for row in selected}
    found: dict[str, dict[str, Any]] = {}
    with input_path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            pair_id = str(record.get("id") or "")
            if pair_id in wanted:
                found[pair_id] = record
                if len(found) == len(wanted):
                    break
    missing = wanted - set(found)
    if missing:
        raise ValueError(f"Selected IDs missing from input: {sorted(missing)[:5]}")
    return found


def validate_selection(
    selected: list[dict[str, Any]],
    expected: dict[str, int],
    require_full_dimension_coverage: bool = True,
) -> dict[str, Any]:
    ids = [row["pair_id"] for row in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("Selected pair_id values are not unique")
    split_counts = Counter(row["split"] for row in selected)
    if dict(split_counts) != expected:
        raise ValueError(f"Unexpected split counts: {dict(split_counts)} != {expected}")
    event_splits: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        event_splits[row["event_group_id"]].add(row["split"])
    leakage = {event: splits for event, splits in event_splits.items() if len(splits) > 1}
    if leakage:
        raise ValueError(f"Event-group leakage detected: {list(leakage.items())[:3]}")
    dimensions = {
        "media": sorted({row["media"] for row in selected}),
        "period": sorted({row["period"] for row in selected}),
        "length_bin": sorted({row["length_bin"] for row in selected}),
        "risk_class": sorted({row["risk_class"] for row in selected}),
    }
    expected_dimensions = {
        "media": ["interfax", "kommersant", "prime", "rbc"],
        "period": ["early_through_2015", "late_2022_onward", "middle_2016_2021"],
        "length_bin": ["long_gt_2400", "medium_801_2400", "short_le_800"],
        "risk_class": sorted(RISK_ORDER),
    }
    if require_full_dimension_coverage and dimensions != expected_dimensions:
        raise ValueError(f"Mandatory dimension coverage failed: {dimensions}")
    return {
        "unique_pair_ids": len(ids),
        "unique_event_groups": len(event_splits),
        "split_counts": dict(sorted(split_counts.items())),
        "event_group_leakage": 0,
        "dimension_coverage": dimensions,
    }


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(
                json.dumps({key: row.get(key) for key in fields}, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
    os.replace(partial, path)


def atomic_write_csv(path: Path, fields: Iterable[str], rows: list[dict[str, Any]]) -> None:
    field_list = list(fields)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=field_list, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in field_list})
    os.replace(partial, path)


def build(
    input_path: Path,
    input_manifest_path: Path,
    output_dir: Path,
    seed: int,
    risk_quotas: dict[str, dict[str, int]],
    overwrite: bool,
    require_full_dimension_coverage: bool = True,
) -> dict[str, Any]:
    input_path = input_path.resolve(strict=True)
    input_manifest_path = input_manifest_path.resolve(strict=True)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = output_dir / "human_validation_selection_v1.jsonl"
    form_a_path = output_dir / "annotation_form_annotator_a.csv"
    form_b_path = output_dir / "annotation_form_annotator_b.csv"
    adjudication_path = output_dir / "adjudication_template.csv"
    manifest_path = output_dir / "human_validation_sample_v1.manifest.json"
    outputs = (selection_path, form_a_path, form_b_path, adjudication_path, manifest_path)
    for path in outputs:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output exists: {path}")

    declared_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    declared_hash = declared_manifest["outputs"]["jsonl"]["sha256"]
    input_hash, input_rows, article_counts, article_events = first_pass(input_path)
    if input_hash != declared_hash:
        raise ValueError(f"C04a input hash mismatch: {input_hash} != {declared_hash}")

    selected, cell_counts, population_summary = second_pass(
        input_path, seed, article_counts, risk_quotas
    )
    expected_split_counts = {
        split: sum(quotas.values()) for split, quotas in risk_quotas.items()
    }
    qa = validate_selection(
        selected, expected_split_counts, require_full_dimension_coverage
    )
    full_records = attach_full_records(input_path, selected)

    selected.sort(key=lambda row: stable_int(SAMPLE_VERSION, seed, "row-order", row["pair_id"]))
    for row in selected:
        row["row_id"] = "HV-" + hashlib.sha256(
            f"{SAMPLE_VERSION}\x1f{seed}\x1f{row['pair_id']}".encode("utf-8")
        ).hexdigest()[:12].upper()

    selection_rows = [{key: row.get(key) for key in SELECTION_FIELDS} for row in selected]
    form_rows: list[dict[str, Any]] = []
    for row in selected:
        record = full_records[row["pair_id"]]
        form_row = {
            "row_id": row["row_id"],
            "source": record.get("media"),
            "published_at": record.get("published_at"),
            "target_company": record.get("target_company"),
            "title": record.get("title") or "",
            "text": record.get("text") or "",
            **{field: "" for field in EMPTY_ANNOTATION_FIELDS},
        }
        form_rows.append(form_row)
    form_a = [{**row, "annotator_id": "annotator_a"} for row in form_rows]
    form_b = [{**row, "annotator_id": "annotator_b"} for row in form_rows]
    adjudication = [
        {
            "row_id": row["row_id"],
            "pair_id": row["pair_id"],
            "split": row["split"],
            **{field: "" for field in ADJUDICATION_FIELDS[3:]},
        }
        for row in selected
    ]

    atomic_write_jsonl(selection_path, selection_rows, SELECTION_FIELDS)
    atomic_write_csv(form_a_path, FORM_FIELDS, form_a)
    atomic_write_csv(form_b_path, FORM_FIELDS, form_b)
    atomic_write_csv(adjudication_path, ADJUDICATION_FIELDS, adjudication)

    distribution: dict[str, Any] = {}
    for dimension in (
        "split",
        "media",
        "period",
        "length_bin",
        "risk_class",
    ):
        distribution[dimension] = dict(
            sorted(Counter(row[dimension] for row in selected).items())
        )
    for flag in (
        "rbc_object_id_date",
        "ambiguous_short_target",
        "multi_target_article",
        "truncated",
        "target_literal_absent",
    ):
        distribution[flag] = int(sum(bool(row[flag]) for row in selected))

    population_cells = []
    selected_by_cell = Counter(
        (row["split"], row["risk_class"], row["media"], row["period"], row["length_bin"])
        for row in selected
    )
    for cell, population in sorted(cell_counts.items()):
        sample_n = selected_by_cell[cell]
        population_cells.append(
            {
                "split": cell[0],
                "risk_class": cell[1],
                "media": cell[2],
                "period": cell[3],
                "length_bin": cell[4],
                "population": population,
                "selected": sample_n,
                "within_stratum_probability": sample_n / population if sample_n else 0.0,
            }
        )

    form_content_digest = hashlib.sha256()
    for row in form_rows:
        form_content_digest.update(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
    manifest = {
        "sample_version": SAMPLE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "input": {
            "jsonl": fingerprint(input_path, input_hash),
            "manifest": fingerprint(input_manifest_path),
            "declared_corpus_version": declared_manifest.get("corpus_version"),
            "rows": input_rows,
        },
        "event_group_proxy": {
            "version": EVENT_PROXY_VERSION,
            "definition": (
                "SHA-256 of Unicode-casefolded alphanumeric tokens in the first 1200 "
                "normalized characters of visible body text; title is fallback only when body is empty"
            ),
            "article_urls_checked": len(article_events),
            "normalized_media_url_conflicts": 0,
            "limitations": [
                "proxy groups exact/near-exact visible leads but is not final semantic event deduplication",
                "reprints with changed leads may remain separate; templated identical leads may be grouped",
                "only the stored excerpt is available, so differences after the excerpt are invisible",
            ],
        },
        "split": {
            "assignment": "event-group hash before pair sampling",
            "probabilities": SPLIT_PROBABILITIES,
            "heldout_rule": (
                "heldout is not used to change prompt or rules; any such change requires a new independent test set"
            ),
        },
        "strata": {
            "period": {
                "early_through_2015": "year <= 2015",
                "middle_2016_2021": "2016 <= year <= 2021",
                "late_2022_onward": "year >= 2022",
            },
            "visible_length": {
                "short_le_800": "stored text characters <= 800",
                "medium_801_2400": "801..2400",
                "long_gt_2400": "> 2400",
            },
            "risk_priority": list(RISK_ORDER),
            "ambiguous_short_target": "one normalized alphanumeric token of at most five characters",
            "multi_target_article": "more than one target pair for normalized media+URL",
            "rbc_object_id_date_sources": sorted(RBC_OBJECT_ID_SOURCES),
            "risk_quotas": risk_quotas,
            "population_cells": population_cells,
        },
        "sampling": {
            "method": (
                "simple deterministic hash-priority sample within mutually exclusive "
                "split×risk×source×period×length cells"
            ),
            "cell_allocation": (
                "fixed risk quotas; balanced source, then period, then length weights with capacity caps"
            ),
            "probability": (
                "overall_inclusion_probability = event-split probability × selected_cell_n / cell_population_N"
            ),
            "design_weight": "inverse overall inclusion probability",
        },
        "population_summary": population_summary,
        "selection_distribution": distribution,
        "qa": {
            **qa,
            "forms_content_sha256_excluding_annotator_id": form_content_digest.hexdigest(),
            "annotation_fields_empty": True,
            "llm_returns_holdings_fields_absent": True,
            "selected_ids_verified_in_input": True,
        },
        "artifacts": {
            "selection": fingerprint(selection_path),
            "annotator_a": fingerprint(form_a_path),
            "annotator_b": fingerprint(form_b_path),
            "adjudication": fingerprint(adjudication_path),
        },
        "notes": [
            "selection contains no LLM labels, returns, holdings or future outcomes",
            "annotation forms are identical except annotator_id and contain empty label/note/confidence fields",
            "300 is an initial validation sample, not an automatically sufficient final sample",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    manifest = build(
        args.input,
        args.input_manifest,
        args.output_dir,
        args.seed,
        DEFAULT_RISK_QUOTAS,
        args.overwrite,
    )
    print(
        json.dumps(
            {
                "sample_version": manifest["sample_version"],
                "seed": manifest["seed"],
                "selection_distribution": manifest["selection_distribution"],
                "qa": manifest["qa"],
                "artifacts": manifest["artifacts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
