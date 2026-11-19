#!/usr/bin/env python3
"""Build and audit the frozen four-source target-level LLM corpus for C04a.

The parent JSONL is treated as immutable.  Accepted records are copied byte for
byte, in their original order, and retain the parent ``id``.  Frank Media rows
are excluded from the derived corpus but are still audited in the manifest.
No model is called by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PARENT_INPUT_VERSION = "target-llm-input-v1"
CORPUS_VERSION = "target-llm-c04a-four-sources-v1"
ALLOWED_MEDIA = ("interfax", "kommersant", "prime", "rbc")
REQUIRED_FIELDS = (
    "id",
    "media",
    "link",
    "target_company",
    "published_at",
    "date_source",
    "date_precision",
    "title",
    "text",
    "text_chars_original",
    "text_truncated",
)
OUTPUT_METADATA_FIELDS = (
    "media",
    "link",
    "target_company",
    "published_at",
    "date_source",
    "date_precision",
    "title",
    "text_chars_original",
    "text_truncated",
)
OBJECT_ID_DATE_SOURCES = ("rbc_freenews_object_id", "rbc_object_id_timestamp")
WHITESPACE_RE = re.compile(r"\s+")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_media(value: Any) -> str:
    return WHITESPACE_RE.sub(" ", str(value or "")).strip().lower()


def normalize_link(value: Any) -> str:
    return WHITESPACE_RE.sub(" ", str(value or "")).strip().rstrip("/").lower()


def natural_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        normalize_media(record.get("media")),
        normalize_link(record.get("link")),
        WHITESPACE_RE.sub(" ", str(record.get("target_company") or "")).strip(),
    )


def expected_pair_id(record: dict[str, Any]) -> str:
    media, link, company = natural_key(record)
    payload = f"{PARENT_INPUT_VERSION}\x1f{media}\x1f{link}\x1f{company}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def is_object_id_date(record: dict[str, Any]) -> bool:
    return (
        normalize_media(record.get("media")) == "rbc"
        and record.get("date_source") in OBJECT_ID_DATE_SOURCES
    )


def year_of(record: dict[str, Any]) -> str:
    value = str(record.get("published_at") or "")
    return value[:4] if len(value) >= 4 and value[:4].isdigit() else "missing"


def target_literal_absent(record: dict[str, Any]) -> bool:
    target = WHITESPACE_RE.sub(" ", str(record.get("target_company") or "")).strip().casefold()
    prompt_text = f"{record.get('title') or ''} {record.get('text') or ''}".casefold()
    return bool(target) and target not in prompt_text


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record.get(key)
        for key in (
            "id",
            "media",
            "link",
            "target_company",
            "published_at",
            "date_source",
            "date_precision",
            "text_chars_original",
            "text_truncated",
        )
    }


def add_example(container: dict[str, list[dict[str, Any]]], name: str, record: dict[str, Any]) -> None:
    if len(container[name]) < 3:
        container[name].append(compact_record(record))


def fingerprint(path: Path, digest: str | None = None) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": digest or sha256_file(path),
    }


def load_llm_outputs(output_dir: Path | None) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if output_dir is None:
        return {}, {"provided": False}
    output_dir = output_dir.resolve(strict=True)
    files = sorted(output_dir.glob("part_*.jsonl")) + sorted(output_dir.glob("part_*.jsonl.partial"))
    records: dict[str, dict[str, Any]] = {}
    counters: Counter[str] = Counter()
    duplicate_examples: list[str] = []
    malformed_examples: list[dict[str, Any]] = []
    for path in files:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                counters["records"] += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    counters["malformed_json"] += 1
                    if len(malformed_examples) < 5:
                        malformed_examples.append(
                            {"file": path.name, "line": line_number, "error": str(exc)}
                        )
                    continue
                pair_id = str(record.get("id") or "")
                if not pair_id:
                    counters["missing_id"] += 1
                    continue
                if pair_id in records:
                    counters["duplicate_id"] += 1
                    if len(duplicate_examples) < 5:
                        duplicate_examples.append(pair_id)
                    continue
                record["_source_file"] = path.name
                record["_matched_parent"] = False
                records[pair_id] = record
                counters[f"status_{record.get('status') or 'missing'}"] += 1
    run_config_path = output_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8")) if run_config_path.exists() else None
    return records, {
        "provided": True,
        "path": str(output_dir),
        "files": [path.name for path in files],
        "scan": dict(sorted(counters.items())),
        "duplicate_id_examples": duplicate_examples,
        "malformed_examples": malformed_examples,
        "run_config": run_config,
    }


def bump(summary: dict[str, Counter[str]], key: str, record: dict[str, Any]) -> None:
    row = summary[key]
    row["pairs"] += 1
    row["has_text"] += bool(record.get("text"))
    row["has_title"] += bool(record.get("title"))
    row["has_published_at"] += bool(record.get("published_at"))
    row["object_id_inferred_date"] += is_object_id_date(record)
    row["truncated"] += bool(record.get("text_truncated"))
    row["target_literal_absent"] += target_literal_absent(record)


def serialize_source_year(summary: dict[str, Counter[str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(summary):
        media, year = key.split("\x1f", 1)
        rows.append({"media": media, "year": year, **dict(sorted(summary[key].items()))})
    return rows


def process(
    parent_input: Path,
    parent_manifest: Path,
    output: Path,
    manifest_path: Path,
    sensitivity_mask: Path,
    llm_output_dir: Path | None,
    overwrite: bool,
) -> dict[str, Any]:
    parent_input = parent_input.resolve(strict=True)
    parent_manifest = parent_manifest.resolve(strict=True)
    output = output.resolve()
    manifest_path = manifest_path.resolve()
    sensitivity_mask = sensitivity_mask.resolve()
    for path in (output, manifest_path, sensitivity_mask):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    llm_outputs, llm_report = load_llm_outputs(llm_output_dir)
    run_config = llm_report.get("run_config") or {}
    expected_runner = run_config.get("runner_version")
    expected_prompt = run_config.get("prompt_version")

    parent_hash = hashlib.sha256()
    output_hash = hashlib.sha256()
    counters: Counter[str] = Counter()
    parent_missing: Counter[str] = Counter()
    output_missing: Counter[str] = Counter()
    by_media: Counter[str] = Counter()
    output_by_media: Counter[str] = Counter()
    pair_year_parent: dict[str, Counter[str]] = defaultdict(Counter)
    pair_year_output: dict[str, Counter[str]] = defaultdict(Counter)
    article_year_parent: dict[str, Counter[str]] = defaultdict(Counter)
    article_year_output: dict[str, Counter[str]] = defaultdict(Counter)
    date_source: Counter[str] = Counter()
    date_precision: Counter[str] = Counter()
    output_date_source: Counter[str] = Counter()
    output_date_precision: Counter[str] = Counter()
    seen_ids: set[bytes] = set()
    seen_natural: set[bytes] = set()
    article_counts: dict[bytes, int] = {}
    output_article_counts: dict[bytes, int] = {}
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reuse: Counter[str] = Counter()
    metadata_conflict_examples: list[dict[str, Any]] = []

    output_partial = output.with_suffix(output.suffix + ".partial")
    mask_partial = sensitivity_mask.with_suffix(sensitivity_mask.suffix + ".partial")
    for path in (output_partial, mask_partial):
        if path.exists():
            if not overwrite:
                raise FileExistsError(f"Partial output exists: {path}")
            path.unlink()

    with (
        parent_input.open("rb") as source,
        output_partial.open("wb") as destination,
        mask_partial.open("w", encoding="utf-8") as mask,
    ):
        for line_number, raw_line in enumerate(source, start=1):
            parent_hash.update(raw_line)
            if not raw_line.strip():
                counters["blank_lines"] += 1
                continue
            counters["parent_rows"] += 1
            try:
                record = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Malformed parent JSONL at line {line_number}: {exc}") from exc

            for field in REQUIRED_FIELDS:
                if field not in record or record[field] in (None, ""):
                    parent_missing[field] += 1
            media, link, company = natural_key(record)
            by_media[media or "missing"] += 1
            date_source[str(record.get("date_source") or "missing")] += 1
            date_precision[str(record.get("date_precision") or "missing")] += 1
            year = year_of(record)
            parent_group = f"{media or 'missing'}\x1f{year}"
            bump(pair_year_parent, parent_group, record)

            article_digest = hashlib.blake2b(f"{media}\x1f{link}".encode("utf-8"), digest_size=16).digest()
            if article_digest not in article_counts:
                article_counts[article_digest] = 0
                article_year_parent[parent_group]["articles"] += 1
                article_year_parent[parent_group]["has_text"] += bool(record.get("text"))
                article_year_parent[parent_group]["has_title"] += bool(record.get("title"))
                article_year_parent[parent_group]["has_published_at"] += bool(record.get("published_at"))
                article_year_parent[parent_group]["object_id_inferred_date"] += is_object_id_date(record)
                article_year_parent[parent_group]["truncated"] += bool(record.get("text_truncated"))
            article_counts[article_digest] += 1
            if article_counts[article_digest] == 2:
                counters["parent_multi_target_articles"] += 1
                add_example(examples, "multi_target_article", record)

            pair_id = str(record.get("id") or "")
            try:
                id_digest = bytes.fromhex(pair_id)
            except ValueError:
                id_digest = hashlib.sha256(pair_id.encode("utf-8")).digest()
                counters["invalid_hex_pair_id"] += 1
            if id_digest in seen_ids:
                counters["duplicate_pair_id"] += 1
                add_example(examples, "duplicate_pair_id", record)
            else:
                seen_ids.add(id_digest)
            natural_digest = hashlib.sha256(f"{media}\x1f{link}\x1f{company}".encode("utf-8")).digest()
            if natural_digest in seen_natural:
                counters["duplicate_natural_key"] += 1
                add_example(examples, "duplicate_natural_key", record)
            else:
                seen_natural.add(natural_digest)
            if pair_id != expected_pair_id(record):
                counters["pair_id_mismatch"] += 1
                add_example(examples, "pair_id_mismatch", record)

            if media in ALLOWED_MEDIA:
                raw_output = llm_outputs.get(pair_id)
                if raw_output is not None:
                    raw_output["_matched_parent"] = True
                    conflicts = [
                        field
                        for field in OUTPUT_METADATA_FIELDS
                        if raw_output.get(field) != record.get(field)
                    ]
                    config_match = (
                        (not expected_runner or raw_output.get("runner_version") == expected_runner)
                        and (not expected_prompt or raw_output.get("prompt_version") == expected_prompt)
                    )
                    if conflicts:
                        reuse["allowed_metadata_conflict"] += 1
                        if len(metadata_conflict_examples) < 5:
                            metadata_conflict_examples.append({"id": pair_id, "fields": conflicts})
                    elif not config_match:
                        reuse["allowed_config_conflict"] += 1
                    elif raw_output.get("status") == "ok":
                        reuse["reusable_success"] += 1
                    elif raw_output.get("status") == "error":
                        reuse["allowed_error"] += 1
                    else:
                        reuse["allowed_unknown_status"] += 1
            else:
                raw_output = llm_outputs.get(pair_id)
                if raw_output is not None:
                    raw_output["_matched_parent"] = True
                    if raw_output.get("status") == "ok":
                        reuse["excluded_frank_success"] += 1
                    elif raw_output.get("status") == "error":
                        reuse["excluded_frank_error"] += 1
                    else:
                        reuse["excluded_other_status"] += 1

            if media not in ALLOWED_MEDIA:
                counters["excluded_rows"] += 1
                continue

            for field in REQUIRED_FIELDS:
                if field not in record or record[field] in (None, ""):
                    output_missing[field] += 1
            output_date_source[str(record.get("date_source") or "missing")] += 1
            output_date_precision[str(record.get("date_precision") or "missing")] += 1
            destination.write(raw_line)
            output_hash.update(raw_line)
            counters["output_rows"] += 1
            output_by_media[media] += 1
            bump(pair_year_output, parent_group, record)
            if article_digest not in output_article_counts:
                output_article_counts[article_digest] = 0
                article_year_output[parent_group]["articles"] += 1
                article_year_output[parent_group]["has_text"] += bool(record.get("text"))
                article_year_output[parent_group]["has_title"] += bool(record.get("title"))
                article_year_output[parent_group]["has_published_at"] += bool(record.get("published_at"))
                article_year_output[parent_group]["object_id_inferred_date"] += is_object_id_date(record)
                article_year_output[parent_group]["truncated"] += bool(record.get("text_truncated"))
            output_article_counts[article_digest] += 1
            if output_article_counts[article_digest] == 2:
                counters["output_multi_target_articles"] += 1
            add_example(examples, f"source_{media}", record)
            is_truncated = bool(record.get("text_truncated"))
            is_target_absent = target_literal_absent(record)
            if is_truncated:
                counters["output_truncated_rows"] += 1
                add_example(examples, "truncated_text", record)
            if is_target_absent:
                counters["target_literal_absent"] += 1
                counters[
                    "target_literal_absent_truncated"
                    if is_truncated
                    else "target_literal_absent_not_truncated"
                ] += 1
                add_example(examples, "target_literal_absent", record)
            if is_object_id_date(record):
                counters["sensitivity_object_id_rows"] += 1
                add_example(examples, "rbc_object_id_date", record)
                mask.write(json.dumps(compact_record(record), ensure_ascii=False, separators=(",", ":")) + "\n")

    os.replace(output_partial, output)
    os.replace(mask_partial, sensitivity_mask)

    unknown_outputs = [
        {"id": pair_id, "file": record.get("_source_file"), "status": record.get("status")}
        for pair_id, record in llm_outputs.items()
        if not record.get("_matched_parent")
    ]
    reuse["unknown_output_id"] = len(unknown_outputs)
    reuse["unprocessed_output_rows"] = counters["output_rows"] - reuse["reusable_success"]
    for key in (
        "reusable_success",
        "excluded_frank_success",
        "allowed_error",
        "excluded_frank_error",
        "allowed_metadata_conflict",
        "allowed_config_conflict",
        "allowed_unknown_status",
        "excluded_other_status",
        "unknown_output_id",
    ):
        reuse.setdefault(key, 0)
    llm_report["mapping"] = dict(sorted(reuse.items()))
    llm_report["metadata_conflict_examples"] = metadata_conflict_examples
    llm_report["unknown_output_examples"] = unknown_outputs[:5]

    for key in (
        "blank_lines",
        "duplicate_pair_id",
        "duplicate_natural_key",
        "invalid_hex_pair_id",
        "pair_id_mismatch",
        "output_truncated_rows",
        "output_multi_target_articles",
        "parent_multi_target_articles",
        "sensitivity_object_id_rows",
        "target_literal_absent",
        "target_literal_absent_not_truncated",
        "target_literal_absent_truncated",
    ):
        counters.setdefault(key, 0)
    for field in REQUIRED_FIELDS:
        parent_missing.setdefault(field, 0)
        output_missing.setdefault(field, 0)

    duplicate_failures = {
        key: counters[key]
        for key in ("duplicate_pair_id", "duplicate_natural_key", "pair_id_mismatch")
        if counters[key]
    }
    manifest = {
        "corpus_version": CORPUS_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent": {
            "input": fingerprint(parent_input, parent_hash.hexdigest()),
            "manifest": fingerprint(parent_manifest),
            "declared_manifest": json.loads(parent_manifest.read_text(encoding="utf-8")),
        },
        "derivation": {
            "algorithm": "stream parent JSONL; copy original line bytes iff lower(media) is in allowed_media",
            "allowed_media": list(ALLOWED_MEDIA),
            "excluded_media": ["frank"],
            "order_preserved": True,
            "records_reserialized": False,
            "pair_id_preserved": True,
            "new_archives_included": False,
            "object_id_dates_main_corpus": True,
            "object_id_sensitivity_rule": (
                "exclude rows where media='rbc' and date_source is one of "
                f"{list(OBJECT_ID_DATE_SOURCES)}"
            ),
        },
        "schema": {
            "required_fields": list(REQUIRED_FIELDS),
            "parent_missing_values": dict(sorted(parent_missing.items())),
            "output_missing_values": dict(sorted(output_missing.items())),
        },
        "counts": dict(sorted(counters.items())),
        "parent_rows_by_media": dict(sorted(by_media.items())),
        "output_rows_by_media": dict(sorted(output_by_media.items())),
        "date_source": dict(sorted(date_source.items())),
        "date_precision": dict(sorted(date_precision.items())),
        "output_date_source": dict(sorted(output_date_source.items())),
        "output_date_precision": dict(sorted(output_date_precision.items())),
        "uniqueness": {
            "pair_ids_seen": len(seen_ids),
            "natural_keys_seen": len(seen_natural),
            "failures": duplicate_failures,
            "natural_key_definition": "lower(media) + normalized URL + exact trimmed target_company",
        },
        "source_year_pairs_parent": serialize_source_year(pair_year_parent),
        "source_year_articles_parent": serialize_source_year(article_year_parent),
        "source_year_pairs_output": serialize_source_year(pair_year_output),
        "source_year_articles_output": serialize_source_year(article_year_output),
        "outputs": {
            "jsonl": fingerprint(output, output_hash.hexdigest()),
            "object_id_sensitivity_mask": fingerprint(sensitivity_mask),
            "manifest": str(manifest_path),
        },
        "llm_output_reuse": llm_report,
        "qa_examples": dict(sorted(examples.items())),
        "notes": [
            "article counts use normalized media+URL; they are not counts of unique news events or original reporting",
            "target_literal_absent is a case-insensitive literal heuristic and is not a relevance judgment",
            "text truncation and target_literal_absent are audited but do not remove rows",
            "LLM status='ok' establishes schema-valid output only, not sentiment quality",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if duplicate_failures:
        raise ValueError(f"Derived corpus failed uniqueness contract: {duplicate_failures}")
    return manifest


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-input", required=True, type=Path)
    parser.add_argument("--parent-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--sensitivity-mask", required=True, type=Path)
    parser.add_argument("--llm-output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    manifest = process(
        args.parent_input,
        args.parent_manifest,
        args.output,
        args.manifest,
        args.sensitivity_mask,
        args.llm_output_dir,
        args.overwrite,
    )
    print(
        json.dumps(
            {
                "corpus_version": manifest["corpus_version"],
                "counts": manifest["counts"],
                "outputs": manifest["outputs"],
                "llm_output_reuse": manifest["llm_output_reuse"].get("mapping"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
