#!/usr/bin/env python3
"""Audit full article lengths and tokenize every frozen C04a prompt.

The text audit joins the immutable C04a article keys back to the exact source
Parquet used by the parent v1 build.  It stores lengths and identifiers only,
never full article bodies.  The token audit calls the running llama.cpp
``/tokenize`` endpoint for both frozen prompt variants for every input row.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import importlib.util
import json
import math
import os
import re
import threading
import time
from array import array
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MEDIA = ("interfax", "kommersant", "prime", "rbc")
CONTEXT_THRESHOLDS = (4096, 8192, 16384, 32768)
SPACE_RE = re.compile(r"\s+")
_thread_local = threading.local()


def load_base(script_dir: Path):
    path = script_dir / "31_run_target_llm_sentiment.py"
    spec = importlib.util.spec_from_file_location("full_token_audit_base", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return SPACE_RE.sub(" ", str(value)).strip()


def normalize_media(value: Any) -> str:
    return clean_text(value).lower()


def normalize_link(value: Any) -> str:
    return clean_text(value).rstrip("/").lower()


def article_key(media: Any, link: Any) -> str:
    return f"{normalize_media(media)}\x1f{normalize_link(link)}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nearest_rank(sorted_values: list[int] | array, probability: float) -> int:
    if not sorted_values:
        raise ValueError("Cannot summarize an empty sequence")
    index = min(len(sorted_values) - 1, max(0, math.ceil(probability * len(sorted_values)) - 1))
    return int(sorted_values[index])


def summarize(values: list[int] | array) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": int(ordered[0]),
        "p50": nearest_rank(ordered, 0.50),
        "p90": nearest_rank(ordered, 0.90),
        "p95": nearest_rank(ordered, 0.95),
        "p99": nearest_rank(ordered, 0.99),
        "p99_9": nearest_rank(ordered, 0.999),
        "max": int(ordered[-1]),
        "sum": int(sum(ordered)),
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def compact_fingerprint(path: Path, include_hash: bool) -> dict[str, Any]:
    stat = path.stat()
    value = {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }
    if include_hash:
        value["sha256"] = sha256_file(path)
    return value


def push_top(heap: list, score: int, identifier: str, row: dict[str, Any], capacity: int) -> None:
    entry = (score, identifier, row)
    if len(heap) < capacity:
        heapq.heappush(heap, entry)
    elif (score, identifier) > (heap[0][0], heap[0][1]):
        heapq.heapreplace(heap, entry)


def first_present(names: list[str], candidates: tuple[str, ...], required: bool = True) -> str | None:
    result = next((name for name in candidates if name in names), None)
    if required and result is None:
        raise ValueError(f"None of {candidates} found in Parquet columns {names}")
    return result


def audit_text(args: argparse.Namespace) -> dict[str, Any]:
    import pyarrow.parquet as pq

    started = time.monotonic()
    articles: dict[str, dict[str, Any]] = {}
    pair_rows = 0
    pair_conflicts = 0
    with args.input.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            pair_rows += 1
            media = normalize_media(record.get("media"))
            if media not in MEDIA:
                raise ValueError(f"Disallowed media in C04a: {media}")
            key = article_key(media, record.get("link"))
            value = {
                "media": media,
                "link": record.get("link"),
                "published_at": record.get("published_at"),
                "pair_id": record.get("id"),
                "saved_chars": len(str(record.get("text") or "")),
                "saved_bytes": len(str(record.get("text") or "").encode("utf-8")),
                "declared_original_chars": int(record.get("text_chars_original") or 0),
                "truncated": bool(record.get("text_truncated")),
            }
            previous = articles.setdefault(key, value)
            if any(previous[field] != value[field] for field in ("saved_chars", "saved_bytes", "declared_original_chars", "truncated")):
                pair_conflicts += 1

    parquet = pq.ParquetFile(args.articles)
    names = parquet.schema_arrow.names
    media_col = first_present(names, ("media", "source_media"))
    link_col = first_present(names, ("link", "url", "fronturl"))
    text_col = first_present(names, ("text", "full_text", "body", "content"))
    date_col = first_present(names, ("published_at", "publish_date", "date"), required=False)
    id_col = first_present(names, ("article_id", "id", "object_id"), required=False)
    columns = list(dict.fromkeys(x for x in (media_col, link_col, text_col, date_col, id_col) if x))

    groups: dict[str, dict[str, array]] = defaultdict(lambda: {
        "full_chars": array("Q"), "full_bytes": array("Q"),
        "saved_chars": array("Q"), "saved_bytes": array("Q"),
    })
    counters: Counter[str] = Counter()
    top: list = []
    matched_keys: set[str] = set()
    original_mismatch_examples: list[dict[str, Any]] = []
    for batch in parquet.iter_batches(columns=columns, batch_size=args.batch_size):
        values = {name: batch.column(batch.schema.get_field_index(name)).to_pylist() for name in columns}
        for index in range(batch.num_rows):
            media = normalize_media(values[media_col][index])
            if media not in MEDIA:
                continue
            link = values[link_col][index]
            key = article_key(media, link)
            c04a = articles.get(key)
            if c04a is None:
                continue
            if key in matched_keys:
                counters["duplicate_snapshot_article_key"] += 1
                continue
            matched_keys.add(key)
            full_text = clean_text(values[text_col][index])
            full_chars = len(full_text)
            full_bytes = len(full_text.encode("utf-8"))
            saved_chars = c04a["saved_chars"]
            saved_bytes = c04a["saved_bytes"]
            for group_name in ("all", media):
                group = groups[group_name]
                group["full_chars"].append(full_chars)
                group["full_bytes"].append(full_bytes)
                group["saved_chars"].append(saved_chars)
                group["saved_bytes"].append(saved_bytes)
            counters["matched_articles"] += 1
            counters["truncated_articles"] += c04a["truncated"]
            counters["full_chars"] += full_chars
            counters["saved_chars"] += saved_chars
            counters["omitted_chars"] += max(0, full_chars - saved_chars)
            counters["full_bytes"] += full_bytes
            counters["saved_bytes"] += saved_bytes
            counters["omitted_bytes"] += max(0, full_bytes - saved_bytes)
            if c04a["declared_original_chars"] != full_chars:
                counters["text_chars_original_mismatch"] += 1
                if len(original_mismatch_examples) < 20:
                    original_mismatch_examples.append({
                        "pair_id": c04a["pair_id"], "media": media, "link": c04a["link"],
                        "declared": c04a["declared_original_chars"], "actual": full_chars,
                    })
            article_id = values[id_col][index] if id_col else None
            published = values[date_col][index] if date_col else c04a["published_at"]
            row = {
                "article_id": None if article_id is None else str(article_id),
                "pair_id_example": c04a["pair_id"],
                "media": media,
                "link": c04a["link"],
                "published_at": None if published is None else str(published),
                "full_chars": full_chars,
                "full_bytes": full_bytes,
                "saved_chars": saved_chars,
                "saved_bytes": saved_bytes,
                "omitted_chars": max(0, full_chars - saved_chars),
                "truncated": c04a["truncated"],
            }
            push_top(top, full_chars, key, row, args.top_n)

    if pair_rows != args.expected_pairs:
        raise ValueError(f"C04a pair count mismatch: {pair_rows} != {args.expected_pairs}")
    if pair_conflicts:
        raise ValueError(f"C04a has {pair_conflicts} within-article text metadata conflicts")
    if len(matched_keys) != len(articles):
        raise ValueError(f"Snapshot join incomplete: {len(matched_keys)} != {len(articles)}")

    result = {
        "version": "c05-full-text-audit-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "c04a": compact_fingerprint(args.input, include_hash=True),
        "articles_snapshot": compact_fingerprint(args.articles, include_hash=args.hash_articles),
        "expected_pairs": args.expected_pairs,
        "unique_c04a_articles": len(articles),
        "parquet_rows": parquet.metadata.num_rows,
        "parquet_columns": names,
        "selected_columns": {"media": media_col, "link": link_col, "text": text_col, "date": date_col, "article_id": id_col},
        "counters": dict(sorted(counters.items())),
        "truncation": {
            "articles": len(articles),
            "truncated_articles": counters["truncated_articles"],
            "truncated_share": counters["truncated_articles"] / len(articles),
            "omitted_chars": counters["omitted_chars"],
            "omitted_bytes": counters["omitted_bytes"],
            "retained_char_share": counters["saved_chars"] / counters["full_chars"],
            "retained_byte_share": counters["saved_bytes"] / counters["full_bytes"],
        },
        "by_group": {
            name: {metric: summarize(values) for metric, values in metrics.items()}
            for name, metrics in sorted(groups.items())
        },
        "top_longest": [entry[2] for entry in sorted(top, reverse=True)],
        "text_chars_original_mismatch_examples": original_mismatch_examples,
    }
    atomic_json(args.output, result)
    return result


def session():
    import requests

    if not hasattr(_thread_local, "session"):
        value = requests.Session()
        value.trust_env = False
        _thread_local.session = value
    return _thread_local.session


def tokenize_record(base: Any, endpoint: str, timeout: int, record: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
    client = session()
    primary = client.post(endpoint, json={"content": base.build_prompt(record)}, timeout=timeout)
    primary.raise_for_status()
    corrective = client.post(endpoint, json={"content": base.build_prompt(record, corrective=True)}, timeout=timeout)
    corrective.raise_for_status()
    compact = {
        "id": str(record["id"]),
        "media": str(record["media"]),
        "truncated": bool(record.get("text_truncated")),
        "text_chars_original": record.get("text_chars_original"),
        "visible_text_chars": len(str(record.get("text") or "")),
    }
    return compact, len(primary.json()["tokens"]), len(corrective.json()["tokens"])


def audit_tokens(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    base = load_base(args.script_dir)
    values: dict[str, dict[str, array]] = defaultdict(lambda: {"primary": array("I"), "corrective": array("I")})
    overflows = {
        str(context): {"primary": 0, "corrective": 0}
        for context in CONTEXT_THRESHOLDS
    }
    top_primary: list = []
    top_corrective: list = []
    count = 0

    def consume(result: tuple[dict[str, Any], int, int]) -> None:
        nonlocal count
        compact, primary, corrective = result
        count += 1
        group_names = ("all", f"media={compact['media']}", f"truncated={str(compact['truncated']).lower()}")
        for group_name in group_names:
            values[group_name]["primary"].append(primary)
            values[group_name]["corrective"].append(corrective)
        for context in CONTEXT_THRESHOLDS:
            overflows[str(context)]["primary"] += primary + args.primary_output_tokens > context
            overflows[str(context)]["corrective"] += corrective + args.corrective_output_tokens > context
        primary_row = {**compact, "prompt_tokens": primary, "output_budget": args.primary_output_tokens, "total_budget": primary + args.primary_output_tokens}
        corrective_row = {**compact, "prompt_tokens": corrective, "output_budget": args.corrective_output_tokens, "total_budget": corrective + args.corrective_output_tokens}
        push_top(top_primary, primary + args.primary_output_tokens, compact["id"], primary_row, args.top_n)
        push_top(top_corrective, corrective + args.corrective_output_tokens, compact["id"], corrective_row, args.top_n)
        if count % args.log_every == 0:
            elapsed = time.monotonic() - started
            print(json.dumps({"processed": count, "pairs_per_second": count / elapsed, "elapsed_seconds": elapsed}), flush=True)

    pending = set()
    submitted = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor, args.input.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if normalize_media(record.get("media")) not in MEDIA:
                raise ValueError(f"Disallowed media in token input: {record.get('media')}")
            pending.add(executor.submit(tokenize_record, base, args.endpoint, args.timeout_seconds, record))
            submitted += 1
            if len(pending) >= args.workers * args.queue_multiplier:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    consume(future.result())
            if args.limit and submitted >= args.limit:
                break
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                consume(future.result())

    expected_count = args.limit or args.expected_pairs
    if count != expected_count:
        raise ValueError(f"Token audit count mismatch: {count} != {expected_count}")
    elapsed = time.monotonic() - started
    result = {
        "version": "c05-full-token-audit-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": compact_fingerprint(args.input, include_hash=True),
        "endpoint": args.endpoint,
        "workers": args.workers,
        "records": count,
        "elapsed_seconds": elapsed,
        "pairs_per_second": count / elapsed,
        "primary_output_tokens": args.primary_output_tokens,
        "corrective_output_tokens": args.corrective_output_tokens,
        "statistics": {
            group: {variant: summarize(metric_values) for variant, metric_values in variants.items()}
            for group, variants in sorted(values.items())
        },
        "overflow_counts": overflows,
        "top_primary_total_budget": [entry[2] for entry in sorted(top_primary, reverse=True)],
        "top_corrective_total_budget": [entry[2] for entry in sorted(top_corrective, reverse=True)],
    }
    atomic_json(args.output, result)
    print(json.dumps({"completed": count, "output": str(args.output), "pairs_per_second": count / elapsed}), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    text = subparsers.add_parser("text")
    text.add_argument("--input", required=True, type=Path)
    text.add_argument("--articles", required=True, type=Path)
    text.add_argument("--output", required=True, type=Path)
    text.add_argument("--expected-pairs", type=int, default=982273)
    text.add_argument("--batch-size", type=int, default=10000)
    text.add_argument("--top-n", type=int, default=20)
    text.add_argument("--hash-articles", action="store_true")
    tokens = subparsers.add_parser("tokens")
    tokens.add_argument("--input", required=True, type=Path)
    tokens.add_argument("--script-dir", required=True, type=Path)
    tokens.add_argument("--endpoint", default="http://127.0.0.1:8081/tokenize")
    tokens.add_argument("--output", required=True, type=Path)
    tokens.add_argument("--expected-pairs", type=int, default=982273)
    tokens.add_argument("--workers", type=int, default=16)
    tokens.add_argument("--queue-multiplier", type=int, default=8)
    tokens.add_argument("--timeout-seconds", type=int, default=60)
    tokens.add_argument("--primary-output-tokens", type=int, default=112)
    tokens.add_argument("--corrective-output-tokens", type=int, default=192)
    tokens.add_argument("--top-n", type=int, default=20)
    tokens.add_argument("--log-every", type=int, default=10000)
    tokens.add_argument("--limit", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output.exists():
        raise SystemExit(f"Output exists; refusing to overwrite: {args.output}")
    if args.mode == "text":
        result = audit_text(args)
        print(json.dumps({"articles": result["unique_c04a_articles"], "output": str(args.output)}))
    else:
        audit_tokens(args)


if __name__ == "__main__":
    main()
