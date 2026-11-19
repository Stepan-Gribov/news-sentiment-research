#!/usr/bin/env python3
"""Measure frozen prompt token counts through the running llama.cpp tokenizer endpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import requests


def load_base(script_dir: Path):
    path = script_dir / "31_run_target_llm_sentiment.py"
    spec = importlib.util.spec_from_file_location("token_audit_base", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def rank(seed: int, pair_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"token-audit-v1\x1f{seed}\x1f{pair_id}".encode()).digest(), "big")


def percentile(values: list[int], p: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1)]


def summarize(values: list[int], context: int, output_tokens: int) -> dict:
    return {
        "n": len(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "overflow_count": sum(value + output_tokens > context for value in values),
        "overflow_share": sum(value + output_tokens > context for value in values) / len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--exclude-selection", type=Path)
    parser.add_argument("--sample-input", type=Path)
    parser.add_argument("--sample-output", type=Path)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8081/tokenize")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--per-source-truncation", type=int, default=250)
    parser.add_argument("--context", type=int, default=4096)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Output exists: {args.output}")
    if bool(args.input) == bool(args.sample_input):
        raise SystemExit("Provide exactly one of --input or --sample-input")
    if args.sample_input:
        sample = [json.loads(line) for line in args.sample_input.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        if not args.exclude_selection:
            raise SystemExit("--exclude-selection is required with --input")
        excluded = {
            json.loads(line)["pair_id"]
            for line in args.exclude_selection.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        reservoirs: dict[tuple[str, bool], list[tuple[int, str, dict]]] = defaultdict(list)
        capacity = args.per_source_truncation
        import heapq
        with args.input.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record["id"] in excluded:
                    continue
                key = (str(record["media"]), bool(record.get("text_truncated")))
                score = rank(args.seed, record["id"])
                heap = reservoirs[key]
                entry = (-score, str(record["id"]), record)
                if len(heap) < capacity:
                    heapq.heappush(heap, entry)
                elif score < -heap[0][0]:
                    heapq.heapreplace(heap, entry)
        expected_keys = {(media, truncated) for media in ("interfax", "kommersant", "prime", "rbc") for truncated in (False, True)}
        if set(reservoirs) != expected_keys:
            raise ValueError(f"Token audit strata mismatch: {sorted(reservoirs)}")
        if any(len(reservoirs[key]) != capacity for key in expected_keys):
            raise ValueError("Token audit has an underfilled source×truncation stratum")
        sample = [entry[2] for key in sorted(reservoirs) for entry in reservoirs[key]]
    if args.sample_output:
        if args.sample_output.exists():
            raise SystemExit(f"Sample output exists: {args.sample_output}")
        args.sample_output.parent.mkdir(parents=True, exist_ok=True)
        partial = args.sample_output.with_suffix(args.sample_output.suffix + ".partial")
        with partial.open("w", encoding="utf-8") as destination:
            for record in sample:
                destination.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(partial, args.sample_output)
    if args.select_only:
        print(json.dumps({"selected": len(sample), "sample_output": str(args.sample_output)}, ensure_ascii=False))
        return
    base = load_base(Path(__file__).resolve().parent)
    session = requests.Session(); session.trust_env = False
    rows = []
    for index, record in enumerate(sample, start=1):
        primary = session.post(args.endpoint, json={"content": base.build_prompt(record)}, timeout=60)
        primary.raise_for_status()
        corrective = session.post(args.endpoint, json={"content": base.build_prompt(record, corrective=True)}, timeout=60)
        corrective.raise_for_status()
        rows.append(
            {
                "id": record["id"],
                "media": record["media"],
                "truncated": bool(record.get("text_truncated")),
                "primary_tokens": len(primary.json()["tokens"]),
                "corrective_tokens": len(corrective.json()["tokens"]),
            }
        )
        if index % 100 == 0:
            print(json.dumps({"tokenized": index}, ensure_ascii=False), flush=True)
    groups = {"all": rows}
    for media in sorted({row["media"] for row in rows}):
        groups[f"media={media}"] = [row for row in rows if row["media"] == media]
    groups["truncated=true"] = [row for row in rows if row["truncated"]]
    groups["truncated=false"] = [row for row in rows if not row["truncated"]]
    result = {
        "version": "actual-gguf-token-audit-v1",
        "seed": args.seed,
        "endpoint": args.endpoint,
        "sample_design": "lowest hash priority, 250 per source-by-truncation stratum; seed externally verified to have zero overlap with frozen human-validation IDs",
        "sample_n": len(rows),
        "context_size": args.context,
        "primary_max_tokens": 112,
        "corrective_max_tokens": 192,
        "primary": {name: summarize([r["primary_tokens"] for r in group], args.context, 112) for name, group in groups.items()},
        "corrective": {name: summarize([r["corrective_tokens"] for r in group], args.context, 192) for name, group in groups.items()},
        "primary_overflows": [
            row for row in rows if row["primary_tokens"] + 112 > args.context
        ],
        "corrective_overflows": [
            row for row in rows if row["corrective_tokens"] + 192 > args.context
        ],
        "sample_ids_sha256": hashlib.sha256("\n".join(sorted(r["id"] for r in rows)).encode()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
