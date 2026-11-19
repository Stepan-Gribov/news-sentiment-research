#!/usr/bin/env python3
"""Select a deterministic 40-pair technical smoke sample, excluding human validation IDs."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import importlib.util
import json
import os
from collections import defaultdict
from pathlib import Path


MEDIA = ("interfax", "kommersant", "prime", "rbc")
SEED = 20260916


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_validation_helpers(script_dir: Path):
    path = script_dir / "38_build_human_validation_sample.py"
    spec = importlib.util.spec_from_file_location("validation_helpers", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def priority(pair_id: str, bucket: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"smoke-v1\x1f{SEED}\x1f{bucket}\x1f{pair_id}".encode()).digest(), "big"
    )


def push(heap: list, record: dict, bucket: str, capacity: int = 64) -> None:
    score = priority(record["id"], bucket)
    entry = (-score, str(record["id"]), record)
    if len(heap) < capacity:
        heapq.heappush(heap, entry)
    elif score < -heap[0][0]:
        heapq.heapreplace(heap, entry)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--exclude-selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise SystemExit("Smoke output exists; refusing to overwrite")
    excluded = {
        json.loads(line)["pair_id"]
        for line in args.exclude_selection.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    helpers = load_validation_helpers(Path(__file__).resolve().parent)
    _, _, article_counts, _ = helpers.first_pass(args.input)
    pools: dict[tuple[str, str], list] = defaultdict(list)
    with args.input.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["id"] in excluded:
                continue
            media = str(record["media"]).lower()
            akey = hashlib.blake2b(helpers.article_key(record).encode(), digest_size=16).digest()
            flags = {
                "truncated": bool(record.get("text_truncated")),
                "target_absent": helpers.target_literal_absent(record),
                "multi_target": article_counts[akey] > 1,
                "rbc_object_id": helpers.rbc_object_id_date(record),
            }
            push(pools[(media, "fill")], record, f"{media}:fill")
            for flag, active in flags.items():
                if active:
                    push(pools[(media, flag)], record, f"{media}:{flag}")

    selected: list[dict] = []
    reasons: dict[str, str] = {}
    used: set[str] = set()
    for media in MEDIA:
        required = ["truncated", "target_absent", "multi_target"]
        if media == "rbc":
            required.insert(0, "rbc_object_id")
        for reason in required:
            choices = sorted((e[2] for e in pools[(media, reason)]), key=lambda r: priority(r["id"], f"{media}:{reason}"))
            choice = next((record for record in choices if record["id"] not in used), None)
            if choice is None:
                raise ValueError(f"No unique smoke candidate for {media}/{reason}")
            selected.append(choice); used.add(choice["id"]); reasons[choice["id"]] = reason
        choices = sorted((e[2] for e in pools[(media, "fill")]), key=lambda r: priority(r["id"], f"{media}:fill"))
        for record in choices:
            if sum(1 for row in selected if row["media"] == media) >= 10:
                break
            if record["id"] not in used:
                selected.append(record); used.add(record["id"]); reasons[record["id"]] = "deterministic_fill"
        if sum(1 for row in selected if row["media"] == media) != 10:
            raise ValueError(f"Could not fill 10 smoke rows for {media}")

    selected.sort(key=lambda r: priority(r["id"], "final-order"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as destination:
        for record in selected:
            destination.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(partial, args.output)
    manifest = {
        "version": "technical-smoke-40-v1",
        "seed": SEED,
        "input_sha256": sha256_file(args.input),
        "excluded_human_validation_ids": len(excluded),
        "rows": len(selected),
        "media_counts": {media: sum(r["media"] == media for r in selected) for media in MEDIA},
        "selection": [
            {
                "id": r["id"],
                "media": r["media"],
                "reason": reasons[r["id"]],
                "truncated": bool(r.get("text_truncated")),
                "target_absent": helpers.target_literal_absent(r),
                "multi_target": article_counts[hashlib.blake2b(helpers.article_key(r).encode(), digest_size=16).digest()] > 1,
                "rbc_object_id": helpers.rbc_object_id_date(r),
            }
            for r in selected
        ],
        "output_sha256": sha256_file(args.output),
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
