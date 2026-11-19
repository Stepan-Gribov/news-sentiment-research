#!/usr/bin/env python3
"""Safety wrapper for the frozen four-source target-level LLM run.

The classifier prompt and parser are imported unchanged from
``31_run_target_llm_sentiment.py``.  This wrapper verifies frozen hashes,
resumes immutable parts, repairs only a torn final partial line, and writes a
separate per-part retry queue for technical errors.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


RUNNER_VERSION = "target-four-sources-safe-runner-v1"
ALLOWED_MEDIA = {"interfax", "kommersant", "prime", "rbc"}
SENTINEL = {
    "target_company": "TARGET",
    "title": "TITLE",
    "text": "TEXT",
}


def load_base_runner(script_dir: Path):
    path = script_dir / "31_run_target_llm_sentiment.py"
    spec = importlib.util.spec_from_file_location("frozen_target_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def part_name(index: int, start: int, end: int) -> str:
    return f"part_{index:05d}_{start:08d}_{end:08d}.jsonl"


def validate_config(config: dict[str, Any], config_path: Path, base: Any) -> dict[str, Path]:
    required = {
        "input_path",
        "input_sha256",
        "input_size_bytes",
        "total_records",
        "media_counts",
        "model_path",
        "model_sha256",
        "base_runner_sha256",
        "safe_runner_sha256",
        "prompt_contract_sha256",
        "server_url",
        "model",
        "part_size",
        "seed",
        "temperature",
        "max_tokens",
        "retry_max_tokens",
        "timeout_seconds",
        "output_dir",
        "retry_dir",
        "scripts_dir",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Run config lacks fields: {missing}")
    endpoint = urlparse(str(config["server_url"]))
    if endpoint.hostname != "127.0.0.1":
        raise ValueError(f"Server must bind through explicit localhost endpoint: {endpoint.hostname}")
    if int(config["part_size"]) <= 0:
        raise ValueError("part_size must be positive")
    if float(config["temperature"]) != 0.0 or int(config["seed"]) != 42:
        raise ValueError("Frozen decoding requires temperature=0 and seed=42")

    script_dir = Path(config["scripts_dir"])
    paths = {
        "input": Path(config["input_path"]),
        "model": Path(config["model_path"]),
        "base_runner": script_dir / "31_run_target_llm_sentiment.py",
        "safe_runner": script_dir / "39_run_target_llm_four_sources.py",
        "output_dir": Path(config["output_dir"]),
        "retry_dir": Path(config["retry_dir"]),
    }
    for name in ("input", "model", "base_runner", "safe_runner"):
        if not paths[name].is_file():
            raise FileNotFoundError(f"Missing {name}: {paths[name]}")
    checks = {
        "model": (paths["model"], config["model_sha256"]),
        "base_runner": (paths["base_runner"], config["base_runner_sha256"]),
        "safe_runner": (paths["safe_runner"], config["safe_runner_sha256"]),
    }
    for name, (path, expected) in checks.items():
        actual = sha256_file(path)
        if actual.lower() != str(expected).lower():
            raise ValueError(f"{name} hash mismatch: {actual} != {expected}")
    prompt_hash = hashlib.sha256(base.build_prompt(SENTINEL).encode("utf-8")).hexdigest()
    if prompt_hash.lower() != str(config["prompt_contract_sha256"]).lower():
        raise ValueError(f"Prompt contract mismatch: {prompt_hash}")
    return paths


def verify_input(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    media: Counter[str] = Counter()
    seen: set[str] = set()
    with path.open("rb") as source:
        for line_number, raw in enumerate(source, start=1):
            digest.update(raw)
            if not raw.strip():
                continue
            record = json.loads(raw)
            pair_id = str(record.get("id") or "")
            if not pair_id or pair_id in seen:
                raise ValueError(f"Missing/duplicate pair_id at input line {line_number}: {pair_id}")
            seen.add(pair_id)
            source_name = str(record.get("media") or "").strip().lower()
            if source_name not in ALLOWED_MEDIA:
                raise ValueError(f"Disallowed media at input line {line_number}: {source_name}")
            media[source_name] += 1
            count += 1
    actual_hash = digest.hexdigest()
    expected_media = {str(k): int(v) for k, v in config["media_counts"].items()}
    if actual_hash.lower() != str(config["input_sha256"]).lower():
        raise ValueError(f"Input hash mismatch: {actual_hash} != {config['input_sha256']}")
    if path.stat().st_size != int(config["input_size_bytes"]):
        raise ValueError("Input size mismatch")
    if count != int(config["total_records"]):
        raise ValueError(f"Input count mismatch: {count} != {config['total_records']}")
    if dict(sorted(media.items())) != dict(sorted(expected_media.items())):
        raise ValueError(f"Input media counts mismatch: {dict(media)} != {expected_media}")
    return {"sha256": actual_hash, "records": count, "media_counts": dict(sorted(media.items()))}


def ensure_run_state(output_dir: Path, config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_hash = sha256_file(config_path)
    state_path = output_dir / "run_state.json"
    frozen = {
        "safe_runner_version": RUNNER_VERSION,
        "config_sha256": config_hash,
        "input_sha256": config["input_sha256"],
        "model_sha256": config["model_sha256"],
        "base_runner_sha256": config["base_runner_sha256"],
        "safe_runner_sha256": config["safe_runner_sha256"],
        "prompt_contract_sha256": config["prompt_contract_sha256"],
        "part_size": config["part_size"],
    }
    if state_path.exists():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        for key, value in frozen.items():
            if previous.get(key) != value:
                raise ValueError(f"Resume state mismatch for {key}: {previous.get(key)!r} != {value!r}")
        return previous
    frozen["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(state_path, frozen)
    return frozen


def recover_jsonl_prefix(path: Path) -> list[dict[str, Any]]:
    """Return valid records and repair only a torn final line."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    records: list[dict[str, Any]] = []
    valid_end = 0
    for index, line in enumerate(lines):
        if not line.strip():
            valid_end += len(line)
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            is_last = index == len(lines) - 1
            if not is_last:
                raise ValueError(f"Malformed non-final JSONL line in {path}: {exc}") from exc
            path.write_bytes(raw[:valid_end])
            return records
        records.append(record)
        valid_end += len(line)
    if raw and not raw.endswith(b"\n"):
        with path.open("ab") as destination:
            destination.write(b"\n")
    return records


def validate_output_records(
    records: list[dict[str, Any]], expected_ids: list[str], complete: bool
) -> None:
    if complete and len(records) != len(expected_ids):
        raise ValueError(f"Completed part has {len(records)} records; expected {len(expected_ids)}")
    if len(records) > len(expected_ids):
        raise ValueError("Output part has more records than its input range")
    ids = [str(record.get("id") or "") for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate output ID in part")
    for index, pair_id in enumerate(ids):
        if pair_id != expected_ids[index]:
            raise ValueError(
                f"Unknown/out-of-order output ID at offset {index}: {pair_id} != {expected_ids[index]}"
            )


def write_retry_queue(path: Path, records: list[dict[str, Any]]) -> None:
    errors = [
        {"id": record["id"], "status": "error", "error": record.get("error"), "source_part": path.stem}
        for record in records
        if record.get("status") == "error"
    ]
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as destination:
        for row in errors:
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(partial, path)


def input_parts(path: Path, part_size: int) -> Iterable[tuple[int, list[dict[str, Any]]]]:
    part: list[dict[str, Any]] = []
    index = 1
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            part.append(json.loads(line))
            if len(part) == part_size:
                yield index, part
                index += 1
                part = []
    if part:
        yield index, part


def run(config_path: Path, max_parts: int = 0) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    script_dir = Path(config["scripts_dir"])
    base = load_base_runner(script_dir)
    paths = validate_config(config, config_path, base)
    input_audit = verify_input(paths["input"], config)
    output_dir = paths["output_dir"]
    retry_dir = paths["retry_dir"]
    retry_dir.mkdir(parents=True, exist_ok=True)
    ensure_run_state(output_dir, config_path, config)
    part_size = int(config["part_size"])
    total = int(config["total_records"])
    total_parts = (total + part_size - 1) // part_size
    stats: Counter[str] = Counter()
    started = time.monotonic()
    newly_completed = 0

    session = base.build_session()
    for part_index, input_records in input_parts(paths["input"], part_size):
        start = (part_index - 1) * part_size
        end = start + len(input_records) - 1
        name = part_name(part_index, start, end)
        final_path = output_dir / name
        partial_path = final_path.with_suffix(final_path.suffix + ".partial")
        retry_path = retry_dir / name.replace("part_", "retry_part_", 1)
        expected_ids = [str(record["id"]) for record in input_records]
        if final_path.exists() and partial_path.exists():
            raise ValueError(f"Both final and partial files exist for part {part_index}")
        if final_path.exists():
            records = recover_jsonl_prefix(final_path)
            validate_output_records(records, expected_ids, complete=True)
            if not retry_path.exists():
                write_retry_queue(retry_path, records)
            stats["completed_parts_skipped"] += 1
            stats["records_skipped"] += len(records)
            continue
        existing = recover_jsonl_prefix(partial_path)
        validate_output_records(existing, expected_ids, complete=False)
        stats["partial_records_resumed"] += len(existing)
        with partial_path.open("a", encoding="utf-8") as destination:
            for record in input_records[len(existing) :]:
                try:
                    result = base.classify_record(
                        record,
                        session,
                        config["server_url"],
                        config["model"],
                        int(config["max_tokens"]),
                        int(config["retry_max_tokens"]),
                        int(config["timeout_seconds"]),
                    )
                    output = base.output_metadata(record)
                    output.update(
                        {
                            "status": "ok",
                            "runner_version": base.RUNNER_VERSION,
                            "safe_runner_version": RUNNER_VERSION,
                            "prompt_version": base.PROMPT_VERSION,
                            **result,
                        }
                    )
                    stats["ok"] += 1
                    stats["used_retry"] += bool(result.get("used_retry"))
                except Exception as exc:
                    output = base.output_metadata(record)
                    output.update(
                        {
                            "status": "error",
                            "runner_version": base.RUNNER_VERSION,
                            "safe_runner_version": RUNNER_VERSION,
                            "prompt_version": base.PROMPT_VERSION,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    stats["error"] += 1
                destination.write(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")
                destination.flush()
        records = recover_jsonl_prefix(partial_path)
        validate_output_records(records, expected_ids, complete=True)
        os.replace(partial_path, final_path)
        write_retry_queue(retry_path, records)
        newly_completed += 1
        stats["parts_completed"] += 1
        if max_parts and newly_completed >= max_parts:
            break

    result = {
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "input_audit": input_audit,
        "total_parts": total_parts,
        "newly_completed_parts": newly_completed,
        "stats": dict(stats),
    }
    atomic_json(output_dir / "last_invocation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True, type=Path)
    parser.add_argument("--max-parts", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(args.run_config, args.max_parts), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
