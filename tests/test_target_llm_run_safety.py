from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "39_run_target_llm_four_sources.py"
SPEC = importlib.util.spec_from_file_location("safe_runner", SCRIPT)
safe_runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = safe_runner
SPEC.loader.exec_module(safe_runner)


def record(pair_id: str, status: str = "ok") -> dict:
    value = {"id": pair_id, "status": status}
    if status == "error":
        value["error"] = "technical failure"
    return value


def test_partial_resume_repairs_torn_tail_without_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "part.jsonl.partial"
    path.write_bytes(
        (json.dumps(record("a")) + "\n" + json.dumps(record("b")) + "\n" + '{"id":"c"').encode()
    )
    recovered = safe_runner.recover_jsonl_prefix(path)
    safe_runner.validate_output_records(recovered, ["a", "b", "c"], complete=False)
    assert [row["id"] for row in recovered] == ["a", "b"]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record("c")) + "\n")
    final = safe_runner.recover_jsonl_prefix(path)
    safe_runner.validate_output_records(final, ["a", "b", "c"], complete=True)
    assert [row["id"] for row in final] == ["a", "b", "c"]


def test_completed_part_skip_contract_and_unknown_id_failure(tmp_path: Path) -> None:
    complete = [record("a"), record("b")]
    safe_runner.validate_output_records(complete, ["a", "b"], complete=True)
    try:
        safe_runner.validate_output_records([record("unknown")], ["a", "b"], complete=False)
    except ValueError as exc:
        assert "Unknown/out-of-order" in str(exc)
    else:
        raise AssertionError("Unknown ID must stop resume")


def test_error_record_is_written_to_separate_retry_queue(tmp_path: Path) -> None:
    retry = tmp_path / "retry.jsonl"
    safe_runner.write_retry_queue(retry, [record("a"), record("b", "error")])
    rows = [json.loads(line) for line in retry.read_text().splitlines()]
    assert rows == [{"id": "b", "status": "error", "error": "technical failure", "source_part": "retry"}]


def test_config_mismatch_stops_resume(tmp_path: Path) -> None:
    config_path = tmp_path / "run_config.json"
    config_path.write_text('{"version":1}\n')
    output = tmp_path / "output"; output.mkdir()
    config = {
        "input_sha256": "a",
        "model_sha256": "b",
        "base_runner_sha256": "c",
        "safe_runner_sha256": "d",
        "prompt_contract_sha256": "e",
        "part_size": 1000,
    }
    safe_runner.ensure_run_state(output, config_path, config)
    config_path.write_text('{"version":2}\n')
    try:
        safe_runner.ensure_run_state(output, config_path, config)
    except ValueError as exc:
        assert "config_sha256" in str(exc)
    else:
        raise AssertionError("Changed config must stop resume")


def test_input_audit_rejects_frank(tmp_path: Path) -> None:
    path = tmp_path / "input.jsonl"
    raw = (json.dumps({"id": "x", "media": "frank"}) + "\n").encode()
    path.write_bytes(raw)
    config = {
        "input_sha256": __import__("hashlib").sha256(raw).hexdigest(),
        "input_size_bytes": len(raw),
        "total_records": 1,
        "media_counts": {"frank": 1},
    }
    try:
        safe_runner.verify_input(path, config)
    except ValueError as exc:
        assert "Disallowed media" in str(exc)
    else:
        raise AssertionError("Frank must be rejected")
