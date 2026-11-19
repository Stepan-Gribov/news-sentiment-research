from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "37_build_target_llm_c04a_corpus.py"
SPEC = importlib.util.spec_from_file_location("target_llm_c04a", SCRIPT)
target_llm_c04a = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = target_llm_c04a
SPEC.loader.exec_module(target_llm_c04a)


def make_record(media: str, link: str, target: str, **updates: object) -> dict[str, object]:
    record: dict[str, object] = {
        "id": "",
        "media": media,
        "link": link,
        "target_company": target,
        "published_at": "2024-01-02T00:00:00",
        "date_source": "article_metadata",
        "date_precision": "day",
        "title": f"Новость о {target}",
        "text": f"Событие затрагивает {target}",
        "text_chars_original": 30,
        "text_truncated": False,
    }
    record.update(updates)
    record["id"] = target_llm_c04a.expected_pair_id(record)
    return record


def write_jsonl(path: Path, records: list[dict[str, object]]) -> bytes:
    raw = b"".join(
        (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        for record in records
    )
    path.write_bytes(raw)
    return raw


def test_filter_preserves_bytes_order_ids_and_keeps_audited_edge_cases(tmp_path: Path) -> None:
    frank = make_record("frank", "https://example.test/frank", "А")
    multi_a = make_record("rbc", "https://example.test/joint/", "А")
    multi_b = make_record("rbc", "https://example.test/joint", "Б")
    inferred = make_record(
        "rbc",
        "https://example.test/inferred",
        "В",
        date_source="rbc_object_id_timestamp",
        date_precision="inferred_timestamp",
    )
    inferred_freenews = make_record(
        "rbc",
        "https://example.test/freenews",
        "Г",
        date_source="rbc_freenews_object_id",
        date_precision="inferred_timestamp",
    )
    truncated_absent = make_record(
        "prime",
        "https://example.test/truncated",
        "Целевая компания",
        title="Общий заголовок",
        text="Фрагмент заканчивается до упоминания цели",
        text_chars_original=9000,
        text_truncated=True,
    )
    records = [frank, multi_a, multi_b, inferred, inferred_freenews, truncated_absent]
    parent = tmp_path / "parent.jsonl"
    write_jsonl(parent, records)
    parent_manifest = tmp_path / "parent.manifest.json"
    parent_manifest.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "output.jsonl"
    manifest_path = tmp_path / "manifest.json"
    mask = tmp_path / "mask.jsonl"

    manifest = target_llm_c04a.process(
        parent, parent_manifest, output, manifest_path, mask, None, False
    )

    expected_raw = b"".join(parent.read_bytes().splitlines(keepends=True)[1:])
    assert output.read_bytes() == expected_raw
    assert [json.loads(line)["id"] for line in output.read_text().splitlines()] == [
        record["id"] for record in records[1:]
    ]
    assert manifest["counts"]["output_rows"] == 5
    assert manifest["counts"]["excluded_rows"] == 1
    assert manifest["counts"]["parent_multi_target_articles"] == 1
    assert manifest["counts"]["output_multi_target_articles"] == 1
    assert manifest["counts"]["sensitivity_object_id_rows"] == 2
    assert manifest["counts"]["target_literal_absent"] == 1
    mask_rows = [json.loads(line) for line in mask.read_text().splitlines()]
    assert [row["id"] for row in mask_rows] == [inferred["id"], inferred_freenews["id"]]


def test_pair_id_and_natural_key_contract_detects_changed_target() -> None:
    record = make_record("kommersant", "HTTPS://EXAMPLE.TEST/A/", "Компания")
    original_id = record["id"]
    record["target_company"] = "Другая компания"
    assert target_llm_c04a.expected_pair_id(record) != original_id
    assert target_llm_c04a.natural_key(record) == (
        "kommersant",
        "https://example.test/a",
        "Другая компания",
    )


def test_reuse_map_separates_allowed_success_frank_and_errors(tmp_path: Path) -> None:
    allowed = make_record("interfax", "https://example.test/ok", "А")
    frank = make_record("frank", "https://example.test/frank", "Б")
    parent = tmp_path / "parent.jsonl"
    write_jsonl(parent, [frank, allowed])
    parent_manifest = tmp_path / "parent.manifest.json"
    parent_manifest.write_text("{}\n", encoding="utf-8")
    parts = tmp_path / "parts"
    parts.mkdir()
    run_config = {
        "runner_version": "target-llm-runner-v1",
        "prompt_version": "target-financial-sentiment-ru-v1",
    }
    (parts / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")
    output_records = []
    for record, status in ((frank, "ok"), (allowed, "error")):
        result = {key: record.get(key) for key in target_llm_c04a.OUTPUT_METADATA_FIELDS}
        result.update(
            {
                "id": record["id"],
                "status": status,
                "runner_version": run_config["runner_version"],
                "prompt_version": run_config["prompt_version"],
            }
        )
        output_records.append(result)
    write_jsonl(parts / "part_00001_00000000_00000001.jsonl", output_records)

    manifest = target_llm_c04a.process(
        parent,
        parent_manifest,
        tmp_path / "output.jsonl",
        tmp_path / "manifest.json",
        tmp_path / "mask.jsonl",
        parts,
        False,
    )
    reuse = manifest["llm_output_reuse"]["mapping"]
    assert reuse["excluded_frank_success"] == 1
    assert reuse["allowed_error"] == 1
    assert reuse["reusable_success"] == 0
    assert reuse["unprocessed_output_rows"] == 1


def test_duplicate_pair_and_natural_key_fail_the_contract(tmp_path: Path) -> None:
    record = make_record("rbc", "https://example.test/duplicate", "Компания")
    parent = tmp_path / "parent.jsonl"
    write_jsonl(parent, [record, dict(record)])
    parent_manifest = tmp_path / "parent.manifest.json"
    parent_manifest.write_text("{}\n", encoding="utf-8")
    try:
        target_llm_c04a.process(
            parent,
            parent_manifest,
            tmp_path / "output.jsonl",
            tmp_path / "manifest.json",
            tmp_path / "mask.jsonl",
            None,
            False,
        )
    except ValueError as exc:
        message = str(exc)
        assert "duplicate_pair_id" in message
        assert "duplicate_natural_key" in message
    else:
        raise AssertionError("Duplicate keys must fail the corpus contract")
