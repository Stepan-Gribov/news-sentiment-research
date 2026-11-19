from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "38_build_human_validation_sample.py"
SPEC = importlib.util.spec_from_file_location("human_validation_sample", SCRIPT)
human_validation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = human_validation
SPEC.loader.exec_module(human_validation)


def make_record(index: int, media: str = "rbc") -> dict[str, object]:
    target = f"Компания{index:04d}"
    body = f"{target} сообщает об уникальном событии номер {index}. " + ("текст " * 150)
    record: dict[str, object] = {
        "id": hashlib.sha256(f"pair-{index}".encode()).hexdigest(),
        "media": media,
        "link": f"https://example.test/{index}",
        "target_company": target,
        "published_at": f"{2012 + index % 15}-01-02T00:00:00",
        "date_source": "article_metadata",
        "date_precision": "day",
        "title": f"Новость {index} о {target}",
        "text": body,
        "text_chars_original": len(body),
        "text_truncated": False,
    }
    return record


def write_fixture(root: Path, rows: int = 160) -> tuple[Path, Path]:
    input_path = root / "input.jsonl"
    raw = b"".join(
        (json.dumps(make_record(i), ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        for i in range(rows)
    )
    input_path.write_bytes(raw)
    input_hash = hashlib.sha256(raw).hexdigest()
    input_manifest = root / "input.manifest.json"
    input_manifest.write_text(
        json.dumps(
            {
                "corpus_version": "fixture",
                "outputs": {"jsonl": {"sha256": input_hash}},
            }
        ),
        encoding="utf-8",
    )
    return input_path, input_manifest


def reduced_quotas() -> dict[str, dict[str, int]]:
    return {
        "development": {name: (4 if name == "standard" else 0) for name in human_validation.RISK_ORDER},
        "heldout": {name: (6 if name == "standard" else 0) for name in human_validation.RISK_ORDER},
    }


def read_selection(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_reproducible_grouped_blinded_sample(tmp_path: Path) -> None:
    input_path, input_manifest = write_fixture(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    changed = tmp_path / "changed"
    manifest_1 = human_validation.build(
        input_path, input_manifest, first, 1234, reduced_quotas(), False, False
    )
    human_validation.build(input_path, input_manifest, second, 1234, reduced_quotas(), False, False)
    human_validation.build(input_path, input_manifest, changed, 9876, reduced_quotas(), False, False)

    selection_1 = read_selection(first / "human_validation_selection_v1.jsonl")
    selection_2 = read_selection(second / "human_validation_selection_v1.jsonl")
    selection_changed = read_selection(changed / "human_validation_selection_v1.jsonl")
    assert [row["pair_id"] for row in selection_1] == [row["pair_id"] for row in selection_2]
    assert [row["pair_id"] for row in selection_1] != [row["pair_id"] for row in selection_changed]
    assert len(selection_1) == len({row["pair_id"] for row in selection_1}) == 10
    assert manifest_1["qa"]["event_group_leakage"] == 0

    with (first / "annotation_form_annotator_a.csv").open(newline="", encoding="utf-8") as handle:
        form_a = list(csv.DictReader(handle))
    with (first / "annotation_form_annotator_b.csv").open(newline="", encoding="utf-8") as handle:
        form_b = list(csv.DictReader(handle))
    assert len(form_a) == len(form_b) == 10
    for row_a, row_b in zip(form_a, form_b):
        assert row_a["annotator_id"] == "annotator_a"
        assert row_b["annotator_id"] == "annotator_b"
        assert {k: v for k, v in row_a.items() if k != "annotator_id"} == {
            k: v for k, v in row_b.items() if k != "annotator_id"
        }
        assert all(row_a[field] == row_b[field] == "" for field in human_validation.EMPTY_ANNOTATION_FIELDS)
        assert not any(key in row_a for key in ("llm_label", "return", "holdings", "pair_id", "split"))


def test_event_proxy_groups_same_visible_reprint_and_split() -> None:
    first = make_record(1, "rbc")
    second = dict(first)
    second["id"] = hashlib.sha256(b"second-pair").hexdigest()
    second["media"] = "interfax"
    second["link"] = "https://other.test/reprint"
    assert human_validation.event_group_id(first) == human_validation.event_group_id(second)
    assert human_validation.assigned_split(123, human_validation.event_group_id(first)) == human_validation.assigned_split(
        123, human_validation.event_group_id(second)
    )
