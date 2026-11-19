from __future__ import annotations

import importlib.util
import sys
from array import array
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "42_audit_full_text_and_tokens.py"
SPEC = importlib.util.spec_from_file_location("full_audit", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def test_exact_nearest_rank_summary() -> None:
    values = array("I", range(1, 1001))
    result = audit.summarize(values)
    assert result == {
        "n": 1000,
        "min": 1,
        "p50": 500,
        "p90": 900,
        "p95": 950,
        "p99": 990,
        "p99_9": 999,
        "max": 1000,
        "sum": 500500,
    }


def test_normalized_article_key_contract() -> None:
    assert audit.article_key(" RBC ", " HTTPS://EXAMPLE.COM/A/ ") == "rbc\x1fhttps://example.com/a"


def test_top_heap_keeps_largest_with_stable_ties() -> None:
    heap = []
    for score, identifier in [(4, "d"), (1, "a"), (3, "c"), (2, "b"), (3, "x")]:
        audit.push_top(heap, score, identifier, {"id": identifier}, 3)
    assert [(x[0], x[1]) for x in sorted(heap, reverse=True)] == [(4, "d"), (3, "x"), (3, "c")]
