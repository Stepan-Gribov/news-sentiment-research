#!/usr/bin/env python3
"""Run resumable target-conditioned LLM sentiment classification from JSONL.

Input is produced by ``30_build_target_llm_input.py``.  Each output line has a
target company, traceable source metadata and a target-level sentiment result;
the potentially large article text is deliberately not duplicated in output.
Completed parts are immutable.  A crash leaves only the current ``.partial``
part, which is resumed from its last valid JSONL record on the next invocation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


RUNNER_VERSION = "target-llm-runner-v1"
PROMPT_VERSION = "target-financial-sentiment-ru-v1"
DEFAULT_SERVER_URL = "http://127.0.0.1:8080/v1/completions"
DEFAULT_MODEL = "Qwen3-4B-Q4_K_M.gguf"
VALID_RELEVANCE = {"relevant", "irrelevant"}
VALID_SENTIMENT = {"positive", "neutral", "negative"}
WHITESPACE_RE = re.compile(r"\s+")


def build_prompt(record: dict[str, Any], corrective: bool = False) -> str:
    correction = ""
    if corrective:
        correction = (
            "\nПредыдущий ответ был невалиден. Верни строго один корректный JSON "
            "по указанной схеме, без markdown и без дополнительного текста."
        )
    return f"""<|im_start|>system
Ты выполняешь target-level классификацию финансовых новостей относительно указанной компании.
Используй только информацию из переданного текста. Не используй внешние знания,
последующие события или фактическое движение ценных бумаг после публикации.

Сначала определи, относится ли содержание новости к TARGET по существу. Если компания
только упомянута или связь с ней недостаточна, верни relevance = \"irrelevant\".

Для релевантной новости оцени с позиции инвестора ожидаемое влияние на TARGET в
ближайший месяц. positive означает улучшение прибыли, денежных потоков, спроса,
перспектив роста, ликвидности, кредитного качества, стоимости компании или снижение
рисков. negative означает ухудшение этих факторов или рост рисков. neutral означает,
что новость относится к TARGET, но не имеет ясного направления, сбалансирована или
содержит недостаточно информации. Оцени последствия именно для TARGET, а не общий тон.

Верни только JSON строго этой схемы:
{{"relevance":"relevant|irrelevant","sentiment_label":"positive|neutral|negative|null","short_rationale":"не более 12 слов"}}
Если relevance = \"irrelevant\", sentiment_label обязан быть null.<|im_end|>
<|im_start|>user
TARGET: {record["target_company"]}

Заголовок: {record.get("title") or "(заголовок отсутствует)"}

Текст новости:
{record["text"]}{correction}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""


def build_session() -> requests.Session:
    session = requests.Session()
    # Do not accidentally send local llama.cpp requests through a proxy.
    session.trust_env = False
    return session


def request_completion(
    session: requests.Session,
    server_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_seconds: int,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "repeat_penalty": 1.0,
        "seed": 42,
        "stop": ["<|im_end|>"],
    }
    response = session.post(server_url, json=payload, timeout=timeout_seconds)
    response.raise_for_status()
    return str(response.json()["choices"][0]["text"]).strip()


def json_candidates(raw_text: str) -> list[dict[str, Any]]:
    """Extract JSON object candidates if the model adds stray surrounding text."""

    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    stripped = raw_text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").removeprefix("json").strip()
    for offset, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(stripped[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            candidates.append(candidate)
    return candidates


def normalize_result(raw_text: str) -> dict[str, Any]:
    errors: list[str] = []
    for candidate in json_candidates(raw_text):
        relevance = str(candidate.get("relevance", "")).strip().lower()
        label_value = candidate.get("sentiment_label")
        label = None if label_value is None else str(label_value).strip().lower()
        rationale = WHITESPACE_RE.sub(" ", str(candidate.get("short_rationale", "")).strip())
        if relevance not in VALID_RELEVANCE:
            errors.append(f"invalid relevance={relevance!r}")
            continue
        if relevance == "irrelevant":
            if label not in (None, "null"):
                errors.append(f"irrelevant response has label={label!r}")
                continue
            return {
                "relevance": "irrelevant",
                "sentiment_label": None,
                "short_rationale": rationale[:500],
            }
        if label not in VALID_SENTIMENT:
            errors.append(f"relevant response has label={label!r}")
            continue
        return {
            "relevance": "relevant",
            "sentiment_label": label,
            "short_rationale": rationale[:500],
        }
    detail = "; ".join(errors) if errors else "no JSON object with the required schema"
    raise ValueError(detail)


def classify_record(
    record: dict[str, Any],
    session: requests.Session,
    server_url: str,
    model: str,
    max_tokens: int,
    retry_max_tokens: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    raw_text = request_completion(
        session,
        server_url,
        model,
        build_prompt(record),
        max_tokens,
        timeout_seconds,
    )
    try:
        result = normalize_result(raw_text)
        result.update(
            {
                "raw_text": raw_text,
                "used_retry": False,
                "max_tokens_used": max_tokens,
            }
        )
        return result
    except Exception as first_error:
        raw_text_retry = request_completion(
            session,
            server_url,
            model,
            build_prompt(record, corrective=True),
            retry_max_tokens,
            timeout_seconds,
        )
        try:
            result = normalize_result(raw_text_retry)
        except Exception as retry_error:
            raise ValueError(
                f"first parse: {first_error}; retry parse: {retry_error}; "
                f"retry raw: {raw_text_retry[:800]!r}"
            ) from retry_error
        result.update(
            {
                "raw_text": raw_text_retry,
                "used_retry": True,
                "max_tokens_used": retry_max_tokens,
                "first_raw_text": raw_text,
            }
        )
        return result


def count_input_records(path: Path) -> int:
    with path.open("r", encoding="utf-8") as source:
        return sum(1 for line in source if line.strip())


def count_valid_jsonl_records(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL in {path} at line {line_number}: {exc}") from exc
            count += 1
    return count


def part_name(part_index: int, start: int, end: int) -> str:
    return f"part_{part_index:05d}_{start:08d}_{end:08d}.jsonl"


def part_layout(total_records: int, part_size: int) -> list[tuple[int, int, int, int]]:
    layout: list[tuple[int, int, int, int]] = []
    for start in range(0, total_records, part_size):
        end = min(start + part_size, total_records) - 1
        part_index = start // part_size + 1
        layout.append((part_index, start, end, end - start + 1))
    return layout


def existing_part_state(
    output_dir: Path, total_records: int, part_size: int
) -> dict[int, dict[str, Any]]:
    """Validate final parts and get safe resume counts for partial parts."""

    state: dict[int, dict[str, Any]] = {}
    for part_index, start, end, expected in part_layout(total_records, part_size):
        final_path = output_dir / part_name(part_index, start, end)
        partial_path = final_path.with_suffix(final_path.suffix + ".partial")
        if final_path.exists() and partial_path.exists():
            raise ValueError(f"Both final and partial versions exist for part {part_index}")
        if final_path.exists():
            actual = count_valid_jsonl_records(final_path)
            if actual != expected:
                raise ValueError(
                    f"Final part {final_path} has {actual} records; expected {expected}. "
                    "Refusing to overwrite a completed artifact."
                )
            state[part_index] = {"complete": True, "resume_count": actual, "path": final_path}
            continue
        if partial_path.exists():
            actual = count_valid_jsonl_records(partial_path)
            if actual > expected:
                raise ValueError(
                    f"Partial part {partial_path} has {actual} records; expected at most {expected}."
                )
            if actual == expected:
                os.replace(partial_path, final_path)
                state[part_index] = {
                    "complete": True,
                    "resume_count": actual,
                    "path": final_path,
                }
            else:
                state[part_index] = {
                    "complete": False,
                    "resume_count": actual,
                    "path": partial_path,
                }
            continue
        state[part_index] = {"complete": False, "resume_count": 0, "path": partial_path}
    return state


def output_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "media": record.get("media"),
        "link": record.get("link"),
        "target_company": record.get("target_company"),
        "published_at": record.get("published_at"),
        "date_source": record.get("date_source"),
        "date_precision": record.get("date_precision"),
        "title": record.get("title"),
        "text_chars_original": record.get("text_chars_original"),
        "text_truncated": record.get("text_truncated"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--part-size", type=int, default=1_000)
    parser.add_argument("--start-part", type=int, default=1)
    parser.add_argument(
        "--max-parts",
        type=int,
        default=0,
        help="Complete at most this many new parts (0 means all available parts).",
    )
    parser.add_argument("--max-tokens", type=int, default=112)
    parser.add_argument("--retry-max-tokens", type=int, default=192)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()

    if min(
        args.part_size,
        args.start_part,
        args.max_tokens,
        args.retry_max_tokens,
        args.timeout_seconds,
    ) <= 0 or args.max_parts < 0:
        raise SystemExit("Part sizes, start part, token limits and timeout must be positive.")

    input_path = args.input.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    total_records = count_input_records(input_path)
    if total_records == 0:
        raise SystemExit("Input JSONL has no non-empty records")
    layout = part_layout(total_records, args.part_size)
    if args.start_part > len(layout):
        raise SystemExit(f"--start-part is outside the available range 1..{len(layout)}")
    state = existing_part_state(output_dir, total_records, args.part_size)

    run_config = {
        "runner_version": RUNNER_VERSION,
        "prompt_version": PROMPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "input_size_bytes": input_path.stat().st_size,
        "total_records": total_records,
        "part_size": args.part_size,
        "server_url": args.server_url,
        "model": args.model,
        "temperature": 0.0,
        "seed": 42,
        "max_tokens": args.max_tokens,
        "retry_max_tokens": args.retry_max_tokens,
        "text_not_duplicated_in_output": True,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_config, ensure_ascii=False), flush=True)

    session = build_session()
    current_part: int | None = None
    current_handle = None
    run_stats: Counter[str] = Counter()
    newly_completed = 0
    started = time.monotonic()

    try:
        with input_path.open("r", encoding="utf-8") as source:
            input_index = 0
            for line in source:
                if not line.strip():
                    continue
                part_index = input_index // args.part_size + 1
                part_start = (part_index - 1) * args.part_size
                part_end = min(part_start + args.part_size, total_records) - 1
                expected = part_end - part_start + 1
                local_index = input_index - part_start
                input_index += 1

                if part_index < args.start_part:
                    continue
                part_state = state[part_index]
                if part_state["complete"]:
                    run_stats["records_skipped_completed_parts"] += 1
                    continue
                if local_index < part_state["resume_count"]:
                    run_stats["records_skipped_partial"] += 1
                    continue

                if current_part != part_index:
                    if current_handle is not None:
                        current_handle.close()
                        raise RuntimeError("Part boundary reached before the prior part was complete")
                    current_part = part_index
                    current_handle = Path(part_state["path"]).open("a", encoding="utf-8")
                    print(
                        f"Starting part {part_index}/{len(layout)} at record {part_start}; "
                        f"resuming after {part_state['resume_count']} records",
                        flush=True,
                    )

                try:
                    record = json.loads(line)
                    result = classify_record(
                        record,
                        session,
                        args.server_url,
                        args.model,
                        args.max_tokens,
                        args.retry_max_tokens,
                        args.timeout_seconds,
                    )
                    output = output_metadata(record)
                    output.update(
                        {
                            "status": "ok",
                            "runner_version": RUNNER_VERSION,
                            "prompt_version": PROMPT_VERSION,
                            **result,
                        }
                    )
                    run_stats["ok"] += 1
                    if result["used_retry"]:
                        run_stats["used_retry"] += 1
                except Exception as exc:
                    output = output_metadata(json.loads(line))
                    output.update(
                        {
                            "status": "error",
                            "runner_version": RUNNER_VERSION,
                            "prompt_version": PROMPT_VERSION,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    run_stats["error"] += 1
                current_handle.write(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")
                current_handle.flush()
                run_stats["records_written"] += 1

                if run_stats["records_written"] % 10 == 0:
                    elapsed = max(time.monotonic() - started, 0.001)
                    print(
                        json.dumps(
                            {
                                "part": part_index,
                                "written_this_invocation": run_stats["records_written"],
                                "ok": run_stats["ok"],
                                "error": run_stats["error"],
                                "records_per_second": round(run_stats["records_written"] / elapsed, 3),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

                if local_index + 1 == expected:
                    current_handle.close()
                    current_handle = None
                    final_path = output_dir / part_name(part_index, part_start, part_end)
                    os.replace(Path(part_state["path"]), final_path)
                    state[part_index] = {
                        "complete": True,
                        "resume_count": expected,
                        "path": final_path,
                    }
                    newly_completed += 1
                    print(f"Completed part {part_index}: {final_path.name}", flush=True)
                    current_part = None
                    if args.max_parts and newly_completed >= args.max_parts:
                        break
    finally:
        if current_handle is not None:
            current_handle.close()

    elapsed_seconds = round(time.monotonic() - started, 3)
    summary = {
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "newly_completed_parts": newly_completed,
        "completed_parts_total": sum(1 for value in state.values() if value["complete"]),
        "total_parts": len(layout),
        "statistics_this_invocation": dict(sorted(run_stats.items())),
    }
    (output_dir / "last_invocation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
