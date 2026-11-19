#!/usr/bin/env python3
"""Compare two local llama.cpp models on the frozen 300-row validation sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests


PROMPT_VERSION = "target-financial-sentiment-ru-v1-chat"
SYSTEM_PROMPT = """Ты выполняешь target-level классификацию финансовых новостей относительно указанной компании.
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
{\"relevance\":\"relevant|irrelevant\",\"sentiment_label\":\"positive|neutral|negative|null\",\"short_rationale\":\"не более 12 слов\"}
Если relevance = \"irrelevant\", sentiment_label обязан быть null."""

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "target_sentiment",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "relevance": {"type": "string", "enum": ["relevant", "irrelevant"]},
                "sentiment_label": {
                    "type": ["string", "null"],
                    "enum": ["positive", "neutral", "negative", None],
                },
                "short_rationale": {"type": "string"},
            },
            "required": ["relevance", "sentiment_label", "short_rationale"],
            "additionalProperties": False,
        },
    },
}

CORRECTION = (
    "\n\nПредыдущий ответ был невалиден. Верни строго один корректный JSON "
    "по указанной схеме, без markdown и без дополнительного текста."
)
VALID_RELEVANCE = {"relevant", "irrelevant"}
VALID_SENTIMENT = {"positive", "neutral", "negative"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_user_prompt(row: dict[str, Any], corrective: bool = False) -> str:
    prompt = (
        f"TARGET: {row['target_company']}\n\n"
        f"Заголовок: {row.get('title') or '(заголовок отсутствует)'}\n\n"
        f"Текст новости:\n{row['text']}"
    )
    return prompt + (CORRECTION if corrective else "")


def json_candidates(raw_text: str) -> list[dict[str, Any]]:
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
        value = candidate.get("sentiment_label")
        sentiment = None if value is None else str(value).strip().lower()
        rationale = " ".join(str(candidate.get("short_rationale", "")).split())
        if relevance not in VALID_RELEVANCE:
            errors.append(f"invalid relevance={relevance!r}")
            continue
        if relevance == "irrelevant":
            if sentiment not in (None, "null"):
                errors.append(f"irrelevant response has label={sentiment!r}")
                continue
            return {
                "relevance": "irrelevant",
                "sentiment_label": None,
                "short_rationale": rationale[:500],
            }
        if sentiment not in VALID_SENTIMENT:
            errors.append(f"relevant response has label={sentiment!r}")
            continue
        return {
            "relevance": "relevant",
            "sentiment_label": sentiment,
            "short_rationale": rationale[:500],
        }
    detail = "; ".join(errors) if errors else "no valid JSON object"
    raise ValueError(detail)


def request_completion(
    session: requests.Session,
    url: str,
    model_name: str,
    row: dict[str, Any],
    corrective: bool,
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(row, corrective)},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "seed": 42,
        "response_format": RESPONSE_FORMAT,
    }
    response = session.post(url, json=payload, timeout=240)
    response.raise_for_status()
    body = response.json()
    content = body["choices"][0]["message"].get("content") or ""
    return str(content).strip(), body.get("usage") or {}


def classify(
    session: requests.Session, url: str, model_name: str, row: dict[str, Any]
) -> dict[str, Any]:
    started = time.monotonic()
    first_raw = ""
    first_usage: dict[str, Any] = {}
    try:
        first_raw, first_usage = request_completion(
            session, url, model_name, row, False, 112
        )
        result = normalize_result(first_raw)
        result.update(
            {
                "valid": True,
                "used_retry": False,
                "raw_text": first_raw,
                "usage": first_usage,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        return result
    except Exception as first_error:
        retry_raw = ""
        try:
            retry_raw, retry_usage = request_completion(
                session, url, model_name, row, True, 192
            )
            result = normalize_result(retry_raw)
            result.update(
                {
                    "valid": True,
                    "used_retry": True,
                    "raw_text": retry_raw,
                    "first_raw_text": first_raw,
                    "first_error": str(first_error),
                    "usage": retry_usage,
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
            return result
        except Exception as retry_error:
            return {
                "valid": False,
                "used_retry": True,
                "relevance": None,
                "sentiment_label": None,
                "short_rationale": "",
                "raw_text": first_raw,
                "retry_raw_text": retry_raw,
                "first_error": str(first_error),
                "retry_error": str(retry_error),
                "elapsed_seconds": time.monotonic() - started,
            }


def load_rows(sample_csv: Path, selection_jsonl: Path) -> list[dict[str, Any]]:
    selection: dict[str, dict[str, Any]] = {}
    with selection_jsonl.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                record = json.loads(line)
                selection[record["row_id"]] = record
    with sample_csv.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 300 or len({row["row_id"] for row in rows}) != 300:
        raise ValueError("Expected exactly 300 unique frozen sample rows")
    for row in rows:
        meta = selection[row["row_id"]]
        row["pair_id"] = meta["pair_id"]
        row["split"] = meta["split"]
        row["design_weight"] = float(meta["design_weight"])
    return rows


def wait_for_server(session: requests.Session, health_url: str, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + 240
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited with code {process.returncode}")
        try:
            response = session.get(health_url, timeout=3)
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(1)
    raise TimeoutError(f"llama-server did not become healthy: {last_error}")


def existing_results(path: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return results
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed result line {line_number}: {exc}") from exc
            results[record["row_id"]] = record
    return results


def write_progress(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def run_model(
    model_key: str,
    model_path: Path,
    rows: list[dict[str, Any]],
    output_dir: Path,
    server_exe: Path,
    port: int,
) -> None:
    result_path = output_dir / f"{model_key}.jsonl"
    progress_path = output_dir / f"{model_key}.progress.json"
    stdout_path = output_dir / f"{model_key}.server.stdout.log"
    stderr_path = output_dir / f"{model_key}.server.stderr.log"
    completed = existing_results(result_path)
    if len(completed) == len(rows):
        return

    command = [
        str(server_exe),
        "--model", str(model_path),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--ctx-size", "4096",
        "--threads", "8",
        "--batch-size", "512",
        "--ubatch-size", "512",
        "--parallel", "1",
        "--n-gpu-layers", "99",
        "--flash-attn", "on",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--reasoning", "off",
        "--reasoning-budget", "0",
        "--chat-template-kwargs", '{"enable_thinking":false}',
    ]
    session = requests.Session()
    session.trust_env = False
    model_started = time.monotonic()
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        process = subprocess.Popen(
            command,
            cwd=server_exe.parent,
            stdout=stdout,
            stderr=stderr,
        )
        try:
            wait_for_server(session, f"http://127.0.0.1:{port}/health", process)
            with result_path.open("a", encoding="utf-8", newline="\n") as destination:
                for index, row in enumerate(rows, start=1):
                    if row["row_id"] in completed:
                        continue
                    result = classify(
                        session,
                        f"http://127.0.0.1:{port}/v1/chat/completions",
                        model_key,
                        row,
                    )
                    output = {
                        "row_id": row["row_id"],
                        "pair_id": row["pair_id"],
                        "split": row["split"],
                        "source": row["source"],
                        "published_at": row["published_at"],
                        "target_company": row["target_company"],
                        "model": model_key,
                        "prompt_version": PROMPT_VERSION,
                        **result,
                    }
                    destination.write(json.dumps(output, ensure_ascii=False) + "\n")
                    destination.flush()
                    completed[row["row_id"]] = output
                    write_progress(
                        progress_path,
                        {
                            "model": model_key,
                            "completed": len(completed),
                            "total": len(rows),
                            "valid": sum(bool(item.get("valid")) for item in completed.values()),
                            "retries": sum(bool(item.get("used_retry")) for item in completed.values()),
                            "elapsed_seconds": time.monotonic() - model_started,
                            "last_row_index": index,
                            "last_row_id": row["row_id"],
                        },
                    )
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def cohen_kappa(pairs: list[tuple[str, str]], labels: list[str]) -> float | None:
    if not pairs:
        return None
    total = len(pairs)
    observed = sum(left == right for left, right in pairs) / total
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    expected = sum(left_counts[label] * right_counts[label] for label in labels) / (total * total)
    if expected == 1:
        return 1.0 if observed == 1 else None
    return (observed - expected) / (1 - expected)


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def fmt_kappa(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def confusion_markdown(
    pairs: list[tuple[str, str]], labels: list[str], left_name: str, right_name: str
) -> str:
    counts = Counter(pairs)
    header = f"| {left_name} \\ {right_name} | " + " | ".join(labels) + " |"
    rule = "|---|" + "---:|" * len(labels)
    rows = [
        f"| {left} | " + " | ".join(str(counts[(left, right)]) for right in labels) + " |"
        for left in labels
    ]
    return "\n".join([header, rule, *rows])


def build_report(
    rows: list[dict[str, Any]], output_dir: Path, left_key: str, right_key: str
) -> dict[str, Any]:
    left = existing_results(output_dir / f"{left_key}.jsonl")
    right = existing_results(output_dir / f"{right_key}.jsonl")
    common_ids = [row["row_id"] for row in rows if row["row_id"] in left and row["row_id"] in right]
    valid_ids = [row_id for row_id in common_ids if left[row_id].get("valid") and right[row_id].get("valid")]

    relevance_pairs = [(left[i]["relevance"], right[i]["relevance"]) for i in valid_ids]
    both_relevant = [i for i in valid_ids if left[i]["relevance"] == right[i]["relevance"] == "relevant"]
    sentiment_pairs = [(left[i]["sentiment_label"], right[i]["sentiment_label"]) for i in both_relevant]

    def combined(record: dict[str, Any]) -> str:
        return "irrelevant" if record["relevance"] == "irrelevant" else str(record["sentiment_label"])

    combined_pairs = [(combined(left[i]), combined(right[i])) for i in valid_ids]
    relevance_agreement = sum(a == b for a, b in relevance_pairs) / len(relevance_pairs) if relevance_pairs else None
    sentiment_agreement = sum(a == b for a, b in sentiment_pairs) / len(sentiment_pairs) if sentiment_pairs else None
    combined_agreement = sum(a == b for a, b in combined_pairs) / len(combined_pairs) if combined_pairs else None

    weights = {row["row_id"]: float(row["design_weight"]) for row in rows}
    weight_total = sum(weights[i] for i in valid_ids)
    weighted_combined = (
        sum(weights[i] for i in valid_ids if combined(left[i]) == combined(right[i])) / weight_total
        if weight_total else None
    )

    relevance_labels = ["irrelevant", "relevant"]
    sentiment_labels = ["negative", "neutral", "positive"]
    combined_labels = ["irrelevant", "negative", "neutral", "positive"]
    distributions = {
        key: Counter(combined(results[i]) for i in valid_ids)
        for key, results in ((left_key, left), (right_key, right))
    }
    elapsed = {
        key: sum(float(results[i].get("elapsed_seconds") or 0) for i in results)
        for key, results in ((left_key, left), (right_key, right))
    }
    retries = {
        key: sum(bool(item.get("used_retry")) for item in results.values())
        for key, results in ((left_key, left), (right_key, right))
    }
    invalid = {
        key: sum(not bool(item.get("valid")) for item in results.values())
        for key, results in ((left_key, left), (right_key, right))
    }

    metrics = {
        "rows": len(rows),
        "common_rows": len(common_ids),
        "valid_pair_rows": len(valid_ids),
        "both_relevant_rows": len(both_relevant),
        "relevance_agreement": relevance_agreement,
        "relevance_kappa": cohen_kappa(relevance_pairs, relevance_labels),
        "sentiment_agreement_both_relevant": sentiment_agreement,
        "sentiment_kappa_both_relevant": cohen_kappa(sentiment_pairs, sentiment_labels),
        "combined_agreement": combined_agreement,
        "combined_design_weighted_agreement": weighted_combined,
        "combined_kappa": cohen_kappa(combined_pairs, combined_labels),
        "distribution": {key: dict(value) for key, value in distributions.items()},
        "retries": retries,
        "invalid": invalid,
        "elapsed_seconds_sum": elapsed,
    }
    (output_dir / "comparison_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    q_dist = distributions[left_key]
    g_dist = distributions[right_key]
    report = f"""# Сравнение Qwen3.5-9B и Gemma 4 E4B на 300 статьях

Выборка: замороженные 300 пар «статья–компания» C04b (120 development, 180 held-out). Промпт одинаков по смыслу; применён нативный chat template каждой модели, reasoning отключён, temperature = 0, seed = 42.

## Итог

- Согласие по relevance: **{pct(relevance_agreement)}**, Cohen’s κ = **{fmt_kappa(metrics['relevance_kappa'])}**.
- Среди {len(both_relevant)} случаев, где обе модели признали новость релевантной, согласие по sentiment: **{pct(sentiment_agreement)}**, κ = **{fmt_kappa(metrics['sentiment_kappa_both_relevant'])}**.
- Полное согласие по итоговому классу `irrelevant/negative/neutral/positive`: **{pct(combined_agreement)}**, κ = **{fmt_kappa(metrics['combined_kappa'])}**.
- Design-weighted согласие итогового класса: **{pct(weighted_combined)}**.
- Невалидные ответы: Qwen — {invalid[left_key]}, Gemma — {invalid[right_key]}; повторный запрос потребовался: Qwen — {retries[left_key]}, Gemma — {retries[right_key]}.

## Распределение итоговых классов

| Модель | irrelevant | negative | neutral | positive |
|---|---:|---:|---:|---:|
| Qwen3.5-9B | {q_dist['irrelevant']} | {q_dist['negative']} | {q_dist['neutral']} | {q_dist['positive']} |
| Gemma 4 E4B | {g_dist['irrelevant']} | {g_dist['negative']} | {g_dist['neutral']} | {g_dist['positive']} |

## Матрица расхождений итогового класса

{confusion_markdown(combined_pairs, combined_labels, 'Qwen', 'Gemma')}

## Ограничение интерпретации

Это сравнение согласованности моделей, а не точности. Без завершённой независимой человеческой разметки нельзя утверждать, какая модель чаще права. Выборка намеренно переобогащена сложными случаями, поэтому невзвешенные доли не являются оценками распределения всего корпуса.
"""
    (output_dir / "comparison_report.md").write_text(report, encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-csv", required=True, type=Path)
    parser.add_argument("--selection-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--qwen-model", required=True, type=Path)
    parser.add_argument("--gemma-model", required=True, type=Path)
    parser.add_argument("--server-exe", type=Path)
    parser.add_argument("--port", type=int, default=8085)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    sample_csv = args.sample_csv.resolve(strict=True)
    selection_jsonl = args.selection_jsonl.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    qwen_model = args.qwen_model.resolve(strict=True)
    gemma_model = args.gemma_model.resolve(strict=True)
    if args.server_exe:
        server_exe = args.server_exe.resolve(strict=True)
    else:
        found = shutil.which("llama-server")
        if not found:
            raise SystemExit("llama-server is not on PATH")
        server_exe = Path(found).resolve(strict=True)

    rows = load_rows(sample_csv, selection_jsonl)
    if args.limit:
        rows = rows[: args.limit]

    manifest = {
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "sample_csv": {"path": str(sample_csv), "sha256": sha256_file(sample_csv)},
        "selection_jsonl": {"path": str(selection_jsonl), "sha256": sha256_file(selection_jsonl)},
        "server_exe": str(server_exe),
        "models": {
            "qwen35_9b_q4km": {"path": str(qwen_model), "sha256": sha256_file(qwen_model)},
            "gemma4_e4b_q4qat": {"path": str(gemma_model), "sha256": sha256_file(gemma_model)},
        },
        "rows": len(rows),
        "temperature": 0.0,
        "seed": 42,
        "reasoning": "off",
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    run_model("qwen35_9b_q4km", qwen_model, rows, output_dir, server_exe, args.port)
    run_model("gemma4_e4b_q4qat", gemma_model, rows, output_dir, server_exe, args.port)
    metrics = build_report(rows, output_dir, "qwen35_9b_q4km", "gemma4_e4b_q4qat")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
