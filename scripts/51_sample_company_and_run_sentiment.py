#!/usr/bin/env python3
"""Freeze random company candidates from the active map; run the existing Gemma prompt.

Source archives and mappings are read-only. New runs refuse existing output folders.
The run can resume from its append-only results and never terminates unrelated servers.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import secrets
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def markdown(rows, results=None):
    labels = {"positive": "+", "negative": "−", "neutral": "0"}
    lines = ["| № | Дата в архиве | СМИ | Метч | Статья" + (" | Оценка" if results is not None else "") + " |",
             "|---|---|---|---|---" + ("|---" if results is not None else "") + "|"]
    for i, row in enumerate(rows, 1):
        title = str(row["title"]).replace("|", " / ").replace("\n", " ").replace("[", "(").replace("]", ")")
        date = str(row["published_at"] or "—")[:10]
        signal = row["match_signal"].replace("name", "имя").replace("industry", "отрасль").replace("product", "продукт")
        source = {"kommersant": "Ъ", "prime": "Прайм", "interfax": "Интерфакс", "rbc": "РБК"}.get(row["source"], row["source"])
        line = f"| {i} | {date} | {source} | {signal} | [{title}]({row['article_url']})"
        if results is not None:
            result = results.get(row["row_id"], {})
            label = ("ошибка" if result and not result.get("valid") else "не относится" if result.get("relevance") == "irrelevant"
                     else labels.get(result.get("sentiment_label"), "ожидает"))
            line += f" | {label}"
        lines.append(line + " |")
    return "\n".join(lines) + "\n"


def prepare(args):
    config = json.loads(args.active_config.read_text(encoding="utf-8"))
    active = Path(config["active_directory"])
    assert (active / config["completion_marker"]).is_file(), "Active mapping lacks SUCCESS"
    registry = pq.read_table(active / config["company_registry"]).to_pylist()
    company = next(r for r in registry if r["company_id"] == args.company_id)
    mapping = active / config["article_company_map"]
    candidates = pq.read_table(mapping, filters=[("company_id", "=", args.company_id)]).to_pylist()
    assert len({r["article_id"] for r in candidates}) == len(candidates), "Duplicate target pairs"
    original_candidates = len(candidates)
    excluded_ids = set()
    if args.exclude_sample:
        excluded_ids = {json.loads(line)["article_id"] for line in args.exclude_sample.read_text(encoding="utf-8").splitlines() if line.strip()}
        candidates = [r for r in candidates if r["article_id"] not in excluded_ids]
    if len(candidates) < args.count:
        raise ValueError("Not enough company candidates")
    seed = args.seed if args.seed is not None else secrets.randbits(64)
    selected = random.Random(seed).sample(candidates, args.count)
    keys = {(r["media"], r["article_url"]): r for r in selected}
    assert len(keys) == len(selected), "Non-unique media/URL keys"
    links = pa.array([r["article_url"] for r in selected])
    articles = Path(config["run_root"]) / "articles_all_from_news_sqlite.parquet"
    stat = articles.stat()
    source = pq.ParquetFile(articles)
    found = {}
    started = time.monotonic()
    for group in range(source.num_row_groups):
        meta = source.read_row_group(group, columns=["media", "link"])
        mask = pc.is_in(meta["link"], value_set=links)
        if not pc.any(mask).as_py():
            continue
        for article in source.read_row_group(group).filter(mask).to_pylist():
            key = (article["media"], article["link"])
            if key not in keys:
                continue
            if article["title"] != keys[key]["title"]:
                raise ValueError(f"Conflicting title for {key}")
            if key in found and article["text"] != found[key]["text"]:
                raise ValueError(f"Conflicting archived texts for {key}")
            found[key] = article
        if len(found) == len(selected):
            break
    if set(found) != set(keys):
        raise ValueError(f"Missing texts for {len(set(keys) - set(found))} selected articles")
    bundled = Path(config["run_root"]) / "python_packages_20260930"
    if bundled.is_dir():
        sys.path.insert(0, str(bundled))
    mapper = load_module(Path(config["run_root"]) / "45_map_articles_to_companies.py", "article_mapper")
    rows = []
    for i, row in enumerate(selected, 1):
        article = found[(row["media"], row["article_url"])]
        if not str(article["text"] or "").strip():
            raise ValueError(f"Selected article lacks text: {row['article_id']}")
        assert mapper.make_article_id(article["media"], article["link"], article["date"], article["title"], article["text"]) == row["article_id"]
        pair_id = "p_" + hashlib.blake2b(f"{row['article_id']}\t{row['company_id']}".encode(), digest_size=16).hexdigest()
        rows.append(dict(row_id=f"SBER-{i:03d}", pair_id=pair_id, article_id=row["article_id"], company_id=row["company_id"],
                         target_company=company["company"], source=row["media"], published_at=row["date"].isoformat() if row["date"] is not None else None,
                         title=row["title"], article_url=row["article_url"], match_signal=row["match_signal"], matched_terms=json.loads(row["matched_terms"]),
                         text=article["text"], text_sha256=hashlib.sha256(article["text"].encode()).hexdigest()))
    assert (stat.st_size, stat.st_mtime_ns) == (articles.stat().st_size, articles.stat().st_mtime_ns)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "sample.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "sample_table.md").write_text(markdown(rows), encoding="utf-8")
    manifest = dict(mapping_version=config["version"], mapping=str(mapping), article_source=str(articles),
                    company_id=args.company_id, target_company=company["company"], available_pairs=len(candidates),
                    original_available_pairs=original_candidates, excluded_previous_sample=str(args.exclude_sample) if args.exclude_sample else None,
                    excluded_previous_articles=original_candidates-len(candidates),
                    selected=len(rows), seed=seed, sampling="uniform unique target pairs, no date or signal filter",
                    source_articles=source.metadata.num_rows, sources=dict(Counter(r["source"] for r in rows)),
                    match_signals=dict(Counter(r["match_signal"] for r in rows)), elapsed_seconds=round(time.monotonic() - started, 2),
                    source_unchanged=True, full_archived_texts_preserved=True)
    write_json(args.output_dir / "sample_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=True), flush=True)


def run(args):
    module = load_module(args.project_root / "scripts/43_compare_sentiment_models.py", "sentiment_comparison")
    rows = [json.loads(line) for line in (args.output_dir / "sample.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    results_path = args.output_dir / "gemma_results.jsonl"
    completed = module.existing_results(results_path)
    if not set(completed) <= {r["row_id"] for r in rows}:
        raise ValueError("Results do not belong to the frozen sample")
    model = args.project_root / "models/llm/gemma-4-E4B_q4_0-it.gguf"
    if not model.is_file() or not args.server_exe.is_file():
        raise FileNotFoundError("Gemma model or llama-server executable missing")
    metadata = dict(model=str(model), model_size_bytes=model.stat().st_size, model_mtime_ns=model.stat().st_mtime_ns,
                    server=str(args.server_exe), prompt_version=module.PROMPT_VERSION,
                    system_prompt=module.SYSTEM_PROMPT, response_format=module.RESPONSE_FORMAT,
                    runner_sha256=module.sha256_file(args.project_root / "scripts/43_compare_sentiment_models.py"),
                    sample_sha256=module.sha256_file(args.output_dir / "sample.jsonl"), max_prompt_tokens=3400,
                    context_size=4096, truncation_policy="Prefix truncated only when token budget is exceeded; original text retained in sample.jsonl")
    manifest_path = args.output_dir / "run_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != metadata:
        raise ValueError("Cannot resume after changing prompt/model/sample")
    write_json(manifest_path, metadata)
    session = requests.Session()
    session.trust_env = False
    base = f"http://127.0.0.1:{args.port}"
    try:
        session.get(base + "/health", timeout=2)
    except requests.RequestException:
        pass
    else:
        raise RuntimeError("Port already occupied; refusing to interfere with another server")
    command = [str(args.server_exe), "--model", str(model), "--host", "127.0.0.1", "--port", str(args.port),
               "--ctx-size", "4096", "--threads", "8", "--batch-size", "512", "--ubatch-size", "512", "--parallel", "1",
               "--n-gpu-layers", "99", "--flash-attn", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
               "--reasoning", "off", "--reasoning-budget", "0", "--chat-template-kwargs", '{"enable_thinking":false}']
    started = time.monotonic()
    def tokenize(row, text):
        request = session.post(base + "/tokenize", json={"content": module.SYSTEM_PROMPT + "\n" + module.build_user_prompt({**row, "text": text}), "add_special": True}, timeout=60)
        request.raise_for_status()
        return len(request.json()["tokens"])
    def progress(status):
        write_json(args.output_dir / "progress.json", dict(status=status, completed=len(completed), total=len(rows),
                   valid=sum(bool(r.get("valid")) for r in completed.values()), retries=sum(bool(r.get("used_retry")) for r in completed.values()),
                   truncated=sum(bool(r.get("text_truncated")) for r in completed.values()), elapsed_seconds=round(time.monotonic() - started, 2)))
    progress("loading_model")
    with (args.output_dir / "server.stdout.log").open("ab") as stdout, (args.output_dir / "server.stderr.log").open("ab") as stderr:
        process = subprocess.Popen(command, cwd=args.server_exe.parent, stdout=stdout, stderr=stderr)
        try:
            module.wait_for_server(session, base + "/health", process)
            with results_path.open("a", encoding="utf-8") as handle:
                for row in rows:
                    if row["row_id"] in completed:
                        continue
                    text = row["text"]
                    original_tokens = tokenize(row, text)
                    if original_tokens > 3400:
                        lo, hi = 0, len(text)
                        while lo < hi:
                            mid = (lo + hi + 1) // 2
                            if tokenize(row, text[:mid]) <= 3400:
                                lo = mid
                            else:
                                hi = mid - 1
                        text = text[:lo]
                    result = module.classify(session, base + "/v1/chat/completions", "gemma4_e4b_q4qat", {**row, "text": text})
                    record = {k: v for k, v in row.items() if k != "text"}
                    record.update(result)
                    record.update(model="gemma4_e4b_q4qat", prompt_version=module.PROMPT_VERSION, original_prompt_tokens=original_tokens,
                                  text_truncated=len(text) < len(row["text"]), input_characters=len(text), input_sha256=hashlib.sha256(text.encode()).hexdigest())
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    completed[row["row_id"]] = record
                    progress("running")
                    print(json.dumps({"completed": len(completed), "total": len(rows), "valid": record["valid"]}), flush=True)
            progress("completed")
            (args.output_dir / "results_table.md").write_text(markdown(rows, completed), encoding="utf-8")
            summary = dict(total=len(rows), valid=sum(bool(r.get("valid")) for r in completed.values()),
                           relevance=dict(Counter(r.get("relevance") or "error" for r in completed.values())),
                           sentiment=dict(Counter(r["sentiment_label"] for r in completed.values() if r.get("valid") and r.get("relevance") == "relevant")),
                           elapsed_seconds=round(time.monotonic() - started, 2), truncated=sum(bool(r.get("text_truncated")) for r in completed.values()))
            write_json(args.output_dir / "summary.json", summary)
            (args.output_dir / "SUCCESS").write_text("All frozen sample rows attempted; see summary for valid/error counts.\n", encoding="utf-8")
            print(json.dumps(summary), flush=True)
        except Exception:
            progress("failed")
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run"])
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--active-config", type=Path)
    parser.add_argument("--company-id", default="c_93c22968778d0f04cae81af8")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--exclude-sample", type=Path, help="Exclude article IDs in a previous sample JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8087)
    parser.add_argument("--server-exe", type=Path, default=Path("./local/AppData/Local/Microsoft/WinGet/Packages/ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe/llama-server.exe"))
    args = parser.parse_args()
    args.active_config = args.active_config or args.project_root / "config/company_lists/active_company_mapping.json"
    (prepare if args.mode == "prepare" else run)(args)


if __name__ == "__main__":
    main()
