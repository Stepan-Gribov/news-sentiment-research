"""Resumable Gemma classification of every frozen title/name candidate.

News, mappings and models are read-only. A private SQLite result ledger is
authoritative; JSONL is a readable projection rebuilt from the ledger on resume.
The target-level prompt/schema are imported unchanged from the pinned script 43.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as futures
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def stamp(path):
    value = path.stat()
    return dict(path=str(path), bytes=value.st_size, mtime_ns=value.st_mtime_ns)


def pair_id(article_id, company_id):
    return "p_" + hashlib.blake2b(f"{article_id}\t{company_id}".encode(), digest_size=16).hexdigest()


@contextmanager
def exclusive_lock(path):
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def prepare(args, comparison, report):
    input_path = args.run_dir / "input.jsonl"
    manifest_path = args.run_dir / "input_manifest.json"
    if (args.run_dir / "INPUT_SUCCESS").is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(stamp(Path(item["path"])) != item for item in manifest["source_stamps"]):
            raise RuntimeError("Frozen input sources changed; refusing unsafe resume")
        if input_path.stat().st_size != manifest["input_bytes"]:
            raise RuntimeError("Frozen input size changed")
        if comparison.sha256_file(input_path) != manifest["input_sha256"]:
            raise RuntimeError("Frozen input hash changed")
        return manifest
    config = json.loads(args.mapping_config.read_text(encoding="utf-8"))
    mapping_dir = Path(config["directory"])
    if not (mapping_dir / config["completion_marker"]).is_file():
        raise RuntimeError("Additional mapping is not complete")
    map_path = mapping_dir / config["article_company_map"]
    registry_path = mapping_dir / config["company_registry"]
    source_stamps = [stamp(path) for path in [args.mapping_config, map_path, registry_path, args.articles]]
    registry = {row["company_id"]: row for row in pq.read_table(registry_path).to_pylist()}
    mapping = pq.read_table(map_path).to_pylist()
    if any(row["company_id"] == "c_90995d77be411a67f2ef2b72" or row["match_signal"] != "name" for row in mapping):
        raise RuntimeError("Input is not the title/name-only map without GAZ")
    keys = collections.defaultdict(list)
    expected_pairs = set()
    for row in mapping:
        key = (row["media"], row["article_url"])
        pid = pair_id(row["article_id"], row["company_id"])
        if pid in expected_pairs:
            raise RuntimeError("Duplicate article/company pair")
        expected_pairs.add(pid)
        keys[key].append(row)
    links = pa.array(sorted({url for _, url in keys}))
    article_source = pq.ParquetFile(args.articles)
    mapper_config = json.loads((args.project_root / "config/company_lists/active_company_mapping.json").read_text(encoding="utf-8"))
    mapper = load(Path(mapper_config["run_root"]) / "scripts/45_map_articles_to_companies.py", "article_identity")
    found = set()
    emitted = 0
    issues = collections.Counter()
    source_counts = collections.Counter()
    partial = input_path.with_suffix(".jsonl.partial")
    digest = hashlib.sha256()
    started = time.monotonic()
    with partial.open("wb") as destination:
        def emit(meta, article, issue=None):
            nonlocal emitted
            company = registry[meta["company_id"]]
            body = str(article.get("text") or "")
            record = dict(row_id=f"TITLE-{emitted+1:08d}", pair_id=pair_id(meta["article_id"], meta["company_id"]),
                article_id=meta["article_id"], company_id=meta["company_id"], target_company=company.get("company_name") or company["company"],
                source=meta["media"], published_at=meta["date"].isoformat() if meta["date"] is not None else None,
                title=meta["title"], article_url=meta["article_url"], match_signal="name", match_scope="title",
                matched_terms=json.loads(meta["matched_terms"]), text=body, text_sha256=hashlib.sha256(body.encode()).hexdigest())
            if issue:
                record["input_issue"] = issue
                issues[issue] += 1
            line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
            destination.write(line)
            digest.update(line)
            emitted += 1
            source_counts[meta["media"]] += 1
        for group in range(article_source.num_row_groups):
            meta = article_source.read_row_group(group, columns=["media", "link"])
            mask = pc.is_in(meta["link"], value_set=links)
            if pc.any(mask).as_py():
                selected = article_source.read_row_group(group, columns=["media", "link", "title", "date", "text"]).filter(mask)
                for article in selected.to_pylist():
                    key = (article["media"], article["link"])
                    if key not in keys or key in found:
                        continue
                    aid = mapper.make_article_id(article["media"], article["link"], article["date"], article["title"], article["text"])
                    for item in keys[key]:
                        if aid != item["article_id"] or article["title"] != item["title"]:
                            raise RuntimeError("Article identity/title disagrees with frozen mapping")
                        emit(item, article, "empty_archived_text" if not str(article["text"] or "").strip() else None)
                    found.add(key)
            report(stage="preparing_input", status="preparing", total=len(mapping), prepared_pairs=emitted,
                   source_row_groups_done=group+1, source_row_groups_total=article_source.num_row_groups)
        for key in keys.keys() - found:
            for item in keys[key]:
                emit(item, {}, "missing_archived_article")
        destination.flush()
        os.fsync(destination.fileno())
    if emitted != len(mapping):
        raise RuntimeError("Prepared pair count mismatch")
    if any(stamp(Path(item["path"])) != item for item in source_stamps):
        raise RuntimeError("Input source changed during preparation")
    os.replace(partial, input_path)
    manifest = dict(total=emitted, unique_articles=len(keys), found_articles=len(found), source_stamps=source_stamps,
        input_sha256=digest.hexdigest(), input_bytes=input_path.stat().st_size, input_issues=dict(issues),
        source_counts=dict(source_counts), archived_texts_preserved=True, source_unchanged=True,
        elapsed_seconds=round(time.monotonic()-started, 2))
    save(manifest_path, manifest)
    (args.run_dir / "INPUT_SUCCESS").write_text("Frozen full-text input reconciled with title/name map.\n", encoding="utf-8")
    print(json.dumps({"input_prepared": manifest}, ensure_ascii=True), flush=True)
    return manifest


def session():
    value = requests.Session()
    value.trust_env = False
    return value


def classify(row, base, comparison):
    if row.get("input_issue"):
        result = dict(valid=False, relevance=None, sentiment_label=None, short_rationale="", used_retry=False,
                      input_issue=row["input_issue"], elapsed_seconds=0)
        return {**{key: value for key, value in row.items() if key != "text"}, **result}
    client = session()
    try:
        def tokens(text):
            response = client.post(base + "/tokenize", json={"content": comparison.SYSTEM_PROMPT + "\n" + comparison.build_user_prompt({**row, "text": text}), "add_special": True}, timeout=60)
            response.raise_for_status()
            return len(response.json()["tokens"])
        text = row["text"]
        original_tokens = tokens(text)
        if original_tokens > 3400:
            low, high = 0, len(text)
            while low < high:
                middle = (low+high+1)//2
                if tokens(text[:middle]) <= 3400:
                    low = middle
                else:
                    high = middle-1
            text = text[:low]
        result = comparison.classify(client, base + "/v1/chat/completions", "gemma4_e4b_q4qat", {**row, "text": text})
        return {**{key: value for key, value in row.items() if key != "text"}, **result,
            "model": "gemma4_e4b_q4qat", "prompt_version": comparison.PROMPT_VERSION,
            "original_prompt_tokens": original_tokens, "text_truncated": len(text) < len(row["text"]),
            "input_characters": len(text), "input_sha256": hashlib.sha256(text.encode()).hexdigest()}
    finally:
        client.close()


def regenerate_jsonl(connection, path):
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        for (body,) in connection.execute("SELECT result_json FROM results ORDER BY task_index"):
            destination.write(body + "\n")
    os.replace(temporary, path)


def execute(args):
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(args.run_dir / ".runner.lock"):
        comparison_path = Path(__file__).with_name("43_compare_sentiment_models.py")
        comparison = load(comparison_path, "frozen_sentiment_contract")
        progress = dict(status="starting", stage="preflight", completed=0, valid=0, errors=0, runner_pid=os.getpid(), total=341799)
        progress_lock = threading.RLock()
        stopped = threading.Event()
        gpu_file = args.run_dir / "resources.jsonl"
        def report(**values):
            with progress_lock:
                progress.update(values)
                progress["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                save(args.run_dir / "progress.json", progress)
        def heartbeat():
            while not stopped.wait(10):
                try:
                    query = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                    if query.returncode == 0:
                        values = [int(value.strip()) for value in query.stdout.splitlines()[0].split(",")]
                        resource = dict(gpu_utilization_percent=values[0], gpu_memory_used_mib=values[1], gpu_memory_total_mib=values[2], gpu_temperature_c=values[3], timestamp_utc=datetime.now(timezone.utc).isoformat())
                        with gpu_file.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(resource) + "\n")
                        report(**resource)
                    else:
                        report()
                except Exception:
                    report()
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        model = args.project_root / "models/llm/gemma-4-E4B_q4_0-it.gguf"
        process = None
        connection = None
        try:
            if not model.is_file() or not args.server_exe.is_file():
                raise FileNotFoundError("Gemma model/server missing")
            if shutil.disk_usage(args.run_dir).free < 5*1024**3:
                raise RuntimeError("Less than 5 GiB disk headroom for frozen input/results")
            manifest = prepare(args, comparison, report)
            immutable = dict(input_sha256=manifest["input_sha256"], prompt_version=comparison.PROMPT_VERSION,
                prompt_sha256=hashlib.sha256(comparison.SYSTEM_PROMPT.encode()).hexdigest(), response_format=comparison.RESPONSE_FORMAT,
                comparison_script_sha256=comparison.sha256_file(comparison_path), runner_script_sha256=comparison.sha256_file(Path(__file__)),
                model=stamp(model), max_prompt_tokens=3400, context_per_slot=4096, temperature=0, seed=42,
                truncation_policy="Same as validated sample: prefix constrained to 3400 tokens; full archived text retained in input.jsonl")
            frozen = args.run_dir / "run_manifest.json"
            if frozen.exists() and json.loads(frozen.read_text(encoding="utf-8")) != immutable:
                raise RuntimeError("Frozen prompt/model/input/runner changed; refusing resume")
            if not frozen.exists():
                save(frozen, immutable)
            connection = sqlite3.connect(args.run_dir / "results.sqlite3")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("CREATE TABLE IF NOT EXISTS results(pair_id TEXT PRIMARY KEY,task_index INTEGER NOT NULL UNIQUE,valid INTEGER NOT NULL,result_json TEXT NOT NULL,elapsed_seconds REAL NOT NULL)")
            connection.commit()
            completed_ids = {row[0] for row in connection.execute("SELECT pair_id FROM results")}
            total = manifest["total"]
            completed = len(completed_ids)
            valid = connection.execute("SELECT COUNT(*) FROM results WHERE valid=1").fetchone()[0]
            output = args.run_dir / "gemma_results.jsonl"
            regenerate_jsonl(connection, output)
            report(total=total, completed=completed, valid=valid, errors=completed-valid, remaining=total-completed)
            if completed == total:
                report(status="completed" if valid == total else "completed_with_errors", stage="completed", eta_seconds=0)
                return
            probe = socket.socket()
            try:
                probe.bind(("127.0.0.1", args.port))
            finally:
                probe.close()
            base = f"http://127.0.0.1:{args.port}"
            client = session()
            report(status="loading_model", stage="loading_model")
            stdout = (args.run_dir / "server.stdout.log").open("ab")
            stderr = (args.run_dir / "server.stderr.log").open("ab")
            parallel = args.parallel
            for attempt in range(2):
                command = [str(args.server_exe), "--model", str(model), "--host", "127.0.0.1", "--port", str(args.port),
                    "--ctx-size", str(4096*parallel), "--threads", "8", "--batch-size", "512", "--ubatch-size", "512",
                    "--parallel", str(parallel), "--device", "Vulkan0", "--n-gpu-layers", "99", "--flash-attn", "on",
                    "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--reasoning", "off", "--reasoning-budget", "0",
                    "--chat-template-kwargs", '{"enable_thinking":false}']
                process = subprocess.Popen(command, cwd=args.server_exe.parent, stdout=stdout, stderr=stderr)
                report(server_pid=process.pid, parallel=parallel)
                try:
                    comparison.wait_for_server(client, base+"/health", process)
                    break
                except Exception:
                    log = "\n".join((args.run_dir / name).read_text(encoding="utf-8", errors="replace")[-15000:] for name in ["server.stderr.log", "server.stdout.log"]).lower()
                    if attempt == 0 and parallel > 1 and any(word in log for word in ["out of memory", "allocation", "failed to allocate"]):
                        process.terminate()
                        process.wait(timeout=20)
                        parallel = 1
                        continue
                    raise
            save(args.run_dir / "engine_config.json", dict(command=command, actual_parallel=parallel, gpu="Vulkan0 RTX4060", server_pid=process.pid))
            recent = collections.deque(maxlen=100)
            errors_in_row = 0
            inference_started = time.monotonic()
            initial_completed = completed
            report(status="running", stage="inference", inference_started_at_utc=datetime.now(timezone.utc).isoformat())
            def rows():
                for index, line in enumerate((args.run_dir / "input.jsonl").open(encoding="utf-8"), 1):
                    row = json.loads(line)
                    if row["pair_id"] not in completed_ids:
                        yield index, row
            iterator = iter(rows())
            pending = {}
            with output.open("a", encoding="utf-8") as destination, futures.ThreadPoolExecutor(max_workers=parallel) as pool:
                def fill():
                    while len(pending) < parallel:
                        try:
                            index, row = next(iterator)
                        except StopIteration:
                            break
                        pending[pool.submit(classify, row, base, comparison)] = index
                fill()
                while pending:
                    if process.poll() is not None:
                        raise RuntimeError("Owned llama-server exited; stopping instead of creating mass errors")
                    done, _ = futures.wait(pending, timeout=10, return_when=futures.FIRST_COMPLETED)
                    if not done:
                        continue
                    for future in done:
                        index = pending.pop(future)
                        result = future.result()
                        body = json.dumps(result, ensure_ascii=False)
                        connection.execute("INSERT INTO results VALUES(?,?,?,?,?)", (result["pair_id"], index, int(bool(result["valid"])), body, float(result.get("elapsed_seconds", 0))))
                        connection.commit()
                        destination.write(body + "\n")
                        destination.flush()
                        completed_ids.add(result["pair_id"])
                        completed += 1
                        valid += bool(result["valid"])
                        errors_in_row = 0 if result["valid"] or result.get("input_issue") else errors_in_row+1
                        now = time.monotonic()
                        recent.append((now, completed))
                        elapsed = now-inference_started
                        overall = (completed-initial_completed)/elapsed if elapsed else 0
                        rate = (recent[-1][1]-recent[0][1])/(recent[-1][0]-recent[0][0]) if len(recent)>10 else overall
                        eta = (total-completed)/rate if rate and completed-initial_completed >= 20 else None
                        report(completed=completed, valid=valid, errors=completed-valid, remaining=total-completed,
                            speed_pairs_per_second=round(rate, 4), speed_pairs_per_hour=round(rate*3600, 1),
                            eta_seconds=round(eta) if eta is not None else None, inference_elapsed_seconds=round(elapsed, 1),
                            last_pair_id=result["pair_id"], last_valid=result["valid"])
                        if completed % 100 == 0:
                            print(json.dumps({"completed": completed, "total": total, "errors": completed-valid, "eta_seconds": eta}), flush=True)
                        if errors_in_row >= 5:
                            raise RuntimeError("Five consecutive inference errors; stopped safely; inspect results/logs")
                    fill()
            regenerate_jsonl(connection, output)
            report(status="completed" if valid == total else "completed_with_errors", stage="completed", eta_seconds=0)
            save(args.run_dir / "summary.json", dict(progress))
            (args.run_dir / ("SUCCESS" if valid == total else "COMPLETE_WITH_ERRORS")).write_text("All frozen pairs attempted; see results and error counts.\n", encoding="utf-8")
        except Exception as error:
            report(status="failed", stage="failed", error=str(error))
            raise
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            if connection is not None:
                connection.close()
            stopped.set()
            thread.join(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--mapping-config", type=Path, required=True)
    parser.add_argument("--articles", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--server-exe", type=Path, default=Path("llama-server"))
    args = parser.parse_args()
    if args.parallel not in (1, 2):
        raise ValueError("Only tested safe parallel values 1 and 2 are allowed")
    execute(args)


if __name__ == "__main__":
    main()
