#!/usr/bin/env python3
"""Small, read-only, standard-library lookup for the shared project graph."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def tokens(value):
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(value)).replace("_", " ")
    return set(re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="*", help="Question or project keywords")
    parser.add_argument("--budget", type=int, default=1500, help="Approximate output-token ceiling")
    parser.add_argument("--vocab", action="store_true", help="Show graph vocabulary matching the query")
    parser.add_argument("--graph", type=Path, default=ROOT / "graphify-out/graph.json")
    args = parser.parse_args()
    if not args.graph.is_file():
        parser.exit(4, "No project graph found. Use a focused source search; graph construction is a separate task.\n")
    data = json.loads(args.graph.read_text(encoding="utf-8"))
    nodes = {str(n["id"]): n for n in data.get("nodes", [])}
    edges = data.get("links", data.get("edges", []))
    vocabulary = set().union(*(tokens(n.get("label", "")) for n in nodes.values())) if nodes else set()
    question = " ".join(args.question)
    requested = tokens(question)
    # Exact vocabulary wins. English expansion only handles a trailing plural s;
    # a five-character prefix would incorrectly equate alpha with alphabetical.
    selected = set()
    for query in requested:
        if query in vocabulary:
            selected.add(query)
        elif query.endswith("s") and query[:-1] in vocabulary:
            selected.add(query[:-1])
        elif not query.isascii() and len(query) >= 5:
            selected.update(v for v in vocabulary if not v.isascii() and len(v) >= 5 and v[:5] == query[:5])
    selected = sorted(t for t in selected if len(t) >= 2)[:12]
    if args.vocab:
        print("Graph vocabulary:", ", ".join(selected) if selected else "no matching label tokens")
        return
    if not selected:
        parser.exit(4, "No matching graph vocabulary. Try a cited filename, project identifier or English keywords; otherwise use rg.\n")
    frequencies = collections.Counter()
    labels = {}
    for key, node in nodes.items():
        labels[key] = tokens(node.get("label", ""))
        frequencies.update(labels[key])
    scores = {key: sum(math.log(1 + len(nodes) / (1 + frequencies[t]))
                       for t in selected if t in label)
              for key, label in labels.items()}
    for key, node in nodes.items():
        if node.get("file_type") in {"concept", "rationale"}:
            scores[key] *= 1.2
        if str(node.get("source_file", "")).endswith("RESEARCH_NOTES.md"):
            scores[key] *= 1.1
    seeds = sorted((key for key in nodes if scores[key] > 0),
                   key=lambda key: (-scores[key], key))[:4]
    adjacency = collections.defaultdict(list)
    for edge in edges:
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if source in nodes and target in nodes and source != target:
            adjacency[source].append((target, edge))
            adjacency[target].append((source, edge))
    candidates = set(seeds)
    for seed in seeds:
        candidates.update(other for other, _ in adjacency[seed]
                          if not nodes[other].get("external") or scores[other] > 0)
    ranked = seeds + sorted(candidates - set(seeds),
                            key=lambda key: (-scores[key], -len(adjacency[key]), key))[:12]
    manifest_path = args.graph.parent / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    indexed = {x["path"]: x for x in manifest.get("files", [])}
    checked = {}

    def freshness(source):
        if not source:
            return "unrecorded"
        path = Path(source)
        if not path.is_absolute():
            path = ROOT / path
        try:
            relative = path.relative_to(ROOT).as_posix()
        except ValueError:
            return "external: verify source"
        if relative in checked:
            return checked[relative]
        stamp = indexed.get(relative)
        if not path.is_file():
            state = "missing: verify path"
        elif stamp is None:
            state = "not in hash manifest: verify source"
        else:
            stat = path.stat()
            state = "unchanged" if (stat.st_size == stamp.get("size_bytes") and
                    stat.st_mtime_ns == stamp.get("mtime_ns")) else (
                    "unchanged (timestamp changed)" if hashlib.sha256(path.read_bytes()).hexdigest() == stamp.get("sha256")
                    else "STALE: read current source")
        checked[relative] = state
        return state

    lines = ["Query expanded from graph vocabulary: " + ", ".join(selected),
             "Graph snapshot: " + str(manifest.get("built_at", "unknown")),
             "Navigation evidence only; verify current sources before claiming results."]
    char_budget = max(600, args.budget * 4)

    def emit(line):
        if sum(len(x) + 1 for x in lines) + len(line) > char_budget:
            return False
        lines.append(line)
        return True

    for key in ranked:
        node = nodes[key]
        source = node.get("source_file", "")
        line = "NODE " + str(node.get("label", key)) + " | " + str(source)
        location = node.get("source_location")
        if location:
            line += ":" + str(location)
        line += " | " + freshness(source)
        if not emit(line):
            break
        detail = node.get("summary") or node.get("description") or node.get("rationale")
        if detail:
            emit("  " + str(detail)[:500])
        for other, edge in sorted(adjacency[key], key=lambda pair: (-scores.get(pair[0], 0), pair[0]))[:3]:
            if other in candidates:
                direction = (str(edge.get("relation", "related")) + " -> ") if str(edge.get("source")) == key else ("<- " + str(edge.get("relation", "related")) + " - ")
                emit("  " + direction +
                     str(nodes[other].get("label", other)) + " [" +
                     str(edge.get("confidence", "unrecorded")) + "]")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
