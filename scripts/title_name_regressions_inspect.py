"""Read-only schema/pointer inspection; never evaluates a regression or LLM."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb


BASE = Path("./local/research-data-service")
ROOT = Path(".")


def quoted(name):
    return '"' + name.replace('"', '""') + '"'


def main():
    c = duckdb.connect(str(BASE / "research.duckdb"), read_only=True,
                       config={"threads": 1, "memory_limit": "512MB"})
    result = {"catalog": str(BASE / "research.duckdb"), "tables": [], "schemas": {}, "samples": {}}
    try:
        result["sources"] = c.execute("SELECT * FROM meta.sources").fetchdf().to_dict("records")
        tables = c.execute("SELECT table_schema,table_name FROM information_schema.tables "
                           "WHERE table_schema IN ('funds','refs','market','analysis','llm') "
                           "ORDER BY 1,2").fetchall()
        result["tables"] = [".".join(x) for x in tables]
        for schema, table in tables:
            name = schema + "." + table
            if schema == "analysis" and any(x in table.lower() for x in ["regress", "coef", "result"]):
                continue  # Do not inspect previous coefficient tables before the spec freeze.
            if schema == "llm":
                continue  # The separate audit reconciles the canonical SQLite directly.
            try:
                qualified = quoted(schema) + "." + quoted(table)
                result["schemas"][name] = c.execute("DESCRIBE SELECT * FROM " + qualified).fetchall()
                if name in {"funds.holdings", "funds.reports", "funds.report_lines", "refs.companies", "market.fund_returns"}:
                    result["samples"][name] = c.execute("SELECT * FROM " + qualified + " LIMIT 2").fetchdf().to_dict("records")
            except Exception as error:
                result["schemas"][name] = {"error": str(error)}
        run = ROOT / "news_data/llm_runs/gemma_title_name_no_gaz_20261008_01"
        result["llm_completion_markers"] = {p: (run/p).exists() for p in ["SUCCESS", "COMPLETE_WITH_ERRORS"]}
        for name in ["input_manifest.json", "run_manifest.json", "summary.json"]:
            if (run/name).exists():
                result[name] = json.loads((run/name).read_text(encoding="utf-8-sig"))
        print(json.dumps(result, ensure_ascii=True, default=str))
    finally:
        c.close()


if __name__ == "__main__":
    main()
