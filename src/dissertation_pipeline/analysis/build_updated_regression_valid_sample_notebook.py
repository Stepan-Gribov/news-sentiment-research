import json
from pathlib import Path


PROJECT_DIR = Path.cwd()
SOURCE_NOTEBOOK = PROJECT_DIR / "LLM_News_Core_Replication.ipynb"
OUTPUT_NOTEBOOK = PROJECT_DIR / "Updated_Regression_Valid_Sample.ipynb"


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise ValueError(f"Pattern not found for replacement: {old[:80]!r}")
    return text.replace(old, new, 1)


def main() -> None:
    nb = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code" and cell.get("cell_type") != "markdown":
            continue

        src = "".join(cell.get("source", []))

        if "# LLM News Sentiment: Core Research Replication" in src:
            src = src.replace(
                "# LLM News Sentiment: Core Research Replication",
                "# Updated Regression Notebook: Valid Sample",
            )
            src = src.replace(
                "This notebook rebuilds the core logic of the original `0.Regressions.ipynb` with the new LLM sentiment results from:",
                (
                    "This notebook rebuilds the core logic of the original `0.Regressions.ipynb` "
                    "using the updated local data layout, but estimates regressions only on the "
                    "valid sample with full control coverage."
                ),
            )
            src += (
                "\n"
                "Valid-sample rule:\n"
                "\n"
                "- estimation sample is restricted to dates `<= 2024-11-30`\n"
                "- this keeps the original control block and return logic methodologically intact\n"
                "- the 2025 append period is intentionally excluded from regression estimation until missing market/control inputs are completed\n"
            )

        if "PROJECT_DIR = Path.cwd()" in src and "BASE_FUNDS_DIR" in src:
            src = replace_once(
                src,
                'OUTPUT_DIR = PROJECT_DIR / "llm_core_replication_outputs"',
                'OUTPUT_DIR = PROJECT_DIR / "updated_valid_sample_outputs"',
            )
            src = replace_once(
                src,
                'Y_PATH = BASE_FUNDS_DIR / "Data/Y/Y_interpolated.csv"',
                'Y_PATH = PROJECT_DIR / "Y_interpolated_updated.csv"',
            )
            src = replace_once(
                src,
                'QUOTES_PATH = BASE_FUNDS_DIR / "Data/quotes_data/All_instr_df.csv"',
                'PORTFOLIO_METRICS_PATH = PROJECT_DIR / "portfolio_metrics/portfolio_metrics_updated.csv"',
            )
            src = replace_once(
                src,
                'PORTFOLIO_VAR_SOURCE_PATH = BASE_FUNDS_DIR / "Code/Analisys/portfolio_variances.xlsx"',
                'ESTIMATION_END = pd.Timestamp("2024-11-30")',
            )
            src = replace_once(
                src,
                'PORTFOLIO_CACHE_PATH = OUTPUT_DIR / "portfolio_metrics.pkl"',
                'PORTFOLIO_CACHE_PATH = OUTPUT_DIR / "portfolio_metrics_valid.pkl"',
            )
            src = replace_once(
                src,
                'FUND_LIQUIDITY_CACHE_PATH = OUTPUT_DIR / "fund_liquidity.pkl"',
                'FUND_LIQUIDITY_CACHE_PATH = OUTPUT_DIR / "fund_liquidity_valid.pkl"',
            )
            src = replace_once(
                src,
                'NEWS_PANEL_CSV_PATH = OUTPUT_DIR / "News_panel_LLM_parts100.csv"',
                'NEWS_PANEL_CSV_PATH = OUTPUT_DIR / "News_panel_LLM_parts100_valid.csv"',
            )
            src = replace_once(
                src,
                'NEWS_PANEL_XLSX_PATH = OUTPUT_DIR / "News_panel_LLM_parts100.xlsx"',
                'NEWS_PANEL_XLSX_PATH = OUTPUT_DIR / "News_panel_LLM_parts100_valid.xlsx"',
            )

        if "def build_portfolio_metrics(" in src:
            start = src.index("def build_portfolio_metrics(")
            replacement = """
def build_portfolio_metrics(
    panel,
    portfolio_metrics_path,
            cache_path,
            lookback_days=90,
        ):
            if cache_path.exists():
                return pd.read_pickle(cache_path)

            metrics = pd.read_csv(portfolio_metrics_path)
            metrics["date"] = pd.to_datetime(metrics["date"])
    metrics = metrics[metrics["date"] <= ESTIMATION_END].copy()
    metrics.to_pickle(cache_path)
    return metrics
"""
            src = src[:start] + replacement

        if 'Y_funds_df = pd.read_csv(Y_PATH, encoding="utf-16", sep="\\\\t", low_memory=False)' in src:
            src = replace_once(
                src,
                'Y_funds_df["date"] = pd.to_datetime(Y_funds_df["date"])',
                'Y_funds_df["date"] = pd.to_datetime(Y_funds_df["date"])\n        Y_funds_df = Y_funds_df[Y_funds_df["date"] <= ESTIMATION_END].copy()',
            )
            src = replace_once(
                src,
                'controls["date"] = pd.to_datetime(controls["date"])',
                (
                    'controls["date"] = pd.to_datetime(controls["date"])\n'
                    '        controls = controls[controls["date"] <= ESTIMATION_END].copy()'
                ),
            )

        if "portfolio_metrics = build_portfolio_metrics(" in src:
            src = replace_once(
                src,
                "    quotes_path=QUOTES_PATH,\n    portfolio_var_source_path=PORTFOLIO_VAR_SOURCE_PATH,\n    cache_path=PORTFOLIO_CACHE_PATH,\n    fund_liquidity_cache_path=FUND_LIQUIDITY_CACHE_PATH,\n    lookback_days=90,",
                "    portfolio_metrics_path=PORTFOLIO_METRICS_PATH,\n    cache_path=PORTFOLIO_CACHE_PATH,\n    lookback_days=90,",
            )

        if 'pif_prices["Дата"] = pd.to_datetime(pif_prices["Дата"])' in src:
            src = replace_once(
                src,
                'pif_prices["Дата"] = pd.to_datetime(pif_prices["Дата"])',
                'pif_prices["Дата"] = pd.to_datetime(pif_prices["Дата"], format="%d.%m.%Y", errors="coerce")\n        pif_prices = pif_prices[pif_prices["Дата"] <= ESTIMATION_END].copy()',
            )

        if 'panel_results_to_df(h0_excess_weighted).head(20)' in src:
            src += """

        print("Valid-sample estimation end:", ESTIMATION_END)
        print("All regressions above are restricted to the fully covered sample.")
"""

        cell["source"] = src.splitlines(keepends=True)

    OUTPUT_NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUTPUT_NOTEBOOK)


if __name__ == "__main__":
    main()
