import json
import textwrap
from pathlib import Path


def _src(text: str) -> str:
    return textwrap.dedent(text).strip("\n") + "\n"


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _src(text).splitlines(keepends=True),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _src(text).splitlines(keepends=True),
    }


cells = [
    markdown_cell(
        """
        # Updated Research Diagnostics: LLM News, New Y, and Market-Metrics Coverage

        This notebook is the strict, non-hallucinated continuation of the original regression workflow for the updated data.

        It does **not** assume that the new period is fully reproducible.
        Instead it checks, in order:

        1. updated holdings panel `Y`
        2. updated LLM news panel `X`
        3. control-variable coverage
        4. quote-resolution coverage
        5. portfolio-metric coverage for `portfolio_var` and `fund_liquidity`
        6. fund-return coverage for `fund_ret` / `excess_return`

        The purpose is scientific: before running regressions on the updated sample, verify whether the data needed by the original thesis design are actually available.
        """
    ),
    code_cell(
        """
        from pathlib import Path
        import json

        import numpy as np
        import pandas as pd

        pd.set_option("display.max_columns", 200)
        pd.set_option("display.width", 200)
        """
    ),
    code_cell(
        """
        PROJECT_DIR = Path.cwd()
        OUTPUT_DIR = PROJECT_DIR / "updated_regression_diagnostics_outputs"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        BASE_FUNDS_DIR = Path("./local/Desktop/my_data/Master thesis/funds")
        MASTER_NEWS_CSV = Path(
            "./local/Desktop/my_data/My Drive/4. Disserartion research/calculations/data/news_links/NEWS_PAGES_PARQ/all_news_merged_unified.csv"
        )
        PARTS_DIR = Path("./local/dev/LLM/outputs/parts(100)")

        COMPANY_MAP_PATH = BASE_FUNDS_DIR / "Code/Analisys/company_type_isin1.xlsx"
        BRIDGE_CSV_PATH = BASE_FUNDS_DIR / "Data/csv/All_news_my_LLM_mapped_processed.csv"
        FUND_PRICES_PATH = BASE_FUNDS_DIR / "Data/Y/funds_01012015_17052025.csv"
        PROCESSED_ALL_PATH = BASE_FUNDS_DIR / "Data/processed_all.xlsx"
        LEGACY_CONTROLS_PATH = BASE_FUNDS_DIR / "Data/Controls/controls.xlsx"

        Y_PATH = PROJECT_DIR / "Y_interpolated_updated.csv"
        PORTFOLIO_METRICS_PATH = PROJECT_DIR / "portfolio_metrics/portfolio_metrics_updated.csv"
        APPEND_PORTFOLIO_METRICS_PATH = PROJECT_DIR / "portfolio_metrics/portfolio_metrics_append_2024_12_2025_11.csv"
        APPEND_FUND_RETURNS_PATH = PROJECT_DIR / "portfolio_metrics/fund_returns_append_available.csv"
        QUOTE_RESOLUTION_PATH = PROJECT_DIR / "quotes_data/moex_resolution_report_append_2024_10_2025_11.csv"
        """
    ),
    code_cell(
        """
        def count_mentions(text, firm, ratio=0.8):
            if pd.isna(text) or pd.isna(firm):
                return 0
            firm = str(firm).strip().lower()
            if not firm:
                return 0
            token = firm[: max(1, int(len(firm) * ratio))]
            return str(text).lower().count(token)


        def load_parts_sentiment(parts_dir: Path) -> pd.DataFrame:
            records = []
            for jsonl_path in sorted(parts_dir.glob("*.jsonl")):
                with jsonl_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        obj = json.loads(line)
                        records.append(
                            {
                                "id": obj["id"],
                                "status": obj.get("status"),
                                "sentiment_label": obj.get("sentiment_label"),
                            }
                        )
            return pd.DataFrame(records)


        def safe_net_sent(pos, neg, neu):
            total = pos + neg + neu
            if total == 0 or pd.isna(total):
                return 0.0
            return (pos - neg) / total
        """
    ),
    markdown_cell(
        """
        ## Load Updated Core Data
        """
    ),
    code_cell(
        """
        Y_funds_df = pd.read_csv(Y_PATH, encoding="utf-16", sep="\\t", low_memory=False)
        Y_funds_df["date"] = pd.to_datetime(Y_funds_df["date"])
        Y_funds_df = Y_funds_df[~Y_funds_df["Company"].str.contains("рбк", case=False, na=False)].copy()

        append_cutoff = pd.Timestamp("2024-11-30")
        Y_append = Y_funds_df[Y_funds_df["date"] > append_cutoff].copy()

        print("Y full:", Y_funds_df.shape, Y_funds_df["date"].min(), Y_funds_df["date"].max())
        print("Y append:", Y_append.shape, Y_append["date"].min(), Y_append["date"].max())
        print("Append unique fund-date pairs:", len(Y_append[["fund_key", "date"]].drop_duplicates()))
        """
    ),
    code_cell(
        """
        portfolio_metrics = pd.read_csv(PORTFOLIO_METRICS_PATH)
        portfolio_metrics["date"] = pd.to_datetime(portfolio_metrics["date"])

        append_metrics = pd.read_csv(APPEND_PORTFOLIO_METRICS_PATH)
        append_metrics["date"] = pd.to_datetime(append_metrics["date"])

        append_fund_returns = pd.read_csv(APPEND_FUND_RETURNS_PATH)
        append_fund_returns["date"] = pd.to_datetime(append_fund_returns["date"])

        quote_resolution = pd.read_csv(QUOTE_RESOLUTION_PATH)

        print("portfolio_metrics full:", portfolio_metrics.shape)
        print("append_metrics:", append_metrics.shape)
        print("append_fund_returns:", append_fund_returns.shape)
        print("quote_resolution:", quote_resolution.shape)
        """
    ),
    markdown_cell(
        """
        ## Rebuild the Updated News Panel

        This reuses the old deterministic article-to-firm bridge and only swaps in the new LLM sentiment labels.
        """
    ),
    code_cell(
        """
        master_links = pd.read_csv(MASTER_NEWS_CSV, usecols=["link"]).reset_index().rename(columns={"index": "id"})
        parts_df = load_parts_sentiment(PARTS_DIR)

        link_sentiment = master_links.merge(parts_df, on="id", how="left", validate="one_to_one")
        assert link_sentiment["link"].duplicated().sum() == 0

        bridge_df = pd.read_csv(
            BRIDGE_CSV_PATH,
            encoding="utf-16",
            sep="^",
            usecols=["link", "date", "news_content", "firm_name"],
        ).drop_duplicates()

        bridge_df = bridge_df.merge(
            link_sentiment[["link", "status", "sentiment_label"]],
            on="link",
            how="left",
            validate="many_to_one",
        )

        sentiment_map = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}
        bridge_df["sentiment"] = bridge_df["sentiment_label"].map(sentiment_map).fillna("Н/Д")
        bridge_df["date"] = pd.to_datetime(bridge_df["date"])
        bridge_df = bridge_df.rename(columns={"firm_name": "firm_in_text"})
        bridge_df["n_matches"] = [
            count_mentions(text, firm)
            for text, firm in zip(bridge_df["news_content"], bridge_df["firm_in_text"])
        ]

        print("bridge rows:", len(bridge_df))
        print("classified share:", bridge_df["status"].notna().mean())
        print(bridge_df["sentiment"].value_counts(dropna=False))
        """
    ),
    code_cell(
        """
        fund_dates = sorted(Y_funds_df["date"].drop_duplicates())
        interval_labels = [
            f"{fund_dates[i + 1].strftime('%Y-%m-%d')}:{fund_dates[i].strftime('%Y-%m-%d')}"
            for i in range(len(fund_dates) - 1)
        ]

        news_df = bridge_df.copy()
        news_df["interval"] = pd.cut(
            news_df["date"],
            bins=fund_dates,
            labels=interval_labels,
            right=True,
        )
        news_df = news_df.dropna(subset=["interval"]).copy()
        news_df["interval"] = news_df["interval"].astype(str)
        news_df["end_date"] = pd.to_datetime(news_df["interval"].str.split(":").str[0])

        news_panel_llm = (
            news_df.assign(
                Positive=(news_df["sentiment"] == "Positive").astype(int),
                Negative=(news_df["sentiment"] == "Negative").astype(int),
                Neutral=(news_df["sentiment"] == "Neutral").astype(int),
            )
            .groupby(["end_date", "firm_in_text"], as_index=False)[["Positive", "Negative", "Neutral"]]
            .sum()
        )

        news_panel_llm["net_sent_score"] = [
            safe_net_sent(p, n, u)
            for p, n, u in zip(
                news_panel_llm["Positive"],
                news_panel_llm["Negative"],
                news_panel_llm["Neutral"],
            )
        ]

        print("news panel:", news_panel_llm.shape)
        print("news panel dates:", news_panel_llm["end_date"].min(), news_panel_llm["end_date"].max())
        """
    ),
    markdown_cell(
        """
        ## Merge Updated X and Updated Y
        """
    ),
    code_cell(
        """
        Panel = pd.merge(
            Y_funds_df,
            news_panel_llm,
            how="inner",
            left_on=["date", "Company"],
            right_on=["end_date", "firm_in_text"],
        ).drop_duplicates()

        Panel = Panel[~Panel["Название фонда"].str.contains("индекс", case=False, na=False)].copy()
        Panel_append = Panel[Panel["date"] > append_cutoff].copy()

        print("Panel full:", Panel.shape)
        print("Panel append:", Panel_append.shape)
        print("Append unique fund-date pairs in merged panel:", len(Panel_append[["fund_key", "date"]].drop_duplicates()))
        """
    ),
    markdown_cell(
        """
        ## Coverage Diagnostics for Portfolio Metrics
        """
    ),
    code_cell(
        """
        append_pairs = Y_append[["date", "fund_key"]].drop_duplicates()
        append_pairs = append_pairs.merge(
            append_metrics,
            on=["date", "fund_key"],
            how="left",
        )

        metric_coverage = (
            append_pairs.groupby("date", as_index=False)
            .agg(
                fund_dates=("fund_key", "size"),
                portfolio_var_nonnull=("portfolio_var", lambda s: s.notna().sum()),
                fund_liquidity_nonnull=("fund_liquidity", lambda s: s.notna().sum()),
            )
        )
        metric_coverage["portfolio_var_share"] = metric_coverage["portfolio_var_nonnull"] / metric_coverage["fund_dates"]
        metric_coverage["fund_liquidity_share"] = metric_coverage["fund_liquidity_nonnull"] / metric_coverage["fund_dates"]

        metric_coverage
        """
    ),
    code_cell(
        """
        print("Any append portfolio_var available:", int(append_pairs["portfolio_var"].notna().sum()))
        print("Any append fund_liquidity available:", int(append_pairs["fund_liquidity"].notna().sum()))

        quote_diag = {
            "requested_ids": int(len(quote_resolution)),
            "resolved_ids": int((quote_resolution["status"] == "resolved").sum()),
            "unresolved_ids": int((quote_resolution["status"] != "resolved").sum()),
        }
        print(quote_diag)
        print()
        print("Resolution by source:")
        print(quote_resolution["source"].value_counts(dropna=False))
        """
    ),
    markdown_cell(
        """
        ## Coverage Diagnostics for Fund Returns
        """
    ),
    code_cell(
        """
        fund_ret_coverage = (
            append_fund_returns.groupby("date", as_index=False)
            .agg(
                fund_dates=("fund_key", "size"),
                fund_ret_nonnull=("fund_ret", lambda s: s.notna().sum()),
                excess_return_nonnull=("excess_return", lambda s: s.notna().sum()),
            )
        )
        fund_ret_coverage["fund_ret_share"] = fund_ret_coverage["fund_ret_nonnull"] / fund_ret_coverage["fund_dates"]
        fund_ret_coverage["excess_return_share"] = fund_ret_coverage["excess_return_nonnull"] / fund_ret_coverage["fund_dates"]

        fund_ret_coverage
        """
    ),
    markdown_cell(
        """
        ## Old Controls Coverage

        The original notebook needs `Flow`, `Imoex`, `imoex_ret`, `Ruonia`, `USD_RUB`, `RVI`.
        This cell shows the current hard limit of the local legacy controls file.
        """
    ),
    code_cell(
        """
        controls = pd.read_excel(LEGACY_CONTROLS_PATH)
        controls["date"] = pd.to_datetime(controls["date"])
        controls = controls.sort_values("date")
        controls["imoex_ret"] = controls["Imoex"].div(controls["Imoex"].shift(1)) - 1

        print("controls coverage:", controls["date"].min(), controls["date"].max(), controls.shape)
        print(controls.tail(12).to_string())
        """
    ),
    markdown_cell(
        """
        ## Scientific Conclusion From the Current Data State

        If the cells above show:

        - zero or near-zero coverage for `portfolio_var` / `fund_liquidity` on the updated period
        - no `excess_return` coverage on the updated period
        - macro controls ending before the new sample ends

        then the updated-period regressions are **not yet identified** under the original research design.

        In that case the next task is not regression running, but data completion:

        1. quote source for the new asset universe
        2. updated macro-controls block
        3. if needed, updated fund-unit-price source for `Пай`
        """
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.9",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output_path = Path.cwd() / "Updated_Research_Diagnostics.ipynb"
output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(output_path)
