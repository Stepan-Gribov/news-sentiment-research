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
        # Updated Main Analysis

        This notebook moves from diagnostics to the actual analysis stage on the updated data.

        It uses:
        - updated fund holdings: `Y_interpolated_updated.csv`
        - updated LLM sentiment panel: `llm_core_replication_outputs/News_panel_LLM_parts100.csv`
        - updated portfolio metrics: `portfolio_metrics/portfolio_metrics_updated.csv`

        The notebook reproduces the main panel-regression blocks from the original research:
        - `H1`: sentiment and portfolio volatility
        - `H2`: sentiment and fund liquidity

        Two specification families are estimated:
        - `full_sample`: uses variables available through the updated period
        - `legacy_controls`: uses the original macro-control block and therefore runs only where those controls exist
        """
    ),
    code_cell(
        """
        from pathlib import Path
        import math

        import numpy as np
        import pandas as pd
        import statsmodels.api as sm

        pd.set_option("display.max_columns", 200)
        pd.set_option("display.width", 220)
        """
    ),
    code_cell(
        """
        try:
            from linearmodels.panel import PanelOLS
        except ImportError as exc:
            raise ImportError(
                "The notebook requires `linearmodels`. Install it with: "
                "`python3 -m pip install --user linearmodels`"
            ) from exc
        """
    ),
    code_cell(
        """
        PROJECT_DIR = Path.cwd()
        OUTPUT_DIR = PROJECT_DIR / "regressions_updated"
        REGR_DIR = OUTPUT_DIR / "regression_tables"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        REGR_DIR.mkdir(parents=True, exist_ok=True)

        LEGACY_FUNDS_DIR = Path("./local/Desktop/my_data/Master thesis/funds")

        Y_PATH = PROJECT_DIR / "Y_interpolated_updated.csv"
        NEWS_PANEL_PATH = PROJECT_DIR / "llm_core_replication_outputs" / "News_panel_LLM_parts100.csv"
        METRICS_PATH = PROJECT_DIR / "portfolio_metrics" / "portfolio_metrics_updated.csv"
        CONTROLS_PATH = LEGACY_FUNDS_DIR / "Data" / "Controls" / "controls.xlsx"
        PROCESSED_ALL_PATH = LEGACY_FUNDS_DIR / "Data" / "processed_all.xlsx"
        """
    ),
    code_cell(
        """
        def first_two_nonzero_round(x):
            if pd.isna(x):
                return x
            absx = abs(x)
            if absx == 0:
                return 0.0
            frac = absx if absx < 1 else absx - math.floor(absx)
            if frac == 0:
                decimals = 0
            else:
                first_pos = math.ceil(-math.log10(frac))
                decimals = first_pos + 1
            return round(x, decimals)


        def results_to_df_two_nonzero(results):
            df = pd.DataFrame(
                {
                    "coef": results.params,
                    "std_err": results.std_errors,
                    "t_stat": results.tstats,
                    "p_value": results.pvalues,
                }
            )
            for col in ["coef", "std_err", "t_stat", "p_value"]:
                df[col] = df[col].apply(first_two_nonzero_round)

            def stars(p):
                if p < 0.01:
                    return "***"
                if p < 0.05:
                    return "**"
                if p < 0.10:
                    return "*"
                return ""

            df["signif"] = df["p_value"].apply(stars)
            df["coef_sig"] = df["coef"].astype(str) + df["signif"]
            return df


        def panel_results_to_df(models):
            rows = []
            for spec_name, res in models.items():
                summary = {
                    "R2_within": res.rsquared_within,
                    "R2_between": res.rsquared_between,
                    "R2_overall": res.rsquared_overall,
                    "N_obs": int(res.nobs),
                }
                for term, val in summary.items():
                    rows.append(
                        {
                            "spec": spec_name,
                            "term": term,
                            "coef": val,
                            "std_err": None,
                            "t_stat": None,
                            "p_value": None,
                            "signif": "",
                            "coef_sig": f"{val:.4f}",
                        }
                    )

                coef_df = results_to_df_two_nonzero(res).reset_index().rename(columns={"index": "term"})
                for _, row in coef_df.iterrows():
                    rows.append(
                        {
                            "spec": spec_name,
                            "term": row["term"],
                            "coef": row["coef"],
                            "std_err": row["std_err"],
                            "t_stat": row["t_stat"],
                            "p_value": row["p_value"],
                            "signif": row["signif"],
                            "coef_sig": row["coef_sig"],
                        }
                    )
            return pd.DataFrame(rows)


        def run_panel_models(
            df,
            entity_col,
            y_var,
            main_vars,
            controls,
            lag=1,
            entity_effects=False,
            time_effects=False,
            cov_type="clustered",
            cluster_entity=True,
            cluster_time=True,
        ):
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index([entity_col, "date"]).sort_index()

            lagged = []
            for var in main_vars:
                lag_var = f"{var}_L{lag}"
                df[lag_var] = df.groupby(level=entity_col)[var].shift(lag)
                lagged.append(lag_var)

            specs = {
                "current": main_vars + controls,
                "lagged": lagged + controls,
                "both": main_vars + lagged + controls,
            }

            results = {}
            for name, vars_ in specs.items():
                work = df[[y_var] + vars_].replace([np.inf, -np.inf], np.nan).dropna()
                exog = sm.add_constant(work[vars_])
                model = PanelOLS(
                    work[y_var],
                    exog,
                    entity_effects=entity_effects,
                    time_effects=time_effects,
                    drop_absorbed=True,
                )
                results[name] = model.fit(
                    cov_type=cov_type,
                    cluster_entity=cluster_entity,
                    cluster_time=cluster_time,
                )
            return results


        def safe_net_sent(p, n, u):
            total = p + n + u
            if total == 0 or pd.isna(total):
                return 0.0
            return (p - n) / total


        def winsorize_series(series, lower=0.01, upper=0.99):
            lo = series.quantile(lower)
            hi = series.quantile(upper)
            return series.clip(lower=lo, upper=hi)
        """
    ),
    code_cell(
        """
        Y_funds_df = pd.read_csv(Y_PATH, encoding="utf-16", sep="\\t", low_memory=False)
        Y_funds_df["date"] = pd.to_datetime(Y_funds_df["date"])
        Y_funds_df = Y_funds_df[~Y_funds_df["Company"].astype(str).str.contains("рбк", case=False, na=False)].copy()

        news_panel_llm = pd.read_csv(NEWS_PANEL_PATH, low_memory=False)
        news_panel_llm["end_date"] = pd.to_datetime(news_panel_llm["end_date"])

        portfolio_metrics = pd.read_csv(METRICS_PATH, low_memory=False)
        portfolio_metrics["date"] = pd.to_datetime(portfolio_metrics["date"])

        controls = pd.read_excel(CONTROLS_PATH)
        controls["date"] = pd.to_datetime(controls["date"])

        fund_assets_wide = pd.read_excel(PROCESSED_ALL_PATH)
        fund_assets_wide["fund_key"] = (
            fund_assets_wide["Название фонда"] + "__" + fund_assets_wide["Номер регистрации"].astype(str)
        )
        fund_assets_wide["date"] = pd.to_datetime(fund_assets_wide["date"])
        fund_classes = fund_assets_wide[
            ["Название фонда", "Название УК", "Номер регистрации", "Тип фонда", "Категория фонда", "fund_key"]
        ].drop_duplicates(subset="fund_key")

        print("Y:", Y_funds_df.shape, Y_funds_df["date"].min(), Y_funds_df["date"].max())
        print("News:", news_panel_llm.shape, news_panel_llm["end_date"].min(), news_panel_llm["end_date"].max())
        print("Metrics:", portfolio_metrics.shape, portfolio_metrics["date"].min(), portfolio_metrics["date"].max())
        print("Controls:", controls.shape, controls["date"].min(), controls["date"].max())
        """
    ),
    code_cell(
        """
        overlap_cols = [
            col for col in news_panel_llm.columns
            if col in Y_funds_df.columns and col not in ["end_date", "firm_in_text"]
        ]

        Panel = pd.merge(
            Y_funds_df,
            news_panel_llm.drop(columns=overlap_cols),
            how="inner",
            left_on=["date", "Company"],
            right_on=["end_date", "firm_in_text"],
        ).drop_duplicates()

        Panel = Panel[~Panel["Название фонда"].astype(str).str.contains("индекс", case=False, na=False)].copy()

        total_assets_on_date = (
            Panel.groupby("date", as_index=False)["Сумма, руб."]
            .sum()
            .rename(columns={"Сумма, руб.": "total_assets_on_date"})
        )
        Panel = Panel.merge(total_assets_on_date, on="date", how="left")
        Panel["share_in_total_assets, %"] = (Panel["Сумма, руб."] / Panel["total_assets_on_date"]) * 100

        sent_total = Panel["Positive"] + Panel["Negative"] + Panel["Neutral"]
        Panel["Net_sent_score_partial"] = np.where(sent_total > 0, (Panel["Positive"] - Panel["Negative"]) / sent_total, 0)
        Panel["Net_sent_score_weighted"] = Panel["Net_sent_score_partial"] * Panel["Доля, %"]
        Panel["Positive_weighted"] = Panel["Positive"] * Panel["Доля, %"]
        Panel["Negative_weighted"] = Panel["Negative"] * Panel["Доля, %"]
        Panel["Neutral_weighted"] = Panel["Neutral"] * Panel["Доля, %"]

        unique_funds_by_date = (
            Panel.groupby("date")["fund_key"]
            .nunique()
            .reset_index()
            .rename(columns={"fund_key": "unique_funds"})
        )

        controls_agg = controls.groupby("date", as_index=False).agg(
            {"Flow": "sum", "Imoex": "mean", "Ruonia": "mean", "USD_RUB": "mean", "RVI": "mean"}
        )
        controls_agg["Flow"] = controls_agg["Flow"] / 1_000_000_000
        controls_agg = controls_agg.sort_values("date").reset_index(drop=True)
        controls_agg["imoex_ret"] = controls_agg["Imoex"].div(controls_agg["Imoex"].shift(1)) - 1
        controls_agg["imoex_log_ret"] = np.log1p(controls_agg["imoex_ret"])

        Panel = Panel.merge(unique_funds_by_date, on="date", how="left")
        Panel["dummy_mar2014"] = (Panel["date"] >= pd.Timestamp("2014-03-01")).astype(int)
        Panel["dummy_feb2022"] = (Panel["date"] >= pd.Timestamp("2022-02-01")).astype(int)
        Panel = Panel.merge(controls_agg, on="date", how="left").drop_duplicates()
        Panel = Panel.sort_values(["fund_key", "firm_in_text", "ISIN конечного актива", "date"])
        Panel["RVI"] = Panel["RVI"].fillna(0)

        print("Panel:", Panel.shape)
        print("Funds:", Panel["fund_key"].nunique(), " Firms:", Panel["firm_in_text"].nunique())
        """
    ),
    code_cell(
        """
        Panel_agg_ft = Panel.groupby(["fund_key", "date"], as_index=False).agg(
            {
                "Класс активов": "sum",
                "Сумма, руб.": "sum",
                "Доля, %": "sum",
                "Net_sent_score_weighted": "sum",
                "Positive_weighted": "sum",
                "Negative_weighted": "sum",
                "Neutral_weighted": "sum",
                "Negative": "sum",
                "Neutral": "sum",
                "Positive": "sum",
                "Flow": "first",
                "Imoex": "first",
                "imoex_ret": "first",
                "Ruonia": "first",
                "USD_RUB": "first",
                "RVI": "first",
                "unique_funds": "first",
                "dummy_mar2014": "first",
                "dummy_feb2022": "first",
            }
        )

        print("Panel_agg_ft:", Panel_agg_ft.shape)
        """
    ),
    code_cell(
        """
        FULL_SAMPLE_CONTROLS = ["unique_funds", "dummy_mar2014", "dummy_feb2022"]
        LEGACY_CONTROLS = ["Flow", "Imoex", "imoex_ret", "Ruonia", "USD_RUB", "unique_funds", "dummy_mar2014", "dummy_feb2022"]
        """
    ),
    markdown_cell("## H1: Portfolio Volatility"),
    code_cell(
        """
        Panel_H1 = (
            portfolio_metrics[["date", "fund_key", "portfolio_var"]]
            .merge(Panel_agg_ft, on=["date", "fund_key"], how="inner")
            .drop_duplicates(subset=["fund_key", "date"])
            .sort_values(["date", "fund_key"], ascending=[False, True])
        )

        Panel_H1["portfolio_vol"] = np.sqrt(Panel_H1["portfolio_var"])
        Panel_H1 = Panel_H1[Panel_H1["portfolio_var"] > 0].copy()

        Panel_H1["net_sent_score"] = [
            safe_net_sent(p, n, u)
            for p, n, u in zip(Panel_H1["Positive"], Panel_H1["Negative"], Panel_H1["Neutral"])
        ]
        sent_total = Panel_H1["Positive"] + Panel_H1["Negative"] + Panel_H1["Neutral"]
        weighted_total = Panel_H1["Positive_weighted"] + Panel_H1["Negative_weighted"] + Panel_H1["Neutral_weighted"]
        Panel_H1["Neg_share"] = np.where(sent_total > 0, Panel_H1["Negative"] / sent_total, 0)
        Panel_H1["Pos_share"] = np.where(sent_total > 0, Panel_H1["Positive"] / sent_total, 0)
        Panel_H1["Neg_share_weighted"] = np.where(weighted_total > 0, Panel_H1["Negative_weighted"] / weighted_total, 0)
        Panel_H1["Pos_share_weighted"] = np.where(weighted_total > 0, Panel_H1["Positive_weighted"] / weighted_total, 0)

        Panel_H1 = Panel_H1[
            (Panel_H1["portfolio_vol"] >= Panel_H1["portfolio_vol"].quantile(0.01))
            & (Panel_H1["portfolio_vol"] <= Panel_H1["portfolio_vol"].quantile(0.99))
        ].copy()

        Panel_H1["portfolio_vol_wins"] = winsorize_series(Panel_H1["portfolio_vol"])
        Panel_H1["log_portfolio_vol"] = np.log(Panel_H1["portfolio_vol"])
        Panel_H1["log_portfolio_var"] = Panel_H1["log_portfolio_vol"] * 2

        for col in ["Negative", "Neutral", "Positive", "Negative_weighted", "Neutral_weighted", "Positive_weighted"]:
            Panel_H1[f"log_{col}"] = np.where(Panel_H1[col] > 0, np.log(Panel_H1[col]), 0)

        Panel_H1["fund_size"] = Panel_H1.groupby(["fund_key", "date"])["Сумма, руб."].transform("sum")
        Panel_H1["RVI"] = Panel_H1["RVI"].ffill()

        print(Panel_H1.shape)
        print(Panel_H1["date"].min(), Panel_H1["date"].max())
        """
    ),
    code_cell(
        """
        h1_full_netsent = run_panel_models(
            df=Panel_H1,
            entity_col="fund_key",
            y_var="log_portfolio_vol",
            main_vars=["Net_sent_score_weighted"],
            controls=FULL_SAMPLE_CONTROLS,
            lag=1,
            entity_effects=True,
            time_effects=False,
        )
        h1_full_components = run_panel_models(
            df=Panel_H1,
            entity_col="fund_key",
            y_var="log_portfolio_vol",
            main_vars=["log_Negative_weighted", "log_Neutral_weighted", "log_Positive_weighted"],
            controls=FULL_SAMPLE_CONTROLS,
            lag=1,
            entity_effects=True,
            time_effects=False,
        )

        h1_legacy_df = Panel_H1.dropna(subset=LEGACY_CONTROLS).copy()
        h1_legacy_netsent = run_panel_models(
            df=h1_legacy_df,
            entity_col="fund_key",
            y_var="log_portfolio_vol",
            main_vars=["Net_sent_score_weighted"],
            controls=LEGACY_CONTROLS,
            lag=1,
            entity_effects=True,
            time_effects=False,
        )
        h1_legacy_components = run_panel_models(
            df=h1_legacy_df,
            entity_col="fund_key",
            y_var="log_portfolio_vol",
            main_vars=["log_Negative_weighted", "log_Neutral_weighted", "log_Positive_weighted"],
            controls=LEGACY_CONTROLS,
            lag=1,
            entity_effects=True,
            time_effects=False,
        )

        panel_results_to_df(h1_full_netsent).to_excel(REGR_DIR / "h1_vol_fullsample_netsent.xlsx", index=False)
        panel_results_to_df(h1_full_components).to_excel(REGR_DIR / "h1_vol_fullsample_components.xlsx", index=False)
        panel_results_to_df(h1_legacy_netsent).to_excel(REGR_DIR / "h1_vol_legacycontrols_netsent.xlsx", index=False)
        panel_results_to_df(h1_legacy_components).to_excel(REGR_DIR / "h1_vol_legacycontrols_components.xlsx", index=False)

        print("=== H1 full_sample: Net_sent_score_weighted ===")
        for spec, result in h1_full_netsent.items():
            print(f"\\n--- {spec} ---")
            print(result)

        print("\\n=== H1 full_sample: sentiment components ===")
        for spec, result in h1_full_components.items():
            print(f"\\n--- {spec} ---")
            print(result)

        print("\\n=== H1 legacy_controls: Net_sent_score_weighted ===")
        for spec, result in h1_legacy_netsent.items():
            print(f"\\n--- {spec} ---")
            print(result)

        print("\\n=== H1 legacy_controls: sentiment components ===")
        for spec, result in h1_legacy_components.items():
            print(f"\\n--- {spec} ---")
            print(result)

        panel_results_to_df(h1_full_netsent)
        """
    ),
    markdown_cell("## H2: Fund Liquidity"),
    code_cell(
        """
        Panel_H2 = (
            portfolio_metrics[["date", "fund_key", "fund_liquidity"]]
            .merge(Panel_agg_ft, on=["date", "fund_key"], how="inner")
            .drop_duplicates(subset=["fund_key", "date"])
            .sort_values(["date", "fund_key"], ascending=[False, True])
        )

        Panel_H2["net_sent_score"] = [
            safe_net_sent(p, n, u)
            for p, n, u in zip(Panel_H2["Positive"], Panel_H2["Negative"], Panel_H2["Neutral"])
        ]
        sent_total = Panel_H2["Positive"] + Panel_H2["Negative"] + Panel_H2["Neutral"]
        weighted_total = Panel_H2["Positive_weighted"] + Panel_H2["Negative_weighted"] + Panel_H2["Neutral_weighted"]
        Panel_H2["Neg_share"] = np.where(sent_total > 0, Panel_H2["Negative"] / sent_total, 0)
        Panel_H2["Pos_share"] = np.where(sent_total > 0, Panel_H2["Positive"] / sent_total, 0)
        Panel_H2["Neg_share_weighted"] = np.where(weighted_total > 0, Panel_H2["Negative_weighted"] / weighted_total, 0)
        Panel_H2["Pos_share_weighted"] = np.where(weighted_total > 0, Panel_H2["Positive_weighted"] / weighted_total, 0)

        Panel_H2 = Panel_H2.merge(
            fund_classes[["Тип фонда", "Категория фонда", "fund_key"]],
            on="fund_key",
            how="left",
        )
        Panel_H2 = Panel_H2[Panel_H2["Категория фонда"].isin(["Акции", "Смешанный"])].copy()
        Panel_H2["RVI"] = Panel_H2["RVI"].fillna(0)

        Panel_H2 = Panel_H2[
            (Panel_H2["fund_liquidity"] >= Panel_H2["fund_liquidity"].quantile(0.01))
            & (Panel_H2["fund_liquidity"] <= Panel_H2["fund_liquidity"].quantile(0.99))
        ].copy()

        Panel_H2["fund_liquidity_wins"] = winsorize_series(Panel_H2["fund_liquidity"])
        Panel_H2["log_fund_liquidity"] = np.where(Panel_H2["fund_liquidity"] > 0, np.log(Panel_H2["fund_liquidity"]), 0)
        Panel_H2 = Panel_H2[Panel_H2["log_fund_liquidity"] != 0].copy()

        for col in ["Negative", "Neutral", "Positive", "Negative_weighted", "Neutral_weighted", "Positive_weighted"]:
            Panel_H2[f"log_{col}"] = np.where(Panel_H2[col] > 0, np.log(Panel_H2[col]), 0)

        Panel_H2["fund_size"] = Panel_H2.groupby(["fund_key", "date"])["Сумма, руб."].transform("sum")

        print(Panel_H2.shape)
        print(Panel_H2["date"].min(), Panel_H2["date"].max())
        """
    ),
    code_cell(
        """
        h2_full_netsent = run_panel_models(
            df=Panel_H2,
            entity_col="fund_key",
            y_var="log_fund_liquidity",
            main_vars=["Net_sent_score_weighted"],
            controls=FULL_SAMPLE_CONTROLS,
            lag=1,
            entity_effects=True,
            time_effects=False,
        )
        h2_full_components = run_panel_models(
            df=Panel_H2,
            entity_col="fund_key",
            y_var="log_fund_liquidity",
            main_vars=["log_Negative_weighted", "log_Neutral_weighted", "log_Positive_weighted"],
            controls=FULL_SAMPLE_CONTROLS,
            lag=1,
            entity_effects=True,
            time_effects=False,
        )

        h2_legacy_df = Panel_H2.dropna(subset=LEGACY_CONTROLS).copy()
        h2_legacy_netsent = run_panel_models(
            df=h2_legacy_df,
            entity_col="fund_key",
            y_var="log_fund_liquidity",
            main_vars=["Net_sent_score_weighted"],
            controls=LEGACY_CONTROLS,
            lag=1,
            entity_effects=True,
            time_effects=False,
        )
        h2_legacy_components = run_panel_models(
            df=h2_legacy_df,
            entity_col="fund_key",
            y_var="log_fund_liquidity",
            main_vars=["log_Negative_weighted", "log_Neutral_weighted", "log_Positive_weighted"],
            controls=LEGACY_CONTROLS,
            lag=1,
            entity_effects=True,
            time_effects=False,
        )

        panel_results_to_df(h2_full_netsent).to_excel(REGR_DIR / "h2_liq_fullsample_netsent.xlsx", index=False)
        panel_results_to_df(h2_full_components).to_excel(REGR_DIR / "h2_liq_fullsample_components.xlsx", index=False)
        panel_results_to_df(h2_legacy_netsent).to_excel(REGR_DIR / "h2_liq_legacycontrols_netsent.xlsx", index=False)
        panel_results_to_df(h2_legacy_components).to_excel(REGR_DIR / "h2_liq_legacycontrols_components.xlsx", index=False)

        print("=== H2 full_sample: Net_sent_score_weighted ===")
        for spec, result in h2_full_netsent.items():
            print(f"\\n--- {spec} ---")
            print(result)

        print("\\n=== H2 full_sample: sentiment components ===")
        for spec, result in h2_full_components.items():
            print(f"\\n--- {spec} ---")
            print(result)

        print("\\n=== H2 legacy_controls: Net_sent_score_weighted ===")
        for spec, result in h2_legacy_netsent.items():
            print(f"\\n--- {spec} ---")
            print(result)

        print("\\n=== H2 legacy_controls: sentiment components ===")
        for spec, result in h2_legacy_components.items():
            print(f"\\n--- {spec} ---")
            print(result)

        panel_results_to_df(h2_full_netsent)
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


output_path = Path(__file__).resolve().parent / "Updated_Main_Analysis.ipynb"
output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(output_path)
