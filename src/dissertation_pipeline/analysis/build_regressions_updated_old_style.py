import json
import textwrap
from pathlib import Path


def _src(text: str) -> str:
    return textwrap.dedent(text).strip("\n") + "\n"


def markdown_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text).splitlines(keepends=True)}


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _src(text).splitlines(keepends=True),
    }


cells = [
    code_cell(
        """
        import os
        import math
        import json
        import pickle
        import datetime
        import re
        import time
        from pathlib import Path

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import statsmodels.api as sm
        from tqdm.auto import tqdm
        from joblib import Parallel, delayed

        from linearmodels.panel import PanelOLS, RandomEffects
        """
    ),
    markdown_cell("Add each fund sum asset col. calculate weight changes as deviation from revaluation /inflow-outflow change"),
    markdown_cell(
        """
        **Current version note**

        This notebook is tied to the rebuilt local `research_v2` dataset.

        News sentiment input is the rebuilt `News_panel_LLM_parts100.csv`, assembled from:
        - article-level LLM sentiment labels in `LLM/outputs/parts(100)`
        - full news metadata and dates from `all_news_merged_unified.csv`

        This means the regression blocks below are now using the updated sentiment markup and updated aggregation.
        """
    ),
    markdown_cell("# Loading Data"),
    code_cell(
        """
        CWD = Path.cwd().resolve()
        root_candidates = [
            CWD,
            CWD.parent,
            CWD / "research_v2",
            CWD.parent / "research_v2",
        ]
        ROOT_DIR = None
        for candidate in root_candidates:
            if (candidate / "data").exists() and (candidate / "outputs").exists():
                ROOT_DIR = candidate
                break
        if ROOT_DIR is None:
            raise FileNotFoundError("Could not locate research_v2 root directory from current working directory.")
        DATA_DIR = ROOT_DIR / "data"
        OUTPUT_DIR = ROOT_DIR / "outputs" / "regression_tables"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        Y_PATH = DATA_DIR / "funds" / "Y_interpolated_updated.csv"
        X_PATH = DATA_DIR / "news" / "News_panel_LLM_parts100.csv"
        CONTROLS_PATH = DATA_DIR / "controls" / "controls.xlsx"
        METRICS_PATH = DATA_DIR / "market" / "portfolio_metrics_updated.csv"
        FUND_PRICES_PATH = DATA_DIR / "reference" / "funds_01012015_17052025.csv"
        PROCESSED_ALL_PATH = DATA_DIR / "reference" / "processed_all.xlsx"

        CACHE_DIR = Path.home() / "Library" / "Caches" / "dissertation_research"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        def materialize_local_cache(src_path, cache_name, chunk_size_mb=16, max_retries=3):
            src_path = Path(src_path)
            if not src_path.exists():
                raise FileNotFoundError(
                    f"Required local file is missing: {src_path}. "
                    "Place Y_interpolated_updated.csv into data/funds/ before running the notebook."
                )
            if src_path.stat().st_size == 0:
                raise ValueError(
                    f"Local file exists but is empty: {src_path}. "
                    "Replace it with a complete Y_interpolated_updated.csv file."
                )
            cache_path = CACHE_DIR / cache_name
            if cache_path.exists():
                if cache_path.stat().st_size > 0:
                    print(f"Using cached file: {cache_path}")
                    return cache_path
                cache_path.unlink()

            tmp_path = cache_path.with_suffix(cache_path.suffix + ".partial")
            for attempt in range(1, max_retries + 1):
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                    print(f"Caching {src_path.name} locally, attempt {attempt}/{max_retries} ...")
                    with src_path.open("rb") as src, tmp_path.open("wb") as dst:
                        while True:
                            chunk = src.read(chunk_size_mb * 1024 * 1024)
                            if not chunk:
                                break
                            dst.write(chunk)
                    if tmp_path.stat().st_size == 0:
                        raise ValueError(f"Cached file is empty after copy: {src_path}")
                    tmp_path.replace(cache_path)
                    print(f"Local cache ready: {cache_path}")
                    return cache_path
                except Exception as e:
                    print(f"Cache attempt {attempt} failed: {e}")
                    time.sleep(min(5 * attempt, 15))

            raise TimeoutError(
                f"Could not materialize local cache for {src_path}. "
                "Google Drive likely did not provide the file in time."
            )
        """
    ),
    code_cell(
        """
        Y_LOCAL_PATH = materialize_local_cache(Y_PATH, "Y_interpolated_updated.csv")
        Y_funds_df = pd.read_csv(Y_LOCAL_PATH, encoding="utf-16", sep="\\t", low_memory=False)
        Y_funds_df["date"] = pd.to_datetime(Y_funds_df["date"])
        Y_funds_df = Y_funds_df[~Y_funds_df["Company"].astype(str).str.contains("рбк", case=False, na=False)].copy()
        Y_funds_df
        """
    ),
    code_cell(
        """
        X_news_df = pd.read_csv(X_PATH, low_memory=False)
        X_news_df["end_date"] = pd.to_datetime(X_news_df["end_date"])
        if "Company" in X_news_df.columns:
            X_news_df = X_news_df[~X_news_df["Company"].astype(str).str.contains("рбк", case=False, na=False)].copy()
        print("News panel source:", X_PATH)
        print("News panel shape:", X_news_df.shape)
        print("News panel date range:", X_news_df["end_date"].min(), "->", X_news_df["end_date"].max())
        print("News panel months:", X_news_df["end_date"].nunique())
        if "firm_in_text" in X_news_df.columns:
            print("Unique firms in news panel:", X_news_df["firm_in_text"].nunique())
        X_news_df.head()
        """
    ),
    markdown_cell(
        """
        The current `X_news_df` is the rebuilt panel used for the rerun after the markup refresh.
        If coefficients differ from older conference materials, this rebuilt input is one of the main reasons.
        """
    ),
    code_cell(
        """
        fund_assets_wide = pd.read_excel(
            PROCESSED_ALL_PATH,
            usecols=["Название фонда", "Название УК", "date", "Номер регистрации", "Тип фонда", "Категория фонда"],
        )
        fund_assets_wide["fund_key"] = fund_assets_wide["Название фонда"] + "__" + fund_assets_wide["Номер регистрации"].astype(str)
        fund_assets_wide["date"] = pd.to_datetime(fund_assets_wide["date"])
        fund_classes = fund_assets_wide[["Название фонда", "Название УК", "Номер регистрации", "Тип фонда", "Категория фонда", "fund_key"]].drop_duplicates(subset="fund_key")
        fund_classes.head()
        """
    ),
    markdown_cell("# Mapping X to Y - Creating Panel"),
    code_cell(
        """
        overlap_cols = [col for col in X_news_df.columns if col in Y_funds_df.columns and col not in ["end_date", "firm_in_text"]]
        Panel = pd.merge(
            Y_funds_df,
            X_news_df.drop(columns=overlap_cols),
            how="inner",
            left_on=["date", "Company"],
            right_on=["end_date", "firm_in_text"],
        )
        Panel = Panel.drop_duplicates().dropna(subset=["date", "fund_key", "Company"])
        Panel
        """
    ),
    code_cell(
        """
        Panel = Panel[~Panel["Название фонда"].astype(str).str.contains("индекс", case=False, na=False)].copy()
        total_assets_on_date = Panel.groupby(["date"])["Сумма, руб."].sum().reset_index()
        total_assets_on_date.rename(columns={"Сумма, руб.": "total_assets"}, inplace=True)
        if "total_assets" in Panel.columns:
            Panel = Panel.drop(columns=["total_assets"])
        Panel = pd.merge(Panel, total_assets_on_date, on="date", how="left")
        Panel["share_in_total_assets, %"] = Panel["Сумма, руб."] / Panel["total_assets"] * 100

        sent_denom = Panel["Positive"] + Panel["Negative"] + Panel["Neutral"]
        Panel["Net_sent_score_partial"] = np.where(sent_denom > 0, (Panel["Positive"] - Panel["Negative"]) / sent_denom, 0)
        Panel["Net_sent_score_weighted"] = Panel["Net_sent_score_partial"] * Panel["Доля, %"]
        Panel["Positive_weighted"] = Panel["Positive"] * Panel["Доля, %"]
        Panel["Negative_weighted"] = Panel["Negative"] * Panel["Доля, %"]
        Panel["Neutral_weighted"] = Panel["Neutral"] * Panel["Доля, %"]
        Panel.head()
        """
    ),
    markdown_cell("## Mapping controls"),
    code_cell(
        """
        unique_funds_by_date = Panel.groupby("date")["fund_key"].nunique().reset_index()
        unique_funds_by_date.rename(columns={"fund_key": "unique_funds"}, inplace=True)

        controls = pd.read_excel(CONTROLS_PATH).reset_index(drop=True)
        controls["date"] = pd.to_datetime(controls["date"])
        controls = controls.groupby(by="date", as_index=False).agg({
            "Flow": "sum",
            "Imoex": "mean",
            "Ruonia": "mean",
            "USD_RUB": "mean",
            "RVI": "mean",
        })
        controls["Flow"] = controls["Flow"] / 1000000000
        controls = controls.sort_values("date").reset_index(drop=True)
        controls["imoex_ret"] = controls["Imoex"].div(controls["Imoex"].shift(1)) - 1
        controls["imoex_log_ret"] = np.log1p(controls["imoex_ret"])
        controls = controls.merge(total_assets_on_date, on="date", how="left")
        controls = controls.merge(unique_funds_by_date, on="date", how="left")
        controls["dummy_mar2014"] = (pd.to_datetime(controls["date"]) >= pd.Timestamp("2014-03-01")).astype(int)
        controls["dummy_feb2022"] = (pd.to_datetime(controls["date"]) >= pd.Timestamp("2022-02-01")).astype(int)
        controls["dummy_sep2018"] = (pd.to_datetime(controls["date"]) >= pd.Timestamp("2018-09-01")).astype(int)
        controls.tail()
        """
    ),
    code_cell(
        """
        Panel = pd.merge(Panel, controls, how="left", on="date")
        Panel.drop_duplicates(inplace=True)
        Panel.sort_values(by=["fund_key", "firm_in_text", "ISIN конечного актива", "date"], inplace=True)
        Panel["RVI"].fillna(0, inplace=True)
        Panel
        """
    ),
    markdown_cell("# Aggregations"),
    markdown_cell("## Aggregation - f, t"),
    code_cell(
        """
        Panel_agg_ft = Panel.groupby(by=["fund_key", "date"], as_index=False).agg({
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
        })
        Panel_agg_ft
        """
    ),
    code_cell(
        """
        topic_cols = [
            "forecast_Negative", "forecast_Neutral", "forecast_Positive",
            "earnings_Negative", "earnings_Neutral", "earnings_Positive",
            "sanctions_Negative", "sanctions_Neutral", "sanctions_Positive",
            "other_Negative", "other_Neutral", "other_Positive",
        ]
        available_topic_cols = [c for c in topic_cols if c in Panel.columns]
        print("Available H5 topic columns:", available_topic_cols)
        """
    ),
    code_cell(
        """
        panel_df = pd.read_csv(METRICS_PATH, low_memory=False)
        panel_df["date"] = pd.to_datetime(panel_df["date"])
        panel_df
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
            df = pd.DataFrame({
                "coef": results.params,
                "std_err": results.std_errors,
                "t_stat": results.tstats,
                "p_value": results.pvalues,
            })
            for col in ["coef", "std_err", "t_stat", "p_value"]:
                df[col] = df[col].apply(first_two_nonzero_round)

            def stars(p):
                if p < 0.01:
                    return "***"
                elif p < 0.05:
                    return "**"
                elif p < 0.10:
                    return "*"
                return ""

            df["signif"] = df["p_value"].apply(stars)
            df["coef_sig"] = df["coef"].astype(str) + df["signif"]
            return df

        def build_term_order(main_vars, controls, lag=1, include_const=True):
            lagged = [f"{v}_L{lag}" for v in main_vars]
            base = ["const"] if include_const else []
            return {
                "current": base + main_vars + controls,
                "lagged": base + lagged + controls,
                "both": base + main_vars + lagged + controls,
            }

        def winsorize_series(s, lower=0.01, upper=0.99):
            s = pd.to_numeric(s, errors="coerce")
            lo = s.quantile(lower)
            hi = s.quantile(upper)
            return s.clip(lower=lo, upper=hi)

        def panel_results_to_df(models, term_order=None):
            rows = []
            for spec_name, res in models.items():
                summary = {
                    "R2_within": res.rsquared_within,
                    "R2_between": res.rsquared_between,
                    "R2_overall": res.rsquared_overall,
                    "N_obs": int(res.nobs),
                }
                for term, val in summary.items():
                    rows.append({
                        "spec": spec_name,
                        "term": term,
                        "coef": val,
                        "std_err": None,
                        "t_stat": None,
                        "p_value": None,
                        "signif": "",
                        "coef_sig": f"{val:.4f}",
                    })
                coef_df = results_to_df_two_nonzero(res).reset_index().rename(columns={"index": "term"})
                if term_order and spec_name in term_order:
                    desired_terms = term_order[spec_name]
                    coef_df = coef_df.set_index("term").reindex(desired_terms).reset_index()
                for _, r in coef_df.iterrows():
                    coef_sig = r["coef_sig"] if pd.notna(r.get("coef_sig")) else ""
                    rows.append({
                        "spec": spec_name,
                        "term": r["term"],
                        "coef": r["coef"],
                        "std_err": r["std_err"],
                        "t_stat": r["t_stat"],
                        "p_value": r["p_value"],
                        "signif": r["signif"] if pd.notna(r.get("signif")) else "",
                        "coef_sig": coef_sig,
                    })
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
            winsorize_cols=None,
            winsor_lower=0.01,
            winsor_upper=0.99,
        ):
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index([entity_col, "date"]).sort_index()
            lagged = []
            for v in main_vars:
                lv = f"{v}_L{lag}"
                df[lv] = df.groupby(level=entity_col)[v].shift(lag)
                lagged.append(lv)
            if winsorize_cols:
                win_targets = [c for c in winsorize_cols if c in df.columns]
                for col in win_targets:
                    df[col] = winsorize_series(df[col], lower=winsor_lower, upper=winsor_upper)

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
        """
    ),
    markdown_cell("## H0"),
    code_cell(
        """
        pif_prices = pd.read_csv(FUND_PRICES_PATH, sep=";", encoding="cp1251", low_memory=False).dropna(subset=["Фонд"])
        pif_prices["Дата"] = pd.to_datetime(pif_prices["Дата"])
        pif_prices.sort_values(["Фонд", "Дата"], ascending=[False, True], inplace=True)
        pif_prices["fund_ret"] = pif_prices.groupby("Фонд")["Пай"].pct_change()

        Panel_etf = pd.merge(Panel, pif_prices[["Дата", "Фонд", "Пай", "СЧА"]], left_on=["date", "Название фонда"], right_on=["Дата", "Фонд"], how="left")
        Panel_etf.sort_values(["Номер регистрации", "date"], ascending=[False, True], inplace=True)
        Panel_etf = pd.merge(Panel_etf, fund_classes[["Тип фонда", "Категория фонда", "fund_key"]], on=["fund_key"], how="left")
        Panel_etf["net_sent_score"] = np.where(
            (Panel_etf["Positive"] + Panel_etf["Negative"] + Panel_etf["Neutral"]) > 0,
            (Panel_etf["Positive"] - Panel_etf["Negative"]) / (Panel_etf["Positive"] + Panel_etf["Negative"] + Panel_etf["Neutral"]),
            0,
        )

        agg = Panel_etf.groupby(["fund_key", "date"], as_index=False)["Пай"].mean()
        agg["fund_ret"] = agg.groupby("fund_key")["Пай"].pct_change()
        if "fund_ret" in Panel_etf.columns:
            Panel_etf = Panel_etf.drop(columns=["fund_ret"])
        Panel_etf = Panel_etf.merge(agg[["fund_key", "date", "fund_ret"]], on=["fund_key", "date"], how="left")

        Panel_etf["excess_return"] = np.where(
            Panel_etf["Категория фонда"].isin(["Акции", "Смешанный"]),
            Panel_etf["fund_ret"] - Panel_etf["imoex_ret"],
            np.nan,
        )

        agg_dict = {
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
            "imoex_log_ret": "first",
            "imoex_ret": "first",
            "Ruonia": "first",
            "USD_RUB": "first",
            "unique_funds": "first",
            "dummy_mar2014": "first",
            "dummy_feb2022": "first",
            "fund_ret": "first",
            "excess_return": "first",
        }
        for c in available_topic_cols:
            agg_dict[c] = "sum"

        Panel_agg_ftc = Panel_etf.groupby(["fund_key", "date"], as_index=False).agg(agg_dict)
        Panel_agg_ftc["net_sent_score"] = np.where(
            (Panel_agg_ftc["Positive"] + Panel_agg_ftc["Negative"] + Panel_agg_ftc["Neutral"]) > 0,
            (Panel_agg_ftc["Positive"] - Panel_agg_ftc["Negative"]) / (Panel_agg_ftc["Positive"] + Panel_agg_ftc["Negative"] + Panel_agg_ftc["Neutral"]),
            0,
        )
        Panel_agg_ftc["fund_size"] = Panel_agg_ftc.groupby(["fund_key", "date"])["Сумма, руб."].transform("sum")
        Panel_H0 = Panel_agg_ftc.copy()
        Panel_H0_raw = Panel_H0.copy()
        Panel_H0 = Panel_H0[(Panel_H0["excess_return"] >= Panel_H0["excess_return"].quantile(0.05)) & (Panel_H0["excess_return"] <= Panel_H0["excess_return"].quantile(0.95))]
        Panel_H0 = Panel_H0[(Panel_H0["fund_ret"] >= Panel_H0["fund_ret"].quantile(0.05)) & (Panel_H0["fund_ret"] <= Panel_H0["fund_ret"].quantile(0.95))]
        Panel_H0["excess_return_wins"] = Panel_H0["excess_return"].clip(lower=Panel_H0["excess_return"].quantile(0.01), upper=Panel_H0["excess_return"].quantile(0.99))
        Panel_H0["fund_ret_wins"] = Panel_H0["fund_ret"].clip(lower=Panel_H0["fund_ret"].quantile(0.01), upper=Panel_H0["fund_ret"].quantile(0.99))
        Panel_H0["log_excess_return"] = Panel_H0["excess_return_wins"].apply(lambda x: 0 if pd.isna(x) or x == 0 else np.log(x + 1))
        Panel_H0["log_fund_ret"] = Panel_H0["fund_ret_wins"].apply(lambda x: 0 if pd.isna(x) or x == 0 else np.log(x + 1))
        Panel_H0[["date", "fund_key", "fund_ret", "excess_return"]].head()
        """
    ),
    code_cell(
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        h0_dep = Panel_H0_raw["excess_return"].replace([np.inf, -np.inf], np.nan).dropna()
        h0_indep = Panel_H0_raw["net_sent_score"].replace([np.inf, -np.inf], np.nan).dropna()

        axes[0].hist(h0_dep, bins=40, color="#4e79a7", edgecolor="white", alpha=0.9)
        axes[0].set_title("H0 data series: excess_return")
        axes[0].set_xlabel("excess_return")
        axes[0].set_ylabel("Frequency")

        axes[1].hist(h0_indep, bins=40, color="#e15759", edgecolor="white", alpha=0.9)
        axes[1].set_title("H0 data series: net_sent_score")
        axes[1].set_xlabel("net_sent_score")
        axes[1].set_ylabel("Frequency")

        plt.tight_layout()
        plt.show()
        """
    ),
    code_cell(
        """
        models = run_panel_models(
            df=Panel_H0,
            entity_col="fund_key",
            y_var="log_excess_return",
            main_vars=["net_sent_score"],
            controls=["Flow", "Imoex", "imoex_ret", "Ruonia", "USD_RUB", "unique_funds", "dummy_mar2014", "dummy_feb2022"],
            lag=1,
            entity_effects=True,
            time_effects=False,
            winsorize_cols=["log_excess_return", "net_sent_score", "net_sent_score_L1"],
        )
        print(models)
        h0_s_df = panel_results_to_df(
            models,
            term_order=build_term_order(
                main_vars=["net_sent_score"],
                controls=["Flow", "Imoex", "imoex_ret", "Ruonia", "USD_RUB", "unique_funds", "dummy_mar2014", "dummy_feb2022"],
                lag=1,
            ),
        )
        h0_s_df.to_excel(OUTPUT_DIR / "0.excess.s.xlsx", index=False)
        h0_s_df
        """
    ),
    code_cell(
        """
        models = run_panel_models(
            df=Panel_H0,
            entity_col="fund_key",
            y_var="excess_return",
            main_vars=["Net_sent_score_weighted"],
            controls=["Flow", "Imoex", "imoex_ret", "Ruonia", "USD_RUB", "unique_funds", "dummy_mar2014", "dummy_feb2022"],
            lag=1,
            entity_effects=True,
            time_effects=False,
            winsorize_cols=["excess_return", "Net_sent_score_weighted", "Net_sent_score_weighted_L1"],
        )
        print(models)
        h0_w_df = panel_results_to_df(
            models,
            term_order=build_term_order(
                main_vars=["Net_sent_score_weighted"],
                controls=["Flow", "Imoex", "imoex_ret", "Ruonia", "USD_RUB", "unique_funds", "dummy_mar2014", "dummy_feb2022"],
                lag=1,
            ),
        )
        h0_w_df.to_excel(OUTPUT_DIR / "0.excess.w.xlsx", index=False)
        h0_w_df
        """
    ),
    markdown_cell("## H1"),
    code_cell(
        """
        Panel_H1 = pd.merge(panel_df[["date", "fund_key", "portfolio_var"]], Panel_agg_ft, on=["date", "fund_key"]).drop_duplicates(subset=["fund_key", "date"], keep="first").sort_values(by=["date", "fund_key"], ascending=[False, True])
        Panel_H1["portfolio_vol"] = np.sqrt(Panel_H1["portfolio_var"])
        Panel_H1["net_sent_score"] = np.where((Panel_H1["Positive"] + Panel_H1["Negative"] + Panel_H1["Neutral"]) > 0, (Panel_H1["Positive"] - Panel_H1["Negative"]) / (Panel_H1["Positive"] + Panel_H1["Negative"] + Panel_H1["Neutral"]), 0)
        Panel_H1["Neg_share"] = Panel_H1["Negative"] / (Panel_H1["Positive"] + Panel_H1["Negative"] + Panel_H1["Neutral"])
        Panel_H1["Pos_share"] = Panel_H1["Positive"] / (Panel_H1["Positive"] + Panel_H1["Negative"] + Panel_H1["Neutral"])
        Panel_H1["Neg_share_weighted"] = Panel_H1["Negative_weighted"] / (Panel_H1["Positive_weighted"] + Panel_H1["Negative_weighted"] + Panel_H1["Neutral_weighted"])
        Panel_H1["Pos_share_weighted"] = Panel_H1["Positive_weighted"] / (Panel_H1["Positive_weighted"] + Panel_H1["Negative_weighted"] + Panel_H1["Neutral_weighted"])
        Panel_H1 = Panel_H1[Panel_H1["portfolio_var"] > 0]
        Panel_H1["RVI"].ffill(inplace=True)
        Panel_H1 = Panel_H1[(Panel_H1["portfolio_vol"] >= Panel_H1["portfolio_vol"].quantile(0.01)) & (Panel_H1["portfolio_vol"] <= Panel_H1["portfolio_vol"].quantile(0.99))]
        Panel_H1["portfolio_vol_wins"] = Panel_H1["portfolio_vol"].clip(lower=Panel_H1["portfolio_vol"].quantile(0.01), upper=Panel_H1["portfolio_vol"].quantile(0.99))
        Panel_H1["log_portfolio_vol"] = np.log(Panel_H1["portfolio_vol"])
        for c in ["Negative", "Neutral", "Positive", "Negative_weighted", "Neutral_weighted", "Positive_weighted"]:
            Panel_H1[f"log_{c}"] = np.where(Panel_H1[c] > 0, np.log(Panel_H1[c]), 0)
        Panel_H1["fund_size"] = Panel_H1.groupby(["fund_key", "date"])["Сумма, руб."].transform("sum")
        Panel_H1["log_portfolio_var"] = Panel_H1["log_portfolio_vol"] * 2
        Panel_H1.head()
        """
    ),
    code_cell(
        """
        models = run_panel_models(
            df=Panel_H1,
            entity_col="fund_key",
            y_var="log_portfolio_vol",
            main_vars=["Net_sent_score_weighted"],
            controls=["Flow", "Imoex", "imoex_ret", "Ruonia", "USD_RUB", "unique_funds", "dummy_mar2014", "dummy_feb2022"],
            lag=1,
            entity_effects=True,
            time_effects=False,
            winsorize_cols=["log_portfolio_vol", "Net_sent_score_weighted", "Net_sent_score_weighted_L1"],
        )
        print(models)
        h1_1_df = panel_results_to_df(
            models,
            term_order=build_term_order(
                main_vars=["Net_sent_score_weighted"],
                controls=["Flow", "Imoex", "imoex_ret", "Ruonia", "USD_RUB", "unique_funds", "dummy_mar2014", "dummy_feb2022"],
                lag=1,
            ),
        )
        h1_1_df.to_excel(OUTPUT_DIR / "1.vol.1.xlsx", index=False)
        h1_1_df
        """
    ),
    code_cell(
        """
        models = run_panel_models(
            df=Panel_H1,
            entity_col="fund_key",
            y_var="log_portfolio_vol",
            main_vars=["log_Negative_weighted", "log_Neutral_weighted", "log_Positive_weighted"],
            controls=["Flow", "Imoex", "imoex_ret", "Ruonia", "USD_RUB", "unique_funds", "dummy_mar2014", "dummy_feb2022"],
            lag=1,
            entity_effects=True,
            time_effects=False,
            winsorize_cols=["log_portfolio_vol", "log_Negative_weighted", "log_Neutral_weighted", "log_Positive_weighted", "log_Negative_weighted_L1", "log_Neutral_weighted_L1", "log_Positive_weighted_L1"],
        )
        print(models)
        h1_3_df = panel_results_to_df(
            models,
            term_order=build_term_order(
                main_vars=["log_Negative_weighted", "log_Neutral_weighted", "log_Positive_weighted"],
                controls=["Flow", "Imoex", "imoex_ret", "Ruonia", "USD_RUB", "unique_funds", "dummy_mar2014", "dummy_feb2022"],
                lag=1,
            ),
        )
        h1_3_df.to_excel(OUTPUT_DIR / "1.vol.3.xlsx", index=False)
        h1_3_df
        """
    ),
    markdown_cell("## H2"),
    code_cell(
        """
        Panel_H2 = pd.merge(panel_df[["date", "fund_key", "fund_liquidity"]], Panel_agg_ft, on=["date", "fund_key"]).drop_duplicates(subset=["fund_key", "date"], keep="first").sort_values(by=["date", "fund_key"], ascending=[False, True])
        Panel_H2["net_sent_score"] = np.where((Panel_H2["Positive"] + Panel_H2["Negative"] + Panel_H2["Neutral"]) > 0, (Panel_H2["Positive"] - Panel_H2["Negative"]) / (Panel_H2["Positive"] + Panel_H2["Negative"] + Panel_H2["Neutral"]), 0)
        Panel_H2["Neg_share"] = Panel_H2["Negative"] / (Panel_H2["Positive"] + Panel_H2["Negative"] + Panel_H2["Neutral"])
        Panel_H2["Pos_share"] = Panel_H2["Positive"] / (Panel_H2["Positive"] + Panel_H2["Negative"] + Panel_H2["Neutral"])
        Panel_H2["RVI"].fillna(0, inplace=True)
        Panel_H2["Neg_share_weighted"] = Panel_H2["Negative_weighted"] / (Panel_H2["Positive_weighted"] + Panel_H2["Negative_weighted"] + Panel_H2["Neutral_weighted"])
        Panel_H2["Pos_share_weighted"] = Panel_H2["Positive_weighted"] / (Panel_H2["Positive_weighted"] + Panel_H2["Negative_weighted"] + Panel_H2["Neutral_weighted"])
        Panel_H2 = pd.merge(Panel_H2, fund_classes[["Тип фонда", "Категория фонда", "fund_key"]], on=["fund_key"], how="left")
        Panel_H2 = Panel_H2[Panel_H2["Категория фонда"].isin(["Акции", "Смешанный"])]
        Panel_H2 = Panel_H2[(Panel_H2["fund_liquidity"] >= Panel_H2["fund_liquidity"].quantile(0.01)) & (Panel_H2["fund_liquidity"] <= Panel_H2["fund_liquidity"].quantile(0.99))]
        Panel_H2["fund_liquidity_wins"] = Panel_H2["fund_liquidity"].clip(lower=Panel_H2["fund_liquidity"].quantile(0.01), upper=Panel_H2["fund_liquidity"].quantile(0.99))
        for c in ["Negative", "Neutral", "Positive", "Negative_weighted", "Neutral_weighted", "Positive_weighted"]:
            Panel_H2[f"log_{c}"] = np.where(Panel_H2[c] > 0, np.log(Panel_H2[c]), 0)
        Panel_H2["log_fund_liquidity"] = np.where(Panel_H2["fund_liquidity"] > 0, np.log(Panel_H2["fund_liquidity"]), 0)
        Panel_H2 = Panel_H2[Panel_H2["log_fund_liquidity"] != 0]
        Panel_H2["fund_size"] = Panel_H2.groupby(["fund_key", "date"])["Сумма, руб."].transform("sum")
        Panel_H2.head()
        """
    ),
    code_cell(
        """
        models = run_panel_models(
            df=Panel_H2,
            entity_col="fund_key",
            y_var="log_fund_liquidity",
            main_vars=["Net_sent_score_weighted"],
            controls=["Flow", "Imoex", "imoex_ret", "Ruonia", "USD_RUB", "unique_funds", "dummy_mar2014", "dummy_feb2022"],
            lag=1,
            entity_effects=True,
            time_effects=False,
            winsorize_cols=["log_fund_liquidity", "Net_sent_score_weighted", "Net_sent_score_weighted_L1"],
        )
        print(models)
        h2_1_df = panel_results_to_df(
            models,
            term_order=build_term_order(
                main_vars=["Net_sent_score_weighted"],
                controls=["Flow", "Imoex", "imoex_ret", "Ruonia", "USD_RUB", "unique_funds", "dummy_mar2014", "dummy_feb2022"],
                lag=1,
            ),
        )
        h2_1_df.to_excel(OUTPUT_DIR / "2.liq.1.xlsx", index=False)
        h2_1_df
        """
    ),
    code_cell(
        """
        models = run_panel_models(
            df=Panel_H2,
            entity_col="fund_key",
            y_var="log_fund_liquidity",
            main_vars=["log_Negative_weighted", "log_Neutral_weighted", "log_Positive_weighted"],
            controls=["Flow", "Imoex", "imoex_ret", "Ruonia", "USD_RUB", "unique_funds", "dummy_mar2014", "dummy_feb2022"],
            lag=1,
            entity_effects=True,
            time_effects=False,
            winsorize_cols=["log_fund_liquidity", "log_Negative_weighted", "log_Neutral_weighted", "log_Positive_weighted", "log_Negative_weighted_L1", "log_Neutral_weighted_L1", "log_Positive_weighted_L1"],
        )
        print(models)
        h2_3_df = panel_results_to_df(
            models,
            term_order=build_term_order(
                main_vars=["log_Negative_weighted", "log_Neutral_weighted", "log_Positive_weighted"],
                controls=["Flow", "Imoex", "imoex_ret", "Ruonia", "USD_RUB", "unique_funds", "dummy_mar2014", "dummy_feb2022"],
                lag=1,
            ),
        )
        h2_3_df.to_excel(OUTPUT_DIR / "2.liq.3.xlsx", index=False)
        h2_3_df
        """
    ),
    markdown_cell("## H3"),
    code_cell(
        """
        Panel_H3_1 = pd.merge(panel_df[["date", "fund_key", "portfolio_var"]], Panel_agg_ft, on=["date", "fund_key"]).drop_duplicates(subset=["fund_key", "date"], keep="first").sort_values(by=["date", "fund_key"], ascending=[False, True])
        Panel_H3_1 = pd.merge(Panel_H3_1, fund_assets_wide.groupby("fund_key").agg({"Категория фонда": "first"}), on="fund_key", how="left")
        Panel_H3_1["Категория фонда"] = Panel_H3_1["Категория фонда"].apply(lambda x: str(x).lower())
        Panel_H3_1["is_stocks_fund"] = Panel_H3_1["Категория фонда"].apply(lambda x: int("акции" in x))
        Panel_H3_1["is_bonds_fund"] = Panel_H3_1["Категория фонда"].apply(lambda x: int("облигации" in x))
        Panel_H3_1["is_comb_fund"] = Panel_H3_1["Категория фонда"].apply(lambda x: int("смешанный" in x))
        Panel_H3_1["portfolio_vol"] = np.sqrt(Panel_H3_1["portfolio_var"])
        Panel_H3_1 = Panel_H3_1[(Panel_H3_1["portfolio_vol"] >= Panel_H3_1["portfolio_vol"].quantile(0.01)) & (Panel_H3_1["portfolio_vol"] <= Panel_H3_1["portfolio_vol"].quantile(0.99))]
        Panel_H3_1["log_portfolio_vol"] = np.where(Panel_H3_1["portfolio_vol"] > 0, np.log(Panel_H3_1["portfolio_vol"]), 0)
        Panel_H3_1 = Panel_H3_1[Panel_H3_1["log_portfolio_vol"] != 0]
        Panel_H3_1["fund_size"] = Panel_H3_1.groupby(["fund_key", "date"])["Сумма, руб."].transform("sum")
        Panel_H3_1["equity_x_netsent"] = Panel_H3_1["is_stocks_fund"] * Panel_H3_1["Net_sent_score_weighted"]
        Panel_H3_1["bond_x_netsent"] = Panel_H3_1["is_bonds_fund"] * Panel_H3_1["Net_sent_score_weighted"]
        Panel_H3_1.head()
        """
    ),
    code_cell(
        """
        models = run_panel_models(
            df=Panel_H3_1,
            entity_col="fund_key",
            y_var="log_portfolio_vol",
            main_vars=["Net_sent_score_weighted", "equity_x_netsent", "bond_x_netsent"],
            controls=["Flow", "Imoex", "imoex_ret", "Ruonia", "USD_RUB", "unique_funds", "dummy_mar2014", "dummy_feb2022"],
            lag=1,
            entity_effects=False,
            time_effects=False,
            winsorize_cols=["log_portfolio_vol", "Net_sent_score_weighted", "equity_x_netsent", "bond_x_netsent", "Net_sent_score_weighted_L1", "equity_x_netsent_L1", "bond_x_netsent_L1"],
        )
        print(models)
        h3_vol_df = panel_results_to_df(
            models,
            term_order=build_term_order(
                main_vars=["Net_sent_score_weighted", "equity_x_netsent", "bond_x_netsent"],
                controls=["Flow", "Imoex", "imoex_ret", "Ruonia", "USD_RUB", "unique_funds", "dummy_mar2014", "dummy_feb2022"],
                lag=1,
            ),
        )
        h3_vol_df.to_excel(OUTPUT_DIR / "3.vol.1_vs_bonds.xlsx", index=False)
        h3_vol_df
        """
    ),
    code_cell(
        """
        Panel_H3_2 = pd.merge(panel_df[["date", "fund_key", "fund_liquidity"]], Panel_agg_ft, on=["date", "fund_key"]).drop_duplicates(subset=["fund_key", "date"], keep="first").sort_values(by=["date", "fund_key"], ascending=[False, True])
        Panel_H3_2 = pd.merge(Panel_H3_2, fund_assets_wide.groupby("fund_key").agg({"Категория фонда": "first"}), on="fund_key", how="left")
        Panel_H3_2["Категория фонда"] = Panel_H3_2["Категория фонда"].apply(lambda x: str(x).lower())
        Panel_H3_2["is_stocks_fund"] = Panel_H3_2["Категория фонда"].apply(lambda x: int("акции" in x))
        Panel_H3_2["is_bonds_fund"] = Panel_H3_2["Категория фонда"].apply(lambda x: int("облигации" in x))
        Panel_H3_2 = Panel_H3_2[(Panel_H3_2["fund_liquidity"] >= Panel_H3_2["fund_liquidity"].quantile(0.01)) & (Panel_H3_2["fund_liquidity"] <= Panel_H3_2["fund_liquidity"].quantile(0.99))]
        Panel_H3_2["log_fund_liquidity"] = np.where(Panel_H3_2["fund_liquidity"] > 0, np.log(Panel_H3_2["fund_liquidity"]), 0)
        Panel_H3_2 = Panel_H3_2[Panel_H3_2["log_fund_liquidity"] != 0]
        Panel_H3_2["equity_x_netsent"] = Panel_H3_2["is_stocks_fund"] * Panel_H3_2["Net_sent_score_weighted"]
        Panel_H3_2["bond_x_netsent"] = Panel_H3_2["is_bonds_fund"] * Panel_H3_2["Net_sent_score_weighted"]
        Panel_H3_2.head()
        """
    ),
    code_cell(
        """
        models = run_panel_models(
            df=Panel_H3_2,
            entity_col="fund_key",
            y_var="log_fund_liquidity",
            main_vars=["Net_sent_score_weighted", "equity_x_netsent", "bond_x_netsent"],
            controls=["Flow", "Imoex", "imoex_ret", "Ruonia", "USD_RUB", "unique_funds", "dummy_mar2014", "dummy_feb2022"],
            lag=1,
            entity_effects=False,
            time_effects=False,
            winsorize_cols=["log_fund_liquidity", "Net_sent_score_weighted", "equity_x_netsent", "bond_x_netsent", "Net_sent_score_weighted_L1", "equity_x_netsent_L1", "bond_x_netsent_L1"],
        )
        print(models)
        h3_liq_df = panel_results_to_df(
            models,
            term_order=build_term_order(
                main_vars=["Net_sent_score_weighted", "equity_x_netsent", "bond_x_netsent"],
                controls=["Flow", "Imoex", "imoex_ret", "Ruonia", "USD_RUB", "unique_funds", "dummy_mar2014", "dummy_feb2022"],
                lag=1,
            ),
        )
        h3_liq_df.to_excel(OUTPUT_DIR / "3.liq.1_vs_bonds.xlsx", index=False)
        h3_liq_df
        """
    ),
    markdown_cell("## H4"),
    code_cell(
        """
        Panel_H4_1 = pd.merge(panel_df[["date", "fund_key", "portfolio_var"]], Panel_agg_ft, on=["date", "fund_key"]).drop_duplicates(subset=["fund_key", "date"], keep="first").sort_values(by=["date", "fund_key"], ascending=[False, True])
        q = 0.9
        Panel_H4_1["fund_size"] = Panel_H4_1.groupby(["fund_key", "date"])["Сумма, руб."].transform("sum")
        Panel_H4_1["is_small"] = Panel_H4_1.groupby("date")["fund_size"].transform(lambda x: (x <= x.quantile(q)).astype(int))
        Panel_H4_1["portfolio_vol"] = np.sqrt(Panel_H4_1["portfolio_var"])
        Panel_H4_1 = Panel_H4_1[(Panel_H4_1["portfolio_vol"] >= Panel_H4_1["portfolio_vol"].quantile(0.01)) & (Panel_H4_1["portfolio_vol"] <= Panel_H4_1["portfolio_vol"].quantile(0.99))]
        for c in ["Negative_weighted", "Neutral_weighted", "Positive_weighted"]:
            Panel_H4_1[f"log_{c}"] = np.where(Panel_H4_1[c] > 0, np.log(Panel_H4_1[c]), 0)
        Panel_H4_1["log_portfolio_vol"] = np.where(Panel_H4_1["portfolio_vol"] > 0, np.log(Panel_H4_1["portfolio_vol"]), 0)
        Panel_H4_1 = Panel_H4_1[Panel_H4_1["log_portfolio_vol"] != 0]
        Panel_H4_1["small_x_netsent"] = Panel_H4_1["is_small"] * Panel_H4_1["Net_sent_score_weighted"]
        Panel_H4_1.head()
        """
    ),
    code_cell(
        """
        models = run_panel_models(
            df=Panel_H4_1,
            entity_col="fund_key",
            y_var="log_portfolio_vol",
            main_vars=["Net_sent_score_weighted", "small_x_netsent"],
            controls=["Flow", "Imoex", "imoex_ret", "Ruonia", "USD_RUB", "unique_funds", "dummy_mar2014", "dummy_feb2022"],
            lag=1,
            entity_effects=False,
            time_effects=False,
            winsorize_cols=["log_portfolio_vol", "Net_sent_score_weighted", "small_x_netsent", "Net_sent_score_weighted_L1", "small_x_netsent_L1"],
        )
        print(models)
        h4_vol_df = panel_results_to_df(
            models,
            term_order=build_term_order(
                main_vars=["Net_sent_score_weighted", "small_x_netsent"],
                controls=["Flow", "Imoex", "imoex_ret", "Ruonia", "USD_RUB", "unique_funds", "dummy_mar2014", "dummy_feb2022"],
                lag=1,
            ),
        )
        h4_vol_df.to_excel(OUTPUT_DIR / "4.vol.1.xlsx", index=False)
        h4_vol_df
        """
    ),
    code_cell(
        """
        Panel_H4_2 = pd.merge(panel_df[["date", "fund_key", "fund_liquidity"]], Panel_agg_ft, on=["date", "fund_key"]).drop_duplicates(subset=["fund_key", "date"], keep="first").sort_values(by=["date", "fund_key"], ascending=[False, True])
        q = 0.8
        Panel_H4_2["fund_size"] = Panel_H4_2.groupby(["fund_key", "date"])["Сумма, руб."].transform("sum")
        Panel_H4_2["is_small"] = Panel_H4_2.groupby("date")["fund_size"].transform(lambda x: (x <= x.quantile(q)).astype(int))
        Panel_H4_2 = Panel_H4_2[(Panel_H4_2["fund_liquidity"] >= Panel_H4_2["fund_liquidity"].quantile(0.01)) & (Panel_H4_2["fund_liquidity"] <= Panel_H4_2["fund_liquidity"].quantile(0.99))]
        Panel_H4_2["log_fund_liquidity"] = np.where(Panel_H4_2["fund_liquidity"] > 0, np.log(Panel_H4_2["fund_liquidity"]), 0)
        Panel_H4_2 = Panel_H4_2[Panel_H4_2["log_fund_liquidity"] != 0]
        Panel_H4_2["small_x_netsent"] = Panel_H4_2["is_small"] * Panel_H4_2["Net_sent_score_weighted"]
        Panel_H4_2.head()
        """
    ),
    code_cell(
        """
        models = run_panel_models(
            df=Panel_H4_2,
            entity_col="fund_key",
            y_var="log_fund_liquidity",
            main_vars=["Net_sent_score_weighted", "small_x_netsent"],
            controls=["Flow", "Imoex", "imoex_ret", "Ruonia", "USD_RUB", "unique_funds", "dummy_mar2014", "dummy_feb2022"],
            lag=1,
            entity_effects=False,
            time_effects=False,
            winsorize_cols=["log_fund_liquidity", "Net_sent_score_weighted", "small_x_netsent", "Net_sent_score_weighted_L1", "small_x_netsent_L1"],
        )
        print(models)
        h4_liq_df = panel_results_to_df(
            models,
            term_order=build_term_order(
                main_vars=["Net_sent_score_weighted", "small_x_netsent"],
                controls=["Flow", "Imoex", "imoex_ret", "Ruonia", "USD_RUB", "unique_funds", "dummy_mar2014", "dummy_feb2022"],
                lag=1,
            ),
        )
        h4_liq_df.to_excel(OUTPUT_DIR / "4.liq.1.xlsx", index=False)
        h4_liq_df
        """
    ),
    markdown_cell("## H5"),
    markdown_cell(
        """
        `H5` depends on topic-level sentiment columns of the old format
        (`forecast_*`, `earnings_*`, `sanctions_*`, `other_*`).
        If the rebuilt news panel does not contain them, the notebook reports that explicitly and does not fabricate results.
        """
    ),
    code_cell(
        """
        required_h5_cols = set([
            "forecast_Negative", "forecast_Neutral", "forecast_Positive",
            "earnings_Negative", "earnings_Neutral", "earnings_Positive",
            "sanctions_Negative", "sanctions_Neutral", "sanctions_Positive",
            "other_Negative", "other_Neutral", "other_Positive",
        ])
        missing_h5_cols = sorted(required_h5_cols - set(Panel.columns))
        if missing_h5_cols:
            print("H5 is blocked on current updated data.")
            print("Missing topic-level sentiment columns:")
            print(missing_h5_cols)
        else:
            print("H5 topic-level columns are available.")
        """
    ),
    code_cell(
        """
        if not missing_h5_cols:
            Panel_H5 = pd.merge(panel_df[["date", "fund_key", "portfolio_var"]], Panel_agg_ftc, on=["date", "fund_key"]).drop_duplicates(subset=["fund_key", "date"], keep="first").sort_values(by=["date", "fund_key"], ascending=[False, True])
            Panel_H5["net_sent_score_reporting"] = np.where((Panel_H5["earnings_Positive"] + Panel_H5["earnings_Negative"] + Panel_H5["earnings_Neutral"]) > 0, (Panel_H5["earnings_Positive"] - Panel_H5["earnings_Negative"]) / (Panel_H5["earnings_Positive"] + Panel_H5["earnings_Negative"] + Panel_H5["earnings_Neutral"]), 0)
            Panel_H5["portfolio_vol"] = np.sqrt(Panel_H5["portfolio_var"])
            Panel_H5 = Panel_H5[(Panel_H5["portfolio_vol"] >= Panel_H5["portfolio_vol"].quantile(0.01)) & (Panel_H5["portfolio_vol"] <= Panel_H5["portfolio_vol"].quantile(0.99))]
            Panel_H5["log_portfolio_vol"] = np.where(Panel_H5["portfolio_vol"] > 0, np.log(Panel_H5["portfolio_vol"]), 0)
            Panel_H5 = Panel_H5[Panel_H5["log_portfolio_vol"] != 0]
            models = run_panel_models(
                df=Panel_H5,
                entity_col="fund_key",
                y_var="log_portfolio_vol",
                main_vars=["net_sent_score_reporting"],
                controls=["Flow", "Imoex", "imoex_ret", "Ruonia", "USD_RUB", "unique_funds", "dummy_mar2014", "dummy_feb2022"],
                lag=1,
                entity_effects=True,
                time_effects=False,
                winsorize_cols=["log_portfolio_vol", "net_sent_score_reporting", "net_sent_score_reporting_L1"],
            )
            print(models)
            h5_df = panel_results_to_df(
                models,
                term_order=build_term_order(
                    main_vars=["net_sent_score_reporting"],
                    controls=["Flow", "Imoex", "imoex_ret", "Ruonia", "USD_RUB", "unique_funds", "dummy_mar2014", "dummy_feb2022"],
                    lag=1,
                ),
            )
            h5_df.to_excel(OUTPUT_DIR / "5.vol.reporting.xlsx", index=False)
            h5_df
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


root = Path(__file__).resolve().parents[1]
out_path = root / "notebooks" / "0.Regressions_updated.ipynb"
out_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(out_path)
