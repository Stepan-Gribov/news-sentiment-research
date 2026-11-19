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
        # LLM News Sentiment: Core Research Replication

        This notebook rebuilds the core logic of the original `0.Regressions.ipynb` with the new LLM sentiment results from:

        - `./local/dev/LLM/outputs/parts(100)`

        The notebook reproduces the **core empirical pipeline**:

        - rebuild an LLM-based news panel aligned to fund-reporting dates
        - merge the panel with fund holdings and macro controls
        - run the main baseline regressions for:
          - `H0`: returns / excess returns
          - `H1`: portfolio volatility and portfolio weights
          - `H2`: fund liquidity

        Methodological note:

        - The new passed data contain article-level sentiment labels only.
        - The original topic-level decomposition used in `H5` (`earnings / forecast / sanctions / other`) is **not** reproduced here because those per-article topic labels are absent in the new input.
        - To avoid re-inventing the article-to-firm mapping, this notebook reuses the existing article-to-firm bridge from `All_news_my_LLM_mapped_processed.csv` and replaces only the sentiment with the new `parts(100)` output.
        - The merge from raw article output to the bridge is done by `link`, because `link` is unique in the master source.
        """
    ),
    code_cell(
        """
        from pathlib import Path
        import json
        import math

        import numpy as np
        import pandas as pd
        import statsmodels.api as sm
        import matplotlib.pyplot as plt
        from tqdm.auto import tqdm

        pd.set_option("display.max_columns", 200)
        pd.set_option("display.width", 200)
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
        OUTPUT_DIR = PROJECT_DIR / "llm_core_replication_outputs"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        REGR_DIR = OUTPUT_DIR / "regression_tables"
        REGR_DIR.mkdir(parents=True, exist_ok=True)

        BASE_FUNDS_DIR = Path("./local/Desktop/my_data/Master thesis/funds")
        MASTER_NEWS_CSV = Path(
            "./local/Desktop/my_data/My Drive/4. Disserartion research/calculations/data/news_links/NEWS_PAGES_PARQ/all_news_merged_unified.csv"
        )
        PARTS_DIR = Path("./local/dev/LLM/outputs/parts(100)")

        COMPANY_MAP_PATH = BASE_FUNDS_DIR / "Code/Analisys/company_type_isin1.xlsx"
        Y_PATH = BASE_FUNDS_DIR / "Data/Y/Y_interpolated.csv"
        CONTROLS_PATH = BASE_FUNDS_DIR / "Data/Controls/controls.xlsx"
        PROCESSED_ALL_PATH = BASE_FUNDS_DIR / "Data/processed_all.xlsx"
        QUOTES_PATH = BASE_FUNDS_DIR / "Data/quotes_data/All_instr_df.csv"
        FUND_PRICES_PATH = BASE_FUNDS_DIR / "Data/Y/funds_01012015_17052025.csv"
        BRIDGE_CSV_PATH = BASE_FUNDS_DIR / "Data/csv/All_news_my_LLM_mapped_processed.csv"
        PORTFOLIO_VAR_SOURCE_PATH = BASE_FUNDS_DIR / "Code/Analisys/portfolio_variances.xlsx"

        PORTFOLIO_CACHE_PATH = OUTPUT_DIR / "portfolio_metrics.pkl"
        FUND_LIQUIDITY_CACHE_PATH = OUTPUT_DIR / "fund_liquidity.pkl"
        NEWS_PANEL_CSV_PATH = OUTPUT_DIR / "News_panel_LLM_parts100.csv"
        NEWS_PANEL_XLSX_PATH = OUTPUT_DIR / "News_panel_LLM_parts100.xlsx"
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

            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=lagged + [y_var])

            specs = {
                "current": main_vars + controls,
                "lagged": lagged + controls,
                "both": main_vars + lagged + controls,
            }
            results = {}
            for name, vars_ in specs.items():
                exog = sm.add_constant(df[vars_].replace([np.inf, -np.inf], np.nan)).dropna()
                y = df.loc[exog.index, y_var]
                model = PanelOLS(
                    y,
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


        def count_mentions(text, firm, ratio=0.8):
            if pd.isna(text) or pd.isna(firm):
                return 0
            firm = str(firm).strip().lower()
            if not firm:
                return 0
            token = firm[: max(1, int(len(firm) * ratio))]
            return str(text).lower().count(token)


        def safe_net_sent(pos, neg, neu):
            total = pos + neg + neu
            if total == 0 or pd.isna(total):
                return 0.0
            return (pos - neg) / total


        def winsorize_series(series, lower=0.01, upper=0.99):
            lo = series.quantile(lower)
            hi = series.quantile(upper)
            return series.clip(lower=lo, upper=hi)


        def build_portfolio_metrics(
            panel,
            quotes_path,
            portfolio_var_source_path,
            cache_path,
            fund_liquidity_cache_path,
            lookback_days=90,
        ):
            if cache_path.exists():
                return pd.read_pickle(cache_path)

            portfolio_var_df = pd.read_excel(
                portfolio_var_source_path,
                usecols=["date", "fund_key", "portfolio_var"],
            )
            portfolio_var_df["date"] = pd.to_datetime(portfolio_var_df["date"])

            quotes = pd.read_csv(quotes_path, encoding="utf-16", sep="\\t", low_memory=False)
            quotes["ISIN"] = quotes["ISIN"].fillna(quotes["SECID"])
            quotes["TRADEDATE"] = pd.to_datetime(quotes["TRADEDATE"])

            quotes["Amihud_illiq"] = quotes["ret"].abs() / quotes["VALUE"]
            quotes.loc[quotes["VALUE"] == 0, "Amihud_illiq"] = np.nan
            illiq_index = (
                quotes.loc[:, ["ISIN", "TRADEDATE", "Amihud_illiq"]]
                .set_index(["ISIN", "TRADEDATE"])
                .sort_index()["Amihud_illiq"]
            )

            portfolios = (
                panel[["date", "fund_key", "ISIN конечного актива", "Доля, %"]]
                .dropna(subset=["ISIN конечного актива", "Доля, %"])
                .sort_values(["date", "fund_key", "Доля, %"], ascending=[True, True, False])
            )

            if fund_liquidity_cache_path.exists():
                fund_liquidity_df = pd.read_pickle(fund_liquidity_cache_path)
            else:
                rows = []
                grouped = portfolios.groupby(["date", "fund_key"], sort=True)
                for (dt, fund_key), grp in tqdm(grouped, total=grouped.ngroups, desc="Fund liquidity"):
                    uniq = grp.drop_duplicates("ISIN конечного актива")
                    weights = pd.Series(
                        uniq["Доля, %"].values,
                        index=uniq["ISIN конечного актива"].astype(str),
                        name="weights",
                    )

                    start = pd.Timestamp(dt) - pd.Timedelta(days=lookback_days - 1)
                    illiq_means = {}
                    for isin in weights.index:
                        try:
                            illiq_ser = illiq_index.loc[isin].loc[start:dt]
                        except KeyError:
                            illiq_ser = pd.Series(dtype=float)
                        if not illiq_ser.empty:
                            illiq_means[isin] = illiq_ser.mean()

                    if illiq_means:
                        w_adj_liq = weights.loc[list(illiq_means.keys())]
                        fund_liquidity = float((w_adj_liq * pd.Series(illiq_means)).sum())
                    else:
                        fund_liquidity = np.nan

                    rows.append(
                        {
                            "date": pd.Timestamp(dt),
                            "fund_key": fund_key,
                            "fund_liquidity": fund_liquidity,
                        }
                    )

                fund_liquidity_df = pd.DataFrame(rows)
                fund_liquidity_df.to_pickle(fund_liquidity_cache_path)

            metrics = portfolio_var_df.merge(
                fund_liquidity_df,
                on=["date", "fund_key"],
                how="outer",
            )
            metrics.to_pickle(cache_path)
            return metrics
        """
    ),
    code_cell(
        """
        company_map = pd.read_excel(COMPANY_MAP_PATH)

        Y_funds_df = pd.read_csv(Y_PATH, encoding="utf-16", sep="\\t", low_memory=False)
        Y_funds_df["date"] = pd.to_datetime(Y_funds_df["date"])
        Y_funds_df = Y_funds_df[~Y_funds_df["Company"].str.contains("рбк", case=False, na=False)].copy()

        fund_assets_wide = pd.read_excel(PROCESSED_ALL_PATH)
        fund_assets_wide["fund_key"] = (
            fund_assets_wide["Название фонда"] + "__" + fund_assets_wide["Номер регистрации"].astype(str)
        )
        fund_assets_wide["date"] = pd.to_datetime(fund_assets_wide["date"])
        fund_classes = fund_assets_wide[
            ["Название фонда", "Название УК", "Номер регистрации", "Тип фонда", "Категория фонда", "fund_key"]
        ].drop_duplicates(subset="fund_key")

        controls = pd.read_excel(CONTROLS_PATH)
        controls["date"] = pd.to_datetime(controls["date"])

        print("Y_funds_df:", Y_funds_df.shape, Y_funds_df["date"].min(), Y_funds_df["date"].max())
        print("controls:", controls.shape, controls["date"].min(), controls["date"].max())
        print("company_map:", company_map.shape)
        """
    ),
    code_cell(
        """
        master_links = pd.read_csv(MASTER_NEWS_CSV, usecols=["link"]).reset_index().rename(columns={"index": "id"})
        parts_df = load_parts_sentiment(PARTS_DIR)

        link_sentiment = master_links.merge(parts_df, on="id", how="left", validate="one_to_one")
        assert link_sentiment["link"].duplicated().sum() == 0, "Expected unique links in the master news source"

        print("master links:", len(master_links))
        print("classified rows:", parts_df["status"].notna().sum())
        print("missing sentiment after id-merge:", link_sentiment["sentiment_label"].isna().sum())
        print(parts_df["status"].value_counts(dropna=False))
        print(parts_df["sentiment_label"].value_counts(dropna=False))
        """
    ),
    code_cell(
        """
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

        for word in ["отчетность", "прогноз", "рынок", "санкции", "мосбиржа", "акции"]:
            prefix = word[: max(1, int(len(word) * 0.75))]
            bridge_df[word] = bridge_df["news_content"].astype(str).str.lower().str.count(prefix)

        coverage = bridge_df["status"].notna().mean()
        print("bridge rows:", len(bridge_df))
        print("link-based sentiment merge coverage:", round(float(coverage), 4))
        print(bridge_df["sentiment"].value_counts(dropna=False))
        """
    ),
    markdown_cell(
        """
        ## Build the New X Panel

        As in the original workflow, news are binned into intervals defined by fund-reporting dates.  
        Articles outside the fund sample are dropped at this stage.
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
            include_lowest=True,
            ordered=True,
        )

        outside_sample = news_df["interval"].isna().sum()
        news_df = news_df.dropna(subset=["interval"]).copy()

        count_pivot = news_df.pivot_table(
            index=["firm_in_text", "interval"],
            columns="sentiment",
            aggfunc="size",
            fill_value=0,
        )

        sum_pivot = news_df.pivot_table(
            index=["firm_in_text", "interval"],
            values=["n_matches", "акции", "мосбиржа", "отчетность", "прогноз", "рынок", "санкции"],
            aggfunc="sum",
            fill_value=0,
        )

        news_panel_llm = (
            count_pivot.reset_index()
            .merge(sum_pivot.reset_index(), on=["firm_in_text", "interval"], how="outer")
            .fillna(0)
        )

        for col in ["Negative", "Neutral", "Positive", "Н/Д"]:
            if col not in news_panel_llm.columns:
                news_panel_llm[col] = 0

        news_panel_llm.insert(
            2,
            "start_date",
            pd.to_datetime(news_panel_llm["interval"].astype(str).str.split(":").str[1]),
        )
        news_panel_llm.insert(
            3,
            "end_date",
            pd.to_datetime(news_panel_llm["interval"].astype(str).str.split(":").str[0]),
        )

        numeric_news_cols = [
            "Negative",
            "Neutral",
            "Positive",
            "Н/Д",
            "n_matches",
            "акции",
            "мосбиржа",
            "отчетность",
            "прогноз",
            "рынок",
            "санкции",
        ]
        news_panel_llm[numeric_news_cols] = news_panel_llm[numeric_news_cols].fillna(0)

        news_panel_llm = company_map.merge(
            news_panel_llm,
            left_on="Company",
            right_on="firm_in_text",
            how="right",
        ).drop_duplicates()

        news_panel_llm.to_csv(NEWS_PANEL_CSV_PATH, index=False)
        try:
            news_panel_llm.to_excel(NEWS_PANEL_XLSX_PATH, index=False)
        except Exception as exc:
            print(f"Warning: failed to write Excel copy: {exc}")

        print("Articles outside the Y sample:", outside_sample)
        print("news_panel_llm shape:", news_panel_llm.shape)
        print(news_panel_llm.head())
        """
    ),
    code_cell(
        """
        overlap_cols = [col for col in news_panel_llm.columns if col in Y_funds_df.columns and col not in ["end_date", "firm_in_text"]]

        Panel = pd.merge(
            Y_funds_df,
            news_panel_llm.drop(columns=overlap_cols),
            how="inner",
            left_on=["date", "Company"],
            right_on=["end_date", "firm_in_text"],
        ).drop_duplicates()

        Panel = Panel[~Panel["Название фонда"].str.contains("индекс", case=False, na=False)].copy()

        total_assets_on_date = Panel.groupby("date", as_index=False)["Сумма, руб."].sum().rename(columns={"Сумма, руб.": "total_assets_on_date"})
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
            {
                "Flow": "sum",
                "Imoex": "mean",
                "Ruonia": "mean",
                "USD_RUB": "mean",
                "RVI": "mean",
            }
        )
        controls_agg["Flow"] = controls_agg["Flow"] / 1_000_000_000
        controls_agg = controls_agg.sort_values("date").reset_index(drop=True)
        controls_agg["imoex_ret"] = controls_agg["Imoex"].div(controls_agg["Imoex"].shift(1)) - 1
        controls_agg["imoex_log_ret"] = np.log1p(controls_agg["imoex_ret"])
        controls_agg = controls_agg.merge(total_assets_on_date.rename(columns={"total_assets_on_date": "total_assets"}), on="date", how="left")
        controls_agg = controls_agg.merge(unique_funds_by_date, on="date", how="left")
        controls_agg["dummy_mar2014"] = (controls_agg["date"] >= pd.Timestamp("2014-03-01")).astype(int)
        controls_agg["dummy_feb2022"] = (controls_agg["date"] >= pd.Timestamp("2022-02-01")).astype(int)

        Panel = Panel.merge(controls_agg, on="date", how="left").drop_duplicates()
        Panel = Panel.sort_values(["fund_key", "firm_in_text", "ISIN конечного актива", "date"])
        Panel["RVI"] = Panel["RVI"].fillna(0)

        print("Panel shape:", Panel.shape)
        print("Dates:", Panel["date"].min(), Panel["date"].max())
        print("Unique funds:", Panel["fund_key"].nunique())
        print("Unique firms:", Panel["firm_in_text"].nunique())
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

        Panel_weights = Panel.groupby(["fund_key", "firm_in_text", "date"], as_index=False).agg(
            {
                "Класс активов": "first",
                "Сумма, руб.": "sum",
                "Доля, %": "sum",
                "Net_sent_score_weighted": "mean",
                "Positive_weighted": "sum",
                "Negative_weighted": "sum",
                "Neutral_weighted": "sum",
                "Negative": "first",
                "Neutral": "first",
                "Positive": "first",
                "Flow": "first",
                "Imoex": "first",
                "imoex_ret": "first",
                "Ruonia": "first",
                "USD_RUB": "first",
                "unique_funds": "first",
                "dummy_mar2014": "first",
                "dummy_feb2022": "first",
            }
        )

        print("Panel_agg_ft:", Panel_agg_ft.shape)
        print("Panel_weights:", Panel_weights.shape)
        """
    ),
    markdown_cell(
        """
        ## Cached Portfolio Metrics

        This is the slowest part of the replication.  
        The first run computes portfolio volatility and fund liquidity from holdings and market quotes and stores the result in a local cache.  
        All later runs reuse the cache.
        """
    ),
    code_cell(
        """
        portfolio_metrics = build_portfolio_metrics(
            panel=Panel,
            quotes_path=QUOTES_PATH,
            portfolio_var_source_path=PORTFOLIO_VAR_SOURCE_PATH,
            cache_path=PORTFOLIO_CACHE_PATH,
            fund_liquidity_cache_path=FUND_LIQUIDITY_CACHE_PATH,
            lookback_days=90,
        )

        print(portfolio_metrics.shape)
        print(portfolio_metrics.head())
        """
    ),
    markdown_cell(
        """
        ## H1: Sentiment and Portfolio Volatility
        """
    ),
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
        """
    ),
    code_cell(
        """
        h1_vol_baseline = run_panel_models(
            df=Panel_H1,
            entity_col="fund_key",
            y_var="log_portfolio_vol",
            main_vars=["Net_sent_score_weighted"],
            controls=["Flow", "Imoex", "imoex_ret", "Ruonia", "USD_RUB", "unique_funds", "dummy_mar2014", "dummy_feb2022"],
            lag=1,
            entity_effects=True,
            time_effects=False,
            cov_type="clustered",
            cluster_entity=True,
            cluster_time=True,
        )

        h1_vol_components = run_panel_models(
            df=Panel_H1,
            entity_col="fund_key",
            y_var="log_portfolio_vol",
            main_vars=["log_Negative_weighted", "log_Neutral_weighted", "log_Positive_weighted"],
            controls=["Flow", "Imoex", "imoex_ret", "Ruonia", "USD_RUB", "unique_funds", "dummy_mar2014", "dummy_feb2022"],
            lag=1,
            entity_effects=True,
            time_effects=False,
            cov_type="clustered",
            cluster_entity=True,
            cluster_time=True,
        )

        panel_results_to_df(h1_vol_baseline).to_excel(REGR_DIR / "1.vol.1_llm_parts100.xlsx", index=False)
        panel_results_to_df(h1_vol_components).to_excel(REGR_DIR / "1.vol.3_llm_parts100.xlsx", index=False)

        panel_results_to_df(h1_vol_baseline).head(20)
        """
    ),
    markdown_cell(
        """
        ## H1: Sentiment and Portfolio Weights
        """
    ),
    code_cell(
        """
        Panel_H1_weights = Panel_weights.copy()
        Panel_H1_weights = Panel_H1_weights[
            (Panel_H1_weights["Доля, %"] >= Panel_H1_weights["Доля, %"].quantile(0.01))
            & (Panel_H1_weights["Доля, %"] <= Panel_H1_weights["Доля, %"].quantile(0.99))
        ].copy()

        Panel_H1_weights["Доля, %_wins"] = winsorize_series(Panel_H1_weights["Доля, %"], lower=0.01, upper=0.98)
        Panel_H1_weights = Panel_H1_weights[Panel_H1_weights["Доля, %_wins"] > 0].copy()
        Panel_H1_weights["fund_firm_key"] = (
            Panel_H1_weights["fund_key"].astype(str) + "||" + Panel_H1_weights["firm_in_text"].astype(str)
        )

        Panel_H1_weights["net_sent_score"] = [
            safe_net_sent(p, n, u)
            for p, n, u in zip(Panel_H1_weights["Positive"], Panel_H1_weights["Negative"], Panel_H1_weights["Neutral"])
        ]

        for col in ["Negative", "Neutral", "Positive", "Negative_weighted", "Neutral_weighted", "Positive_weighted"]:
            Panel_H1_weights[f"log_{col}"] = np.where(Panel_H1_weights[col] > 0, np.log(Panel_H1_weights[col]), 0)

        Panel_H1_weights["fund_size"] = Panel_H1_weights.groupby(["fund_key", "date"])["Сумма, руб."].transform("sum")
        print(Panel_H1_weights.shape)
        """
    ),
    code_cell(
        """
        h1_weights_baseline = run_panel_models(
            df=Panel_H1_weights,
            entity_col="fund_firm_key",
            y_var="Доля, %_wins",
            main_vars=["Net_sent_score_weighted"],
            controls=["Flow", "Imoex", "imoex_ret", "Ruonia", "USD_RUB", "unique_funds", "dummy_feb2022"],
            lag=1,
            entity_effects=True,
            time_effects=False,
            cov_type="clustered",
            cluster_entity=True,
            cluster_time=True,
        )

        h1_weights_components = run_panel_models(
            df=Panel_H1_weights,
            entity_col="fund_firm_key",
            y_var="Доля, %_wins",
            main_vars=["log_Negative_weighted", "log_Positive_weighted"],
            controls=["Flow", "Imoex", "imoex_ret", "Ruonia", "USD_RUB", "unique_funds", "dummy_feb2022"],
            lag=1,
            entity_effects=True,
            time_effects=False,
            cov_type="clustered",
            cluster_entity=True,
            cluster_time=True,
        )

        panel_results_to_df(h1_weights_baseline).to_excel(REGR_DIR / "1.weights.1_llm_parts100.xlsx", index=False)
        panel_results_to_df(h1_weights_components).to_excel(REGR_DIR / "1.weights.3_llm_parts100.xlsx", index=False)

        panel_results_to_df(h1_weights_baseline).head(20)
        """
    ),
    markdown_cell(
        """
        ## H2: Sentiment and Fund Liquidity
        """
    ),
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
        """
    ),
    code_cell(
        """
        h2_liq_baseline = run_panel_models(
            df=Panel_H2,
            entity_col="fund_key",
            y_var="log_fund_liquidity",
            main_vars=["Net_sent_score_weighted"],
            controls=["Flow", "Imoex", "imoex_ret", "Ruonia", "USD_RUB", "unique_funds", "dummy_mar2014", "dummy_feb2022"],
            lag=1,
            entity_effects=True,
            time_effects=False,
            cov_type="clustered",
            cluster_entity=True,
            cluster_time=True,
        )

        h2_liq_components = run_panel_models(
            df=Panel_H2,
            entity_col="fund_key",
            y_var="log_fund_liquidity",
            main_vars=["log_Negative_weighted", "log_Neutral_weighted", "log_Positive_weighted"],
            controls=["Flow", "Imoex", "imoex_ret", "Ruonia", "USD_RUB", "unique_funds", "dummy_mar2014", "dummy_feb2022"],
            lag=1,
            entity_effects=True,
            time_effects=False,
            cov_type="clustered",
            cluster_entity=True,
            cluster_time=True,
        )

        panel_results_to_df(h2_liq_baseline).to_excel(REGR_DIR / "2.liq.1_llm_parts100.xlsx", index=False)
        panel_results_to_df(h2_liq_components).to_excel(REGR_DIR / "2.liq.3_llm_parts100.xlsx", index=False)

        panel_results_to_df(h2_liq_baseline).head(20)
        """
    ),
    markdown_cell(
        """
        ## H0: Sentiment and Fund Returns
        """
    ),
    code_cell(
        """
        pif_prices = pd.read_csv(FUND_PRICES_PATH, sep=";", encoding="cp1251", low_memory=False).dropna(subset=["Фонд"])
        pif_prices["Дата"] = pd.to_datetime(pif_prices["Дата"])
        pif_prices = pif_prices.sort_values(["Фонд", "Дата"], ascending=[False, True])

        Panel_etf = Panel.merge(
            pif_prices[["Дата", "Фонд", "Пай", "СЧА"]],
            left_on=["date", "Название фонда"],
            right_on=["Дата", "Фонд"],
            how="left",
        )
        Panel_etf = Panel_etf.sort_values(["Номер регистрации", "date"], ascending=[False, True])
        Panel_etf = Panel_etf.merge(
            fund_classes[["Тип фонда", "Категория фонда", "fund_key"]],
            on="fund_key",
            how="left",
        )

        agg_prices = Panel_etf.groupby(["fund_key", "date"], as_index=False)["Пай"].mean()
        agg_prices["fund_ret"] = agg_prices.groupby("fund_key")["Пай"].pct_change()
        Panel_etf = Panel_etf.merge(agg_prices[["fund_key", "date", "fund_ret"]], on=["fund_key", "date"], how="left")

        Panel_etf["excess_return"] = np.where(
            Panel_etf["Категория фонда"].isin(["Акции", "Смешанный"]),
            Panel_etf["fund_ret"] - Panel_etf["imoex_ret"],
            np.nan,
        )

        Panel_H0 = Panel_etf.groupby(["fund_key", "date"], as_index=False).agg(
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
        )

        Panel_H0["net_sent_score"] = [
            safe_net_sent(p, n, u)
            for p, n, u in zip(Panel_H0["Positive"], Panel_H0["Negative"], Panel_H0["Neutral"])
        ]
        Panel_H0["fund_size"] = Panel_H0.groupby(["fund_key", "date"])["Сумма, руб."].transform("sum")

        Panel_H0 = Panel_H0[
            (Panel_H0["excess_return"] >= Panel_H0["excess_return"].quantile(0.05))
            & (Panel_H0["excess_return"] <= Panel_H0["excess_return"].quantile(0.95))
        ].copy()
        Panel_H0 = Panel_H0[
            (Panel_H0["fund_ret"] >= Panel_H0["fund_ret"].quantile(0.05))
            & (Panel_H0["fund_ret"] <= Panel_H0["fund_ret"].quantile(0.95))
        ].copy()

        Panel_H0["excess_return_wins"] = winsorize_series(Panel_H0["excess_return"])
        Panel_H0["fund_ret_wins"] = winsorize_series(Panel_H0["fund_ret"])
        Panel_H0["log_excess_return"] = Panel_H0["excess_return_wins"].apply(lambda x: 0 if x == 0 else np.log1p(x))
        Panel_H0["log_fund_ret"] = Panel_H0["fund_ret_wins"].apply(lambda x: 0 if x == 0 else np.log1p(x))

        print(Panel_H0.shape)
        """
    ),
    code_cell(
        """
        h0_excess_simple = run_panel_models(
            df=Panel_H0,
            entity_col="fund_key",
            y_var="log_excess_return",
            main_vars=["net_sent_score"],
            controls=["Flow", "Imoex", "imoex_ret", "Ruonia", "USD_RUB", "unique_funds", "dummy_feb2022"],
            lag=1,
        )

        h0_excess_weighted = run_panel_models(
            df=Panel_H0,
            entity_col="fund_key",
            y_var="log_excess_return",
            main_vars=["Net_sent_score_weighted"],
            controls=["Flow", "Imoex", "imoex_ret", "Ruonia", "USD_RUB", "unique_funds", "dummy_feb2022"],
            lag=1,
        )

        h0_fund_simple = run_panel_models(
            df=Panel_H0,
            entity_col="fund_key",
            y_var="log_fund_ret",
            main_vars=["net_sent_score"],
            controls=["Flow", "Imoex", "imoex_ret", "Ruonia", "USD_RUB", "unique_funds", "dummy_feb2022"],
            lag=1,
        )

        h0_fund_weighted = run_panel_models(
            df=Panel_H0,
            entity_col="fund_key",
            y_var="log_fund_ret",
            main_vars=["Net_sent_score_weighted"],
            controls=["Flow", "Imoex", "imoex_ret", "Ruonia", "USD_RUB", "unique_funds", "dummy_feb2022"],
            lag=1,
        )

        panel_results_to_df(h0_excess_simple).to_excel(REGR_DIR / "0.excess.s_llm_parts100.xlsx", index=False)
        panel_results_to_df(h0_excess_weighted).to_excel(REGR_DIR / "0.excess.w_llm_parts100.xlsx", index=False)
        panel_results_to_df(h0_fund_simple).to_excel(REGR_DIR / "0.simple.s_llm_parts100.xlsx", index=False)
        panel_results_to_df(h0_fund_weighted).to_excel(REGR_DIR / "0.simple.w_llm_parts100.xlsx", index=False)

        panel_results_to_df(h0_excess_weighted).head(20)
        """
    ),
    markdown_cell(
        """
        ## Final Note

        This notebook intentionally stops at the **core replication** stage.

        Not reproduced from the original notebook:

        - `H5` topic-level decomposition, because the new passed sentiment files do not contain article-level topic classes
        - older exploratory cells and duplicate robustness fragments from the source notebook

        Everything above is based on the new `parts(100)` sentiment output and the existing deterministic research inputs (`Y`, controls, quotes, prices, article-to-firm bridge).
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

output_path = Path.cwd() / "LLM_News_Core_Replication.ipynb"
output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(output_path)
