import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "regressions_updated"
REGR_DIR = OUTPUT_DIR / "regression_tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REGR_DIR.mkdir(parents=True, exist_ok=True)

LEGACY_FUNDS_DIR = Path("./local/Desktop/my_data/Master thesis/funds")


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
        {"coef": results.params, "std_err": results.std_errors, "t_stat": results.tstats, "p_value": results.pvalues}
    )
    for col in ["coef", "std_err", "t_stat", "p_value"]:
        df[col] = df[col].apply(first_two_nonzero_round)
    df["signif"] = df["p_value"].map(lambda p: "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else "")))
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
            rows.append({"spec": spec_name, "term": term, "coef": val, "std_err": None, "t_stat": None, "p_value": None, "signif": "", "coef_sig": f"{val:.4f}"})
        coef_df = results_to_df_two_nonzero(res).reset_index().rename(columns={"index": "term"})
        for _, row in coef_df.iterrows():
            rows.append({"spec": spec_name, "term": row["term"], "coef": row["coef"], "std_err": row["std_err"], "t_stat": row["t_stat"], "p_value": row["p_value"], "signif": row["signif"], "coef_sig": row["coef_sig"]})
    return pd.DataFrame(rows)


def run_panel_models(df, entity_col, y_var, main_vars, controls, lag=1):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index([entity_col, "date"]).sort_index()
    lagged = []
    for var in main_vars:
        lag_var = f"{var}_L{lag}"
        df[lag_var] = df.groupby(level=0)[var].shift(lag)
        lagged.append(lag_var)
    specs = {
        "current": main_vars + controls,
        "lagged": lagged + controls,
        "both": main_vars + lagged + controls,
    }
    out = {}
    for name, vars_ in specs.items():
        work = df[[y_var] + vars_].replace([np.inf, -np.inf], np.nan).dropna()
        exog = sm.add_constant(work[vars_])
        out[name] = PanelOLS(work[y_var], exog, entity_effects=True, time_effects=False, drop_absorbed=True).fit(
            cov_type="clustered", cluster_entity=True, cluster_time=True
        )
    return out


def main():
    y_df = pd.read_csv(PROJECT_DIR / "Y_interpolated_updated.csv", sep="\t", encoding="utf-16", low_memory=False)
    y_df["date"] = pd.to_datetime(y_df["date"])
    news = pd.read_csv(PROJECT_DIR / "llm_core_replication_outputs" / "News_panel_LLM_parts100.csv", low_memory=False)
    news["end_date"] = pd.to_datetime(news["end_date"])
    metrics = pd.read_csv(PROJECT_DIR / "portfolio_metrics" / "portfolio_metrics_updated.csv", low_memory=False)
    metrics["date"] = pd.to_datetime(metrics["date"])
    controls = pd.read_excel(LEGACY_FUNDS_DIR / "Data" / "Controls" / "controls.xlsx")
    controls["date"] = pd.to_datetime(controls["date"])
    fund_assets_wide = pd.read_excel(LEGACY_FUNDS_DIR / "Data" / "processed_all.xlsx")
    fund_assets_wide["fund_key"] = fund_assets_wide["Название фонда"] + "__" + fund_assets_wide["Номер регистрации"].astype(str)
    fund_classes = fund_assets_wide[["fund_key", "Категория фонда"]].drop_duplicates("fund_key")

    overlap_cols = [c for c in news.columns if c in y_df.columns and c not in ["end_date", "firm_in_text"]]
    panel = pd.merge(y_df, news.drop(columns=overlap_cols), how="inner", left_on=["date", "Company"], right_on=["end_date", "firm_in_text"]).drop_duplicates()
    panel = panel[~panel["Название фонда"].astype(str).str.contains("индекс", case=False, na=False)].copy()
    sent_total = panel["Positive"] + panel["Negative"] + panel["Neutral"]
    panel["Net_sent_score_partial"] = np.where(sent_total > 0, (panel["Positive"] - panel["Negative"]) / sent_total, 0)
    panel["Net_sent_score_weighted"] = panel["Net_sent_score_partial"] * panel["Доля, %"]
    panel["Positive_weighted"] = panel["Positive"] * panel["Доля, %"]
    panel["Negative_weighted"] = panel["Negative"] * panel["Доля, %"]
    panel["Neutral_weighted"] = panel["Neutral"] * panel["Доля, %"]
    unique_funds_by_date = panel.groupby("date")["fund_key"].nunique().reset_index().rename(columns={"fund_key": "unique_funds"})
    controls_agg = controls.groupby("date", as_index=False).agg({"Flow": "sum", "Imoex": "mean", "Ruonia": "mean", "USD_RUB": "mean", "RVI": "mean"})
    controls_agg["Flow"] = controls_agg["Flow"] / 1_000_000_000
    controls_agg = controls_agg.sort_values("date").reset_index(drop=True)
    controls_agg["imoex_ret"] = controls_agg["Imoex"].div(controls_agg["Imoex"].shift(1)) - 1
    panel = panel.merge(unique_funds_by_date, on="date", how="left")
    panel["dummy_mar2014"] = (panel["date"] >= pd.Timestamp("2014-03-01")).astype(int)
    panel["dummy_feb2022"] = (panel["date"] >= pd.Timestamp("2022-02-01")).astype(int)
    panel = panel.merge(controls_agg, on="date", how="left").drop_duplicates()

    panel_agg = panel.groupby(["fund_key", "date"], as_index=False).agg(
        {"Сумма, руб.": "sum", "Net_sent_score_weighted": "sum", "Positive_weighted": "sum", "Negative_weighted": "sum", "Neutral_weighted": "sum", "Negative": "sum", "Neutral": "sum", "Positive": "sum", "unique_funds": "first", "dummy_mar2014": "first", "dummy_feb2022": "first", "Flow": "first", "Imoex": "first", "imoex_ret": "first", "Ruonia": "first", "USD_RUB": "first", "RVI": "first"}
    )

    full_controls = ["unique_funds", "dummy_mar2014", "dummy_feb2022"]

    h1 = metrics[["date", "fund_key", "portfolio_var"]].merge(panel_agg, on=["date", "fund_key"], how="inner").drop_duplicates(["fund_key", "date"])
    h1["portfolio_vol"] = np.sqrt(h1["portfolio_var"])
    h1 = h1[h1["portfolio_var"] > 0].copy()
    h1 = h1[(h1["portfolio_vol"] >= h1["portfolio_vol"].quantile(0.01)) & (h1["portfolio_vol"] <= h1["portfolio_vol"].quantile(0.99))].copy()
    h1["log_portfolio_vol"] = np.log(h1["portfolio_vol"])
    for col in ["Negative_weighted", "Neutral_weighted", "Positive_weighted"]:
        h1[f"log_{col}"] = np.where(h1[col] > 0, np.log(h1[col]), 0)

    h2 = metrics[["date", "fund_key", "fund_liquidity"]].merge(panel_agg, on=["date", "fund_key"], how="inner").drop_duplicates(["fund_key", "date"])
    h2 = h2.merge(fund_classes, on="fund_key", how="left")
    h2 = h2[h2["Категория фонда"].isin(["Акции", "Смешанный"])].copy()
    h2 = h2[(h2["fund_liquidity"] >= h2["fund_liquidity"].quantile(0.01)) & (h2["fund_liquidity"] <= h2["fund_liquidity"].quantile(0.99))].copy()
    h2["log_fund_liquidity"] = np.where(h2["fund_liquidity"] > 0, np.log(h2["fund_liquidity"]), 0)
    h2 = h2[h2["log_fund_liquidity"] != 0].copy()
    for col in ["Negative_weighted", "Neutral_weighted", "Positive_weighted"]:
        h2[f"log_{col}"] = np.where(h2[col] > 0, np.log(h2[col]), 0)

    print("H1 sample:", h1.shape, h1["date"].min(), h1["date"].max())
    print("H2 sample:", h2.shape, h2["date"].min(), h2["date"].max())

    print("Run H1 netsent", flush=True)
    h1_res = run_panel_models(h1, "fund_key", "log_portfolio_vol", ["Net_sent_score_weighted"], full_controls)
    print("Run H1 components", flush=True)
    h1_components = run_panel_models(
        h1,
        "fund_key",
        "log_portfolio_vol",
        ["log_Negative_weighted", "log_Neutral_weighted", "log_Positive_weighted"],
        full_controls,
    )

    print("Run H2 netsent", flush=True)
    h2_res = run_panel_models(h2, "fund_key", "log_fund_liquidity", ["Net_sent_score_weighted"], full_controls)
    print("Run H2 components", flush=True)
    h2_components = run_panel_models(
        h2,
        "fund_key",
        "log_fund_liquidity",
        ["log_Negative_weighted", "log_Neutral_weighted", "log_Positive_weighted"],
        full_controls,
    )

    h1_df = panel_results_to_df(h1_res)
    h1_comp_df = panel_results_to_df(h1_components)
    h2_df = panel_results_to_df(h2_res)
    h2_comp_df = panel_results_to_df(h2_components)

    h1_df.to_excel(REGR_DIR / "h1_vol_fullsample_netsent.xlsx", index=False)
    h1_comp_df.to_excel(REGR_DIR / "h1_vol_fullsample_components.xlsx", index=False)
    h2_df.to_excel(REGR_DIR / "h2_liq_fullsample_netsent.xlsx", index=False)
    h2_comp_df.to_excel(REGR_DIR / "h2_liq_fullsample_components.xlsx", index=False)

    summary = pd.concat(
        [
            h1_df.assign(model="H1_vol", block="netsent"),
            h1_comp_df.assign(model="H1_vol", block="components"),
            h2_df.assign(model="H2_liq", block="netsent"),
            h2_comp_df.assign(model="H2_liq", block="components"),
        ],
        ignore_index=True,
    )
    summary.to_excel(REGR_DIR / "updated_analysis_summary.xlsx", index=False)
    print("saved:", REGR_DIR / "h1_vol_fullsample_netsent.xlsx")
    print("saved:", REGR_DIR / "h1_vol_fullsample_components.xlsx")
    print("saved:", REGR_DIR / "h2_liq_fullsample_netsent.xlsx")
    print("saved:", REGR_DIR / "h2_liq_fullsample_components.xlsx")
    print("saved:", REGR_DIR / "updated_analysis_summary.xlsx")


if __name__ == "__main__":
    main()
