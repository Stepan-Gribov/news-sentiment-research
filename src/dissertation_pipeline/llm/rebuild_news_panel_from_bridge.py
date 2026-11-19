import json
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = PROJECT_DIR / "llm_core_replication_outputs" / "News_panel_LLM_parts100.csv"
OUTPUT_XLSX_PATH = PROJECT_DIR / "llm_core_replication_outputs" / "News_panel_LLM_parts100.xlsx"

PARTS_DIR = Path("./local/dev/LLM/outputs/parts(100)")
MASTER_NEWS_CSV = Path(
    "./local/Desktop/my_data/My Drive/4. Disserartion research/calculations/data/news_links/NEWS_PAGES_PARQ/all_news_merged_unified.csv"
)
BRIDGE_CSV_PATH = Path("./local/Desktop/my_data/Master thesis/funds/Data/csv/All_news_my_LLM_mapped_processed.csv")
COMPANY_MAP_PATH = Path("./local/Desktop/my_data/Master thesis/funds/Code/Analisys/company_type_isin1.xlsx")
Y_PATH = PROJECT_DIR / "Y_interpolated_updated.csv"


def load_parts_sentiment(parts_dir: Path) -> pd.DataFrame:
    rows = []
    for jsonl_path in sorted(parts_dir.glob("*.jsonl")):
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                obj = json.loads(line)
                rows.append(
                    {
                        "id": obj.get("id"),
                        "status": obj.get("status"),
                        "sentiment_label": obj.get("sentiment_label"),
                    }
                )
    return pd.DataFrame(rows)


def count_mentions(text, firm, ratio=0.8):
    if pd.isna(text) or pd.isna(firm):
        return 0
    text = str(text).lower()
    firm = str(firm).lower()
    if not firm:
        return 0
    prefix_len = max(1, int(len(firm) * ratio))
    return text.count(firm[:prefix_len])


def main():
    print("[1/6] load company map and Y", flush=True)
    company_map = pd.read_excel(COMPANY_MAP_PATH)

    y_df = pd.read_csv(Y_PATH, sep="\t", encoding="utf-16", low_memory=False)
    y_df["date"] = pd.to_datetime(y_df["date"])
    fund_dates = sorted(y_df["date"].drop_duplicates())

    print("[2/6] load master links and parts sentiment", flush=True)
    master_links = pd.read_csv(MASTER_NEWS_CSV, usecols=["link"]).reset_index().rename(columns={"index": "id"})
    parts_df = load_parts_sentiment(PARTS_DIR)
    link_sentiment = master_links.merge(parts_df, on="id", how="left", validate="one_to_one")

    print("[3/6] load and normalize bridge", flush=True)
    bridge_raw = pd.read_csv(
        BRIDGE_CSV_PATH,
        encoding="utf-16",
        sep="\t",
        engine="python",
        on_bad_lines="skip",
    )
    col = bridge_raw.columns[0]
    bridge_df = bridge_raw[col].str.split("^", expand=True)
    bridge_df = bridge_df.iloc[:, :8].copy()
    bridge_df.columns = [
        "news_source",
        "title",
        "link",
        "date",
        "news_content",
        "firm_list",
        "firm_name",
        "legacy_sentiment",
    ]
    bridge_df = bridge_df[["link", "date", "news_content", "firm_name"]].dropna(subset=["link", "date", "firm_name"])
    bridge_df = bridge_df.drop_duplicates()

    print("[4/6] merge bridge with sentiment", flush=True)
    bridge_df = bridge_df.merge(
        link_sentiment[["link", "status", "sentiment_label"]],
        on="link",
        how="left",
        validate="many_to_one",
    )

    sentiment_map = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}
    bridge_df["sentiment"] = bridge_df["sentiment_label"].map(sentiment_map).fillna("Н/Д")
    bridge_df["date"] = pd.to_datetime(bridge_df["date"], errors="coerce")
    bridge_df = bridge_df.dropna(subset=["date"]).rename(columns={"firm_name": "firm_in_text"})

    bridge_df["n_matches"] = [
        count_mentions(text, firm)
        for text, firm in zip(bridge_df["news_content"], bridge_df["firm_in_text"])
    ]
    for word in ["отчетность", "прогноз", "рынок", "санкции", "мосбиржа", "акции"]:
        prefix = word[: max(1, int(len(word) * 0.75))]
        bridge_df[word] = bridge_df["news_content"].astype(str).str.lower().str.count(prefix)

    print("[5/6] build interval panel", flush=True)
    interval_labels = [
        f"{fund_dates[i + 1].strftime('%Y-%m-%d')}:{fund_dates[i].strftime('%Y-%m-%d')}"
        for i in range(len(fund_dates) - 1)
    ]
    bridge_df["interval"] = pd.cut(
        bridge_df["date"],
        bins=fund_dates,
        labels=interval_labels,
        include_lowest=True,
        ordered=True,
    )
    bridge_df = bridge_df.dropna(subset=["interval"]).copy()

    count_pivot = bridge_df.pivot_table(
        index=["firm_in_text", "interval"],
        columns="sentiment",
        aggfunc="size",
        fill_value=0,
    )
    sum_pivot = bridge_df.pivot_table(
        index=["firm_in_text", "interval"],
        values=["n_matches", "акции", "мосбиржа", "отчетность", "прогноз", "рынок", "санкции"],
        aggfunc="sum",
        fill_value=0,
    )

    panel = (
        count_pivot.reset_index()
        .merge(sum_pivot.reset_index(), on=["firm_in_text", "interval"], how="outer")
        .fillna(0)
    )
    for colname in ["Negative", "Neutral", "Positive", "Н/Д"]:
        if colname not in panel.columns:
            panel[colname] = 0

    panel.insert(2, "start_date", pd.to_datetime(panel["interval"].astype(str).str.split(":").str[1]))
    panel.insert(3, "end_date", pd.to_datetime(panel["interval"].astype(str).str.split(":").str[0]))

    print("[6/6] merge company map and save outputs", flush=True)
    panel = company_map.merge(panel, left_on="Company", right_on="firm_in_text", how="right").drop_duplicates()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUTPUT_PATH, index=False)
    try:
        panel.to_excel(OUTPUT_XLSX_PATH, index=False)
    except Exception as exc:
        print(f"Excel export warning: {exc}")

    print("saved:", OUTPUT_PATH)
    print("shape:", panel.shape)
    print("date range:", panel["end_date"].min(), panel["end_date"].max())


if __name__ == "__main__":
    main()
