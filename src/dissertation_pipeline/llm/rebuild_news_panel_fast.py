import json
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = PROJECT_DIR / "llm_core_replication_outputs" / "News_panel_LLM_parts100.csv"

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


def main():
    print("[1/5] load Y and company map", flush=True)
    y_df = pd.read_csv(Y_PATH, sep="\t", encoding="utf-16", low_memory=False)
    y_df["date"] = pd.to_datetime(y_df["date"])
    fund_dates = sorted(y_df["date"].drop_duplicates())
    company_map = pd.read_excel(COMPANY_MAP_PATH)

    print("[2/5] load master links and parts", flush=True)
    master_links = pd.read_csv(MASTER_NEWS_CSV, usecols=["link"]).reset_index().rename(columns={"index": "id"})
    parts_df = load_parts_sentiment(PARTS_DIR)
    link_sentiment = master_links.merge(parts_df, on="id", how="left", validate="one_to_one")

    print("[3/5] load bridge", flush=True)
    bridge_raw = pd.read_csv(BRIDGE_CSV_PATH, encoding="utf-16", sep="\t", engine="python", on_bad_lines="skip")
    bridge = bridge_raw[bridge_raw.columns[0]].str.split("^", expand=True).iloc[:, :7].copy()
    bridge.columns = ["news_source", "title", "link", "date", "news_content", "firm_list", "firm_name"]
    bridge = bridge[["link", "date", "firm_name"]].dropna().drop_duplicates()
    bridge["date"] = pd.to_datetime(bridge["date"], errors="coerce")
    bridge = bridge.dropna(subset=["date"]).rename(columns={"firm_name": "firm_in_text"})

    print("[4/5] merge and aggregate", flush=True)
    bridge = bridge.merge(
        link_sentiment[["link", "status", "sentiment_label"]],
        on="link",
        how="left",
        validate="many_to_one",
    )
    sentiment_map = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}
    bridge["sentiment"] = bridge["sentiment_label"].map(sentiment_map).fillna("Н/Д")

    interval_labels = [
        f"{fund_dates[i + 1].strftime('%Y-%m-%d')}:{fund_dates[i].strftime('%Y-%m-%d')}"
        for i in range(len(fund_dates) - 1)
    ]
    bridge["interval"] = pd.cut(
        bridge["date"],
        bins=fund_dates,
        labels=interval_labels,
        include_lowest=True,
        ordered=True,
    )
    bridge = bridge.dropna(subset=["interval"]).copy()

    count_pivot = bridge.pivot_table(
        index=["firm_in_text", "interval"],
        columns="sentiment",
        aggfunc="size",
        fill_value=0,
    ).reset_index()

    for col in ["Negative", "Neutral", "Positive", "Н/Д"]:
        if col not in count_pivot.columns:
            count_pivot[col] = 0
    count_pivot["n_matches"] = 0
    for col in ["акции", "мосбиржа", "отчетность", "прогноз", "рынок", "санкции"]:
        count_pivot[col] = 0

    count_pivot.insert(2, "start_date", pd.to_datetime(count_pivot["interval"].astype(str).str.split(":").str[1]))
    count_pivot.insert(3, "end_date", pd.to_datetime(count_pivot["interval"].astype(str).str.split(":").str[0]))

    panel = company_map.merge(count_pivot, left_on="Company", right_on="firm_in_text", how="right").drop_duplicates()

    print("[5/5] save", flush=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUTPUT_PATH, index=False)
    print("saved:", OUTPUT_PATH, flush=True)
    print("shape:", panel.shape, flush=True)
    print("date range:", panel["end_date"].min(), panel["end_date"].max(), flush=True)


if __name__ == "__main__":
    main()
