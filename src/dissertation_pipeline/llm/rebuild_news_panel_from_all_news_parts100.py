from __future__ import annotations

import json
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path("./local/Desktop/my_data/Dissertation/research_v2")
Y_PATH = PROJECT_ROOT / "data" / "funds" / "Y_interpolated_updated.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "news" / "News_panel_LLM_parts100.csv"

ALL_NEWS_TEXT_PATH = Path(
    "./local/Library/CloudStorage/analyst@example.org/My Drive/4. Disserartion research/calculations/code/LLM/outputs/all_news.jsonl"
)
MASTER_NEWS_META_CSV = Path(
    "./local/Desktop/my_data/My Drive/4. Disserartion research/calculations/data/news_links/NEWS_PAGES_PARQ/all_news_merged_unified.csv"
)
PARTS_DIR = Path(
    "./local/Library/CloudStorage/analyst@example.org/My Drive/4. Disserartion research/calculations/code/LLM/outputs/parts(100)"
)

KEYWORDS = ["акции", "мосбиржа", "отчетность", "прогноз", "рынок", "санкции"]
SENTIMENT_MAP = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}


def norm_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def count_mentions(text: object, firm: object, ratio: float = 0.8) -> int:
    if pd.isna(text) or pd.isna(firm):
        return 0
    firm_norm = norm_text(firm)
    if not firm_norm:
        return 0
    token = firm_norm[: max(1, int(len(firm_norm) * ratio))]
    return norm_text(text).count(token)


def load_y_inputs() -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    y_df = pd.read_csv(Y_PATH, encoding="utf-16", sep="\t", low_memory=False)
    y_df["date"] = pd.to_datetime(y_df["date"])
    fund_dates = sorted(y_df["date"].dropna().drop_duplicates().tolist())
    return y_df, fund_dates


def build_company_map(y_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Конечный актив", "ISIN конечного актива", "Asset_type", "Company"]
    company_map = y_df[cols].dropna(subset=["Company"]).drop_duplicates().copy()
    company_map["company_key"] = company_map["Company"].map(norm_text)
    company_map = company_map.drop_duplicates(subset=["company_key", "Конечный актив", "ISIN конечного актива", "Asset_type"])
    return company_map


def load_sentiment_labels(parts_dir: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    for jsonl_path in sorted(parts_dir.glob("*.jsonl")):
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                obj = json.loads(line)
                if obj.get("status") != "ok":
                    continue
                label = obj.get("sentiment_label")
                if label in SENTIMENT_MAP:
                    labels[int(obj["id"])] = SENTIMENT_MAP[label]
    return labels


def interval_index(article_date: pd.Timestamp, fund_dates: list[pd.Timestamp]) -> int | None:
    if pd.isna(article_date) or article_date < fund_dates[0] or article_date > fund_dates[-1]:
        return None
    idx = bisect_left(fund_dates[1:], article_date)
    if idx < 0 or idx >= len(fund_dates) - 1:
        return None
    return idx


def aggregate_news(fund_dates: list[pd.Timestamp], labels_by_id: dict[int, str]) -> pd.DataFrame:
    rows: dict[tuple[str, int], dict[str, object]] = defaultdict(dict)
    total_rows = 0
    matched_rows = 0

    for chunk in pd.read_csv(
        MASTER_NEWS_META_CSV,
        usecols=["company", "date", "text"],
        chunksize=50000,
        low_memory=False,
    ):
        chunk = chunk.reset_index(drop=True)
        chunk["row_id"] = range(total_rows, total_rows + len(chunk))
        total_rows += len(chunk)
        chunk["sentiment"] = chunk["row_id"].map(labels_by_id)
        chunk = chunk.dropna(subset=["sentiment"]).copy()
        if chunk.empty:
            continue

        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        chunk = chunk.dropna(subset=["date", "company"]).copy()
        if chunk.empty:
            continue

        for row in chunk.itertuples(index=False):
            idx = interval_index(row.date, fund_dates)
            if idx is None:
                continue

            firm_in_text = row.company
            matched_rows += 1
            key = (norm_text(firm_in_text), idx)
            rec = rows[key]
            if not rec:
                rec.update(
                    {
                        "firm_in_text": firm_in_text,
                        "interval_idx": idx,
                        "start_date": fund_dates[idx],
                        "end_date": fund_dates[idx + 1],
                        "interval": f"{fund_dates[idx + 1].strftime('%Y-%m-%d')}:{fund_dates[idx].strftime('%Y-%m-%d')}",
                        "Positive": 0,
                        "Negative": 0,
                        "Neutral": 0,
                        "Н/Д": 0,
                        "n_matches": 0,
                        "акции": 0,
                        "мосбиржа": 0,
                        "отчетность": 0,
                        "прогноз": 0,
                        "рынок": 0,
                        "санкции": 0,
                    }
                )

            rec[row.sentiment] += 1
            article_text = row.text
            rec["n_matches"] += count_mentions(article_text, firm_in_text)
            text_norm = norm_text(article_text)
            for word in KEYWORDS:
                prefix = word[: max(1, int(len(word) * 0.75))]
                rec[word] += text_norm.count(prefix)

    panel = pd.DataFrame(rows.values())
    print(f"all_news rows scanned: {total_rows}")
    print(f"classified articles matched into intervals: {matched_rows}")
    return panel


def main() -> None:
    print("[1/4] load Y and company map", flush=True)
    y_df, fund_dates = load_y_inputs()
    company_map = build_company_map(y_df)
    print("Y rows:", len(y_df), "date range:", y_df["date"].min(), y_df["date"].max(), flush=True)
    print("fund dates:", len(fund_dates), flush=True)
    print("company map rows:", len(company_map), "unique companies:", company_map["company_key"].nunique(), flush=True)

    print("[2/4] load sentiment labels", flush=True)
    labels_by_id = load_sentiment_labels(PARTS_DIR)
    print("successful classified ids:", len(labels_by_id), flush=True)
    print("all_news text file available:", ALL_NEWS_TEXT_PATH.exists(), flush=True)
    print("metadata csv source:", MASTER_NEWS_META_CSV, flush=True)

    print("[3/4] aggregate full all_news layer", flush=True)
    news_panel = aggregate_news(fund_dates, labels_by_id)
    news_panel["company_key"] = news_panel["firm_in_text"].map(norm_text)

    print("[4/4] merge with company map and save", flush=True)
    panel = company_map.merge(news_panel, on="company_key", how="right")
    panel = panel.drop(columns=["company_key"])

    ordered_cols = [
        "Конечный актив",
        "ISIN конечного актива",
        "Asset_type",
        "Company",
        "firm_in_text",
        "interval",
        "start_date",
        "end_date",
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
    for col in ordered_cols:
        if col not in panel.columns:
            panel[col] = pd.NA
    panel = panel[ordered_cols].drop_duplicates()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUTPUT_PATH, index=False)

    print("saved:", OUTPUT_PATH, flush=True)
    print("shape:", panel.shape, flush=True)
    print("date range:", panel["end_date"].min(), panel["end_date"].max(), flush=True)
    print(
        "nonzero months:",
        int(
            panel.groupby("end_date")[["Positive", "Negative", "Neutral"]]
            .sum()
            .sum(axis=1)
            .gt(0)
            .sum()
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
