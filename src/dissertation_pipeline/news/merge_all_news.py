from pathlib import Path
import pandas as pd

ROOT = Path(r"./local/My Drive\4. Disserartion research\calculations\data\news_links\NEWS_PAGES_PARQ")
OUTPUT = ROOT / "all_news_merged_unified.parquet"

FILES = {
    "frank": ROOT / "frank_merged_full.parquet",
    "interfax": ROOT / "interfax_merged_full.parquet",
    "kommersant": ROOT / "kommersant_merged_full.parquet",
    "prime": ROOT / "prime_merged_full.parquet",
    "rbc": ROOT / "rbc_merged_full.parquet",
}


def first_non_empty(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series("", index=df.index, dtype="object")
    for col in columns:
        if col in df.columns:
            values = df[col].fillna("").astype(str).str.strip()
            result = result.mask(result.eq(""), values)
    return result


# канонические названия колонок -> возможные исходные названия
ALIASES = {
    "media": ["media", "source_media", "dataset_origin", "source"],
    "company": ["company", "company_hint", "query", "search_query"],
    "title": ["title", "header"],
    "link": ["link", "url", "fronturl"],
    "date": ["date", "timestamp", "publish_date"],
    "category": ["category"],
    "text": [
        "article_text_extracted",
        "news_content",
        "article_text",
        "content_txt",
        "text",
        "body",
    ],
}

frames = []

for media_name, file_path in FILES.items():
    if not file_path.exists():
        print(f"SKIP missing file: {file_path}")
        continue

    print(f"Reading: {file_path.name}")
    df = pd.read_parquet(file_path).copy()

    # если явной колонки с названием СМИ нет, подставляем имя файла
    if "media" not in df.columns:
        df["media"] = media_name

    # собираем канонические колонки
    unified = pd.DataFrame(index=df.index)
    for new_col, old_cols in ALIASES.items():
        unified[new_col] = first_non_empty(df, old_cols)

    # какие исходные колонки считаем дублирующими по смыслу
    duplicate_source_cols = set()
    for cols in ALIASES.values():
        duplicate_source_cols.update(cols)

    # оставляем прочие полезные колонки, которые не дублируют канонические
    extra_cols = [c for c in df.columns if c not in duplicate_source_cols and c not in unified.columns]

    df_final = pd.concat([unified, df[extra_cols]], axis=1)

    # чистим пустые ссылки и дубли
    if "link" in df_final.columns:
        df_final["link"] = df_final["link"].fillna("").astype(str).str.strip()
        df_final = df_final[df_final["link"] != ""]
        df_final = df_final.drop_duplicates(subset=["media", "link"], keep="first")

    frames.append(df_final)

if not frames:
    raise RuntimeError("Не удалось прочитать ни один merged parquet.")

result = pd.concat(frames, ignore_index=True, sort=False)

# финальная дедупликация после общего merge
result = result.drop_duplicates(subset=["media", "link"], keep="first")

result.to_parquet(OUTPUT, index=False)
print(f"Saved: {OUTPUT}")
print(f"Rows: {len(result)}")
print("Columns:", list(result.columns))
