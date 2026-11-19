from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent

OLD_Y_PATH = Path(
    "./local/Desktop/my_data/Master thesis/funds/Data/Y/Y_interpolated.csv"
)
NEW_FUNDS_INFO_DIR = Path(
    "./local/Library/CloudStorage/analyst@example.org/My Drive/4. Disserartion research/calculations/data/funds_info"
)

OUTPUT_UPDATED_Y = CURRENT_DIR / "Y_interpolated_updated.csv"
OUTPUT_APPEND_ONLY = CURRENT_DIR / "Y_actual_append_2024_2025.csv"
OUTPUT_DIAGNOSTICS = CURRENT_DIR / "Y_actual_append_diagnostics.csv"

DATE_RE = re.compile(r"(\d{2}_\d{2}_\d{4})$")


def normalize_month_end(dt: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(dt).to_period("M").to_timestamp("M")


def extract_date_and_prefix(path: Path) -> tuple[pd.Timestamp, str]:
    match = DATE_RE.search(path.stem)
    if not match:
        raise ValueError(f"Cannot extract date from filename: {path.name}")
    raw_date = pd.to_datetime(match.group(1), format="%d_%m_%Y")
    prefix = path.stem[: -(len(match.group(1)) + 1)]
    return normalize_month_end(raw_date), prefix


def target_asset_class(prefix: str) -> str:
    if prefix in {"russian_stocks", "foreign_stocks"}:
        return "stocks"
    return prefix


def clean_string_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})


def parse_from_text(asset_name: object, asset_class: str) -> tuple[object, object]:
    if pd.isna(asset_name):
        return pd.NA, pd.NA

    text = str(asset_name).strip()
    if not text:
        return pd.NA, pd.NA

    if asset_class == "metals":
        return text, text

    if asset_class == "funds":
        return text, text

    if asset_class == "etf_funds":
        parts = [part.strip() for part in text.split(",")]
        if len(parts) >= 2:
            return parts[0], parts[1] if parts[1] else "ETF"
        return text, text

    if asset_class == "bonds":
        parts = [part.strip() for part in text.split(",", 1)]
        if len(parts) == 2:
            return parts[0], parts[1]
        return text, "облигация"

    if asset_class == "stocks":
        parts = [part.strip() for part in text.split(",")]
        if len(parts) >= 2:
            return parts[0], parts[1]
        return text, text

    return text, pd.NA


def load_old_y() -> pd.DataFrame:
    old = pd.read_csv(OLD_Y_PATH, sep="\t", encoding="utf-16", low_memory=False)
    old["date"] = pd.to_datetime(old["date"])
    return old


def load_new_asset_rows(min_date_exclusive: pd.Timestamp) -> pd.DataFrame:
    frames = []

    for path in sorted(NEW_FUNDS_INFO_DIR.glob("*.xls")):
        date, prefix = extract_date_and_prefix(path)
        if date <= min_date_exclusive:
            continue

        df = pd.read_excel(path, skiprows=2).iloc[:-4].copy()
        if "Название фонда" not in df.columns:
            continue

        if prefix not in {
            "funds",
            "etf_funds",
            "bonds",
            "russian_stocks",
            "foreign_stocks",
            "metals",
        }:
            continue

        if "ISIN конечного актива" not in df.columns:
            df["ISIN конечного актива"] = pd.NA

        df.insert(0, "date", date)
        df["source_prefix"] = prefix
        df["Класс активов"] = target_asset_class(prefix)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    assets = pd.concat(frames, ignore_index=True)
    assets = assets[assets["Название фонда"].notna()].copy()
    assets = assets[~assets["Название фонда"].astype(str).str.fullmatch("Итого", na=False)].copy()

    assets["Название фонда"] = clean_string_series(assets["Название фонда"])
    assets["Номер регистрации"] = clean_string_series(assets["Номер регистрации"])
    assets["ISIN Фонда"] = clean_string_series(assets["ISIN Фонда"])
    assets["Название УК"] = clean_string_series(assets["Название УК"])
    assets["Конечный актив"] = clean_string_series(assets["Конечный актив"])
    assets["ISIN конечного актива"] = clean_string_series(assets["ISIN конечного актива"])

    assets["Сумма, руб."] = pd.to_numeric(assets["Сумма, руб."], errors="coerce")
    assets["Доля, %"] = pd.to_numeric(assets["Доля, %"], errors="coerce")
    assets["fund_key"] = assets["Название фонда"] + "__" + assets["Номер регистрации"]

    assets = assets.dropna(subset=["date", "fund_key", "Конечный актив"])
    assets = assets.drop_duplicates(subset=["date", "fund_key", "Конечный актив"], keep="first")
    return assets


def enrich_with_historical_mapping(new_assets: pd.DataFrame, old_y: pd.DataFrame) -> pd.DataFrame:
    map_asset = old_y[
        ["Конечный актив", "Company", "Asset_type", "ISIN конечного актива"]
    ].drop_duplicates()
    map_isin = old_y[
        ["ISIN конечного актива", "Company", "Asset_type"]
    ].dropna(subset=["ISIN конечного актива"]).drop_duplicates("ISIN конечного актива")

    merged = new_assets.merge(map_asset, on="Конечный актив", how="left")
    merged = merged.rename(
        columns={
            "ISIN конечного актива_x": "ISIN конечного актива",
            "ISIN конечного актива_y": "ISIN_hist",
        }
    )
    merged = merged.merge(
        map_isin,
        on="ISIN конечного актива",
        how="left",
        suffixes=("", "_by_isin"),
    )

    merged["Company"] = merged["Company"].combine_first(merged["Company_by_isin"])
    merged["Asset_type"] = merged["Asset_type"].combine_first(merged["Asset_type_by_isin"])

    parsed = merged.apply(
        lambda row: parse_from_text(row["Конечный актив"], row["Класс активов"]),
        axis=1,
        result_type="expand",
    )
    parsed.columns = ["Company_parsed", "Asset_type_parsed"]

    merged["Company"] = merged["Company"].combine_first(parsed["Company_parsed"])
    merged["Asset_type"] = merged["Asset_type"].combine_first(parsed["Asset_type_parsed"])

    merged = merged.drop(
        columns=["Company_by_isin", "Asset_type_by_isin", "ISIN_hist"],
        errors="ignore",
    )
    return merged


def compute_group_metrics(df: pd.DataFrame) -> pd.DataFrame:
    group_keys = ["fund_key", "date"]
    grouped = df.groupby(group_keys, dropna=False)

    share_sum = grouped["Доля, %"].transform("sum")
    amount_sum = grouped["Сумма, руб."].transform("sum")

    df["classified_share"] = share_sum
    df["total_assets"] = amount_sum / share_sum
    df.loc[share_sum.isna() | share_sum.eq(0), "total_assets"] = pd.NA
    df["is_interpolated"] = 0
    return df


def assemble_append_frame(old_y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    min_new_date = old_y["date"].max()
    new_assets = load_new_asset_rows(min_new_date)
    if new_assets.empty:
        return pd.DataFrame(columns=old_y.columns), pd.DataFrame()

    enriched = enrich_with_historical_mapping(new_assets, old_y)
    enriched = compute_group_metrics(enriched)

    append_df = enriched[
        [
            "date",
            "fund_key",
            "Название фонда",
            "Номер регистрации",
            "ISIN Фонда",
            "Название УК",
            "Класс активов",
            "Конечный актив",
            "Company",
            "Asset_type",
            "ISIN конечного актива",
            "Сумма, руб.",
            "Доля, %",
            "total_assets",
            "classified_share",
            "is_interpolated",
        ]
    ].copy()

    diagnostics = (
        enriched.assign(
            company_missing=enriched["Company"].isna().astype(int),
            asset_type_missing=enriched["Asset_type"].isna().astype(int),
        )[
            [
                "date",
                "source_prefix",
                "Класс активов",
                "fund_key",
                "Конечный актив",
                "ISIN конечного актива",
                "Company",
                "Asset_type",
                "company_missing",
                "asset_type_missing",
            ]
        ]
        .sort_values(["date", "source_prefix", "fund_key", "Конечный актив"])
        .reset_index(drop=True)
    )

    return append_df, diagnostics


def combine_without_duplicates(old_y: pd.DataFrame, append_df: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([old_y, append_df], ignore_index=True)
    combined = combined.sort_values(["fund_key", "Конечный актив", "date"])
    combined = combined.drop_duplicates(
        subset=["fund_key", "Конечный актив", "date"],
        keep="first",
    ).reset_index(drop=True)
    return combined


def main() -> None:
    old_y = load_old_y()
    append_df, diagnostics = assemble_append_frame(old_y)

    if append_df.empty:
        print("No new rows found after", old_y["date"].max().date())
        return

    combined = combine_without_duplicates(old_y, append_df)

    append_df.to_csv(OUTPUT_APPEND_ONLY, sep="\t", encoding="utf-16", index=False)
    diagnostics.to_csv(OUTPUT_DIAGNOSTICS, index=False)
    combined.to_csv(OUTPUT_UPDATED_Y, sep="\t", encoding="utf-16", index=False)

    print("Old Y shape:", old_y.shape)
    print("Old max date:", old_y["date"].max().date())
    print("Append shape:", append_df.shape)
    print("Append min/max date:", append_df["date"].min().date(), append_df["date"].max().date())
    print("Combined shape:", combined.shape)
    print("Combined max date:", combined["date"].max().date())
    print("Company missing share in append:", round(append_df["Company"].isna().mean(), 4))
    print("Asset_type missing share in append:", round(append_df["Asset_type"].isna().mean(), 4))
    print("Saved:", OUTPUT_UPDATED_Y)
    print("Saved:", OUTPUT_APPEND_ONLY)
    print("Saved:", OUTPUT_DIAGNOSTICS)


if __name__ == "__main__":
    main()
