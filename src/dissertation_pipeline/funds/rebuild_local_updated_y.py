from __future__ import annotations

from pathlib import Path

import pandas as pd


OLD_Y_PATH = Path("./local/Desktop/my_data/Master thesis/funds/Data/Y/Y_interpolated.csv")
APPEND_Y_PATH = Path(
    "./local/Library/CloudStorage/analyst@example.org/My Drive/4. Disserartion research/calculations/code/Y_actual_append_2024_2025.csv"
)
OUTPUT_DIR = Path("./local/Desktop/my_data/Dissertation/research_v2/data/funds")
OUTPUT_Y_PATH = OUTPUT_DIR / "Y_interpolated_updated.csv"


def read_y(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-16", sep="\t", low_memory=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    old_y = read_y(OLD_Y_PATH)
    append_y = read_y(APPEND_Y_PATH)

    old_y["date"] = pd.to_datetime(old_y["date"])
    append_y["date"] = pd.to_datetime(append_y["date"])

    required_cols = [
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

    old_y = old_y[required_cols].copy()
    append_y = append_y[required_cols].copy()

    combined = pd.concat([old_y, append_y], ignore_index=True)
    combined = combined.sort_values(["fund_key", "Конечный актив", "date"])
    combined = combined.drop_duplicates(
        subset=["fund_key", "Конечный актив", "date"],
        keep="first",
    ).reset_index(drop=True)

    combined["date"] = pd.to_datetime(combined["date"]).dt.strftime("%Y-%m-%d")
    combined.to_csv(OUTPUT_Y_PATH, encoding="utf-16", sep="\t", index=False)

    print(f"saved: {OUTPUT_Y_PATH}")
    print(f"rows_old: {len(old_y)}")
    print(f"rows_append: {len(append_y)}")
    print(f"rows_combined: {len(combined)}")
    print(f"date_min: {combined['date'].min()}")
    print(f"date_max: {combined['date'].max()}")


if __name__ == "__main__":
    main()
