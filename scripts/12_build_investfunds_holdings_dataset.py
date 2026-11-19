from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

from dissertation_pipeline.paths import PATHS


DEFAULT_RUN_DIR = PATHS.funds_dir / "investfunds_fund_holdings" / "runs" / "2026_07_01"
DEFAULT_OUTPUT_DIR = PATHS.funds_dir / "investfunds_fund_holdings" / "processed"
ERROR_COLUMNS = [
    "source_path",
    "fund_id",
    "fund_name_file",
    "report_date_from_filename",
    "download_id",
    "error_type",
    "error",
]
TOP_ASSET_CATEGORY_MARKERS = [
    "денежные средства",
    "акции",
    "облигации",
    "паи",
    "etf",
    "драгоценные металлы",
    "недвижимость",
    "векселя",
    "доли в ооо",
    "дебиторская задолженность",
    "производные",
    "прочие активы",
]


@dataclass
class ParsedWorkbook:
    source_path: str
    fund_id: str
    fund_name_file: str
    download_id: str
    report_date_from_filename: str
    fund_name_xls: str | None
    report_date: str | None
    registration_number: str | None
    management_company: str | None
    fund_type_category: str | None
    net_asset_value_rub: float | None
    fund_info_url: str | None
    total_assets_thousand_rub: float | None
    total_obligations_thousand_rub: float | None
    asset_rows_json: str
    obligation_rows_json: str


def parse_number(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace("\u00a0", " ").strip()
    if not text:
        return None
    text = text.replace(" ", "").replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def clean_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def parse_filename(path: Path) -> dict[str, str]:
    fund_id = path.parent.name
    stem = path.stem
    match = re.match(r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<fund>.+)_(?P<download_id>\d+)$", stem)
    if not match:
        return {
            "fund_id": fund_id,
            "report_date_from_filename": "",
            "fund_name_file": stem,
            "download_id": "",
        }
    return {
        "fund_id": fund_id,
        "report_date_from_filename": match.group("date"),
        "fund_name_file": match.group("fund").replace("_", " "),
        "download_id": match.group("download_id"),
    }


def find_row(df: pd.DataFrame, pattern: str) -> int | None:
    regex = re.compile(pattern, flags=re.IGNORECASE)
    for idx, row in df.iterrows():
        text = " ".join(str(x) for x in row.dropna().tolist())
        if regex.search(text):
            return int(idx)
    return None


def metadata_value(df: pd.DataFrame, label: str) -> str | None:
    for _, row in df.iterrows():
        left = clean_text(row.iloc[1] if len(row) > 1 else None)
        if left and label.casefold() in left.casefold():
            return clean_text(row.iloc[2] if len(row) > 2 else None)
    return None


def fund_name_from_title(df: pd.DataFrame) -> str | None:
    for _, row in df.iterrows():
        text = clean_text(row.iloc[1] if len(row) > 1 else None)
        if not text:
            continue
        match = re.search(r'фонда\s+"([^"]+)"', text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def classify_asset_row(name: str, registration_number: str | None, current_subcategory: str | None) -> str:
    low = name.casefold()
    if low.startswith("итого сумма активов"):
        return "asset_total"
    if name.startswith("-"):
        return "asset_subcategory"
    if registration_number:
        return "asset_item"
    if current_subcategory and not any(marker in low for marker in TOP_ASSET_CATEGORY_MARKERS):
        return "asset_item"
    if "задолженность" in low:
        return "asset_category"
    if low in {"etf"}:
        return "asset_category"
    return "asset_category"


def parse_workbook(path: Path) -> tuple[ParsedWorkbook, list[dict[str, object]]]:
    file_meta = parse_filename(path)
    # xlrd prints OLE2 consistency warnings directly to stdout for many legacy XLS files.
    # They are noisy but not fatal, so keep progress output readable.
    with contextlib.redirect_stdout(io.StringIO()):
        df = pd.read_excel(path, sheet_name=0, header=None, engine="xlrd")
    df = df.dropna(how="all").reset_index(drop=True)

    asset_header_idx = find_row(df, r"Название вида актива|ценной бумаги")
    obligation_header_idx = find_row(df, r"Обязательства\s+Сумма обязательств|Обязательства")
    if asset_header_idx is None:
        raise ValueError("asset header not found")

    fund_name_xls = fund_name_from_title(df)
    report_date_raw = metadata_value(df, "Дата отчетности")
    report_date = None
    if report_date_raw:
        parsed = pd.to_datetime(report_date_raw, dayfirst=True, errors="coerce")
        report_date = parsed.date().isoformat() if pd.notna(parsed) else report_date_raw

    common = {
        "source_path": str(path),
        "fund_id": file_meta["fund_id"],
        "fund_name_file": file_meta["fund_name_file"],
        "download_id": file_meta["download_id"],
        "report_date_from_filename": file_meta["report_date_from_filename"],
        "fund_name_xls": fund_name_xls,
        "report_date": report_date,
        "registration_number": metadata_value(df, "Регистрационный номер фонда"),
        "management_company": metadata_value(df, "Управляющая компания"),
        "fund_type_category": metadata_value(df, "Тип и категория фонда"),
        "net_asset_value_rub": parse_number(metadata_value(df, "СЧА на дату отчетности")),
        "fund_info_url": metadata_value(df, "Подробная информация о фонде"),
    }

    long_rows: list[dict[str, object]] = []
    asset_rows_for_json: list[dict[str, object]] = []
    obligation_rows_for_json: list[dict[str, object]] = []
    total_assets = None
    total_obligations = None

    asset_end_idx = obligation_header_idx if obligation_header_idx is not None else len(df)
    current_category: str | None = None
    current_subcategory: str | None = None
    for row_order, (_, row) in enumerate(df.iloc[asset_header_idx + 1 : asset_end_idx].iterrows(), start=1):
        name = clean_text(row.iloc[1] if len(row) > 1 else None)
        if not name:
            continue
        registration = clean_text(row.iloc[2] if len(row) > 2 else None)
        amount = parse_number(row.iloc[3] if len(row) > 3 else None)
        share = parse_number(row.iloc[4] if len(row) > 4 else None)
        row_type = classify_asset_row(name, registration, current_subcategory)
        if row_type == "asset_category":
            current_category = name
            current_subcategory = None
        elif row_type == "asset_subcategory":
            current_subcategory = name.lstrip("-").strip()
        elif row_type == "asset_total":
            total_assets = amount

        item = {
            **common,
            "table_type": "asset",
            "row_order": row_order,
            "row_type": row_type,
            "parent_category": current_category,
            "parent_subcategory": current_subcategory,
            "item_name": name,
            "registration_number_security": registration,
            "amount_thousand_rub": amount,
            "share_pct": share,
            "liability_amount_thousand_rub": None,
        }
        long_rows.append(item)
        asset_rows_for_json.append(
            {
                "row_order": row_order,
                "row_type": row_type,
                "parent_category": current_category,
                "parent_subcategory": current_subcategory,
                "item_name": name,
                "registration_number_security": registration,
                "amount_thousand_rub": amount,
                "share_pct": share,
            }
        )

    if obligation_header_idx is not None:
        for row_order, (_, row) in enumerate(df.iloc[obligation_header_idx + 1 :].iterrows(), start=1):
            name = clean_text(row.iloc[1] if len(row) > 1 else None)
            if not name:
                continue
            amount = parse_number(row.iloc[2] if len(row) > 2 else None)
            row_type = "obligation_total" if name.casefold().startswith("итого сумма обязательств") else "obligation_item"
            if row_type == "obligation_total":
                total_obligations = amount
            item = {
                **common,
                "table_type": "obligation",
                "row_order": row_order,
                "row_type": row_type,
                "parent_category": None,
                "parent_subcategory": None,
                "item_name": name,
                "registration_number_security": None,
                "amount_thousand_rub": None,
                "share_pct": None,
                "liability_amount_thousand_rub": amount,
            }
            long_rows.append(item)
            obligation_rows_for_json.append(
                {
                    "row_order": row_order,
                    "row_type": row_type,
                    "item_name": name,
                    "liability_amount_thousand_rub": amount,
                }
            )

    wide = ParsedWorkbook(
        **common,
        total_assets_thousand_rub=total_assets,
        total_obligations_thousand_rub=total_obligations,
        asset_rows_json=json.dumps(asset_rows_for_json, ensure_ascii=False),
        obligation_rows_json=json.dumps(obligation_rows_for_json, ensure_ascii=False),
    )
    return wide, long_rows


def load_manifest(run_dir: Path) -> pd.DataFrame:
    manifest = run_dir / "manifest.csv"
    if not manifest.exists():
        return pd.DataFrame()
    df = pd.read_csv(manifest)
    if "xls_path" not in df.columns:
        return pd.DataFrame()
    return df


def resolve_xls_files(run_dir: Path, sample_files: int | None = None) -> list[Path]:
    manifest = load_manifest(run_dir)
    files: list[Path] = []
    if not manifest.empty:
        mask = manifest.get("xls_path").notna()
        if "status" in manifest.columns:
            mask &= manifest["status"].astype(str).isin(["downloaded", "exists", "exists_global"])
        for value in manifest.loc[mask, "xls_path"].dropna().astype(str).drop_duplicates():
            path = Path(value)
            if path.exists() and path.suffix.lower() == ".xls":
                files.append(path)
    if not files:
        files = sorted((run_dir / "xls").glob("*/*.xls"))
    files = sorted(set(files))
    if sample_files:
        files = files[:sample_files]
    return files


def write_outputs(wide_rows: list[ParsedWorkbook], long_rows: list[dict[str, object]], errors: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    wide_df = pd.DataFrame([asdict(row) for row in wide_rows])
    long_df = pd.DataFrame(long_rows)
    errors_df = pd.DataFrame(errors, columns=ERROR_COLUMNS)

    parquet_available = importlib.util.find_spec("pyarrow") is not None or importlib.util.find_spec("fastparquet") is not None
    if parquet_available:
        wide_df.to_parquet(output_dir / "fund_holdings_wide.parquet", index=False)
        long_df.to_parquet(output_dir / "fund_holdings_long.parquet", index=False)

    wide_df.to_csv(output_dir / "fund_holdings_wide.csv", index=False, encoding="utf-8-sig")
    wide_df.to_csv(output_dir / "fund_holdings_wide.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")
    long_df.to_csv(output_dir / "fund_holdings_long.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")
    errors_df.to_csv(output_dir / "fund_holdings_parse_errors.csv", index=False, encoding="utf-8-sig")

    summary = {
        "wide_rows": len(wide_df),
        "long_rows": len(long_df),
        "errors": len(errors_df),
        "parquet_written": parquet_available,
        "wide_parquet": str(output_dir / "fund_holdings_wide.parquet") if parquet_available else None,
        "long_parquet": str(output_dir / "fund_holdings_long.parquet") if parquet_available else None,
        "wide_csv": str(output_dir / "fund_holdings_wide.csv"),
        "wide_csv_gz": str(output_dir / "fund_holdings_wide.csv.gz"),
        "long_csv_gz": str(output_dir / "fund_holdings_long.csv.gz"),
        "errors_csv": str(output_dir / "fund_holdings_parse_errors.csv"),
    }
    (output_dir / "fund_holdings_build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build combined InvestFunds holdings datasets from downloaded XLS files.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-files", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = resolve_xls_files(args.run_dir, args.sample_files)
    if not files:
        raise FileNotFoundError(f"No XLS files found under {args.run_dir}")

    print(f"xls files selected: {len(files):,}", flush=True)
    wide_rows: list[ParsedWorkbook] = []
    long_rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []

    for i, path in enumerate(files, start=1):
        try:
            wide, rows = parse_workbook(path)
            wide_rows.append(wide)
            long_rows.extend(rows)
        except Exception as exc:  # noqa: BLE001
            meta = parse_filename(path)
            errors.append(
                {
                    "source_path": str(path),
                    "fund_id": meta["fund_id"],
                    "fund_name_file": meta["fund_name_file"],
                    "report_date_from_filename": meta["report_date_from_filename"],
                    "download_id": meta["download_id"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        if i == 1 or i % 500 == 0 or i == len(files):
            print(
                f"processed={i:,}/{len(files):,} wide={len(wide_rows):,} long={len(long_rows):,} errors={len(errors):,}",
                flush=True,
            )

    write_outputs(wide_rows, long_rows, errors, args.output_dir)
    print(f"done output_dir={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
