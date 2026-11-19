from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

from dissertation_pipeline.paths import PATHS


DEFAULT_Y_PATH = PATHS.funds_dir / "current" / "Y_interpolated_updated.csv"
OUTPUT_NAMES = PATHS.config_dir / "company_names_from_fund_assets.txt"
OUTPUT_FULL_NAMES = PATHS.config_dir / "company_names_from_fund_assets_full.txt"
OUTPUT_TOP500_NAMES = PATHS.config_dir / "company_names_from_fund_assets_top500.txt"
OUTPUT_DIAGNOSTICS = PATHS.config_dir / "company_names_from_fund_assets_sources.csv"
OUTPUT_RAW = PATHS.config_dir / "company_names_from_fund_assets_raw.csv"
ACTIVE_NAMES = PATHS.company_names_file
ACTIVE_BACKUP = PATHS.config_dir / "company_names_before_fund_assets.txt"

READ_COLUMNS = [
    "date",
    "Класс активов",
    "Конечный актив",
    "Company",
    "Asset_type",
    "ISIN конечного актива",
    "Сумма, руб.",
    "Доля, %",
]

LEGAL_PATTERNS = [
    r"публичное акционерное общество",
    r"открытое акционерное общество",
    r"закрытое акционерное общество",
    r"акционерное общество",
    r"общество с ограниченной ответственностью",
    r"публичная компания",
    r"международная компания",
    r"\bмкпао\b",
    r"\bпао\b",
    r"\bоао\b",
    r"\bзао\b",
    r"\bао\b",
    r"\bооо\b",
    r"\bнко\b",
    r"\bнпао\b",
    r"\bpublic joint stock company\b",
    r"\bpublic joint-stock company\b",
    r"\bjoint stock company\b",
    r"\bjoint-stock company\b",
    r"\bpjsc\b",
    r"\bojsc\b",
    r"\bcjsc\b",
    r"\bjsc\b",
    r"\bllc\b",
    r"\bltd\b",
    r"\bplc\b",
    r"\bincorporated\b",
    r"\binc\b",
    r"\bcorp\b",
    r"\bcorporation\b",
    r"\bcompany\b",
    r"\bco\b",
]

SECURITY_TAIL_PATTERNS = [
    r"\bord\s*shs\b",
    r"\borderinary shares?\b",
    r"\bcommon shares?\b",
    r"\bpreferred shares?\b",
    r"\bpreference shares?\b",
    r"\bclass\s+[a-z]\b",
    r"\bords?\b",
    r"\bprefs?\b",
    r"\badr\b",
    r"\bgdr\b",
    r"\bрдр\b",
    r"\bгдр\b",
    r"\bакци[ия]\b",
    r"\bакц[.]?\b",
    r"\bобыкновенн\w*\b",
    r"\bпривилегированн\w*\b",
    r"\bао\b",
    r"\bап\b",
    r"\bоб[.]?\b",
]

DROP_PATTERNS = [
    r"\bi?nav\b",
    r"\betf\b",
    r"\bбпиф\b",
    r"\bзпиф\b",
    r"\bипиф\b",
    r"\bопиф\b",
    r"\bпиф\b",
    r"\bпа[ий]\b",
    r"\bфонд\b",
    r"\bfund\b",
    r"\bindex\b",
    r"\bиндекс\b",
    r"расч[её]тная цена",
    r"\bденежн\w+\s+средств\w*\b",
    r"\bден[.]?\s*средств\w*\b",
    r"\bвалют\w*\b",
    r"\bдепозит\w*\b",
    r"\bкэш\b",
    r"\bcash\b",
    r"\bитого\b",
    r"\btotal\b",
    r"\bоблигация иностранного государства\b",
    r"\bкомпания\b$",
    r"\bmanagement\b",
    r"\bсопф\b",
    r"\bинфраструктурные облигации\b",
    r"^group$",
]

GOVERNMENT_PATTERNS = [
    r"\bросси[яи]\b",
    r"\bроссийская федерация\b",
    r"\brussian federation\b",
    r"\bминистерство\b",
    r"\bминфин\b",
    r"\bдепартамент\b",
    r"\bказначейств\w*\b",
    r"\bправительств\w*\b",
    r"\bадминистраци\w*\b",
    r"\bмуниципальн\w*\b",
    r"\bфедеральн\w+\s+за[её]м\w*\b",
    r"\bофз\b",
    r"\bобласть\b",
    r"\bкрай\b",
    r"\bреспублика\b",
    r"\bгород\b",
    r"\bсубъект\b",
    r"\bказахстан\b",
    r"\bбеларусь\b",
    r"\bтурция\b",
    r"\bегипет\b",
    r"\bюар\b",
    r"\bоман\b",
    r"\bkazakhstan\b",
    r"\bbelarus\b",
    r"\bturkey\b",
    r"\begypt\b",
    r"\bsouth africa\b",
    r"\boman\b",
]

BRAND_REPLACEMENTS = [
    (r".*\bавтодор\b.*", "Автодор"),
    (r".*\bunitedhealth\b.*", "UnitedHealth Group"),
    (r".*\bсбербанк\w*(?:\s+россии)?\b.*", "Сбербанк"),
    (r".*\bлукойл\b.*", "ЛУКОЙЛ"),
    (r".*\bгазпром\s+нефть\b.*", "Газпром нефть"),
    (r".*\bгазпром\s+газораспределение\b.*", "Газпром газораспределение"),
    (r".*\bгазпром\b.*", "Газпром"),
    (r".*\bгмк\s+норильский\s+никель\b.*", "Норильский никель"),
    (r".*\bнорильский\s+никель\b.*", "Норильский никель"),
    (r".*\bноволипецкий\s+металлургический\s+комбинат\b.*", "НЛМК"),
    (r".*\bмагнитогорский\s+металлургический\s+комбинат\b.*", "ММК"),
    (r".*\bалроса\b.*", "АЛРОСА"),
    (r".*\bафк\s+система\b.*", "АФК Система"),
    (r".*\bакционерная\s+финансовая\s+корпорация\s+система\b.*", "АФК Система"),
    (r".*\bроссийские\s+сети\b.*", "Россети"),
    (r".*\bроссети\b.*", "Россети"),
    (r".*\bфск\s+еэс\b.*", "ФСК ЕЭС"),
    (r".*\bмосковская\s+биржа\b.*", "Московская Биржа"),
    (r".*\bмосбиржа\b.*", "Московская Биржа"),
    (r".*\bяндекс\b.*", "Яндекс"),
    (r".*\byandex\b.*", "Яндекс"),
    (r".*\bт[-\s]?технологи\w*\b.*", "Т-Технологии"),
    (r".*\bтинькофф\b.*", "Т-Технологии"),
    (r".*\bт\s*банк\b.*", "Т-Банк"),
    (r".*\bмобильные\s+телесистемы\b.*", "МТС"),
    (r".*\bмтс\b.*", "МТС"),
    (r".*\bвтб\b.*", "ВТБ"),
    (r".*\bмосковский\s+кредитный\s+банк\b.*", "Московский Кредитный банк"),
    (r".*\bмкб\b.*", "Московский Кредитный банк"),
    (r".*\bрусал\b.*", "РУСАЛ"),
    (r".*\brusal\b.*", "РУСАЛ"),
    (r".*\bсеверсталь\b.*", "Северсталь"),
    (r".*\bмагнит\b.*", "Магнит"),
    (r".*\bноватэк\b.*", "НОВАТЭК"),
    (r".*\bроснефть\b.*", "Роснефть"),
    (r".*\bтатнефть\b.*", "Татнефть"),
    (r".*\bсургутнефтегаз\b.*", "Сургутнефтегаз"),
    (r".*\bростелеком\b.*", "Ростелеком"),
    (r".*\bинтер\s*рао\b.*", "Интер РАО"),
]


def normalize_spaces(value: str) -> str:
    value = value.replace("\u00a0", " ")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s+([,.;:])", r"\1", value)
    return value.strip(" \t\r\n,.;:-")


def strip_wrapping_quotes(value: str) -> str:
    value = value.replace("“", '"').replace("”", '"').replace("„", '"')
    value = value.replace("«", '"').replace("»", '"').replace("`", '"')
    value = value.strip()
    while len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1].strip()
    return value


def has_pattern(value: str, patterns: list[str]) -> bool:
    lowered = value.casefold()
    return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns)


def apply_brand_replacements(value: str) -> str:
    lowered = value.casefold()
    for pattern, replacement in BRAND_REPLACEMENTS:
        if re.match(pattern, lowered, flags=re.IGNORECASE):
            return replacement
    return value


def clean_company_name(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None

    text = str(value)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\bRU[A-Z0-9]{10}\b", " ", text)
    text = re.sub(r"\b[A-Z]{2}[A-Z0-9]{10}\b", " ", text)
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"№\s*гос[. ]*\s*рег[.:]?.*$", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\([^)]*(?:ранее|бывш|isin|вып|issue|bond|обл)[^)]*\)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*\)", " ", text)
    text = strip_wrapping_quotes(text)
    text = re.sub(r"[\"'«»]", " ", text)
    text = re.sub(r"[_/]", " ", text)
    text = normalize_spaces(text)
    if not text:
        return None

    if has_pattern(text, DROP_PATTERNS) or has_pattern(text, GOVERNMENT_PATTERNS):
        return None

    cleaned = text
    for pattern in LEGAL_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\(\s*\)", " ", cleaned)
    for pattern in SECURITY_TAIL_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\(\s*\)", " ", cleaned)

    cleaned = re.sub(r"\bregs?\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bsec\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bnew\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b\d{4,}[A-ZА-ЯЁ0-9-]*\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*[-–—]\s*", " ", cleaned)
    cleaned = normalize_spaces(cleaned)
    cleaned = strip_wrapping_quotes(cleaned)
    cleaned = apply_brand_replacements(cleaned)
    cleaned = normalize_spaces(cleaned)

    if not cleaned or len(cleaned) < 2:
        return None
    if cleaned.isdigit():
        return None
    if has_pattern(cleaned, DROP_PATTERNS) or has_pattern(cleaned, GOVERNMENT_PATTERNS):
        return None
    if not re.search(r"[A-Za-zА-Яа-яЁё]", cleaned):
        return None
    return cleaned


def preferred_display_name(names: pd.Series) -> str:
    values = [str(v) for v in names.dropna().tolist() if str(v).strip()]
    if not values:
        return ""
    # Prefer shorter Russian brand-like forms, but keep English names for foreign holdings.
    values = sorted(set(values), key=lambda x: (len(x), x.casefold()))
    return values[0]


def build_company_list(y_path: Path, chunksize: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg: dict[str, dict[str, object]] = {}
    raw_rows: list[dict[str, object]] = []
    total_rows = 0

    for i, chunk in enumerate(
        pd.read_csv(
            y_path,
            sep="\t",
            encoding="utf-16",
            usecols=lambda col: col in READ_COLUMNS,
            chunksize=chunksize,
            low_memory=False,
        ),
        start=1,
    ):
        total_rows += len(chunk)
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        chunk["clean_name"] = chunk["Company"].map(clean_company_name)
        valid = chunk[chunk["clean_name"].notna()].copy()

        grouped = valid.groupby("clean_name", dropna=False)
        for clean_name, g in grouped:
            item = agg.setdefault(
                str(clean_name),
                {
                    "company_name": str(clean_name),
                    "raw_examples": set(),
                    "asset_classes": set(),
                    "asset_types": set(),
                    "row_count": 0,
                    "date_count": 0,
                    "first_date": pd.NaT,
                    "last_date": pd.NaT,
                    "sum_abs_rub": 0.0,
                    "sum_weight": 0.0,
                    "max_weight": 0.0,
                },
            )
            item["row_count"] = int(item["row_count"]) + len(g)
            item["date_count"] = int(item["date_count"]) + g["date"].nunique(dropna=True)
            first_date = g["date"].min()
            last_date = g["date"].max()
            if pd.isna(item["first_date"]) or (pd.notna(first_date) and first_date < item["first_date"]):
                item["first_date"] = first_date
            if pd.isna(item["last_date"]) or (pd.notna(last_date) and last_date > item["last_date"]):
                item["last_date"] = last_date
            item["sum_abs_rub"] = float(item["sum_abs_rub"]) + pd.to_numeric(g.get("Сумма, руб."), errors="coerce").abs().sum(skipna=True)
            item["sum_weight"] = float(item["sum_weight"]) + pd.to_numeric(g.get("Доля, %"), errors="coerce").sum(skipna=True)
            item["max_weight"] = max(float(item["max_weight"]), float(pd.to_numeric(g.get("Доля, %"), errors="coerce").max(skipna=True) or 0))
            item["raw_examples"].update(g["Company"].dropna().astype(str).drop_duplicates().head(8).tolist())
            item["asset_classes"].update(g["Класс активов"].dropna().astype(str).drop_duplicates().head(8).tolist())
            item["asset_types"].update(g["Asset_type"].dropna().astype(str).drop_duplicates().head(8).tolist())

        raw = (
            chunk.groupby(["Company", "clean_name"], dropna=False)
            .agg(
                row_count=("Company", "size"),
                first_date=("date", "min"),
                last_date=("date", "max"),
            )
            .reset_index()
        )
        raw_rows.extend(raw.to_dict("records"))
        print(
            f"chunk={i} rows={total_rows:,} valid_companies={valid['clean_name'].nunique():,} accumulated={len(agg):,}",
            flush=True,
        )

    rows = []
    for item in agg.values():
        rows.append(
            {
                "company_name": item["company_name"],
                "raw_examples": "; ".join(sorted(item["raw_examples"], key=str.casefold)[:12]),
                "asset_classes": "; ".join(sorted(item["asset_classes"], key=str.casefold)),
                "asset_types": "; ".join(sorted(item["asset_types"], key=str.casefold)[:12]),
                "row_count": item["row_count"],
                "date_count": item["date_count"],
                "first_date": item["first_date"],
                "last_date": item["last_date"],
                "sum_abs_rub": item["sum_abs_rub"],
                "sum_weight": item["sum_weight"],
                "max_weight": item["max_weight"],
            }
        )

    diagnostics = pd.DataFrame(rows)
    if not diagnostics.empty:
        diagnostics["last_date"] = pd.to_datetime(diagnostics["last_date"])
        diagnostics["first_date"] = pd.to_datetime(diagnostics["first_date"])
        diagnostics = diagnostics[
            diagnostics["asset_classes"].astype(str).str.contains(r"\b(?:stocks|bonds)\b", regex=True, na=False)
        ].copy()
        diagnostics = diagnostics.sort_values(
            ["last_date", "sum_weight", "row_count", "company_name"],
            ascending=[False, False, False, True],
        )

    raw_df = pd.DataFrame(raw_rows)
    if not raw_df.empty:
        raw_df = (
            raw_df.groupby(["Company", "clean_name"], dropna=False)
            .agg(row_count=("row_count", "sum"), first_date=("first_date", "min"), last_date=("last_date", "max"))
            .reset_index()
            .sort_values(["clean_name", "row_count"], ascending=[True, False])
        )

    return diagnostics, raw_df


def write_names(diagnostics: pd.DataFrame, output_path: Path, top_n: int | None = None) -> list[str]:
    names = diagnostics["company_name"].dropna().astype(str).drop_duplicates().tolist()
    if top_n is not None:
        names = names[:top_n]
    output_path.write_text("\n".join(names) + "\n", encoding="utf-8")
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build news-search company names from actual fund holdings.")
    parser.add_argument("--y-path", type=Path, default=DEFAULT_Y_PATH)
    parser.add_argument("--chunksize", type=int, default=150_000)
    parser.add_argument("--top-n", type=int, default=500, help="Number of names to activate for news search.")
    parser.add_argument("--activate", action="store_true", help="Overwrite config/company_names.txt with the new list.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.y_path.exists() or args.y_path.stat().st_size == 0:
        raise FileNotFoundError(f"Y holdings file is missing or empty: {args.y_path}")

    PATHS.config_dir.mkdir(parents=True, exist_ok=True)
    diagnostics, raw_df = build_company_list(args.y_path, args.chunksize)
    diagnostics.to_csv(OUTPUT_DIAGNOSTICS, index=False)
    raw_df.to_csv(OUTPUT_RAW, index=False)
    full_names = write_names(diagnostics, OUTPUT_FULL_NAMES)
    names = write_names(diagnostics, OUTPUT_NAMES, top_n=args.top_n)
    write_names(diagnostics, OUTPUT_TOP500_NAMES, top_n=500)

    if args.activate:
        if ACTIVE_NAMES.exists() and ACTIVE_NAMES.stat().st_size > 0 and not ACTIVE_BACKUP.exists():
            shutil.copy2(ACTIVE_NAMES, ACTIVE_BACKUP)
        write_names(diagnostics, ACTIVE_NAMES, top_n=args.top_n)

    print(f"source: {args.y_path}", flush=True)
    print(f"full companies after filters: {len(full_names):,}", flush=True)
    print(f"active/search companies: {len(names):,}", flush=True)
    print(f"names: {OUTPUT_NAMES}", flush=True)
    print(f"full names: {OUTPUT_FULL_NAMES}", flush=True)
    print(f"top500 names: {OUTPUT_TOP500_NAMES}", flush=True)
    print(f"diagnostics: {OUTPUT_DIAGNOSTICS}", flush=True)
    print(f"raw mapping: {OUTPUT_RAW}", flush=True)
    if args.activate:
        print(f"active names updated: {ACTIVE_NAMES}", flush=True)
        print(f"previous active backup: {ACTIVE_BACKUP}", flush=True)
    print("top 30 names:", flush=True)
    for name in names[:30]:
        print(name, flush=True)


if __name__ == "__main__":
    main()
