from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Optional

import _bootstrap  # noqa: F401

import pandas as pd

from dissertation_pipeline.paths import PATHS


DEFAULT_HOLDINGS_LONG = (
    PATHS.funds_dir
    / "investfunds_fund_holdings"
    / "processed"
    / "fund_holdings_long.csv.gz"
)
OUTPUT_DIR = PATHS.config_dir / "company_lists"
ACTIVE_NAMES = PATHS.company_names_file

ISIN_RE = re.compile(r"\b[A-Z]{2}[A-Z0-9]{10}\b")
DATE_RE = re.compile(r"\b\d{1,2}[a-z]{3}\d{4}\b", flags=re.IGNORECASE)

READ_COLUMNS = [
    "fund_id",
    "report_date",
    "table_type",
    "row_type",
    "parent_category",
    "parent_subcategory",
    "item_name",
    "registration_number_security",
    "amount_thousand_rub",
    "share_pct",
]

LEGAL_PATTERNS = [
    r"публичное акционерное общество",
    r"открытое акционерное общество",
    r"закрытое акционерное общество",
    r"акционерное общество",
    r"общество с ограниченной ответственностью",
    r"международная компания",
    r"\bмкпао\b",
    r"\bпао\b",
    r"\bоао\b",
    r"\bзао\b",
    r"\bао\b",
    r"\bооо\b",
    r"\bнко\b",
    r"\bнпао\b",
    r"\bpublic joint[- ]stock company\b",
    r"\bjoint[- ]stock company\b",
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
    r"\borderinary shares?\b",
    r"\bcommon shares?\b",
    r"\bpreferred shares?\b",
    r"\bpreference shares?\b",
    r"\bclass\s+[a-z]\b",
    r"\bord(?:\s+shs?)?\b",
    r"\bpref(?:\s+shs?)?\b",
    r"\bregs?\b",
    r"\badr\b",
    r"\bgdr\b",
    r"\bрдр\b",
    r"\bгдр\b",
    r"\bакци[ия]\b",
    r"\bакц[.]?\b",
    r"\bобыкновенн\w*\b",
    r"\bпривилегированн\w*\b",
    r"\bоблигаци\w*\b",
    r"\bбиржев\w+\s+облигаци\w*\b",
    r"\bеврооблигаци\w*\b",
    r"\bао\b",
    r"\bап\b",
    r"\bоб[.]?\b",
    r"\bб[оo]\b",
    r"\bбо\b",
    r"\bсер(?:ия|ии)?\b",
    r"\bвып(?:уск)?\b",
    r"\bissue\b",
    r"\bnote\b",
    r"\bbond\b",
]

DROP_PATTERNS = [
    r"\bi?nav\b",
    r"\betf\b",
    r"\bбпиф\b",
    r"\bзпиф\b",
    r"\bипиф\b",
    r"\bопиф\b",
    r"\bпиф\b",
    r"\bпаев\w*\b",
    r"\bпа[ий]\b",
    r"\bфонд\w*\b",
    r"\bfund\b",
    r"\btrust\b",
    r"\bspdr\b",
    r"\bindex\b",
    r"\bиндекс\b",
    r"^золото\b",
    r"\bgold\s+shares?\b",
    r"\bбиржев\w*\b",
    r"расч[её]тная цена",
    r"\bденежн\w+\s+средств\w*\b",
    r"\bден[.]?\s*средств\w*\b",
    r"\bвалют\w*\b",
    r"\bдепозит\w*\b",
    r"\bдебиторск\w+\s+задолженн\w*\b",
    r"\bкредиторск\w+\s+задолженн\w*\b",
    r"\bдрагоценн\w+\s+металл\w*\b",
    r"\bкэш\b",
    r"\bcash\b",
    r"\bитого\b",
    r"\btotal\b",
    r"\bmanagement\b",
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
    r"\bсубъект\b",
    r"\bобласть\b",
    r"\bкрай\b",
    r"\bреспублика\b",
    r"\bгород\b",
    r"\bkazakhstan\b",
    r"\bbelarus\b",
    r"\bturkey\b",
    r"\begypt\b",
    r"\bsouth africa\b",
    r"\boman\b",
    r"\bказахстан\b",
    r"\bбеларусь\b",
    r"\bтурция\b",
    r"\bегипет\b",
    r"\bюар\b",
    r"\bоман\b",
    r"\bузбекистан\b",
]

BRAND_REPLACEMENTS = [
    (r".*\bсбербанк\w*(?:\s+россии)?\b.*", "Сбербанк"),
    (r".*\bлукойл\b.*", "ЛУКОЙЛ"),
    (r".*\bгазпром\s+нефть\b.*", "Газпром нефть"),
    (r".*\bгазпром\s+газораспределение\b.*", "Газпром газораспределение"),
    (r".*\bгазпромбанк\b.*", "Газпромбанк"),
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
    (r".*\bрос\s+агро\b.*", "Русагро"),
    (r".*\brusagro\b.*", "Русагро"),
    (r".*\bкорпоративный\s+центр\s+икс\b.*", "X5 Retail Group"),
    (r".*\bсеверсталь\b.*", "Северсталь"),
    (r".*\bмагнит\b.*", "Магнит"),
    (r".*\bноватэк\b.*", "НОВАТЭК"),
    (r".*\bроснефть\b.*", "Роснефть"),
    (r".*\bтатнефть\b.*", "Татнефть"),
    (r".*\bсургутнефтегаз\b.*", "Сургутнефтегаз"),
    (r".*\bростелеком\b.*", "Ростелеком"),
    (r".*\bинтер\s*рао\b.*", "Интер РАО"),
    (r".*\bрусгидро\b.*", "РусГидро"),
    (r".*\bполюс\b.*", "Полюс"),
    (r".*\bфосагро\b.*", "ФосАгро"),
    (r".*\baeroflot\b.*", "Аэрофлот"),
    (r".*\bаэрофлот\b.*", "Аэрофлот"),
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


def extract_isin(*values: object) -> str:
    for value in values:
        if value is None or pd.isna(value):
            continue
        match = ISIN_RE.search(str(value).upper())
        if match:
            return match.group(0)
    return ""


def classify_security_scope(parent_category: object, parent_subcategory: object, item_name: object) -> str:
    text = " ".join(
        str(v)
        for v in [parent_category, parent_subcategory, item_name]
        if v is not None and not pd.isna(v)
    ).casefold()
    if re.search(r"акци|депозитарн\w+\s+расписк|adr|gdr|рдр|гдр|shares?", text):
        return "shares"
    if re.search(r"облигац|еврооблигац|\bbond\b|\bnote\b", text):
        return "bonds"
    if re.search(r"\betf\b|бпиф|зпиф|опиф|ипиф|пай|фонд", text):
        return "funds"
    return "other"


def clean_search_name(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None

    text = str(value)
    text = text.replace("\u00a0", " ")
    text = ISIN_RE.sub(" ", text.upper()) if text.isascii() else ISIN_RE.sub(" ", text)
    text = DATE_RE.sub(" ", text)
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"№\s*гос[. ]*\s*рег[.:]?.*$", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\([^)]*(?:isin|вып|issue|bond|обл|офз|note)[^)]*\)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*\)", " ", text)
    text = re.sub(r"[\"'«»]", " ", strip_wrapping_quotes(text))
    text = re.sub(r"[_/]", " ", text)
    text = re.sub(r"\b\d+(?:[.,]\d+)?\s*%\b", " ", text)
    text = re.sub(r"\b\d{1,2}[a-z]{3}\d{4}\b", " ", text, flags=re.IGNORECASE)
    text = normalize_spaces(text)
    if not text:
        return None

    if has_pattern(text, DROP_PATTERNS) or has_pattern(text, GOVERNMENT_PATTERNS):
        return None

    cleaned = re.split(r"\s+-\s+", text, maxsplit=1)[0]
    # Most security rows are issuer first, issue/type after comma.
    parts = [normalize_spaces(part) for part in cleaned.split(",") if normalize_spaces(part)]
    if parts:
        cleaned = parts[0]

    for pattern in LEGAL_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    for pattern in SECURITY_TAIL_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\b\d{2,}[A-ZА-ЯЁ0-9-]*\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\(\s*\)", " ", cleaned)
    cleaned = re.sub(r"\s+[-–—]\s+", " ", cleaned)
    cleaned = normalize_spaces(strip_wrapping_quotes(cleaned))
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


def join_limited(values: pd.Series, limit: int = 10) -> str:
    unique = []
    seen = set()
    for value in values.dropna().astype(str):
        value = normalize_spaces(value)
        if not value or value.casefold() in seen:
            continue
        seen.add(value.casefold())
        unique.append(value)
        if len(unique) >= limit:
            break
    return "; ".join(unique)


def build_tables(holdings_path: Path, chunksize: int, sample_rows: Optional[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_parts: list[pd.DataFrame] = []
    total_rows = 0
    total_items = 0

    reader = pd.read_csv(
        holdings_path,
        compression="gzip",
        usecols=lambda col: col in READ_COLUMNS,
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk_id, chunk in enumerate(reader, start=1):
        if sample_rows is not None:
            remaining = sample_rows - total_rows
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)

        total_rows += len(chunk)
        chunk = chunk[(chunk["table_type"] == "asset") & (chunk["row_type"] == "asset_item")].copy()
        if chunk.empty:
            print(f"chunk={chunk_id} rows={total_rows:,} asset_items=0 kept=0", flush=True)
            continue

        total_items += len(chunk)
        chunk["report_date"] = pd.to_datetime(chunk["report_date"], errors="coerce")
        chunk["security_code"] = chunk["registration_number_security"].fillna("").astype(str).map(normalize_spaces)
        chunk["isin"] = [
            extract_isin(sec, raw)
            for sec, raw in zip(chunk["security_code"], chunk["item_name"])
        ]
        chunk["security_scope"] = [
            classify_security_scope(cat, subcat, name)
            for cat, subcat, name in zip(
                chunk["parent_category"],
                chunk["parent_subcategory"],
                chunk["item_name"],
            )
        ]
        chunk["news_search_name"] = chunk["item_name"].map(clean_search_name)
        valid = chunk[chunk["news_search_name"].notna()].copy()
        if valid.empty:
            print(f"chunk={chunk_id} rows={total_rows:,} asset_items={total_items:,} kept=0", flush=True)
            continue

        valid["amount_thousand_rub"] = pd.to_numeric(valid["amount_thousand_rub"], errors="coerce")
        valid["share_pct"] = pd.to_numeric(valid["share_pct"], errors="coerce")
        group_cols = ["news_search_name", "isin", "security_code", "security_scope"]
        detail = (
            valid.groupby(group_cols, dropna=False)
            .agg(
                row_count=("item_name", "size"),
                fund_count=("fund_id", "nunique"),
                date_count=("report_date", "nunique"),
                first_date=("report_date", "min"),
                last_date=("report_date", "max"),
                sum_amount_thousand_rub=("amount_thousand_rub", "sum"),
                sum_share_pct=("share_pct", "sum"),
                max_share_pct=("share_pct", "max"),
                parent_categories=("parent_category", join_limited),
                raw_item_examples=("item_name", join_limited),
            )
            .reset_index()
        )
        detail_parts.append(detail)
        print(
            f"chunk={chunk_id} rows={total_rows:,} asset_items={total_items:,} "
            f"kept_rows={len(valid):,} groups={len(detail):,}",
            flush=True,
        )

        if sample_rows is not None and total_rows >= sample_rows:
            break

    if not detail_parts:
        return pd.DataFrame(), pd.DataFrame()

    detail_all = pd.concat(detail_parts, ignore_index=True)
    detail = (
        detail_all.groupby(["news_search_name", "isin", "security_code", "security_scope"], dropna=False)
        .agg(
            row_count=("row_count", "sum"),
            fund_count=("fund_count", "max"),
            date_count=("date_count", "max"),
            first_date=("first_date", "min"),
            last_date=("last_date", "max"),
            sum_amount_thousand_rub=("sum_amount_thousand_rub", "sum"),
            sum_share_pct=("sum_share_pct", "sum"),
            max_share_pct=("max_share_pct", "max"),
            parent_categories=("parent_categories", join_limited),
            raw_item_examples=("raw_item_examples", join_limited),
        )
        .reset_index()
    )
    detail = detail.sort_values(
        ["security_scope", "last_date", "sum_share_pct", "row_count", "news_search_name"],
        ascending=[True, False, False, False, True],
    )

    by_name = (
        detail.groupby("news_search_name", dropna=False)
        .agg(
            scopes=("security_scope", lambda s: "; ".join(sorted(set(s.dropna().astype(str))))),
            isin_codes=("isin", lambda s: "; ".join(sorted({v for v in s.dropna().astype(str) if v}))),
            security_codes=("security_code", lambda s: "; ".join(sorted({v for v in s.dropna().astype(str) if v})[:20])),
            row_count=("row_count", "sum"),
            fund_count=("fund_count", "max"),
            date_count=("date_count", "max"),
            first_date=("first_date", "min"),
            last_date=("last_date", "max"),
            sum_amount_thousand_rub=("sum_amount_thousand_rub", "sum"),
            sum_share_pct=("sum_share_pct", "sum"),
            max_share_pct=("max_share_pct", "max"),
            raw_item_examples=("raw_item_examples", join_limited),
        )
        .reset_index()
    )
    by_name["is_share_name"] = by_name["scopes"].str.contains("shares", na=False)
    by_name = by_name.sort_values(
        ["is_share_name", "last_date", "sum_share_pct", "row_count", "news_search_name"],
        ascending=[False, False, False, False, True],
    )
    return detail, by_name


def write_names(names: list[str], output_path: Path) -> None:
    output_path.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")


def archive_old_company_lists(current_outputs: set[Path]) -> Path | None:
    candidates = [
        path
        for path in PATHS.config_dir.iterdir()
        if path.is_file()
        and path.name.startswith("company_")
        and path.name != ACTIVE_NAMES.name
        and path.resolve() not in {p.resolve() for p in current_outputs}
    ]
    if not candidates:
        return None

    archive_dir = PATHS.config_dir / "company_lists" / "archive_previous"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for path in candidates:
        target = archive_dir / path.name
        if target.exists():
            target = archive_dir / f"{path.stem}.bak{path.suffix}"
        shutil.move(str(path), str(target))
    return archive_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build news-search company names and ISIN mapping from combined InvestFunds holdings."
    )
    parser.add_argument("--holdings-path", type=Path, default=DEFAULT_HOLDINGS_LONG)
    parser.add_argument("--chunksize", type=int, default=150_000)
    parser.add_argument("--sample-rows", type=int, default=None, help="Read only N source rows for a quick smoke test.")
    parser.add_argument("--top-n", type=int, default=500, help="Number of names to activate in config/company_names.txt.")
    parser.add_argument(
        "--active-scope",
        choices=["shares", "all"],
        default="shares",
        help="Use only share/DR names or all cleaned corporate security names for active news search.",
    )
    parser.add_argument("--activate", action="store_true", help="Overwrite config/company_names.txt.")
    parser.add_argument("--archive-old", action="store_true", help="Move old company-list files to config/company_lists/archive_previous.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.holdings_path.exists() or args.holdings_path.stat().st_size == 0:
        raise FileNotFoundError(f"Holdings dataset is missing or empty: {args.holdings_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detail, by_name = build_tables(args.holdings_path, args.chunksize, args.sample_rows)
    if detail.empty or by_name.empty:
        raise RuntimeError("No company names were extracted from holdings dataset.")

    detail_path = OUTPUT_DIR / "holdings_security_name_isin_mapping.csv"
    names_path = OUTPUT_DIR / "holdings_company_names.csv"
    full_names_path = OUTPUT_DIR / "holdings_company_names_full.txt"
    top_names_path = OUTPUT_DIR / f"holdings_company_names_top{args.top_n}.txt"
    share_names_path = OUTPUT_DIR / "holdings_company_names_shares_full.txt"

    detail.to_csv(detail_path, index=False)
    by_name.to_csv(names_path, index=False)

    full_names = by_name["news_search_name"].dropna().astype(str).drop_duplicates().tolist()
    share_names = (
        by_name.loc[by_name["is_share_name"], "news_search_name"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    active_source = share_names if args.active_scope == "shares" else full_names
    active_names = active_source[: args.top_n]

    write_names(full_names, full_names_path)
    write_names(share_names, share_names_path)
    write_names(active_names, top_names_path)

    current_outputs = {detail_path, names_path, full_names_path, top_names_path, share_names_path}
    backup_path = None
    if args.activate:
        if ACTIVE_NAMES.exists() and ACTIVE_NAMES.stat().st_size > 0:
            backup_path = OUTPUT_DIR / "company_names_before_holdings_dataset.txt"
            shutil.copy2(ACTIVE_NAMES, backup_path)
            current_outputs.add(backup_path)
        write_names(active_names, ACTIVE_NAMES)

    archive_dir = archive_old_company_lists(current_outputs) if args.archive_old else None

    print(f"source: {args.holdings_path}", flush=True)
    print(f"security rows after cleaning: {len(detail):,}", flush=True)
    print(f"unique search names all scopes: {len(full_names):,}", flush=True)
    print(f"unique share/DR search names: {len(share_names):,}", flush=True)
    print(f"active scope: {args.active_scope}", flush=True)
    print(f"active names written: {len(active_names):,}", flush=True)
    print(f"mapping table: {detail_path}", flush=True)
    print(f"name diagnostics: {names_path}", flush=True)
    print(f"active list copy: {top_names_path}", flush=True)
    print(f"full all names: {full_names_path}", flush=True)
    print(f"full share names: {share_names_path}", flush=True)
    if args.activate:
        print(f"active config updated: {ACTIVE_NAMES}", flush=True)
        print(f"active config backup: {backup_path}", flush=True)
    if archive_dir:
        print(f"old company-list files archived: {archive_dir}", flush=True)
    print("top 40 active names:", flush=True)
    for name in active_names[:40]:
        print(name, flush=True)


if __name__ == "__main__":
    main()
