from __future__ import annotations

import argparse
import re
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

from dissertation_pipeline.paths import PATHS


DEFAULT_HOLDINGS_LONG = (
    PATHS.funds_dir
    / "investfunds_fund_holdings"
    / "processed"
    / "fund_holdings_long.csv"
)
DEFAULT_SHARES_INFO = PATHS.funds_dir / "current" / "Shares_info.csv"
OUTPUT_DIR = PATHS.config_dir / "company_lists"

ISIN_RE = re.compile(r"\b[A-Z]{2}[A-Z0-9]{10}\b")

READ_COLUMNS = [
    "fund_id",
    "report_date",
    "item_name",
    "registration_number_security",
    "amount_thousand_rub",
]

LEGAL_PATTERNS = [
    r"публичное акционерное общество",
    r"открытое акционерное общество",
    r"закрытое акционерное общество",
    r"акционерное общество",
    r"общество с ограниченной ответственностью",
    r"\bпао\b",
    r"\bоао\b",
    r"\bзао\b",
    r"\bооо\b",
    r"\bао\b",
    r"\bнко\b",
    r"\bмкпао\b",
    r"\bнпао\b",
]

GENERIC_EXACT = {
    "",
    "россия",
    "российская федерация",
    "аргентина",
    "беларусь",
    "казахстан",
    "республика казахстан",
    "турция",
    "египет",
    "узбекистан",
    "азербайджан",
    "республика беларусь",
    "практика",
    "ликвидность",
    "регент",
    "операторская компания",
    "партнерские инвестиции",
    "независимые инвестиции",
    "капитальные инвестиции",
    "системное развитие",
    "созидание и развитие",
    "развитие",
    "инвестиции",
    "финансы",
    "финанс",
    "капитал",
    "гарант",
    "перспектива",
    "инвест проект",
    "инвест-проект",
    "инвестиционный альянс",
    "мв финанс",
    "ритейл бел финанс",
    "синх-финанс",
    "ктж финанс",
    "рвк-финанс",
    "боржоми",
    "боржоми финанс",
    "казахстанские коммунальные системы",
    "евразийский банк развития",
}

GENERIC_PATTERNS = [
    r"\bофз\b",
    r"\bфедеральн\w+\s+за[её]м",
    r"\bминфин\b",
    r"\bминистерство\b",
    r"\bдепартамент\b",
    r"\bказначейств",
    r"\bправительств",
    r"\bмуниципальн",
    r"\bобласть\b",
    r"\bкрай\b",
    r"\bреспублика\b",
    r"\bиностранн\w+\s+государств",
    r"\bгород\b",
    r"\bсубъект\b",
    r"\bденежн\w+\s+средств",
    r"\bдепозит",
    r"\bдебиторск\w+\s+задолж",
    r"\bкредиторск\w+\s+задолж",
    r"\bвалют",
    r"\bпаев",
    r"\bфонд",
    r"\bбпиф\b",
    r"\bпиф\b",
    r"\betf\b",
    r"\bиндекс\b",
    r"^золото\b",
    r"\bсозидание\s+и\s+развитие\b",
]

BRAND_RULES = [
    (r".*\bсбербанк(?:\s+россии)?\b.*", "Сбербанк"),
    (r".*\bлукойл\b.*", "ЛУКОЙЛ"),
    (r".*\bгазпром\s+нефть\b.*", "Газпром нефть"),
    (r".*\bгазпром\s+капитал\b.*", "Газпром"),
    (r".*\bгазпромбанк\b.*", "Газпромбанк"),
    (r".*\bгпб\s+финанс\b.*", "Газпромбанк"),
    (r".*\bгазпром\b.*", "Газпром"),
    (r".*\bнорильский\s+никель\b.*", "Норильский никель"),
    (r".*\bгмк\s+норильский\s+никель\b.*", "Норильский никель"),
    (r".*\bроснефть\b.*", "Роснефть"),
    (r".*\bтатнефть\b.*", "Татнефть"),
    (r".*\bноватэк\b.*", "НОВАТЭК"),
    (r".*\bсургутнефтегаз\b.*", "Сургутнефтегаз"),
    (r".*\bафк\s+система\b.*", "АФК Система"),
    (r".*\bяндекс\b.*", "Яндекс"),
    (r".*\bмкпао\s+яндекс\b.*", "Яндекс"),
    (r".*\bсфо\s+сплит\s+финанс\b.*яндекс.*", "Яндекс"),
    (r".*\bозон\b.*", "Озон"),
    (r".*\bмкпао\s+озон\b.*", "Озон"),
    (r".*\bхэдхантер\b.*", "Хэдхантер"),
    (r".*\bмкпао\s+хэдхантер\b.*", "Хэдхантер"),
    (r".*\bт[-\s]?технологи\w*\b.*", "Т-Технологии"),
    (r".*\bт[-\s]?финанс\b.*", "Т-Технологии"),
    (r".*\bт\s*банк\b.*", "Т-Банк"),
    (r".*\bтинькофф\b.*", "Т-Технологии"),
    (r".*\bткс\s+холдинг\b.*", "Т-Технологии"),
    (r".*\bбанк\s+втб\b.*", "ВТБ"),
    (r".*\bвтб\b.*", "ВТБ"),
    (r".*\bальфа[-\s]?банк\b.*", "Альфа-Банк"),
    (r".*\bроссельхозбанк\b.*", "Россельхозбанк"),
    (r".*\bмосковский\s+кредитный\s+банк\b.*", "Московский Кредитный банк"),
    (r".*\bмкб\b.*", "Московский Кредитный банк"),
    (r".*\bбанк\s+санкт[-\s]?петербург\b.*", "Банк Санкт-Петербург"),
    (r".*\bсовкомбанк\b.*", "Совкомбанк"),
    (r".*\bхкф\s+банк\b.*", "ХКФ Банк"),
    (r".*\bавто\s+финанс\s+банк\b.*", "Авто Финанс Банк"),
    (r".*\bросэксимбанк\b.*", "Росэксимбанк"),
    (r".*\bрусал\b.*", "РУСАЛ"),
    (r".*\bалроса\b.*", "АЛРОСА"),
    (r".*\bсеверсталь\b.*", "Северсталь"),
    (r".*\bмагнит\b.*", "Магнит"),
    (r".*\bмтс\b.*", "МТС"),
    (r".*\bмобильные\s+телесистемы\b.*", "МТС"),
    (r".*\bмегафон\b.*", "МегаФон"),
    (r".*\bмэйл\.?\s*ру\s+финанс\b.*", "ВК"),
    (r".*\bвк\b.*", "ВК"),
    (r".*\bнлмк\b.*", "НЛМК"),
    (r".*\bммк\b.*", "ММК"),
    (r".*\bполюс\b.*", "Полюс"),
    (r".*\bрусгидро\b.*", "РусГидро"),
    (r".*\bинтер\s*рао\b.*", "Интер РАО"),
    (r".*\bроссети\b.*", "Россети"),
    (r".*\bфск\s+еэс\b.*", "ФСК ЕЭС"),
    (r".*\bаэрофлот\b.*", "Аэрофлот"),
    (r".*\bфосагро\b.*", "ФосАгро"),
    (r".*\bростелеком\b.*", "Ростелеком"),
    (r".*\bтранснефть\b.*", "Транснефть"),
    (r".*\bмосковская\s+биржа\b.*", "Московская Биржа"),
    (r".*\bмосбиржа\b.*", "Московская Биржа"),
    (r".*\bевраз\b.*", "Евраз"),
    (r".*\bевразхолдинг\s+финанс\b.*", "Евраз"),
    (r".*\bикс\s*5\b.*", "Икс 5"),
    (r".*\bкорпоративный\s+центр\s+икс\s*5\b.*", "Икс 5"),
    (r".*\bгруппа\s+компаний\s+самолет\b.*", "Самолет"),
    (r".*\bсамолет\b.*", "Самолет"),
    (r".*\bгруппа\s+лср\b.*", "Группа ЛСР"),
    (r".*\bлср\b.*", "Группа ЛСР"),
    (r".*\bсэтл\s+групп\b.*", "Сэтл Групп"),
    (r".*\bпик[-\s]?корпорация\b.*", "ПИК"),
    (r".*\bпик\b.*", "ПИК"),
    (r".*\bэталон\s+финанс\b.*", "Эталон"),
    (r".*\bлидер[-\s]?инвест\b.*", "Эталон"),
    (r".*\bэталон\b.*", "Эталон"),
    (r".*\bсегежа\b.*", "Сегежа Групп"),
    (r".*\bчеркизово\b.*", "Группа Черкизово"),
    (r".*\bсибур\b.*", "СИБУР"),
    (r".*\bсуэк[-\s]?финанс\b.*", "СУЭК"),
    (r".*\bсуэк\b.*", "СУЭК"),
    (r".*\bметаллоинвест\b.*", "Металлоинвест"),
    (r".*\bхк\s+металлоинвест\b.*", "Металлоинвест"),
    (r".*\bкамаз\b.*", "КАМАЗ"),
    (r".*\bюжуралзолото\b.*", "Южуралзолото"),
    (r".*\bсовкомфлот\b.*", "Совкомфлот"),
    (r".*\bпочта\s+россии\b.*", "Почта России"),
    (r".*\bржд\b.*", "РЖД"),
    (r".*\bгтлк\b.*", "ГТЛК"),
    (r".*\bвэб\.?рф\b.*", "ВЭБ.РФ"),
    (r".*\bвэб[-\s]?лизинг\b.*", "ВЭБ.РФ"),
    (r".*\bавтодор\b.*", "Автодор"),
    (r".*\bдом\.?рф\b.*", "ДОМ.РФ"),
    (r".*\bо\s*кей\s+финанс\b.*", "ОКЕЙ"),
    (r".*\bокей\s+финанс\b.*", "ОКЕЙ"),
    (r".*\bо`кей\b.*", "ОКЕЙ"),
    (r".*\bо'?кей\b.*", "ОКЕЙ"),
    (r".*\bборец\s+капитал\b.*", "Группа Борец"),
    (r".*\bрвк[-\s]?инвест\b.*", "РВК"),
    (r".*\bмираторг\s+финанс\b.*", "Мираторг"),
    (r".*\bвис\s+финанс\b.*", "ВИС"),
    (r".*\bо1\s+пропертиз\s+финанс\b.*", "О1 Пропертиз"),
    (r".*\bгсп[-\s]?финанс\b.*", "ГСП"),
    (r".*\bкиви\s+финанс\b.*", "КИВИ"),
    (r".*\bрсг[-\s]?финанс\b.*", "РСГ"),
    (r".*\bинк[-\s]?капитал\b.*", "ИНК"),
    (r".*\bавтобан[-\s]?финанс\b.*", "Автобан"),
    (r".*\bкузбассэнерго[-\s]?финанс\b.*", "Кузбассэнерго"),
    (r".*\bвымпелком[-\s]?инвест\b.*", "Вымпелком"),
    (r".*\bрмк[-\s]?финанс\b.*", "РМК"),
    (r".*\bютэйр[-\s]?финанс\b.*", "ЮТэйр"),
    (r".*\bпрофмедиа\s+финанс\b.*", "ПрофМедиа"),
    (r".*\bсоллерс[-\s]?финанс\b.*", "Соллерс"),
    (r".*\bдикси\s*[-]?\s*финанс\b.*", "Дикси"),
    (r".*\bмагнит\s+финанс\b.*", "Магнит"),
    (r".*\bпромтрактор[-\s]?финанс\b.*", "Промтрактор"),
    (r".*\bлизинговая\s+компания\s+уралсиб\b.*", "УРАЛСИБ"),
    (r".*\bцентр[-\s]?инвест\b.*", "Центр-инвест"),
    (r".*\bренессанс\s+капитал\b.*", "Ренессанс Капитал"),
    (r".*\bфпк\s+гарант[-\s]?инвест\b.*", "Гарант-Инвест"),
    (r".*\bмд\s+медикал\s+груп\b.*", "Мать и дитя"),
    (r".*\bтрубная\s+металлургическая\s+компания\b.*", "ТМК"),
    (r".*\bпервая\s+грузовая\s+компания\b.*", "ПГК"),
]


def normalize_spaces(value: str) -> str:
    value = value.replace("\u00a0", " ")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s+([,.;:])", r"\1", value)
    return value.strip(" \t\r\n,.;:-")


def clean_base_name(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"\([^)]*(?:пао|ао|оао|зао|ооо|мкпао|нпао)[^)]*\)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*\)", " ", text)
    text = text.replace("«", " ").replace("»", " ").replace('"', " ")
    text = normalize_spaces(text)
    for pattern in LEGAL_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    text = normalize_spaces(text)
    return text


def optimize_name(initial_name: object) -> tuple[str, str]:
    base = clean_base_name(initial_name)
    lowered = base.casefold()

    if lowered in GENERIC_EXACT:
        return "", "generic_name"
    utility_match = re.search(r"\b(тгк|огк)[-\s]*(\d+)\b", lowered, flags=re.IGNORECASE)
    if utility_match:
        return f"{utility_match.group(1).upper()}-{utility_match.group(2)}", "include"
    if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in GENERIC_PATTERNS):
        return "", "government_or_non_company"

    for pattern, replacement in BRAND_RULES:
        if re.match(pattern, lowered, flags=re.IGNORECASE):
            return replacement, "include"

    text = base
    text = re.sub(r"\b(облигаци\w*|акци\w*|биржев\w+|выпуск|серия)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d{2,}[A-ZА-ЯЁ0-9-]*\b", " ", text, flags=re.IGNORECASE)
    text = normalize_spaces(text)

    if not text or text.casefold() in GENERIC_EXACT:
        return "", "generic_name"
    if len(text) <= 2 and not text.isupper():
        return "", "too_short_or_ambiguous"
    if len(text.split()) == 1 and text.casefold() in {"практика", "регент", "ликвидность"}:
        return "", "generic_name"
    return text, "include"


def extract_isins(values: object) -> str:
    values_text = "" if values is None or pd.isna(values) else str(values)
    return "; ".join(sorted(set(ISIN_RE.findall(values_text.upper()))))


def load_moex_share_isin_map(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["optimized_name", "moex_share_isin_candidates"])

    shares = pd.read_csv(
        path,
        encoding="utf-16",
        sep="\t",
        usecols=lambda col: col in {"SHORTNAME", "SECID", "ISIN"},
        low_memory=False,
    )
    shares = shares.dropna(subset=["SHORTNAME", "ISIN"]).copy()
    shares["optimized_name"] = shares["SHORTNAME"].map(lambda x: optimize_name(x)[0])
    shares = shares[shares["optimized_name"].ne("")]
    shares["ISIN"] = shares["ISIN"].astype(str).str.strip()
    out = (
        shares.groupby("optimized_name", as_index=False)
        .agg(moex_share_isin_candidates=("ISIN", lambda s: "; ".join(sorted(set(s.dropna().astype(str)))[:10])))
    )
    return out


def build_top500(holdings_path: Path, top_n: int) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(
        holdings_path,
        usecols=lambda col: col in READ_COLUMNS,
        chunksize=250_000,
        low_memory=False,
    ):
        chunk = chunk.loc[chunk["registration_number_security"].notna()].copy()
        if chunk.empty:
            continue
        chunk["initial_name"] = chunk["item_name"].astype(str).str.split(",").str[0].str.strip()
        reg = chunk["registration_number_security"].astype(str).str.strip().str.upper()
        is_us_isin = reg.str.startswith("US", na=False)
        has_latin_name = chunk["initial_name"].str.contains(r"[A-Za-z]", regex=True, na=False)
        chunk = chunk.loc[~is_us_isin & ~has_latin_name].copy()
        if chunk.empty:
            continue
        chunk["amount_thousand_rub"] = pd.to_numeric(chunk["amount_thousand_rub"], errors="coerce").fillna(0)
        parts.append(chunk)

    if not parts:
        raise RuntimeError("No rows after initial Russian-name/security filters.")

    data = pd.concat(parts, ignore_index=True)
    agg = (
        data.groupby("initial_name", as_index=False)
        .agg(
            total_amount_thousand_rub=("amount_thousand_rub", "sum"),
            rows_count=("initial_name", "size"),
            fund_count=("fund_id", "nunique"),
            date_count=("report_date", "nunique"),
            first_date=("report_date", "min"),
            last_date=("report_date", "max"),
            registration_numbers=(
                "registration_number_security",
                lambda s: "; ".join(sorted(set(s.dropna().astype(str)))[:30]),
            ),
        )
        .sort_values("total_amount_thousand_rub", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    agg.insert(0, "rank_by_total_amount", range(1, len(agg) + 1))
    agg["isin_codes"] = agg["registration_numbers"].map(extract_isins)
    cleaned = agg["initial_name"].map(optimize_name)
    agg["optimized_name"] = [item[0] for item in cleaned]
    agg["decision"] = [item[1] for item in cleaned]
    agg["include_in_news_search"] = agg["decision"].eq("include") & agg["optimized_name"].ne("")
    agg["total_amount_rub"] = agg["total_amount_thousand_rub"] * 1000
    return agg


def write_unique_names(review: pd.DataFrame, output_path: Path) -> list[str]:
    included = review[review["include_in_news_search"]].copy()
    included = included.sort_values(["total_amount_thousand_rub", "rows_count"], ascending=False)
    names = included["optimized_name"].dropna().astype(str).drop_duplicates().tolist()
    output_path.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build manually reviewable news-search securities names from dff top-N holdings."
    )
    parser.add_argument("--holdings-path", type=Path, default=DEFAULT_HOLDINGS_LONG)
    parser.add_argument("--shares-info-path", type=Path, default=DEFAULT_SHARES_INFO)
    parser.add_argument("--top-n", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.holdings_path.exists():
        raise FileNotFoundError(args.holdings_path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    review = build_top500(args.holdings_path, args.top_n)
    moex_map = load_moex_share_isin_map(args.shares_info_path)
    review = review.merge(moex_map, on="optimized_name", how="left")
    review["moex_share_isin_candidates"] = review["moex_share_isin_candidates"].fillna("")

    cols = [
        "rank_by_total_amount",
        "include_in_news_search",
        "decision",
        "initial_name",
        "optimized_name",
        "isin_codes",
        "moex_share_isin_candidates",
        "registration_numbers",
        "total_amount_thousand_rub",
        "total_amount_rub",
        "rows_count",
        "fund_count",
        "date_count",
        "first_date",
        "last_date",
    ]
    review = review[cols]

    review_path = OUTPUT_DIR / "top500_securities_news_search_review.csv"
    included_path = OUTPUT_DIR / "top500_securities_news_search_names.txt"
    excluded_path = OUTPUT_DIR / "top500_securities_news_search_excluded.csv"

    review.to_csv(review_path, index=False)
    review.loc[~review["include_in_news_search"]].to_csv(excluded_path, index=False)
    names = write_unique_names(review, included_path)

    print(f"source: {args.holdings_path}", flush=True)
    print(f"top_n raw names: {len(review):,}", flush=True)
    print(f"included raw rows: {int(review['include_in_news_search'].sum()):,}", flush=True)
    print(f"unique optimized names: {len(names):,}", flush=True)
    print(f"review: {review_path}", flush=True)
    print(f"included names: {included_path}", flush=True)
    print(f"excluded review: {excluded_path}", flush=True)
    print("top 40 optimized names:", flush=True)
    for name in names[:40]:
        print(name, flush=True)
    print("excluded top rows:", flush=True)
    print(
        review.loc[~review["include_in_news_search"], ["rank_by_total_amount", "initial_name", "decision"]]
        .head(30)
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
