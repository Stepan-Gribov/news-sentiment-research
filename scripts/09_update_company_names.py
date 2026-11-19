from __future__ import annotations

import csv
import re
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd
import requests

from dissertation_pipeline.paths import PATHS


MOEX_GLOBAL_SECURITIES_URL = "https://iss.moex.com/iss/securities.json"
MOEX_SHARES_COLLECTION_URL = (
    "https://iss.moex.com/iss/securitygroups/stock_shares/"
    "collections/stock_shares_all/securities.json"
)

OUTPUT_DIAGNOSTICS = PATHS.config_dir / "company_names_sources.csv"
OUTPUT_ALL_CANDIDATES = PATHS.config_dir / "company_names_candidates.csv"
OUTPUT_ALL_NAMES = PATHS.config_dir / "company_names_moex_all.txt"
OUTPUT_SHARE_NAMES = PATHS.config_dir / "company_names_shares.txt"
OUTPUT_SHARE_FULL_NAMES = PATHS.config_dir / "company_names_shares_full.txt"
OUTPUT_SHARE_TOP_NAMES = PATHS.config_dir / "company_names_shares_top500.txt"
OUTPUT_BOND_NAMES = PATHS.config_dir / "company_names_bonds.txt"
OUTPUT_SHARE_ISSUERS = PATHS.config_dir / "company_issuers_shares.csv"
OUTPUT_SHARE_TOP_ISSUERS = PATHS.config_dir / "company_issuers_shares_top500.csv"
OUTPUT_BOND_ISSUERS = PATHS.config_dir / "company_issuers_bonds.csv"
BACKUP_PATH = PATHS.config_dir / "company_names_previous.txt"

LEGACY_NAME_FILES = [
    PATHS.project_root / "legacy_archive" / "data_loading" / "rus_stocks_names.txt",
    PATHS.project_root / "legacy_archive" / "data_loading" / "names.csv",
]

REQUEST_TIMEOUT = 30
REQUEST_RETRIES = 5
PAGE_SLEEP_SECONDS = 0.15
SHARE_TOP_N = 500

ALLOWED_SHARE_TYPES = {
    "common_share",
    "preferred_share",
    "depositary_receipt",
    "russian_depositary_receipt",
}

ALLOWED_BOND_TYPES = {
    "corporate_bond",
    "exchange_bond",
    "euro_bond",
}

ISSUER_SOURCE_COLUMNS = ["emitent_title", "name"]

STOP_WORD_PATTERNS = [
    r"\bi?nav\b",
    r"\betf\b",
    r"\bбпиф\b",
    r"\bзпиф\b",
    r"\bипиф\b",
    r"\bпиф\b",
    r"расчетная цена",
    r"расч[её]тная цена",
    r"управляющ\w+ компани\w+",
    r"\bфонд\b",
    r"\bпай\b",
    r"\bитого\b",
    r"\btotal\b",
    r"обыкновенн\w+ акци\w+",
    r"прост\w+ акци\w+",
    r"акци\w+ обыкновенн\w+",
    r"привилегированн\w+ акци\w+",
]

LEGAL_PATTERNS = [
    r"публичное акционерное общество",
    r"открытое акционерное общество",
    r"закрытое акционерное общество",
    r"акционерное общество",
    r"общество с ограниченной ответственностью",
    r"публичная компания",
    r"\bпао\b",
    r"\bоао\b",
    r"\bзао\b",
    r"\bао\b",
    r"\bооо\b",
    r"\bнко\b",
    r"\bpublic joint stock company\b",
    r"\bpublic joint-stock company\b",
    r"\bjoint stock company\b",
    r"\bjoint-stock company\b",
    r"\bpjsc\b",
    r"\bojsc\b",
    r"\bcjsc\b",
    r"\bjsc\b",
    r"\bllc\b",
    r"\boao\b",
    r"\bzao\b",
    r"\bao\b",
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
    r"\bpreferred shares?\b",
    r"\bords?\b",
    r"\bprefs?\b",
    r"\badr\b",
    r"\bgdr\b",
    r"\bрдр\b",
    r"\bгдр\b",
    r"\bао\b",
    r"\bап\b",
    r"\bап[.]?\b",
    r"\bао[.]?\b",
    r"\bоб\.\b",
    r"\bоб\b",
]


@dataclass(frozen=True)
class NameCandidate:
    name: str
    source: str
    secid: str | None = None
    raw_value: str | None = None


def request_json(session: requests.Session, url: str, params: dict) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            wait = min(2**attempt, 30)
            print(f"request failed attempt={attempt}/{REQUEST_RETRIES} wait={wait}s url={url}: {exc}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"MOEX request failed after retries: {url}") from last_error


def fetch_moex_table(url: str, params: dict, block: str = "securities") -> pd.DataFrame:
    rows: list[list] = []
    columns: list[str] | None = None
    start = 0
    session = requests.Session()

    total: int | None = None
    page_size: int | None = None

    while True:
        page_params = {**params, "iss.meta": "off", "start": start}
        payload = request_json(session, url, page_params)
        section = payload[block]
        if columns is None:
            columns = section["columns"]
        data = section["data"]
        print(f"MOEX {url.rsplit('/iss/', 1)[-1]} start={start} rows={len(data)}", flush=True)
        if not data:
            break
        rows.extend(data)

        cursor = payload.get(f"{block}.cursor")
        if cursor and cursor.get("data"):
            cursor_columns = cursor["columns"]
            cursor_row = dict(zip(cursor_columns, cursor["data"][0]))
            total = int(cursor_row.get("TOTAL") or cursor_row.get("total") or 0)
            page_size = int(cursor_row.get("PAGESIZE") or cursor_row.get("pagesize") or len(data))

        start += len(data)
        time.sleep(PAGE_SLEEP_SECONDS)
        if total is not None and start >= total:
            break
        if total is None and page_size is None and len(data) < 100:
            break

    return pd.DataFrame(rows, columns=columns or [])


def read_legacy_names(path: Path) -> list[str]:
    if not path.exists():
        return []
    encodings = ["utf-8-sig", "utf-16", "cp1251"]
    for encoding in encodings:
        try:
            text = path.read_text(encoding=encoding)
            return [line.strip() for line in text.splitlines() if line.strip()]
        except UnicodeError:
            continue
    print(f"skip unreadable legacy names file: {path}", flush=True)
    return []


def normalize_spaces(value: str) -> str:
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s+([,.;:])", r"\1", value)
    return value.strip(" \t\r\n,.;:-")


def strip_wrapping_quotes(value: str) -> str:
    value = value.replace("“", '"').replace("”", '"').replace("„", '"')
    value = value.replace("«", '"').replace("»", '"').replace("`", '"')
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1]
    return value.strip()


def clean_company_name(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"^[+]+", "", text)
    text = re.sub(r"^i(?=[А-ЯЁа-яё])", "", text)
    text = re.sub(r"^[Aa](?=кционерное\s+общество)", "А", text)
    text = re.sub(r"^акции\s+пр[.]?\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^акц[.]?\s*пр[.]?\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^ап\s+", "", text, flags=re.IGNORECASE)
    text = strip_wrapping_quotes(text)
    text = re.sub(r"[()]", " ", text)
    text = re.sub(r"[\"'«»]", " ", text)
    text = normalize_spaces(text)
    if not text:
        return None

    lowered = text.casefold()
    for pattern in STOP_WORD_PATTERNS:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            return None

    cleaned = text
    for pattern in LEGAL_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    for pattern in SECURITY_TAIL_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s*[-–—]\s*", " ", cleaned)
    cleaned = re.sub(r"\s*/\s*", " ", cleaned)
    cleaned = normalize_spaces(cleaned)
    cleaned = strip_wrapping_quotes(cleaned)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.strip(".,;:-")
    if not cleaned:
        return None
    if len(cleaned) < 2:
        return None
    if cleaned.casefold() in {"ао", "ап", "об", "ord", "pref", "shares"}:
        return None
    if re.fullmatch(r"\d+", cleaned):
        return None
    if re.search(r"\d$", cleaned):
        return None
    return cleaned


def clean_issuer_search_name(value: object) -> str | None:
    cleaned = clean_company_name(value)
    if not cleaned:
        return None

    cleaned = re.sub(r"\b\d{2,4}[A-ZА-ЯЁ]{1,3}\b$", "", cleaned).strip()
    cleaned = re.sub(r"\s+\d+\s*[A-ZА-ЯЁa-zа-яё]{1,4}$", "", cleaned).strip()
    cleaned = re.sub(r"\b\d+\s*(?:вып|серия|серии)\b.*$", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;-")

    replacements = [
        (r"^Газпром газораспределение\b.*", "Газпром газораспределение"),
        (r"^Газпром газорасп\b.*", "Газпром газораспределение"),
        (r"^Газпром газораспр\b.*", "Газпром газораспределение"),
        (r"^Газпром межрегионгаз\b.*", "Газпром межрегионгаз"),
        (r"^Нефтяная компания ЛУКОЙЛ$", "ЛУКОЙЛ"),
        (r"^НК ЛУКОЙЛ$", "ЛУКОЙЛ"),
        (r"^Обл[.]?\s+ЛУКОЙЛ$", "ЛУКОЙЛ"),
        (r"^ЛУКОЙЛ\s+[по]$", "ЛУКОЙЛ"),
        (r"^АК Сберегательн\w+ банк\w* РФ$", "Сбербанк"),
        (r"^Сбербанк России$", "Сбербанк"),
        (r"^Сбербанк\s+[пp]$", "Сбербанк"),
        (r"^Новолипецкий металлургический комбинат$", "НЛМК"),
        (r"^Магнитогорский металлургический комбинат$", "ММК"),
        (r"^Акционерная компания АЛРОСА$", "АЛРОСА"),
        (r"^Акционерная нефтяная Компания Башнефть$", "Башнефть"),
        (r"^Акционерная финансовая корпорация Система$", "АФК Система"),
        (r"^Горно металлургическая компания Норильский никель$", "Норильский Никель"),
        (r"^Международная компания ЯНДЕКС$", "ЯНДЕКС"),
        (r"^МКПАО ЯНДЕКС$", "ЯНДЕКС"),
        (r"^Международная компания Т Технологии$", "Т-Технологии"),
        (r"^ТКСХолд$", "ТКС Холдинг"),
        (r"^ТКС Холдинг\b.*", "ТКС Холдинг"),
        (r"^РАО Норильский никель$", "Норильский Никель"),
        (r"^Федеральная сетевая компания Россети$", "Россети"),
        (r"^Российские сети$", "Россети"),
        (r"^Холдинг МРСК$", "Россети"),
        (r"^Россети\b.*", "Россети"),
        (r"^МРСК\b.*", "МРСК"),
        (r"^ТНС энерго\b.*", "ТНС энерго"),
        (r"^Квадра\b.*", "Квадра"),
        (r"^Волжская МРК\b.*", "Волжская МРК"),
        (r"^Volzhskaya MRK\b.*", "Volzhskaya MRK"),
        (r"^авиастр[.]?корп\b.*", "Объединенная авиастроительная корпорация"),
    ]
    for pattern, replacement in replacements:
        if re.search(pattern, cleaned, flags=re.IGNORECASE):
            cleaned = replacement
            break

    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;-")
    if not cleaned or len(cleaned) < 2:
        return None
    return cleaned


def has_cyrillic(value: str) -> bool:
    return bool(re.search(r"[а-яёА-ЯЁ]", value))


def cyrillic_count(value: str) -> int:
    return len(re.findall(r"[а-яёА-ЯЁ]", value))


def looks_like_security_issue(candidate: NameCandidate) -> bool:
    raw = (candidate.raw_value or candidate.name or "").casefold()
    secid = (candidate.secid or "").casefold()

    if candidate.source.startswith("legacy:"):
        return False

    if "emitent_title" in candidate.source:
        return False

    if "shortname" in candidate.source.upper() or "SHORTNAME" in candidate.source:
        if "-" in raw:
            return True
        if re.search(r"\b[аa]п\b", raw):
            return True
        if re.search(r"\d$", raw):
            return True

    if "name" in candidate.source.casefold():
        if secid and "-" in secid:
            return True
        if re.search(r"-\s*(?:[аa]о|[аa]п)?-?\s*\d+\s*[a-zа-яё]{0,3}\b", raw):
            return True
        if re.search(r"\s+\d+\s*[a-zа-яё]{1,3}\b$", candidate.name.casefold()):
            return True
        if re.search(r"\b\d+\s*вып\b", raw):
            return True
        if re.search(r"\b\d{3,}[a-zа-яё]*\b", raw):
            return True
        if re.search(r"(^|\s)об\.", raw):
            return True

    return False


def is_share_row(row: pd.Series) -> bool:
    group = str(row.get("group", "")).casefold()
    typ = str(row.get("type", "")).casefold()
    return group == "stock_shares" and typ in ALLOWED_SHARE_TYPES


def is_bond_row(row: pd.Series) -> bool:
    group = str(row.get("group", "")).casefold()
    typ = str(row.get("type", "")).casefold()
    return group in {"stock_bonds", "stock_eurobond"} and typ in ALLOWED_BOND_TYPES


def issuer_key(row: pd.Series) -> str:
    for column in ["emitent_id", "emitent_inn", "emitent_title"]:
        value = row.get(column)
        if value is not None and not pd.isna(value) and str(value).strip():
            return f"{column}:{str(value).strip()}"
    return f"name:{canonical_key(str(row.get('name') or row.get('shortname') or ''))}"


def build_issuer_table(rows: pd.DataFrame, source_group: str) -> tuple[list[str], pd.DataFrame]:
    if rows.empty:
        return [], pd.DataFrame()

    issuer_rows = []
    grouped = rows.groupby(rows.apply(issuer_key, axis=1), dropna=False)
    for key, group in grouped:
        aliases = []
        raw_titles = []
        for _, row in group.iterrows():
            full_aliases_for_row = []
            for column, priority in [("emitent_title", 0), ("name", 2)]:
                raw_value = row.get(column)
                alias = clean_issuer_search_name(raw_value)
                if alias and cyrillic_count(alias) >= 2:
                    aliases.append((alias, priority))
                    full_aliases_for_row.append(alias)
                    raw_titles.append(str(raw_value))
            short_alias = clean_issuer_search_name(row.get("shortname"))
            if short_alias and cyrillic_count(short_alias) >= 2 and not re.search(r"[-\d]$", str(row.get("shortname") or "")):
                short_key = canonical_key(short_alias)
                full_keys = [canonical_key(alias) for alias in full_aliases_for_row]
                if any(short_key and (short_key in full_key or full_key in short_key) for full_key in full_keys):
                    aliases.append((short_alias, 1))
                    raw_titles.append(str(row.get("shortname")))

        if not aliases:
            continue

        by_key: dict[str, tuple[str, int]] = {}
        for alias, priority in aliases:
            key_alias = canonical_key(alias)
            current = by_key.get(key_alias)
            if current is None or priority < current[1] or (priority == current[1] and len(alias) < len(current[0])):
                by_key[key_alias] = (alias, priority)
        compressed_aliases = [item[0] for item in sorted(by_key.values(), key=lambda x: (x[1], len(x[0]), x[0].casefold()))]
        primary = compressed_aliases[0]
        issuer_rows.append(
            {
                "search_name": primary,
                "issuer_key": key,
                "source_group": source_group,
                "emitent_id": "; ".join(sorted(set(group["emitent_id"].dropna().astype(str)))) if "emitent_id" in group else "",
                "emitent_inn": "; ".join(sorted(set(group["emitent_inn"].dropna().astype(str)))) if "emitent_inn" in group else "",
                "secids": "; ".join(sorted(set(group["secid"].dropna().astype(str)))[:100]) if "secid" in group else "",
                "security_count": len(group),
                "active_security_count": int(pd.to_numeric(group.get("is_traded", 0), errors="coerce").fillna(0).sum()),
                "raw_titles_sample": " | ".join(sorted(set(raw_titles))[:20]),
                "aliases_sample": " | ".join(compressed_aliases[:20]),
            }
        )

    issuer_df = pd.DataFrame(issuer_rows)
    if issuer_df.empty:
        return [], issuer_df

    collapsed = []
    issuer_df["search_key"] = issuer_df["search_name"].map(canonical_key)
    for _, group in issuer_df.groupby("search_key", dropna=False):
        search_name = sorted(set(group["search_name"]), key=lambda x: (x.isupper(), len(x), x.casefold()))[0]
        collapsed.append(
            {
                "search_name": search_name,
                "issuer_keys": "; ".join(sorted(set(group["issuer_key"].astype(str)))[:100]),
                "source_group": source_group,
                "emitent_id": "; ".join(sorted(set("; ".join(group["emitent_id"]).split("; ")))[:100]),
                "emitent_inn": "; ".join(sorted(set("; ".join(group["emitent_inn"]).split("; ")))[:100]),
                "secids": "; ".join(sorted(set("; ".join(group["secids"]).split("; ")))[:150]),
                "issuer_count": len(group),
                "security_count": int(group["security_count"].sum()),
                "active_security_count": int(group["active_security_count"].sum()),
                "raw_titles_sample": " | ".join(sorted(set(" | ".join(group["raw_titles_sample"]).split(" | ")))[:30]),
                "aliases_sample": " | ".join(sorted(set(" | ".join(group["aliases_sample"]).split(" | ")))[:30]),
            }
        )

    out_df = pd.DataFrame(collapsed)
    out_df = out_df[out_df["search_name"].astype(str).str.strip().ne("")]
    out_df = out_df.sort_values("search_name", key=lambda s: s.str.casefold()).reset_index(drop=True)
    names = sorted(out_df["search_name"].tolist(), key=sort_key)
    return names, out_df


def select_top_share_issuers(share_issuers: pd.DataFrame, top_n: int = SHARE_TOP_N) -> pd.DataFrame:
    work = share_issuers.copy()
    name = work["search_name"].astype(str)
    noise_mask = (
        name.str.match(r"^\s*\d", na=False)
        | name.str.contains("таксомотор", case=False, regex=False, na=False)
        | name.str.contains("автокомбинат", case=False, regex=False, na=False)
    )
    work = work[~noise_mask].copy()
    work["active_security_count"] = pd.to_numeric(work.get("active_security_count", 0), errors="coerce").fillna(0)
    work["security_count"] = pd.to_numeric(work.get("security_count", 0), errors="coerce").fillna(0)
    work["issuer_count"] = pd.to_numeric(work.get("issuer_count", 0), errors="coerce").fillna(0)
    work["has_active_security"] = (work["active_security_count"] > 0).astype(int)
    work["name_len"] = work["search_name"].astype(str).str.len()
    work = work.sort_values(
        ["has_active_security", "active_security_count", "security_count", "issuer_count", "name_len", "search_name"],
        ascending=[False, False, False, False, True, True],
        key=lambda col: col.str.casefold() if col.name == "search_name" else col,
    )
    work = work.head(top_n).drop(columns=["has_active_security", "name_len"])
    return work.reset_index(drop=True)


def fetch_security_group(group: str) -> pd.DataFrame:
    df = fetch_moex_table(
        MOEX_GLOBAL_SECURITIES_URL,
        {"group_by": "group", "group_by_filter": group},
    )
    if not df.empty:
        df["source_group"] = group
    return df


def collect_issuer_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    shares = fetch_security_group("stock_shares")
    shares = shares[shares.apply(is_share_row, axis=1)].copy()

    bond_parts = []
    for group in ["stock_bonds", "stock_eurobond"]:
        df = fetch_security_group(group)
        if not df.empty:
            bond_parts.append(df[df.apply(is_bond_row, axis=1)].copy())
    bonds = pd.concat(bond_parts, ignore_index=True, sort=False) if bond_parts else pd.DataFrame()
    return shares, bonds


def collect_candidates() -> tuple[pd.DataFrame, list[NameCandidate]]:
    candidates: list[NameCandidate] = []

    global_df = fetch_moex_table(
        MOEX_GLOBAL_SECURITIES_URL,
        {"group_by": "group", "group_by_filter": "stock_shares"},
    )
    global_df = global_df[global_df.apply(is_share_row, axis=1)].copy()
    global_df["source_table"] = "moex_global_stock_shares"

    collection_df = fetch_moex_table(MOEX_SHARES_COLLECTION_URL, {})
    collection_df["source_table"] = "moex_stock_shares_all_collection"

    source_rows = []
    if not global_df.empty:
        source_rows.append(global_df)
        for _, row in global_df.iterrows():
            secid = str(row.get("secid") or "")
            for column in ["shortname", "name", "emitent_title"]:
                cleaned = clean_company_name(row.get(column))
                if cleaned:
                    if column == "shortname" and not has_cyrillic(cleaned):
                        continue
                    candidates.append(NameCandidate(cleaned, f"moex_global:{column}", secid, str(row.get(column))))

    if not collection_df.empty:
        source_rows.append(collection_df)
        for _, row in collection_df.iterrows():
            secid = str(row.get("SECID") or "")
            for column in ["SHORTNAME", "NAME", "LATNAME"]:
                cleaned = clean_company_name(row.get(column))
                if cleaned:
                    if column == "SHORTNAME" and not has_cyrillic(cleaned):
                        continue
                    candidates.append(NameCandidate(cleaned, f"moex_collection:{column}", secid, str(row.get(column))))

    for legacy_file in LEGACY_NAME_FILES:
        for raw_name in read_legacy_names(legacy_file):
            cleaned = clean_company_name(raw_name)
            if cleaned:
                candidates.append(NameCandidate(cleaned, f"legacy:{legacy_file.name}", None, raw_name))

    raw_sources = pd.concat(source_rows, ignore_index=True, sort=False) if source_rows else pd.DataFrame()
    return raw_sources, candidates


def canonical_key(name: str) -> str:
    return re.sub(r"[^0-9a-zа-яё]+", "", name.casefold())


def sort_key(name: str) -> tuple[int, str]:
    has_cyrillic = bool(re.search(r"[а-яёА-ЯЁ]", name))
    return (0 if has_cyrillic else 1, name.casefold())


def unique_sorted_names(names: list[str]) -> list[str]:
    by_key: dict[str, str] = {}
    for name in names:
        key = canonical_key(name)
        if not key:
            continue
        current = by_key.get(key)
        if current is None or (name.isupper(), len(name), name.casefold()) < (
            current.isupper(),
            len(current),
            current.casefold(),
        ):
            by_key[key] = name
    return sorted(by_key.values(), key=sort_key)


def build_name_tables(candidates: list[NameCandidate]) -> tuple[list[str], pd.DataFrame]:
    grouped: dict[str, dict] = {}
    for candidate in candidates:
        key = canonical_key(candidate.name)
        if not key:
            continue
        if key not in grouped:
            grouped[key] = {
                "name": candidate.name,
                "sources": set(),
                "secids": set(),
                "raw_values": set(),
            }
        current = grouped[key]["name"]
        if len(candidate.name) < len(current) or (len(candidate.name) == len(current) and candidate.name < current):
            grouped[key]["name"] = candidate.name
        grouped[key]["sources"].add(candidate.source)
        if candidate.secid:
            grouped[key]["secids"].add(candidate.secid)
        if candidate.raw_value:
            grouped[key]["raw_values"].add(candidate.raw_value)

    rows = []
    for item in grouped.values():
        rows.append(
            {
                "company_name": item["name"],
                "sources": "; ".join(sorted(item["sources"])),
                "secids": "; ".join(sorted(item["secids"])[:30]),
                "raw_values_sample": " | ".join(sorted(item["raw_values"])[:10]),
                "source_count": len(item["sources"]),
                "secid_count": len(item["secids"]),
            }
        )
    diagnostics = pd.DataFrame(rows).sort_values("company_name", key=lambda s: s.str.casefold()).reset_index(drop=True)

    names = sorted(diagnostics["company_name"].tolist(), key=sort_key)
    return names, diagnostics


def shortname_matches_full_name(candidate: NameCandidate, secid_context: dict[str, set[str]]) -> bool:
    if "shortname" not in candidate.source.casefold():
        return True
    if not candidate.secid:
        return False
    name_key = canonical_key(candidate.name)
    if not name_key:
        return False
    for context_key in secid_context.get(candidate.secid, set()):
        if name_key in context_key or context_key in name_key:
            return True
    return False


def is_main_news_name(candidate: NameCandidate, secid_context: dict[str, set[str]]) -> bool:
    if candidate.source.startswith("legacy:"):
        return True
    if "LATNAME" in candidate.source:
        return False
    if looks_like_security_issue(candidate):
        return False
    if not shortname_matches_full_name(candidate, secid_context):
        return False
    return cyrillic_count(candidate.name) >= 2


def write_names(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(names) + "\n", encoding="utf-8")


def main() -> None:
    PATHS.config_dir.mkdir(parents=True, exist_ok=True)

    if PATHS.company_names_file.exists():
        shutil.copy2(PATHS.company_names_file, BACKUP_PATH)
        print(f"backup saved: {BACKUP_PATH}", flush=True)

    share_rows, bond_rows = collect_issuer_tables()
    share_full_names, share_issuers = build_issuer_table(share_rows, "stock_shares")
    bond_names, bond_issuers = build_issuer_table(bond_rows, "stock_bonds_and_eurobonds")
    share_top_issuers = select_top_share_issuers(share_issuers, SHARE_TOP_N)
    share_top_names = share_top_issuers["search_name"].tolist()
    all_names = unique_sorted_names(share_full_names + bond_names)

    if share_rows.empty:
        raise RuntimeError("No MOEX share rows were loaded; company list was not updated.")
    if len(share_full_names) < 300:
        raise RuntimeError(f"Unexpectedly small share issuer list: {len(share_full_names)} names")
    if len(share_top_names) != SHARE_TOP_N:
        raise RuntimeError(f"Unexpected top share list size: {len(share_top_names)} names")

    write_names(PATHS.company_names_file, share_top_names)
    write_names(OUTPUT_SHARE_NAMES, share_top_names)
    write_names(OUTPUT_SHARE_TOP_NAMES, share_top_names)
    write_names(OUTPUT_SHARE_FULL_NAMES, share_full_names)
    write_names(OUTPUT_BOND_NAMES, bond_names)
    write_names(OUTPUT_ALL_NAMES, all_names)

    share_issuers.to_csv(OUTPUT_SHARE_ISSUERS, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    share_top_issuers.to_csv(OUTPUT_SHARE_TOP_ISSUERS, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    bond_issuers.to_csv(OUTPUT_BOND_ISSUERS, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    # Backward-compatible audit filenames used by earlier checks.
    share_issuers.to_csv(
        PATHS.config_dir / "company_names_main_sources.csv",
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL,
    )
    all_issuers = pd.concat([share_issuers, bond_issuers], ignore_index=True, sort=False)
    all_issuers.to_csv(OUTPUT_DIAGNOSTICS, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    all_issuers.to_csv(OUTPUT_ALL_CANDIDATES, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    print(f"company names written: {PATHS.company_names_file}", flush=True)
    print(f"share names written: {OUTPUT_SHARE_NAMES}", flush=True)
    print(f"full share names written: {OUTPUT_SHARE_FULL_NAMES}", flush=True)
    print(f"top share names written: {OUTPUT_SHARE_TOP_NAMES}", flush=True)
    print(f"bond names written: {OUTPUT_BOND_NAMES}", flush=True)
    print(f"all issuer names written: {OUTPUT_ALL_NAMES}", flush=True)
    print(f"share issuers written: {OUTPUT_SHARE_ISSUERS}", flush=True)
    print(f"top share issuers written: {OUTPUT_SHARE_TOP_ISSUERS}", flush=True)
    print(f"bond issuers written: {OUTPUT_BOND_ISSUERS}", flush=True)
    print(f"share security rows: {len(share_rows)}", flush=True)
    print(f"bond security rows: {len(bond_rows)}", flush=True)
    print(f"unique share search names full: {len(share_full_names)}", flush=True)
    print(f"unique share search names top: {len(share_top_names)}", flush=True)
    print(f"unique bond search names: {len(bond_names)}", flush=True)
    print(f"unique all issuer names: {len(all_names)}", flush=True)
    print("first 30 names:", flush=True)
    for name in share_top_names[:30]:
        print(name, flush=True)


if __name__ == "__main__":
    main()
