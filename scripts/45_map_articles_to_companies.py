#!/usr/bin/env python3
"""Map archived news articles to companies using names, industries and products.

The matcher scans each article once with a token-level Aho-Corasick automaton.
Russian words are Snowball-stemmed, so common case/number endings match the
base-form keywords in the company workbook. Row groups are processed by worker
processes and merged into one Parquet output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import sys
import tempfile
import time
import unicodedata
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import nltk
import openpyxl
import pyarrow as pa
import pyarrow.parquet as pq
from nltk.stem.snowball import SnowballStemmer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTICLES = Path.home() / "dev/dissertation_supp/news_data/merged/articles_total_rbc_dates_restored.parquet"
DEFAULT_OUTPUT = Path.home() / "dev/dissertation_supp/news_data/merged/article_company_map_algorithmic_v3.parquet"
SHEET_NAME = "По компаниям"
TOKEN_RE = re.compile(r"[a-zа-я0-9]+", re.IGNORECASE)
PAREN_RE = re.compile(r"\(([^()]*)\)")
CYRILLIC_RE = re.compile(r"[а-я]", re.IGNORECASE)

# Terms that are too broad to identify an issuer in article text on their own.
GENERIC_KEYWORDS = {
    "актив", "алмаз", "банк", "бизнес", "бюджет", "вклад", "газ", "генерация", "госдолг", "государство",
    "деятельность", "добыча", "инвестиции", "инвестиция", "инфраструктура", "карта",
    "кредит", "магазин", "машина", "медицина", "металл", "нефть", "облигация", "оборудование",
    "оплата", "платёж", "платеж", "покупатель", "продукт", "производство", "проект",
    "регион", "сервис", "связь", "система", "средство", "сталь", "стройматериал", "товар",
    "торговля", "услуга", "уголь", "финансы", "финансирование", "цена", "экономика",
}

# Explicit user exception (2026-10-06): common banking topics are useful
# candidates even without issuer names and even with many keyword owners.
BANK_KEYWORD_POLICY_VERSION = "bank-topics-20261006-v1"
BANK_KEYWORD_EXCEPTIONS = {"industry": {"банк"}, "product": {"кредит", "вклад"}}
BANK_INDUSTRY_RE = re.compile(r"\bбанк(?:и|а|ов|ам|ами|ах|ом|е)?\b|\bбанковск[а-я]*\b")


def is_bank_company(company: dict[str, Any]) -> bool:
    industry = normalize_text(company.get("industry", ""))
    if BANK_INDUSTRY_RE.search(industry):
        return True
    # A non-banking subsidiary must not become a bank just because its brand
    # contains 'банк'. The fallback is only for an absent classification.
    return not industry.strip() and "банк" in normalize_text(company.get("name", company.get("company", "")))


def bank_keyword_exception(company: dict[str, Any], signal: str, keyword: str) -> bool:
    eligible = company.get("bank_product_eligible", is_bank_company(company)) if signal == "product" else is_bank_company(company)
    return eligible and normalize_text(keyword) in BANK_KEYWORD_EXCEPTIONS.get(signal, set())


def company_signal_keywords(company: dict[str, Any], signal: str) -> list[str]:
    keywords = split_keywords(company.get(signal, ""))
    if (company.get("bank_product_eligible", is_bank_company(company)) if signal == "product" else is_bank_company(company)):
        keywords.extend(sorted(BANK_KEYWORD_EXCEPTIONS.get(signal, set())))
    return sorted(set(keywords))

LEGAL_TOKENS = {
    "ао", "зао", "оао", "ооо", "пао", "пao", "нпф", "гк", "группа", "компания",
    "компаний", "холдинг", "холдинговая", "плс", "инк", "корпорация",
}
AMBIGUOUS_NAMES = {
    "открытие",
    "альянс", "восток", "калита", "кокс", "победа", "прогресс", "регион", "резерв",
    "россия", "москва", "беларусь", "казахстан", "самолет", "синергия", "капитал",
    "лента", "система", "союз", "старт", "труд", "центр", "ювелит", "азот", "урал", "якутия",
}
AMBIGUOUS_NAME_PHRASES = {"городской супермаркет"}
PARENTHETICAL_DESCRIPTORS = {"государственная компания"}

OUTPUT_SCHEMA = pa.schema(
    [
        ("article_id", pa.string()),
        ("date", pa.timestamp("ns")),
        ("media", pa.string()),
        ("title", pa.string()),
        ("company", pa.string()),
        ("company_id", pa.string()),
        ("match_signal", pa.string()),
        ("matched_terms", pa.string()),
        ("article_url", pa.string()),
    ]
)

_WORKER_PF = None
_WORKER_TRIE: list[dict[str, int]] = []
_WORKER_FAIL: list[int] = []
_WORKER_OUTPUTS: list[list[int]] = []
_WORKER_REFS: list[list[tuple[str, str, str, str, bool, bool, tuple[str, ...]]]] = []
_WORKER_STEMMER = None
_WORKER_MAX_OWNERS = 30
_WORKER_EXCLUDED_MEDIA: set[str] = set()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return unicodedata.normalize("NFKC", str(value)).casefold().replace("ё", "е")


def tokenize(value: Any) -> list[str]:
    return TOKEN_RE.findall(normalize_text(value))


def stem_tokens(tokens: list[str]) -> tuple[str, ...]:
    assert _WORKER_STEMMER is not None
    return tuple(_stem_token(token) for token in tokens)


@lru_cache(maxsize=200_000)
def _stem_token(token: str) -> str:
    if CYRILLIC_RE.search(token):
        return _WORKER_STEMMER.stem(token)
    return token


def clean_keyword(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def split_keywords(value: Any) -> list[str]:
    return [clean_keyword(part) for part in str(value or "").split(";") if clean_keyword(part)]


def company_id_for(name: str, isin_values: Any) -> str:
    identity = f"{normalize_text(name)}\t{normalize_text(isin_values)}".encode("utf-8")
    return "c_" + hashlib.blake2b(identity, digest_size=12).hexdigest()


def load_company_rows(workbook_path: Path, industry_overrides: Path | None = None) -> list[dict[str, str]]:
    wb = openpyxl.load_workbook(workbook_path, read_only=True, data_only=True)
    if SHEET_NAME not in wb.sheetnames:
        raise ValueError(f"В книге нет листа «{SHEET_NAME}»: {workbook_path}")
    ws = wb[SHEET_NAME]
    rows: list[dict[str, str]] = []
    for row in ws.iter_rows(min_row=2, max_col=5, values_only=True):
        name = str(row[0] or "").strip()
        if not name:
            continue
        rows.append(
            {
                "name": name,
                "company_id": company_id_for(name, row[2]),
                "industry": str(row[3] or "").strip(),
                "product": str(row[4] or "").strip(),
            }
        )
    wb.close()
    if industry_overrides is not None:
        policy = json.loads(industry_overrides.read_text(encoding="utf-8"))
        overrides = policy["original_company_overrides"]
        unknown = set(overrides) - {r["company_id"] for r in rows}
        if unknown:
            raise ValueError(f"Unknown original company IDs: {sorted(unknown)}")
        for company in rows:
            value = overrides.get(company["company_id"])
            if value is not None:
                if len(split_keywords(value["industry"])) > 1:
                    raise ValueError("More than one primary industry")
                company["bank_product_eligible"] = is_bank_company(company)
                company["original_industry"] = company["industry"]
                company["industry"] = value["industry"]
                company["primary_industry_explicit"] = True
    return rows


def name_variants(name: str) -> set[str]:
    variants = {name.strip()}
    parentheticals = PAREN_RE.findall(name)
    safely_removable = True
    for inner in parentheticals:
        marker = re.match(r"^(?:бывш\.?|ранее|прежнее название)\s*(.+)$", inner.strip(), flags=re.I)
        if marker:
            variants.add(marker.group(1).strip())
            continue

        inner_normalized = normalize_text(inner).strip()
        inner_tokens = tokenize(inner)
        if inner_normalized.startswith("не сущ") or inner_normalized in PARENTHETICAL_DESCRIPTORS:
            continue
        if inner_tokens and all(token in LEGAL_TOKENS for token in inner_tokens):
            continue

        # An unmarked parenthetical can be an alternate brand/acronym (keep it
        # as its own alias), or a location (do not turn the location into a name).
        safely_removable = False
        if len(inner_tokens) == 1:
            one_word_alias = inner.strip()
            acronym_or_latin_brand = (
                one_word_alias.isupper()
                or bool(re.search(r"[0-9-]", one_word_alias))
                or bool(re.search(r"[a-z]", one_word_alias))
            )
            if not acronym_or_latin_brand:
                continue
        if not (len(inner_tokens) == 1 and inner_tokens[0] in AMBIGUOUS_NAMES):
            variants.add(inner.strip())

    if parentheticals and safely_removable:
        without_parentheses = PAREN_RE.sub(" ", name).strip()
        if without_parentheses:
            variants.add(without_parentheses)

    # Keep a useful issuer name before an alternate-name annotation, but not
    # generic phrases such as "городской супермаркет".
    if parentheticals:
        core_name = name.split("(", 1)[0].strip()
        core_tokens = tokenize(core_name)
        core_phrase = " ".join(core_tokens)
        if core_name and core_phrase not in AMBIGUOUS_NAME_PHRASES and not (
            len(core_tokens) == 1 and core_tokens[0] in AMBIGUOUS_NAMES
        ):
            variants.add(core_name)

    for variant in list(variants):
        words = tokenize(variant)
        while words and words[0] in LEGAL_TOKENS:
            words.pop(0)
        while words and words[-1] in LEGAL_TOKENS:
            words.pop()
        if words and not all(word in LEGAL_TOKENS for word in words):
            variants.add(" ".join(words))
        if len(words) > 2 and words[:2] == ["группа", "компаний"]:
            variants.add(" ".join(words[2:]))
        if len(words) > 1 and words[0] in {"группа", "гк"}:
            variants.add(" ".join(words[1:]))

    clean = set()
    for variant in variants:
        tokens = tokenize(variant)
        if not tokens or all(word in LEGAL_TOKENS for word in tokens):
            continue
        if len(tokens) == 1 and tokens[0] in LEGAL_TOKENS:
            continue
        clean.add(variant)
    return clean


def build_pattern_records(companies: list[dict[str, str]], max_keyword_owners: int) -> list[dict[str, Any]]:
    baseline_products = None
    if any(c.get("primary_industry_explicit") for c in companies):
        original = []
        for company in companies:
            restored = dict(company)
            if restored.pop("primary_industry_explicit", False):
                restored["industry"] = restored.pop("original_industry")
                restored.pop("bank_product_eligible", None)
            original.append(restored)
        baseline_products = {(ref[0], tuple(record["tokens"]), ref[3])
                             for record in build_pattern_records(original, max_keyword_owners)
                             for ref in record["refs"] if ref[2] == "product"}
    patterns: dict[tuple[str, ...], list[tuple[str, str, str, str, bool, bool]]] = defaultdict(list)
    keyword_owners: dict[tuple[str, ...], set[str]] = defaultdict(set)
    bank_company_ids = {company["company_id"] for company in companies if is_bank_company(company)}
    bank_product_ids = {company["company_id"] for company in companies if company.get("bank_product_eligible", is_bank_company(company))}
    primary_industry_ids = {company["company_id"] for company in companies if company.get("primary_industry_explicit")}

    for company in companies:
        cid = company["company_id"]
        name = company["name"]
        for variant in name_variants(name):
            tokens = tokenize(variant)
            if not tokens:
                continue
            if " ".join(tokens) in AMBIGUOUS_NAME_PHRASES:
                continue
            # Avoid noisy direct matches on very short, ordinary company names.
            if len(tokens) == 1 and len(tokens[0]) < 3:
                continue
            # A one-word name that is also a common place or ordinary noun is
            # too ambiguous to identify an issuer, even when it appears in a title.
            if len(tokens) == 1 and tokens[0] in AMBIGUOUS_NAMES:
                continue
            ambiguous = len(tokens) == 1 and tokens[0] in AMBIGUOUS_NAMES
            patterns[stem_tokens(tokens)].append((cid, name, "name", name, ambiguous, False, tuple(tokens)))

        for signal in ("industry", "product"):
            for keyword in company_signal_keywords(company, signal):
                raw_tokens = tokenize(keyword)
                if not raw_tokens:
                    continue
                primary_exception = signal == "industry" and company.get("primary_industry_explicit")
                if not primary_exception and not bank_keyword_exception(company, signal, keyword) and len(raw_tokens) == 1 and (len(raw_tokens[0]) < 4 or raw_tokens[0] in GENERIC_KEYWORDS):
                    continue
                pattern = stem_tokens(raw_tokens)
                patterns[pattern].append((cid, name, signal, keyword, False, False, tuple(raw_tokens)))
                keyword_owners[pattern].add(cid)
                # Snowball stems singular 'кредит' to 'кред', but declined
                # forms such as 'кредиты/кредитам' to 'кредит'. Retain both
                # sequences for this explicit bank exception, not for all terms.
                if bank_keyword_exception(company, signal, keyword) and normalize_text(keyword) == "кредит":
                    plural_pattern = stem_tokens(["кредиты"])
                    if plural_pattern != pattern:
                        patterns[plural_pattern].append((cid, name, signal, keyword, False, False, tuple(raw_tokens)))
                        keyword_owners[plural_pattern].add(cid)

    overshared = {
        pattern
        for pattern, owners in keyword_owners.items()
        if len(owners) > max_keyword_owners
    }

    clean_patterns: list[dict[str, Any]] = []
    for pattern, refs in patterns.items():
        kept = []
        for ref in refs:
            explicit_bank_exception = ref[0] in (bank_product_ids if ref[2] == "product" else bank_company_ids) and normalize_text(ref[3]) in BANK_KEYWORD_EXCEPTIONS.get(ref[2], set())
            primary_exception = ref[2] == "industry" and ref[0] in primary_industry_ids
            if ref[2] == "product" and baseline_products is not None:
                if (ref[0], pattern, ref[3]) not in baseline_products:
                    continue
            elif ref[2] != "name" and pattern in overshared and not explicit_bank_exception and not primary_exception:
                continue
            title_only = ref[2] != "name"
            kept.append((*ref[:5], title_only, ref[6]))
        if kept and pattern:
            clean_patterns.append({"tokens": pattern, "refs": kept})
    return clean_patterns


def build_automaton(pattern_records: list[dict[str, Any]]) -> tuple[list[dict[str, int]], list[int], list[list[int]], list[list[tuple[str, str, str, str, bool, bool, tuple[str, ...]]]]]:
    trie: list[dict[str, int]] = [{}]
    outputs: list[list[int]] = [[]]
    refs: list[list[tuple[str, str, str, str, bool, bool, tuple[str, ...]]]] = []

    for record in pattern_records:
        state = 0
        for token in record["tokens"]:
            next_state = trie[state].get(token)
            if next_state is None:
                next_state = len(trie)
                trie[state][token] = next_state
                trie.append({})
                outputs.append([])
            state = next_state
        pattern_id = len(refs)
        refs.append(record["refs"])
        outputs[state].append(pattern_id)

    fail = [0] * len(trie)
    queue: deque[int] = deque()
    for child in trie[0].values():
        queue.append(child)

    while queue:
        state = queue.popleft()
        for token, child in trie[state].items():
            queue.append(child)
            fallback = fail[state]
            while fallback and token not in trie[fallback]:
                fallback = fail[fallback]
            fail[child] = trie[fallback].get(token, 0)
            outputs[child].extend(outputs[fail[child]])

    return trie, fail, outputs, refs


def worker_init(
    articles_path: str,
    pattern_records: list[dict[str, Any]],
    max_keyword_owners: int,
    excluded_media: list[str],
) -> None:
    global _WORKER_PF, _WORKER_TRIE, _WORKER_FAIL, _WORKER_OUTPUTS, _WORKER_REFS
    global _WORKER_STEMMER, _WORKER_MAX_OWNERS, _WORKER_EXCLUDED_MEDIA
    _WORKER_PF = pq.ParquetFile(articles_path)
    _WORKER_STEMMER = SnowballStemmer("russian")
    _stem_token.cache_clear()
    _WORKER_MAX_OWNERS = max_keyword_owners
    _WORKER_EXCLUDED_MEDIA = {normalize_text(media) for media in excluded_media}
    _WORKER_TRIE, _WORKER_FAIL, _WORKER_OUTPUTS, _WORKER_REFS = build_automaton(pattern_records)


def canonical_url(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parts = urlsplit(raw)
    if not parts.scheme or not parts.netloc:
        return normalize_text(raw)
    scheme = parts.scheme.casefold()
    host = parts.netloc.casefold()
    if host.startswith("www."):
        host = host[4:]
    if (scheme == "https" and host.endswith(":443")) or (scheme == "http" and host.endswith(":80")):
        host = host.rsplit(":", 1)[0]
    path = parts.path.rstrip("/") or "/"
    ignored = {"yclid", "gclid", "fbclid", "ref", "from", "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content"}
    query = urlencode(sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k.casefold() not in ignored))
    return urlunsplit((scheme, host, path, query, ""))


def make_article_id(media: Any, link: Any, published: Any, title: Any, text: Any) -> str:
    canonical = canonical_url(link)
    if canonical:
        payload = f"{normalize_text(media)}\t{canonical}"
    else:
        date_value = published.isoformat() if hasattr(published, "isoformat") else str(published or "")
        text_fingerprint = hashlib.blake2b(str(text or "").encode("utf-8"), digest_size=8).hexdigest()
        payload = f"{normalize_text(media)}\t{date_value}\t{normalize_text(title)}\t{text_fingerprint}"
    return "a_" + hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def scan_tokens(tokens: tuple[str, ...] | list[str]) -> set[tuple[int, int]]:
    state = 0
    found: set[tuple[int, int]] = set()
    for end_index, token in enumerate(tokens):
        while state and token not in _WORKER_TRIE[state]:
            state = _WORKER_FAIL[state]
        state = _WORKER_TRIE[state].get(token, 0)
        for pattern_id in _WORKER_OUTPUTS[state]:
            found.add((pattern_id, end_index))
    return found


def surface_compatible(surface: tuple[str, ...], article_tokens: tuple[str, ...] | list[str], end_index: int) -> bool:
    start = end_index - len(surface) + 1
    if start < 0:
        return False
    for expected, actual in zip(surface, article_tokens[start : end_index + 1]):
        if expected == actual:
            continue
        if not CYRILLIC_RE.search(expected) or not CYRILLIC_RE.search(actual):
            return False
        shared = 0
        for left, right in zip(expected, actual):
            if left != right:
                break
            shared += 1
        minimum_prefix = 3 if min(len(expected), len(actual)) <= 4 else 4
        if shared < minimum_prefix:
            return False
    return True


def add_pattern_hits(
    pattern_hits: set[tuple[int, int]],
    matches: dict[str, dict[str, Any]],
    in_title: bool,
    surface_tokens: tuple[str, ...] | list[str],
) -> None:
    for pattern_id, end_index in pattern_hits:
        for cid, company_name, signal, matched_term, ambiguous, title_only, pattern_surface in _WORKER_REFS[pattern_id]:
            if signal == "name" and ambiguous and not in_title:
                continue
            if signal != "name" and not in_title:
                continue
            if not surface_compatible(pattern_surface, surface_tokens, end_index):
                continue
            entry = matches.setdefault(
                cid,
                {"company": company_name, "signals": set(), "terms": defaultdict(set)},
            )
            label = "name" if signal == "name" else signal
            entry["signals"].add(label)
            entry["terms"][label].add(matched_term)


def process_row_group(task: tuple[int, int | None, str]) -> dict[str, Any]:
    row_group_index, max_rows, tmp_dir = task
    assert _WORKER_PF is not None
    table = _WORKER_PF.read_row_group(
        row_group_index,
        columns=["media", "title", "link", "date", "text"],
        use_threads=True,
    )
    if max_rows is not None and max_rows < table.num_rows:
        table = table.slice(0, max_rows)

    media_values = table.column("media").to_pylist()
    title_values = table.column("title").to_pylist()
    link_values = table.column("link").to_pylist()
    date_values = table.column("date").to_pylist()
    text_values = table.column("text").to_pylist()
    output_rows: list[dict[str, Any]] = []
    excluded_rows = 0

    for media, title, link, published, body in zip(
        media_values, title_values, link_values, date_values, text_values
    ):
        if normalize_text(media) in _WORKER_EXCLUDED_MEDIA:
            excluded_rows += 1
            continue
        title = str(title or "")
        body = str(body or "")
        title_surface = tokenize(title)
        body_surface = tokenize(body)
        title_tokens = stem_tokens(title_surface)
        body_tokens = stem_tokens(body_surface)
        matches: dict[str, dict[str, Any]] = {}
        add_pattern_hits(scan_tokens(title_tokens), matches, in_title=True, surface_tokens=title_surface)
        add_pattern_hits(scan_tokens(body_tokens), matches, in_title=False, surface_tokens=body_surface)
        if not matches:
            continue

        article_id = make_article_id(media, link, published, title, body)
        for cid, match in matches.items():
            ordered_signals = [s for s in ("name", "industry", "product") if s in match["signals"]]
            terms = {s: sorted(match["terms"][s]) for s in ordered_signals}
            output_rows.append(
                {
                    "article_id": article_id,
                    "date": published,
                    "media": str(media or ""),
                    "title": title,
                    "company": match["company"],
                    "company_id": cid,
                    "match_signal": ";".join(ordered_signals),
                    "matched_terms": json.dumps(terms, ensure_ascii=False, separators=(",", ":")),
                    "article_url": str(link or ""),
                }
            )

    part_path = Path(tmp_dir) / f"part-{row_group_index:05d}.parquet"
    if output_rows:
        out_table = pa.Table.from_pylist(output_rows, schema=OUTPUT_SCHEMA)
        pq.write_table(out_table, part_path, compression="snappy", use_dictionary=["media", "company", "company_id", "match_signal"])
    return {
        "row_group": row_group_index,
        "rows_read": table.num_rows,
        "excluded_rows": excluded_rows,
        "matches": len(output_rows),
        "part": str(part_path) if output_rows else None,
    }


def latest_company_workbook() -> Path:
    candidates = sorted(
        ROOT.glob("outputs/*/*classification*.xlsx"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("Не нашёл актуальную таблицу классификации компаний в outputs/.")
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES, help="Исходный parquet с новостями")
    parser.add_argument("--companies", type=Path, default=None, help="XLSX с листом «По компаниям»; по умолчанию самый новый файл классификации")
    parser.add_argument("--industry-overrides", type=Path, help="Versioned canonical single-industry policy; does not modify the source workbook")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Итоговый parquet с парами статья-компания")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="Число процессов; по умолчанию все логические ядра")
    parser.add_argument("--max-articles", type=int, default=None, help="Ограничить число первых статей для пробного запуска")
    parser.add_argument("--max-keyword-owners", type=int, default=15, help="Не учитывать общие отраслевые/продуктовые ключи, встречающиеся у большего числа компаний")
    parser.add_argument("--exclude-media", action="append", default=["frank"], help="Исключить СМИ по значению столбца media; Frank исключён по умолчанию")
    parser.add_argument("--include-frank", action="store_true", help="Не исключать Frank Media")
    parser.add_argument("--overwrite", action="store_true", help="Разрешить заменить целевой выходной файл")
    return parser.parse_args()


def main() -> int:
    global _WORKER_STEMMER
    args = parse_args()
    articles_path = args.articles.expanduser().resolve()
    companies_path = (args.companies or latest_company_workbook()).expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not articles_path.is_file():
        raise FileNotFoundError(f"Файл новостей не найден: {articles_path}")
    if not companies_path.is_file():
        raise FileNotFoundError(f"Файл компаний не найден: {companies_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Выход уже существует (укажи --overwrite): {output_path}")
    if args.workers < 1:
        raise ValueError("--workers должен быть не меньше 1")
    if args.max_articles is not None and args.max_articles < 1:
        raise ValueError("--max-articles должен быть не меньше 1")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    companies = load_company_rows(companies_path, args.industry_overrides)
    excluded_media = {normalize_text(media) for media in args.exclude_media}
    if args.include_frank:
        excluded_media.discard("frank")
    _WORKER_STEMMER = SnowballStemmer("russian")
    _stem_token.cache_clear()
    pattern_records = build_pattern_records(companies, args.max_keyword_owners)
    input_file = pq.ParquetFile(articles_path)
    total_rows = input_file.metadata.num_rows
    row_group_count = input_file.metadata.num_row_groups
    row_groups = list(range(row_group_count))
    if args.max_articles is not None:
        row_groups = []
        remaining = args.max_articles
        for index in range(row_group_count):
            if remaining <= 0:
                break
            row_groups.append(index)
            remaining -= input_file.metadata.row_group(index).num_rows

    run_rows = min(total_rows, args.max_articles) if args.max_articles is not None else total_rows
    workers = min(args.workers, len(row_groups))
    start = time.monotonic()
    print(
        f"Статей: {run_rows:,}; компаний: {len(companies):,}; рабочих процессов: {workers}; "
        f"ключевых последовательностей: {len(pattern_records):,}; "
        f"исключённые СМИ: {', '.join(sorted(excluded_media)) or 'нет'}",
        flush=True,
    )

    with tempfile.TemporaryDirectory(prefix=f".{output_path.stem}_", dir=output_path.parent) as tmp_dir:
        tasks: list[tuple[int, int | None, str]] = []
        remaining = args.max_articles
        for index in row_groups:
            group_rows = input_file.metadata.row_group(index).num_rows
            take = None
            if remaining is not None:
                take = min(group_rows, remaining)
                remaining -= take
            tasks.append((index, take, tmp_dir))

        results: dict[int, dict[str, Any]] = {}
        rows_done = 0
        excluded_done = 0
        matches_done = 0
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=worker_init,
            initargs=(str(articles_path), pattern_records, args.max_keyword_owners, sorted(excluded_media)),
        ) as executor:
            futures = {executor.submit(process_row_group, task): task[0] for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                results[result["row_group"]] = result
                rows_done += result["rows_read"]
                excluded_done += result["excluded_rows"]
                matches_done += result["matches"]
                elapsed = max(time.monotonic() - start, 0.001)
                print(
                    f"Готово групп: {len(results)}/{len(tasks)}; статей: {rows_done:,}/{run_rows:,}; "
                    f"исключено: {excluded_done:,}; "
                    f"связей: {matches_done:,}; скорость: {rows_done / elapsed:,.0f} статей/с",
                    flush=True,
                )

        writer = pq.ParquetWriter(output_path, OUTPUT_SCHEMA, compression="snappy", use_dictionary=["media", "company", "company_id", "match_signal"])
        try:
            for index in sorted(results):
                part = results[index]["part"]
                if not part:
                    continue
                part_file = pq.ParquetFile(part)
                try:
                    for batch in part_file.iter_batches(batch_size=50_000):
                        writer.write_batch(batch)
                finally:
                    # Windows keeps the temporary part locked until the reader
                    # is explicitly closed; TemporaryDirectory then cannot clean
                    # up after an otherwise successful merge.
                    part_file.close()
        finally:
            writer.close()

    elapsed = time.monotonic() - start
    output_meta = pq.ParquetFile(output_path).metadata
    audit_path = output_path.with_suffix(output_path.suffix + ".audit.json")
    audit = {
        "article_source": str(articles_path),
        "company_source": str(companies_path),
        "article_rows_read": rows_done,
        "articles_excluded": excluded_done,
        "excluded_media": sorted(excluded_media),
        "articles_included": rows_done - excluded_done,
        "articles_in_source": total_rows,
        "companies_in_source": len(companies),
        "mapping_rows": output_meta.num_rows,
        "worker_processes": workers,
        "max_keyword_owners": args.max_keyword_owners,
        "keyword_patterns": len(pattern_records),
        "bank_keyword_policy": BANK_KEYWORD_POLICY_VERSION,
        "name_alias_policy": "no-bare-otkritie-20261008-v1",
        "blocked_single_word_name_aliases": sorted(AMBIGUOUS_NAMES),
        "industry_overrides": str(args.industry_overrides) if args.industry_overrides else None,
        "primary_industry_explicit": bool(args.industry_overrides),
        "bank_keyword_exceptions": {k: sorted(v) for k, v in BANK_KEYWORD_EXCEPTIONS.items()},
        "elapsed_seconds": round(elapsed, 2),
        "articles_per_second": round(rows_done / max(elapsed, 0.001), 2),
        "id_rule": "BLAKE2b-128(media + canonical URL); title/date/text fingerprint if URL is missing",
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**audit, "output": str(output_path), "audit": str(audit_path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
