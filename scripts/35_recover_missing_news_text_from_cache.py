"""Recover missing article text from already downloaded page archives only."""

from __future__ import annotations

import argparse
import re
import gc
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from bs4 import BeautifulSoup, SoupStrainer

try:
    import lxml  # noqa: F401

    HTML_PARSER = "lxml"
except ImportError:
    HTML_PARSER = "html.parser"


SUPPLEMENT_ROOT = Path("./market_data/raw_external_2026-09-30")
NEWS_FILE = Path("./news_data/merged/articles_total_rbc_dates_restored.parquet")
PAGE_ROOT = Path("./news_data/article_pages")
DATE_LIMIT = pd.Timestamp("2026-05-29 23:59:59")
PAGE_KEYS = {
    "rbc": ["fronturl", "url", "link"],
    "kommersant": ["url", "fronturl", "link"],
    "prime": ["link", "fronturl", "url"],
    "interfax": ["url", "fronturl", "link"],
}
MAX_PAGE_SOURCE_CHARS = 4_000_000


def normalize_link(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def class_has(value: object, token: str) -> bool:
    if isinstance(value, (list, tuple)):
        return token in value
    if isinstance(value, str):
        return token in value.split()
    return False


def text_from_nodes(nodes: list) -> str:
    parts = []
    seen = set()
    for node in nodes:
        value = re.sub(r"\s+", " ", node.get_text(" ", strip=True)).replace("\xa0", " ").strip()
        if value and value not in seen:
            parts.append(value)
            seen.add(value)
    return "\n".join(parts).strip()


def extract_cached_html(source: str, html: object) -> tuple[str, str]:
    if not isinstance(html, str) or not html.strip():
        return "", ""
    if len(html) > MAX_PAGE_SOURCE_CHARS:
        return "", f"page_source exceeds {MAX_PAGE_SOURCE_CHARS:,} characters; not parsed"

    if source == "kommersant":
        soup = BeautifulSoup(html, HTML_PARSER, parse_only=SoupStrainer(class_=lambda value: class_has(value, "doc__text")))
        text = text_from_nodes(soup.select(".doc__text"))
        if not text:
            soup = BeautifulSoup(html, HTML_PARSER, parse_only=SoupStrainer("article"))
            text = text_from_nodes(soup.select("article p"))
        return text, ""

    if source == "interfax":
        soup = BeautifulSoup(html, HTML_PARSER, parse_only=SoupStrainer(attrs={"itemprop": "articleBody"}))
        text = text_from_nodes(soup.select('[itemprop="articleBody"] p'))
        if not text:
            soup = BeautifulSoup(html, HTML_PARSER, parse_only=SoupStrainer("article"))
            text = text_from_nodes(soup.select("article p"))
        return text, ""

    if source == "prime":
        soup = BeautifulSoup(html, HTML_PARSER, parse_only=SoupStrainer(class_=lambda value: class_has(value, "article__body") or class_has(value, "article__text")))
        text = text_from_nodes(soup.select(".article__body .article__text"))
        if not text:
            soup = BeautifulSoup(html, HTML_PARSER, parse_only=SoupStrainer(attrs={"itemprop": "articleBody"}))
            body = soup.select_one('[itemprop="articleBody"]')
            text = text_from_nodes([body]) if body else ""
        return text, ""

    if source == "rbc":
        soup = BeautifulSoup(html, HTML_PARSER, parse_only=SoupStrainer(attrs={"itemprop": "articleBody"}))
        body = soup.select_one('[itemprop="articleBody"]')
        text = text_from_nodes([body]) if body else ""
        if not text:
            soup = BeautifulSoup(html, HTML_PARSER, parse_only=SoupStrainer(class_=lambda value: class_has(value, "article__text") or class_has(value, "article__content")))
            for selector in (".article__text", ".article__content"):
                body = soup.select_one(selector)
                text = text_from_nodes([body]) if body else ""
                if text:
                    break
        if not text:
            soup = BeautifulSoup(html, HTML_PARSER, parse_only=SoupStrainer("main"))
            text = text_from_nodes([node for node in soup.select("main p") if not node.get_text(" ", strip=True).startswith("Фото:")])
        return text, ""

    return "", "unsupported source"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=SUPPLEMENT_ROOT / "news")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    target = output / "missing_text_from_cached_pages.csv"
    if target.exists():
        raise FileExistsError(f"Refusing to replace existing output: {target}")

    targets: dict[tuple[str, str], dict] = {}
    news_parquet = pq.ParquetFile(NEWS_FILE)
    for batch_number, batch in enumerate(news_parquet.iter_batches(batch_size=5000, columns=["media", "company", "title", "link", "date", "text"]), start=1):
        frame = batch.to_pandas()
        dates = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame[
            frame["media"].isin(["rbc", "kommersant", "prime", "interfax"])
            & (dates <= DATE_LIMIT)
            & (frame["text"].isna() | frame["text"].astype(str).str.strip().eq(""))
            & frame["link"].notna()
        ]
        for row in frame.itertuples(index=False):
            media, company, title, raw_link, raw_date, _ = row
            link = normalize_link(raw_link)
            if not link:
                continue
            key = (media, link)
            item = targets.get(key)
            if item is None:
                item = {"media": media, "link": link, "date": raw_date, "title": title, "companies": set(), "source_rows_with_blank_text": 0}
                targets[key] = item
            if isinstance(company, str) and company:
                item["companies"].add(company)
            item["source_rows_with_blank_text"] += 1
        if batch_number % 20 == 0:
            print(f"[news master] scanned {batch_number} batches; unique missing links={len(targets)}", flush=True)
    target_rows = [
        {"media": item["media"], "link": item["link"], "date": item["date"], "title": item["title"], "company_mentions_from_source_rows": " | ".join(sorted(item["companies"])), "source_rows_with_blank_text": item["source_rows_with_blank_text"]}
        for item in targets.values()
    ]
    pending = {(r["media"], r["link"]): r for r in target_rows}
    found: dict[tuple[str, str], dict] = {}
    page_errors = []
    files = sorted(PAGE_ROOT.rglob("*.parquet"))
    page_candidates: dict[Path, dict[int, set[str]]] = {}
    for n, path in enumerate(files, start=1):
        source = next((media for media in PAGE_KEYS if media in path.parts), None)
        if source not in PAGE_KEYS:
            if n % 100 == 0:
                print(f"[cached news links] scanned {n}/{len(files)} files; matched {len(page_candidates)} files", flush=True)
            continue
        try:
            parquet = pq.ParquetFile(path)
            schema = parquet.schema_arrow
            names = set(schema.names)
            link_col = next((column for column in PAGE_KEYS[source] if column in names), None)
            if not link_col:
                continue
            text_cols = [column for column in ["article_text_extracted", "article_text", "body", "content_txt"] if column in names]
            if "page_source" not in names:
                text_cols += [column for column in ["page_source_error", "error"] if column in names]
        except Exception as exc:
            page_errors.append({"file": str(path), "error": str(exc)})
            if n % 100 == 0:
                print(f"[cached news links] scanned {n}/{len(files)} files; matched {len(page_candidates)} files", flush=True)
            continue
        for group_index in range(parquet.metadata.num_row_groups):
            try:
                links = parquet.read_row_group(group_index, columns=[link_col]).column(0).to_pylist()
            except Exception as exc:
                page_errors.append({"file": str(path), "row_group": group_index, "error": str(exc)})
                continue
            group_keys = set()
            for value in links:
                link = normalize_link(value)
                key = (source, link)
                if not link or key not in pending:
                    continue
                if key not in found or not found[key].get("recovered_text"):
                    group_keys.add(link)
            if group_keys:
                page_candidates.setdefault(path, {})[group_index] = group_keys
        if n % 100 == 0:
            print(f"[cached news links] scanned {n}/{len(files)} files; relevant files={len(page_candidates)}", flush=True)

    # Load text columns only for groups with a URL that is actually missing from the master.
    html_candidates: dict[Path, dict[int, set[str]]] = {}
    for file_number, (path, groups) in enumerate(page_candidates.items(), start=1):
        source = next(media for media in PAGE_KEYS if media in path.parts)
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        link_col = next(column for column in PAGE_KEYS[source] if column in names)
        text_cols = [column for column in ["article_text_extracted", "article_text", "content_txt", "body"] if column in names]
        for group_index, group_links in groups.items():
            try:
                columns = [link_col] + text_cols + [column for column in ["page_source_error", "error"] if column in names]
                table = parquet.read_row_group(group_index, columns=columns)
                rows = table.to_pydict()
            except Exception as exc:
                page_errors.append({"file": str(path), "row_group": group_index, "error": str(exc)})
                continue
            for row_index, value in enumerate(rows.get(link_col, [])):
                link = normalize_link(value)
                if link not in group_links:
                    continue
                key = (source, link)
                text = ""
                text_column = ""
                for column in ("article_text_extracted", "article_text", "content_txt", "body"):
                    values = rows.get(column, [])
                    text_value = values[row_index] if row_index < len(values) else None
                    if isinstance(text_value, str) and text_value.strip():
                        text = text_value.strip()
                        text_column = column
                        break
                candidate = {
                    **pending[key], "recovered_text": text,
                    "page_archive_file": str(path),
                    "page_source_error": rows.get("page_source_error", rows.get("error", [""]))[row_index] or "",
                    "cached_page_found": True,
                    "extraction_source_column": text_column,
                }
                if key not in found or len(text or "") > len(found[key].get("recovered_text", "")):
                    found[key] = candidate
                if not text and "page_source" in names:
                    html_candidates.setdefault(path, {}).setdefault(group_index, set()).add(link)
            del table, rows
            gc.collect()
        if file_number % 25 == 0 or file_number == len(page_candidates):
            print(f"[cached text columns] scanned {file_number}/{len(page_candidates)} relevant files; text links={sum(bool(v.get('recovered_text')) for v in found.values())}", flush=True)

    # Prefer files with many needed URLs per byte and parse each article URL once.
    def candidate_density(item: tuple[Path, dict[int, set[str]]]) -> float:
        path, groups = item
        match_count = sum(len(links) for links in groups.values())
        return match_count / max(path.stat().st_size, 1)

    html_items = sorted(html_candidates.items(), key=candidate_density, reverse=True)
    html_attempted: set[tuple[str, str]] = set()
    html_candidate_keys = {
        (next(media for media in PAGE_KEYS if media in path.parts), link)
        for path, groups in html_items for links in groups.values() for link in links
    }

    # Page HTML is streamed in batches and each still-missing URL is parsed at most once.
    for file_number, (path, groups) in enumerate(html_items, start=1):
        source = next(media for media in PAGE_KEYS if media in path.parts)
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        link_col = next(column for column in PAGE_KEYS[source] if column in names)
        columns = [link_col, "page_source"] + [column for column in ["page_source_error", "error"] if column in names]
        for group_index, group_links in groups.items():
            links_to_read = {
                link for link in group_links
                if (source, link) not in html_attempted
                and not found.get((source, link), {}).get("recovered_text")
            }
            if not links_to_read:
                continue
            try:
                batches = parquet.iter_batches(batch_size=256, row_groups=[group_index], columns=columns)
            except Exception as exc:
                page_errors.append({"file": str(path), "row_group": group_index, "error": str(exc)})
                continue
            for batch_number, batch in enumerate(batches, start=1):
                rows = batch.to_pydict()
                for row_index, value in enumerate(rows.get(link_col, [])):
                    link = normalize_link(value)
                    if link not in links_to_read:
                        continue
                    key = (source, link)
                    if key in html_attempted:
                        continue
                    if found.get(key, {}).get("recovered_text"):
                        html_attempted.add(key)
                        continue
                    html_values = rows.get("page_source", [])
                    html = html_values[row_index] if row_index < len(html_values) else ""
                    text, parse_error = extract_cached_html(source, html)
                    error_values = rows.get("page_source_error", rows.get("error", []))
                    source_error = error_values[row_index] if row_index < len(error_values) else ""
                    found[key] = {
                        **pending[key], "recovered_text": text,
                        "page_archive_file": str(path),
                        "page_source_error": parse_error or source_error or "",
                        "cached_page_found": True,
                        "extraction_source_column": "page_source HTML parser" if text else "",
                    }
                    if isinstance(html, str) and html.strip():
                        html_attempted.add(key)
                del batch, rows
                gc.collect()
                if batch_number % 20 == 0:
                    print(
                        f"[cached HTML] file {file_number}/{len(html_candidates)} "
                        f"{path.name}; batch {batch_number}; "
                        f"text links={sum(bool(v.get('recovered_text')) for v in found.values())}",
                        flush=True,
                    )
        if file_number % 25 == 0 or file_number == len(html_candidates):
            print(
                f"[cached HTML] parsed {file_number}/{len(html_items)} candidate files; "
                f"attempted URLs={len(html_attempted)}/{len(html_candidate_keys)}; "
                f"text links={sum(bool(v.get('recovered_text')) for v in found.values())}",
                flush=True,
            )

    all_records = []
    for key, item in pending.items():
        if key in found:
            record = found[key]
            record["text_recovered"] = bool(record.get("recovered_text", "").strip())
            record["recovery_status"] = "recovered_from_local_page_archive" if record["text_recovered"] else "cached_page_found_but_no_text"
        else:
            record = {**item, "recovered_text": "", "page_archive_file": "", "page_source_error": "", "cached_page_found": False, "extraction_source_column": "", "text_recovered": False, "recovery_status": "no_matching_local_page"}
        all_records.append(record)
    result = pd.DataFrame(all_records)
    result.to_csv(target, index=False, encoding="utf-8-sig")
    result.to_parquet(output / "missing_text_from_cached_pages.parquet", index=False, compression="zstd")
    recovered = result[result["text_recovered"]].copy()
    unrecovered = result[~result["text_recovered"]].copy()
    recovered.to_csv(output / "recovered_missing_article_text.csv", index=False, encoding="utf-8-sig")
    recovered.to_parquet(output / "recovered_missing_article_text.parquet", index=False, compression="zstd")
    unrecovered.to_csv(output / "unrecovered_article_text_gaps.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(page_errors).to_csv(output / "page_archive_read_errors.csv", index=False, encoding="utf-8-sig")
    print({"target_rows_unique_articles": len(result), "recovered_texts": len(recovered), "not_recovered": len(unrecovered), "matched_cached_pages": int(result["cached_page_found"].sum()), "read_errors": len(page_errors), "media": result.groupby("media")["recovery_status"].value_counts().to_dict()}, flush=True)


if __name__ == "__main__":
    main()
