import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup


ROOT_DIR = Path(r"./local/NEWS_PAGES_PARQ")
OUTPUT_COLUMN = "article_text_extracted"
MAX_WORKERS = 1


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_joined_paragraphs(parts):
    cleaned = []
    seen = set()

    for part in parts:
        txt = normalize_text(part)
        if not txt:
            continue
        if txt in seen:
            continue
        seen.add(txt)
        cleaned.append(txt)

    return "\n".join(cleaned).strip()


def extract_from_nodes(nodes):
    return clean_joined_paragraphs([node.get_text(" ", strip=True) for node in nodes])


def extract_interfax_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    nodes = soup.select('[itemprop="articleBody"] p')
    if nodes:
        return extract_from_nodes(nodes)
    nodes = soup.select("article p")
    return extract_from_nodes(nodes)


def extract_kommersant_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    nodes = soup.select(".doc__text")
    if nodes:
        return extract_from_nodes(nodes)
    nodes = soup.select("article p")
    return extract_from_nodes(nodes)


def extract_prime_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    nodes = soup.select(".article__body .article__text")
    if nodes:
        return extract_from_nodes(nodes)

    body = soup.select_one('[itemprop="articleBody"]')
    if body:
        text = normalize_text(body.get_text(" ", strip=True))
        return text

    return ""


def extract_rbc_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for selector in ['[itemprop="articleBody"]', ".article__text", ".article__content"]:
        body = soup.select_one(selector)
        if body:
            text = normalize_text(body.get_text(" ", strip=True))
            if text:
                return text

    nodes = soup.select("main p")
    parts = []

    for node in nodes:
        txt = normalize_text(node.get_text(" ", strip=True))
        if not txt:
            continue
        if txt.startswith("Фото:"):
            continue
        parts.append(txt)

    return clean_joined_paragraphs(parts)


def extract_frank_text(row) -> str:
    content_txt = normalize_text(str(row.get("content_txt", "")))
    if content_txt:
        return content_txt

    html = row.get("content", "")
    if not isinstance(html, str) or not html.strip():
        html = row.get("page_source", "")
    if not isinstance(html, str) or not html.strip():
        return ""

    soup = BeautifulSoup(html, "html.parser")
    nodes = soup.select("article p")
    return extract_from_nodes(nodes)


def detect_source(folder_name: str) -> str:
    name = folder_name.lower()
    if "interfax" in name:
        return "interfax"
    if "kommersant" in name:
        return "kommersant"
    if "prime" in name:
        return "prime"
    if "rbc" in name:
        return "rbc"
    if "frank" in name:
        return "frank"
    raise ValueError(f"Unknown source folder: {folder_name}")


def extract_row_text(row, source: str) -> str:
    if source == "frank":
        return extract_frank_text(row)

    html_col = "page_source"
    html = row.get(html_col, "")
    if (not isinstance(html, str) or not html.strip()) and source == "rbc":
        return normalize_text(str(row.get("body", "")))
    if not isinstance(html, str) or not html.strip():
        return ""

    if source == "interfax":
        return extract_interfax_text(html)
    if source == "kommersant":
        return extract_kommersant_text(html)
    if source == "prime":
        return extract_prime_text(html)
    if source == "rbc":
        extracted = extract_rbc_text(html)
        if extracted:
            return extracted
        return normalize_text(str(row.get("body", "")))

    return ""


def needs_fill(value) -> bool:
    return not isinstance(value, str) or not value.strip()


def process_one_file(parquet_path: Path, source: str) -> dict[str, int]:
    df = pd.read_parquet(parquet_path)

    if OUTPUT_COLUMN not in df.columns:
        df[OUTPUT_COLUMN] = ""

    pending_mask = df[OUTPUT_COLUMN].apply(needs_fill)
    pending_count = int(pending_mask.sum())

    if pending_count == 0:
        print(f"Skipped: {parquet_path.name} (already filled)")
        return {"pending": 0, "filled": 0, "remaining": 0}

    print(f"Processing: {parquet_path.name} | rows to fill: {pending_count}")

    for idx in df.index[pending_mask]:
        df.at[idx, OUTPUT_COLUMN] = extract_row_text(df.loc[idx], source)

    filled_count = int((pending_mask & ~df[OUTPUT_COLUMN].apply(needs_fill)).sum())
    remaining_count = int(df[OUTPUT_COLUMN].apply(needs_fill).sum())
    temp_path = parquet_path.with_name(f".{parquet_path.name}.extract.tmp")
    df.to_parquet(temp_path, index=False)
    temp_path.replace(parquet_path)
    print(f"Saved: {parquet_path} | filled={filled_count} | remaining={remaining_count}")
    return {"pending": pending_count, "filled": filled_count, "remaining": remaining_count}


def is_already_processed(parquet_path: Path) -> bool:
    try:
        df = pd.read_parquet(parquet_path, columns=[OUTPUT_COLUMN])
    except Exception:
        return False

    if OUTPUT_COLUMN not in df.columns:
        return False

    return int(df[OUTPUT_COLUMN].apply(needs_fill).sum()) == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract article text from downloaded news-page parquet files.")
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    parser.add_argument("--media", nargs="*", choices=["frank", "interfax", "kommersant", "prime", "rbc"])
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--sample-files", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    folders = sorted([p for p in args.root.iterdir() if p.is_dir()])
    if args.media:
        folders = [folder for folder in folders if detect_source(folder.name) in args.media]
    if not folders:
        raise RuntimeError(f"No source folders found in {args.root}")

    grand_total = {"pending": 0, "filled": 0, "remaining": 0}
    for folder in folders:
        source = detect_source(folder.name)
        parquet_files = sorted(folder.glob("*.parquet"))
        if args.sample_files:
            parquet_files = parquet_files[: args.sample_files]

        print(f"\n=== {folder.name} | files: {len(parquet_files)} ===")

        pending_files = [p for p in parquet_files if not is_already_processed(p)]
        print(f"Pending files: {len(pending_files)}")

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(process_one_file, parquet_path, source): parquet_path
                for parquet_path in pending_files
            }

            for future in as_completed(futures):
                parquet_path = futures[future]
                try:
                    stats = future.result()
                    for key in grand_total:
                        grand_total[key] += stats[key]
                except Exception as e:
                    print(f"FAILED: {parquet_path.name} -> {e}")

    print(
        "Extraction summary: "
        f"pending={grand_total['pending']}, filled={grand_total['filled']}, "
        f"remaining={grand_total['remaining']}"
    )


if __name__ == "__main__":
    main()
