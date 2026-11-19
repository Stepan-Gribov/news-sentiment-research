"""Record what headless Chrome receives for supplied RBC URLs."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from dissertation_pipeline.news.rbc_archive import (
    _build_browser,
    extract_next_material,
    parse_html_article,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait", type=float, default=45)
    args = parser.parse_args()
    driver = _build_browser()
    results = []
    try:
        for url in args.url:
            result = {"requested_url": url}
            try:
                driver.get(url)
                deadline = time.monotonic() + args.wait
                material = {}
                html = ""
                while time.monotonic() < deadline:
                    time.sleep(0.5)
                    html = driver.page_source
                    try:
                        material = extract_next_material(html)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        material = {}
                    if material:
                        break
                result.update(
                    final_url=driver.current_url,
                    title=driver.title,
                    html_bytes=len(html.encode("utf-8", errors="ignore")),
                    material_found=bool(material),
                    body_sample=driver.find_element("tag name", "body").text[:2000],
                )
                fallback = parse_html_article(html, url, driver.current_url)
                result.update(
                    fallback_status=fallback["parse_status"],
                    fallback_title=fallback["title"],
                    fallback_text_length=fallback["text_length"],
                )
            except Exception as exc:
                result["exception"] = f"{type(exc).__name__}: {exc}"
            results.append(result)
    finally:
        try:
            driver.quit()
        except Exception:
            pass
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
