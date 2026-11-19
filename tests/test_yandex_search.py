from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from dissertation_pipeline.news.search_engines.yandex import (
    DEFAULT_QUERY_PROFILES,
    DateWindow,
    SearchPage,
    YandexWindowCollector,
    build_query,
    monthly_windows,
    parse_search_xml,
    split_window,
)


SAMPLE_XML = """<?xml version="1.0" encoding="utf-8"?>
<yandexsearch version="1.0">
  <response date="20260719T120000">
    <found priority="all">321</found>
    <results>
      <grouping>
        <group>
          <doc id="1">
            <url>https://example.com/gazprom-disclosure</url>
            <domain>example.com</domain>
            <title><hlword>Газпром</hlword> раскрыл отчетность по МСФО</title>
            <modtime>20260131</modtime>
            <passages>
              <passage>Опубликована <hlword>отчетность</hlword> эмитента.</passage>
            </passages>
            <mime-type>text/html</mime-type>
          </doc>
        </group>
      </grouping>
    </results>
  </response>
</yandexsearch>
""".encode("utf-8")


class YandexSearchTests(unittest.TestCase):
    def test_monthly_windows_use_exclusive_end(self) -> None:
        windows = monthly_windows("2026-01-15", "2026-03-01")
        self.assertEqual(
            [(window.start, window.end) for window in windows],
            [
                (date(2026, 1, 15), date(2026, 1, 31)),
                (date(2026, 2, 1), date(2026, 2, 28)),
            ],
        )

    def test_split_window_has_no_overlap_or_gap(self) -> None:
        left, right = split_window(DateWindow(date(2026, 1, 1), date(2026, 1, 31)))
        self.assertEqual(left.end.toordinal() + 1, right.start.toordinal())
        self.assertEqual(left.days + right.days, 31)

    def test_query_contains_profile_and_date_range(self) -> None:
        query = build_query(
            "Газпром",
            DEFAULT_QUERY_PROFILES["disclosure"],
            DateWindow(date(2026, 1, 1), date(2026, 1, 31)),
        )
        self.assertIn("Газпром", query)
        self.assertIn("раскрытие информации", query)
        self.assertIn("date:20260101..20260131", query)

    def test_parse_xml_preserves_nested_highlight_text(self) -> None:
        page = parse_search_xml(SAMPLE_XML, page=0, groups_on_page=100)
        self.assertEqual(page.total_found, 321)
        self.assertEqual(len(page.rows), 1)
        self.assertEqual(page.rows[0]["link"], "https://example.com/gazprom-disclosure")
        self.assertEqual(page.rows[0]["title"], "Газпром раскрыл отчетность по МСФО")
        self.assertIn("отчетность эмитента", page.rows[0]["snippet"])

    def test_saturated_window_is_split_until_unsaturated(self) -> None:
        class FakeClient:
            groups_on_page = 100

            def search_page(self, query_text, page, cache_path):
                date_range = query_text.rsplit("date:", 1)[1]
                start_text, end_text = date_range.split("..")
                start = date.fromisoformat(
                    f"{start_text[:4]}-{start_text[4:6]}-{start_text[6:]}"
                )
                end = date.fromisoformat(
                    f"{end_text[:4]}-{end_text[4:6]}-{end_text[6:]}"
                )
                days = (end - start).days + 1
                if days > 2:
                    total = 300
                    page_size = 100 if page < 2 else 50
                else:
                    total = 1
                    page_size = 1 if page == 0 else 0
                rows = [
                    {
                        "rank": page * 100 + i + 1,
                        "link": f"https://example.com/{start_text}/{page}/{i}",
                        "domain": "example.com",
                        "title": "Газпром раскрыл отчетность",
                        "snippet": "Существенный факт эмитента",
                        "engine_date": start_text,
                        "mime_type": "text/html",
                        "saved_copy_url": "",
                    }
                    for i in range(page_size)
                ]
                return SearchPage(rows=rows, total_found=total, raw_xml=b"<xml/>", cache_path=cache_path)

        with TemporaryDirectory() as tmp:
            collector = YandexWindowCollector(
                FakeClient(),
                raw_dir=Path(tmp),
                min_window_days=1,
            )
            _, diagnostics = collector.collect_adaptive(
                company="Газпром",
                profile=DEFAULT_QUERY_PROFILES["disclosure"],
                window=DateWindow(date(2026, 1, 1), date(2026, 1, 4)),
            )

        self.assertEqual(len(diagnostics), 3)
        self.assertTrue(diagnostics[0]["split"])
        self.assertFalse(diagnostics[1]["saturated"])
        self.assertFalse(diagnostics[2]["saturated"])


if __name__ == "__main__":
    unittest.main()
