from __future__ import annotations

import unittest

from dissertation_pipeline.news.search_engines.gdelt import (
    DEFAULT_GDELT_PROFILES,
    build_gdelt_query,
    parse_gdelt_json,
)


SAMPLE_JSON = """{
  "articles": [{
    "url": "https://example.com/gazprom",
    "url_mobile": "",
    "title": "Газпром раскрыл отчетность по МСФО",
    "seendate": "20260501T120000Z",
    "socialimage": "https://example.com/image.jpg",
    "domain": "example.com",
    "language": "Russian",
    "sourcecountry": "Russia"
  }]
}""".encode("utf-8")


class GdeltSearchTests(unittest.TestCase):
    def test_query_uses_translated_term_and_russian_source_filter(self) -> None:
        query = build_gdelt_query("gazprom", DEFAULT_GDELT_PROFILES["disclosure"])
        self.assertIn("gazprom", query)
        self.assertIn("information disclosure", query)
        self.assertIn("sourcelang:russian", query)

    def test_parse_article_list(self) -> None:
        response = parse_gdelt_json(SAMPLE_JSON)
        self.assertEqual(len(response.rows), 1)
        self.assertEqual(response.rows[0]["domain"], "example.com")
        self.assertEqual(response.rows[0]["language"], "Russian")


if __name__ == "__main__":
    unittest.main()
