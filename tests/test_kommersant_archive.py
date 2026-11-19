from pathlib import Path

from dissertation_pipeline.news.kommersant_archive import (
    HttpClient,
    HttpResult,
    download_candidate,
    extract_doc_id,
    parse_article_html,
    parse_sitemap_locations,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_extract_doc_id():
    assert extract_doc_id("https://www.kommersant.ru/doc/8906920") == 8906920
    assert extract_doc_id("https://www.kommersant.ru/doc/8906920?from=main") == 8906920
    assert extract_doc_id("https://www.kommersant.ru/archive") is None


def test_parse_sitemap_locations():
    xml = """<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url><loc>https://www.kommersant.ru/doc/1</loc><lastmod>1991-07-08</lastmod></url>
    </urlset>"""
    assert parse_sitemap_locations(xml) == [
        {"loc": "https://www.kommersant.ru/doc/1", "lastmod": "1991-07-08"}
    ]


def test_parse_modern_article_without_storing_html():
    html = """
    <html><head>
      <link rel="canonical" href="https://www.kommersant.ru/doc/8906920">
      <meta property="article:published_time" content="2026-08-24T13:30:24+03:00">
      <meta property="og:title" content="Ритейл в поисках устойчивости">
    </head><body>
      <article class="doc js-article" data-article-docsid="8906920"
        data-article-rubric-name="Потребительский рынок"
        data-article-publishing-name="Review Online">
        <div class="article_text_wrapper">
          <p class="doc__text">Первый абзац.</p>
          <p class="doc__thought">Повторная цитата.</p>
          <p class="doc__text">Второй абзац.</p>
          <p class="doc__text document_authors">Автор Тестов</p>
        </div>
      </article>
    </body></html>
    """
    row = parse_article_html(html, "https://www.kommersant.ru/doc/8906920")
    assert row["doc_id"] == 8906920
    assert row["timestamp"] == "2026-08-24T13:30:24+03:00"
    assert row["category"] == "Потребительский рынок"
    assert row["authors"] == "Автор Тестов"
    assert row["article_text_extracted"] == "Первый абзац.\nВторой абзац."
    assert "html" not in row


def test_parse_legacy_font_and_br_article():
    html = """
    <html><head><link rel="canonical" href="https://www.kommersant.ru/doc/1"></head>
    <body><article class="doc js-article" data-analytics-date="1991-07-08"
      data-article-title="8&amp;mdash;14 июля">
      <div class="doc__body"><div class="article_text_wrapper">
        <font><b>Первый старый абзац.</b></font><br>Второй старый абзац.<br>
      </div></div>
    </article></body></html>
    """
    row = parse_article_html(html, "https://www.kommersant.ru/doc/1")
    assert row["doc_id"] == 1
    assert row["title"] == "8—14 июля"
    assert row["timestamp"] == "1991-07-08"
    assert row["article_text_extracted"] == "Первый старый абзац.\nВторой старый абзац."


def test_nonretryable_external_redirect_is_terminal():
    class FakeClient:
        def get(self, url):
            return HttpResult(
                status=400,
                final_url="https://www.autopilot.ru/issues/auto/1998/10/example.html",
                text="",
                response_bytes=37,
                error="HTTP 400",
            )

    row = download_candidate(
        FakeClient(),
        {
            "doc_id": 593100,
            "url": "https://www.kommersant.ru/doc/593100",
            "sitemap_month": "1998-10",
            "sitemap_lastmod": "",
        },
    )
    assert row["parse_status"] == "external_redirect"
    assert row["http_status"] == 400
