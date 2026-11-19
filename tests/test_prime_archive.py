from dissertation_pipeline.news.prime_archive import (
    extract_article_identity,
    parse_article_html,
    parse_sitemap_locations,
    sitemap_month,
)


def test_extract_article_identity_with_and_without_slug() -> None:
    assert extract_article_identity("https://1prime.ru/20100129/758835772.html") == (
        "20100129",
        758835772,
    )
    assert extract_article_identity("https://1prime.ru/20260831/aeroflot--872822255.html") == (
        "20260831",
        872822255,
    )
    assert extract_article_identity("https://1prime.ru/services/") is None


def test_sitemap_month_and_xml_parser() -> None:
    url = "https://1prime.ru/sitemap_article.xml?date_start=20100101&date_end=20100131"
    assert sitemap_month(url) == "2010-01"
    xml = f'''<?xml version="1.0"?>
    <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <sitemap><loc>{url.replace('&', '&amp;')}</loc><lastmod>2010-01-31</lastmod></sitemap>
    </sitemapindex>'''
    assert parse_sitemap_locations(xml) == [{"loc": url, "lastmod": "2010-01-31"}]


def test_parse_prime_article_html() -> None:
    html = '''
    <html><head>
      <link rel="canonical" href="https://1prime.ru/20260831/aeroflot--872822255.html">
      <meta property="article:published_time" content="20260831T1406+0300">
      <meta property="article:author" content="ПРАЙМ">
      <meta property="article:section" content="Новости">
      <meta property="article:tag" content="Бизнес">
      <meta property="article:tag" content="РОССИЯ">
      <meta name="description" content="Краткое описание">
    </head><body>
      <h1 class="article__second-title">Заголовок новости</h1>
      <div class="article__body">
        <div class="article__text"><p>Первый абзац.</p></div>
        <div class="article__text"><p>Второй абзац.</p></div>
      </div>
    </body></html>
    '''
    row = parse_article_html(html, "https://1prime.ru/20260831/aeroflot--872822255.html")
    assert row["parse_status"] == "ok"
    assert row["article_id"] == 872822255
    assert row["title"] == "Заголовок новости"
    assert row["published_at"] == "2026-08-31T14:06:00+03:00"
    assert row["publication_date"] == "2026-08-31"
    assert row["date_source"] == "page_metadata"
    assert row["tags"] == "Бизнес; РОССИЯ"
    assert row["article_text_extracted"] == "Первый абзац.\nВторой абзац."
