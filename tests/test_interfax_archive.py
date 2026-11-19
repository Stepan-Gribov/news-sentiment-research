from dissertation_pipeline.news.interfax_archive import (
    extract_article_identity,
    parse_article_html,
    parse_sitemap_locations,
    sitemap_group,
)


def test_article_filter_accepts_editorial_and_rejects_service_pages() -> None:
    assert extract_article_identity("https://www.interfax.ru/business/1081091") == (
        "business",
        1081091,
    )
    assert extract_article_identity("https://www.interfax.ru/ConfederationsCup2017/30923") == (
        "confederationscup2017",
        30923,
    )
    assert extract_article_identity("https://www.interfax.ru/photo/7343/76244") is None
    assert extract_article_identity("https://www.interfax.ru/pressreleases/1080967") is None
    assert extract_article_identity("https://www.interfax.ru/news/2008/02/11") is None


def test_sitemap_group_and_xml_parser() -> None:
    assert sitemap_group("https://www.interfax.ru/SEO_SiteMap_18.xml") == ("map-018", 18)
    assert sitemap_group("https://www.interfax.ru/SEO_SiteMapLastChanges.xml") == ("latest", None)
    xml = '''<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://www.interfax.ru/russia/914</loc><lastmod>2008-02-20</lastmod></url>
    </urlset>'''
    assert parse_sitemap_locations(xml) == [
        {"loc": "https://www.interfax.ru/russia/914", "lastmod": "2008-02-20"}
    ]


def test_parse_interfax_article_html() -> None:
    html = '''
    <html><head>
      <meta property="og:url" content="https://www.interfax.ru/business/1081091">
      <meta property="article:published_time" content="2026-03-31T10:29+03:00">
      <meta property="article:section" content="Экономика">
      <meta property="article:tag" content="РФ">
      <meta property="article:tag" content="газ">
      <meta name="description" content="Краткое описание">
    </head><body>
      <article itemprop="articleBody">
        <h1 itemprop="headline">Заголовок Интерфакса</h1>
        <p>Первый абзац.</p><p></p><p>Второй абзац.</p>
        <div itemprop="author"><meta itemprop="name" content="Интерфакс"></div>
      </article>
    </body></html>
    '''
    row = parse_article_html(html, "https://www.interfax.ru/business/1081091")
    assert row["parse_status"] == "ok"
    assert row["article_id"] == 1081091
    assert row["publication_date"] == "2026-03-31"
    assert row["date_source"] == "page_metadata"
    assert row["section"] == "Экономика"
    assert row["tags"] == "РФ; газ"
    assert row["article_text_extracted"] == "Первый абзац.\nВторой абзац."
