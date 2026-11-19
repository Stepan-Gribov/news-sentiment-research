import gzip
import json
from concurrent.futures import Future

from dissertation_pipeline.news.rbc_archive import (
    download_content_part,
    extract_material_identity,
    markdown_to_text,
    part_is_complete,
    parse_material,
    sitemap_group,
)


def test_rbc_article_url_filter() -> None:
    assert extract_material_identity(
        "https://www.rbc.ru/politics/25/03/2003/5703b5579a7947783a5a4602"
    ) == ("politics", "5703b5579a7947783a5a4602", "2003-03-25")
    assert extract_material_identity(
        "https://www.rbc.ru/rbcfreenews/6a96bc150e0da85b90a0cf2a"
    ) == ("rbcfreenews", "6a96bc150e0da85b90a0cf2a", "")
    assert extract_material_identity("https://www.rbc.ru/story/6a02fe3a9a79472cfbafe6bf") is None
    assert extract_material_identity("https://www.rbc.ru/person/5f1234569a79470000000000") is None


def test_sitemap_group() -> None:
    assert sitemap_group("https://www.rbc.ru/sitemap_991.xml") == ("map-0991", 991)


def test_parse_material_and_markdown() -> None:
    material = {
        "id": "6a96ab51884e1199ae9cdb76",
        "fronturl": "https://www.rbc.ru/business/01/09/2026/6a96ab51884e1199ae9cdb76",
        "title": "Заголовок РБК",
        "publishDate": "Tue, 01 Sep 2026 12:00:00 +0300",
        "modifDate": "Tue, 01 Sep 2026 12:10:00 +0300",
        "metaDescription": "Описание статьи",
        "bodyMd": "Первый абзац.\n\nВторой с [ссылкой](https://example.com).",
        "category": {"name": "Бизнес"},
        "categories": [{"name": "Бизнес"}],
        "tags": [{"title": "Wildberries"}],
        "author": [{"name": "Иван Иванов"}],
        "tickers": [{"ticker": "WILD"}],
        "companyCards": [{"name": "РВБ"}],
        "projectNick": "rbcnews",
        "type": "article",
        "isPaywall": False,
    }
    row = parse_material(
        material,
        "https://www.rbc.ru/business/01/09/2026/6a96ab51884e1199ae9cdb76",
    )
    assert row["parse_status"] == "ok"
    assert row["publication_date"] == "2026-09-01"
    assert row["title"] == "Заголовок РБК"
    assert row["authors"] == "Иван Иванов"
    assert row["tickers"] == "WILD"
    assert row["companies"] == "РВБ"
    assert "Второй с ссылкой." in row["article_text_extracted"]
    assert markdown_to_text("A [link](https://example.com)") == "A link"


def test_content_part_persists_request_error_without_blocking(tmp_path) -> None:
    candidates = [{"material_id": "ok-id"}, {"material_id": "failed-id"}]
    results = {
        "ok-id": {
            "requested_material_id": "ok-id",
            "parse_status": "ok",
            "is_paywall": False,
        },
        "failed-id": {
            "requested_material_id": "failed-id",
            "parse_status": "request_error",
            "error": "browser_timeout",
        },
    }

    class ImmediateExecutor:
        def submit(self, _function, candidate):
            future = Future()
            future.set_result(results[candidate["material_id"]])
            return future

    path = tmp_path / "part-00000.jsonl.gz"
    stats = download_content_part(
        ImmediateExecutor(),
        candidates,
        path,
        force=False,
        output_format="jsonl.gz",
    )

    assert stats["ok"] == 1
    assert stats["errors"] == 1
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    assert [row["requested_material_id"] for row in rows] == ["ok-id", "failed-id"]
    assert rows[1]["is_paywall"] is None
    assert part_is_complete(path, ["ok-id", "failed-id"], "jsonl.gz") is False
