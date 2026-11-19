"""Audit and document the separate external raw-data package.

This script creates only source-coverage and data-dictionary files. It does not
alter source datasets or calculate analytical variables.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


DATA_ROOT = Path("./")
PACKAGE = DATA_ROOT / "market_data/raw_external_2026-09-30"
NEWS_MASTER = DATA_ROOT / "news_data/merged/articles_total_rbc_dates_restored.parquet"
NEWS_LIMIT = pd.Timestamp("2026-05-29 23:59:59")
CORE_MEDIA = ["kommersant", "prime", "interfax", "rbc"]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def moscow_dates_from_ms(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, unit="ms", utc=True, errors="coerce").dt.tz_convert("Europe/Moscow").dt.tz_localize(None)


def add_coverage(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    file_path: str,
    source: str,
    source_url: str,
    frequency: str,
    units: str,
    date_field: str,
    frame: pd.DataFrame,
    dates: pd.Series | None = None,
    duplicate_keys: list[str] | None = None,
    notes: str = "",
) -> None:
    valid_dates = dates.dropna() if dates is not None else pd.Series(dtype="object")
    if len(valid_dates):
        minimum, maximum = str(valid_dates.min()), str(valid_dates.max())
    else:
        minimum, maximum = "", ""
    duplicate_key_rows = int(frame.duplicated(duplicate_keys).sum()) if duplicate_keys else int(frame.duplicated().sum())
    rows.append({
        "dataset": dataset,
        "file_path": file_path,
        "source": source,
        "source_url_or_api": source_url,
        "frequency": frequency,
        "units": units,
        "date_field": date_field,
        "actual_min_date": minimum,
        "actual_max_date": maximum,
        "rows": int(len(frame)),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows_on_documented_key": duplicate_key_rows,
        "duplicate_key": ",".join(duplicate_keys or []),
        "notes": notes,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-generated-docs", action="store_true",
        help="refresh only generated coverage/documentation outputs in the new package",
    )
    args = parser.parse_args()
    expected_news = [
        PACKAGE / "news/missing_text_from_cached_pages.csv",
        PACKAGE / "news/recovered_missing_article_text.csv",
        PACKAGE / "news/unrecovered_article_text_gaps.csv",
        PACKAGE / "news/page_archive_read_errors.csv",
    ]
    absent = [path for path in expected_news if not path.exists()]
    if absent:
        raise FileNotFoundError("News cache recovery has not finished; missing: " + ", ".join(map(str, absent)))

    output_paths = [
        PACKAGE / "source_coverage.csv",
        PACKAGE / "unresolved_series.csv",
        PACKAGE / "investfunds/investfunds_daily_count_source_date_gaps.csv",
        PACKAGE / "news/existing_news_text_coverage_by_media.csv",
        PACKAGE / "README_Data_Dictionary.md",
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.refresh_generated_docs:
        raise FileExistsError("Refusing to replace existing documentation outputs: " + ", ".join(map(str, existing)))

    coverage: list[dict[str, Any]] = []
    indices_dir = PACKAGE / "moex/indices"
    for path in sorted(indices_dir.glob("*_daily_history_2012-01-01_2026-08-31.csv")):
        frame = read_csv(path)
        if "TRADEDATE" in frame:
            dates = pd.to_datetime(frame["TRADEDATE"], errors="coerce").dt.strftime("%Y-%m-%d")
        else:
            dates = None
        code = path.name.split("_daily_history_")[0]
        add_coverage(
            coverage, dataset=f"MOEX index {code}", file_path=str(path.relative_to(PACKAGE)),
            source="Moscow Exchange ISS", source_url=f"https://iss.moex.com/iss/history/engines/stock/markets/index/securities/{code}.json",
            frequency="daily trading sessions", units="index points; raw ISS CLOSE/OPEN/HIGH/LOW and source fields retained",
            date_field="TRADEDATE", frame=frame, dates=dates, duplicate_keys=["TRADEDATE"],
        )

    usd_path = PACKAGE / "central_bank/USD_RUB_official_daily.csv"
    usd = read_csv(usd_path)
    usd_dates = pd.to_datetime(usd["Date"], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")
    add_coverage(
        coverage, dataset="Official USD/RUB rate", file_path=str(usd_path.relative_to(PACKAGE)),
        source="Bank of Russia", source_url="https://www.cbr.ru/scripts/XML_dynamic.asp?date_req1=01%2F01%2F2012&date_req2=31%2F08%2F2026&VAL_NM_RQ=R01235",
        frequency="official fixing dates", units="source XML fields Nominal, Value, VunitRate; VunitRate is RUB per USD",
        date_field="Date", frame=usd, dates=usd_dates, duplicate_keys=["Date"],
        notes="No source fixing was returned for 2026-08-30 or 2026-08-31; no value was added.",
    )

    ruonia_path = PACKAGE / "central_bank/RUONIA_daily.csv"
    ruonia = read_csv(ruonia_path)
    ruonia_dates = pd.to_datetime(ruonia["DT"], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")
    add_coverage(
        coverage, dataset="RUONIA and source calculation fields", file_path=str(ruonia_path.relative_to(PACKAGE)),
        source="Bank of Russia", source_url="https://cbr.ru/hd_base/ruonia/dynamics/",
        frequency="daily published RUONIA observations", units="ruo in percent p.a.; vol in RUB bn; remaining columns as source",
        date_field="DT", frame=ruonia, dates=ruonia_dates, duplicate_keys=["DT"],
        notes=f"Core rate ruo missing values={int(ruonia['ruo'].isna().sum())}; auxiliary source fields are blank for parts of the early period.",
    )

    universe_path = PACKAGE / "moex/securities/holdings_exact_isin_universe.csv"
    universe = read_csv(universe_path)
    holdings_dates = pd.concat([pd.to_datetime(universe[c], errors="coerce") for c in ["exact_first_date", "exact_last_date"]]).dropna()
    add_coverage(
        coverage, dataset="Exact holdings ISIN universe", file_path=str(universe_path.relative_to(PACKAGE)),
        source="Project exact-fund-holdings archive", source_url="source archive referenced in project; see project data manifest",
        frequency="one row per unique ISIN", units="unique instrument identifiers and exact holdings observation coverage",
        date_field="exact_first_date/exact_last_date", frame=universe, dates=holdings_dates.dt.strftime("%Y-%m-%d"),
        duplicate_keys=["requested_isin"], notes="3,262 exact-holdings ISINs; holdings source archive ends 2026-05-29.",
    )

    quote_files = [
        (PACKAGE / "moex/securities/existing_local_All_instr_exact_holdings_quotes_2024-10_2025-11.csv", "Existing local MOEX share/stock history matched through unique SECID mapping", "daily trading sessions"),
        (PACKAGE / "moex/securities/daily_quote_gap_2026-07-01_2026-08-31/quotes_exact_holdings_2026-07-01_2026-08-31.csv", "MOEX ISS; exact-holdings ISIN mapped to SECID", "daily trading sessions"),
    ]
    for path, source_note, frequency in quote_files:
        quotes = read_csv(path)
        date_series = pd.to_datetime(quotes["TRADEDATE"], errors="coerce").dt.strftime("%Y-%m-%d")
        sec_column = "HOLDINGS_ISIN_BY_MOEX_MAPPING" if "HOLDINGS_ISIN_BY_MOEX_MAPPING" in quotes else "REQUESTED_ISIN"
        add_coverage(
            coverage, dataset=path.stem, file_path=str(path.relative_to(PACKAGE)), source=source_note,
            source_url="https://iss.moex.com/iss/ (historical securities endpoint)", frequency=frequency,
            units="raw MOEX ISS quote, turnover and instrument fields; no derived returns/volatility",
            date_field="TRADEDATE", frame=quotes, dates=date_series,
            duplicate_keys=[c for c in ["SECID", "BOARDID", "TRADEDATE"] if c in quotes],
        )

    unresolved_path = PACKAGE / "moex/securities/unresolved_moex_identifiers.csv"
    unresolved = read_csv(unresolved_path)
    add_coverage(
        coverage, dataset="Unresolved exact-holdings ISIN to MOEX SECID matches", file_path=str(unresolved_path.relative_to(PACKAGE)),
        source="MOEX ISS security finder", source_url="https://iss.moex.com/iss/securities.json?q={ISIN}",
        frequency="mapping audit snapshot", units="ISIN and source mapping status", date_field="exact_first_date/exact_last_date",
        frame=unresolved, dates=pd.concat([pd.to_datetime(unresolved["exact_first_date"], errors="coerce"), pd.to_datetime(unresolved["exact_last_date"], errors="coerce")]).dropna().dt.strftime("%Y-%m-%d"),
        duplicate_keys=["requested_isin"], notes="Only exact ISIN records were accepted as SECID matches.",
    )

    invest_root = PACKAGE / "investfunds"
    nav_path = invest_root / "fund_nav_and_unit_value_public_chart_data_verified.parquet"
    nav = pd.read_parquet(nav_path)
    nav["source_date_moscow"] = moscow_dates_from_ms(nav["timestamp_ms_source"])
    for data_key, part in nav.groupby("data_key", sort=True):
        add_coverage(
            coverage, dataset=f"Investfunds public {data_key} series", file_path=str(nav_path.relative_to(PACKAGE)),
            source="Investfunds public chart JSON", source_url="https://investfunds.ru/funds/{fund_id}/?action=chartData",
            frequency="source chart observation (daily as returned)", units="pay: unit price as published; sca: NAV as published; no resampling",
            date_field="timestamp_ms_source (Moscow calendar date)", frame=part,
            dates=part["source_date_moscow"].dt.strftime("%Y-%m-%d"),
            duplicate_keys=["fund_id", "data_key", "timestamp_ms_source"],
            notes=f"{part['fund_id'].nunique()} of 1,108 holdings fund IDs have points in this series.",
        )

    ranks_path = invest_root / "public_monthly_fund_inflow_ranking_verified.csv"
    ranks = read_csv(ranks_path)
    rank_dates = pd.to_datetime(ranks["ranking_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    add_coverage(
        coverage, dataset="Investfunds per-fund public attracted-funds ranking", file_path=str(ranks_path.relative_to(PACKAGE)),
        source="Investfunds public ranking", source_url="https://investfunds.ru/fund-rankings/fund-inflow/",
        frequency="monthly ranking snapshots", units="attracted funds, RUB (not a per-fund net flow series)",
        date_field="ranking_date", frame=ranks, dates=rank_dates, duplicate_keys=["ranking_date", "fund_id"],
        notes="61 of 176 requested monthly pages contained no parsed fund rows; see the paired failure list.",
    )

    stats_path = invest_root / "public_market_statistics_raw/fund_count_and_market_net_flows_public_source_points.parquet"
    stats = pd.read_parquet(stats_path)
    stats["source_date_moscow"] = moscow_dates_from_ms(stats["timestamp_ms_source"])
    series_names = {"0": "all funds", "1": "open", "2": "interval", "3": "closed", "5": "exchange-traded"}
    flow_names = {"4": "market total", "4001": "equity", "4003": "bond", "4004": "mixed", "4006": "money market", "4023": "precious metals"}
    for (key, series_id), part in stats.groupby(["data_key_source", "series_id_source"], sort=True):
        is_count = key == "quant"
        label = series_names.get(str(series_id), str(series_id)) if is_count else flow_names.get(str(series_id), str(series_id))
        if is_count:
            dates = part["source_date_moscow"].dt.strftime("%Y-%m-%d")
            frequency, units, period = "daily source observations", "number of funds", "daily count"
        else:
            dates = part["source_date_moscow"].dt.strftime("%Y-%m")
            frequency, units, period = "monthly source observations", "RUB; aggregate public-market flow as returned", "month"
        add_coverage(
            coverage, dataset=f"Investfunds {period}: {label}", file_path=str(stats_path.relative_to(PACKAGE)),
            source="Investfunds public market statistics page", source_url="https://investfunds.ru/funds-statistics/",
            frequency=frequency, units=units, date_field="timestamp_ms_source (Moscow calendar)", frame=part,
            dates=dates, duplicate_keys=["data_key_source", "series_id_source", "timestamp_ms_source"],
            notes=f"source series ID={series_id}; value_source is retained unchanged.",
        )

    # Preserve the two dates missing from the public daily count response; do not fill them.
    gaps = []
    for series_id, part in stats[stats["data_key_source"].eq("quant")].groupby("series_id_source"):
        actual = set(part["source_date_moscow"].dt.strftime("%Y-%m-%d"))
        if not actual:
            continue
        expected_start = min(actual)
        expected = pd.date_range(expected_start, "2026-08-31", freq="D").strftime("%Y-%m-%d")
        for missing_date in sorted(set(expected) - actual):
            gaps.append({"series_id_source": series_id, "series_name": series_names.get(str(series_id), str(series_id)), "missing_source_date": missing_date, "status": "no source point; not filled"})
    gaps_path = PACKAGE / "investfunds/investfunds_daily_count_source_date_gaps.csv"
    pd.DataFrame(gaps).to_csv(gaps_path, index=False, encoding="utf-8-sig")

    # News master date/text status is summarized in streaming batches, not copied or modified.
    master = pq.ParquetFile(NEWS_MASTER)
    per_media: dict[str, dict[str, Any]] = {media: {"rows": 0, "blank_text_rows": 0, "missing_date_rows": 0, "dates": []} for media in CORE_MEDIA}
    for batch in master.iter_batches(batch_size=20000, columns=["media", "date", "text"]):
        frame = batch.to_pandas()
        dates = pd.to_datetime(frame["date"], errors="coerce")
        keep = frame["media"].isin(CORE_MEDIA) & (dates <= NEWS_LIMIT)
        selected = frame.loc[keep].copy()
        selected_dates = dates.loc[keep]
        blank = selected["text"].isna() | selected["text"].astype(str).str.strip().eq("")
        for media in CORE_MEDIA:
            mask = selected["media"].eq(media)
            info = per_media[media]
            info["rows"] += int(mask.sum())
            info["blank_text_rows"] += int((mask & blank).sum())
            info["missing_date_rows"] += int(pd.to_datetime(selected.loc[mask, "date"], errors="coerce").isna().sum())
            valid = selected_dates[mask].dropna()
            if len(valid):
                info["dates"].extend([valid.min(), valid.max()])
    news_coverage = []
    for media, info in per_media.items():
        dt = info["dates"]
        news_coverage.append({
            "media": media,
            "source_file": str(NEWS_MASTER),
            "rows_through_2026-05-29": info["rows"],
            "blank_text_rows": info["blank_text_rows"],
            "missing_date_rows": info["missing_date_rows"],
            "min_date": min(dt).date().isoformat() if dt else "",
            "max_date": max(dt).date().isoformat() if dt else "",
        })
    news_coverage_path = PACKAGE / "news/existing_news_text_coverage_by_media.csv"
    pd.DataFrame(news_coverage).to_csv(news_coverage_path, index=False, encoding="utf-8-sig")
    for path, dataset, note in [
        (PACKAGE / "news/recovered_missing_article_text.parquet", "Recovered news article text", "Only text recovered from already cached local article pages; no full-corpus redownload."),
        (PACKAGE / "news/unrecovered_article_text_gaps.csv", "Unrecovered news article text gaps", "Unique article URL/status list for manual or later source retrieval."),
    ]:
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = read_csv(path)
        date_col = "date"
        dates = pd.to_datetime(frame[date_col], errors="coerce").dt.strftime("%Y-%m-%d") if date_col in frame else None
        add_coverage(
            coverage, dataset=dataset, file_path=str(path.relative_to(PACKAGE)), source="Local article-page cache recovery",
            source_url="Article URLs retained in row-level link field", frequency="unique missing article URL",
            units="article text/status", date_field="date", frame=frame, dates=dates,
            duplicate_keys=[key for key in ["media", "link"] if key in frame], notes=note,
        )

    audit_path = PACKAGE / "news/missing_text_from_cached_pages.csv"
    audit = read_csv(audit_path)
    audit_dates = pd.to_datetime(audit["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    add_coverage(
        coverage, dataset="News local-cache recovery audit", file_path=str(audit_path.relative_to(PACKAGE)),
        source="Existing article-page cache; no network download", source_url="article URLs retained in each record",
        frequency="one row per unique media/article URL", units="source text, text-recovery status and local cache provenance",
        date_field="date", frame=audit, dates=audit_dates, duplicate_keys=["media", "link"],
        notes="One record per unique article link with blank text in the existing master; master archive was not modified.",
    )
    page_error_path = PACKAGE / "news/page_archive_read_errors.csv"
    page_errors = read_csv(page_error_path)
    add_coverage(
        coverage, dataset="News local page-cache read errors", file_path=str(page_error_path.relative_to(PACKAGE)),
        source="Existing article-page cache", source_url="local cache file paths in `file` field",
        frequency="one row per unreadable local cache file", units="read error description",
        date_field="", frame=page_errors, notes="These cache files could not be inspected and are not counted as text-recovered.",
    )

    coverage_path = PACKAGE / "source_coverage.csv"
    pd.DataFrame(coverage).to_csv(coverage_path, index=False, encoding="utf-8-sig")

    gap_rank = read_csv(invest_root / "public_monthly_fund_inflow_ranking_fetch_failures_verified.csv")
    gap_nav = read_csv(invest_root / "fund_nav_and_unit_value_fetch_gaps_verified.csv")
    missing_text = read_csv(PACKAGE / "news/unrecovered_article_text_gaps.csv")
    recovered_count = int(audit["text_recovered"].fillna(False).astype(bool).sum())
    recovered_by_media = audit.loc[audit["text_recovered"].fillna(False).astype(bool)].groupby("media").size().to_dict()
    cached_page_count = int(audit["cached_page_found"].fillna(False).astype(bool).sum())
    cached_no_text_count = cached_page_count - recovered_count
    no_cache_count = len(audit) - cached_page_count
    unresolved_moex = len(unresolved)
    no_quote = read_csv(PACKAGE / "moex/securities/unresolved_moex_histories_2026-07-01_2026-08-31.csv")
    quote_status = read_csv(PACKAGE / "moex/securities/moex_history_status_2026-07-01_2026-08-31.csv")
    quote_type_counts = quote_status.groupby("type", dropna=False).size().to_dict()
    quote_ok_by_type = (
        quote_status[quote_status["status_history"].eq("ok")]
        .groupby("type").size().to_dict()
    )
    quote_type_summary = "; ".join(
        f"{kind}: {int(count)} resolved, {int(quote_ok_by_type.get(kind, 0))} with rows"
        for kind, count in sorted(quote_type_counts.items(), key=lambda x: str(x[0]))
    )
    corporate_bond_quote_rows = int(quote_ok_by_type.get("corporate_bond", 0))
    usd_last = usd_dates.dropna().max()
    README = f"""# Исходные рыночные данные для исследования — Data Dictionary

Пакет: `{PACKAGE}`. Он содержит только исходные ряды/файлы и диагностические списки покрытия. Исходные архивы проекта не перезаписаны. Производные доходности, волатильность, ликвидность, HHI, логарифмы, кризисные индикаторы и регрессии здесь не рассчитывались. Значения из источников не заполнялись нулями и не интерполировались.

Целевой срок рядов — 31.08.2026; целевая дата holdings — 29.05.2026. Обзор фактических рядов и пропусков — в [`source_coverage.csv`](source_coverage.csv). Каждая запись там сообщает источник/API, частоту, единицы, фактические min/max, строки, пустые ячейки и дубликаты на ключе, указанном в таблице.

Уже были в проекте: исходный holdings до 29.05.2026, общий новостной архив с датами, локальные частичные котировки (2024–2025) и индексные файлы до 2025. В новую папку выгружены ряды ЦБ, MOEX-индексы до августа 2026, публичные Investfunds ряды/статистика, котировочный топ-ап только за июль–август 2026 и отдельный результат восстановления текстов из локального кэша. Holdings и исходный новостной архив повторно не выгружались.

## Что собрано

| Блок | Исходные данные и фактическое покрытие | Файлы |
|---|---|---|
| Банк России | USD/RUB: {len(usd)} фиксов, {usd_dates.dropna().min()}–{usd_last}; RUONIA: {len(ruonia)} наблюдений, {ruonia_dates.dropna().min()}–{ruonia_dates.dropna().max()}. Основная ставка `ruo` без пропусков; ранние вспомогательные поля частично пусты. | `central_bank/` |
| MOEX — индексы | 12 исходных рядов. IMOEX, RVI, RGBI/RGBITR и актуальные корпоративные индексы доступны до 31.08.2026. `RUCBITR`/`RUCBICP` прекращаются 31.05.2023; новые корпоративные серии сохранены отдельно, без сшивания методологически разных рядов. | `moex/indices/` |
| MOEX — бумаги из holdings | Вселенная — 3 262 уникальных ISIN. SECID разрешён для 781, не разрешён для {unresolved_moex}. Имеются 4 187 локальных котировочных строк по 18 бумагам за 03.10.2024–28.11.2025 и 1 084 строки по тем же 18 бумагам за 01.07–31.08.2026. У {len(no_quote)} разрешённых ISIN в июле–августе нет торговых строк. Разбивка разрешённых идентификаторов и историй по типам: {quote_type_summary}. Корпоративных облигаций с торговыми строками в топ-апе: {corporate_bond_quote_rows}. | `moex/securities/`; списки: [`unresolved_moex_identifiers.csv`](moex/securities/unresolved_moex_identifiers.csv), [`unresolved_moex_histories_2026-07-01_2026-08-31.csv`](moex/securities/unresolved_moex_histories_2026-07-01_2026-08-31.csv) |
| Investfunds — цена пая / СЧА | `pay` и `sca`: по 1 219 610 точек, по 1 051 из 1 108 fund ID, даты 01.01.2012–31.08.2026. У 57 ID на каждый показатель нет точек; точный список в файле gaps. | `investfunds/fund_nav_and_unit_value_public_chart_data_verified.csv/.parquet` |
| Investfunds — net flows рынка и количество фондов | Открытая статистическая страница отдаёт месячные агрегированные притоки/оттоки по категориям (не по каждому фонду) и ежедневный count фондов по типу за целевой период. Сохранены исходные timestamp/value и IDs. | `investfunds/public_market_statistics_raw/` |
| Investfunds — рейтинги фондов | 48 341 строка рейтинга по 649 ID за месячные даты 2012–2026; 61/176 запрошенных страниц не содержали строк. Показатель «привлечённые средства» не является чистым потоком фонда. | `investfunds/public_monthly_fund_inflow_ranking_verified.csv/.parquet`, failure list |
| Новости | Исходный общий архив оставлен на месте. До 29.05.2026 по четырём СМИ проверены текст и дата; доступные тексты из локального page-cache сохранены отдельно, исходный мастер-архив не изменён. Полный результат проверки ссылок и ошибки кеша документированы отдельно. | `news/` |

## Data Dictionary

- `central_bank/USD_RUB_official_daily.csv`: исходные XML-поля ЦБ `Date`, `Id`, `Nominal`, `Value`, `VunitRate`; источник — динамика официальных курсов, код USD `R01235`. Фактический последний доступный фиксинг — {usd_last}; на 30–31.08.2026 отдельного значения endpoint не вернуло, поэтому даты не дополнены.
- `central_bank/RUONIA_daily.csv`: исходные поля XLSX ЦБ `DT`, `ruo`, `vol`, `T`, `C`, процентили/границы, `StatusXML`, `DateUpdate`. Пустые вспомогательные значения сохранены как есть.
- `moex/indices/*csv`: оригинальные поля MOEX ISS; в частности `TRADEDATE`, `SECID`, `CLOSE`, `VALUE`, `VOLUME`, `DURATION`, `YIELD`. `CLOSE` индекса — пункты; остальные значения остаются в единицах ISS.
- Котировки бумаг MOEX сохраняют биржевые поля: `BOARDID`, `TRADEDATE`, `SECID`, `NUMTRADES`, `VALUE`, `CLOSE`, параметры облигаций (`MATDATE`, `DURATION`, купонные/доходностные поля), если источник их вернул. `HOLDINGS_ISIN_BY_MOEX_MAPPING` — дополнительный ключ связи через точное соответствие ISIN–SECID; исходные поля не заменяет. Из старой локальной выгрузки исключены только ранее рассчитанные столбцы `ret`, `volatility`, `ret_volatility`, `yield_volatility`.
- `investfunds/fund_nav_and_unit_value_public_chart_data_verified.*`: `fund_id`, `data_key` (`pay`/`sca`), исходные `timestamp_ms_source`, `value_source`, название/tooltip серии. `pay` — опубликованная стоимость пая; `sca` — опубликованная СЧА. Timestamp переводится в московскую календарную дату только для отчёта покрытия, в data-файле сохраняется оригинальное значение.
- `investfunds/public_monthly_fund_inflow_ranking_verified.*`: исходные поля рейтинга, включая `ranking_date`, ID/название фонда, rank, специализацию, географию и текстовую величину привлечённых средств в RUB.
- `investfunds/public_market_statistics_raw/fund_count_and_market_net_flows_public_source_points.*`: `data_key_source` (`quant`/`inflow`), `data_view_source`, `series_id_source`, исходные `timestamp_ms_source`, `value_source`, запрошенный диапазон. `quant`: число фондов; `inflow`: агрегированный месячный поток в RUB, не индивидуальный фондовый ряд.
- Новости: `missing_text_from_cached_pages.csv` содержит уникальную пару СМИ–ссылка, исходную дату/заголовок и статус проверки локального page-cache; recovered/unrecovered файлы содержат ссылку и результат извлечения. `page_archive_read_errors.csv` содержит локальные ошибки чтения.

## Не покрытые данные и ограничения

- Полный MOEX-архив индивидуальных торгов за 2008–30.06.2026 описан в [MOEX_QUOTES.md](<./local/Library/CloudStorage/analyst@example.org/My Drive/4. Disserartion research/calculations/code/dissertation_pipeline/docs/MOEX_QUOTES.md>), но домашний ПК в этой сессии не отвечал по SSH. Поэтому его физическое наличие, целостность и фактические даты сейчас не подтверждены; новый пакет содержит только явно перечисленные локальные периоды и июль–августовскую дозагрузку.
- Публичный MOEX ISS manual указывает, что исторические торговые данные по отдельным бумагам (кроме индексов) доступны по подписке. В этой выгрузке точная маппинг-проверка дала {unresolved_moex} unmatched ISIN; см. CSV выше. Исторические рейтинги, отдельный полный архив дивидендов/корпоративных действий и исторические shares-outstanding/капитализация не получены. MOEX указывает корпоративные события и рейтинговые данные в Центре корпоративной информации, доступ предоставляется по запросу/договору.
- Полные per-fund net flows и полный датированный именованный список действующих ПИФ не получены. Публичные рейтинги Investfunds не покрывают 61 месячную страницу и показывают «привлечённые средства», а не чистые притоки/оттоки. В открытом Investfunds API заявлены фондовая статистика и NAV/паи, но доступ запрашивается у поставщика. Вместо списка имён получен публичный ежедневный агрегат количества фондов по типам.
- Новости: из {len(audit):,} ссылок с пустым текстом локально восстановлено {recovered_count:,} текста ({recovered_by_media.get('rbc', 0):,} РБК, {recovered_by_media.get('kommersant', 0):,} Коммерсантъ); {len(missing_text):,} ссылок остаются без текста. Из них для {cached_no_text_count:,} кеш-страница нашлась, но извлечь текст не удалось; для {no_cache_count:,} ссылок кеш не найден. {len(page_errors)} файла кеша Коммерсанта не читались из-за ошибок Parquet-footer. Полный корпус заново не скачивался; ссылки с датой/источником находятся в `news/unrecovered_article_text_gaps.csv`.
- Ежедневный count Investfunds не содержит точки для 20.07.2022 и 23.09.2022 в исходном ответе (по затронутым series ID); пустые даты перечислены в `investfunds/investfunds_daily_count_source_date_gaps.csv` и не заполнены.

## Первичные источники

- [Банк России: динамика официальных курсов и XML API](https://www.cbr.ru/development/SXML/) / [динамика официальных валютных курсов](https://www.cbr.ru/currency_base/dynamics/)
- [Банк России: дневная динамика RUONIA](https://cbr.ru/hd_base/ruonia/dynamics/)
- [MOEX ISS: интерфейс](https://www.moex.com/a2920), [руководство ISS](https://www.moex.com/files/4be999zbzp80bx2bgmwayrtyx0)
- [MOEX: индексы корпоративных облигаций](https://www.moex.com/a7478), [Центр корпоративной информации](https://www.moex.com/tsentr-korporativnoj-informatsii)
- [Investfunds: публичная статистика рынка ПИФ](https://investfunds.ru/funds-statistics/), [рейтинг привлечённых средств](https://investfunds.ru/fund-rankings/fund-inflow/), [API и порядок доступа](https://investfunds.ru/api/)
"""
    readme_path = PACKAGE / "README_Data_Dictionary.md"
    readme_path.write_text(README, encoding="utf-8")
    unresolved_rows = [
        {"source_block": "Bank of Russia USD/RUB", "uncovered_item": "2026-08-30 to 2026-08-31 fixing", "detail": f"Last returned source date is {usd_last}; no value filled.", "file_or_status": str(usd_path.relative_to(PACKAGE))},
        {"source_block": "MOEX securities", "uncovered_item": f"{unresolved_moex} exact holdings ISINs without exact SECID mapping", "detail": "See the row-level mapping audit.", "file_or_status": str(unresolved_path.relative_to(PACKAGE))},
        {"source_block": "MOEX securities", "uncovered_item": f"{len(no_quote)} mapped ISINs with empty 2026-07-01 to 2026-08-31 history", "detail": "Empty response is not treated as zero price or turnover.", "file_or_status": "moex/securities/unresolved_moex_histories_2026-07-01_2026-08-31.csv"},
        {"source_block": "MOEX bonds", "uncovered_item": "No exact-holdings row matched in the existing local non-share quote archive", "detail": "The existing 2024–2025 non-share file was left untouched; this package does not claim it covers holdings bonds.", "file_or_status": "analysis_outputs/quotes_data/moex_nonshare_history.csv (existing project file)"},
        {"source_block": "MOEX securities", "uncovered_item": "Full historical security history, corporate actions, ratings, issuer share counts/capitalisation", "detail": "Home PC archive unavailable over SSH; MOEX historical security data and CCI items require subscriber/access confirmation.", "file_or_status": "docs/MOEX_QUOTES.md (documentation only; not file-verified)"},
        {"source_block": "Investfunds", "uncovered_item": f"{len(gap_rank)} monthly per-fund ranking pages without parsed fund rows", "detail": "The public ranking metric is attracted funds, not net flow.", "file_or_status": "investfunds/public_monthly_fund_inflow_ranking_fetch_failures_verified.csv"},
        {"source_block": "Investfunds", "uncovered_item": f"{gap_nav['fund_id'].nunique()} fund IDs with no public pay/SCA series", "detail": "No series point returned; no values added.", "file_or_status": "investfunds/fund_nav_and_unit_value_fetch_gaps_verified.csv"},
        {"source_block": "Investfunds", "uncovered_item": "Complete historical monthly net flows by individual fund and named active-fund roster by month", "detail": "Public market-level flows/count are available; per-fund historical series/register requires additional source access.", "file_or_status": "investfunds/public_market_statistics_raw/"},
        {"source_block": "Investfunds", "uncovered_item": f"{len(gaps)} missing daily market count points", "detail": "Source API returned no point for these dates/series; no interpolation.", "file_or_status": "investfunds/investfunds_daily_count_source_date_gaps.csv"},
        {"source_block": "News", "uncovered_item": f"{len(missing_text)} unique article links without text recovered locally", "detail": "Cache-only recovery; the full news archive was not downloaded again.", "file_or_status": "news/unrecovered_article_text_gaps.csv"},
        {"source_block": "News", "uncovered_item": f"{len(page_errors)} unreadable local Kommersant cache files", "detail": "Parquet footer was invalid; source pages could not be inspected.", "file_or_status": "news/page_archive_read_errors.csv"},
    ]
    pd.DataFrame(unresolved_rows).to_csv(PACKAGE / "unresolved_series.csv", index=False, encoding="utf-8-sig")
    print({"source_coverage_rows": len(coverage), "news_media_rows": len(news_coverage), "count_dates_missing": len(gaps), "unresolved_series_rows": len(unresolved_rows), "readme": str(readme_path)}, flush=True)


if __name__ == "__main__":
    main()
