from __future__ import annotations

import argparse
import concurrent.futures
import pickle
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests


ISS_URL = "https://iss.moex.com/iss"
DEFAULT_FROM_DATE = "2012-01-01"
DEFAULT_TIMEOUT = 30
DEFAULT_RETRIES = 5
ROLLING_WINDOW = 30

# Lower rank means "prefer this board first" when the same instrument-date
# appears on several boards in the raw MOEX history.
BOARD_RANK = {
    "TQBR": 1,
    "TQTF": 2,
    "TQOB": 3,
    "TQCB": 4,
    "TQIR": 5,
    "TQPI": 6,
    "TQTD": 7,
    "EQRP": 8,
    "FQBR": 9,
}


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"


def _progress_line(stage: str, completed: int, total: int, start_time: float) -> None:
    elapsed = max(time.perf_counter() - start_time, 1e-9)
    rate = completed / elapsed
    remaining = max(total - completed, 0)
    eta = remaining / rate if rate > 0 else float("inf")
    share = completed / total if total else 1.0
    print(
        f"[{stage}] {completed}/{total} ({share:.1%}) | "
        f"elapsed={_format_duration(elapsed)} | eta={_format_duration(eta)}"
    )


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    funds_root = script_dir.parent.parent

    parser = argparse.ArgumentParser(
        description=(
            "Download MOEX price histories needed to reconstruct portfolio "
            "returns, volatility and liquidity for the thesis pipeline."
        )
    )
    parser.add_argument(
        "--universe-file",
        type=Path,
        default=funds_root / "Data" / "Y" / "Y_interpolated.csv",
        help="Path to Y_interpolated.csv with column 'ISIN конечного актива'.",
    )
    parser.add_argument(
        "--securities-export",
        type=Path,
        default=(
            funds_root
            / "Code"
            / "Data loading"
            / "Export_ru_securities-list_20250203.csv"
        ),
        help="Local MOEX securities export used for fast ISIN -> ticker mapping.",
    )
    parser.add_argument(
        "--quotes-out",
        type=Path,
        default=funds_root / "Data" / "quotes_data" / "All_instr_df.csv",
        help="Combined quotes output in the format used by the regressions.",
    )
    parser.add_argument(
        "--quotes-dict-out",
        type=Path,
        default=funds_root / "Data" / "Processed_data" / "all_instr_quotes_dict.pkl",
        help="Pickle with per-SECID raw histories.",
    )
    parser.add_argument(
        "--mapping-out",
        type=Path,
        default=funds_root / "Data" / "quotes_data" / "moex_resolution_report.csv",
        help="CSV report describing how each requested instrument was resolved.",
    )
    parser.add_argument(
        "--imoex-out",
        type=Path,
        default=funds_root / "Data" / "Controls" / "imoex_df.xlsx",
        help="Path for IMOEX control series.",
    )
    parser.add_argument(
        "--rvi-out",
        type=Path,
        default=funds_root / "Data" / "Controls" / "rvi_df.xlsx",
        help="Path for RVI control series.",
    )
    parser.add_argument(
        "--from-date",
        default=DEFAULT_FROM_DATE,
        help="History start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--to-date",
        default=pd.Timestamp.today().strftime("%Y-%m-%d"),
        help="History end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel workers for MOEX history downloads.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Download only the first N unique instrument ids from the universe.",
    )
    parser.add_argument(
        "--skip-controls",
        action="store_true",
        help="Do not download IMOEX and RVI control series.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def request_json(
    url: str,
    params: Optional[dict] = None,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> dict:
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # pragma: no cover - network branch
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(2 ** (attempt - 1), 10))
    raise RuntimeError(f"Request failed for {url} params={params}: {last_error}")


def load_universe(universe_file: Path) -> List[str]:
    df = pd.read_csv(
        universe_file,
        sep="\t",
        encoding="utf-16",
        usecols=["ISIN конечного актива"],
        low_memory=False,
    )
    values = (
        df["ISIN конечного актива"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    values = values[values.ne("")]
    unique_values = pd.Index(values.unique()).tolist()
    return unique_values


def market_from_supertype(supertype: str) -> str:
    text = str(supertype).strip().lower()
    if "облигац" in text:
        return "bonds"
    return "shares"


def load_local_reference(export_path: Path) -> Dict[str, List[dict]]:
    usecols = ["TRADE_CODE", "ISIN", "SUPERTYPE", "EMITENT_FULL_NAME"]
    ref = pd.read_csv(
        export_path,
        encoding="cp1251",
        usecols=usecols,
        low_memory=False,
    )
    ref = ref.dropna(subset=["TRADE_CODE", "ISIN"]).copy()
    ref["ISIN"] = ref["ISIN"].astype(str).str.strip()
    ref["TRADE_CODE"] = ref["TRADE_CODE"].astype(str).str.strip()
    ref = ref[ref["ISIN"].ne("") & ref["TRADE_CODE"].ne("")]
    ref["market"] = ref["SUPERTYPE"].map(market_from_supertype)

    grouped: Dict[str, List[dict]] = {}
    for isin, group in ref.groupby("ISIN", sort=False):
        rows = group.to_dict("records")
        rows.sort(
            key=lambda row: (
                0 if row["market"] == "shares" else 1,
                0 if str(row["TRADE_CODE"]).upper() == isin.upper() else 1,
            )
        )
        grouped[isin] = rows
    return grouped


def resolve_via_local(requested_id: str, local_ref: Dict[str, List[dict]]) -> Optional[dict]:
    rows = local_ref.get(requested_id)
    if not rows:
        return None
    row = rows[0]
    return {
        "requested_id": requested_id,
        "secid": row["TRADE_CODE"],
        "market": row["market"],
        "source": "local_export",
        "supertype": row.get("SUPERTYPE"),
        "resolved_name": row.get("EMITENT_FULL_NAME"),
    }


def market_from_search_row(row: pd.Series) -> str:
    group = str(row.get("group", "")).lower()
    sec_type = str(row.get("type", "")).lower()
    board = str(row.get("primary_boardid", "")).upper()

    if "bond" in group or "bond" in sec_type or board.startswith("TQO"):
        return "bonds"
    return "shares"


def resolve_via_iss_search(requested_id: str) -> Optional[dict]:
    payload = request_json(
        f"{ISS_URL}/securities.json",
        params={"q": requested_id, "iss.meta": "off", "limit": 50},
    )
    block = payload.get("securities", {})
    rows = block.get("data", [])
    columns = block.get("columns", [])
    if not rows or not columns:
        return None

    search_df = pd.DataFrame(rows, columns=columns)
    if "isin" not in search_df.columns:
        return None

    if "secid" not in search_df.columns:
        search_df["secid"] = None

    requested_upper = requested_id.upper()
    exact = search_df[
        search_df["isin"].fillna("").astype(str).str.upper().eq(requested_upper)
        | search_df["secid"].fillna("").astype(str).str.upper().eq(requested_upper)
    ].copy()
    if exact.empty:
        return None

    for col in ["is_traded", "primary_boardid", "marketprice_boardid"]:
        if col not in exact.columns:
            exact[col] = None

    exact["board_rank"] = exact["primary_boardid"].map(
        lambda x: BOARD_RANK.get(str(x).upper(), 999)
    )
    exact = exact.sort_values(
        by=["is_traded", "board_rank"],
        ascending=[False, True],
    )
    row = exact.iloc[0]
    return {
        "requested_id": requested_id,
        "secid": row["secid"],
        "market": market_from_search_row(row),
        "source": "iss_search",
        "supertype": row.get("type"),
        "resolved_name": row.get("name"),
    }


def resolve_universe(
    requested_ids: Iterable[str],
    local_ref: Dict[str, List[dict]],
    workers: int = 8,
) -> pd.DataFrame:
    requested_ids = list(requested_ids)
    total = len(requested_ids)
    results_by_id: Dict[str, dict] = {}
    unresolved_ids: List[str] = []
    start_time = time.perf_counter()

    for idx, requested_id in enumerate(requested_ids, start=1):
        resolution = resolve_via_local(requested_id, local_ref)
        if resolution is None:
            unresolved_ids.append(requested_id)
        else:
            resolution["status"] = "resolved"
            results_by_id[requested_id] = resolution

        if idx % 250 == 0 or idx == total:
            _progress_line("resolve-local", idx, total, start_time)

    print(
        f"Local reference resolved {len(results_by_id)}/{total}; "
        f"remaining for ISS search: {len(unresolved_ids)}"
    )

    if unresolved_ids:
        iss_start = time.perf_counter()

        def _job(requested_id: str) -> tuple[str, Optional[dict], Optional[str]]:
            try:
                resolution = resolve_via_iss_search(requested_id)
                return requested_id, resolution, None
            except Exception as exc:  # pragma: no cover - network branch
                return requested_id, None, str(exc)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(workers, 1)) as executor:
            future_map = {
                executor.submit(_job, requested_id): requested_id
                for requested_id in unresolved_ids
            }
            for idx, future in enumerate(concurrent.futures.as_completed(future_map), start=1):
                requested_id, resolution, error = future.result()
                if resolution is None:
                    results_by_id[requested_id] = {
                        "requested_id": requested_id,
                        "secid": None,
                        "market": None,
                        "source": "iss_unresolved" if error is None else "iss_error",
                        "supertype": None,
                        "resolved_name": None,
                        "status": "unresolved",
                        "resolve_error": error,
                    }
                else:
                    resolution["status"] = "resolved"
                    resolution["resolve_error"] = None
                    results_by_id[requested_id] = resolution

                if idx % 100 == 0 or idx == len(unresolved_ids):
                    _progress_line("resolve-iss", idx, len(unresolved_ids), iss_start)

    resolved_rows = []
    for requested_id in requested_ids:
        row = results_by_id.get(requested_id)
        if row is None:
            row = {
                "requested_id": requested_id,
                "secid": None,
                "market": None,
                "source": "unresolved",
                "supertype": None,
                "resolved_name": None,
                "status": "unresolved",
                "resolve_error": None,
            }
        elif "resolve_error" not in row:
            row["resolve_error"] = None
        resolved_rows.append(row)

    return pd.DataFrame(resolved_rows)


def fetch_history_for_instrument(secid: str, market: str, from_date: str, to_date: str) -> pd.DataFrame:
    start = 0
    all_rows: List[list] = []
    columns: Optional[List[str]] = None
    url = f"{ISS_URL}/history/engines/stock/markets/{market}/securities/{secid}.json"

    while True:
        payload = request_json(
            url,
            params={
                "from": from_date,
                "till": to_date,
                "start": start,
                "iss.meta": "off",
            },
        )
        history = payload.get("history", {})
        rows = history.get("data", [])
        if not rows:
            break
        if columns is None:
            columns = history.get("columns", [])
        all_rows.extend(rows)
        start += len(rows)

    if columns is None:
        return pd.DataFrame()
    return pd.DataFrame(all_rows, columns=columns)


def download_histories(
    resolved_df: pd.DataFrame,
    from_date: str,
    to_date: str,
    workers: int,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    resolved_only = (
        resolved_df[resolved_df["status"] == "resolved"][["secid", "market"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    targets = [tuple(x) for x in resolved_only.to_records(index=False)]
    history_dict: Dict[str, pd.DataFrame] = {}
    rows = []

    def _job(secid: str, market: str) -> Tuple[str, str, pd.DataFrame, Optional[str]]:
        try:
            df = fetch_history_for_instrument(secid=secid, market=market, from_date=from_date, to_date=to_date)
            return secid, market, df, None
        except Exception as exc:  # pragma: no cover - network branch
            return secid, market, pd.DataFrame(), str(exc)

    total = len(targets)
    if total == 0:
        return history_dict, pd.DataFrame()

    download_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(workers, 1)) as executor:
        future_map = {
            executor.submit(_job, secid, market): (secid, market)
            for secid, market in targets
        }
        for idx, future in enumerate(concurrent.futures.as_completed(future_map), start=1):
            secid, market = future_map[future]
            job_secid, job_market, df, error = future.result()
            if error is None:
                history_dict[job_secid] = df
                rows.append(
                    {
                        "secid": job_secid,
                        "market": job_market,
                        "history_rows": len(df),
                        "download_status": "ok",
                        "download_error": None,
                    }
                )
            else:
                rows.append(
                    {
                        "secid": secid,
                        "market": market,
                        "history_rows": 0,
                        "download_status": "error",
                        "download_error": error,
                    }
                )

            if idx % 25 == 0 or idx == total:
                _progress_line("download", idx, total, download_start)

    return history_dict, pd.DataFrame(rows)


def select_preferred_board_rows(quotes_df: pd.DataFrame) -> pd.DataFrame:
    if quotes_df.empty:
        return quotes_df.copy()

    df = quotes_df.copy()
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"], errors="coerce")
    if "VALUE" in df.columns:
        df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
    df["board_rank"] = df["BOARDID"].map(lambda x: BOARD_RANK.get(str(x).upper(), 999))
    df["value_nonzero"] = df["VALUE"].fillna(0).gt(0).astype(int) if "VALUE" in df.columns else 0

    df = df.sort_values(
        by=["SECID", "TRADEDATE", "value_nonzero", "VALUE", "board_rank"],
        ascending=[True, True, False, False, True],
    )
    df = df.drop_duplicates(subset=["SECID", "TRADEDATE"], keep="first")
    df = df.drop(columns=["board_rank", "value_nonzero"], errors="ignore")
    return df


def add_return_features(quotes_df: pd.DataFrame) -> pd.DataFrame:
    if quotes_df.empty:
        return quotes_df.copy()

    df = quotes_df.copy()
    numeric_cols = [
        "LEGALCLOSEPRICE",
        "CLOSE",
        "MARKETPRICE2",
        "MARKETPRICE3",
        "VALUE",
        "YIELDCLOSE",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "ISIN" not in df.columns:
        df["ISIN"] = pd.NA

    df["instrument_key"] = df["ISIN"].fillna("").astype(str).str.strip()
    df["instrument_key"] = df["instrument_key"].replace("", pd.NA).fillna(df["SECID"])

    df["price_for_ret"] = pd.NA
    for col in ["LEGALCLOSEPRICE", "CLOSE", "MARKETPRICE3", "MARKETPRICE2"]:
        if col in df.columns:
            df["price_for_ret"] = df["price_for_ret"].fillna(df[col])

    df = df.sort_values(by=["instrument_key", "TRADEDATE", "SECID"])
    df["ret"] = df.groupby("instrument_key")["price_for_ret"].pct_change(fill_method=None)
    df["volatility"] = df.groupby("instrument_key")["ret"].transform(
        lambda s: s.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).std()
    )
    df["ret_volatility"] = df["volatility"]

    if "YIELDCLOSE" in df.columns:
        df["yield_volatility"] = df.groupby("instrument_key")["YIELDCLOSE"].transform(
            lambda s: s.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).std()
        )
    else:
        df["yield_volatility"] = pd.NA

    df = df.drop(columns=["instrument_key", "price_for_ret"], errors="ignore")
    return df


def build_combined_quotes(history_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not history_dict:
        return pd.DataFrame()

    frames = []
    for secid, df in history_dict.items():
        if df.empty:
            continue
        temp = df.copy()
        temp["REQUESTED_SECID"] = secid
        frames.append(temp)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = select_preferred_board_rows(combined)
    combined = add_return_features(combined)
    return combined


def fetch_index_history(index_secid: str, from_date: str, to_date: str) -> pd.DataFrame:
    start = 0
    all_rows: List[list] = []
    columns: Optional[List[str]] = None
    url = f"{ISS_URL}/history/engines/stock/markets/index/securities/{index_secid}.json"

    while True:
        payload = request_json(
            url,
            params={
                "from": from_date,
                "till": to_date,
                "start": start,
                "iss.meta": "off",
            },
        )
        history = payload.get("history", {})
        rows = history.get("data", [])
        if not rows:
            break
        if columns is None:
            columns = history.get("columns", [])
        all_rows.extend(rows)
        start += len(rows)

    if columns is None:
        return pd.DataFrame()
    return pd.DataFrame(all_rows, columns=columns)


def main() -> None:
    args = parse_args()

    print("Loading instrument universe...")
    universe = load_universe(args.universe_file)
    if args.limit is not None:
        universe = universe[: args.limit]
    print(f"Unique requested identifiers: {len(universe)}")

    print("Loading local MOEX reference export...")
    local_ref = load_local_reference(args.securities_export)

    print("Resolving identifiers to MOEX SECIDs...")
    resolved_df = resolve_universe(universe, local_ref, workers=args.workers)

    print("Downloading MOEX histories...")
    history_dict, download_stats = download_histories(
        resolved_df=resolved_df,
        from_date=args.from_date,
        to_date=args.to_date,
        workers=args.workers,
    )

    if not download_stats.empty:
        resolved_df = resolved_df.merge(download_stats, on=["secid", "market"], how="left")
    else:
        resolved_df["history_rows"] = pd.NA
        resolved_df["download_status"] = pd.NA
        resolved_df["download_error"] = pd.NA

    print("Building combined quotes table...")
    quotes_df = build_combined_quotes(history_dict)

    ensure_parent(args.mapping_out)
    resolved_df.to_csv(args.mapping_out, index=False)
    print(f"Resolution report saved to: {args.mapping_out}")

    if quotes_df.empty:
        print("No quote history was downloaded. Combined quotes file was not written.")
    else:
        ensure_parent(args.quotes_out)
        quotes_df.to_csv(args.quotes_out, index=False, encoding="utf-16", sep="\t")
        print(f"Combined quotes saved to: {args.quotes_out}")
        print(f"Combined quotes shape: {quotes_df.shape}")

    ensure_parent(args.quotes_dict_out)
    with open(args.quotes_dict_out, "wb") as f:
        pickle.dump(history_dict, f)
    print(f"Per-SECID history dict saved to: {args.quotes_dict_out}")

    if not args.skip_controls:
        print("Downloading IMOEX and RVI controls...")
        imoex_df = fetch_index_history("IMOEX", from_date=args.from_date, to_date=args.to_date)
        rvi_df = fetch_index_history("RVI", from_date=args.from_date, to_date=args.to_date)

        ensure_parent(args.imoex_out)
        ensure_parent(args.rvi_out)
        imoex_df.to_excel(args.imoex_out, index=False)
        rvi_df.to_excel(args.rvi_out, index=False)
        print(f"IMOEX saved to: {args.imoex_out}")
        print(f"RVI saved to: {args.rvi_out}")

    summary = {
        "requested_identifiers": len(universe),
        "resolved_identifiers": int((resolved_df["status"] == "resolved").sum()),
        "downloaded_secids": len(history_dict),
        "combined_rows": int(len(quotes_df)),
    }
    print("Summary:", summary)


if __name__ == "__main__":
    main()
