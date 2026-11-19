import argparse
import concurrent.futures
import sys
import time
from pathlib import Path

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent
QUOTES_DIR = BASE_DIR / "quotes_data"
Y_PATH = BASE_DIR / "Y_interpolated_updated.csv"
SHARES_INFO_PATH = BASE_DIR / "Shares_info.csv"

BONDS_INFO_URL = "https://iss.moex.com/iss/securitygroups/stock_bonds/collections/stock_bonds_all/securities.json"
ETF_INFO_URL = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQTF/securities.json"
IMOEX_URL = "https://iss.moex.com/iss/history/engines/stock/markets/index/securities/IMOEX.json"
RVI_URL = "https://iss.moex.com/iss/history/engines/stock/markets/index/securities/RVI.json"


def make_session():
    session = requests.Session()
    session.headers.update({"User-Agent": "thesis-moex-loader/1.0"})
    return session


def fetch_json(session, url, params=None, timeout=30, retries=8, backoff=1.8):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            wait = backoff ** (attempt - 1)
            print(
                f"[retry] url={url} attempt={attempt}/{retries} wait={wait:.1f}s err={exc}",
                flush=True,
            )
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {url} after {retries} retries") from last_error


def fetch_paginated_table(session, url, block_name, params=None, limit=1000):
    params = dict(params or {})
    start = 0
    rows = []
    cols = None
    while True:
        page_params = dict(params)
        page_params["start"] = start
        page_params["limit"] = limit
        payload = fetch_json(session, url, params=page_params)
        block = payload[block_name]
        data = block["data"]
        if not data:
            break
        if cols is None:
            cols = block["columns"]
        rows.extend(data)
        start += len(data)
        print(f"[page] {url.split('/iss/')[-1]} rows={start}", flush=True)
    return pd.DataFrame(rows, columns=cols or [])


def load_target_isins(date_from):
    usecols = ["date", "ISIN конечного актива"]
    y_df = pd.read_csv(Y_PATH, sep="\t", encoding="utf-16", usecols=usecols)
    y_df["date"] = pd.to_datetime(y_df["date"])
    y_df["ISIN конечного актива"] = y_df["ISIN конечного актива"].astype(str).str.strip()
    y_df = y_df[y_df["ISIN конечного актива"].ne("") & y_df["ISIN конечного актива"].ne("nan")]
    if date_from:
        y_df = y_df[y_df["date"] >= pd.Timestamp(date_from)]
    return set(y_df["ISIN конечного актива"].unique())


def load_share_isins():
    shares = pd.read_csv(SHARES_INFO_PATH, sep="\t", encoding="utf-16", usecols=["ISIN"])
    shares["ISIN"] = shares["ISIN"].astype(str).str.strip()
    shares = shares[shares["ISIN"].ne("") & shares["ISIN"].ne("nan")]
    return set(shares["ISIN"].unique())


def build_target_mapping(session, target_isins):
    bonds_df = fetch_paginated_table(session, BONDS_INFO_URL, "securities")
    etf_payload = fetch_json(session, ETF_INFO_URL, params={"iss.meta": "off"})
    etf_df = pd.DataFrame(etf_payload["securities"]["data"], columns=etf_payload["securities"]["columns"])

    bonds_keep = [c for c in ["SECID", "ISIN", "SHORTNAME", "NAME", "ISSUENAME", "LATNAME"] if c in bonds_df.columns]
    etf_keep = [c for c in ["SECID", "ISIN", "BOARDID", "SHORTNAME", "SECNAME"] if c in etf_df.columns]
    bonds_df = bonds_df[bonds_keep].drop_duplicates()
    etf_df = etf_df[etf_keep].drop_duplicates()
    bonds_df["source_market"] = "bonds"
    etf_df["source_market"] = "shares"

    mapping = pd.concat([bonds_df, etf_df], ignore_index=True, sort=False)
    mapping["ISIN"] = mapping["ISIN"].astype(str).str.strip()
    mapping = mapping[mapping["ISIN"].isin(target_isins)].drop_duplicates(subset=["ISIN", "SECID", "source_market"])
    return bonds_df, etf_df, mapping


def fetch_security_history(session, secid, market, date_from, date_till):
    url = f"https://iss.moex.com/iss/history/engines/stock/markets/{market}/securities/{secid}.json"
    return fetch_paginated_table(
        session,
        url,
        "history",
        params={"from": date_from, "till": date_till},
    )


def download_histories(mapping_df, date_from, date_till, workers, sample):
    if sample:
        mapping_df = mapping_df.head(sample).copy()
    tasks = list(mapping_df[["ISIN", "SECID", "source_market"]].itertuples(index=False, name=None))
    total = len(tasks)
    if total == 0:
        return pd.DataFrame(), []

    done = 0
    failures = []
    frames = []
    start_ts = time.perf_counter()

    def worker(task):
        isin, secid, market = task
        session = make_session()
        hist = fetch_security_history(session, secid, market, date_from, date_till)
        if not hist.empty:
            hist["REQUESTED_ISIN"] = isin
            hist["SOURCE_SECID"] = secid
            hist["SOURCE_MARKET"] = market
        return isin, secid, market, hist

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(worker, task): task for task in tasks}
        for future in concurrent.futures.as_completed(future_map):
            isin, secid, market = future_map[future]
            done += 1
            try:
                _, _, _, hist = future.result()
                if not hist.empty:
                    frames.append(hist)
                else:
                    failures.append({"ISIN": isin, "SECID": secid, "source_market": market, "reason": "empty_history"})
            except Exception as exc:
                failures.append({"ISIN": isin, "SECID": secid, "source_market": market, "reason": str(exc)})
            elapsed = time.perf_counter() - start_ts
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (total - done) / rate if rate > 0 else 0.0
            print(
                f"[download] {done}/{total} ok_frames={len(frames)} fail={len(failures)} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
                flush=True,
            )

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined, failures


def fetch_index_history(session, url, name, date_from, date_till):
    df = fetch_paginated_table(session, url, "history", params={"from": date_from, "till": date_till})
    if not df.empty:
        df["INDEX_NAME"] = name
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date-from", default="2024-10-01")
    parser.add_argument("--date-till", default="2025-12-31")
    parser.add_argument("--target-date-from", default="2024-12-01")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args()

    QUOTES_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_sample_{args.sample}" if args.sample else ""

    print("[stage] load target isins", flush=True)
    target_isins = load_target_isins(args.target_date_from)
    share_isins = load_share_isins()
    missing_isins = sorted(target_isins - share_isins)
    print(f"target_isins={len(target_isins)} share_isins={len(share_isins)} missing_nonshare_candidates={len(missing_isins)}", flush=True)

    session = make_session()

    print("[stage] fetch mappings", flush=True)
    bonds_df, etf_df, mapping_df = build_target_mapping(session, set(missing_isins))
    matched_isins = set(mapping_df["ISIN"].unique()) if not mapping_df.empty else set()
    unmatched_isins = sorted(set(missing_isins) - matched_isins)
    print(
        f"bonds_rows={len(bonds_df)} etf_rows={len(etf_df)} matched_isins={len(matched_isins)} unmatched_isins={len(unmatched_isins)}",
        flush=True,
    )

    bonds_df.to_csv(QUOTES_DIR / f"moex_bonds_universe{suffix}.csv", index=False)
    etf_df.to_csv(QUOTES_DIR / f"moex_etf_universe{suffix}.csv", index=False)
    mapping_df.to_csv(QUOTES_DIR / f"moex_nonshare_mapping{suffix}.csv", index=False)
    pd.DataFrame({"ISIN": unmatched_isins}).to_csv(QUOTES_DIR / f"moex_nonshare_unmatched{suffix}.csv", index=False)

    print("[stage] download histories", flush=True)
    history_df, failures = download_histories(
        mapping_df=mapping_df,
        date_from=args.date_from,
        date_till=args.date_till,
        workers=args.workers,
        sample=args.sample,
    )
    history_path = QUOTES_DIR / f"moex_nonshare_history{suffix}.csv"
    history_df.to_csv(history_path, index=False)
    pd.DataFrame(failures).to_csv(QUOTES_DIR / f"moex_nonshare_history_failures{suffix}.csv", index=False)

    print("[stage] download indexes", flush=True)
    imoex_df = fetch_index_history(session, IMOEX_URL, "IMOEX", args.date_from, args.date_till)
    rvi_df = fetch_index_history(session, RVI_URL, "RVI", args.date_from, args.date_till)
    imoex_df.to_csv(QUOTES_DIR / f"IMOEX_history{suffix}.csv", index=False)
    rvi_df.to_csv(QUOTES_DIR / f"RVI_history{suffix}.csv", index=False)

    print("[done]", flush=True)
    print(f"saved_history={history_path}", flush=True)
    print(f"history_rows={len(history_df)} failures={len(failures)}", flush=True)
    print(f"matched_isins={len(matched_isins)} unmatched_isins={len(unmatched_isins)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise
