"""Seal the versioned factor package only after independent raw/SQL QA passes."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import pandas as pd

ROOT=Path('./analysis_outputs/h3_alpha_20261104/factors')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(p,d):
    q=Path(str(p)+'.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n',encoding='utf-8');q.replace(p)


def main():
    ready=ROOT/'ready_manifest.json'
    if ready.exists():
        m=json.loads(ready.read_text(encoding='utf-8'))
        assert all(sha(x['path'])==x['sha256'] for x in m['files'].values())
        print('Already sealed and hashes match');return
    qa=json.loads((ROOT/'qa_independent.json').read_text(encoding='utf-8'))
    assert qa['status']=='PASS' and qa['checks_fail']==0
    m=json.loads((ROOT/'collection_manifest.json').read_text(encoding='utf-8'))
    for x in m['files'].values():assert sha(x['path'])==x['sha256']
    for x in m['reused_sources']:assert sha(x['path'])==x['sha256']
    spec=ROOT/'code/h3_alpha_spec_20261104.json'
    specdata=json.loads(spec.read_text(encoding='utf-8'))
    assert 'Index(d+1 calendar day)' in specdata['fund_alpha']['risk_free']
    primary=pd.read_parquet(ROOT/'factors_monthly.parquet')
    assert primary.shift_days.eq(1).all()
    summary={}
    for flag in ['primary_complete','resident_net_complete','corporate_sensitivity_complete']:
        x=primary.loc[primary[flag],'month']
        summary[flag]=dict(first=x.min(),last=x.max(),months=len(x))
    text='''# Official H3 risk factors — 04.11.2026

## Use this file

`factors_monthly.parquet`, join key `month` (YYYY-MM). Returns are simple monthly decimal returns, not percent and not annualised. The wide file has an explicit calendar grid; filter `primary_complete` for the primary factor model and still enforce the separate fund-history/valuation/distribution gates. Missing values are not zero.

Primary: `rf_m`, `MCFTR_return_m`, `RGBITR_return_m`. Excess factors: `equity_excess_m=MCFTR_return_m-rf_m`, `bond_excess_m=RGBITR_return_m-rf_m`. Optional resident-net sensitivity: `MCFTRR_return_m` or `equity_resident_net_excess_m`. Corporate sensitivity: `credit_factor_m=RUCBTRNS_return_m-RGBITR_return_m` (do not subtract RF a second time). `legacy_credit_factor_m` is the separate RUCBITR-minus-RGBITR series, not a splice or the default credit factor.

## Official identity and coverage

MOEX ISS security description and board metadata are preserved in `raw/moex/{SECID}/metadata.json`; the full paginated historical responses and request URLs/timestamps/hashes are retained. MCFTR is RUB gross total return (`CALCMODE=TRB`), MCFTRR is RUB resident-net total return (`TRN`), and RGBITR is RUB government-bond total income (`TR`). Neither IMOEX price nor RGBI clean-price returns were substituted. Daily `CLOSE` and the reported full-day session `3` were used; duplicate dates or other sessions are rejected rather than selected silently.

MCFTR, MCFTRR and RGBITR cover 03.12.2007–03.11.2026 in this collection. RUCBITR is an official RUB corporate-bond total-income index, but ends 31.05.2023. RUCBTRNS is a separate official RUB corporate-bond total-income index with downloadable history from 29.12.2018. They are not concatenated. Corporate duration/credit membership is not treated as identical to every bond fund; this factor is a sensitivity rather than a complete bond-risk model. These are current historical archives, not evidence of original publication vintages or historical investability.

MOEX monthly returns use the final published close in each actual calendar month divided by the final close in the immediately preceding actual calendar month minus one. Missing months are not bridged. `normalized/moex_monthly_endpoints.parquet` retains actual start/end trade dates, values, counts and calendar staleness. `month_start` denotes the prior calendar month end, and `month_end` denotes the current calendar month end, not necessarily a trading day. Observed long trading-date gaps are separately retained in `long_observed_trading_gaps.csv`; daily calendar gaps are not filled with invented observations or returns. November 2026 is incomplete and no November monthly return is supplied.

## RUONIA: two conventions kept separate

The CBR methodology, Chapter 2.1, defines the dated official index through accumulation over [T,t), with actual calendar-year day counts (365/366), compound accumulation over calculation days and simple accumulation over intervening non-calculation days. The official dated Index(t) therefore includes interest before date t; it is not a published close-of-t investment index. The project's end-of-day conversion is `I_close(d)=official Index(d+1 calendar day)`. Consequently the PRIMARY monthly RF is Index(first calendar day of next month)/Index(first calendar day of month)-1. This is a project valuation convention consistent with the official accumulation formula, not a claim that CBR labels the series as a market close or that the value was publicly known on the prior evening. Exact official index dates are saved. Historical source vintages/publication lags are not established by the archive.

The earlier DAR calculation used shift0: Index(current calendar month end)/Index(previous calendar month end)-1. Its output is retained in `factors_monthly_shift0_DIAGNOSTIC.parquet`; never mix it into the primary training or future returns. `ruonia_convention_comparison.csv` quantifies the difference on each month. No annual rate divided by 12 or fixed ACT/365 reconstruction is used, and RF is never automatically assigned as a fund's cash/deposit income.

Existing official RUONIA XML and normalized CSV from `money_market_ruonia_20261027` were reused by SHA. Only an overlapping 20.10.2026–03.11.2026 CBR SOAP extension was downloaded. All seven overlap dates agree; the existing 5,021-row DAR Parquet also agrees on every date. Original inputs and old DAR outputs remain unchanged. The combined official daily index has 6,141 calendar dates, 11.01.2010–03.11.2026. RF is unavailable before the first official index; the first full monthly RF is February 2010. No synthetic 2008–2009 RF was created.

## Technical readiness and provenance

`factor_history_availability.csv` reports only factor-side past-60/min-36/last-12-min-9/rank/condition readiness and complete future horizons. It does not assert that any fund has sufficient valid unit-return history. The new H3 specification independently controls non-EOM report origins, source valuation dates, fund identity and distribution adjustments. Source paths/hashes, raw HTTPS request logs and the independent QA are in `ready_manifest.json`. Collection failure records for two optional documents are retained historically; the bond-index list was subsequently downloaded successfully using certifi plus the Windows ROOT trust store with certificate verification enabled. No verify=False, database DDL, catalog rebuild, service restart or live-Gemma mutation was used.

Sources:
- MOEX official index history API: https://iss.moex.com/iss/history/engines/stock/markets/index/boards/RTSI/securities/MCFTR.json
- MOEX total-return equity family: https://www.moex.com/ru/index/totalreturn/MCFTR?real=1
- Corporate index: https://www.moex.com/ru/index/RUCBTRNS
- Official bond-index list: https://fs.moex.com/f/17321/bond-indices-list.pdf
- CBR current methodology linked by the official index page: https://www.cbr.ru/Content/Document/File/135651/Meth_ruonia_20220429.pdf
- CBR user guide: https://www.cbr.ru/content/document/file/128244/ruonia_avg_userguide.pdf
- CBR official SOAP source: https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx
'''
    (ROOT/'README.md').write_text(text,encoding='utf-8')
    files={str(p.relative_to(ROOT)).replace('\\','/'):dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
        for p in ROOT.rglob('*') if p.is_file() and p.name not in ['ready_manifest.json','collection_status.json'] and not p.name.endswith('.tmp')}
    readydata=dict(status='READY_FOR_ALPHA',factors_ready=True,sealed_at_utc=datetime.now(timezone.utc).isoformat(),
        output_root=str(ROOT),as_of=m['as_of'],primary_path=str(ROOT/'factors_monthly.parquet'),
        primary_sha256=sha(ROOT/'factors_monthly.parquet'),key='month YYYY-MM',return_units='simple monthly decimal returns',
        primary_columns=['rf_m','MCFTR_return_m','RGBITR_return_m'],primary_RF_shift_days=1,
        sensitivity_columns=['MCFTRR_return_m','credit_factor_m','legacy_credit_factor_m'],
        shift0_diagnostic_path=str(ROOT/'factors_monthly_shift0_DIAGNOSTIC.parquet'),
        coverage=summary,daily_series=m['series'],reused_sources=m['reused_sources'],
        latest_factor_complete_origin=qa['factor_only_latest_origins'],
        qa=dict(status='PASS',checks_pass=qa['checks_pass'],checks_fail=qa['checks_fail'],
            path=str(ROOT/'qa_independent.json'),sha256=sha(ROOT/'qa_independent.json')),
        spec=dict(path='config/h3_alpha_spec_20261104.json',snapshot=str(spec),sha256=sha(spec),spec_id=specdata['spec_id']),
        files=files,source_manifest=str(ROOT/'collection_manifest.json'),
        warnings=['RUONIA unavailable in 2008-2009; first complete RF February 2010.',
            'RUCBITR and RUCBTRNS are separate factors, never spliced.',
            'November 2026 partial: no monthly returns.',
            'Factor readiness does not establish fund-return/H3 readiness.'])
    write_json(ready,readydata)
    write_json(ROOT/'collection_status.json',dict(status='READY_FOR_ALPHA',factors_ready=True,
        updated_at_utc=datetime.now(timezone.utc).isoformat(),ready_manifest=str(ready),ready_sha256=sha(ready),
        primary_path=readydata['primary_path'],primary_sha256=readydata['primary_sha256']))
    print(json.dumps({k:readydata[k] for k in ['status','primary_path','primary_sha256','coverage','latest_factor_complete_origin','qa']},ensure_ascii=False))


if __name__=='__main__':
    main()
