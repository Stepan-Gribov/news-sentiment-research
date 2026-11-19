"""Collect official MOEX TR factors and CBR RUONIA; run only on research-host.

New versioned outputs only. Existing RF inputs are reused by hash, not rewritten.
Resumable raw HTTP responses retain URL, request time, TLS verification and SHA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path('./analysis_outputs/h3_alpha_20261104/factors')
OLD = Path('./analysis_outputs/money_market_ruonia_20261027')
DAR = Path('./analysis_outputs/h23_hypotheses_20261103/h3_mechanism/inputs/ruonia_index_daily.parquet')
SERIES = {'MCFTR': ('RTSI', 'TRB'), 'MCFTRR': ('RTSI', 'TRN'),
          'RGBITR': ('SNDX', 'TR'), 'RUCBITR': ('SNDX', 'TR'),
          'RUCBTRNS': ('RTSI', 'TR')}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(p, d):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + '.tmp')
    q.write_text(json.dumps(d, ensure_ascii=False, indent=2, default=str, allow_nan=False) + '\n', encoding='utf-8')
    q.replace(p)


def frame(block):
    return pd.DataFrame(block['data'], columns=block['columns'])


class Collector:
    def __init__(self, root, as_of):
        self.root, self.as_of = root, pd.Timestamp(as_of).normalize()
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'DissertationResearch/20261104 (official public daily index history)'
        self.rawlog = root / 'raw_requests.jsonl'
        self.warnings = []
        self.coverage = {}
        self.reused = []
        root.mkdir(parents=True, exist_ok=True)

    def status(self, status, **kw):
        write_json(self.root/'collection_status.json', dict(status=status, updated_at_utc=now(),
            as_of=str(self.as_of.date()), coverage=self.coverage, **kw))

    def fetch(self, rel, url, *, params=None, body=None, headers=None, validator=None):
        p = self.root / rel
        if p.exists():
            raw = p.read_bytes()
            if validator:
                validator(raw)
            return raw
        p.parent.mkdir(parents=True, exist_ok=True)
        last = None
        for attempt in range(4):
            try:
                r = self.session.request('POST' if body else 'GET', url, params=params, data=body,
                    headers=headers, timeout=(15, 60))  # default certificate verification stays enabled
                r.raise_for_status()
                raw = r.content
                if validator:
                    validator(raw)
                with p.open('xb') as f:
                    f.write(raw)
                entry = dict(path=str(p), url=r.url, method='POST' if body else 'GET',
                    downloaded_at_utc=now(), status_code=r.status_code, bytes=len(raw),
                    sha256=sha(p), tls_verify=True, content_type=r.headers.get('Content-Type'),
                    request_body_sha256=hashlib.sha256(body).hexdigest() if body else None)
                with self.rawlog.open('a', encoding='utf-8') as f:
                    f.write(json.dumps(entry, ensure_ascii=False)+'\n')
                return raw
            except Exception as e:
                last = e
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(f'Official download failed: {url}: {last}')

    def index(self, code, board, calcmode):
        def valid_meta(b):
            j = json.loads(b)
            d = frame(j['description']).set_index('name').value.to_dict()
            assert d['SECID'] == code and d['CALCMODE'] == calcmode and d['CURRENCYID'] == 'RUB', d
        meta = json.loads(self.fetch(f'raw/moex/{code}/metadata.json',
            f'https://iss.moex.com/iss/securities/{code}.json', params={'iss.meta': 'off'}, validator=valid_meta))
        desc = frame(meta['description']).set_index('name').value.to_dict()
        boards = frame(meta['boards'])
        assert ((boards.boardid == board) & (boards.engine == 'stock')).any()
        all_rows, totals = [], set()
        start = 0
        while True:
            url = f'https://iss.moex.com/iss/history/engines/stock/markets/index/boards/{board}/securities/{code}.json'
            params = {'from': '2007-12-01', 'till': str(self.as_of.date()), 'start': start,
                      'iss.meta': 'off', 'limit': 100, 'tradingsession': 3}
            def valid_page(b):
                j = json.loads(b)
                assert 'history' in j and 'history.cursor' in j
                assert {'SECID', 'TRADEDATE', 'CLOSE'}.issubset(j['history']['columns'])
            j = json.loads(self.fetch(f'raw/moex/{code}/history_{start:06}.json', url,
                                     params=params, validator=valid_page))
            cur = frame(j['history.cursor']).iloc[0]
            totals.add(int(cur.TOTAL))
            assert int(cur.INDEX) == start and len(totals) == 1, 'Pagination changed during collection'
            d = frame(j['history'])
            if d.empty:
                assert start >= int(cur.TOTAL), 'Unexpected empty page'
                break
            d['raw_page'] = f'history_{start:06}.json'
            all_rows.append(d)
            start += len(d)
            if start >= int(cur.TOTAL):
                break
        d = pd.concat(all_rows, ignore_index=True)
        assert len(d) == next(iter(totals))
        assert d.SECID.eq(code).all() and d.BOARDID.eq(board).all()
        assert d.TRADINGSESSION.astype(str).eq('3').all(), 'Unexpected session: no implicit session selection'
        d['date'] = pd.to_datetime(d.TRADEDATE)
        d['index_value'] = pd.to_numeric(d.CLOSE, errors='coerce')
        assert not d.date.duplicated().any()
        assert d.index_value.notna().all() and np.isfinite(d.index_value).all() and d.index_value.gt(0).all()
        assert d.date.between('2007-12-01', self.as_of).all()
        d = d.sort_values('date')
        d['series_id'], d['currency'], d['calcmode'] = code, 'RUB', calcmode
        d['previous_observed_date'] = d.date.shift(1)
        d['daily_return'] = d.index_value / d.index_value.shift(1) - 1
        d['calendar_gap_days'] = (d.date - d.previous_observed_date).dt.days
        out = d[['series_id','date','index_value','daily_return','previous_observed_date','calendar_gap_days',
                 'BOARDID','TRADINGSESSION','currency','calcmode','raw_page']]
        p = self.root/f'normalized/{code}_daily.parquet'
        p.parent.mkdir(exist_ok=True)
        out.to_parquet(p, index=False)
        self.coverage[code] = dict(rows=len(out), first=str(out.date.min().date()), last=str(out.date.max().date()),
            name=desc['NAME'], calcmode=calcmode, currency='RUB', board=board, raw_pages=len(all_rows),
            raw_total=next(iter(totals)), path=str(p), sha256=sha(p),
            max_calendar_gap_days=int(out.calendar_gap_days.max()))
        if code == 'RUCBITR':
            self.warnings.append('RUCBITR is discontinued: retain as a separate legacy sensitivity; never splice into RUCBTRNS.')
        self.status('COLLECTING', completed_series=list(self.coverage))
        print(json.dumps(self.coverage[code], ensure_ascii=False), flush=True)
        return out

    def ruonia(self):
        old_manifest = json.loads((OLD/'download_manifest.json').read_text(encoding='utf-8'))
        for name in ['RuoniaSVXML.xml', 'ruonia_index_daily.csv']:
            p = OLD/name
            assert sha(p) == old_manifest['files'][name]['sha256']
            self.reused.append(dict(path=str(p), sha256=sha(p), role='verified existing official RF source'))
        assert sha(DAR) == 'd1902b7876bdf64495fe4cb055c11a6ff111f675265b4dede963503f36a2f0ef'
        self.reused.append(dict(path=str(DAR), sha256=sha(DAR), role='existing shift0 DAR input; unchanged'))
        old = pd.read_csv(OLD/'ruonia_index_daily.csv', parse_dates=['date'])
        # A short overlap is intentionally requested to detect revisions before appending new dates.
        begin = (old.date.max()-pd.Timedelta(days=6)).date()
        body = (f'<?xml version="1.0" encoding="utf-8"?><soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">'
                f'<soap:Body><RuoniaSVXML xmlns="http://web.cbr.ru/"><fromDate>{begin}</fromDate>'
                f'<ToDate>{self.as_of.date()}</ToDate></RuoniaSVXML></soap:Body></soap:Envelope>').encode()
        request = self.root/'raw/cbr/extension_request.xml'
        request.parent.mkdir(parents=True, exist_ok=True)
        if not request.exists():
            request.write_bytes(body)
        assert request.read_bytes() == body
        def valid_xml(b):
            assert ET.fromstring(b).findall('.//ra'), 'No RUONIA observations'
        raw = self.fetch('raw/cbr/RuoniaSVXML_extension.xml', 'https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx',
            body=body, headers={'Content-Type':'text/xml; charset=utf-8', 'SOAPAction':'http://web.cbr.ru/RuoniaSVXML'},
            validator=valid_xml)
        new = pd.DataFrame([{'date':pd.Timestamp(e.findtext('DT')[:10]),
                             'ruonia_index':float(e.findtext('RUONIA_Index'))} for e in ET.fromstring(raw).findall('.//ra')])
        overlap = old.merge(new,on='date',suffixes=('_old','_new'))
        assert len(overlap)>=5 and np.allclose(overlap.ruonia_index_old,overlap.ruonia_index_new,atol=2e-14,rtol=0), 'CBR historical revision'
        d = pd.concat([old, new[new.date>old.date.max()]],ignore_index=True).sort_values('date')
        assert not d.date.duplicated().any()
        assert d.date.diff().dropna().eq(pd.Timedelta(days=1)).all()
        assert d.ruonia_index.gt(0).all() and np.isfinite(d.ruonia_index).all()
        check = pd.read_parquet(DAR)
        common = d.merge(check,on='date',suffixes=('_combined','_dar'))
        assert len(common)==len(check) and np.allclose(common.ruonia_index_combined,common.ruonia_index_dar,atol=2e-14,rtol=0)
        p = self.root/'normalized/ruonia_index_daily.parquet'
        d.to_parquet(p,index=False)
        self.coverage['RUONIA'] = dict(rows=len(d),first=str(d.date.min().date()),last=str(d.date.max().date()),
            extension_overlap_rows=len(overlap),dar_overlap_rows=len(common),path=str(p),sha256=sha(p))
        return d

    def monthly(self, daily, rf):
        periods = pd.period_range('2007-12', self.as_of.to_period('M'), freq='M')
        records = []
        for code,d in daily.items():
            groups = {m:g for m,g in d.groupby(d.date.dt.to_period('M'))}
            for m in periods:
                end, start = m.end_time.normalize(), (m-1).end_time.normalize()
                g, prev = groups.get(m), groups.get(m-1)
                a = None if prev is None else prev.iloc[-1]
                b = None if g is None else g.iloc[-1]
                complete = end <= self.as_of
                reason = ('PARTIAL_CALENDAR_MONTH' if not complete else 'NO_CURRENT_MONTH_OBSERVATION' if b is None
                          else 'NO_PREVIOUS_MONTH_OBSERVATION' if a is None else 'OK')
                records.append(dict(series_id=code,month=str(m),month_start=start,month_end=end,
                    month_complete=complete,n_observations=0 if g is None else len(g),
                    index_start_date=pd.NaT if a is None else a.date,index_end_date=pd.NaT if b is None else b.date,
                    index_start_value=np.nan if a is None else a.index_value,
                    index_end_value=np.nan if b is None else b.index_value,
                    start_staleness_days=np.nan if a is None else (start-a.date).days,
                    end_staleness_days=np.nan if b is None else (end-b.date).days,
                    return_m=b.index_value/a.index_value-1 if reason=='OK' else np.nan,status=reason))
        endpoints = pd.DataFrame(records)
        endpoints.to_parquet(self.root/'normalized/moex_monthly_endpoints.parquet',index=False)
        rfs = []
        idx = rf.set_index('date').ruonia_index
        for m in periods:
            start,end = (m-1).end_time.normalize(),m.end_time.normalize()
            for shift in [1,0]:
                a,b = start+pd.Timedelta(days=shift),end+pd.Timedelta(days=shift)
                x,y=idx.get(a,np.nan),idx.get(b,np.nan)
                reason = ('PARTIAL_CALENDAR_MONTH' if end>self.as_of else 'MISSING_EXACT_RF_ENDPOINT'
                          if pd.isna(x) or pd.isna(y) else 'OK')
                # Also require every calendar date in the official accumulation span.
                expected=(b-a).days+1
                span=int(idx.loc[(idx.index>=a)&(idx.index<=b)].notna().sum())
                if reason=='OK' and span!=expected:
                    reason='MISSING_INTERNAL_RF_DATE'
                rfs.append(dict(month=str(m),month_start=start,month_end=end,shift_days=shift,
                    convention='close_d_plus_1_PRIMARY' if shift else 'dated_index_shift0_DIAGNOSTIC',
                    rf_index_start_date=a,rf_index_end_date=b,rf_index_start_value=x,rf_index_end_value=y,
                    rf_m=y/x-1 if reason=='OK' else np.nan,status=reason,calendar_index_dates=span))
        rfs=pd.DataFrame(rfs)
        rfs.to_parquet(self.root/'normalized/ruonia_monthly_both_conventions.parquet',index=False)
        wide=endpoints.pivot(index='month',columns='series_id',values='return_m')
        for shift in [1,0]:
            out=rfs[rfs.shift_days.eq(shift)].copy().merge(wide.add_suffix('_return_m'),on='month',how='left',validate='1:1')
            out=out.rename(columns={'status':'rf_status'})
            out['equity_excess_m']=out.MCFTR_return_m-out.rf_m
            out['bond_excess_m']=out.RGBITR_return_m-out.rf_m
            out['equity_resident_net_excess_m']=out.MCFTRR_return_m-out.rf_m
            out['credit_factor_m']=out.RUCBTRNS_return_m-out.RGBITR_return_m
            out['legacy_credit_factor_m']=out.RUCBITR_return_m-out.RGBITR_return_m
            out['primary_complete']=out[['rf_m','MCFTR_return_m','RGBITR_return_m']].notna().all(axis=1)
            out['resident_net_complete']=out[['rf_m','MCFTRR_return_m','RGBITR_return_m']].notna().all(axis=1)
            out['corporate_sensitivity_complete']=out[['rf_m','MCFTR_return_m','RGBITR_return_m','credit_factor_m']].notna().all(axis=1)
            name='factors_monthly.parquet' if shift else 'factors_monthly_shift0_DIAGNOSTIC.parquet'
            out.to_parquet(self.root/name,index=False)
            out.to_csv(self.root/name.replace('.parquet','.csv'),index=False)
        return endpoints,rfs

    def run(self):
        if (self.root/'ready_manifest.json').exists():
            print('Already sealed: no second collection created',flush=True)
            return
        self.status('COLLECTING')
        daily={code:self.index(code,*args) for code,args in SERIES.items()}
        rf=self.ruonia()
        endpoints,rfs=self.monthly(daily,rf)
        docs={
          'raw/methodology/cbr_ruonia_userguide.pdf':'https://www.cbr.ru/content/document/file/128244/ruonia_avg_userguide.pdf',
          'raw/methodology/moex_equity_totalreturn.html':'https://www.moex.com/ru/index/totalreturn/MCFTR?real=1',
          'raw/methodology/moex_bond_index_list.pdf':'https://fs.moex.com/f/17321/bond-indices-list.pdf',
          'raw/methodology/moex_corporate_indices_20260227.pdf':'https://www.moex.com/files/4r1n7dn8qhqqdxhjddw2s8av4p',
        }
        for rel,url in docs.items():
            try:
                self.fetch(rel,url,validator=(lambda b: b.startswith(b'%PDF') or (_ for _ in ()).throw(ValueError('Not PDF'))) if rel.endswith('.pdf') else None)
            except Exception as e:
                self.warnings.append(str(e))
        main=pd.read_parquet(self.root/'factors_monthly.parquet')
        checks=dict(unique_series_dates=all(not x.date.duplicated().any() for x in daily.values()),
            positive_finite_index_values=all(np.isfinite(x.index_value).all() and x.index_value.gt(0).all() for x in daily.values()),
            official_metadata_calcmode_and_currency_verified=True,
            pagination_totals_verified=True,ruonia_reused_SHA_and_overlap_verified=True,
            unique_month_series=not endpoints.duplicated(['series_id','month']).any(),
            partial_month_returns_all_missing=bool(endpoints.loc[~endpoints.month_complete,'return_m'].isna().all()),
            missing_endpoint_returns_not_zero_filled=bool(endpoints.loc[endpoints.status.ne('OK'),'return_m'].isna().all()),
            primary_RF_uses_shift1=bool(main.shift_days.eq(1).all()),
            complete_primary_2013_2026Oct=bool(main.loc[main.month.between('2013-01','2026-10'),'primary_complete'].all()),
            complete_primary_months_2013_2026Oct=int(main.loc[main.month.between('2013-01','2026-10'),'primary_complete'].sum())==166)
        write_json(self.root/'qa_collection.json',dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks))
        assert all(checks.values()),checks
        assert all(sha(x['path'])==x['sha256'] for x in self.reused)
        files={str(p.relative_to(self.root)).replace('\\','/'):dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
               for p in self.root.rglob('*') if p.is_file() and p.name not in ['collection_status.json','ready_manifest.json','collection_manifest.json']}
        manifest=dict(status='COLLECTED_AWAITING_INDEPENDENT_QA',collected_at_utc=now(),as_of=str(self.as_of.date()),
            series=self.coverage,reused_sources=self.reused,files=files,collection_checks=checks,warnings=self.warnings,
            primary_RF_convention='RF_m=official Index(first calendar day of next month)/official Index(first day of month)-1; I_close(d)=Index(d+1 calendar day)',
            shift0_diagnostic='RF_m=Index(calendar month-end)/Index(previous calendar month-end)-1; separate file; old DAR unchanged',
            MOEX_monthly_convention='Last published CLOSE in each actual calendar month, divided by last published CLOSE in previous actual calendar month, minus 1; no missing-month bridging; no partial month returns; endpoint dates/staleness saved.',
            no_fund_cash_imputation=True,no_price_index_substitution=True,no_corporate_index_splicing=True,
            factor_units='simple monthly returns in decimal, not percent, not annualized',
            downstream_contract=dict(path=str(self.root/'factors_monthly.parquet'),key='month YYYY-MM',
                primary=['rf_m','MCFTR_return_m','RGBITR_return_m'],
                sensitivity=['MCFTRR_return_m','credit_factor_m','legacy_credit_factor_m'],
                credit_factor='RUCBTRNS_return_m - RGBITR_return_m; no RF subtraction twice',
                valid_primary_months=main.loc[main.primary_complete,'month'].tolist()))
        write_json(self.root/'collection_manifest.json',manifest)
        self.status('COLLECTED_AWAITING_INDEPENDENT_QA',manifest=str(self.root/'collection_manifest.json'))
        print('COLLECTION_DONE',json.dumps(dict(coverage=self.coverage,checks=checks,warnings=self.warnings),ensure_ascii=False),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,default=ROOT);a.add_argument('--as-of',default='2026-11-03');o=a.parse_args()
    if os.name!='nt' or o.root.resolve()!=ROOT.resolve():
        raise SystemExit('Run on research-host with the designated new output root only')
    c=Collector(o.root,o.as_of)
    try:
        c.run()
    except Exception as e:
        c.status('FAILED_RESUMABLE',error=f'{type(e).__name__}: {e}')
        raise
