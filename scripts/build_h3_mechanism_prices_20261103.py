"""Quote-only inputs for separately labelled cash-excess price DAR, on research-host.

No RF convention is imposed here, no regression and no source mutation.
"""
from pathlib import Path
import datetime as dt
import hashlib
import json
import time

import duckdb

B=Path('./analysis_outputs')
OUT=B/'h23_hypotheses_20261103/h3_mechanism/data'
PANEL=B/'h1_entries_trim_20261102/samples/AUGMENTED_P99.parquet'
POSITIONS=B/'portfolio_units_20261030/portfolio_units.parquet'
CATALOG='./local/research-data-service/research.duckdb'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as s:
        for z in iter(lambda:s.read(4*1024*1024),b''):h.update(z)
    return h.hexdigest()


def main():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'price_data_ready.json').exists():raise RuntimeError('Completed quote data already exists')
    source_hashes={'panel':sha(PANEL),'positions':sha(POSITIONS)}
    assert source_hashes['panel']=='8bd9cfd2eef8668036c8c6c95c9863d8edbc1923d2c4d627d5b1306d03cfd43c'
    assert source_hashes['positions']=='4226356fb5177b978a3a28a045d61d66c334d60d70b168f85bc6830f80f57329'
    c=duckdb.connect(CATALOG,read_only=True,config={'threads':2,'memory_limit':'2GB','temp_directory':str(OUT/'tmp')})
    def run(sql):return c.execute(sql)
    def one(sql):return run(sql).fetchone()[0]
    def records(sql):return run(sql).fetchdf().to_dict('records')
    panel=str(PANEL).replace('\\','/');positions=str(POSITIONS).replace('\\','/')
    run(f'''CREATE TEMP TABLE base AS SELECT p.row_id,p.fund_id,p.company_id,p.security_id,
      p.interval_id current_interval_id,p.start_date current_start_date,p.end_date current_end_date,
      p.opening_report_id,p.closing_report_id,p.quantity_open,p.quantity_close,p.delta_quantity_units,
      sign(p.delta_quantity_units) trade_sign,
      CASE WHEN p.delta_quantity_units IS NULL OR NOT isfinite(p.delta_quantity_units) THEN 'unknown_delta'
           WHEN p.delta_quantity_units=0 THEN 'no_trade' ELSE 'known_nonzero_delta' END trade_reason,
      coalesce(z.asset_class,a.asset_class) asset_class,
      CASE WHEN z.unit_price_rub>0 AND isfinite(z.unit_price_rub) THEN 'closing_position_quote'
           WHEN a.price_isin IS NOT NULL AND a.secid IS NOT NULL AND a.boardid IS NOT NULL THEN 'opening_identity_market_fallback'
           ELSE 'no_price_identity' END p0_source_kind,
      CASE WHEN z.unit_price_rub>0 AND isfinite(z.unit_price_rub) THEN z.price_isin ELSE a.price_isin END requested_isin,
      CASE WHEN z.unit_price_rub>0 AND isfinite(z.unit_price_rub) THEN z.secid ELSE a.secid END secid,
      CASE WHEN z.unit_price_rub>0 AND isfinite(z.unit_price_rub) THEN z.boardid ELSE a.boardid END boardid,
      z.unit_price_rub closing_position_price_rub,z.quote_date closing_position_quote_date,
      z.quote_age_days closing_position_quote_age_days,z.quote_close closing_position_raw_close,
      z.currency closing_position_currency,z.facevalue closing_position_facevalue,
      z.faceunit closing_position_faceunit,z.quote_market closing_position_market,
      z.accint closing_position_accint,z.quote_source_file closing_position_quote_source_file
      FROM read_parquet('{panel}') p
      LEFT JOIN read_parquet('{positions}') z ON p.fund_id=z.fund_id AND p.closing_report_id=z.report_id AND p.security_id=z.security_id
      LEFT JOIN read_parquet('{positions}') a ON p.fund_id=a.fund_id AND p.opening_report_id=a.report_id AND p.security_id=a.security_id''')
    assert one('SELECT count(*) FROM base')==221899
    assert one('SELECT count(distinct row_id) FROM base')==221899
    run('''CREATE TEMP TABLE identity_requests AS SELECT DISTINCT requested_isin,secid,boardid,cast(current_end_date AS DATE) origin_date
      FROM base WHERE requested_isin IS NOT NULL AND secid IS NOT NULL AND boardid IS NOT NULL''')
    run('''CREATE TEMP TABLE qraw AS SELECT upper(trim(q.REQUESTED_ISIN)) requested_isin,q.SECID secid,q.BOARDID boardid,
      try_cast(q.TRADEDATE AS DATE) quote_date,q.CLOSE raw_close,q.VALUE trading_value,upper(q.CURRENCYID) currency,
      q.FACEVALUE facevalue,upper(q.FACEUNIT) faceunit,q.SOURCE_MARKET source_market,q.ACCINT accint,q.source_file,
      CASE WHEN q.SOURCE_MARKET='bonds' AND q.FACEVALUE>0 AND isfinite(q.FACEVALUE) AND upper(q.FACEUNIT) IN ('RUB','RUR','SUR')
        THEN q.CLOSE*q.FACEVALUE/100 WHEN q.SOURCE_MARKET<>'bonds' THEN q.CLOSE END unit_price_rub
      FROM market.security_quotes q
      INNER JOIN (SELECT DISTINCT requested_isin,secid,boardid FROM identity_requests) r
      ON upper(trim(q.REQUESTED_ISIN))=r.requested_isin AND q.SECID=r.secid AND q.BOARDID=r.boardid
      WHERE try_cast(q.TRADEDATE AS DATE)>=(SELECT min(origin_date)-INTERVAL '7 days' FROM identity_requests)
        AND try_cast(q.TRADEDATE AS DATE)<=(SELECT max(origin_date)+INTERVAL '3 months' FROM identity_requests)
        AND q.CLOSE>0 AND isfinite(q.CLOSE)''')
    conflicts=one('''SELECT count(*) FROM (SELECT requested_isin,secid,boardid,quote_date FROM qraw
      WHERE unit_price_rub>0 AND isfinite(unit_price_rub) AND currency IN ('RUB','RUR','SUR') GROUP BY ALL
      HAVING count(distinct unit_price_rub)>1 OR count(distinct currency)>1 OR count(distinct source_market)>1)''')
    run('''CREATE TEMP TABLE qclean AS SELECT requested_isin,secid,boardid,quote_date,
      max(raw_close) raw_close,max(trading_value) trading_value,max(currency) currency,max(facevalue) facevalue,
      max(faceunit) faceunit,max(source_market) source_market,max(accint) accint,min(source_file) source_file,
      max(unit_price_rub) unit_price_rub FROM qraw
      WHERE unit_price_rub>0 AND isfinite(unit_price_rub) AND currency IN ('RUB','RUR','SUR')
      GROUP BY requested_isin,secid,boardid,quote_date
      HAVING count(distinct unit_price_rub)=1 AND count(distinct currency)=1 AND count(distinct source_market)=1''')
    run('''CREATE TEMP TABLE p0fallback AS SELECT r.*,q.quote_date,q.raw_close,q.unit_price_rub,
      q.currency,q.facevalue,q.faceunit,q.source_market,q.accint,q.source_file,
      date_diff('day',q.quote_date,r.origin_date) quote_age_days
      FROM identity_requests r ASOF LEFT JOIN qclean q
      ON r.requested_isin=q.requested_isin AND r.secid=q.secid AND r.boardid=q.boardid AND r.origin_date>=q.quote_date''')
    run('''CREATE TEMP TABLE origins AS SELECT b.*,
      CASE WHEN b.p0_source_kind='closing_position_quote' THEN b.closing_position_price_rub ELSE f.unit_price_rub END p0_price_rub,
      CASE WHEN b.p0_source_kind='closing_position_quote' THEN b.closing_position_quote_date ELSE f.quote_date END p0_quote_date,
      CASE WHEN b.p0_source_kind='closing_position_quote' THEN b.closing_position_quote_age_days ELSE f.quote_age_days END p0_quote_age_days,
      CASE WHEN b.p0_source_kind='closing_position_quote' THEN b.closing_position_raw_close ELSE f.raw_close END p0_raw_close,
      CASE WHEN b.p0_source_kind='closing_position_quote' THEN b.closing_position_currency ELSE f.currency END p0_currency,
      CASE WHEN b.p0_source_kind='closing_position_quote' THEN b.closing_position_facevalue ELSE f.facevalue END p0_facevalue,
      CASE WHEN b.p0_source_kind='closing_position_quote' THEN b.closing_position_faceunit ELSE f.faceunit END p0_faceunit,
      CASE WHEN b.p0_source_kind='closing_position_quote' THEN b.closing_position_market ELSE f.source_market END p0_source_market,
      CASE WHEN b.p0_source_kind='closing_position_quote' THEN b.closing_position_accint ELSE f.accint END p0_accint,
      CASE WHEN b.p0_source_kind='closing_position_quote' THEN b.closing_position_quote_source_file ELSE f.source_file END p0_quote_source_file
      FROM base b LEFT JOIN p0fallback f ON b.requested_isin=f.requested_isin AND b.secid=f.secid AND b.boardid=f.boardid AND cast(b.current_end_date AS DATE)=f.origin_date''')
    assert one('SELECT count(*) FROM origins')==221899
    run('''CREATE TEMP TABLE future_requests AS SELECT r.*,h.horizon,
      cast(r.origin_date+h.horizon*INTERVAL '1 month' AS DATE) future_target_date
      FROM identity_requests r CROSS JOIN (VALUES(1),(2),(3)) h(horizon)''')
    run('''CREATE TEMP TABLE future_prices AS SELECT r.*,q.quote_date future_quote_date,
      q.raw_close future_raw_close,q.unit_price_rub future_price_rub,q.currency future_currency,
      q.facevalue future_facevalue,q.faceunit future_faceunit,q.source_market future_source_market,
      q.accint future_accint,q.source_file future_quote_source_file,
      date_diff('day',q.quote_date,r.future_target_date) future_quote_age_days
      FROM future_requests r ASOF LEFT JOIN qclean q
      ON r.requested_isin=q.requested_isin AND r.secid=q.secid AND r.boardid=q.boardid AND r.future_target_date>=q.quote_date''')
    run('''CREATE TEMP TABLE candidates AS SELECT o.*,h.horizon,
      cast(o.current_end_date+h.horizon*INTERVAL '1 month' AS DATE) future_target_date,
      f.future_quote_date,f.future_raw_close,f.future_price_rub,f.future_currency,f.future_facevalue,f.future_faceunit,
      f.future_source_market,f.future_accint,f.future_quote_source_file,f.future_quote_age_days,
      CASE WHEN o.requested_isin IS NULL OR o.secid IS NULL OR o.boardid IS NULL THEN 'missing_quote_identity'
       WHEN o.p0_price_rub IS NULL OR o.p0_price_rub<=0 OR NOT isfinite(o.p0_price_rub) THEN 'missing_p0_price'
       WHEN o.p0_quote_date>o.current_end_date OR o.p0_quote_age_days<0 OR o.p0_quote_age_days>7 THEN 'invalid_or_stale_p0_quote'
       WHEN o.p0_currency IS NULL OR o.p0_currency NOT IN ('RUB','RUR','SUR') THEN 'unsupported_p0_currency'
       WHEN f.future_price_rub IS NULL THEN 'no_future_quote'
       WHEN f.future_quote_date<=o.current_end_date THEN 'no_strict_future_quote'
       WHEN f.future_quote_age_days<0 OR f.future_quote_age_days>7 THEN 'stale_future_quote'
       WHEN f.future_quote_date>cast(o.current_end_date+h.horizon*INTERVAL '1 month' AS DATE) THEN 'future_quote_after_target'
       WHEN f.future_currency IS NULL OR f.future_currency NOT IN ('RUB','RUR','SUR') THEN 'unsupported_future_currency'
       WHEN f.future_source_market IS DISTINCT FROM o.p0_source_market THEN 'market_basis_changed'
       ELSE 'eligible' END price_reason
      FROM origins o CROSS JOIN (VALUES(1),(2),(3)) h(horizon)
      LEFT JOIN future_prices f ON o.requested_isin=f.requested_isin AND o.secid=f.secid AND o.boardid=f.boardid
      AND cast(o.current_end_date AS DATE)=f.origin_date AND h.horizon=f.horizon''')
    run('''CREATE TEMP TABLE final AS SELECT *,price_reason='eligible' price_eligible,
      price_reason='eligible' AND trade_reason='known_nonzero_delta' mechanism_price_eligible,
      CASE WHEN price_reason='eligible' THEN future_price_rub/p0_price_rub-1 END future_price_return,
      CASE WHEN trade_reason<>'known_nonzero_delta' THEN trade_reason ELSE price_reason END mechanism_price_reason
      FROM candidates''')
    assert one('SELECT count(*) FROM final')==221899*3
    assert one('SELECT count(*) FROM (SELECT row_id,horizon FROM final GROUP BY ALL HAVING count(*)<>1)')==0
    assert one("SELECT count(*) FROM final WHERE price_eligible AND NOT (p0_quote_date<=current_end_date AND future_quote_date>current_end_date AND future_quote_date<=future_target_date AND p0_quote_age_days BETWEEN0AND7 AND future_quote_age_days BETWEEN0AND7)".replace('BETWEEN0AND7','BETWEEN 0 AND 7'))==0
    assert one('SELECT count(*) FROM final WHERE price_eligible AND (NOT isfinite(future_price_return) OR future_price_return< -1)')==0
    assert one('SELECT count(*) FROM final WHERE NOT price_eligible AND future_price_return IS NOT NULL')==0
    source_files=[r[0] for r in run('SELECT DISTINCT source_file FROM qraw WHERE source_file IS NOT NULL ORDER BY1'.replace('ORDER BY1','ORDER BY 1')).fetchall()]
    source_metadata=[dict(path=s,bytes=Path(s).stat().st_size,mtime_ns=Path(s).stat().st_mtime_ns) for s in source_files]
    filename=OUT/'row_future_prices.parquet'
    run(f"COPY final TO '{str(filename).replace(chr(92),'/')}' (FORMAT PARQUET,COMPRESSION ZSTD)")
    schema=records('DESCRIBE final')
    audit=dict(status='PRICE_DATA_READY_RF_PENDING',utc=dt.datetime.now(dt.timezone.utc).isoformat(),host='research-host',
      source_hashes=source_hashes,source_panel=str(PANEL),source_positions=str(POSITIONS),catalog=CATALOG,
      rows=221899*3,source_rows=221899,unique_price_identities=one('SELECT count(*) FROM identity_requests'),
      queried_quote_rows=one('SELECT count(*) FROM qraw'),clean_quote_rows=one('SELECT count(*) FROM qclean'),
      conflicting_quote_identity_dates=conflicts,source_quote_files=len(source_files),
      source_kind_counts=records('SELECT p0_source_kind,count(*) n_rows FROM origins GROUP BY ALL'),
      fallback_p0_resolved=one("SELECT count(*) FROM origins WHERE p0_source_kind='opening_identity_market_fallback' AND p0_price_rub>0 AND p0_quote_age_days BETWEEN 0 AND 7"),
      trade_reason_counts=records('SELECT trade_reason,count(*) n_rows FROM origins GROUP BY ALL'),
      horizon_flow=records('SELECT horizon,count(*) n_rows,sum(price_eligible::INT) price_eligible,sum(mechanism_price_eligible::INT) mechanism_price_eligible,count(distinct fund_id) source_funds FROM final GROUP BY ALL ORDER BY horizon'),
      reasons=records('SELECT horizon,mechanism_price_reason,count(*) n_rows FROM final GROUP BY ALL ORDER BY 1,2'),
      valid_ranges=records('SELECT horizon,min(current_end_date) first_origin,max(current_end_date) last_origin,min(future_quote_date) first_future_quote,max(future_quote_date) last_future_quote,count(distinct fund_id) funds,count(distinct company_id) companies,count(distinct security_id) securities FROM final WHERE mechanism_price_eligible GROUP BY ALL ORDER BY horizon'),
      conventions=dict(future_target='actual closing date + DateOffset months(h), h=1,2,3',price_normalization='RUB shares CLOSE; bonds clean CLOSE*FACEVALUE/100; no ACCINT, payouts or FX substitution',
        board_policy='Exact source closing quote identity; missing closing quote may use exact opening identity with new as-of-closing lookup.',
        temporal_policy='P0 quote<=origin, age<=7; P1 origin<quote<=future target, age<=7; no future quote after target.',
        RF='Not calculated here: root must explicitly freeze index-date convention before attaching cash benchmark.',
        outcome='No alpha or DAR outcome yet; future_price_return only.',no_new_fits=True,no_LLM=True),
      fields=schema,source_inputs_unchanged=(sha(PANEL)==source_hashes['panel'] and sha(POSITIONS)==source_hashes['positions']),
      file=dict(path=str(filename),bytes=filename.stat().st_size,sha256=sha(filename)),elapsed_seconds=time.monotonic()-started)
    (OUT/'quote_sources.json').write_text(json.dumps(source_metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    def dump(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n',encoding='utf-8')
    dump('price_data_audit.json',audit)
    manifest=dict(status='PRICE_DATA_READY_RF_PENDING',files={n:dict(sha256=sha(OUT/n),bytes=(OUT/n).stat().st_size) for n in ['row_future_prices.parquet','price_data_audit.json','quote_sources.json']},builder_sha256=sha(__file__),source_hashes=source_hashes)
    dump('price_hash_manifest.json',manifest)
    dump('price_data_ready.json',dict(ready=True,RF_ready=False,manifest_sha256=sha(OUT/'price_hash_manifest.json')))
    print(json.dumps({k:audit[k] for k in ['status','rows','unique_price_identities','queried_quote_rows','fallback_p0_resolved','horizon_flow','valid_ranges','file','elapsed_seconds']},ensure_ascii=True,default=str))
    c.close()


if __name__=='__main__':main()
