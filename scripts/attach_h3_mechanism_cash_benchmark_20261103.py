"""Attach an explicitly specified calendar-index convention; no regressions."""
import argparse
from pathlib import Path
import datetime as dt
import hashlib
import json

import duckdb
import numpy as np
import pandas as pd

B=Path('./analysis_outputs')
OUT=B/'h23_hypotheses_20261103/h3_mechanism/data'
QUOTES=OUT/'row_future_prices.parquet'
PANEL=B/'h1_entries_trim_20261102/samples/AUGMENTED_P99.parquet'
LAGS=B/'h1_sentiment_lag_20261103/data/row_lag_map.parquet'
EXPECTED={'quotes':'c1adc7ff47a0863504e25fdc1deb746e94cf8015d986e3dd85752d7f92428293',
    'panel':'8bd9cfd2eef8668036c8c6c95c9863d8edbc1923d2c4d627d5b1306d03cfd43c',
    'lags':'8a2ac91bb414238c219c7a6b2165dd9cce62ff92a15f8f42b2be8de63ef626a8'}


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as s:
        for z in iter(lambda:s.read(4*1024*1024),b''):h.update(z)
    return h.hexdigest()


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--index',required=True);a.add_argument('--index-sha256',required=True)
    a.add_argument('--date-column',required=True);a.add_argument('--value-column',required=True)
    a.add_argument('--rf-date-shift-days',type=int,choices=[0,1],required=True)
    o=a.parse_args()
    if (OUT/'mechanism_data_ready.json').exists():raise RuntimeError('Completed mechanism data is immutable')
    paths={'quotes':QUOTES,'panel':PANEL,'lags':LAGS,'rf_index':Path(o.index)}
    hashes={k:sha(v) for k,v in paths.items()}
    for k,v in EXPECTED.items():assert hashes[k]==v,k
    assert hashes['rf_index']==o.index_sha256
    raw=pd.read_parquet(o.index) if Path(o.index).suffix.lower()=='.parquet' else pd.read_csv(o.index,dtype=str)
    index=pd.DataFrame(dict(index_date=pd.to_datetime(raw[o.date_column],errors='raise').dt.normalize(),
        index_value=pd.to_numeric(raw[o.value_column],errors='raise')))
    assert index.index_date.notna().all() and np.isfinite(index.index_value).all() and index.index_value.gt(0).all()
    assert not index.index_date.duplicated().any()
    assert len(pd.date_range(index.index_date.min(),index.index_date.max(),freq='D'))==len(index)
    p=pd.read_parquet(PANEL,columns=['row_id','fund_id','company_id','interval_id','start_date','end_date','net_sentiment'])
    l=pd.read_parquet(LAGS,columns=['row_id','current_interval_id','fund_id','company_id','current_start_date','current_end_date',
        'net_sentiment_lag1','lag_eligible_primary','lag_reason','lag_company_period_id','lag_interval_id'])
    check=p.merge(l,on='row_id',validate='one_to_one',suffixes=('_p','_l'))
    assert len(check)==221899
    for x,y in [('fund_id_p','fund_id_l'),('company_id_p','company_id_l'),('interval_id','current_interval_id'),
        ('start_date','current_start_date'),('end_date','current_end_date')]:assert check[x].eq(check[y]).all(),(x,y)
    signals=p[['row_id','net_sentiment']].merge(l[['row_id','net_sentiment_lag1','lag_eligible_primary','lag_reason','lag_company_period_id','lag_interval_id']],on='row_id',validate='one_to_one')
    assert signals.net_sentiment.notna().all()
    c=duckdb.connect(':memory:',config={'threads':2,'memory_limit':'1GB','temp_directory':str(OUT/'rf_tmp')})
    c.register('index_data',index);c.register('signals',signals)
    quote_path=str(QUOTES).replace('\\','/')
    index_source=str(Path(o.index)).replace('\\','/').replace("'","''")
    c.execute(f'''CREATE TEMP TABLE rf_candidates AS SELECT q.*,s.net_sentiment,s.net_sentiment_lag1,
      s.lag_eligible_primary,s.lag_reason,s.lag_company_period_id,s.lag_interval_id,
      {o.rf_date_shift_days}::INTEGER rf_date_shift_days,
      '{index_source}' rf_source_file,'{hashes['rf_index']}' rf_source_sha256,
      cast(q.current_end_date+INTERVAL '{o.rf_date_shift_days} days' AS DATE) rf_index_start_date,
      cast(q.future_target_date+INTERVAL '{o.rf_date_shift_days} days' AS DATE) rf_index_end_date,
      b.index_value rf_index_start_value,e.index_value rf_index_end_value,
      b.index_value IS NOT NULL AND e.index_value IS NOT NULL rf_eligible,
      CASE WHEN b.index_value IS NULL THEN 'missing_start_index' WHEN e.index_value IS NULL THEN 'missing_end_index' ELSE 'eligible' END rf_reason,
      CASE WHEN b.index_value IS NOT NULL AND e.index_value IS NOT NULL THEN e.index_value/b.index_value-1 END rf_return
      FROM read_parquet('{quote_path}') q
      LEFT JOIN signals s ON q.row_id=s.row_id
      LEFT JOIN index_data b ON b.index_date=cast(q.current_end_date+INTERVAL '{o.rf_date_shift_days} days' AS DATE)
      LEFT JOIN index_data e ON e.index_date=cast(q.future_target_date+INTERVAL '{o.rf_date_shift_days} days' AS DATE)''')
    c.execute('''CREATE TEMP TABLE mechanism AS SELECT *,mechanism_price_eligible AND rf_eligible mechanism_eligible,
      mechanism_price_eligible AND rf_eligible AND lag_eligible_primary mechanism_lag_eligible,
      CASE WHEN NOT mechanism_price_eligible THEN mechanism_price_reason WHEN NOT rf_eligible THEN rf_reason ELSE 'eligible' END mechanism_reason,
      CASE WHEN mechanism_price_eligible AND rf_eligible THEN trade_sign*(future_price_return-rf_return) END dar_qprice_cash_excess,
      CASE WHEN mechanism_price_eligible AND rf_eligible THEN 100*trade_sign*(future_price_return-rf_return) END dar_qprice_cash_excess_pct,
      CASE WHEN trade_reason='known_nonzero_delta' THEN trade_sign*net_sentiment END alignment_current,
      CASE WHEN trade_reason='known_nonzero_delta' AND lag_eligible_primary THEN trade_sign*net_sentiment_lag1 END alignment_lag1
      FROM rf_candidates''')
    c.execute('''CREATE TEMP TABLE final_outcomes AS SELECT *,rf_return cash_return,
      dar_qprice_cash_excess_pct dar_q_cash_price_pp,mechanism_eligible outcome_eligible,
      mechanism_reason outcome_reason,alignment_current align_current,alignment_lag1 align_lag1 FROM mechanism''')
    def one(sql):return c.execute(sql).fetchone()[0]
    def rows(sql):return c.execute(sql).fetchdf().to_dict('records')
    assert one('SELECT count(*) FROM mechanism')==665697
    assert one('SELECT count(*) FROM (SELECT row_id,horizon FROM mechanism GROUP BY ALL HAVING count(*)<>1)')==0
    assert one('SELECT count(*) FROM mechanism WHERE mechanism_eligible AND NOT isfinite(dar_qprice_cash_excess)')==0
    assert one('SELECT count(*) FROM mechanism WHERE NOT mechanism_eligible AND dar_qprice_cash_excess IS NOT NULL')==0
    assert one('SELECT count(*) FROM mechanism WHERE mechanism_lag_eligible AND alignment_lag1 IS NULL')==0
    assert one('SELECT count(*) FROM mechanism WHERE net_sentiment IS NULL')==0
    target=OUT/'row_future_mechanism.parquet'
    c.execute(f"COPY final_outcomes TO '{str(target).replace(chr(92),'/')}' (FORMAT PARQUET,COMPRESSION ZSTD)")
    audit=dict(status='MECHANISM_DATA_READY_NO_FITS',utc=dt.datetime.now(dt.timezone.utc).isoformat(),host='research-host',
      source_paths={k:str(v) for k,v in paths.items()},source_sha256=hashes,index_date_column=o.date_column,index_value_column=o.value_column,
      rf_date_shift_days=o.rf_date_shift_days,index_first_date=str(index.index_date.min().date()),index_last_date=str(index.index_date.max().date()),
      index_rows=len(index),calendar_index_complete=True,calendar_index_only_no_fill=True,
      outcome_formula='sign(delta_quantity_units)*(future normalized RUB clean-price return - RUONIA accumulated-index return)',
      outcome_units='fraction; dar_qprice_cash_excess_pct is multiplied by100',
      signal_formula='alignment_current=trade_sign*Net; alignment_lag1=trade_sign*Netlag1; lag is exact previous report interval',
      not_fund_alpha=True,not_matched_risk_alpha=True,not_observed_trade_quantities=True,no_fits=True,
      horizon_flow=rows('SELECT horizon,count(*) n_rows,sum(mechanism_price_eligible::INT) price_eligible,sum(rf_eligible::INT) rf_eligible,sum(mechanism_eligible::INT) mechanism_eligible,sum(mechanism_lag_eligible::INT) mechanism_lag_eligible FROM mechanism GROUP BY ALL ORDER BY horizon'),
      reasons=rows('SELECT horizon,mechanism_reason,count(*) n_rows FROM mechanism GROUP BY ALL ORDER BY 1,2'),
      outcome_distribution=rows('SELECT horizon,min(dar_qprice_cash_excess_pct) minimum,max(dar_qprice_cash_excess_pct) maximum,quantile_cont(abs(dar_qprice_cash_excess_pct),0.95) abs_p95,quantile_cont(abs(dar_qprice_cash_excess_pct),0.99) abs_p99 FROM mechanism WHERE mechanism_eligible GROUP BY ALL ORDER BY horizon'),
      fields=rows('DESCRIBE final_outcomes'),output=dict(path=str(target),sha256=sha(target),bytes=target.stat().st_size))
    assert all(sha(v)==hashes[k] for k,v in paths.items())
    def dump(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n',encoding='utf-8')
    dump('mechanism_data_audit.json',audit)
    manifest=dict(status='MECHANISM_DATA_READY_NO_FITS',source_sha256=hashes,builder_sha256=sha(__file__),files={n:dict(sha256=sha(OUT/n),bytes=(OUT/n).stat().st_size) for n in ['row_future_mechanism.parquet','mechanism_data_audit.json']})
    dump('mechanism_hash_manifest.json',manifest)
    dump('mechanism_data_ready.json',dict(ready=True,no_fits=True,manifest_sha256=sha(OUT/'mechanism_hash_manifest.json')))
    print(json.dumps({k:audit[k] for k in ['status','rf_date_shift_days','horizon_flow','outcome_distribution','output']},ensure_ascii=True,default=str))
    c.close()


if __name__=='__main__':main()
