"""Pre-fit absolute price-DAR trimming calibration on unique price windows."""
from pathlib import Path
import datetime as dt
import hashlib
import json

import duckdb

OUT=Path('./analysis_outputs/h23_hypotheses_20261103/h3_mechanism/data')
SOURCE=OUT/'row_future_mechanism.parquet'
EXPECTED='4be88815ecd6bb2ef8e1ea4b5b84a8a261e8c71b2d788f6953007a194f9c25b8'
QUANTILES=[.99,.975,.95,.90]


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as s:
        for z in iter(lambda:s.read(4*1024*1024),b''):h.update(z)
    return h.hexdigest()


def main():
    assert sha(SOURCE)==EXPECTED
    if (OUT/'outcome_trim_calibration.json').exists():raise RuntimeError('Calibration already exists')
    c=duckdb.connect(':memory:',config={'threads':2,'memory_limit':'1GB'})
    p=str(SOURCE).replace('\\','/')
    c.execute(f'''CREATE TEMP TABLE eligible AS SELECT *,abs(dar_q_cash_price_pp) abs_dar_pp,
      sha256(cast(horizon AS VARCHAR)||chr(9)||requested_isin||chr(9)||secid||chr(9)||boardid||chr(9)||
        cast(cast(current_end_date AS DATE) AS VARCHAR)||chr(9)||cast(cast(future_target_date AS DATE) AS VARCHAR)) price_window_id
      FROM read_parquet('{p}') WHERE outcome_eligible AND trade_reason='known_nonzero_delta' AND trade_sign<>0''')
    assert c.execute('SELECT count(*) FROM eligible WHERE price_window_id IS NULL').fetchone()[0]==0
    c.execute('''CREATE TEMP TABLE conflicts AS SELECT price_window_id,horizon,requested_isin,secid,boardid,
      current_end_date,future_target_date,count(*) n_rows,count(distinct abs_dar_pp) n_abs_dar,
      count(distinct p0_price_rub) n_p0,count(distinct future_price_rub) n_p1,
      count(distinct cash_return) n_cash_return,count(distinct p0_quote_date) n_p0_dates,
      count(distinct future_quote_date) n_p1_dates,
      min(abs_dar_pp) min_abs_dar,max(abs_dar_pp) max_abs_dar
      FROM eligible GROUP BY ALL HAVING n_abs_dar<>1 OR n_p0<>1 OR n_p1<>1 OR n_cash_return<>1 OR n_p0_dates<>1 OR n_p1_dates<>1''')
    conflicts=c.execute('SELECT * FROM conflicts LIMIT 20').fetchdf().to_dict('records')
    if conflicts:
        (OUT/'outcome_trim_conflicts.json').write_text(json.dumps(conflicts,indent=2,default=str)+'\n')
        raise RuntimeError('Conflicting values inside exact price-window key; calibration not produced')
    c.execute('''CREATE TEMP TABLE windows AS SELECT price_window_id,horizon,requested_isin,secid,boardid,
      current_end_date,future_target_date,min(abs_dar_pp) abs_dar_pp,count(*) source_rows
      FROM eligible GROUP BY ALL''')
    rows=[]
    for h in [1,2,3]:
        nrows=c.execute('SELECT count(*) FROM eligible WHERE horizon=?',[h]).fetchone()[0]
        nwindows=c.execute('SELECT count(*) FROM windows WHERE horizon=?',[h]).fetchone()[0]
        ids=[r[0] for r in c.execute('SELECT price_window_id FROM windows WHERE horizon=? ORDER BY price_window_id',[h]).fetchall()]
        cuts=c.execute('SELECT quantile_cont(abs_dar_pp,[.99,.975,.95,.90]) FROM windows WHERE horizon=?',[h]).fetchone()[0]
        thresholds=[]
        for quantile,cutoff in zip(QUANTILES,cuts):
            removed_rows=c.execute('SELECT count(*) FROM eligible WHERE horizon=? AND abs_dar_pp>?',[h,cutoff]).fetchone()[0]
            removed_windows=c.execute('SELECT count(*) FROM windows WHERE horizon=? AND abs_dar_pp>?',[h,cutoff]).fetchone()[0]
            ties_rows=c.execute('SELECT count(*) FROM eligible WHERE horizon=? AND abs_dar_pp=?',[h,cutoff]).fetchone()[0]
            thresholds.append(dict(quantile=quantile,cutoff_abs_dar_pp=cutoff,removed_rows=removed_rows,
                kept_rows=nrows-removed_rows,removed_unique_windows=removed_windows,
                kept_unique_windows=nwindows-removed_windows,rows_equal_cutoff_retained=ties_rows))
        rows.append(dict(horizon=h,original_source_rows=c.execute(f"SELECT count(*) FROM read_parquet('{p}') WHERE horizon=?",[h]).fetchone()[0],
            eligible_source_rows=nrows,unique_price_windows=nwindows,
            eligible_window_id_hash=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),quantiles=thresholds))
    result=dict(status='PASS_PREFIT_UNIQUE_PRICE_WINDOW_CALIBRATION',utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        execution_host='research-host',source=str(SOURCE),source_sha256=EXPECTED,
        key=['horizon','requested_isin','requested_secid','requested_boardid','current_end_date','future_target_date'],
        source_key_column_aliases={'requested_secid':'secid','requested_boardid':'boardid'},
        price_window_id_formula='SHA256(tab-separated horizon,requested_isin,secid,boardid,YYYY-MM-DD origin,YYYY-MM-DD future target)',
        unique_window_hash_formula='SHA256(LF-joined sorted unique price_window_id; no trailing newline)',
        eligibility='outcome_eligible AND known nonzero deltaQ; before fund-focus, support, lag and quantity-Y filters',
        metric='abs(dar_q_cash_price_pp), percentage points; absolute value removes trade-direction sign before deduplication',
        within_key_checks=['one exact absDAR','one exact P0','one exact P1','one exact cash return','one exact P0 quote date','one exact P1 quote date'],
        conflicting_exact_keys=0,no_averaging_of_conflicts=True,quantile_method='DuckDB quantile_cont, linear interpolation over unique exact price windows',
        exclusion_rule='Exclude absDAR > horizon-specific cutoff; retain equality ties',
        equal_unique_window_weight=True,no_regressions=True,no_LLM=True,source_changed=False,horizons=rows)
    assert sha(SOURCE)==EXPECTED
    target=OUT/'outcome_trim_calibration.json'
    target.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
    vtbr=c.execute('''SELECT row_id,horizon,fund_id,company_id,security_id,requested_isin,secid,boardid,
      current_end_date,future_target_date,p0_source_kind,p0_price_rub,p0_quote_date,p0_raw_close,
      p0_quote_source_file,future_price_rub,future_quote_date,future_raw_close,future_quote_source_file,
      quantity_open,quantity_close,delta_quantity_units,trade_sign,future_price_return,cash_return,dar_q_cash_price_pp,price_window_id
      FROM eligible WHERE secid='VTBR' ORDER BY abs_dar_pp DESC,horizon,row_id LIMIT 20''').fetchdf().to_dict('records')
    evidence=dict(status='RAW_VTBR_PRICE_SCALE_BREAK_EVIDENCE_NOT_ADJUSTED',source=str(SOURCE),source_sha256=EXPECTED,
        note='Unadjusted source CLOSE spans a very large price-scale change; this evidence alone does not certify an economically comparable return or apply a corporate-action correction.',
        records=vtbr,max_records=20,all_examples_excluded_at_P99=all(abs(r['dar_q_cash_price_pp'])>next(x for x in rows if x['horizon']==r['horizon'])['quantiles'][0]['cutoff_abs_dar_pp'] for r in vtbr))
    (OUT/'vtbr_raw_price_break_examples.json').write_text(json.dumps(evidence,ensure_ascii=False,default=str,allow_nan=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(calibration=result,calibration_sha256=sha(target),evidence_sha256=sha(OUT/'vtbr_raw_price_break_examples.json')),ensure_ascii=True))
    c.close()


if __name__=='__main__':main()
