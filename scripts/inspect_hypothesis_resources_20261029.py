"""Bounded read-only feasibility inventory; never fits a model."""
import json
from pathlib import Path
import sys

import duckdb
import pandas as pd
import pyarrow.parquet as pq

BASE = Path('./analysis_outputs/title_name_regressions_20261026_w2_v2')
NEW = Path('./analysis_outputs/title_name_hypothesis_research_20261029')
out = {'new_root_exists': NEW.exists(), 'files': {}, 'catalog': {}}
if NEW.exists():
    out['existing_new_files'] = [p.name for p in NEW.iterdir()]
    for name in ['checkpoint.json', 'summary.json']:
        if (NEW/name).exists(): out['existing_'+name] = json.loads((NEW/name).read_text(encoding='utf-8-sig'))
for name in ['fund_security_panel.parquet','holdings_inventory.parquet','all_true_intervals.parquet',
             'report_quality.parquet','event_representatives.parquet','company_month_signal.parquet']:
    path = BASE/name
    table = pq.ParquetFile(path)
    frame = pd.read_parquet(path)
    item = dict(rows=len(frame), schema=str(table.schema_arrow), example=frame.head(2).to_dict('records'))
    if name == 'holdings_inventory.parquet':
        item['bucket_asset_counts'] = frame.groupby(['bucket_kind','asset_type'],dropna=False).size().reset_index(name='n').to_dict('records')
        for col in ['cash_currency','currency','residual_label','category','category_name','asset_name']:
            if col in frame: item[col+'_sample'] = frame[col].dropna().astype(str).unique()[:30].tolist()
    if name == 'all_true_intervals.parquet':
        possible = frame.opening_full_portfolio & frame.closing_full_portfolio & ~frame.opening_passive_explicit & ~frame.closing_passive_explicit
        item['reconciled_adjacent'] = int(possible.sum())
        item['durations_days'] = (frame.loc[possible,'end_date']-frame.loc[possible,'start_date']).dt.days.describe().to_dict()
    out['files'][name] = item
connection = duckdb.connect('./local/research-data-service/research.duckdb',read_only=True,config={'threads':2,'memory_limit':'2GB'})
try:
    out['catalog']['tables'] = connection.execute("select table_schema,table_name from information_schema.tables where table_schema in ('funds','market','refs') order by 1,2").fetchdf().to_dict('records')
    out['catalog']['sources'] = connection.execute("select * from meta.sources where source_id like 'market.%' or source_id like 'funds.%'").fetchdf().to_dict('records')
    for table in ['funds.holdings','funds.reports','market.security_quotes','market.fund_returns']:
        item = {'schema': connection.execute('describe '+table).fetchdf().to_dict('records')}
        item['example'] = connection.execute('select * from '+table+' limit 2').fetchdf().to_dict('records')
        out['catalog'][table] = item
finally: connection.close()
mapping = Path('./databaseresources/market/moex_quotes_2008_2026/mapping.csv')
out['quote_mapping_sample'] = pd.read_csv(mapping,nrows=3).to_dict('records')
print(json.dumps(out,ensure_ascii=True,default=str,allow_nan=True))
