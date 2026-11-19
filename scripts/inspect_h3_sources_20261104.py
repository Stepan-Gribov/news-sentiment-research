"""Footer-only H3 source schema probe. Never prints source data records."""
import json
from pathlib import Path
import pyarrow.parquet as pq

ROOT=Path('./analysis_outputs/h3_alpha_20261104')
BASE=Path('./analysis_outputs')
PATHS={
    'quantity_panel':BASE/'h1_entries_trim_20261102/samples/AUGMENTED_P99.parquet',
    'portfolio_units':BASE/'portfolio_units_20261030/portfolio_units.parquet',
    'calendar':BASE/'title_name_regressions_20261026_w2_v2/all_true_intervals.parquet',
    'focus_map':BASE/'h1_macro_no_time_20261031/fund_interval_classification.parquet',
    'scheduled_events':BASE/'title_name_regressions_20261026_w2_v2/scheduled_corporate_events.parquet',
}
result={}
for label,path in PATHS.items():
    if not path.exists():result[label]={'exists':False,'path':str(path)};continue
    f=pq.ParquetFile(path)
    result[label]=dict(path=str(path),rows=f.metadata.num_rows,
        columns={field.name:str(field.type) for field in f.schema_arrow})
(ROOT/'runtime').mkdir(exist_ok=True,parents=True)
(ROOT/'runtime/source_schema_probe.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
