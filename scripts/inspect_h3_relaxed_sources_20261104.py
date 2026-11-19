"""Read-only source/schema and state probe; no regression or source mutation."""
import json
from pathlib import Path
import duckdb
import pyarrow.parquet as pq
BASE=Path('./analysis_outputs')
PATHS={
 'contributions':BASE/'h3_alpha_20261104/panels/nts_ee5a8bf9b450137a/quantity_news_contributions.parquet',
 'candidates':BASE/'h1_net_fund_fe_20261030/regression/all_candidate_panel.parquet',
 'quantity_decisions':BASE/'h1_entries_trim_20261102/row_decisions.parquet',
 'net':BASE/'h1_net_fund_fe_20261030/net_sentiment_company_period.parquet',
 'lag_windows':BASE/'h1_sentiment_lag_20261103/data/lag_company_windows.parquet',
 'registry':Path('./databaseresources/mapping/title-name-20261008-no-gaz/company_registry.parquet'),
 'approved_scope':BASE/'h1_macro_no_time_20261031/inputs/approved_news_company_scope.parquet',
 'events':BASE/'title_name_regressions_20261026_w2_v2/event_representatives.parquet',
}
c=duckdb.connect(':memory:');c.execute('SET threads=2')
out={}
for label,path in PATHS.items():
 if not path.exists():out[label]={'exists':False};continue
 p=pq.ParquetFile(path);out[label]={'rows':p.metadata.num_rows,'columns':p.schema_arrow.names,'bytes':path.stat().st_size}
 literal="'"+str(path).replace("'","''")+"'"
 c.execute(f'CREATE VIEW {label} AS SELECT * FROM read_parquet({literal})')
out['news_state_counts']=c.execute('SELECT period_kind,state,count(*) n,sum(n_undated_company_candidates>0) rows_with_undated FROM net GROUP BY ALL ORDER BY 1,2').fetchdf().to_dict('records')
out['registry_company_overlap']=c.execute('SELECT count(DISTINCT n.company_id) companies,count(DISTINCT n.company_id) FILTER(WHERE r.company_id IS NOT NULL) in_registry FROM net n LEFT JOIN registry r USING(company_id)').fetchdf().to_dict('records')
out['observed_outside_registry']=c.execute('SELECT count(*) n FROM net n LEFT JOIN registry r USING(company_id) WHERE n.net_sentiment IS NOT NULL AND r.company_id IS NULL').fetchone()[0]
out['quantity_mutated_gate_counts']=c.execute('SELECT inherited_target,eligible_for_score,count(*) n,count(*) FILTER(WHERE B_Q IS NOT NULL AND NOT scheduled_redemption_in_interval) pure_quantity_action_measurable FROM contributions GROUP BY ALL').fetchdf().to_dict('records')
out['event_dates']=c.execute('SELECT min(event_date) first_date,max(event_date) last_date,count(*) n,sum(event_date IS NULL) undated,count(DISTINCT company_id) companies FROM events').fetchdf().to_dict('records')
print(json.dumps(out,ensure_ascii=False,default=str,indent=2))
(BASE/'h3_relaxed_news_20261104/runtime/source_probe.json').write_text(json.dumps(out,ensure_ascii=False,default=str,indent=2),encoding='utf-8')
c.close()
