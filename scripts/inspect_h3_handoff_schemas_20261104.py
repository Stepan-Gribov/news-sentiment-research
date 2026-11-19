"""Footer-only schemas of the new, sealed H3 handoffs; no observations."""
import json
from pathlib import Path
import pyarrow.parquet as pq
from h3_alpha_io_20261104 import ROOT
paths={
    'returns':ROOT/'fund_history/v1_existing_cache/fund_monthly_returns.parquet',
    'crosswalk':ROOT/'fund_history/v1_existing_cache/fund_series_crosswalk.parquet',
    'factors':ROOT/'factors/factors_monthly.parquet',
    'cash':ROOT/'fund_history/action_evidence/public_cash_events_20261103T215753.parquet',
    'profile_actions':ROOT/'fund_history/action_evidence/profile_action_status_20261103T215753.parquet',
}
print(json.dumps({k:{'rows':pq.ParquetFile(p).metadata.num_rows,
    'columns':pq.read_schema(p).names} for k,p in paths.items()},indent=2))
