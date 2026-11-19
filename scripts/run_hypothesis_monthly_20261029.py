"""First bounded package from the existing sealed v2 panel; no source rebuild."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

BASE=Path('./analysis_outputs/title_name_regressions_20261026_w2_v2')
sys.path.insert(0,str(BASE/'code'))
import pandas as pd
from title_name_regressions_core import sha256_file,freeze_json,IntegrityError
from hypothesis_methods_20261029 import model,mde,holm,save,clean

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
root=args.output;root.mkdir(parents=True,exist_ok=True)
if (root/'monthly_results.json').exists(): raise IntegrityError('Completed monthly package already exists; never rerun blindly')
plan=json.loads((root/'config/analysis_plan.json').read_text())
if plan['status']!='FROZEN_BEFORE_NEW_OUTCOME_COEFFICIENTS':raise IntegrityError('Missing frozen analysis plan')
source=BASE/'fund_security_panel.parquet'
manifest=dict(plan_sha256=sha256_file(root/'config/analysis_plan.json'),base_panel_sha256=sha256_file(source),
              base_summary_sha256=sha256_file(BASE/'summary.json'),assumptions_sha256=sha256_file(root/'config/assumptions.json'),
              code_hashes={name:sha256_file(root/'code'/name) for name in ['hypothesis_methods_20261029.py','run_hypothesis_monthly_20261029.py']},
              created_at_utc=datetime.now(timezone.utc).isoformat())
freeze_json(root/'monthly_manifest.json',manifest)
save(root/'checkpoint.json',dict(stage='ESTIMATING_MONTHLY_FIXED_MODELS',updated_at_utc=datetime.now(timezone.utc).isoformat()))
frame=pd.read_parquet(source)
records=[];reference=None
for identifier,family,options in [('M0_DW_REFERENCE','monthly_DW_reference',{}),
                                  ('M1_DW_EQUITY_INTERACTION','asset_heterogeneity',dict(interaction=True)),
                                  ('M2_DW_RELATIVE','relative_allocation',dict(relative=True))]:
    record,result=model(frame,identifier=identifier,family=family,**options);records.append(record)
    save(root/(identifier+'.json'),record)
    if identifier=='M0_DW_REFERENCE':reference=result
    print(json.dumps(dict(model=identifier,status=record['status'],N=record.get('diagnostics',{}).get('nobs'))),flush=True)
old=json.loads((BASE/'summary.json').read_text())
old_rows={row['term']:row for row in old['coefficients'] if row['model']=='EXPLORATORY_DW_S_main'}
if reference is not None:
    diff=max(abs(row['beta']-old_rows[row['term']]['beta']) for row in reference.table())
    if diff>1e-10:raise IntegrityError('Reference DW coefficients do not reproduce v2')
else:diff=None
tests=holm([test for record in records for test in record['tests']])
output=dict(status='MONTHLY_PACKAGE_COMPLETED_EXTENDED_PENDING',manifest=manifest,models=records,tests=tests,
            reference_beta_max_abs_difference=diff,power_MDE=mde(reference),
            completed_at_utc=datetime.now(timezone.utc).isoformat(),new_data_or_LLM_run=False)
save(root/'monthly_results.json',output)
pd.DataFrame(tests).to_csv(root/'monthly_tests.csv',index=False)
pd.DataFrame([row for record in records for row in record['coefficients']]).to_csv(root/'monthly_coefficients.csv',index=False)
save(root/'checkpoint.json',dict(stage='MONTHLY_COMPLETED_EXTENDED_PENDING',models=len(records),
                                completed_at_utc=output['completed_at_utc'],no_source_rebuild=True))
print(json.dumps(clean(dict(status=output['status'],reference_beta_max_abs_difference=diff))),flush=True)
