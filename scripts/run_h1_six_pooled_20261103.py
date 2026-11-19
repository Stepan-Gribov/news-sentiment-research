"""One frozen pooled-reference H1 SIX regression; bulk remains on research-host."""
import os
for name in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ.setdefault(name, '1')
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

def sha(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value

def save(path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2), encoding='utf-8')

def run(plan_path):
    started = time.perf_counter()
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    root = Path(plan['output_root']); root.mkdir(parents=True, exist_ok=True)
    if (root/'summary.json').exists(): raise FileExistsError('Completed output exists')
    d = pd.read_parquet(plan['input_panel']).sort_values(['fund_id','end_date','security_id','interval_id'], kind='stable').reset_index(drop=True)
    rh = hashlib.sha256('\n'.join(d.row_id.astype(str)).encode()).hexdigest()
    assert (len(d),d.fund_id.nunique(),d.company_id.nunique(),rh)==(plan['expected_rows'],plan['expected_funds'],plan['expected_companies'],plan['expected_row_hash'])
    assert np.isfinite(d[[plan['outcome']]+plan['targets']+plan['controls']].astype(float)).all().all()
    manifest = dict(source=plan['input_panel'],source_sha256=sha(plan['input_panel']),source_row_hash=rh,plan_sha256=sha(plan_path),runner_sha256=sha(__file__),base_helper_sha256=sha(plan['base_helper']),lag_helper_sha256=sha(plan['lag_helper']))
    save(root/'input_manifest.json',manifest); save(root/'frozen_plan.json',plan)
    base=load(plan['base_helper'],'basepooledsix');lag=load(plan['lag_helper'],'lagpooledsix')
    used,res,terms,dropped=lag.fit_native(d,plan['controls'],base,True)
    assert set(terms)==set(plan['targets']+plan['controls']) and not dropped
    (root/'native.summary.txt').write_text(str(res.summary),encoding='utf-8')
    (root/'native.summary.html').write_text(res.summary.as_html(),encoding='utf-8')
    ci=res.conf_int();rows=[]
    for term in res.params.index:
        rows.append(dict(term=term,beta=float(res.params[term]),se=float(res.std_errors[term]),p=float(res.pvalues[term]),ci_low=float(ci.loc[term].iloc[0]),ci_high=float(ci.loc[term].iloc[1])))
    pd.DataFrame(rows).to_csv(root/'coefficients.csv',index=False,encoding='utf-8-sig')
    save(root/'full_covariance.json',dict(terms=list(res.params.index),matrix=res.cov.to_numpy()))
    tests=lag.distributed_tests(res);save(root/'targetcovariance.json',tests)
    summary=dict(status='COMPLETE_PENDING_INDEPENDENT_QA',model=plan['model'],nobs=int(res.nobs),funds=int(d.fund_id.nunique()),companies=int(d.company_id.nunique()),securities=int(d.security_id.nunique()),first_start_date=str(d.start_date.min()),last_end_date=str(d.end_date.max()),fund_focus_counts=d.fund_focus.value_counts().to_dict(),minimum_history_filter=None,singleton_funds=int(d.groupby('fund_id').size().eq(1).sum()),df_resid=int(res.df_resid),rsquared_within=float(res.rsquared_within),coefficients=rows,distributed_tests=tests,manifest=manifest,plan=plan,elapsed_seconds=time.perf_counter()-started)
    save(root/'summary.json',summary)
    print(json.dumps(clean(summary),ensure_ascii=True),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);run(p.parse_args().plan)
