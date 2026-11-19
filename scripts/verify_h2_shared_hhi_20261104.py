"""Post-grid native verification of four already-computed HHI-only models.

The pooled native result is reused; only three absent native representations
are estimated. The frozen search grid and original shortlist are unchanged.
"""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


def main():
    if sys.platform != 'win32': raise RuntimeError('Run on research-host only')
    root = Path('./analysis_outputs/h2_significance_search_20261104')
    spec = importlib.util.spec_from_file_location('h2search_native', root/'code/run_h2_significance_search_20261104.py')
    runner = importlib.util.module_from_spec(spec); sys.modules[spec.name] = runner; spec.loader.exec_module(runner)
    import pandas as pd
    plan = json.loads((root/'frozen_plan.json').read_text(encoding='utf-8'))
    helper = runner.load(plan['h2_helper'], 'h2_shared_hhi_helper')
    manifest = json.loads((root/'source_manifest.json').read_text(encoding='utf-8'))
    protected = ['frozen_plan.json','all_models.csv','all_models.csv.gz','model_catalog.json','summary.json','shortlist.csv','native_results.json','completion.json']
    before = {name:runner.sha(root/name) for name in protected}
    assert runner.sha(root/'data/canonical_signed_panel.parquet') == manifest['generated_data']['canonical_signed_panel.parquet']['sha256']
    assert runner.sha(root/'data/sample_masks.parquet') == manifest['generated_data']['sample_masks.parquet']['sha256']
    source = pd.read_parquet(root/'data/canonical_signed_panel.parquet')
    masks = pd.read_parquet(root/'data/sample_masks.parquet')
    assert source.row_id.equals(masks.row_id)
    models = {m['model_id']:m for m in json.loads((root/'model_catalog.json').read_text(encoding='utf-8'))}
    old_native = json.loads((root/'native_results.json').read_text(encoding='utf-8'))
    with gzip.open(root/'fit_details.json.gz','rt',encoding='utf-8') as f: fast = json.load(f)
    destination = root/'shared_hhi'; destination.mkdir(exist_ok=True)
    if (root/'shared_hhi_native.json').exists(): raise FileExistsError('Completed verification exists')
    started=time.perf_counter(); results={}; checks=[]; newly_estimated=[]
    for group in plan['groups']:
        model=f'A__{group}__CURRENT_ONLY__SIXMASK_01'; definition=models[model]
        assert definition['controls']==['closing_HHI'] and definition['outcome_column']=='quantity_change_pct'
        if model in old_native:
            result=old_native[model]; origin='EXACT_EXISTING_NATIVE_REUSED'
        else:
            frame=source.loc[masks[definition['sample_id']]].reset_index(drop=True)
            assert helper.row_hash(frame)==definition['row_hash'] and len(frame)==definition['nobs']
            result=runner.native(frame,definition,helper,destination); origin='POST_GRID_NATIVE_VERIFICATION'
            newly_estimated.append(model)
        checks.append(runner.verify(fast[model],result))
        results[model]=dict(**result,verification_origin=origin)
    assert len(newly_estimated)==3
    unchanged={name:runner.sha(root/name)==digest for name,digest in before.items()}
    assert all(unchanged.values())
    report=dict(status='NATIVE_VERIFIED_PENDING_INDEPENDENT_QA',purpose='Post-grid presentation of one common current-only HHI specification in all four groups',
        models=results,new_native_verification_fits=3,reused_native_models=1,verification=dict(PASS=len(checks),FAIL=0,checks=checks),
        elapsed_seconds=time.perf_counter()-started,frozen_grid_and_shortlist_unchanged=unchanged,
        frozen_search_labels=660,frozen_search_native_unique_fits=58,no_new_search_specifications=True)
    runner.save(root/'shared_hhi_native.json',report)
    print(json.dumps(dict(status=report['status'],new_native_fits=3,reused=1,PASS=len(checks),FAIL=0,elapsed=report['elapsed_seconds'])))


if __name__=='__main__': main()
