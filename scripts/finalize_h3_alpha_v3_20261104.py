"""Resume metadata only after completed v3 numerical stages. Never reruns fits."""
from __future__ import annotations
import argparse
import copy
import html
import json
import os
from pathlib import Path
import shutil
import time

import pandas as pd

from h3_alpha_io_20261104 import ROOT,file_record,sha,save_json,seal_directory,checkpoint,utc
from run_h3_alpha_20261104 import compare_versions,audit_sensitivity_registry


def run(root):
    root=Path(root)
    if (root/'manifest.json').exists():
        raise RuntimeError('Already sealed; metadata finalizer will not modify completed results')
    frozen=json.loads((root/'input_manifest.json').read_text(encoding='utf-8'))
    inputs=frozen['inputs']
    for record in inputs.values():file_record(record['path'],record['sha256'])
    # All numerical products are already present. Load their actual saved
    # metadata, not a refit or re-aggregation of estimated coefficients.
    saved={}
    for p in (root/'regression').rglob('fit.json'):
        info=json.loads(p.read_text(encoding='utf-8'));saved[info['model_id']]=info
    table=pd.read_csv(root/'models.csv');results=[]
    if len(table)!=114:raise AssertionError('Numerical registry incomplete')
    for row in table.to_dict('records'):
        identifier=row['model_id']
        if identifier in saved:info=copy.deepcopy(saved[identifier])
        else:
            reference=row.get('reuse_of_primary_model_id')
            if not isinstance(reference,str) or reference not in saved:
                raise AssertionError('Missing numerical fit '+identifier)
            info=copy.deepcopy(saved[reference]);info.update(model_id=identifier,sensitivity=row['sensitivity'],
                phase=int(row['phase']),reuse_of_primary_model_id=reference)
        results.append(info)
    returns=pd.read_parquet(root/'monthly_returns_adapter.parquet')
    factors=pd.read_parquet(root/'monthly_factors_adapter.parquet')
    origins=pd.read_parquet(root/'origin_eligibility.parquet')
    broad=pd.read_parquet(inputs['nts_broad']['path'])
    diagnostic=pd.read_parquet(root/'alpha/UNIT_VALUE_KNOWN_ADJUSTMENTS/HISTORY60_MIN36/alpha.parquet')
    tr=pd.read_parquet(root/'alpha/TR_VERIFIED/HISTORY60_MIN36/alpha.parquet')
    snapshot=json.loads(Path(inputs['handoff']['path']).read_text(encoding='utf-8'))
    if snapshot['fund_release']!='v3_action_aware':raise AssertionError('This finalizer is v3-specific')
    audit=audit_sensitivity_registry(results,returns,factors,origins,broad)
    original_audit=json.loads((root/'frozen_sensitivity_audit.json').read_text(encoding='utf-8'))
    if audit!=original_audit:raise AssertionError('Saved frozen-sensitivity audit changed')
    comparison=compare_versions(root,snapshot['compare_with_run'])
    save_json(root/'comparison_v1_v3.json',comparison)
    source=Path(__file__).parent
    for key,file in [('metadata_finalizer',Path(__file__)),('metadata_helper',source/'run_h3_alpha_20261104.py')]:
        dest=root/'code'/(file.stem+'_'+sha(file)[:12]+file.suffix)
        if not dest.exists():shutil.copy2(file,dest)
        inputs['postprocess:'+key]=file_record(dest,sha(file))
    save_json(root/'metadata_completion.json',dict(stage='FINALIZED_WITHOUT_NUMERIC_RERUN',utc=utc(),
        pid=os.getpid(),reason='Comparison join must include nonoverlapping phase; numerical outputs unchanged.',
        original_input_manifest_sha256=sha(root/'input_manifest.json'),postprocessing_inputs={k:v for k,v in inputs.items() if k.startswith('postprocess:')}))
    summary=dict(status='COMPUTED_PENDING_INDEPENDENT_NUMERICAL_AND_RESEARCH_QA',output_root=str(root),
        version=frozen['version'],created_utc=utc(),spec_id=frozen['spec_id'],spec_sha256=inputs['spec']['sha256'],
        input_manifest=inputs,fund_release=snapshot['fund_release'],sensitivity_registry_status=audit['status'],
        previous_run=snapshot['compare_with_run'],source_funds=int(origins.fund_id.nunique()),selected_origins=len(origins),
        primary_NTS_origins=int(origins.nts_pair_eligible.sum()),
        primary_NTS_origins_in_mandate_universe=int((origins.nts_pair_eligible&origins.universe_eligible).sum()),
        monthly_returns=dict(observed_eligible=int(returns.diagnostic_eligible.sum()),TR_verified=int(returns.TR_eligible.sum()),
            funds_observed_eligible=int(returns.loc[returns.diagnostic_eligible,'fund_id'].nunique())),
        diagnostic_eligible_alpha_by_horizon=diagnostic.groupby('horizon').alpha.count().to_dict(),
        model_labels=len(results),estimated_model_labels=sum(r['status']=='ESTIMATED' for r in results),
        primary_TR_H3_status='NOT_ESTIMABLE' if not tr.alpha.notna().any() else 'ESTIMATED_PENDING_QA',
        original_AT_branch_status='NOT_AVAILABLE',independent_QA_required=True,publication_interpretation_ready=False,
        numerical_rerun_for_metadata_fix=False,
        limitations=['UNIT_VALUE_KNOWN_ADJUSTMENTS is factor-adjusted unit-value growth, not verified total-return alpha.',
            'Known record/payment months excluded; ex-dates and complete payout histories are unverified.',
            'Q-value changes include fund flows and valuation/clean-bond-price approximation, not pure discretionary trading.',
            'One-way fund clustering omits dependence across funds; generated-regressor/outcome uncertainty remains.',
            'No new LLM, H2 calculation, control search, trimming or diagnostic sensitivity Cartesian grid.'])
    save_json(root/'summary.json',summary);save_json(root/'all_model_details.json',results)
    page='<!doctype html><meta charset="utf-8"><title>H3 v3</title><style>body{font:18px system-ui;margin:35px}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}</style><h1>H3 v3: численный расчёт завершён, независимый QA ожидается</h1>'
    page+='<p>TR_VERIFIED: '+summary['primary_TR_H3_status']+'. Диагностическая ветка не заменяет полную доходность.</p>'
    page+=table.loc[table.sensitivity.eq('PRIMARY'),['branch','group','horizon','variant','status','nobs','funds']].to_html(index=False)
    page+='<h2>Полнота frozen sensitivities</h2>'+pd.DataFrame(audit['families']).to_html(index=False)
    page+='<h2>Сравнение v1/v3</h2>'+pd.DataFrame(comparison['coverage_comparison']).to_html(index=False)
    (root/'review.html').write_text(page,encoding='utf-8')
    for record in inputs.values():file_record(record['path'],record['sha256'])
    seal_directory(root,dict(status=summary['status'],inputs=inputs,summary='summary.json',version=frozen['version']))
    save_json(ROOT/'runtime/alpha_ready.json',dict(ready=True,output=str(root),manifest_sha256=sha(root/'manifest.json'),summary=summary))
    checkpoint('V3_COMPUTED_METADATA_FINALIZED_QA_PENDING',output=str(root),summary={k:v for k,v in summary.items() if k!='input_manifest'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);a=p.parse_args();run(a.run_root)
