"""Metadata/table-only revision of finished bridge estimates. Never fits."""
import copy,hashlib,json
from pathlib import Path
import pandas as pd

DATA=Path('./analysis_outputs/title_name_exact_price_bridge_20261029/v2_deterministic_absorption')
ROOT=DATA/'packaging_v1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
source=json.loads((DATA/'summary.json').read_text())
if source['status']!='EXACT_PRICE_BRIDGE_COMPLETED' or len(source['records'])!=20 or len(source['tests'])!=72:raise ValueError('Unexpected numeric package')
if (ROOT/'summary.json').exists():raise ValueError('Completed packaging revision already exists')
out=copy.deepcopy(source);added=[]
for index,record in enumerate(out['records']):
    convention=record['cell'].rsplit('__',1)[-1]
    if convention not in ['full_FE_rank','nested_FE_adjusted']:raise ValueError('Unrecognized locked cell convention')
    if 'convention' in record and record['convention']!=convention:raise ValueError('Contradictory convention')
    if 'convention' not in record:record['convention']=convention;added.append(index)
comparison=copy.deepcopy(out)
for index in added:del comparison['records'][index]['convention']
if comparison!=source:raise ValueError('Metadata repair changed scientific content')
out['data_root']=str(DATA)
out['packaging_revision']=dict(status='PACKAGING_ONLY_COMPLETED',output_root=str(ROOT),
    original_summary_sha256=sha(DATA/'summary.json'),original_checkpoint=json.loads((DATA/'checkpoint.json').read_text()),
    conventions_added_to_record_indices=added,numeric_payload_before_sha256=canonical(source),
    numeric_payload_after_metadata_removed_sha256=canonical(comparison),numerical_content_identical=True,
    export_script_sha256=sha(Path(__file__)),new_fits=False,change='Add convention decoded from already locked cell name; export display effects with correct units.')
ROOT.mkdir(exist_ok=True)
(ROOT/'summary.json').write_text(json.dumps(out,ensure_ascii=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
pd.DataFrame([row for record in out['records'] for row in record['coefficients']]).to_csv(ROOT/'coefficients.csv',index=False)
pd.DataFrame(out['tests']).to_csv(ROOT/'tests.csv',index=False)
lines=['# Exact-window price bridge — 29 October 2026','','Metadata/table-only packaging; no refit.', '',
 '| Model | Clusters | FE DoF | N | Positive effect | p+ | Negative effect | p− | H1 p | H2 p | Units |',
 '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
fmt=lambda value:'NA' if value is None else f'{value:.6g}'
for r in out['records']:
    rows={v['term']:v for v in r['coefficients']};a=rows.get('S_positive',{});b=rows.get('S_negative',{})
    lines.append('| '+' | '.join([r['model'],'+'.join(r['clusters']),r['convention'],str(r['N']),fmt(a.get('effect_display')),fmt(a.get('p')),
        fmt(b.get('effect_display')),fmt(b.get('p')),fmt((r.get('H1') or {}).get('intersection_union_p')),
        fmt((r.get('H2') or {}).get('p_abs')),a.get('display_units',r.get('reason','NA'))])+' |')
lines+=['','All raw coefficients, CI and fixed-cell/family/global Holm are in JSON/CSV.',
 'E1/E1B: weight basis points. E2/E3/E4: log percentage points, not weight bps. No H3 alpha.',
 'Price-only adjustment omits payouts by explicit user assumption. E3 is an implied position-growth proxy, not directly observed quantities.']
(ROOT/'tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
checkpoint=dict(stage='PACKAGING_COMPLETED_NO_REFIT',summary_sha256=sha(ROOT/'summary.json'),tables_sha256=sha(ROOT/'tables.md'),
    estimated_models=out['estimated_model_count'],estimated_covariance_cells=out['estimated_covariance_cells'],fixed_test_slots=72)
(ROOT/'checkpoint.json').write_text(json.dumps(checkpoint,indent=2)+'\n',encoding='utf-8')
print(json.dumps(checkpoint))
