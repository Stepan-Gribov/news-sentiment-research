"""Read-only data/mask audit for the fixed November 4 H2 sensitivity grid."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare(plan):
    h2 = load(plan['h2_helper'], 'h2_mask_helper')
    lag = load(plan['lag_helper'], 'h2_mask_lag_helper')
    focus = load(plan['focus_helper'], 'h2_mask_focus_helper')
    assert sha(plan['h2_helper']) == plan['h2_helper_sha256']
    for key, value in plan['expected_source_sha256'].items():
        assert sha(plan[key]) == value, key
    source = h2.ordered(pd.read_parquet(plan['input_panel']))
    assert len(source) == 221899 and source.row_id.is_unique
    frame = h2.add_signed_parts(lag.join_lag(source, pd.read_parquet(plan['lag_map'])))
    pd.testing.assert_frame_equal(source, frame[source.columns], check_exact=True)
    support = pd.read_csv(plan['focus_support'], dtype={'fund_id': str})
    computed = focus.focus_support_table(frame)
    cols = ['fund_focus', 'fund_id', 'fund_rows', 'fund_intervals', 'fund_end_dates']
    support['fund_id'] = support.fund_id.astype(str); computed['fund_id'] = computed.fund_id.astype(str)
    pd.testing.assert_frame_equal(support[cols].sort_values(cols[:2]).reset_index(drop=True),
        computed[cols].sort_values(cols[:2]).reset_index(drop=True), check_dtype=False)
    cutoff_masks, threshold_audit = h2.cutoff_masks(frame, h2.ordered(pd.read_parquet(plan['cutoff_reference_sample'])), plan['cutoffs'])
    common = frame.complete_current_lag1.astype(bool)
    masks = {}; catalog = []; support_records = []
    for group in plan['groups']:
        group_mask = pd.Series(True, index=frame.index) if group == 'pooled' else frame.fund_focus.eq(group)
        group_support = focus.support_table(frame.loc[group_mask])
        group_support['fund_id'] = group_support.fund_id.astype(str)
        support_records += [dict(group=group, **r) for r in group_support.to_dict('records')]
        for rule in plan['support_rules']:
            history = group_mask.copy()
            if rule['rows'] is not None:
                eligible = set(group_support.loc[group_support.fund_rows.ge(rule['rows']) & group_support.fund_intervals.ge(rule['intervals']), 'fund_id'])
                history &= frame.fund_id.astype(str).isin(eligible)
            for cut in plan['cutoffs']:
                name = f"B__{group}__{rule['tag']}__{cut['tag']}"
                mask = history & cutoff_masks[cut['tag']] & common
                part = frame.loc[mask]
                masks[name] = mask
                catalog.append(dict(sample_id=name, family='B', group=group, support=rule['tag'], cutoff=cut['tag'],
                    pre_lag_pre_cut_rows=int(history.sum()), excluded_by_cut=int((history & ~cutoff_masks[cut['tag']]).sum()),
                    excluded_lag_after_cut=int((history & cutoff_masks[cut['tag']] & ~common).sum()),
                    rows=len(part), funds=int(part.fund_id.nunique()), companies=int(part.company_id.nunique()),
                    ordered_row_hash=h2.row_hash(part)))
    for definition in plan['exact_samples']:
        group = definition['group']; path = definition['input_panel']
        assert sha(path) == definition['expected_source_sha256']
        exact = h2.ordered(pd.read_parquet(path))
        assert h2.row_hash(exact) == definition['expected_ordered_row_hash']
        source_mask = frame.row_id.isin(exact.row_id)
        pd.testing.assert_frame_equal(frame.loc[source_mask, exact.columns].reset_index(drop=True), exact, check_exact=True)
        name = 'A__' + group
        masks[name] = source_mask
        catalog.append(dict(sample_id=name, family='A', group=group, support='NONE' if group == 'pooled' else 'R60_I24',
            cutoff='P99', rows=len(exact), funds=int(exact.fund_id.nunique()), companies=int(exact.company_id.nunique()), ordered_row_hash=h2.row_hash(exact)))
        continuing = source_mask & frame.quantity_open.gt(0) & frame.quantity_close.gt(0)
        part = frame.loc[continuing]
        name = 'C_CONTINUING__' + group
        masks[name] = continuing
        catalog.append(dict(sample_id=name, family='C', group=group, support='exact_A_continuing', cutoff='P99',
            rows=len(part), funds=int(part.fund_id.nunique()), companies=int(part.company_id.nunique()), ordered_row_hash=h2.row_hash(part)))
    return frame, masks, catalog, support_records, threshold_audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--definition', required=True)
    args = parser.parse_args()
    if sys.platform != 'win32': raise RuntimeError('Data audit runs only on research-host')
    plan = json.loads(Path(args.definition).read_text(encoding='utf-8'))
    frame, masks, catalog, support, cuts = prepare(plan)
    root = Path(plan['output_root'])
    result = dict(status='PREFIT_NO_COEFFICIENTS', original_rows=len(frame),
        common_lag_rows=int(frame.complete_current_lag1.sum()), samples=catalog,
        unique_B_masks=len({s['ordered_row_hash'] for s in catalog if s['family'] == 'B'}),
        thresholds=cuts, declared_model_count=660, family_counts={'A':520,'B':128,'C':12})
    (root/'prefit_mask_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    pd.DataFrame(catalog).to_csv(root/'prefit_samples.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(support).to_csv(root/'prefit_fund_support.csv',index=False,encoding='utf-8-sig')
    print(json.dumps({k:v for k,v in result.items() if k not in ['samples','thresholds']}),flush=True)
