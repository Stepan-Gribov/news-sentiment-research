"""Read-only checks of the completed v2 package; no fits or input mutations."""
from pathlib import Path
import hashlib
import json
import sys

import duckdb
import numpy as np
import pandas as pd

root = Path('./analysis_outputs/title_name_regressions_20261026_w2_v2')
sys.path.insert(0, str(root / 'code'))
from title_name_regressions_crosswalk import maturity_interval_flags


def read(name):
    return json.loads((root / name).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


summary = read('summary.json')
frozen = read('run_manifest.json')
prepared = read('preparation.json')
spec = read('config/spec.json')
checks = {}
checks['completed'] = read('checkpoint.json')['stage'] == 'completed'
checks['spec_hash'] = digest(root / 'config/spec.json') == frozen['spec_sha256'] == '1983f70382045ae9819b8d1a528b952e00cb2b0cc606a4eab6b7de49a4b25397'
checks['audit_hash'] = digest(root / 'config/audit.json') == frozen['audit_sha256']
checks['R4_hash'] = digest(root / 'config/r4_review.json') == '9eff5476906d58d52215b352c7a8f905eaa258f7c6d19a9cfa2af49c03fd1395'
checks['code_unchanged'] = all(digest(root / 'code' / name) == sha for name, sha in frozen['code_hashes'].items())
source_checks = {}
for source in frozen['source_hashes']:
    path = Path(source['path'])
    source_checks[str(path)] = not path.exists() if source.get('exists') is False else digest(path) == source['sha256']
checks['source_contents_unchanged'] = all(source_checks.values())
sealed = prepared['preparation_provenance']['derived_output_hashes']
checks['sealed_preparation_unchanged'] = all(digest(root / name) == sha for name, sha in sealed.items())
checks['new_preparation_fingerprint'] = summary['preparation_fingerprint'] == prepared['preparation_provenance']['identity']['fingerprint'] and summary['preparation_fingerprint'] != 'a1d4956d4b732b89dc33026c594c12a5e4a801a658287c2a5ab209ec365e6a8d'
checks['prepared_manifest_hash'] = digest(root / 'preparation.json') == frozen['preparation_manifest_sha256']
checks['required_outputs_present'] = all((root / name).exists() for name in spec['expected_outputs'])

temporal = pd.read_csv(root / 'temporal_exclusion_audit.csv')
temporal_dates = pd.to_datetime(temporal.report_date)
checks['R4_only_31_early_rows'] = len(temporal) == 31 and bool(temporal_dates.lt('2018-01-01').all())
edges = pd.read_parquet(root / 'w2_edge_selection_audit.parquet')
checks['no_all_period_code_ban'] = bool(edges.reason.eq('ACCEPTED_CURRENT_STATIC_ONLY').sum() == 3068 and not edges.reason.eq('UNRESOLVED_TEMPORAL_OR_IDENTITY_REVIEW').any())

inventory = pd.read_parquet(root / 'holdings_inventory.parquet', columns=['report_id', 'security_id', 'bucket_kind', 'amount_thousand_rub'])
checks['unique_inventory_keys'] = not inventory.duplicated(['report_id', 'security_id']).any()
observed = inventory.loc[inventory.bucket_kind.eq('instrument_leaf')].groupby('report_id').amount_thousand_rub.sum(min_count=1)
connection = duckdb.connect(spec['resources']['catalog'], read_only=True, config={'threads': 1, 'memory_limit': '1GB'})
try:
    expected = connection.execute('SELECT report_id, sum(amount_thousand_rub) amount FROM funds.holdings GROUP BY report_id').fetchdf().set_index('report_id').amount
finally:
    connection.close()
comparison = expected.rename('source').to_frame().join(observed.rename('derived'), how='outer', validate='one_to_one')
comparable = comparison.dropna()
relative = (comparable.source - comparable.derived).abs() / np.maximum(comparable.source.abs(), 1)
checks['all_raw_leaf_values_preserved'] = bool(relative.le(1e-10).all() and comparison.source.isna().eq(comparison.derived.isna()).all())

signal = pd.read_parquet(root / 'company_month_signal.parquet')
valid = signal.loc[signal.S.notna()]
checks['unique_signal_keys'] = not signal.duplicated(['company_id', 'month']).any()
checks['signal_unit_sample_sd'] = abs(valid.S.std(ddof=1) - 1) < 1e-10
checks['signal_FE_orthogonality'] = bool(max(valid.groupby('company_id').S.mean().abs().max(), valid.groupby('month').S.mean().abs().max()) < 1e-7)
checks['signal_sample_minima'] = bool(valid.groupby('company_id').size().ge(6).all() and valid.groupby('month').company_id.nunique().ge(10).all())

panel = pd.read_parquet(root / 'fund_security_panel.parquet')
checks['unique_panel_keys'] = not panel.duplicated(['fund_id', 'security_id', 'start_date', 'end_date']).any()
checks['true_EOM_endpoints'] = bool(panel.start_date.dt.is_month_end.all() and panel.end_date.dt.is_month_end.all())
checks['consecutive_calendar_months'] = bool(((panel.end_date.dt.year - panel.start_date.dt.year) * 12 + panel.end_date.dt.month - panel.start_date.dt.month).eq(1).all())
checks['within_frozen_period'] = bool(panel.month.ge('2018-01').all() and panel.month.le('2026-05').all())
checks['DW_is_raw_weight_change'] = bool(np.allclose(panel.DW, panel.weight_close - panel.weight_open, atol=1e-12, rtol=1e-12))
checks['unverified_AT_missing'] = bool(panel.AT.isna().all())
checks['historical_and_corporate_action_flags_retained'] = bool(not panel.historical_identity_verified.any() and not panel.corporate_action_history_verified.any())
checks['explicit_Russian_residence_retained'] = bool(panel.residence_documented_by_disclosure.eq(True).all())
facts = pd.read_parquet(root / 'scheduled_corporate_events.parquet')
checks['scheduled_maturity_affected_targets_absent'] = not maturity_interval_flags(panel, facts).any()
checks['maturity_not_actual_payment'] = bool(facts.interpretation.eq('SCHEDULED_NOT_CONFIRMED_ACTUAL_REDEMPTION').all())
joined = panel[['company_id', 'month', 'S']].merge(signal[['company_id', 'month', 'S']], on=['company_id', 'month'], how='left', validate='many_to_one', suffixes=('_panel', '_natural'))
checks['natural_signal_join_no_expansion'] = bool(len(joined) == len(panel) and np.allclose(joined.S_panel, joined.S_natural, equal_nan=True))
nts = pd.read_parquet(root / 'fund_month_NTS.parquet')
checks['NTS_not_fabricated'] = bool(nts.NTS.isna().all())
checks['baseline_H1_H2_blocked'] = all(summary['hypotheses'][name]['status'] == 'NOT_ESTIMABLE_BASELINE' for name in ['H1', 'H2'])
checks['all_H3_horizons_blocked'] = all(value['status'] == 'NOT_ESTIMABLE' for value in summary['hypotheses']['H3'])
checks['independent_small_actual_estimator_check'] = summary['independent_small_calculation']['status'] == 'PASS'
flow = pd.read_csv(root / 'primary_exclusion_flow.csv')
checks['primary_exclusion_totals_reconcile'] = all(group.n.sum() == group.denominator.iloc[0] for _, group in flow.groupby('unit'))
rows = pd.DataFrame(summary['coefficients'])
checks['unique_model_coefficient_keys'] = not rows.duplicated(['model', 'term']).any()
checks['covariance_failures_not_zero_SE'] = bool(rows.loc[rows.inference_status.ne('ESTIMABLE'), ['se', 'p', 'ci95_low', 'ci95_high']].isna().all().all())
checks['all_estimated_models_exploratory'] = all(value['status'] in {'ESTIMATED_EXPLORATORY_NOT_BASELINE_H1_H2', 'NOT_ESTIMABLE'} for value in summary['models'])
main = next(value for value in summary['models'] if value['model'] == 'EXPLORATORY_DW_S_main')
checks['main_FE_clusters_retained'] = main['diagnostics']['fixed_effects'] == ['fund_security_id', 'month'] and main['diagnostics']['cluster_dimensions'] == ['fund_id', 'company_id']
result = dict(status='PASS' if all(checks.values()) else 'FAIL', checks={key: bool(value) for key, value in checks.items()},
    verified_sources=len(source_checks), sealed_outputs=len(sealed), preparation_fingerprint=summary['preparation_fingerprint'],
    raw_leaf_reports_compared=len(comparison), max_relative_leaf_value_difference=float(relative.max()),
    temporal_excluded_rows=len(temporal), temporal_excluded_rows_2018_onward=int(temporal_dates.ge('2018-01-01').sum()),
    panel_rows=len(panel), panel_funds=int(panel.fund_id.nunique()), panel_companies=int(panel.company_id.nunique()),
    panel_securities=int(panel.security_id.nunique()), signal_company_months=len(valid),
    main_n=main['diagnostics']['nobs'], actual_independent_small_check=summary['independent_small_calculation'],
    summary_sha256=digest(root / 'summary.json'), regression_csv_sha256=digest(root / 'regression_results.csv'),
    regression_markdown_sha256=digest(root / 'regression_tables.md'),
    only_read_checks=True, refits_performed=False)
print(json.dumps(result, ensure_ascii=True, allow_nan=False))
raise SystemExit(0 if all(checks.values()) else 1)
