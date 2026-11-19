"""Finish the locked finite plan on research-host using existing immutable resources."""
import hashlib
import json
from pathlib import Path
import sys
import time

BASE = Path('./analysis_outputs/title_name_regressions_20261026_w2_v2')
ROOT = Path('./analysis_outputs/title_name_hypothesis_research_20261029')
sys.path.insert(0, str(BASE / 'code'))
import duckdb
import numpy as np
import pandas as pd
from hypothesis_methods_20261029 import model, holm, save, clean
from hypothesis_extended_20261029 import target_panel, window_signal, quote_pairs, cash_labels, price_adjustment
from title_name_regressions_core import sha256_file, IntegrityError


def announce(stage, **details):
    data = dict(stage=stage, updated_at_utc=pd.Timestamp.now(tz='UTC').isoformat(), **details)
    save(ROOT / 'control_checkpoint.json', data)
    print(json.dumps(clean(data)), flush=True)


def duration_description(frame):
    days = (frame.end_date - frame.start_date).dt.days
    return dict(rows=len(frame), intervals=int(frame.interval_id.nunique()),
                quantiles=days.quantile([0, .25, .5, .75, .9, .99, 1]).to_dict(),
                bins=[dict(label=label, rows=int(mask.sum()),
                           intervals=int(frame.loc[mask, 'interval_id'].nunique()),
                           funds=int(frame.loc[mask, 'fund_id'].nunique()),
                           companies=int(frame.loc[mask, 'company_id'].nunique()))
                      for label, mask in [('<=35', days.le(35)), ('36-93', days.between(36, 93)),
                                          ('94-366', days.between(94, 366)), ('>366', days.gt(366))]])


def main():
    started = time.monotonic()
    ROOT.mkdir(exist_ok=True)
    lock = ROOT / '.control_extended_owner'
    mode = 'a' if '--resume-after-failed-attempt' in sys.argv else 'x'
    with lock.open(mode, encoding='utf-8') as f:
        f.write('Control owns this extended pass. Never rerun a finished package.\n')
    if (ROOT / 'extended_results.json').exists():
        raise IntegrityError('Extended package already exists')
    monthly = json.loads((ROOT / 'monthly_results.json').read_text(encoding='utf-8'))
    plan = ROOT / 'config/analysis_plan.json'
    assert sha256_file(plan) == monthly['manifest']['plan_sha256']
    models = list(monthly['models'])
    announce('LOADING_SEALED_INPUTS')
    inventory = pd.read_parquet(BASE / 'holdings_inventory.parquet')
    intervals = pd.read_parquet(BASE / 'all_true_intervals.parquet')
    monthly_panel = pd.read_parquet(BASE / 'fund_security_panel.parquet')
    events = pd.read_parquet(BASE / 'event_representatives.parquet')
    facts = pd.read_parquet(BASE / 'scheduled_corporate_events.parquet')
    monthly_intervals = intervals.loc[intervals.preliminary_eligible].copy()
    _, monthly_universe, monthly_audit = target_panel(inventory, monthly_intervals, facts)
    c = duckdb.connect('./local/research-data-service/research.duckdb', read_only=True,
                       config={'threads': 2, 'memory_limit': '2GB'})
    announce('EXTRACTING_EXISTING_PRICE_PAIRS', monthly_intervals=len(monthly_intervals))
    pairs, quote_audit = quote_pairs(monthly_universe, c)
    pairs.to_parquet(ROOT / 'accepted_price_pairs.parquet', index=False)
    announce('ESTIMATING_MATCHED_PRICE_MODELS', accepted_price_pairs=len(pairs))
    values = inventory[['report_id', 'security_id', 'amount_thousand_rub']]
    if values.duplicated(['report_id', 'security_id']).any():
        raise IntegrityError('Inventory values are not unique')
    matched = monthly_panel.merge(pairs, on=['security_id', 'start_date', 'end_date'],
                                  how='inner', validate='many_to_one')
    matched = matched.merge(values.rename(columns={'report_id': 'opening_report_id',
                                                  'amount_thousand_rub': 'reported_value_open'}),
                            on=['opening_report_id', 'security_id'], how='left', validate='many_to_one')
    matched = matched.merge(values.rename(columns={'report_id': 'closing_report_id',
                                                  'amount_thousand_rub': 'reported_value_close'}),
                            on=['closing_report_id', 'security_id'], how='left', validate='many_to_one')
    matched = matched.loc[matched.reported_value_open.gt(0) & matched.reported_value_close.gt(0)].copy()
    matched['price_adjusted_value_growth'] = (np.log(matched.reported_value_close / matched.reported_value_open)
                                              - np.log(matched.price_close / matched.price_open))
    matched.to_parquet(ROOT / 'price_matched_panel.parquet', index=False)
    for identifier, outcome in [('M2B_DW_RELATIVE_PRICE_MATCHED', 'DW'),
                                ('M3_PRICE_ADJUSTED_VALUE_GROWTH', 'price_adjusted_value_growth')]:
        record, result = model(matched, identifier=identifier, family='relative_allocation',
                               outcome=outcome, relative=True)
        models.append(record)
        save(ROOT / (identifier + '.json'), record)
        announce('PRICE_MATCHED_MODEL_COMPLETE', model=identifier, status=record['status'],
                 N=record.get('diagnostics', {}).get('nobs'))
    allowed_cash, cash_evidence = cash_labels(c)
    full, coverage = price_adjustment(monthly_universe, pairs, allowed_cash)
    coverage.to_csv(ROOT / 'whole_portfolio_price_coverage.csv', index=False)
    save(ROOT / 'cash_zero_assumption_evidence.json',
         dict(allowed_security_keys=sorted(allowed_cash), labels=cash_evidence))
    priced_panel = monthly_panel.merge(full[['interval_id', 'security_id', 'AT_PRICE']],
                                       on=['interval_id', 'security_id'], how='left', validate='one_to_one')
    priced_panel = priced_panel.loc[priced_panel.AT_PRICE.notna()].copy()
    for identifier, outcome in [('M5_PRICE_AT_FULL_PORTFOLIO', 'AT_PRICE'),
                                ('M5B_DW_PRICE_AT_MATCHED', 'DW')]:
        record, result = model(priced_panel, identifier=identifier, family='full_portfolio_price_AT',
                               outcome=outcome)
        models.append(record)
        save(ROOT / (identifier + '.json'), record)
    announce('BUILDING_EXACT_DISCLOSURE_WINDOWS')
    eligible = (intervals.opening_full_portfolio.eq(True) & intervals.closing_full_portfolio.eq(True)
                & ~intervals.opening_passive_explicit & ~intervals.closing_passive_explicit
                & intervals.end_date.between(pd.Timestamp('2018-01-01'), pd.Timestamp('2026-05-29')))
    actual_intervals = intervals.loc[eligible].copy()
    actual, _, actual_audit = target_panel(inventory, actual_intervals, facts)
    durations = duration_description(actual)
    # Conservative observed source-history boundary. Do not call a truncated lag verified no-news.
    source_starts = events.groupby('source', observed=True).event_date.min()
    history_start = source_starts.max()
    prior_start = actual.start_date - (actual.end_date - actual.start_date)
    left_censored = prior_start.lt(history_start)
    censored_rows = int(left_censored.sum())
    actual = actual.loc[~left_censored].copy()
    signal, signal_audit = window_signal(actual, events)
    signal.to_parquet(ROOT / 'exact_window_signal.parquet', index=False)
    signal_fields = ['company_id', 'start_date', 'end_date', 'S', 'Tone', 'lag_log_news_count',
                     'news_status', 'n_relevant']
    actual = actual.merge(signal[signal_fields], on=['company_id', 'start_date', 'end_date'],
                          how='left', validate='many_to_one')
    actual.to_parquet(ROOT / 'exact_actual_window_panel.parquet', index=False)
    record, result = model(actual, identifier='M4_DW_EXACT_ACTUAL_WINDOWS',
                           family='exact_disclosure_windows', extra_controls=('log_interval_days',))
    if result is not None:
        fitted = actual.reset_index(drop=True).loc[result.sample.index]
        durations['estimation_sample'] = duration_description(fitted)
    models.append(record)
    save(ROOT / (record['model'] + '.json'), record)
    announce('EXACT_WINDOW_MODEL_COMPLETE', status=record['status'],
             N=record.get('diagnostics', {}).get('nobs'))
    observed_quantity = dict(model='QUANTITY_OBSERVED', status='NOT_ESTIMABLE',
                             reason='No actual holding quantity/unit field in sealed holdings; audit raw XLS separately')
    h3 = dict(model='H3_ALPHA', status='NOT_ESTIMABLE', horizons=[1, 2, 3],
              reason='Registered fund returns cover only December 2024-November 2025; no 36 prior monthly returns or verified longer unit-value series/factor set')
    tests = holm([t for m in models for t in m['tests']])
    # A separately labelled global sensitivity prevents choosing a family only after seeing its p-values.
    order = sorted(range(len(tests)), key=lambda i: tests[i]['p'] if tests[i]['p'] is not None else 1)
    last = 0.0
    for rank, index in enumerate(order):
        p = tests[index]['p']
        last = max(last, min(1.0, (len(tests) - rank) * (1 if p is None else p)))
        tests[index]['p_holm_all_locked_tests_sensitivity'] = last
    rows = [row for m in models for row in m['coefficients']]
    checks = dict(plan_hash_unchanged=sha256_file(plan) == monthly['manifest']['plan_sha256'],
                  reference_exact=monthly['reference_beta_max_abs_difference'] <= 1e-10,
                  unique_monthly_keys=not monthly_panel.duplicated(['interval_id', 'security_id']).any(),
                  unique_price_keys=not pairs.duplicated(['security_id', 'start_date', 'end_date']).any(),
                  no_future_prices=bool(pairs.quote_open_date.le(pairs.start_date).all()
                                        and pairs.quote_close_date.le(pairs.end_date).all()),
                  unique_window_signal=not signal.duplicated(['company_id', 'start_date', 'end_date']).any(),
                  unique_actual_panel=not actual.duplicated(['interval_id', 'security_id']).any(),
                  actual_panel_size_preserved=len(actual) == actual_audit['target_after_maturity'] - censored_rows,
                  finite_M3_values=bool(np.isfinite(matched.price_adjusted_value_growth).all()),
                  full_portfolio_no_unpriced_remainder=bool(coverage.loc[coverage.eligible_full_price_AT, 'uncovered_weight'].le(1e-12).all()))
    if not all(checks.values()):
        raise IntegrityError('New panel validation failed: ' + str(checks))
    output = dict(status='FINITE_EXPLORATORY_PACKAGE_COMPLETED', client_date='2026-10-29',
                  completed_at_utc=pd.Timestamp.now(tz='UTC').isoformat(), elapsed_seconds=time.monotonic() - started,
                  output_root=str(ROOT), prior_results_known=True, new_external_data=False,
                  plan_sha256=sha256_file(plan), models=models, tests=tests, coefficients=rows,
                  power_MDE=monthly['power_MDE'], conditional_models=[observed_quantity, h3], checks=checks,
                  monthly_audit=monthly_audit, quote_audit=quote_audit,
                  matched_sample=dict(rows=len(matched), funds=int(matched.fund_id.nunique()),
                                      companies=int(matched.company_id.nunique()),
                                      securities=int(matched.security_id.nunique())),
                  full_portfolio_price_coverage=dict(intervals=len(coverage),
                      eligible_intervals=int(coverage.eligible_full_price_AT.sum()), target_rows=len(priced_panel),
                      median_opening_coverage=float(coverage.coverage_fraction.median()),
                      mean_opening_coverage=float(coverage.coverage_fraction.mean())),
                  actual_window_audit=actual_audit, duration_description=durations,
                  source_history_audit=dict(observed_start_by_source=source_starts.to_dict(),
                      conservative_history_start=history_start, censored_target_rows=censored_rows,
                      boundary_limit='Earliest observed mapped story per selected source, not proof of complete original archives'),
                  exact_signal=signal_audit,
                  limitations=['All new models are post-hoc exploratory; no guarantee or proof from a p threshold',
                      'Current static identities and unverified active mandates remain provisional',
                      'Observed price adjustment omits payouts and unverified corporate actions by user assumption',
                      'M3 is implied position growth on continuing positions; unit is log-growth, not weight basis points',
                      'M4 pools actual intervals of different lengths; not a monthly response',
                      'Cluster uncertainty is conditional on LLM labels and generated S'],
                  code_hashes={p.name:sha256_file(p) for p in (ROOT / 'code').glob('*.py')},
                  control_code_hashes={p.name:sha256_file(p) for p in (ROOT / 'control_code').glob('*.py')})
    save(ROOT / 'extended_results.json', output)
    pd.DataFrame(rows).to_csv(ROOT / 'all_coefficients.csv', index=False)
    pd.DataFrame(tests).to_csv(ROOT / 'all_tests.csv', index=False)
    announce('COMPLETED', estimated=sum(m['status'].startswith('ESTIMATED') for m in models),
             total_fixed_models=len(models), seconds=output['elapsed_seconds'])
    print(json.dumps(dict(status=output['status'], path=str(ROOT / 'extended_results.json'))), flush=True)
    c.close()


if __name__ == '__main__':
    main()
