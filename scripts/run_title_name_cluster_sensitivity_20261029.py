"""Fixed covariance sensitivity, executed on research-host; original point estimates preserved."""
from dataclasses import replace
from itertools import combinations
from pathlib import Path
import json
import sys
import hashlib
import time

BASE = Path('./analysis_outputs/title_name_regressions_20261026_w2_v2')
PREVIOUS = Path('./analysis_outputs/title_name_hypothesis_research_20261029')
ROOT = Path('./analysis_outputs/title_name_cluster_sensitivity_20261029')
sys.path.insert(0, str(BASE / 'code'))
sys.path.insert(0, str(PREVIOUS / 'code'))
import numpy as np
import pandas as pd
from scipy import stats
from hypothesis_methods_20261029 import model, save, clean
from title_name_regressions_core import IntegrityError, sha256_file
from title_name_regressions_estimation import (
    within_transform, fixed_effect_rank, _cluster_meat,
    h1_directional_test, h2_absolute_reaction_test,
)


def adjust_holm(tests, key):
    order = sorted(range(len(tests)), key=lambda i: 1 if tests[i]['p'] is None else tests[i]['p'])
    last = 0.0
    for rank, i in enumerate(order):
        p = 1 if tests[i]['p'] is None else tests[i]['p']
        last = max(last, min(1.0, (len(tests) - rank) * p))
        tests[i][key] = last


def nested_rank(frame, effects, clusters):
    details = {}
    remaining = []
    for effect in effects:
        is_nested = {}
        for cluster in clusters:
            maximum = int(frame.groupby(effect, observed=True)[cluster].nunique().max())
            is_nested[cluster] = maximum == 1
        nested = any(is_nested.values())
        details[effect] = dict(categories=int(frame[effect].nunique()), nested_in=is_nested,
                               excluded_from_DoF=nested)
        if not nested:
            remaining.append(effect)
    if len(remaining) == 2:
        rank = fixed_effect_rank(frame[remaining[0]], frame[remaining[1]])[0]
    elif len(remaining) == 1:
        rank = int(frame[remaining[0]].nunique())
    else:
        rank = 1  # Retain the overall intercept in the finite-sample adjustment.
    return rank, details


def cluster_covariance(frame, x, residual, clusters, correction_fe_rank, cache):
    n, k = x.shape
    scores = x * residual[:, None]
    meat = np.zeros((k, k))
    counts = {}
    for size in range(1, len(clusters) + 1):
        for subset in combinations(clusters, size):
            subset = tuple(subset)
            if subset not in cache:
                group = frame[subset[0]] if size == 1 else pd.MultiIndex.from_frame(frame[list(subset)])
                cache[subset] = _cluster_meat(scores, group)
            part, groups = cache[subset]
            counts['|'.join(subset)] = groups
            meat += (1 if size % 2 else -1) * groups / (groups - 1) * part
    denominator = n - k - correction_fe_rank
    if denominator <= 0:
        raise IntegrityError('No correction degrees of freedom')
    q = (n - 1) / denominator
    bread = np.linalg.inv(x.T @ x)
    covariance = q * bread @ meat @ bread
    covariance = (covariance + covariance.T) / 2
    return covariance, counts, q, denominator


def main():
    start = time.monotonic()
    ROOT.mkdir(exist_ok=True)
    with (ROOT / '.writer_lock').open('x') as f:
        f.write('single covariance sensitivity writer\n')
    plan = json.loads((ROOT / 'plan.json').read_text(encoding='utf-8'))
    assert plan['status'] == 'FIXED_BEFORE_NEW_COVARIANCE_RESULTS'
    previous = json.loads((PREVIOUS / 'extended_results.json').read_text(encoding='utf-8'))
    old_models = {m['model']: m for m in previous['models']}
    paths = {'M0_DW_REFERENCE': BASE / 'fund_security_panel.parquet',
             'M4_DW_EXACT_ACTUAL_WINDOWS': PREVIOUS / 'exact_actual_window_panel.parquet'}
    schemes = [(c, 'substantive') for c in plan['substantive_covariances']]
    schemes += [(c, 'incomplete_dependence_diagnostic') for c in plan['incomplete_dependence_diagnostics']]
    all_records, all_coefficients, all_tests = [], [], []
    validations = []
    for identifier in plan['models']:
        frame = pd.read_parquet(paths[identifier])
        options = {'extra_controls': ('log_interval_days',)} if identifier.startswith('M4') else {}
        original, result = model(frame, identifier=identifier, family='covariance_sensitivity', **options)
        if result is None:
            raise IntegrityError('Previously fitted model failed reference reconstruction')
        old = old_models[identifier]
        old_beta = {row['term']: row['beta'] for row in old['coefficients']}
        max_beta_difference = max(abs(float(b) - old_beta[name]) for name, b in zip(result.names, result.beta))
        assert max_beta_difference < 1e-10
        assert result.diagnostics['nobs'] == old['diagnostics']['nobs']
        # Extend the exact retained sample by calendar dimensions without changing row selection.
        d = result.sample.copy()
        d['quarter'] = pd.PeriodIndex(d.month, freq='M').asfreq('Q').astype(str)
        projected, iterations = within_transform(d[['DW'] + result.names].to_numpy(float),
                                                 [d.fund_security_id, d.month])
        y, x = projected[:, 0], projected[:, 1:]
        residual = y - x @ result.beta
        assert np.max(np.abs(residual - result.residuals)) < 1e-8
        cache = {}
        covariance_max_error = None
        source_hash = sha256_file(paths[identifier])
        sample_hash = hashlib.sha256(pd.util.hash_pandas_object(
            d[['fund_id', 'fund_security_id', 'company_id', 'month']], index=True).values.tobytes()).hexdigest()
        for clusters, interpretation in schemes:
            reduced_rank, nesting = nested_rank(d, ['fund_security_id', 'month'], clusters)
            for convention in ['full_FE_rank', 'nested_FE_adjusted']:
                correction_fe_rank = result.diagnostics['fe_rank'] if convention == 'full_FE_rank' else reduced_rank
                cov, counts, q, denominator = cluster_covariance(d, x, residual, clusters,
                                                                correction_fe_rank, cache)
                df = min(counts[c] for c in clusters) - 1
                if clusters == ['fund_id', 'company_id'] and convention == 'full_FE_rank':
                    covariance_max_error = float(np.max(np.abs(cov - result.covariance)))
                    assert covariance_max_error < 1e-10
                diag = dict(result.diagnostics, inference_df=df, cluster_dimensions=clusters,
                            covariance_min_eigenvalue=float(np.linalg.eigvalsh(cov).min()),
                            negative_coefficient_variances=int((np.diag(cov) < 0).sum()))
                current = replace(result, covariance=cov, diagnostics=diag)
                cell = identifier + '__' + '_'.join(clusters) + '__' + convention
                h1 = h1_directional_test(current)
                h2 = h2_absolute_reaction_test(current)
                coefficients = []
                tests = []
                for row in current.table():
                    if row['term'] not in ['S_positive', 'S_negative']:
                        continue
                    row = clean(row)
                    row.update(cell=cell, model=identifier, clusters='+'.join(clusters), convention=convention,
                               effect_bps=10000 * row['beta'],
                               se_bps=None if row['se'] is None else 10000 * row['se'],
                               ci95_low_bps=None if row['ci95_low'] is None else 10000 * row['ci95_low'],
                               ci95_high_bps=None if row['ci95_high'] is None else 10000 * row['ci95_high'])
                    coefficients.append(row)
                    tests.append(dict(cell=cell, model=identifier, test=row['term'], p=row['p']))
                tests += [dict(cell=cell, model=identifier, test='H1_directional_IUT',
                               p=h1.get('intersection_union_p')),
                          dict(cell=cell, model=identifier, test='H2_absolute_union_null', p=h2.get('p_abs'))]
                adjust_holm(tests, 'p_holm_four_tests_in_cell')
                record = dict(cell=cell, model=identifier, dependence_scope=interpretation,
                              clusters=clusters, cluster_counts=counts, convention=convention,
                              correction_fe_rank=correction_fe_rank, full_fe_rank=result.diagnostics['fe_rank'],
                              small_sample_q=q, correction_denominator=denominator, nesting=nesting,
                              inference_df=df, N=len(d), funds=int(d.fund_id.nunique()),
                              companies=int(d.company_id.nunique()), months=int(d.month.nunique()),
                              min_month=str(d.month.min()), max_month=str(d.month.max()),
                              covariance_min_eigenvalue=diag['covariance_min_eigenvalue'],
                              negative_coefficient_variances=diag['negative_coefficient_variances'],
                              sample_hash=sample_hash, within_R2=result.diagnostics['within_r2'],
                              H1=h1, H2=h2, coefficients=coefficients, tests=tests)
                all_records.append(record)
                all_coefficients.extend(coefficients)
                all_tests.extend(tests)
        validations.append(dict(model=identifier, source_panel_sha256=source_hash,
                                unchanged_beta_max_abs_error=max_beta_difference,
                                reconstructed_original_covariance_max_abs_error=covariance_max_error,
                                N=len(d), sample_hash=sample_hash, projection_iterations=iterations))
        save(ROOT / 'checkpoint.json', dict(stage='MODEL_COMPLETED', model=identifier,
                                            covariance_cells=len(all_records)))
        print(json.dumps(dict(model=identifier, N=len(d), cells=len(all_records))), flush=True)
    adjust_holm(all_tests, 'p_holm_all_locked_covariance_tests_sensitivity')
    by_test = {(r['cell'], r['test']): r for r in all_tests}
    for record in all_records:
        record['tests'] = [by_test[(t['cell'], t['test'])] for t in record['tests']]
    output = dict(status='COVARIANCE_SENSITIVITY_COMPLETED', output_root=str(ROOT),
                  completed_at_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                  elapsed_seconds=time.monotonic() - start,
                  plan_sha256=sha256_file(ROOT / 'plan.json'),
                  runner_sha256=sha256_file(Path(__file__)),
                  records=all_records, coefficients=all_coefficients, tests=all_tests,
                  validations=validations,
                  interpretation='Same estimates and sample across covariance cells. Nested-FE correction is a disclosed alternative finite-sample convention. Single-cluster diagnostics leave a dependence dimension uncontrolled. Do not select the smallest p as proof; DW and original AT/alpha remain distinct.')
    save(ROOT / 'summary.json', output)
    pd.DataFrame(all_coefficients).to_csv(ROOT / 'coefficients.csv', index=False)
    pd.DataFrame(all_tests).to_csv(ROOT / 'tests.csv', index=False)
    save(ROOT / 'checkpoint.json', dict(stage='COMPLETED', covariance_cells=len(all_records),
                                        elapsed_seconds=output['elapsed_seconds']))
    print(json.dumps(dict(status=output['status'], cells=len(all_records), seconds=output['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
