"""H1-only Python comparison report. Actual estimation runs on research-host."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import html
import json
import os
import sys

BASE = Path('./analysis_outputs/title_name_regressions_20261026_w2_v2')
BRIDGE = Path('./analysis_outputs/title_name_exact_price_bridge_20261029/v2_deterministic_absorption')
ROOT = Path('./analysis_outputs/h1_panel_report_20261030')
sys.path.insert(0, str(BASE / 'code'))
import numpy as np
import pandas as pd
from scipy import stats
from title_name_regressions_core import sha256_file
from title_name_regressions_estimation import fit_panel_fe, h1_directional_test

LABELS = {'S_positive': 'Positive news, S+', 'S_negative': 'Negative news, |S-|',
          'opening_weight': 'Opening weight, p.p.', 'lag_log_news_count': 'Lag log(1 + news count)'}
TERMS = list(LABELS)


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def stars(p):
    return '***' if p < .01 else '**' if p < .05 else '*' if p < .1 else ''


def cells(result):
    return {c['term']: c for c in result['coefficients']}


def comparison_rows(results):
    rows = []
    for term in TERMS:
        values = []
        for result in results:
            c = cells(result).get(term)
            values.append(None if c is None else [f"{c['display_beta']:.4f}{stars(c['p'])}", f"({c['display_se']:.4f})"])
        rows.append(dict(label=LABELS[term], values=values, kind='coefficient'))
    for label, key in [('Fund x security FE', None), ('Fund x interval FE', None),
                       ('Intercept', None), ('Observations', 'nobs'), ('Funds', 'funds'),
                       ('Issuers', 'companies'), ('Fund-security pairs', 'unit_effects'),
                       ('Within R-squared', 'within_r2')]:
        values = []
        for result in results:
            value = result['diagnostics'].get(key) if key else 'Yes'
            if label == 'Intercept': value = 'Absorbed by FE'
            elif key == 'within_r2': value = f'{value:.4f}'
            elif key: value = f'{value:,}'
            values.append([str(value)])
        rows.append(dict(label=label, values=values, kind='statistic'))
    rows.append(dict(label='Primary H1 directional p', values=[['--'], ['--'], [f"{results[2]['H1']['intersection_union_p']:.4f}"]], kind='statistic'))
    return rows


def render_reports(output):
    results = output['models']
    rows = comparison_rows(results)
    output['comparison_rows'] = rows
    lines = ['H1: PANEL FIXED-EFFECT REGRESSION COMPARISON',
             'Dependent variable: 100 * [log(V_close/V_open) - log(P_close/P_open)]',
             'Continuing-position price-adjusted growth proxy; not whole-portfolio AT.',
             'Common sample: June2018-May2026;93 observed closing months;28-94day actual intervals.',
             'CGM covariance: fund + issuer clusters; CR1 with exact full absorbed FE rank.',
             'Stars use RAW TWO-SIDED coefficient p: *p<.10, **p<.05, ***p<.01.',
             'SE in parentheses. Columns1/2 auxiliary; column3 is the primary H1 specification.', '',
             f"{'':<34}{'(1) Positive only':>22}{'(2) Negative only':>22}{'(3) Joint: PRIMARY':>22}", '=' * 100]
    for row in rows:
        vals = ['--' if value is None else value[0] for value in row['values']]
        lines.append(f"{row['label']:<34}" + ''.join(f'{v:>22}' for v in vals))
        if row['kind'] == 'coefficient':
            lines.append(' ' * 34 + ''.join(f"{('--' if value is None else value[1]):>22}" for value in row['values']))
    lines += ['=' * 100, 'Displayed dependent variable is in log percentage points; opening weight is in percentage points.',
              'NAV, HHI and interval duration are absorbed by fund-specific interval FE.',
              'Primary H1 requires beta_positive>0 AND beta_negative<0; stars on an auxiliary column do not establish H1.',
              'The primary directional IUT uses the pre-existing one-sided sign alternative; stars above use two-sided p.',
              'Post-hoc exploratory report; no multiplicity-adjusted confirmation is claimed.', '']
    for result in results:
        dg = result['diagnostics']
        lines += ['=' * 100, result['title'],
                  f"Estimator: absorbed OLS with two fixed effects; N={dg['nobs']}; residual df={dg['residual_df']}; inference df={dg['inference_df']}",
                  f"Fund-security FE groups={dg['unit_effects']}; fund-interval FE groups={dg['time_effects']}; connected components={dg['fe_connected_components']}; exact FE rank={dg['fe_rank']}",
                  f"Clusters: funds={dg['fund_clusters']}, issuers={dg['company_clusters']}; within R2={dg['within_r2']:.8f}",
                  'term                             coefficient       std.err.        t           p                 95% CI']
        for c in result['coefficients']:
            lines.append(f"{c['term']:<31} {c['display_beta']: 13.7f} {c['display_se']: 13.7f} {c['t']: 10.4f} {c['p']: .8f} [{c['display_ci_low']: .7f}, {c['display_ci_high']: .7f}]")
        test = result['all_slope_Wald_F']
        lines += [f"All-slope robust Wald F({test['df_num']},{test['df_denom']})={test['F']:.6f}; p={test['p']:.8g}; this is NOT the H1 sign test."]
        if result['H1'] is not None: lines += ['Primary H1 directional IUT: ' + json.dumps(result['H1'])]
        lines.append('')
    (ROOT / 'h1_panel_report.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    body = []
    for index, row in enumerate(rows):
        kind = 'coef' if row['kind'] == 'coefficient' else 'stat'
        if index == 4: kind += ' section'
        values = []
        for value in row['values']:
            text = '&mdash;' if value is None else '<br>'.join(html.escape(s) for s in value)
            values.append('<td>' + text + '</td>')
        body.append('<tr class="' + kind + '"><td>' + html.escape(row['label']) + '</td>' + ''.join(values) + '</tr>')
    page = '<!doctype html><meta charset="utf-8"><title>H1 panel regression</title>'
    page += '<style>body{background:white;color:#111;font:18px Georgia,"Times New Roman",serif;max-width:1050px;margin:45px auto;padding:0 25px}h1{font-size:25px;text-align:center}p{text-align:center}table{border-collapse:collapse;width:100%;margin-top:25px}th,td{padding:8px 12px;text-align:center}th:first-child,td:first-child{text-align:left;width:40%}thead{border-top:2px solid #222;border-bottom:1px solid #222}tr.section{border-top:1px solid #555}tr.coef td{padding-top:12px;padding-bottom:12px;line-height:1.6}tbody{border-bottom:2px solid #222}footer{font-size:14px;line-height:1.55;margin-top:14px}small{font-size:15px}</style>'
    page += '<h1>H1. News and price-adjusted position growth</h1><p>Dependent variable: 100 &times; [log(V<sub>1</sub>/V<sub>0</sub>) &minus; log(P<sub>1</sub>/P<sub>0</sub>)]</p>'
    page += '<p><small>June 2018 &ndash; May 2026 &middot; Identical observations in all columns</small></p><table><thead><tr><th></th><th>(1)<br>Positive only</th><th>(2)<br>Negative only</th><th>(3)<br>Joint &mdash; primary</th></tr></thead><tbody>' + ''.join(body) + '</tbody></table>'
    page += '<footer>Standard errors in parentheses. * p &lt; 0.10; ** p &lt; 0.05; *** p &lt; 0.01 (raw two-sided p).<br>Two-way CGM clustering by fund and issuer; full absorbed-rank finite-sample correction.<br>NAV, HHI and duration are absorbed by fund-interval FE. S-negative is the magnitude of the negative signal.<br>Columns (1) and (2) are auxiliary; only column (3) is the joint H1 test. This outcome is a continuing-position proxy, not whole-portfolio AT.</footer>'
    (ROOT / 'h1_panel_table.html').write_text(page, encoding='utf-8')
    save(ROOT / 'h1_panel_results.json', output)


def main():
    if os.name != 'nt': raise RuntimeError('Actual data and fits must stay on research-host')
    if (ROOT / 'h1_panel_results.json').exists(): raise RuntimeError('Completed report is immutable')
    ROOT.mkdir(parents=True, exist_ok=True)
    plan = json.loads((ROOT / 'plan.json').read_text(encoding='utf-8'))
    assert plan['status'] == 'FIXED_BEFORE_TWO_AUXILIARY_H1_FITS'
    reference_path = BRIDGE / 'packaging_v1/summary.json'
    reference = json.loads(reference_path.read_text(encoding='utf-8'))
    for name, expected in reference['manifest']['base_code_hashes'].items():
        assert sha256_file(BASE / 'code' / name) == expected, name
    sample_path = Path(plan['sample'])
    sample_sha = sha256_file(sample_path)
    d = pd.read_parquet(sample_path)
    keys = ['fund_id', 'security_id', 'start_date', 'end_date']
    hash_frame = lambda cols: hashlib.sha256(pd.util.hash_pandas_object(d[cols], index=False).values.tobytes()).hexdigest()
    assert hash_frame(keys) == plan['sample_hash']
    assert hash_frame(keys + ['S']) == plan['sealed_S_hash']
    assert len(d) == 45974
    absorbed = {c: int(d.groupby('interval_id', observed=True)[c].nunique().max()) for c in plan['absorbed_controls']}
    assert all(n == 1 for n in absorbed.values())
    models = []
    for spec in plan['models']:
        targets = spec['targets']
        result = fit_panel_fe(d, plan['outcome'].split(' = ')[0], targets + plan['common_controls'],
                              time='interval_id', allow_absorbed=plan['common_controls'],
                              allow_collinear=plan['common_controls'], drop_singletons=False)
        assert len(result.sample) == len(d) and result.sample.index.equals(d.index)
        assert result.names == targets + plan['common_controls']
        coefficients = result.table()
        for c in coefficients:
            scale = 1 if c['term'] == 'opening_weight' else 100
            c.update(display_beta=c['beta'] * scale, display_se=c['se'] * scale,
                     display_ci_low=c['ci95_low'] * scale, display_ci_high=c['ci95_high'] * scale,
                     stars=stars(c['p']))
        assert np.linalg.eigvalsh(result.covariance).min() > 0
        k = len(result.names)
        wald = float(result.beta @ np.linalg.solve(result.covariance, result.beta) / k)
        dg = {key: int(value) if isinstance(value, (np.integer,)) else float(value) if isinstance(value, (np.floating,)) else value for key, value in result.diagnostics.items()}
        models.append(dict(column=spec['column'], title=spec['column'] + ' ' + spec['role'],
                           targets=targets, coefficients=coefficients, diagnostics=dg,
                           all_slope_Wald_F=dict(F=wald, df_num=k, df_denom=dg['inference_df'], p=float(stats.f.sf(wald, k, dg['inference_df']))),
                           H1=h1_directional_test(result) if len(targets) == 2 else None))
        print(spec['column'], 'N', dg['nobs'], 'within_R2', dg['within_r2'], flush=True)
    ref = next(r for r in reference['records'] if r['model'] == 'E3_PRICE_ADJUSTED_VALUE_GROWTH_EXACT_MATCHED' and r['convention'] == 'full_FE_rank' and len(r['clusters']) == 2)
    old = {c['term']: c for c in ref['coefficients']}
    errors = {key: max(abs(c[key] - old[c['term']][key]) for c in models[2]['coefficients']) for key in ['beta', 'se', 'p', 'ci95_low', 'ci95_high']}
    assert all(error < 1e-7 for error in errors.values()), errors
    assert abs(models[2]['diagnostics']['within_r2'] - ref['within_R2']) < 1e-9
    assert abs(models[2]['H1']['intersection_union_p'] - ref['H1']['intersection_union_p']) < 1e-7
    assert sha256_file(sample_path) == sample_sha
    output = dict(status='COMPLETED_H1_ONLY_THREE_COLUMN_REPORT', created_at_utc=datetime.now(timezone.utc).isoformat(),
                  output_root=str(ROOT), estimation_host='research-host', plan_sha256=sha256_file(ROOT / 'plan.json'),
                  runner_sha256=sha256_file(Path(__file__)), sample_path=str(sample_path), sample_sha256=sample_sha,
                  sample_hash=hash_frame(keys), sealed_S_hash=hash_frame(keys + ['S']), reference_sha256=sha256_file(reference_path),
                  structural_absorption=absorbed, primary_reproduction_max_errors=errors, models=models,
                  all_models_identical_observations=True, H2_H3_estimated=False,
                  interpretation='Primary model is column3. One-signal auxiliary models do not replace the joint directional H1. Price-adjusted continuing-position proxy, not whole-portfolio AT. Coefficient stars are raw two-sided p, not multiplicity-adjusted confirmation.')
    render_reports(output)
    print('COMPLETED', str(ROOT), flush=True)


if __name__ == '__main__':
    main()
