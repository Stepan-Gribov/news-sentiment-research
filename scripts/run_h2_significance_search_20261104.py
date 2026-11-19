"""Frozen 660-label H2 specification search, computed only on research-host.

Family A uses exact cluster sufficient statistics; B/C use direct within OLS
and CGM scores. Every shortlisted unique design is verified using PanelOLS.
No input, historical result, or LLM database is mutated.
"""
from __future__ import annotations
import os
for _key in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ.setdefault(_key, '1')
import argparse
import copy
import datetime as dt
import gzip
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import socket
import sys
import time
import traceback
import types
import warnings
import numpy as np
import pandas as pd
try:
    from threadpoolctl import threadpool_limits
except ImportError:
    from contextlib import nullcontext
    def threadpool_limits(limits): return nullcontext()

PARTS = ['net_positive', 'net_negative', 'net_positive_lag1', 'net_negative_lag1']
VARIANTS = ['CURRENT_ONLY', 'CURRENT_PLUS_LAG1']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def clean(v):
    if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [clean(x) for x in v]
    if isinstance(v, np.ndarray): return clean(v.tolist())
    if isinstance(v, np.generic): return clean(v.item())
    if isinstance(v, float) and not np.isfinite(v): return None
    return v


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.suffix == '.gz':
        with gzip.open(path, 'wt', encoding='utf-8') as f: f.write(text)
    else: path.write_text(text, encoding='utf-8')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


def setup_within(d, outcome, controls):
    names = PARTS + list(controls)
    raw = d[[outcome] + names].astype(float)
    if not np.isfinite(raw).all().all(): raise ValueError('Nonfinite estimation value')
    within = (raw - raw.groupby(d.fund_id.astype(str), sort=False).transform('mean')).to_numpy()
    scales = np.sqrt(np.mean(within * within, axis=0))
    if np.any(scales <= 0) or not np.isfinite(scales).all(): raise ValueError('Absorbed full design')
    return names, scales, within / scales


def cluster_groups(d):
    groups = [d.fund_id.astype(str), d.company_id.astype(str), pd.MultiIndex.from_frame(d[['fund_id', 'company_id']].astype(str))]
    return [(sign, key, *pd.factorize(g, sort=False)) for sign, key, g in zip([1, 1, -1], ['fund', 'company', 'intersection'], groups)]


class MomentSolver:
    """Exact CGM covariance via quadratic cluster moments for all A subsets."""
    def __init__(self, d, outcome, controls, output):
        self.n = len(d); self.f = int(d.fund_id.nunique()); self.controls = list(controls)
        self.names, self.scales, z = setup_within(d, outcome, controls)
        self.gram = z[:, 1:].T @ z[:, 1:]; self.xy = z[:, 1:].T @ z[:, 0]; self.yy = float(z[:, 0] @ z[:, 0])
        self.ii, self.jj = np.triu_indices(z.shape[1]); q = len(self.ii); self.T = np.zeros((q, q)); self.cluster_counts = {}
        for sign, key, codes, labels in cluster_groups(d):
            ng = len(labels); self.cluster_counts[key] = ng
            if ng < 2: raise ValueError('Fewer than two clusters')
            moments = np.empty((ng, q))
            for j, (a, b) in enumerate(zip(self.ii, self.jj)):
                moments[:, j] = np.bincount(codes, weights=z[:, a] * z[:, b], minlength=ng)
            self.T += sign * ng / (ng - 1) * (self.n - 1) / self.n * (moments.T @ moments)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, gram=self.gram, xy=self.xy, yy=self.yy, scales=self.scales,
            T=self.T, ii=self.ii, jj=self.jj, names=np.asarray(self.names), n=self.n, funds=self.f)

    def fit(self, controls, distributed):
        nt = 4 if distributed else 2
        ix = list(range(nt)) + [4 + self.controls.index(c) for c in controls]
        gram = self.gram[np.ix_(ix, ix)]; bread = np.linalg.inv(gram); b = bread @ self.xy[ix]
        df = self.n - self.f - len(ix)
        if df <= 0: raise ValueError('Nonpositive residual degrees of freedom')
        fullb = np.zeros(len(self.names)); fullb[ix] = b; r = np.r_[1., -fullb]
        W = []
        for t in range(nt):
            factor = self.scales[0] / self.scales[1 + t]
            v = np.zeros(len(self.names) + 1); v[1 + np.asarray(ix)] = bread[t] * factor
            w = v[self.ii] * r[self.jj] + v[self.jj] * r[self.ii]
            w[self.ii == self.jj] *= .5; W.append(w)
        W = np.asarray(W)
        covariance = W @ self.T @ W.T * self.n / df
        beta = b[:nt] * self.scales[0] / self.scales[1:1 + nt]
        r2 = float((2 * b @ self.xy[ix] - b @ gram @ b) / self.yy)
        return beta, covariance, df, r2


class DirectSolver:
    """Direct rowwise within regression for the limited B/C designs."""
    def __init__(self, d, outcome, controls):
        self.d = d; self.n = len(d); self.f = int(d.fund_id.nunique()); self.controls = list(controls)
        self.names, self.scales, self.z = setup_within(d, outcome, controls)
        self.groups = cluster_groups(d)

    def fit(self, controls, distributed):
        nt = 4 if distributed else 2
        ix = list(range(nt)) + [4 + self.controls.index(c) for c in controls]
        x = self.z[:, 1 + np.asarray(ix)]; y = self.z[:, 0]
        gram = x.T @ x; bread = np.linalg.inv(gram); beta = bread @ (x.T @ y); residual = y - x @ beta
        df = self.n - self.f - len(ix)
        if df <= 0: raise ValueError('Nonpositive residual degrees of freedom')
        meat = np.zeros_like(gram)
        for sign, _, codes, labels in self.groups:
            ng = len(labels)
            if ng < 2: raise ValueError('Fewer than two clusters')
            scores = np.column_stack([np.bincount(codes, weights=x[:, j] * residual, minlength=ng) for j in range(len(ix))])
            meat += sign * ng / (ng - 1) * (self.n - 1) / self.n * (scores.T @ scores)
        covariance = bread @ meat @ bread * self.n / df
        factors = self.scales[0] / self.scales[1 + np.asarray(ix)]
        covariance *= np.outer(factors, factors)
        r2 = float(1 - residual @ residual / (y @ y))
        return beta[:nt] * factors[:nt], covariance[:nt, :nt], df, r2


def target_result(values, distributed, helper):
    beta, covariance, df, r2 = values; terms = PARTS if distributed else PARTS[:2]
    fake = types.SimpleNamespace(params=pd.Series(beta, index=terms), cov=pd.DataFrame(covariance, index=terms, columns=terms), df_resid=df)
    return dict(status='ESTIMATED', df_resid=int(df), rsquared_within=float(r2), target_terms=terms,
        target_beta=beta, target_covariance=covariance, h2_tests=helper.h2_tests(fake, distributed))


def specs(plan, catalog):
    result = []; by_id = {m['sample_id']: m for m in catalog}
    def add(model_id, family, group, variant, sample_id, controls, mask, outcome, tag):
        sample = by_id[sample_id]
        result.append(dict(model_id=model_id, family=family, group=group, variant=variant, sample_id=sample_id,
            row_hash=sample['ordered_row_hash'], outcome_column=outcome, outcome_tag=tag, controls=controls,
            control_mask=mask, n_controls=len(controls), support=sample['support'], cutoff=sample['cutoff'],
            nobs=sample['rows'], funds=sample['funds'], companies=sample['companies']))
    for group in plan['groups']:
        for variant in VARIANTS:
            for mask in list(range(64)) + ['ALL16']:
                controls = plan['all16_controls'] if mask == 'ALL16' else [c for j, c in enumerate(plan['six_controls']) if mask >> j & 1]
                label = 'ALL16' if mask == 'ALL16' else f'SIXMASK_{mask:02d}'
                add(f'A__{group}__{variant}__{label}', 'A', group, variant, 'A__' + group, controls, mask, 'quantity_change_pct', 'RAW_PERCENT')
    for sample in catalog:
        if sample['family'] != 'B': continue
        for variant in VARIANTS:
            add(sample['sample_id'] + '__' + variant, 'B', sample['group'], variant, sample['sample_id'],
                plan['six_controls'], 63, 'quantity_change_pct', 'RAW_PERCENT')
    for group in plan['groups']:
        for alternative in plan['family_C']['alternatives']:
            tag = alternative['tag']; sample_id = ('A__' if tag == 'SYMMETRIC_ALL' else 'C_CONTINUING__') + group
            add(f'C__{group}__{tag}', 'C', group, 'CURRENT_PLUS_LAG1', sample_id, plan['six_controls'], 63,
                alternative['dependent_column'], tag)
    assert len(result) == 660
    return result


def cachekey(spec):
    return (spec['row_hash'], spec['outcome_column'], spec['variant'], tuple(spec['controls']))


def flatten(spec, fit):
    rec = {k: v for k, v in spec.items() if k != 'controls'}
    rec['controls_included'] = '|'.join(spec['controls'])
    rec.update({k: fit.get(k) for k in ['status', 'df_resid', 'rsquared_within', 'reason']})
    for scope, test in fit.get('h2_tests', {}).get('tests', {}).items():
        for key in ['beta_positive', 'beta_negative', 'p_abs', 'expected_directional_signs', 'negative_absolute_minus_positive_absolute']:
            rec[scope + '_' + key] = test[key]
        rec[scope + '_p_difference'] = test['difference_zero']['p_native']; rec[scope + '_p_sum'] = test['sum_zero']['p_native']
        rec[scope + '_status'] = test['status']
        for sign in ['positive', 'negative']:
            coefficient = test[sign + '_coefficient']
            for out, key in [('se', 'se_native'), ('p', 'p_native'), ('ci_low', 'ci95_native_low'), ('ci_high', 'ci95_native_high')]:
                rec[f'{scope}_{sign}_{out}'] = coefficient[key]
    return rec


def native(d, spec, helper, root):
    from linearmodels.panel import PanelOLS
    targets = PARTS if spec['variant'] == 'CURRENT_PLUS_LAG1' else PARTS[:2]
    terms, dropped = helper.select_regressors(d, targets, spec['controls'])
    if terms != targets + spec['controls'] or dropped: raise AssertionError('Frozen controls/required targets changed')
    idx = pd.MultiIndex.from_arrays([d.fund_id.astype(str), pd.to_datetime(d.end_date)], names=['fund_id', 'report_date'])
    x = d[terms].astype(float).copy(); x.index = idx; x.insert(0, 'const', 1.)
    y = pd.Series(d[spec['outcome_column']].to_numpy(float), index=idx, name=spec['outcome_column'])
    clusters = pd.DataFrame({'fund_cluster': pd.Categorical(d.fund_id.astype(str)).codes,
        'company_cluster': pd.Categorical(d.company_id.astype(str)).codes}, index=idx)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = PanelOLS(y, x, entity_effects=True, time_effects=False, singletons=True, drop_absorbed=False, check_rank=True).fit(
            cov_type='clustered', clusters=clusters, cluster_time=False, cluster_entity=False,
            group_debias=True, debiased=True, auto_df=False, count_effects=True)
        assert result.nobs == len(d) and result.resids.index.equals(idx)
        summary_txt = str(result.summary); summary_html = result.summary.as_html(); ci = result.conf_int()
        coefficients = [dict(term=term, beta=float(result.params[term]), se_native=float(result.std_errors[term]),
            t_native=float(result.tstats[term]), p_native=float(result.pvalues[term]),
            ci95_native_low=float(ci.loc[term].iloc[0]), ci95_native_high=float(ci.loc[term].iloc[1])) for term in result.params.index]
        tests = helper.h2_tests(result, spec['variant'] == 'CURRENT_PLUS_LAG1')
        full_covariance = dict(terms=list(result.params.index), matrix=result.cov.to_numpy())
    model = spec['model_id']; dest = root / 'native'; dest.mkdir(exist_ok=True)
    (dest / (model + '.summary.txt')).write_text(summary_txt, encoding='utf-8')
    (dest / (model + '.summary.html')).write_text(summary_html, encoding='utf-8')
    record = dict(**spec, status='ESTIMATED_NATIVE', terms=list(result.params.index), coefficients=coefficients,
        full_covariance=full_covariance, h2_tests=tests, df_resid=int(result.df_resid), rsquared_within=float(result.rsquared_within),
        warnings=list(dict.fromkeys(str(w.message) for w in caught)), covariance_repair=False)
    save(dest / (model + '.json'), record)
    return record


def verify(fast, native_result):
    check = dict(model_id=native_result['model_id'], passed=True)
    def close(left, right):
        return np.allclose(np.asarray(left, dtype=float), np.asarray(right, dtype=float), rtol=2e-7, atol=2e-8, equal_nan=True)
    assert fast['df_resid'] == native_result['df_resid']
    assert np.isclose(fast['rsquared_within'], native_result['rsquared_within'], rtol=1e-8, atol=1e-10)
    for scope, test in fast['h2_tests']['tests'].items():
        actual = native_result['h2_tests']['tests'][scope]
        for key in ['beta_positive', 'beta_negative', 'p_abs']:
            assert close(test[key], actual[key]), (native_result['model_id'], scope, key)
        assert np.allclose(test['target_covariance'], actual['target_covariance'], rtol=2e-7, atol=2e-8, equal_nan=True)
        for sign in ['positive', 'negative']:
            for key in ['se_native', 'p_native', 'ci95_native_low', 'ci95_native_high']:
                assert close(test[sign + '_coefficient'][key], actual[sign + '_coefficient'][key])
    return check


def shortlist(frame, plan):
    reasons = {}
    def add(model, reason): reasons.setdefault(model, []).append(reason)
    for record in frame.loc[frame.family.eq('A') & frame.control_mask.astype(str).isin(['0', '63', 'ALL16'])].to_dict('records'):
        add(record['model_id'], 'FIXED_A_BASELINE_' + str(record['control_mask']))
    for family in ['A', 'B']:
        part = frame.loc[frame.family.eq(family)]
        for (group, variant), sub in part.groupby(['group', 'variant'], sort=False):
            for scope in plan['h2_scopes'][variant]:
                valid = sub.loc[np.isfinite(pd.to_numeric(sub[scope + '_p_abs'], errors='coerce'))]
                if len(valid):
                    model = valid.sort_values([scope + '_p_abs', 'model_id'], kind='stable').iloc[0].model_id
                    add(model, f'MIN_P_ABS_{family}_{group}_{variant}_{scope}')
    for model in frame.loc[frame.family.eq('C'), 'model_id']: add(model, 'ALL_ALTERNATIVE_OUTCOMES')
    assert len(reasons) <= plan['selection_policy']['maximum_native_shortlist_labels']
    return reasons


def render(root, summary, shortlist_frame, minima):
    columns = ['model_id', 'selection_reasons', 'nobs', 'funds', 'companies', 'n_controls', 'rsquared_within',
        'current_p_abs', 'lag1_p_abs', 'total_current_plus_lag1_p_abs', 'controls_included']
    body = ['<!doctype html><html lang="ru"><meta charset="utf-8"><title>H2: ограниченный поиск спецификаций</title>',
        '<style>body{font:15px system-ui;margin:36px;color:#172337;line-height:1.45}p{max-width:1200px}table{border-collapse:collapse;font-size:12px}td,th{padding:7px;border-bottom:1px solid #ccd4df;text-align:left}.scroll{overflow:auto}a{color:#155b99}</style>',
        '<h1>H2: 660 сохранённых спецификаций</h1>',
        '<p>Исследовательский поиск после просмотра прежних результатов. Все варианты сохранены, включая незначимые. Минимальный p выбран из сетки и не является новым подтверждающим тестом. Поправки на множественные проверки по указанию пользователя не применялись.</p>',
        '<p>A: 64 поднабора SIX и ALL16, четыре прежние точные выборки, current-only и current+lag (520). B: SIX, четыре группы × четыре прежних P99/P97,5/P95/P90 × четыре правила истории × два дизайна (128). История считается до потерь по лагу и отсечений. Current-only использует те же common-lag строки. C: по четыре модели continuing-percent, continuing-log и symmetric-all (12).</p>',
        '<p>Во всех моделях fund FE, кластеры фонд/эмитент, const, без временных эффектов и временных кластеров. H2 — равенство модулей двух коэффициентов. p_abs=max(p_diff,p_sum); малый p_diff сам по себе не означает H2. Для сумм current+lag используется полная ковариация. Все знаки, компоненты тестов, SE и CI доступны в полном CSV.</p>',
        '<p>C меняет шкалу и/или состав, поэтому не заменяет основную raw-percent H2: log=100 ln(Qclose/Qopen) только при строго положительных Q; symmetric=200ΔQ/(Qclose+Qopen), входы+200 и выходы−200. Новых отсечений по преобразованному Y нет. Исходный quantity_change_pct не изменён.</p>',
        '<p>Статус: <b>' + html.escape(summary['status']) + '</b>; независимая проверка ожидается.</p>',
        '<h2>Минимумы по семействам, группам и тестам</h2><div class="scroll">' + pd.DataFrame(minima).to_html(index=False, border=0, float_format=lambda x: f'{x:.6g}') + '</div>',
        '<h2>Проверенные в PanelOLS контрольные и выбранные модели</h2><div class="scroll">' + shortlist_frame[columns].to_html(index=False, border=0, float_format=lambda x: f'{x:.6g}') + '</div>',
        '<p><a href="all_models.csv.gz">Все 660 моделей CSV.gz</a> · <a href="all_h2_tests.csv">Все тесты и знаки</a> · <a href="shortlist.csv">Короткий список</a> · <a href="native_coefficients.csv">Native коэффициенты</a> · <a href="native_results.json">Native параметры/полная ковариация</a> · <a href="summary.json">Сводка JSON</a> · <a href="source_manifest.json">Источники</a> · <a href="sample_catalog.json">Все masks</a> · <a href="frozen_plan.json">План до коэффициентов</a></p><ul>']
    for model in summary['native_models']:
        body.append('<li>' + html.escape(model) + ': <a href="native/' + model + '.summary.txt">TXT</a> · <a href="native/' + model + '.summary.html">HTML</a></li>')
    body.append('</ul><p>Большие канонические данные, строки масок и sufficient statistics остаются на research-host. Ни данные LLM, ни прежние результаты и документы не изменялись.</p></html>')
    (root / 'review.html').write_text('\n'.join(body), encoding='utf-8')


def execute(args):
    if sys.platform != 'win32': raise RuntimeError('Estimation is restricted to research-host')
    plan_path = Path(args.plan); assert sha(plan_path) == args.plan_sha
    plan = json.loads(plan_path.read_text(encoding='utf-8')); root = Path(plan['output_root'])
    assert plan['status'] == 'FIXED_BEFORE_NEW_COEFFICIENTS' and plan['model_counts']['total_labels'] == 660
    assert plan['targets'] == PARTS and plan['variants'] == VARIANTS
    assert not plan['multiple_testing_adjustments'] and not plan['time_FE'] and not plan['cluster_time'] and not plan['covariance_repair']
    assert not any(p.lower() == 'databaseresources' for p in root.parts)
    assert sha(plan['preflight_helper_path']) == plan['preflight_helper_sha256']
    assert sha(plan['prefit_mask_audit_path']) == plan['prefit_mask_audit_sha256']
    if (root / 'summary.json').exists(): raise FileExistsError('Completed search may not be overwritten')
    lock = root / 'run.lock'; fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd)
    started = time.perf_counter(); native_results = {}; native_by_key = {}; checks = []; fit_results = {}; cached = {}; allrows = []
    def checkpoint(state, **extra):
        v = dict(state=state, elapsed_seconds=time.perf_counter() - started, **extra)
        save(root / 'checkpoint.json', v); print(json.dumps(clean(v), ensure_ascii=True), flush=True)
    try:
        save(root / 'frozen_plan.json', plan); checkpoint('PREPARING_FIXED_MASKS')
        preflight = load(plan['preflight_helper_path'], 'h2_search_preflight')
        helper = load(plan['h2_helper'], 'h2_search_helpers')
        source_paths = {k: plan[k] for k in plan['expected_source_sha256']}
        source_paths.update({'exact_' + d['group']: d['input_panel'] for d in plan['exact_samples']})
        source_paths.update(h2_helper=plan['h2_helper'], preflight_helper=plan['preflight_helper_path'])
        sources = {k: dict(path=p, sha256_before=sha(p), bytes=Path(p).stat().st_size) for k, p in source_paths.items()}
        frame, masks, catalog, support, cuts = preflight.prepare(plan)
        preaudit = json.loads(Path(plan['prefit_mask_audit_path']).read_text(encoding='utf-8'))
        assert catalog == preaudit['samples']
        numeric_q = frame[['quantity_open', 'quantity_close']].astype(float)
        assert np.isfinite(numeric_q).all().all() and numeric_q.ge(0).all().all()
        q0 = numeric_q.quantity_open; q1 = numeric_q.quantity_close; continuing = q0.gt(0) & q1.gt(0)
        assert (q0 + q1).gt(0).all()
        alt = pd.DataFrame({'row_id': frame.row_id})
        alt['h2_y_continuing_percent'] = frame.quantity_change_pct.where(continuing)
        alt['h2_y_continuing_log'] = np.nan
        alt.loc[continuing, 'h2_y_continuing_log'] = 100 * (np.log(q1.loc[continuing]) - np.log(q0.loc[continuing]))
        alt['h2_y_symmetric'] = 200 * (q1 - q0) / (q1 + q0)
        assert np.isfinite(alt.loc[continuing].drop(columns='row_id')).all().all()
        assert np.isfinite(alt.h2_y_symmetric).all()
        assert np.allclose(alt.loc[q0.eq(0), 'h2_y_symmetric'], 200, rtol=0, atol=1e-12)
        assert np.allclose(alt.loc[q1.eq(0), 'h2_y_symmetric'], -200, rtol=0, atol=1e-12)
        data = root / 'data'; data.mkdir(exist_ok=True)
        frame.to_parquet(data / 'canonical_signed_panel.parquet', index=False, compression='zstd')
        mask_frame = pd.DataFrame({'row_id': frame.row_id, **masks})
        mask_frame.to_parquet(data / 'sample_masks.parquet', index=False, compression='zstd')
        alt.to_parquet(data / 'alternative_outcomes.parquet', index=False, compression='zstd')
        for column in alt.columns:
            if column != 'row_id': frame[column] = alt[column]
        save(root / 'sample_catalog.json', catalog); save(root / 'thresholds.json', cuts)
        pd.DataFrame(support).to_csv(root / 'fund_support.csv', index=False, encoding='utf-8-sig')
        models = specs(plan, catalog); bymodel = {s['model_id']: s for s in models}
        save(root / 'model_catalog.json', models)
        def subset(spec):
            d = frame.loc[masks[spec['sample_id']]].reset_index(drop=True)
            assert helper.row_hash(d) == spec['row_hash'] and len(d) == spec['nobs']
            assert d.complete_current_lag1.all()
            return d
        def store(spec, values=None, issue=None):
            key = cachekey(spec)
            if key in cached:
                fit = fit_results[cached[key]]; alias = cached[key]
            else:
                fit = target_result(values, spec['variant'] == 'CURRENT_PLUS_LAG1', helper) if issue is None else dict(status='NOT_ESTIMABLE', reason=issue)
                cached[key] = spec['model_id']; alias = None
            fit_results[spec['model_id']] = fit
            rec = flatten(spec, fit); rec['exact_alias_of'] = alias; allrows.append(rec)
            return fit
        def get_native(spec):
            key = cachekey(spec)
            if key in native_by_key: return native_results[native_by_key[key]]
            checkpoint('NATIVE_VERIFY', model_id=spec['model_id'])
            result = native(subset(spec), spec, helper, root)
            checks.append(verify(fit_results[spec['model_id']], result))
            native_results[spec['model_id']] = result; native_by_key[key] = spec['model_id']
            return result
        with threadpool_limits(limits=1):
            for group in plan['groups']:
                group_specs = [s for s in models if s['family'] == 'A' and s['group'] == group]
                checkpoint('A_BUILD_SUFFICIENT_STATISTICS', group=group)
                solver = MomentSolver(subset(group_specs[0]), 'quantity_change_pct', plan['all16_controls'], root / 'statistics' / (group + '.npz'))
                for spec in group_specs:
                    try: values = solver.fit(spec['controls'], spec['variant'] == 'CURRENT_PLUS_LAG1')
                    except (ValueError, np.linalg.LinAlgError) as error: store(spec, issue=str(error))
                    else: store(spec, values)
                for spec in group_specs:
                    if spec['control_mask'] == 'ALL16': get_native(spec)
                checkpoint('A_GROUP_COMPLETE', group=group, labels=len(group_specs))
            checkpoint('A_COMPLETE', labels=520, unique_fast_fits=len(cached))
            direct = None; direct_key = None
            for index, spec in enumerate([s for s in models if s['family'] in ['B', 'C']]):
                if cachekey(spec) in cached: store(spec); continue
                key = (spec['row_hash'], spec['outcome_column'])
                try:
                    if key != direct_key:
                        direct = DirectSolver(subset(spec), spec['outcome_column'], plan['six_controls']); direct_key = key
                    values = direct.fit(spec['controls'], spec['variant'] == 'CURRENT_PLUS_LAG1')
                except (ValueError, np.linalg.LinAlgError) as error: store(spec, issue=str(error))
                else: store(spec, values)
                if index % 16 == 15 or spec['family'] == 'C': checkpoint('GRID_PROGRESS', family=spec['family'], model_id=spec['model_id'], completed_labels=len(allrows))
            checkpoint('B_C_COMPLETE', labels=len(allrows), unique_fast_fits=len(cached))
            allframe = pd.DataFrame(allrows)
            assert len(allframe) == 660 and allframe.model_id.is_unique
            assert len(cached) == plan['model_counts']['expected_unique_sample_outcome_design_control_fits']
            allframe.to_csv(root / 'all_models.csv', index=False, encoding='utf-8-sig')
            allframe.to_csv(root / 'all_models.csv.gz', index=False, encoding='utf-8-sig', compression='gzip')
            reasons = shortlist(allframe, plan)
            selected = allframe.loc[allframe.model_id.isin(reasons)].copy()
            selected['selection_reasons'] = selected.model_id.map(lambda m: '|'.join(reasons[m]))
            native_mapping = {}
            for model in selected.model_id:
                spec = bymodel[model]
                if fit_results[model]['status'] != 'ESTIMATED':
                    native_mapping[model] = None; continue
                result = get_native(spec); native_mapping[model] = result['model_id']
            selected['native_model_id'] = selected.model_id.map(native_mapping)
            selected.to_csv(root / 'shortlist.csv', index=False, encoding='utf-8-sig')
        save(root / 'native_results.json', native_results)
        native_coefs = [dict(model_id=model, **c) for model, result in native_results.items() for c in result['coefficients']]
        pd.DataFrame(native_coefs).to_csv(root / 'native_coefficients.csv', index=False, encoding='utf-8-sig')
        testrows = []
        for record in allrows:
            for scope in plan['h2_scopes'][record['variant']]:
                fields = {k[len(scope) + 1:]: v for k, v in record.items() if k.startswith(scope + '_')}
                testrows.append({**{k: record[k] for k in ['model_id', 'family', 'group', 'variant', 'nobs', 'funds', 'companies']}, 'scope': scope, **fields})
        tests_frame = pd.DataFrame(testrows); assert len(tests_frame) == 1332
        tests_frame.to_csv(root / 'all_h2_tests.csv', index=False, encoding='utf-8-sig')
        save(root / 'all_target_covariance.json', {m: {k: f.get(k) for k in ['status', 'target_terms', 'target_beta', 'target_covariance', 'df_resid']} for m, f in fit_results.items()})
        save(root / 'fit_details.json.gz', fit_results)
        save(root / 'native_verification.json', dict(PASS=len(checks), FAIL=0, checks=checks))
        counts = []; minima = []
        for (family, group, variant), sub in allframe.groupby(['family', 'group', 'variant'], sort=False):
            for scope in plan['h2_scopes'][variant]:
                pcol = scope + '_p_abs'; valid = sub.loc[np.isfinite(pd.to_numeric(sub[pcol], errors='coerce'))]
                count = dict(family=family, group=group, variant=variant, scope=scope, labels=len(sub), finite=len(valid),
                    p_lt_05=int((valid[pcol] < .05).sum()), minimum_p=float(valid[pcol].min()) if len(valid) else None)
                counts.append(count)
                if len(valid):
                    row = valid.sort_values([pcol, 'model_id']).iloc[0]
                    minima.append(dict(**count, model_id=row.model_id, nobs=int(row.nobs), controls=row.controls_included,
                        beta_positive=row[scope + '_beta_positive'], beta_negative=row[scope + '_beta_negative'],
                        expected_directional_signs=row[scope + '_expected_directional_signs']))
        pd.DataFrame(counts).to_csv(root / 'family_counts.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(minima).to_csv(root / 'minima.csv', index=False, encoding='utf-8-sig')
        for item in sources.values():
            item['sha256_after'] = sha(item['path']); item['unchanged'] = item['sha256_before'] == item['sha256_after']
            assert item['unchanged']
        assert sha(plan_path) == args.plan_sha
        manifest = dict(sources=sources, plan_sha256=args.plan_sha, runner_sha256=sha(__file__),
            all_sources_unchanged=True, execution_host=socket.gethostname(), bulk_location=str(data),
            generated_data={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in data.iterdir()})
        save(root / 'source_manifest.json', manifest)
        summary = dict(status='COMPLETE_PENDING_INDEPENDENT_QA', plan=plan, model_labels=660, unique_fast_fits=len(cached),
            exact_alias_labels=int(allframe.exact_alias_of.notna().sum()), shortlist_labels=len(selected),
            native_models=list(native_results), native_unique_fits=len(native_results), native_verification=dict(PASS=len(checks),FAIL=0),
            family_counts=counts, minima=minima, shortlist=selected.to_dict('records'),
            manifest=manifest, elapsed_seconds=time.perf_counter()-started, independent_QA='PENDING',
            all_results_preserved=True, selected_minima_are_exploratory=True, multiple_testing_adjustments=False)
        save(root / 'summary.json', summary); render(root, summary, selected, minima)
        completion = dict(status=summary['status'], model_labels=660, unique_fast_fits=len(cached), native_unique_fits=len(native_results),
            elapsed_seconds=time.perf_counter()-started, completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        save(root / 'completion.json', completion); checkpoint('COMPLETE_PENDING_INDEPENDENT_QA', **{k:v for k,v in completion.items() if k not in ['status','elapsed_seconds']})
        save(root / 'output_manifest.json', {str(p.relative_to(root)).replace('\\','/'):dict(bytes=p.stat().st_size,sha256=sha(p))
            for p in root.rglob('*') if p.is_file() and p.name not in ['run.lock','output_manifest.json']})
    except Exception as error:
        save(root / 'failure.json', dict(error=str(error),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started)); raise
    finally: lock.unlink(missing_ok=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--plan-sha',required=True)
    execute(parser.parse_args())
