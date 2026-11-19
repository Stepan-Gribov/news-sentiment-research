"""Four frozen H2 signed-part SIX regressions on the previous H1 samples.

Actual estimation is restricted to research-host. This runner never rebuilds a sample,
updates a source, or searches controls/thresholds. All four results are retained.
"""
from __future__ import annotations

import os
for _name in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ.setdefault(_name, '1')

import argparse
import datetime as dt
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import platform
import socket
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd

CONTROLS = ['closing_HHI', 'bond_indicator', 'imoex_ret', 'RVI', 'dummy_mar2014', 'dummy_sep2018']
PARTS = ['net_positive', 'net_negative', 'net_positive_lag1', 'net_negative_lag1']
GROUPS = ['pooled', 'equity', 'bond', 'mixed']
SCOPES = ['current', 'lag1', 'total_current_plus_lag1']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def save(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load(path):
    spec = importlib.util.spec_from_file_location('frozen_h2_helper', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sign_counts(frame, source, period_id):
    records = {}
    for scope, data in [('regression_rows', frame), ('distinct_company_windows', frame.drop_duplicates(['company_id', period_id]))]:
        values = pd.to_numeric(data[source], errors='coerce')
        records[scope] = dict(n=len(values), positive=int(values.gt(0).sum()),
            negative=int(values.lt(0).sum()), zero=int(values.eq(0).sum()),
            missing=int(values.isna().sum()), minimum=float(values.min()), maximum=float(values.max()))
    return records


def render(root, summary, comparisons, coefficients, testrows):
    # A readable small report; the native outputs and exact covariances remain linked.
    sections = ['<!doctype html><html lang="ru"><meta charset="utf-8">',
        '<title>H2 — четыре модели SIX с текущим сигналом и лагом</title>',
        '<style>body{font:16px system-ui;max-width:1450px;margin:36px auto;padding:0 22px;color:#182536;line-height:1.45}table{border-collapse:collapse;font-size:13px;width:100%}th,td{border-bottom:1px solid #d5dbe2;padding:7px;text-align:right}th:first-child,td:first-child{text-align:left}.scroll{overflow-x:auto}a{color:#155da1}code{font-size:13px}h2{margin-top:34px}</style>',
        '<h1>H2: асимметрия абсолютной реакции, SIX + lag1</h1>',
        '<p>Четыре заранее зафиксированные модели на тех же строках, что последние H1: общая выборка, акционные, облигационные и смешанные фонды. Положительная часть = max(Net, 0); отрицательная часть = max(−Net, 0). Обе части и их точный предыдущий отчётный лаг включены одновременно. Y — процентное изменение расчётного количества бумаг.</p>',
        '<p>Контроли: closing_HHI, bond_indicator, imoex_ret, RVI, dummy_mar2014, dummy_sep2018. Fund FE, кластеры фонд/эмитент; без временных эффектов и временных кластеров. Шесть контролей перенесены из прежнего исследовательского выбора H1; нового перебора не было. Const сохранена.</p>',
        '<p>Общая выборка без порога 60/24. Три группы сохраняют прежний отбор ≥60 строк и ≥24 интервалов до потерь по лагу; они не составляют полного разбиения общей выборки. Прежние P99=322,777164%, условные входы +100% и выходы −100%, NA и точные лаги не менялись. Смешанные 60/24 используют прежний точный alias файла 30/24.</p>',
        '<p>H2: |β+|=|β−|. Это объединение двух нулей β+−β−=0 ИЛИ β++β−=0. Единый p_abs=max(p_diff,p_sum), отклонение требует отклонить оба компонента. Это intersection-union тест одной гипотезы, не поправка на множественные проверки. Показаны обычные двусторонние native p, без поправок и без изменения ковариации. Для суммы current+lag используется полная ковариация четырёх коэффициентов.</p>',
        '<p><b>Статус:</b> ' + html.escape(summary['status']) + '; независимая проверка: ' + html.escape(summary['independent_QA']) + '.</p>',
        '<h2>Четыре модели</h2><div class="scroll">' + comparisons.to_html(index=False, border=0, float_format=lambda x: f'{x:.7g}') + '</div>',
        '<h2>H2: текущий сигнал, лаг и сумма</h2><div class="scroll">' + testrows.to_html(index=False, border=0, float_format=lambda x: f'{x:.7g}') + '</div>',
        '<p><a href="summary.json">Полные результаты JSON</a> · <a href="summary.csv">Сводка CSV</a> · <a href="coefficients.csv">Все коэффициенты/SE/p/CI</a> · <a href="full_covariance.json">Полная ковариация</a> · <a href="h2_tests.json">Все контрасты H2</a> · <a href="sign_counts.json">Знаки сигналов</a> · <a href="source_manifest.json">Источники и SHA</a> · <a href="frozen_plan.json">Фиксированный план</a></p>']
    for record in summary['models']:
        group = record['group']
        sections += ['<h2>' + html.escape(record['label_ru']) + '</h2>',
            '<p><a href="' + group + '.summary.txt">Оригинальный TXT</a> · <a href="' + group + '.summary.html">Оригинальный HTML</a></p>',
            '<div class="scroll">' + coefficients.loc[coefficients.group.eq(group)].drop(columns='group').to_html(index=False, border=0, float_format=lambda x: f'{x:.7g}') + '</div>']
    sections.append('<p>Все четыре результата сохранены независимо от значимости. Связи являются условными ассоциациями; абсолютная асимметрия сама по себе не доказывает более сильную отрицательную реакцию, правильные знаки H1, доходность или причинный эффект.</p></html>')
    (root / 'review.html').write_text('\n'.join(sections), encoding='utf-8')


def run(plan_path, expected_plan_sha):
    if sys.platform != 'win32': raise RuntimeError('Actual regression execution must occur on research-host')
    plan_path = Path(plan_path)
    if sha(plan_path) != expected_plan_sha: raise AssertionError('Fixed plan SHA mismatch')
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    assert plan['status'] == 'FIXED_BEFORE_FOUR_FITS'
    assert plan['model_count'] == 4 and [g['group'] for g in plan['groups']] == GROUPS
    assert plan['controls'] == CONTROLS and plan['targets'] == PARTS
    assert plan['fixed_effects'] == ['fund_id'] and plan['clusters'] == ['fund_id', 'company_id']
    assert not plan['time_FE'] and not plan['cluster_time'] and not plan['multiple_testing_adjustments']
    assert plan['h2_scopes'] == SCOPES and not plan['covariance_repair']
    assert plan['panel_options'] == dict(entity_effects=True, time_effects=False, singletons=True, drop_absorbed=False, check_rank=True)
    assert plan['covariance_options'] == dict(cov_type='clustered', cluster_entity=False, cluster_time=False,
        group_debias=True, debiased=True, auto_df=False, count_effects=True)
    assert sha(plan['helper_path']) == plan['helper_sha256']
    root = Path(plan['output_root'])
    if any(part.lower() == 'databaseresources' for part in root.parts): raise ValueError('Protected output path')
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'summary.json').exists() or (root / 'completion.json').exists():
        raise FileExistsError('Completed estimates may not be overwritten or repeated')
    lock = root / 'run.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, json.dumps(dict(pid=os.getpid(), plan_sha256=expected_plan_sha)).encode())
    os.close(descriptor)
    started = time.perf_counter()
    helper = load(plan['helper_path'])
    sources = {}
    def checkpoint(state, **extra):
        data = dict(state=state, elapsed_seconds=time.perf_counter() - started) | extra
        save(root / 'checkpoint.json', data)
        print(json.dumps(clean(data), ensure_ascii=True), flush=True)
    try:
        save(root / 'frozen_plan.json', plan)
        manifest = dict(plan_sha256=expected_plan_sha, runner_sha256=sha(__file__), helper_sha256=sha(plan['helper_path']),
            execution_host=socket.gethostname(), platform=platform.platform(), python=sys.version,
            started_utc=dt.datetime.now(dt.timezone.utc).isoformat(), sources=sources)
        for definition in plan['groups']:
            group = definition['group']; path = definition['input_panel']
            actual = sha(path)
            if actual != definition['expected_source_sha256']: raise AssertionError('Source SHA mismatch: ' + group)
            sources[group] = dict(path=path, sha256_before=actual, bytes=Path(path).stat().st_size)
        save(root / 'source_manifest_before.json', manifest)
        records = []; all_coefs = []; all_tests = {}; covariance = {}; all_signs = {}; samples = {}; comparison = []; testrows = []
        (root / 'samples').mkdir(exist_ok=True)
        for definition in plan['groups']:
            group = definition['group']
            checkpoint('READING_EXACT_SAMPLE', group=group)
            original = helper.ordered(pd.read_parquet(definition['input_panel']))
            actual_shape = (len(original), original.fund_id.nunique(), original.company_id.nunique(), helper.row_hash(original))
            expected_shape = tuple(definition[k] for k in ['expected_rows', 'expected_funds', 'expected_companies', 'expected_ordered_row_hash'])
            assert actual_shape == expected_shape, (group, actual_shape, expected_shape)
            assert original.row_id.is_unique
            assert np.isfinite(original[['quantity_change_pct', 'net_sentiment', 'net_sentiment_lag1'] + CONTROLS].astype(float)).all().all()
            assert original.complete_current_lag1.all()
            assert original.net_sentiment.between(-1, 1).all() and original.net_sentiment_lag1.between(-1, 1).all()
            entries = original.quantity_open.eq(0) & original.quantity_close.gt(0)
            exits = original.quantity_open.gt(0) & original.quantity_close.eq(0)
            continuing = original.quantity_open.gt(0) & original.quantity_close.gt(0)
            assert (entries | exits | continuing).all()
            assert original.loc[entries, 'quantity_change_pct'].eq(100).all()
            assert original.loc[exits, 'quantity_change_pct'].eq(-100).all()
            assert original.loc[continuing, 'quantity_change_pct'].le(plan['existing_upper_tail_rule']['cutoff_percent'] + 1e-10).all()
            signed = helper.add_signed_parts(original)
            pd.testing.assert_frame_equal(original, signed[original.columns], check_exact=True)
            all_signs[group] = {source: sign_counts(signed, source, period_id) for source, period_id in
                [('net_sentiment', 'company_period_id'), ('net_sentiment_lag1', 'lag_company_period_id')]}
            used_path = root / 'samples' / (group + '__CURRENT_PLUS_LAG1_SIX.parquet')
            if used_path.exists(): raise FileExistsError('Used sample already exists: ' + str(used_path))
            signed.to_parquet(used_path, index=False, compression='zstd')
            samples[group] = dict(path=str(used_path), sha256=sha(used_path), source_path=definition['input_panel'],
                source_sha256=sources[group]['sha256_before'], ordered_row_hash=helper.row_hash(signed), rows=len(signed),
                funds=int(signed.fund_id.nunique()), companies=int(signed.company_id.nunique()), source_columns_preserved_exactly=True,
                additional_columns=PARTS, columns=list(signed.columns), entries=int(entries.sum()), exits=int(exits.sum()),
                continuing=int(continuing.sum()), minimum_history_filter=definition['minimum_history_filter'])
            checkpoint('FITTING_FIXED_MODEL', group=group, rows=len(signed))
            fitted_start = time.perf_counter()
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                used, result, terms, dropped = helper.fit_native(signed, CONTROLS, True)
                assert terms == PARTS + CONTROLS and not dropped
                assert int(result.nobs) == len(original)
                pd.testing.assert_frame_equal(original, used[original.columns], check_exact=True)
                text = str(result.summary); native_html = result.summary.as_html()
                ci = result.conf_int()
                coefs = [dict(group=group, term=term, beta=float(result.params[term]), se_native=float(result.std_errors[term]),
                    t_native=float(result.tstats[term]), p_native=float(result.pvalues[term]),
                    ci95_native_low=float(ci.loc[term].iloc[0]), ci95_native_high=float(ci.loc[term].iloc[1]))
                    for term in result.params.index]
                tests = helper.h2_tests(result, True)
                fullcov = dict(terms=list(result.params.index), matrix=result.cov.to_numpy(), df_resid=int(result.df_resid),
                    covariance_options=plan['covariance_options'], covariance_repair=False)
            fit_seconds = time.perf_counter() - fitted_start
            (root / (group + '.summary.txt')).write_text(text, encoding='utf-8')
            (root / (group + '.summary.html')).write_text(native_html, encoding='utf-8')
            save(root / (group + '.full_covariance.json'), fullcov)
            covariance[group] = fullcov; all_coefs += coefs; all_tests[group] = tests
            record = dict(group=group, label_ru=definition['label_ru'], model=group.upper() + '__CURRENT_PLUS_LAG1_SIX',
                status='ESTIMATED', nobs=int(result.nobs), funds=int(original.fund_id.nunique()),
                companies=int(original.company_id.nunique()), securities=int(original.security_id.nunique()),
                first_start_date=str(original.start_date.min()), last_end_date=str(original.end_date.max()),
                fund_focus_counts=original.fund_focus.value_counts(dropna=False).to_dict(),
                singleton_funds=int(original.groupby('fund_id').size().eq(1).sum()),
                minimum_history_filter=definition['minimum_history_filter'], df_resid=int(result.df_resid),
                rsquared_within=float(result.rsquared_within), rsquared=float(result.rsquared),
                rsquared_between=float(result.rsquared_between), rsquared_overall=float(result.rsquared_overall),
                used_terms=list(result.params.index), dropped_terms=dropped, coefficients=coefs, h2_tests=tests,
                ordered_row_hash=helper.row_hash(original), fit_seconds=fit_seconds,
                warnings=list(dict.fromkeys(str(w.message) for w in captured)))
            records.append(record)
            item = {key: record[key] for key in ['group', 'nobs', 'funds', 'companies', 'df_resid', 'rsquared_within']}
            for scope in SCOPES:
                test = tests['tests'][scope]
                item.update({scope + '_' + key: test[key] for key in ['beta_positive', 'beta_negative', 'p_abs', 'expected_directional_signs']})
                testrows.append(dict(group=group, scope=scope, status=test['status'], beta_positive=test['beta_positive'],
                    positive_se=test['positive_coefficient']['se_native'], positive_p=test['positive_coefficient']['p_native'],
                    beta_negative=test['beta_negative'], negative_se=test['negative_coefficient']['se_native'],
                    negative_p=test['negative_coefficient']['p_native'], p_difference=test['difference_zero']['p_native'],
                    p_sum=test['sum_zero']['p_native'], p_abs=test['p_abs'],
                    expected_directional_signs=test['expected_directional_signs'],
                    negative_absolute_minus_positive_absolute=test['negative_absolute_minus_positive_absolute']))
            comparison.append(item)
            save(root / (group + '.result.json'), record)
            checkpoint('MODEL_SAVED', group=group, nobs=int(result.nobs), fit_seconds=fit_seconds,
                p_abs={scope: tests['tests'][scope]['p_abs'] for scope in SCOPES})
        assert len(records) == plan['model_count']
        for group, source in sources.items():
            source['sha256_after'] = sha(source['path'])
            source['unchanged'] = source['sha256_before'] == source['sha256_after']
            assert source['unchanged'], 'Input changed: ' + group
        assert sha(plan_path) == expected_plan_sha and sha(plan['helper_path']) == manifest['helper_sha256']
        assert sha(__file__) == manifest['runner_sha256']
        manifest['all_sources_unchanged'] = True
        manifest['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        save(root / 'source_manifest.json', manifest)
        save(root / 'samples_manifest.json', samples); save(root / 'sign_counts.json', all_signs)
        save(root / 'full_covariance.json', covariance); save(root / 'h2_tests.json', all_tests)
        coefficients = pd.DataFrame(all_coefs); comparisons = pd.DataFrame(comparison); tests_frame = pd.DataFrame(testrows)
        coefficients.to_csv(root / 'coefficients.csv', index=False, encoding='utf-8-sig')
        comparisons.to_csv(root / 'summary.csv', index=False, encoding='utf-8-sig')
        tests_frame.to_csv(root / 'h2_tests.csv', index=False, encoding='utf-8-sig')
        summary = dict(status='COMPLETE_PENDING_INDEPENDENT_QA', run_id=plan['run_id'], specification=plan,
            models=records, samples=samples, source_manifest=manifest, sign_counts=all_signs,
            unique_native_fits=4, elapsed_seconds=time.perf_counter() - started,
            total_fit_seconds=sum(r['fit_seconds'] for r in records), independent_QA='PENDING_SEPARATE_REVIEW',
            no_selection_by_significance=True, multiple_testing_adjustments=False,
            h2_iut_is_not_multiple_testing_correction=True, source_inputs_unchanged=True)
        save(root / 'summary.json', summary)
        render(root, summary, comparisons, coefficients, tests_frame)
        completion = dict(status='COMPLETE_PENDING_INDEPENDENT_QA', unique_native_fits=4,
            elapsed_seconds=time.perf_counter() - started, total_fit_seconds=summary['total_fit_seconds'],
            plan_sha256=expected_plan_sha, runner_sha256=manifest['runner_sha256'],
            completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        save(root / 'completion.json', completion)
        checkpoint('COMPLETE', **completion)
        save(root / 'output_manifest.json', {str(path.relative_to(root)): dict(bytes=path.stat().st_size, sha256=sha(path))
            for path in root.rglob('*') if path.is_file() and path.name not in ['run.lock', 'output_manifest.json']})
        return 0
    except Exception as error:
        save(root / 'failure.json', dict(status='FAILED', error=str(error), traceback=traceback.format_exc(),
            plan_sha256=expected_plan_sha, elapsed_seconds=time.perf_counter() - started))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha', required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.plan, args.plan_sha))
