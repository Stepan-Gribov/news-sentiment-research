"""Frozen H1 sensitivity: certified zero entries coded +100 and upper-P99 trim.

Actual research execution is Windows-only. --fixture-only uses synthetic rows.
Earlier sources, helpers, samples and estimates are never modified.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd


MICRO = ['opening_weight', 'closing_log_NAV', 'closing_HHI',
         'log_news_count_interval', 'log_interval_days', 'bond_indicator']
MACRO = ['Imoex', 'imoex_ret', 'Ruonia', 'USD_RUB', 'RVI',
         'Flow_billion_RUB', 'unique_funds', 'dummy_mar2014',
         'dummy_feb2022', 'dummy_sep2018']
KEYS = ['row_id', 'fund_id', 'security_id', 'company_id', 'interval_id',
        'company_period_id', 'opening_report_id', 'closing_report_id',
        'start_date', 'end_date']
FLAGS = ['target_security_candidate', 'issuer_identity_conflict',
         'asset_class_conflict', 'opening_position_observed',
         'closing_position_present', 'closing_position_reported',
         'opening_full_portfolio', 'closing_full_portfolio',
         'known_opening_absence_zero', 'known_closing_absence_zero',
         'unknown_opening_absence', 'unknown_closing_absence', 'certified_exit',
         'opening_passive_explicit', 'closing_passive_explicit',
         'eligible_primary', 'complete_cases', 'complete_NET_CONTROLS',
         'exclusion_reason', 'reason_NET_CONTROLS', 'w2_static_link_used']
DETAIL = ['fund_name', 'asset_name', 'isin', 'price_isin', 'company', 'asset_type',
          'quantity_open', 'quantity_close', 'quantity_change_pct',
          'closing_weight', 'delta_quantity_units', 'delta_share_pp',
          'net_sentiment', 'n_relevant', 'n_candidates', 'n_valid',
          'n_positive', 'n_negative', 'n_neutral', 'state',
          'n_errors_or_unattempted', 'n_undated_company_candidates',
          'source_path', 'open__quantity_est', 'close__quantity_est',
          'open__is_countable', 'close__is_countable']
DERIVED = ['original_quantity_change_pct', 'entry_zero_to_positive',
           'entry_opening_zero_confirmed', 'complete_exit',
           'positive_base_eligible', 'augmented_eligible',
           'exclusion_reason_augmented', 'trimmed_upper_p99',
           'retained_primary', 'monthly_interval', 'interval_days']
PLANNED = ['REFERENCE_OLD', 'OLD_BASE_AFTER_P99', 'AUGMENTED_UNTRIMMED',
           'AUGMENTED_P99', 'MONTHLY_AUGMENTED_P99', 'FOCUS_EQUITY_P99',
           'FOCUS_BOND_P99', 'FOCUS_MIXED_P99']


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def truth(series):
    return series.eq(True).fillna(False).astype(bool)


def finite(series):
    return pd.Series(np.isfinite(pd.to_numeric(series, errors='coerce')),
                     index=series.index)


def row_hash(frame):
    return hashlib.sha256('\n'.join(frame.row_id.astype(str)).encode()).hexdigest()


def ordered(frame):
    return frame.sort_values(['fund_id', 'end_date', 'security_id', 'interval_id'],
                            kind='stable').reset_index(drop=True)


def counts(frame):
    return dict(rows=len(frame), funds=int(frame.fund_id.nunique()),
                companies=int(frame.company_id.nunique()),
                entries=int(frame.entry_zero_to_positive.sum()) if 'entry_zero_to_positive' in frame else None,
                exits=int(frame.complete_exit.sum()) if 'complete_exit' in frame else None,
                first_start_date=str(frame.start_date.min()) if len(frame) else None,
                last_end_date=str(frame.end_date.max()) if len(frame) else None)


def prepare(frame, controls, index_unknown_policy='retain_flagged'):
    """Recreate every former eligibility gate, changing only the zero-entry Y."""
    required = set(KEYS) | set(controls) | {
        'quantity_open', 'quantity_close', 'quantity_change_pct', 'net_sentiment',
        'target_security_candidate', 'issuer_identity_conflict',
        'asset_class_conflict', 'unknown_opening_absence',
        'unknown_closing_absence', 'eligible_primary',
        'opening_position_observed', 'known_opening_absence_zero',
        'opening_full_portfolio', 'index_status', 'exclude_index_fund', 'fund_focus'}
    if required - set(frame):
        raise ValueError('Missing candidate fields: ' + str(sorted(required - set(frame))))
    d = frame.copy()
    if d.row_id.isna().any() or d.row_id.duplicated().any():
        raise ValueError('Candidate row identities missing or duplicated')
    for column in ['start_date', 'end_date']:
        d[column] = pd.to_datetime(d[column], errors='raise')
    q0 = pd.to_numeric(d.quantity_open, errors='coerce')
    q1 = pd.to_numeric(d.quantity_close, errors='coerce')
    d['original_quantity_change_pct'] = pd.to_numeric(d.quantity_change_pct, errors='coerce')
    valid = finite(q0) & finite(q1) & q0.ge(0) & q1.ge(0)
    positive_base = valid & q0.gt(0)
    certified_zero = (truth(d.opening_position_observed)
                      | (truth(d.known_opening_absence_zero) & truth(d.opening_full_portfolio)))
    d['entry_opening_zero_confirmed'] = valid & q0.eq(0) & certified_zero
    d['entry_zero_to_positive'] = d.entry_opening_zero_confirmed & q1.gt(0)
    d['complete_exit'] = positive_base & q1.eq(0)
    calculated = pd.Series(np.nan, index=d.index, dtype=float)
    calculated.loc[positive_base] = 100 * (q1.loc[positive_base] / q0.loc[positive_base] - 1)
    # On positive-base observations, preserve the stored outcome after checking
    # its arithmetic; do not regenerate a slightly different legacy float.
    check = positive_base & finite(d.original_quantity_change_pct)
    if not np.isclose(calculated.loc[check], d.loc[check, 'original_quantity_change_pct'],
                      rtol=1e-10, atol=1e-8).all():
        raise AssertionError('Stored positive-base Y disagrees with Q0/Q1')
    d['quantity_change_pct'] = d.original_quantity_change_pct.where(positive_base)
    d.loc[d.entry_zero_to_positive, 'quantity_change_pct'] = 100.0
    reason = pd.Series('', index=d.index, dtype='string')
    def reject(mask, label):
        nonlocal reason
        reason = reason.mask(mask.fillna(True), reason + label + ';')
    reject(~truth(d.target_security_candidate), 'not_target_security')
    reject(d.company_id.isna(), 'company_link_missing')
    reject(truth(d.issuer_identity_conflict), 'issuer_identity_conflict')
    reject(truth(d.asset_class_conflict), 'asset_class_conflict')
    reject(truth(d.unknown_opening_absence), 'unknown_opening_absence')
    reject(truth(d.unknown_closing_absence), 'unknown_closing_absence')
    reject(~truth(d.eligible_primary), 'sentiment_not_primary')
    reject(~finite(d.net_sentiment), 'missing_net_sentiment')
    reject(~valid, 'invalid_or_unknown_quantities')
    reject(valid & q0.eq(0) & q1.eq(0), 'zero_to_zero_no_relative_outcome')
    reject(valid & q0.eq(0) & q1.gt(0) & ~certified_zero, 'opening_zero_not_certified')
    reject(~finite(d.quantity_change_pct), 'missing_or_nonfinite_outcome')
    reject(d.index_status.isna() | d.exclude_index_fund.isna(), 'fund_classification_missing')
    reject(d.index_status.eq('index') | truth(d.exclude_index_fund), 'identified_index_fund')
    if index_unknown_policy == 'exclude':
        reject(~d.index_status.eq('nonindex_verified'), 'not_verified_nonindex')
    elif index_unknown_policy != 'retain_flagged':
        raise ValueError('Unrecognized index policy')
    for control in controls:
        reject(~finite(d[control]), 'missing_control_' + control)
    d['exclusion_reason_augmented'] = reason.str.rstrip(';')
    d['augmented_eligible'] = reason.eq('')
    d['positive_base_eligible'] = d.augmented_eligible & positive_base
    d['interval_days'] = (d.end_date - d.start_date).dt.days
    monthdiff = (d.end_date.dt.year-d.start_date.dt.year)*12+d.end_date.dt.month-d.start_date.dt.month
    d['monthly_interval'] = monthdiff.eq(1) & d.interval_days.between(20, 40, inclusive='both')
    if (d.loc[d.augmented_eligible, 'quantity_change_pct'] < -100-1e-8).any():
        raise AssertionError('Eligible outcome below valid complete exit -100')
    return d


def threshold_and_masks(d, upper_quantile=.99):
    q0 = pd.to_numeric(d.quantity_open, errors='coerce')
    q1 = pd.to_numeric(d.quantity_close, errors='coerce')
    population = ordered(d.loc[d.positive_base_eligible & q0.gt(0) & q1.gt(0)])
    if population.empty:
        raise ValueError('No eligible continuing positions for the frozen percentile')
    cutoff = float(population.quantity_change_pct.quantile(upper_quantile, interpolation='linear'))
    if not np.isfinite(cutoff):
        raise ValueError('Nonfinite P99 cutoff')
    d['trimmed_upper_p99'] = (d.positive_base_eligible & q0.gt(0) & q1.gt(0)
                              & d.quantity_change_pct.gt(cutoff))
    d['retained_primary'] = d.augmented_eligible & ~d.trimmed_upper_p99
    assert not (d.trimmed_upper_p99 & (d.entry_zero_to_positive | d.complete_exit)).any()
    assert (d.retained_primary | d.trimmed_upper_p99).equals(d.augmented_eligible)
    threshold = dict(upper_quantile=float(upper_quantile), quantile_interpolation='linear',
        cutoff_percent=cutoff, cutoff_population_rows=len(population),
        cutoff_population_row_hash=row_hash(population),
        cutoff_population_description='Eligible old main nonindex macro-complete continuing Q0>0,Q1>0; before conditional entries',
        cutoff_population_quantiles={str(q):float(population.quantity_change_pct.quantile(q, interpolation='linear'))
                                     for q in [0, .01, .25, .5, .75, .9, .95, .99, .999, 1]},
        trim_rule='Exclude continuing Y strictly > common cutoff; ties retained',
        entries_always_retained=True, exits_always_retained=True,
        threshold_recomputed_per_model_or_group=False, lower_tail_trim=False,
        winsorization=False, excluded_rows=int(d.trimmed_upper_p99.sum()),
        entry_value_percent=100,
        interpretation='Exploratory user-requested sensitivity; extremes are not proven errors. Conditional +100 entry codes are not literal percentage growth from zero.')
    return d, threshold


def holm_adjust(coefficients, groups):
    family = ['FOCUS_' + group.upper() + '_P99' for group in groups]
    items = []
    for name in family:
        found = [r for r in coefficients if r['model'] == name and r['term'] == 'net_sentiment']
        p = found[0]['p_native'] if found else None
        valid = p is not None and np.isfinite(p) and 0 <= p <= 1
        items.append((name, float(p) if valid else 1.0, valid))
    largest = 0.; adjusted = {}
    for i, (name, p, valid) in enumerate(sorted(items, key=lambda x:x[1])):
        largest = max(largest, min(1., (len(items)-i)*p))
        adjusted[name] = largest if valid else None
    for r in coefficients:
        r['p_native_holm_focus_family'] = adjusted.get(r['model']) if r['term'] == 'net_sentiment' else None
    return dict(method='Holm step-down; unestimated tests retain p=1 slots',
                planned_family=family, adjusted_pvalues=adjusted)


def export(frame, root, name, controls):
    columns = list(dict.fromkeys(c for c in KEYS+DETAIL+FLAGS+DERIVED+controls+list(frame.columns)
                 if c in frame and (c in KEYS+DETAIL+FLAGS+DERIVED+controls
                     or c.startswith(('index_', 'fund_focus', 'opening_', 'closing_', 'Flow_', 'Imoex_', 'imoex_', 'Ruonia_', 'RVI_', 'USD_RUB_'))
                     or c == 'exclude_index_fund')))
    stem = root / 'samples' / name
    stem.parent.mkdir(exist_ok=True)
    frame[columns].to_parquet(stem.with_suffix('.parquet'), index=False, compression='zstd')
    frame[columns].to_csv(stem.with_suffix('.csv.gz'), index=False, encoding='utf-8-sig', compression='gzip')
    return dict(parquet=str(stem.with_suffix('.parquet')),csv_gz=str(stem.with_suffix('.csv.gz')),
                rows=len(frame),row_hash=row_hash(frame),exported_columns=columns)


def render(root, summary, comparison, native):
    threshold = summary['threshold']['cutoff_percent']
    parts = ['<!doctype html><html lang="ru"><meta charset="utf-8"><title>H1: входы и P99</title>',
      '<style>body{font:16px system-ui;max-width:1350px;margin:32px auto;padding:0 20px;color:#17212d}table{border-collapse:collapse;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}pre{padding:16px;background:#f5f6f8;overflow:auto;font:13px monospace}a{color:#075ba5}</style>',
      '<h1>H1: новые позиции и отсечение верхнего 1%</h1>',
      '<p>Для Q₀&gt;0: Y=100×(Q₁/Q₀−1). Подтверждённый вход Q₀=0,Q₁&gt;0 кодируется условными +100%; это не буквальный процент роста из нуля. Полные выходы сохраняют −100%. Неизвестное отсутствие не заменяется нулём, отсутствующий sentiment остаётся NULL.</p>',
      f'<p>Единый P99={threshold:.8g}% рассчитан один раз по допустимым продолжающимся позициям до добавления входов. Удалены только значения строго выше порога; равные порогу сохранены. Новые входы и полные выходы не отсекаются. Это исследовательская проверка устойчивости после предыдущего результата; экстремумы не объявлены ошибками.</p>',
      '<p>Фиксированные эффекты фондов; ошибки кластеризованы по фонду и компании. Временных эффектов и кластеров по времени нет. Те же 6 микроконтролей и 10 макроконтролей; индикатор входа не добавлялся. Выявленные индексные фонды исключены; остальные статусы сохранены явно. Месячная проверка требует соседних календарных месяцев и 20–40 дней.</p>',
      comparison.to_html(index=False,border=0,float_format=lambda x:f'{x:.7g}'),
      '<p>Сравнение OLD_BASE_AFTER_P99 с AUGMENTED_P99 показывает влияние включения условных входов при одном пороге. Отдельная значимость группы не доказывает различий между группами; для трёх групп дана поправка Holm. Группы фондов могут пересекаться по времени.</p>',
      '<p>Отрицательная дисперсия отдельного контроля или отрицательный robust F означают неопределённость соответствующего вывода, а не p=1. Исходный Python-вывод сохранён; отрицательная ковариация не исправлялась.</p>',
      '<p><a href="summary.json">Все метрики JSON</a> · <a href="coefficients.csv">Все коэффициенты</a> · <a href="threshold.json">Порог</a> · <a href="sample_flow.json">Отбор</a> · <a href="row_decisions.csv.gz">Все строки и причины</a> · <a href="excluded_upper_p99.csv.gz">Исключённые экстремумы</a> · <a href="regression_preview.csv">Пример использованных данных</a> · <a href="source_manifest.json">Источники</a></p>']
    for name,text in native.items():
        parts.append('<h2>'+html.escape(name)+'</h2><pre>'+html.escape(text)+'</pre>')
    parts.append('</html>')
    (root/'report.html').write_text('\n'.join(parts),encoding='utf-8')


def execute(args):
    if sys.platform != 'win32':
        raise RuntimeError('Actual research processing must run on research-host; use --fixture-only locally')
    plan_path = Path(args.plan)
    if sha(plan_path) != args.plan_sha:
        raise ValueError('Frozen plan SHA256 mismatch')
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if plan['planned_models'] != PLANNED:
        raise ValueError('Unexpected model suite')
    if plan['entry_value_percent'] != 100 or plan['upper_quantile'] != .99 or plan['quantile_interpolation'] != 'linear':
        raise ValueError('Outcome/cutoff convention differs from frozen implementation')
    if plan.get('entry_flag_as_control') or plan.get('time_FE') or plan.get('cluster_time'):
        raise ValueError('Entry controls/time effects/time clusters are not authorized by this plan')
    controls = plan['baseline_controls'] + plan['macro_controls']
    if controls != MICRO + MACRO or plan.get('clusters') != ['fund_id','company_id']:
        raise ValueError('Frozen control/cluster set changed')
    root = Path(plan['output_root'])
    if any(part.lower() == 'databaseresources' for part in root.parts):
        raise ValueError('Protected input root cannot contain output')
    root.mkdir(parents=True,exist_ok=True)
    if (root/'completion.json').exists():
        raise FileExistsError('Completed output version may not be overwritten')
    base_path = args.base_helper or plan['base_helper']
    macro_path = args.macro_runner_helper or plan['macro_runner_helper']
    base = load_module(base_path,'h1_entries_base')
    macro = load_module(macro_path,'h1_entries_macro_fit')
    paths = {key:plan[key] for key in ['all_candidate_panel','old_primary_sample',
             'macro_controls_source','classification_source','news_company_source']}
    paths.update(base_helper=base_path,macro_runner_helper=macro_path)
    before = {}
    for key,path in paths.items():
        actual = sha(path)
        if actual != plan['expected_source_sha256'][key]:
            raise ValueError('Source hash mismatch: '+key)
        before[key] = dict(path=path,sha256=actual,size_bytes=Path(path).stat().st_size)
    lock = root/'run.lock'
    descriptor = os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(descriptor,json.dumps(dict(pid=os.getpid(),plan_sha256=args.plan_sha)).encode())
    os.close(descriptor)
    started = time.time()
    def save(name,value): base.save(root/name,value)
    def checkpoint(state,**extra):
        record=dict(state=state,pid=os.getpid(),elapsed_seconds=time.time()-started,**extra)
        save('checkpoint.json',record);print(json.dumps(record),flush=True)
    try:
        save('frozen_plan.json',plan)
        checkpoint('READING_INPUTS')
        wanted=list(dict.fromkeys(KEYS+DETAIL+FLAGS+MICRO))
        d=macro.read_frame(paths['all_candidate_panel'],wanted)
        classification=macro.read_frame(paths['classification_source'])
        if classification.duplicated(['interval_id','fund_id']).any():
            raise ValueError('Duplicate classification interval/fund keys')
        mapcols=[c for c in classification if c in ['interval_id','fund_id'] or c not in d]
        d=d.merge(classification[mapcols],on=['interval_id','fund_id'],how='left',validate='many_to_one',sort=False)
        macrotable=macro.read_frame(paths['macro_controls_source'])
        if macrotable.interval_id.duplicated().any():
            raise ValueError('Duplicate macro interval keys')
        maccols=[c for c in macrotable if c=='interval_id' or c not in d]
        d=d.merge(macrotable[maccols],on='interval_id',how='left',validate='many_to_one',sort=False)
        del classification,macrotable
        checkpoint('RECONSTRUCTING_ELIGIBILITY',candidate_rows=len(d))
        d=prepare(d,controls,plan['index_unknown_policy'])
        old=ordered(macro.read_frame(paths['old_primary_sample']))
        prior=ordered(d.loc[d.positive_base_eligible])
        if len(old)!=plan['expected_old_primary_rows'] or not np.array_equal(prior.row_id.astype(str),old.row_id.astype(str)):
            added=set(prior.row_id)-set(old.row_id);missing=set(old.row_id)-set(prior.row_id)
            raise AssertionError(f'Positive-base universe changed: prior={len(prior)} old={len(old)} added={len(added)} missing={len(missing)}')
        for column in ['quantity_open','quantity_close','quantity_change_pct','net_sentiment']+controls:
            if not np.allclose(prior[column].astype(float),old[column].astype(float),rtol=1e-12,atol=1e-10,equal_nan=True):
                raise AssertionError('Legacy values changed for '+column)
        d,threshold=threshold_and_masks(d,plan['upper_quantile'])
        if threshold['cutoff_population_rows']!=204152 or not np.isclose(threshold['cutoff_percent'],322.7771642245575,rtol=1e-12,atol=1e-10):
            raise AssertionError('Cutoff differs from the independent frozen-source prefit calculation')
        if int(d.trimmed_upper_p99.sum())!=2042 or int((d.positive_base_eligible & d.complete_exit).sum())!=9460:
            raise AssertionError('Continuing-tail/exit counts differ from independent prefit calculation')
        save('threshold.json',threshold)
        # All candidate decisions remain inspectable, including all failed gates.
        d.to_parquet(root/'row_decisions.parquet',index=False,compression='zstd')
        d.to_csv(root/'row_decisions.csv.gz',index=False,encoding='utf-8-sig',compression='gzip')
        excluded=ordered(d.loc[d.trimmed_upper_p99])
        excluded.to_csv(root/'excluded_upper_p99.csv.gz',index=False,encoding='utf-8-sig',compression='gzip')
        primary=ordered(d.loc[d.retained_primary])
        old_clean=ordered(d.loc[d.positive_base_eligible & ~d.trimmed_upper_p99])
        augmented=ordered(d.loc[d.augmented_eligible])
        old_aug=ordered(d.loc[d.positive_base_eligible])
        assert set(primary.row_id)==set(old_clean.row_id)|set(d.loc[d.augmented_eligible & d.entry_zero_to_positive,'row_id'])
        assert set(old_clean.row_id).isdisjoint(set(excluded.row_id))
        specs=[('REFERENCE_OLD',old_aug),('OLD_BASE_AFTER_P99',old_clean),
               ('AUGMENTED_UNTRIMMED',augmented),('AUGMENTED_P99',primary),
               ('MONTHLY_AUGMENTED_P99',primary.loc[primary.monthly_interval])]
        specs.extend(('FOCUS_'+group.upper()+'_P99',primary.loc[primary.fund_focus.eq(group)]) for group in plan['fund_focus_groups'])
        flow=dict(all_candidates=counts(d),reconstructed_old_main=counts(old_aug),
            augmented_untrimmed=counts(augmented),primary=counts(primary),
            trimmed=counts(excluded),old_base_after_p99=counts(old_clean),
            entries_all_candidates=int(d.entry_zero_to_positive.sum()),
            eligible_entries=int((d.entry_zero_to_positive & d.augmented_eligible).sum()),
            excluded_candidate_reason_combinations=d.loc[~d.augmented_eligible,'exclusion_reason_augmented'].value_counts().to_dict(),
            index_status_all=d.index_status.value_counts(dropna=False).to_dict(),
            index_status_primary=primary.index_status.value_counts(dropna=False).to_dict(),
            focus_primary=primary.fund_focus.value_counts(dropna=False).to_dict(),
            old_row_ids_exactly_reproduced=True,old_values_unchanged=True,
            source_old_sample_row_hash=row_hash(old),reconstructed_old_sample_row_hash=row_hash(old_aug),
            unknown_absence_remains_missing=True,missing_news_not_imputed=True,
            original_outcome_preserved_column='original_quantity_change_pct',
            conditional_entries_are_not_literal_percent_growth=True,
            index_unknown_policy=plan['index_unknown_policy'])
        save('sample_flow.json',flow)
        del d,old,prior
        rows=[];models=[];samples={};native={}
        for name,sample in specs:
            sample=ordered(sample)
            checkpoint('FITTING_MODEL',model=name,**counts(sample))
            samples[name]=export(sample,root,name,controls)
            problem=macro.model_status(sample,plan['minimum_model_rows'])
            if problem:
                models.append(dict(model=name,status='NOT_ESTIMABLE',reason=problem,
                                   nobs=len(sample),funds=int(sample.fund_id.nunique()),companies=int(sample.company_id.nunique())))
                continue
            try:
                fitted,result,terms,dropped=macro.fit(sample,controls,base)
                record,coefficients=base.describe_result(fitted,result,name,'quantity_change_pct',terms,dropped)
                record['covariance_options']['cluster_time']=False
                record.update(entry_dummy_control=False,common_cutoff_percent=threshold['cutoff_percent'],
                              conditional_entry_rows=int(sample.entry_zero_to_positive.sum()),
                              complete_exit_rows=int(sample.complete_exit.sum()))
                robustf=result.f_statistic_robust
                fvalue=float(robustf.stat);fp=float(robustf.pval)
                record['joint_robust_F_inference']=dict(statistic=fvalue,native_p=fp,
                    status='VALID' if np.isfinite(fvalue) and fvalue>=0 and np.isfinite(fp) else 'UNDEFINED_COVARIANCE',
                    negative_or_nonfinite_is_not_valid_p1=True)
                if name=='REFERENCE_OLD' and plan.get('verify_old_reference',True):
                    if not np.isclose(result.params['net_sentiment'],3720.505450722427,rtol=1e-7,atol=1e-5):
                        raise AssertionError('Old main coefficient did not reproduce')
                    if not np.isclose(result.std_errors['net_sentiment'],3467.319967071551,rtol=1e-7,atol=1e-5):
                        raise AssertionError('Old main covariance did not reproduce')
                    if not np.isclose(result.pvalues['net_sentiment'],.2832631281146265,rtol=1e-7,atol=1e-8):
                        raise AssertionError('Old main p-value did not reproduce')
            except Exception as error:
                if name=='REFERENCE_OLD': raise
                models.append(dict(model=name,status='NOT_ESTIMABLE',reason=f'{type(error).__name__}: {error}',
                                   nobs=len(sample),funds=int(sample.fund_id.nunique()),companies=int(sample.company_id.nunique())))
                continue
            text=str(result.summary)
            (root/(name+'.summary.txt')).write_text(text,encoding='utf-8')
            (root/(name+'.summary.html')).write_text(result.summary.as_html(),encoding='utf-8')
            native[name]=text;models.append(record);rows.extend(coefficients)
        adjustment=holm_adjust(rows,plan['holm_focus_groups'])
        coef=pd.DataFrame(rows)
        coef.to_csv(root/'coefficients.csv',index=False,encoding='utf-8-sig')
        comparison=[]
        for record in models:
            item={key:record.get(key) for key in ['model','status','nobs','funds','companies','rsquared','rsquared_within','conditional_entry_rows','complete_exit_rows']}
            match=next((r for r in rows if r['model']==record['model'] and r['term']=='net_sentiment'),{})
            item.update({key:match.get(key) for key in ['beta','se_native','ci95_native_low','ci95_native_high','p_native','p_native_holm_focus_family']})
            comparison.append(item)
        comparison=pd.DataFrame(comparison)
        comparison.to_csv(root/'comparison.csv',index=False,encoding='utf-8-sig')
        primary.head(300).to_csv(root/'regression_preview.csv',index=False,encoding='utf-8-sig')
        unchanged={key:sha(path)==before[key]['sha256'] for key,path in paths.items()}
        if not all(unchanged.values()): raise AssertionError('A source changed during execution')
        save('source_manifest.json',dict(sources=before,source_hashes_unchanged_after_run=unchanged,
                                       plan_sha256=args.plan_sha,runner_sha256=sha(__file__)))
        save('data_dictionary.json',dict(outcome='quantity_change_pct',
            original_quantity_change_pct='Original stored outcome; NA for zero-opening entries',
            quantity_change_pct='Original percent growth for positive Q0; conditional +100 for certified positive entry',
            entry_zero_to_positive='Q0=0,Q1>0 and explicit opening observation or certified full-report absence; eligibility still separately required',
            entry_opening_zero_confirmed='Finite nonnegative quantities, Q0=0 and certification; may include zero-to-zero excluded rows',
            complete_exit='Finite Q0>0,Q1=0; eligibility separately required',
            positive_base_eligible='All former identity/target/news/quantity/control/index/macro gates with Q0>0',
            augmented_eligible='Same gates plus conditional entries',
            trimmed_upper_p99='Eligible continuing Q0>0,Q1>0 with original Y strictly above the fixed cutoff',
            retained_primary='augmented_eligible and not trimmed_upper_p99',
            monthly_interval='One calendar-month difference and 20–40 days inclusive',
            regression_controls=controls,rows='Fund × security × actual report interval',
            net_sentiment='Unsplit raw net sentiment for exact company/report window; missing not imputed',
            macro_controls='Frozen previous source values with endpoint/flow conventions; no real-time-vintage availability claim'))
        ok=any(m['model']==plan['primary_model'] and m['status']=='ESTIMATED' for m in models)
        summary=dict(status='COMPLETE' if ok else 'PRIMARY_NOT_ESTIMABLE',specification=plan,
            plan_sha256=args.plan_sha,models=models,samples=samples,threshold=threshold,
            sample_flow=flow,holm_focus_adjustment=adjustment,elapsed_seconds=time.time()-started,
            interpretation=plan['interpretation'],independent_QA='PENDING_SEPARATE_REVIEW')
        save('summary.json',summary)
        render(root,summary,comparison,native)
        checkpoint(summary['status'],models=len(models))
        save('completion.json',dict(status=summary['status'],plan_sha256=args.plan_sha,
             elapsed_seconds=time.time()-started,finished_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        return 0 if ok else 2
    except Exception as error:
        save('failure.json',dict(status='FAILED',error_type=type(error).__name__,message=str(error),
             traceback=traceback.format_exc(),plan_sha256=args.plan_sha))
        raise
    finally:
        lock.unlink(missing_ok=True)


def fixture(args):
    records=[]
    scenarios=[('continue',10.,12.,True,False,True,False,False),
        ('exit',10.,0.,True,False,True,False,False),
        ('entry_observed',0.,5.,True,False,False,False,False),
        ('entry_absence',0.,5.,False,True,True,False,False),
        ('unknown_open',0.,5.,False,False,False,True,False),
        ('zero_zero',0.,0.,True,False,True,False,False),
        ('unknown_close',10.,0.,True,False,True,False,True),
        ('uncertified_zero',0.,5.,False,False,False,False,False),
        ('outlier',1.,100000.,True,False,True,False,False)]
    for fund in range(4):
        for j,(name,q0,q1,observed,known,full,uo,uc) in enumerate(scenarios):
            rec=dict(row_id=f'{fund}-{name}',fund_id=str(fund),security_id=name,
                company_id=str(j%3),interval_id=f'i{fund}',company_period_id=f'cp{fund}',
                opening_report_id='r0',closing_report_id='r1',start_date=pd.Timestamp('2020-01-31'),
                end_date=pd.Timestamp('2020-02-29'),quantity_open=q0,quantity_close=q1,
                quantity_change_pct=100*(q1/q0-1) if q0>0 else np.nan,
                opening_position_observed=observed,known_opening_absence_zero=known,
                opening_full_portfolio=full,unknown_opening_absence=uo,unknown_closing_absence=uc,
                target_security_candidate=True,issuer_identity_conflict=False,asset_class_conflict=False,
                eligible_primary=True,net_sentiment=.2,index_status='source_classified_nonindex',
                exclude_index_fund=False,fund_focus='equity',control=.4)
            records.append(rec)
    d=prepare(pd.DataFrame(records),['control'])
    assert d.positive_base_eligible.sum()==12 and d.augmented_eligible.sum()==20
    assert not d.loc[d.security_id.isin(['unknown_open','unknown_close','zero_zero','uncertified_zero']),'augmented_eligible'].any()
    assert d.loc[d.augmented_eligible & d.entry_zero_to_positive,'quantity_change_pct'].eq(100).all()
    # A single extreme ensures P99 trims strictly above, while ties are retained.
    extreme=d.index[d.security_id.eq('outlier')][-1]
    d.loc[extreme,'quantity_change_pct']=2e8
    d.loc[extreme,'quantity_close']=2000001.
    d,cut=threshold_and_masks(d)
    assert d.trimmed_upper_p99.sum()==1
    assert d.loc[d.complete_exit & d.augmented_eligible,'retained_primary'].all()
    assert d.loc[d.entry_zero_to_positive & d.augmented_eligible,'retained_primary'].all()
    assert d.monthly_interval.all()
    tied=d.copy();tied.loc[tied.positive_base_eligible & ~tied.complete_exit,'quantity_change_pct']=100.
    tied,threshold=threshold_and_masks(tied)
    assert threshold['cutoff_percent']==100 and not tied.trimmed_upper_p99.any()
    altered=d.copy();altered.loc[0,'quantity_change_pct']=900
    try: prepare(altered,['control'])
    except AssertionError: pass
    else: raise AssertionError('Arithmetic corruption was not detected')
    monthly=d.copy();monthly.loc[0,'end_date']=pd.Timestamp('2020-04-30')
    monthly.loc[1,'end_date']=pd.Timestamp('2020-03-01')
    monthly=prepare(monthly.drop(columns=DERIVED,errors='ignore'),['control'])
    assert not monthly.loc[0,'monthly_interval'] and not monthly.loc[1,'monthly_interval']
    rows=[dict(model='FOCUS_EQUITY_P99',term='net_sentiment',p_native=.01),
          dict(model='FOCUS_BOND_P99',term='net_sentiment',p_native=.03)]
    h=holm_adjust(rows,['equity','bond','mixed'])
    assert np.isclose(h['adjusted_pvalues']['FOCUS_EQUITY_P99'],.03)
    assert np.isclose(h['adjusted_pvalues']['FOCUS_BOND_P99'],.06)
    base=load_module(args.base_helper or Path(__file__).with_name('run_h1_net_fund_fe_20261030.py'),'h1_entries_fixture_base')
    helper=load_module(args.macro_runner_helper or Path(__file__).with_name('run_h1_macro_no_time_20261031.py'),'h1_entries_fixture_macro')
    rng=np.random.default_rng(20261102)
    rows=[]
    for fund in range(8):
        for month in range(6):
            for company in range(4):
                net,control,noise=rng.normal(size=3)
                rows.append(dict(fund_id=str(fund),company_id=str(company),security_id=str(company),
                    interval_id=f'i{fund}-{month}',row_id=f'r{fund}-{month}-{company}',
                    start_date=pd.Timestamp('2020-01-31')+pd.offsets.MonthEnd(month),
                    end_date=pd.Timestamp('2020-01-31')+pd.offsets.MonthEnd(month+1),
                    net_sentiment=net,control=control,absorbed=float(fund),
                    quantity_change_pct=2.5*net+.4*control+.6*fund+noise,
                    opening_passive_explicit=False,closing_passive_explicit=False,closing_full_portfolio=True))
    synthetic,result,terms,dropped=helper.fit(pd.DataFrame(rows),['control','absorbed'],base)
    within=synthetic[['quantity_change_pct']+terms]-synthetic[['quantity_change_pct']+terms].groupby(synthetic.fund_id).transform('mean')
    independent=np.linalg.lstsq(within[terms],within.quantity_change_pct,rcond=None)[0]
    assert np.allclose(result.params[terms],independent,rtol=1e-10,atol=1e-10)
    independent_cov=base.cluster_covariance_fixture(synthetic,within[terms].to_numpy(),result.resids.to_numpy(),result.df_resid)
    assert np.allclose(result.cov.loc[terms,terms],independent_cov,rtol=1e-10,atol=1e-10)
    record,coefs=base.describe_result(synthetic,result,'SYNTHETIC','quantity_change_pct',terms,dropped)
    assert record['included_effects']==['Entity'] and record['time_effects'] is False
    assert set(c['term'] for c in coefs)==set(['const']+terms)
    assert 'absorbed' not in terms
    print(json.dumps(dict(status='PASS',scope='synthetic_only',checks=22,cutoff=cut['cutoff_percent'],
        native_rows=len(synthetic),fund_FE_only=True,two_way_covariance_independently_verified=True)))
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan')
    parser.add_argument('--plan-sha')
    parser.add_argument('--base-helper')
    parser.add_argument('--macro-runner-helper')
    parser.add_argument('--fixture-only',action='store_true')
    args=parser.parse_args()
    if args.fixture_only: return fixture(args)
    if not args.plan or not args.plan_sha: parser.error('--plan and --plan-sha are required')
    return execute(args)


if __name__=='__main__':
    raise SystemExit(main())
