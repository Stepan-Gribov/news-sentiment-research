"""Run the versioned H1 macro/index-universe comparison on research-host.

No inputs are modified. --fixture-only uses synthetic observations exclusively.
The caller supplies a frozen plan and its SHA256 before any coefficients exist.
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


BASE_CONTROLS = ['opening_weight', 'closing_log_NAV', 'closing_HHI',
    'log_news_count_interval', 'log_interval_days', 'bond_indicator']
MACRO_CONTROLS = ['Imoex', 'imoex_ret', 'Ruonia', 'USD_RUB', 'RVI',
    'Flow_billion_RUB', 'unique_funds', 'dummy_mar2014', 'dummy_feb2022',
    'dummy_sep2018']
POSITION_COLUMNS = ['report_id', 'fund_id', 'usable_weight', 'asset_class',
    'asset_type', 'bucket_kind', 'asset_name', 'is_reported_position', 'is_exit',
    'is_countable', 'full_report', 'security_id']
EXCEPTION_COLUMNS = ['fund_id', 'security_id', 'company_id', 'interval_id',
    'fund_name', 'asset_name', 'company',
    'company_period_id', 'start_date', 'end_date', 'state', 'eligible_primary',
    'net_sentiment', 'n_candidates', 'n_attempted', 'n_valid', 'n_relevant',
    'n_positive', 'n_negative', 'n_neutral', 'classification_coverage',
    'n_errors_or_unattempted', 'n_undated_company_candidates', 'target_security_candidate',
    'issuer_identity_conflict', 'asset_class_conflict', 'complete_cases',
    'exclusion_reason', 'complete_NET_CONTROLS', 'reason_NET_CONTROLS']


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


def row_hash(frame):
    return hashlib.sha256('\n'.join(frame.row_id.astype(str)).encode()).hexdigest()


def ordered(frame):
    return frame.sort_values(['fund_id', 'end_date', 'security_id', 'interval_id'],
        kind='stable').reset_index(drop=True)


def read_frame(path, wanted=None):
    path = Path(path)
    if path.suffix.lower() == '.parquet':
        import pyarrow.parquet as pq
        available = pq.read_schema(path).names
        columns = [c for c in wanted if c in available] if wanted else None
        return pd.read_parquet(path, columns=columns)
    frame = pd.read_csv(path, low_memory=False)
    return frame[[c for c in wanted if c in frame]] if wanted else frame


def finite_rows(frame, columns):
    missing = sorted(set(columns) - set(frame))
    if missing:
        raise ValueError(f'Missing required regression columns: {missing}')
    return pd.Series(np.isfinite(frame[columns].apply(pd.to_numeric,
        errors='coerce').to_numpy(float)).all(axis=1), index=frame.index)


def fit(sample, controls, base, covariance='clustered'):
    from linearmodels.panel import PanelOLS
    sample = ordered(sample)
    terms, dropped = base.select_regressors(sample, controls)
    index = pd.MultiIndex.from_arrays([sample.fund_id.astype(str),
        pd.to_datetime(sample.end_date)], names=['fund_id', 'report_date'])
    x = sample[terms].astype(float).copy()
    x.index = index
    x.insert(0, 'const', 1.0)
    y = pd.Series(sample.quantity_change_pct.to_numpy(float), index=index,
        name='quantity_change_pct')
    model = PanelOLS(y, x, entity_effects=True, time_effects=False,
        other_effects=None, singletons=True, drop_absorbed=False, check_rank=True)
    options = dict(cov_type=covariance, debiased=True, auto_df=False,
        count_effects=True)
    if covariance == 'clustered':
        clusters = pd.DataFrame({
            'fund_cluster': pd.Categorical(sample.fund_id.astype(str)).codes,
            'company_cluster': pd.Categorical(sample.company_id.astype(str)).codes},
            index=index)
        options.update(clusters=clusters, cluster_time=False,
            cluster_entity=False, group_debias=True)
    result = model.fit(**options)
    if result.nobs != len(sample) or not result.resids.index.equals(index):
        raise AssertionError('Native estimator changed row count/order')
    return sample, result, terms, dropped


def residual_diagnostics(sample, result):
    squared = np.square(result.resids.to_numpy(float))
    total = float(squared.sum())
    tests = {
        'growth_over_10000_percent': sample.quantity_change_pct.gt(10000),
        'HHI_outside_0_1': sample.closing_HHI.lt(0) | sample.closing_HHI.gt(1.001),
        'opening_weight_outside_0_1': sample.opening_weight.lt(0) | sample.opening_weight.gt(1.001),
    }
    if 'closing_weight' in sample:
        tests['closing_weight_outside_0_1'] = sample.closing_weight.lt(0) | sample.closing_weight.gt(1.001)
    flags = {}
    for name, mask in tests.items():
        mask = mask.fillna(False).to_numpy(bool)
        contribution = float(squared[mask].sum())
        flags[name] = dict(rows=int(mask.sum()), residual_SS=contribution,
            residual_SS_fraction=contribution / total if total else None)
    return dict(residual_SS=total, flags=flags,
        largest_10_residual_SS_fraction=float(np.sort(squared)[-10:].sum()/total) if total else None,
        interpretation='Descriptive contributions to fitted residual variance; no observations removed and no causal influence claim.')


def export_sample(frame, root, model):
    stem = root / 'samples' / model
    stem.parent.mkdir(exist_ok=True)
    wanted=set(['row_id','fund_id','fund_name','security_id','asset_name','isin',
        'price_isin','company_id','company','interval_id','company_period_id',
        'opening_report_id','closing_report_id','start_date','end_date',
        'quantity_open','quantity_close','quantity_change_pct','delta_quantity_units',
        'delta_share_pp','net_sentiment','n_relevant','state','eligible_primary',
        'source_path','exclude_index_fund','w2_static_link_used',
        'residence_documented_by_disclosure','focus_bond','focus_mixed',
        'net_x_focus_bond','net_x_focus_mixed']+BASE_CONTROLS+MACRO_CONTROLS)
    prefixes=('index_','fund_focus','opening_','closing_','Imoex_','imoex_',
        'Ruonia_','USD_RUB_','RVI_','Flow_')
    columns=[c for c in frame if c in wanted or c.startswith(prefixes)]
    exported=frame[columns]
    exported.to_parquet(stem.with_suffix('.parquet'), index=False, compression='zstd')
    exported.to_csv(stem.with_suffix('.csv.gz'), index=False, encoding='utf-8-sig',
        compression='gzip')
    return dict(parquet=str(stem.with_suffix('.parquet')),
        csv_gz=str(stem.with_suffix('.csv.gz')), rows=len(frame), row_hash=row_hash(frame),
        exported_columns=columns,complete_original_input_retained_at_source=True)


def model_status(sample, minimum_rows):
    if len(sample) < minimum_rows:
        return f'Fewer than {minimum_rows} complete rows'
    if sample.fund_id.nunique() < 2 or sample.company_id.nunique() < 2:
        return 'Fewer than two funds or two company clusters'
    return None


def holm_focus_adjustment(coefficients, focus_groups):
    """Predeclared subgroup family; unestimated hypotheses retain p=1 slots."""
    family=['NET_MACRO_FOCUS_'+str(focus).upper() for focus in focus_groups]
    values=[]
    for name in family:
        match=[row for row in coefficients if row['model']==name and row['term']=='net_sentiment']
        p=float(match[0]['p_native']) if match else 1.0
        values.append((name,p if np.isfinite(p) else 1.0,bool(match)))
    maximum=0.0; adjusted={}
    for rank,(name,p,estimated) in enumerate(sorted(values,key=lambda x:x[1])):
        maximum=max(maximum,min(1.0,(len(values)-rank)*p))
        adjusted[name]=maximum if estimated else None
    for row in coefficients:
        row['p_native_holm_focus_family']=(adjusted.get(row['model'])
            if row['term']=='net_sentiment' else None)
    return dict(method='Holm step-down',two_sided_native_pvalues=True,
        planned_family=family,unestimated_tests_use_p1_family_slots=True,
        adjusted_pvalues=adjusted)


def render_report(root, summary, comparison, native):
    table = comparison.to_html(index=False, border=0, float_format=lambda x: f'{x:.6g}')
    text = ['<!doctype html><html lang="ru"><meta charset="utf-8">',
        '<title>H1: макроконтроли и исключение выявленных индексных фондов</title>',
        '<style>body{font:16px system-ui;max-width:1300px;margin:36px auto;padding:0 18px;color:#17212d}table{border-collapse:collapse;font-size:14px}th,td{padding:9px;border-bottom:1px solid #dce1e7;text-align:right}th:first-child,td:first-child{text-align:left}pre{font:13px ui-monospace,monospace;white-space:pre;overflow:auto;padding:18px;background:#f4f6f8}a{color:#075ba5}</style>',
        '<h1>H1: NetSentiment и изменение количества бумаг</h1>',
        '<p>Y = 100 × (Q₁/Q₀ − 1). Фиксированные эффекты фондов. Кластеры ошибок: фонд и компания; кластеризации по времени и временных фиксированных эффектов нет. Нулевые новости и неизвестное покрытие не заменяются нейтральным сентиментом.</p>',
        '<p>ORIGINAL_MICRO — историческая проверка прежней модели, включая индексные фонды. В новых моделях исключены подтверждённые индексные фонды целиком за всю историю. Индексность остальных фондов установлена не всегда; их отдельный статус и политика включения записаны в sample_flow. BASELINE_COMMON и NET_MACRO_COMMON используют одинаковые строки. Наблюдения с экстремальными значениями сохранены.</p>',
        '<p>Макроконтроли описывают состояние рынка на отчётную дату/за интервал. Исторические пересмотренные ряды не являются доказанными доступными управляющему в реальном времени винтажами.</p>', table,
        '<p>Для заранее заданных групп фондов приведены также p-values с поправкой Holm. Разница в значимости отдельных групп сама по себе не доказывает различия их коэффициентов; при доступной оценке отдельный pooled interaction Wald проверяет равенство склонов.</p>',
        '<p>Файлы: <a href="summary.json">summary.json</a>, <a href="coefficients.csv">все коэффициенты</a>, <a href="sample_flow.json">отбор наблюдений</a>, <a href="universe_metadata.json">классификация фондов</a>, <a href="macro_metadata.json">макроконтроли</a>, <a href="exception_summary.json">новостное покрытие и исключения</a>, <a href="source_manifest.json">использованные файлы</a>.</p>']
    for name, native_text in native.items():
        text.append(f'<h2>{html.escape(name)}</h2><pre>{html.escape(native_text)}</pre>')
    failed = [m for m in summary['models'] if m.get('status') != 'ESTIMATED']
    if failed:
        text.append('<h2>Неоценённые модели</h2><pre>'+html.escape(json.dumps(failed,ensure_ascii=False,indent=2))+'</pre>')
    text.append('</html>')
    (root / 'report.html').write_text('\n'.join(text), encoding='utf-8')


def execute(args):
    plan_path = Path(args.plan)
    if sha(plan_path) != args.plan_sha:
        raise ValueError('Frozen plan SHA256 mismatch')
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    root = Path(plan['output_root'])
    if 'databaseresources' in [x.lower() for x in root.parts]:
        raise ValueError('Output cannot be within protected inputs')
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'completion.json').exists():
        raise FileExistsError('This completed version must not be overwritten')
    base = load_module(args.base_helper, 'h1_previous_helper')
    macro = load_module(args.macro_helper, 'h1_macro_helper')
    universe = load_module(args.universe_helper, 'h1_universe_helper')
    source_paths={k:plan[k] for k in ['input_panel','interval_source','all_candidate_panel','positions_source','fund_reports_source','news_company_source'] if plan.get(k)}
    source_paths.update(base_helper=args.base_helper,macro_helper=args.macro_helper,
        universe_helper=args.universe_helper,plan=str(plan_path))
    initial_stats={k:(Path(v).stat().st_size,Path(v).stat().st_mtime_ns) for k,v in source_paths.items()}
    for key,expected in plan.get('expected_source_sha256',{}).items():
        if key not in source_paths or sha(source_paths[key])!=expected:
            raise ValueError(f'Frozen source checksum mismatch: {key}')
    lock = root / 'run.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, json.dumps({'pid':os.getpid(),'plan_sha256':args.plan_sha}).encode())
    os.close(descriptor)
    started = time.time()
    def save(name, value):
        base.save(root / name, value)
    def checkpoint(state, **extra):
        save('checkpoint.json', dict(state=state, pid=os.getpid(),
            elapsed_seconds=time.time()-started, **extra))
        print(json.dumps(dict(state=state, **extra)), flush=True)
    try:
        checkpoint('READING_INPUTS')
        panel = ordered(read_frame(plan['input_panel']))
        if panel.row_id.duplicated().any():
            raise ValueError('Duplicate input row_id')
        controls = plan.get('baseline_controls', BASE_CONTROLS)
        macros = plan.get('macro_controls', MACRO_CONTROLS)
        if len(panel) != int(plan.get('expected_original_rows', 305163)):
            raise ValueError('Prior primary sample row count changed')
        if not finite_rows(panel,['quantity_change_pct','net_sentiment']+controls).all():
            raise ValueError('Prior sample no longer meets its complete-case definition')
        intervals = read_frame(plan['interval_source'])
        for column in ['start_date','end_date']:
            intervals[column] = pd.to_datetime(intervals[column])
            panel[column] = pd.to_datetime(panel[column])
        if intervals.interval_id.duplicated().any():
            raise ValueError('Full interval source contains duplicate interval_id')
        full_counts = intervals.groupby('end_date',as_index=False).fund_id.nunique().rename(columns={'fund_id':'unique_funds'})
        registry_path = plan.get('fund_reports_source')
        if registry_path:
            registry = read_frame(registry_path)
        else:
            import duckdb
            connection = duckdb.connect(plan['catalog_path'], read_only=True)
            try:
                registry = connection.execute('SELECT report_id,fund_id,report_date,fund_name_xls,fund_type_category FROM funds.reports').fetchdf()
            finally:
                connection.close()
        positions = read_frame(plan['positions_source'], POSITION_COLUMNS) if plan.get('positions_source') else None
        checkpoint('CLASSIFYING_FUND_UNIVERSE')
        universe_options = dict(plan.get('universe_options',{}))
        unknown_policy = universe_options.pop('index_unknown_policy',None)
        if unknown_policy not in ['exclude','retain_flagged']:
            raise ValueError('Plan must freeze index_unknown_policy: exclude or retain_flagged')
        mapping, universe_meta = universe.classify_funds(intervals,registry=registry,
            positions=positions,**universe_options)
        del positions
        if mapping.duplicated(['interval_id','fund_id']).any():
            raise ValueError('Fund universe mapping is not unique')
        mapping.to_parquet(root / 'fund_interval_classification.parquet', index=False, compression='zstd')
        mapping.to_csv(root / 'fund_interval_classification.csv.gz',index=False,encoding='utf-8-sig',compression='gzip')
        save('universe_metadata.json', universe_meta)
        required_map = ['interval_id','fund_id','index_status','exclude_index_fund','fund_focus']
        if set(required_map)-set(mapping):
            raise ValueError('Universe helper omitted mandatory mapping fields')
        mapping_columns = [c for c in mapping if c in ['interval_id','fund_id'] or c not in panel]
        joined = panel.merge(mapping[mapping_columns],on=['interval_id','fund_id'],how='left',validate='many_to_one',sort=False)
        if joined.index_status.isna().any():
            raise ValueError('Some regression intervals have no universe mapping')
        index_free = joined.index_status.ne('index') & ~joined.exclude_index_fund.fillna(False).astype(bool)
        strict_statuses=plan.get('strict_nonindex_statuses',['nonindex_verified'])
        if unknown_policy == 'exclude':
            index_free &= joined.index_status.isin(strict_statuses)
        nonindex = joined.loc[index_free].copy()
        macro_options = dict(plan.get('macro_builder_options',{}))
        if macro_options.get('flow_mode') != 'latest_completed' or macro_options.get('flow_month_label') != 'previous_month':
            raise ValueError('Primary plan must explicitly fix the reviewed Flow timing convention')
        if macro_options.get('validate_hashes') is not True:
            raise ValueError('Actual macro-source hashes must be validated')
        checkpoint('BUILDING_MACRO_CONTROLS',nonindex_rows=len(nonindex))
        macro_table, macro_meta = macro.build_macro_controls(intervals,plan['source_root'],
            unique_funds_by_date=full_counts,**macro_options)
        macro_table.to_parquet(root / 'macro_controls_by_interval.parquet',index=False,compression='zstd')
        macro_table.to_csv(root / 'macro_controls_by_interval.csv.gz',index=False,encoding='utf-8-sig',compression='gzip')
        save('macro_metadata.json',macro_meta)
        existing = set(nonindex)-{'interval_id'}
        macro_columns = [c for c in macro_table if c=='interval_id' or c not in existing]
        nonindex = ordered(nonindex.merge(macro_table[macro_columns],on='interval_id',how='left',validate='many_to_one',sort=False))
        complete = finite_rows(nonindex,['quantity_change_pct','net_sentiment']+controls+macros)
        common = nonindex.loc[complete].copy()
        missing_macro = {c:int(pd.to_numeric(nonindex[c],errors='coerce').isna().sum()) for c in macros}
        flow = dict(original_rows=len(panel),original_funds=int(panel.fund_id.nunique()),
            index_status_rows=joined.index_status.value_counts(dropna=False).to_dict(),
            index_status_funds=joined.groupby('index_status').fund_id.nunique().to_dict(),
            index_unknown_policy=unknown_policy,nonindex_rows=len(nonindex),
            nonindex_funds=int(nonindex.fund_id.nunique()),macro_complete_rows=len(common),
            macro_complete_funds=int(common.fund_id.nunique()),missing_macro_rows_nonexclusive=missing_macro,
            focus_rows=nonindex.fund_focus.value_counts(dropna=False).to_dict(),
            full_universe_fund_counts_scope='All source true intervals before news, index, and complete-case filters',
            missing_news_not_imputed=True,extreme_values_not_trimmed=True)
        save('sample_flow.json',flow)
        candidates = read_frame(plan['all_candidate_panel'],EXCEPTION_COLUMNS)
        company_ids = None
        if plan.get('news_company_source'):
            company_ids = read_frame(plan['news_company_source'],['company_id']).company_id.dropna().astype(str).unique()
        exception_report = universe.exception_summary(candidates,news_company_ids=company_ids)
        exception_reasons=universe.sentiment_exception_reasons(candidates,news_company_ids=company_ids)
        audit_columns=[c for c in ['fund_id','fund_name','security_id','asset_name',
            'company_id','company','interval_id','company_period_id','start_date',
            'end_date','state','net_sentiment','eligible_primary','complete_cases',
            'complete_NET_CONTROLS','exclusion_reason'] if c in candidates]
        exception_rows=candidates[audit_columns].copy()
        exception_rows['sentiment_exception_reason']=exception_reasons.to_numpy()
        exception_rows=exception_rows.merge(mapping[required_map],on=['interval_id','fund_id'],
            how='left',validate='many_to_one',sort=False)
        exception_rows['new_universe_eligible']=exception_rows.index_status.notna() & \
            exception_rows.index_status.ne('index') & ~exception_rows.exclude_index_fund.fillna(False).astype(bool)
        if unknown_policy=='exclude':
            exception_rows['new_universe_eligible'] &= exception_rows.index_status.isin(strict_statuses)
        exception_rows.to_csv(root/'exception_rows.csv.gz',index=False,encoding='utf-8-sig',compression='gzip')
        case_column=next((c for c in ['complete_NET_CONTROLS','complete_cases'] if c in exception_rows),None)
        cross_keys=['sentiment_exception_reason','index_status','new_universe_eligible']
        if case_column:
            cross_keys.append(case_column)
        exception_report['prior_case_index_news_cross_counts']=exception_rows.groupby(
            cross_keys,dropna=False).size().reset_index(name='rows').to_dict(orient='records')
        exception_report['row_level_export']='exception_rows.csv.gz'
        save('exception_summary.json',exception_report)
        del candidates,exception_rows
        specs = [('ORIGINAL_MICRO',panel,controls),('NONINDEX_MICRO',nonindex,controls),
            ('BASELINE_COMMON',common,controls),('NET_MACRO_COMMON',common,controls+macros)]
        focus_groups=plan.get('fund_focus_groups',['equity','bond','mixed'])
        for focus in focus_groups:
            specs.append(('NET_MACRO_FOCUS_'+str(focus).upper(),common.loc[common.fund_focus.eq(focus)],controls+macros))
        if unknown_policy=='retain_flagged' and plan.get('strict_nonindex_secondary',True):
            strict=common.loc[common.index_status.isin(strict_statuses)].copy()
            specs.extend([('BASELINE_VERIFIED_COMMON_SECONDARY',strict,controls),
                ('NET_MACRO_VERIFIED_NONINDEX_SECONDARY',strict,controls+macros)])
        interaction_name='NET_MACRO_FOCUS_INTERACTION_SECONDARY'
        interaction_terms=['net_x_focus_bond','net_x_focus_mixed']
        interaction_result=dict(status='NOT_REQUESTED')
        if plan.get('focus_interaction_wald',True):
            core=['equity','bond','mixed']
            pooled=common.loc[common.fund_focus.isin(core)].copy()
            enough=all(model_status(pooled.loc[pooled.fund_focus.eq(focus)],
                int(plan.get('minimum_model_rows',30))) is None for focus in core)
            if enough:
                for focus in ['bond','mixed']:
                    pooled['focus_'+focus]=pooled.fund_focus.eq(focus).astype(float)
                    pooled['net_x_focus_'+focus]=pooled.net_sentiment*pooled['focus_'+focus]
                specs.append((interaction_name,pooled,controls+macros+
                    ['focus_bond','focus_mixed']+interaction_terms))
                interaction_result=dict(status='PENDING',reference_group='equity',
                    null_hypothesis='Equal net-sentiment slopes for equity, bond, and mixed funds',
                    common_control_slopes=True,fund_focus_main_effects_included=True,
                    secondary=True)
            else:
                interaction_result=dict(status='NOT_ESTIMABLE',
                    reason='At least one prespecified focus group lacks minimum rows or clusters')
        if plan.get('security_subgroups',False):
            specs.extend([('NET_MACRO_SECURITY_BOND',common.loc[common.bond_indicator.eq(1)],controls+macros),
                ('NET_MACRO_SECURITY_NONBOND',common.loc[common.bond_indicator.eq(0)],controls+macros)])
        rows=[];models=[];native={};samples={};residual={}
        minimum_rows=int(plan.get('minimum_model_rows',30))
        for name,sample,terms in specs:
            checkpoint('FITTING_MODEL',model=name,rows=len(sample))
            reason=model_status(sample,minimum_rows)
            if reason:
                if name=='ORIGINAL_MICRO':
                    raise ValueError('Historical reference is not estimable: '+reason)
                models.append(dict(model=name,status='NOT_ESTIMABLE',reason=reason,nobs=len(sample),
                    funds=int(sample.fund_id.nunique()),companies=int(sample.company_id.nunique())))
                continue
            try:
                sample,result,retained,dropped=fit(sample,terms,base)
                record,coefficients=base.describe_result(sample,result,name,'quantity_change_pct',retained,dropped)
            except Exception as error:
                if name=='ORIGINAL_MICRO':
                    raise
                models.append(dict(model=name,status='NOT_ESTIMABLE',reason=f'{type(error).__name__}: {error}',
                    nobs=len(sample),funds=int(sample.fund_id.nunique()),companies=int(sample.company_id.nunique())))
                continue
            record['covariance_options']['cluster_time']=False
            record['is_historical_reproduction']=name=='ORIGINAL_MICRO'
            record['index_unknown_policy']=unknown_policy
            if name==interaction_name:
                if all(term in result.params.index for term in interaction_terms):
                    restriction=np.zeros((2,len(result.params)))
                    for number,term in enumerate(interaction_terms):
                        restriction[number,result.params.index.get_loc(term)]=1
                    try:
                        covariance=restriction@result.cov.to_numpy()@restriction.T
                        eigenvalues=np.linalg.eigvalsh((covariance+covariance.T)/2)
                        interaction_result['interaction_covariance_eigenvalues']=eigenvalues.tolist()
                        if not np.isfinite(eigenvalues).all() or eigenvalues.min()<0:
                            raise ValueError('Interaction covariance is nonfinite or not positive semidefinite; no repair applied')
                        test=result.wald_test(restriction=restriction,value=np.zeros(2))
                        interaction_result.update(status='ESTIMATED',statistic=float(test.stat),
                            p_value=float(test.pval),df=int(test.df),distribution=str(test.dist_name),
                            rows=len(sample),funds=int(sample.fund_id.nunique()),
                            companies=int(sample.company_id.nunique()),source_row_hash=row_hash(sample))
                        if not np.isfinite(test.stat) or test.stat<0 or not np.isfinite(test.pval):
                            interaction_result.update(status='UNDEFINED_COVARIANCE',
                                reason='Native Wald statistic is negative or nonfinite; no PSD repair applied')
                    except Exception as error:
                        interaction_result.update(status='NOT_ESTIMABLE',reason=f'{type(error).__name__}: {error}')
                else:
                    interaction_result.update(status='NOT_ESTIMABLE',
                        reason='A required sentiment interaction is structurally collinear/absorbed')
            if name=='ORIGINAL_MICRO' and plan.get('verify_original_reference',True):
                if not np.isclose(result.params['net_sentiment'],-451.5911315954303,rtol=1e-7,atol=1e-5):
                    raise AssertionError('Original H1 coefficient did not reproduce')
                if not np.isclose(result.std_errors['net_sentiment'],4078.619816999392,rtol=1e-7,atol=1e-5):
                    raise AssertionError('Original H1 covariance did not reproduce')
            text=str(result.summary)
            (root/(name+'.summary.txt')).write_text(text,encoding='utf-8')
            (root/(name+'.summary.html')).write_text(result.summary.as_html(),encoding='utf-8')
            native[name]=text;models.append(record);rows.extend(coefficients)
            samples[name]=export_sample(sample,root,name)
            if name=='NET_MACRO_COMMON':
                preview_columns=[c for c in ['fund_name','fund_id','asset_name','security_id',
                    'company','company_id','start_date','end_date','quantity_open','quantity_close',
                    'quantity_change_pct','net_sentiment','n_relevant','index_status','fund_focus',
                    'fund_focus_source']+controls+macros+['row_id'] if c in sample]
                sample[preview_columns].head(200).to_csv(root/'regression_preview.csv',
                    index=False,encoding='utf-8-sig')
            residual[name]=residual_diagnostics(sample,result)
            if name=='NET_MACRO_COMMON' and plan.get('robust_sensitivity',False):
                robust_sample,robust,robust_terms,robust_dropped=fit(sample,terms,base,'robust')
                if not np.allclose(result.params,robust.params,rtol=1e-10,atol=1e-8):
                    raise AssertionError('Covariance-only sensitivity changed slopes')
                robust_name='NET_MACRO_COMMON_HC1_SECONDARY'
                rec,coefs=base.describe_result(robust_sample,robust,robust_name,'quantity_change_pct',robust_terms,robust_dropped)
                rec['covariance_options']=dict(cov_type='robust',cluster_time=False,debiased=True,auto_df=False,count_effects=True)
                rec['secondary_covariance_only']=True
                rec['conservative_cluster_df']=None
                for coefficient in coefs:
                    for field in ['conservative_cluster_df','p_cluster_t_two_sided',
                            'ci95_cluster_t_low','ci95_cluster_t_high','p_H1_positive_one_sided']:
                        coefficient[field]=None
                models.append(rec);rows.extend(coefs);native[robust_name]=str(robust.summary)
                (root/(robust_name+'.summary.txt')).write_text(str(robust.summary),encoding='utf-8')
                (root/(robust_name+'.summary.html')).write_text(robust.summary.as_html(),encoding='utf-8')
        if 'BASELINE_COMMON' in samples and 'NET_MACRO_COMMON' in samples:
            assert samples['BASELINE_COMMON']['row_hash']==samples['NET_MACRO_COMMON']['row_hash']
        if 'BASELINE_VERIFIED_COMMON_SECONDARY' in samples and 'NET_MACRO_VERIFIED_NONINDEX_SECONDARY' in samples:
            assert samples['BASELINE_VERIFIED_COMMON_SECONDARY']['row_hash']==samples['NET_MACRO_VERIFIED_NONINDEX_SECONDARY']['row_hash']
        multiple_testing=holm_focus_adjustment(rows,plan.get('holm_focus_groups',['equity','bond','mixed']))
        coefficients=pd.DataFrame(rows)
        coefficients.to_csv(root/'coefficients.csv',index=False,encoding='utf-8-sig')
        comparison=[]
        for record in models:
            item={k:record.get(k) for k in ['model','status','nobs','funds','companies','first_start_date','last_end_date','rsquared_within']}
            matches=[c for c in rows if c['model']==record['model'] and c['term']=='net_sentiment']
            if matches:
                item.update({k:matches[0].get(k) for k in ['beta','se_native','t_native','p_native','ci95_native_low','ci95_native_high','p_H1_positive_one_sided','p_native_holm_focus_family']})
            else:
                item['reason']=record.get('reason')
            comparison.append(item)
        comparison=pd.DataFrame(comparison)
        comparison.to_csv(root/'comparison.csv',index=False,encoding='utf-8-sig')
        primary_ok=any(m['model']=='NET_MACRO_COMMON' and m.get('status')=='ESTIMATED' for m in models)
        summary=dict(status='COMPLETE' if primary_ok else 'PRIMARY_NOT_ESTIMABLE',
            specification=plan,plan_sha256=args.plan_sha,models=models,samples=samples,
            multiple_testing=multiple_testing,focus_interaction_wald=interaction_result,
            full_index_identification_guaranteed=universe_meta.get('full_index_identification_guaranteed',False),
            covariance='fund/company clusters; no time clustering',fund_FE_only=True,
            no_time_FE=True,no_sentiment_imputation=True,no_outlier_trimming=True,
            elapsed_seconds=time.time()-started,sample_flow=flow)
        save('summary.json',summary);save('residual_diagnostics.json',residual)
        save('focus_interaction_wald.json',interaction_result)
        for key,value in source_paths.items():
            if initial_stats[key]!=(Path(value).stat().st_size,Path(value).stat().st_mtime_ns):
                raise ValueError(f'Source changed during execution: {key}')
        manifest={k:dict(path=str(v),bytes=Path(v).stat().st_size,sha256=sha(v)) for k,v in source_paths.items()}
        manifest['macro_raw_sources']=macro_meta['source_files']
        manifest['catalog_registry_read']=plan.get('catalog_path') if not registry_path else None
        manifest['source_inputs_modified']=False
        save('source_manifest.json',manifest)
        save('variable_dictionary.json',dict(macro=macro_meta['variable_dictionary'],
            dependent='100*(quantity_close/quantity_open-1), defined only with positive opening quantity',
            sentiment='Original strict fully classified raw net; no-news/outside-universe/errors remain missing',
            universe=universe_meta,baseline_controls=controls))
        render_report(root,summary,comparison,native)
        checkpoint(summary['status'],models=len(models))
        save('completion.json',dict(status=summary['status'],plan_sha256=args.plan_sha,
            finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),models=len(models),
            report=str(root/'report.html')))
        return 0 if primary_ok else 2
    except Exception as error:
        save('failure.json',dict(status='FAILED',error_type=type(error).__name__,
            message=str(error),traceback=traceback.format_exc(),plan_sha256=args.plan_sha))
        raise
    finally:
        lock.unlink(missing_ok=True)


def fixture(args):
    base=load_module(args.base_helper,'h1_previous_fixture_helper')
    macro=load_module(args.macro_helper,'h1_macro_fixture_helper')
    universe=load_module(args.universe_helper,'h1_universe_fixture_helper')
    checks=[macro.self_test()]
    universe_check=universe.self_test()
    rng=np.random.default_rng(49182)
    records=[]
    for fund in range(8):
        for date in range(7):
            for company in range(4):
                signal,control,noise=rng.normal(size=3)
                records.append(dict(fund_id=str(fund),company_id=str(company),
                    security_id=str(company),interval_id=f'{fund}-{date}',
                    end_date=pd.Timestamp('2020-01-31')+pd.Timedelta(days=31*date),
                    net_sentiment=signal,control=control,absorbed=float(fund),
                    quantity_change_pct=2.5*signal-.4*control+.5*fund+noise))
    sample=pd.DataFrame(records)
    sample,result,terms,dropped=fit(sample,['control','absorbed'],base)
    robust_sample,robust,_,_=fit(sample,['control','absorbed'],base,'robust')
    assert result.nobs==len(records)
    assert 'absorbed' not in terms and any(x['term']=='absorbed' for x in dropped)
    assert np.allclose(result.params,robust.params)
    assert result.included_effects==['Entity']
    assert abs(result.params['net_sentiment']-2.5)<.3
    assert sample.equals(robust_sample)
    holm_rows=[dict(model='NET_MACRO_FOCUS_EQUITY',term='net_sentiment',p_native=.01),
        dict(model='NET_MACRO_FOCUS_BOND',term='net_sentiment',p_native=.03)]
    adjustment=holm_focus_adjustment(holm_rows,['equity','bond','mixed'])
    assert np.isclose(adjustment['adjusted_pvalues']['NET_MACRO_FOCUS_EQUITY'],.03)
    assert np.isclose(adjustment['adjusted_pvalues']['NET_MACRO_FOCUS_BOND'],.06)
    assert adjustment['adjusted_pvalues']['NET_MACRO_FOCUS_MIXED'] is None
    print(json.dumps(dict(status='PASS',scope='synthetic_only',native_fit_checks=6,
        macro_fixture=checks,universe_fixture=universe_check,holm_checks=3),default=str))
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan')
    parser.add_argument('--plan-sha')
    parser.add_argument('--base-helper',default=str(Path(__file__).with_name('run_h1_net_fund_fe_20261030.py')))
    parser.add_argument('--macro-helper',default=str(Path(__file__).with_name('h1_macro_controls_20261031.py')))
    parser.add_argument('--universe-helper',default=str(Path(__file__).with_name('h1_fund_universe_20261031.py')))
    parser.add_argument('--fixture-only',action='store_true')
    args=parser.parse_args()
    if args.fixture_only:
        return fixture(args)
    if not args.plan or not args.plan_sha:
        parser.error('--plan and --plan-sha are required for an actual-data run')
    return execute(args)


if __name__=='__main__':
    raise SystemExit(main())
