"""Frozen H1 current/previous-actual-interval Net models on identical samples.

Actual research execution is Windows-only and requires a final frozen plan.
Synthetic fixtures never read research data. Existing outputs are immutable.
"""
from __future__ import annotations

import argparse
import copy
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


MICRO=['opening_weight','closing_log_NAV','closing_HHI','log_news_count_interval','log_interval_days','bond_indicator']
MACRO=['Imoex','imoex_ret','Ruonia','USD_RUB','RVI','Flow_billion_RUB','unique_funds','dummy_mar2014','dummy_feb2022','dummy_sep2018']
CURRENT='net_sentiment'
LAG='net_sentiment_lag1'
VARIANTS=['CURRENT_COMMON','CURRENT_PLUS_LAG1']
COEF_FIELDS=['model','outcome','term','beta','se_native','t_native','p_native','ci95_native_low','ci95_native_high']
SOURCE_KEYS=['input_panel','focus_summary','focus_coefficients','focus_support','focus_helper','interval_source',
             'net_source','lag_map','lag_windows','lag_data_audit','base_helper','macro_runner_helper']


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,str(path));module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def row_hash(frame):return hashlib.sha256('\n'.join(frame.row_id.astype(str)).encode()).hexdigest()


def ordered(frame):return frame.sort_values(['fund_id','end_date','security_id','interval_id'],kind='stable').reset_index(drop=True)


def truth(series):return series.eq(True).fillna(False).astype(bool)


def finite(series):return pd.Series(np.isfinite(pd.to_numeric(series,errors='coerce')),index=series.index)


def join_lag(frame,mapping):
    required=['row_id','current_interval_id','fund_id','company_id','current_start_date','current_end_date',
        'lag_interval_id','lag_start_date','lag_end_date','lag_closing_report_id','lag_company_id',
        'lag_company_period_id',LAG,'lag_eligible_primary','lag_reason','lag_source_kind']
    if set(required)-set(mapping):raise ValueError('Lag map missing fields: '+str(sorted(set(required)-set(mapping))))
    if frame.row_id.isna().any()or frame.row_id.duplicated().any():raise ValueError('Source row IDs invalid')
    if mapping.row_id.isna().any()or mapping.row_id.duplicated().any():raise ValueError('Lag map row IDs invalid')
    if set(frame.row_id)!=set(mapping.row_id):raise AssertionError('Lag map must cover exactly the frozen source rows')
    aligned=mapping.set_index('row_id').reindex(frame.row_id)
    identity=[('fund_id','fund_id'),('company_id','company_id'),('interval_id','current_interval_id')]
    for source,mapped in identity:
        if not np.array_equal(frame[source].astype(str),aligned[mapped].astype(str)):
            raise AssertionError('Lag map current identity mismatch: '+source)
    for source,mapped in [('start_date','current_start_date'),('end_date','current_end_date')]:
        if not np.array_equal(pd.to_datetime(frame[source]).to_numpy(),pd.to_datetime(aligned[mapped]).to_numpy()):
            raise AssertionError('Lag map current dates mismatch: '+source)
    columns=[c for c in mapping if c=='row_id'or c.startswith('lag_')or c==LAG]
    if set(columns)-{'row_id'} & set(frame):raise ValueError('Lag fields already exist in source')
    # Assign the already one-to-one aligned columns, avoiding merge coercion of
    # original nullable-string keys or other literal source-column dtypes.
    joined=frame.copy()
    for column in columns:
        if column=='row_id':continue
        values=aligned[column].reset_index(drop=True);values.index=frame.index
        joined[column]=values
    if not np.array_equal(joined.row_id.astype(str),frame.row_id.astype(str)):raise AssertionError('Lag join changed row order')
    for column in frame:
        if not joined[column].equals(frame[column]):raise AssertionError('Lag join changed original column: '+column)
    has_previous=joined.lag_interval_id.notna()
    previous_start=pd.to_datetime(joined.lag_start_date)
    previous_end=pd.to_datetime(joined.lag_end_date)
    if not np.array_equal(previous_end.loc[has_previous].to_numpy(dtype='datetime64[ns]'),
                          pd.to_datetime(joined.loc[has_previous,'start_date']).to_numpy(dtype='datetime64[ns]')):
        raise AssertionError('Lag does not end at current interval start')
    if not previous_start.loc[has_previous].lt(previous_end.loc[has_previous]).all():raise AssertionError('Lag interval is empty or reversed')
    if not joined.loc[has_previous,'lag_company_id'].astype(str).eq(joined.loc[has_previous,'company_id'].astype(str)).all():
        raise AssertionError('Lag issuer differs from current issuer')
    if not joined.loc[has_previous,'lag_closing_report_id'].astype(str).eq(joined.loc[has_previous,'opening_report_id'].astype(str)).all():
        raise AssertionError('Lag report chain does not meet current opening report')
    eligible=truth(joined.lag_eligible_primary)
    if (eligible&~has_previous).any():raise AssertionError('Eligible lag has no predecessor interval')
    if (eligible&~finite(joined[LAG])).any():raise AssertionError('Eligible lag score is not finite')
    if (finite(joined[LAG])&~eligible).any():raise AssertionError('Ineligible lag was filled with a numeric score')
    if 'lag_n_relevant'in joined and not pd.to_numeric(joined.loc[eligible,'lag_n_relevant'],errors='coerce').gt(0).all():
        raise AssertionError('Eligible lag has no relevant denominator')
    joined['complete_current_lag1']=eligible&finite(joined[LAG])
    return joined


def fit_native(sample,controls,base,distributed):
    from linearmodels.panel import PanelOLS
    sample=ordered(sample)
    declared=([LAG]if distributed else[])+list(controls)
    terms,dropped=base.select_regressors(sample,declared)
    required=[CURRENT,LAG]if distributed else[CURRENT]
    missing=set(required)-set(terms)
    if missing:raise ValueError('Required sentiment target is absorbed/collinear: '+str(sorted(missing)))
    index=pd.MultiIndex.from_arrays([sample.fund_id.astype(str),pd.to_datetime(sample.end_date)],names=['fund_id','report_date'])
    x=sample[terms].astype(float).copy();x.index=index;x.insert(0,'const',1.)
    y=pd.Series(sample.quantity_change_pct.to_numpy(float),index=index,name='quantity_change_pct')
    clusters=pd.DataFrame({'fund_cluster':pd.Categorical(sample.fund_id.astype(str)).codes,
        'company_cluster':pd.Categorical(sample.company_id.astype(str)).codes},index=index)
    model=PanelOLS(y,x,entity_effects=True,time_effects=False,other_effects=None,
        singletons=True,drop_absorbed=False,check_rank=True)
    result=model.fit(cov_type='clustered',clusters=clusters,cluster_time=False,cluster_entity=False,
        group_debias=True,debiased=True,auto_df=False,count_effects=True)
    if result.nobs!=len(sample)or not result.resids.index.equals(index):raise AssertionError('Estimator changed rows/order')
    return sample,result,terms,dropped


def distributed_tests(result):
    from scipy import stats
    targets=[CURRENT,LAG]
    covariance=result.cov.loc[targets,targets].to_numpy(float)
    beta=result.params.loc[targets].to_numpy(float)
    joint=dict(status='UNDEFINED_COVARIANCE',statistic=None,p_value=None,df=2,distribution='chi2(2)')
    total=dict(status='UNDEFINED_COVARIANCE',beta=float(beta.sum()),variance=None,se_native=None,
        ci95_native_low=None,ci95_native_high=None,p_native=None,native_df_resid=int(result.df_resid))
    eigenvalues=None
    if np.isfinite(covariance).all():
        eigenvalues=np.linalg.eigvalsh((covariance+covariance.T)/2)
        if eigenvalues.min()>0:
            restriction=np.zeros((2,len(result.params)))
            for j,term in enumerate(targets):restriction[j,result.params.index.get_loc(term)]=1.
            wald=result.wald_test(restriction=restriction,value=np.zeros(2))
            if np.isfinite(wald.stat)and wald.stat>=0 and np.isfinite(wald.pval):
                joint.update(status='ESTIMATED',statistic=float(wald.stat),p_value=float(wald.pval),
                    df=int(wald.df),distribution=str(wald.dist_name))
        variance=float(covariance.sum());total['variance']=variance
        if variance>0 and np.isfinite(variance)and result.df_resid>0:
            se=float(np.sqrt(variance));critical=float(stats.t.ppf(.975,result.df_resid))
            total.update(status='ESTIMATED',se_native=se,ci95_native_low=float(beta.sum()-critical*se),
                ci95_native_high=float(beta.sum()+critical*se),p_native=float(2*stats.t.sf(abs(beta.sum()/se),result.df_resid)))
    joint['target_covariance_eigenvalues']=eigenvalues.tolist()if eigenvalues is not None else None
    return dict(target_terms=targets,target_covariance=covariance.tolist(),
        var_current=float(covariance[0,0]),var_lag=float(covariance[1,1]),cov_current_lag=float(covariance[0,1]),
        joint_current_lag_zero=joint,sum_current_lag=total,covariance_repair=False,
        interpretation='Sum is the linear combination of current and one previous-interval coefficients, not a structural long-run multiplier.')


def target_status(coefficients,term):
    row=next((r for r in coefficients if r['term']==term),{})
    if not row:return'NOT_ESTIMATED'
    fields=['beta','se_native','p_native','ci95_native_low','ci95_native_high']
    ok=all(row.get(k)is not None and np.isfinite(row[k])for k in fields)
    return'FINITE'if ok and row['se_native']>0 and 0<=row['p_native']<=1 else'UNDEFINED'


def old_metrics(previous,coefficients):
    rows=[]
    for record in previous['models']:
        item={k:record.get(k)for k in ['model','fund_focus','status','nobs','funds','companies','rsquared_within']}
        selected=coefficients.loc[coefficients.model.eq(record['model'])&coefficients.term.eq(CURRENT)]
        target=selected.iloc[0].to_dict()if len(selected)==1 else{}
        item.update({k:target.get(k)for k in ['beta','se_native','ci95_native_low','ci95_native_high','p_native']})
        rows.append(item)
    return pd.DataFrame(rows)


def render(root,summary,comparison,old,native):
    parts=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>H1: текущий сентимент и предыдущий интервал</title>',
        '<style>body{font:16px system-ui;max-width:1600px;margin:32px auto;padding:0 20px;color:#17212d}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}pre{padding:16px;background:#f5f6f8;overflow:auto;font:12px monospace}a{color:#075ba5}</style>',
        '<h1>H1: текущий NetSentiment и один точный отчётный лаг</h1>',
        '<p>Лаг — сентимент той же компании за непосредственно предыдущий фактический интервал раскрытия этого фонда. Предыдущее закрытие должно совпадать с текущим открытием. Это не сдвиг сохранившихся строк и не обязательно календарный месяц. Отсутствующее покрытие, невалидная классификация или отсутствие предыдущего интервала остаются пропусками.</p>',
        '<p>Для каждой прежней маски обе модели используют одни и те же строки с доступным текущим и лагированным score. CURRENT_COMMON содержит текущий score; CURRENT_PLUS_LAG1 содержит текущий и предыдущий. Старые support-пороги зафиксированы до исключений по лагу и после них не пересчитываются. Старый полный результат показан отдельно.</p>',
        '<p>Y, условные +100% для подтверждённых входов, P99, 16 контролей, fund FE и кластеры фонд/компания не менялись. Временных эффектов/кластеров нет. Показаны обычные двусторонние native p-values. Совместный Wald и сумма коэффициентов используют полную ковариацию; неопределённые тесты не заменены числами.</p>',
        '<h2>Прежние результаты на полных прежних выборках</h2><div class="scroll">'+old.to_html(index=False,border=0,float_format=lambda x:f'{x:.6g}')+'</div>',
        '<h2>Общая выборка: текущий score и текущий плюс лаг</h2><div class="scroll">'+comparison.to_html(index=False,border=0,float_format=lambda x:f'{x:.6g}')+'</div>',
        '<p><a href="coefficients.csv">Все коэффициенты</a> · <a href="distributed_tests.csv">Совместный тест и сумма</a> · <a href="old_metrics.csv">Прежние метрики</a> · <a href="sample_flow.csv">Потеря наблюдений из-за лага</a> · <a href="lag_sample_flow.json">Причины пропусков</a> · <a href="regression_preview.csv">Пример данных</a> · <a href="summary.json">Все метрики</a> · <a href="source_manifest.json">Источники</a></p>',
        '<p>Тип фонда определяется на начало текущего интервала и может меняться. Результат в одной группе не равен результату для всех фондов. Значимый отрицательный коэффициент не подтверждает положительно направленную H1. Лаговая связь не доказывает причинность, переоценку или скорость включения новостей в цены. Совпадающие маски используют один расчёт и не являются независимыми повторениями.</p>']
    for name,text in native.items():parts.append('<h3>'+html.escape(name)+'</h3><pre>'+html.escape(text)+'</pre>')
    parts.append('</html>');(root/'report.html').write_text('\n'.join(parts),encoding='utf-8')


def execute(args):
    if sys.platform!='win32':raise RuntimeError('Actual research execution must run on research-host')
    planpath=Path(args.plan)
    if sha(planpath)!=args.plan_sha:raise ValueError('Frozen plan SHA256 mismatch')
    plan=json.loads(planpath.read_text(encoding='utf-8'))
    if not plan['status'].startswith('FIXED'):raise ValueError('Plan is not final and fixed before coefficients')
    controls=plan['baseline_controls']+plan['macro_controls']
    if controls!=MICRO+MACRO or plan.get('time_FE')or plan.get('cluster_time')or plan.get('clusters')!=['fund_id','company_id']:
        raise ValueError('Frozen estimator conventions changed')
    if plan.get('multiple_testing_adjustments')is not False:raise ValueError('Plan must preserve requested unadjusted inference')
    if plan['required_targets']!=[CURRENT,LAG]:raise ValueError('Unexpected target definitions')
    root=Path(plan['output_root'])
    if any(part.lower()=='databaseresources'for part in root.parts):raise ValueError('Protected output location')
    root.mkdir(parents=True,exist_ok=True)
    if(root/'completion.json').exists():raise FileExistsError('Completed version cannot be overwritten')
    paths={key:plan[key]for key in SOURCE_KEYS};sources={}
    for key,path in paths.items():
        actual=sha(path)
        if actual!=plan['expected_source_sha256'].get(key):raise ValueError('Frozen source mismatch/missing hash: '+key)
        sources[key]=dict(path=path,sha256=actual,size_bytes=Path(path).stat().st_size)
    base=load_module(paths['base_helper'],'h1_lag_base');macro=load_module(paths['macro_runner_helper'],'h1_lag_macro')
    focus=load_module(paths['focus_helper'],'h1_lag_focus_masks')
    if focus.planned_models()!=plan['base_models']or plan['planned_models']!=[name+'__'+variant for name in plan['base_models']for variant in VARIANTS]:
        raise ValueError('Frozen 36/72 model definitions changed')
    previous=json.loads(Path(paths['focus_summary']).read_text(encoding='utf-8'))
    if previous['specification']['expected_source_sha256']['input_panel']!=plan['expected_source_sha256']['input_panel']:
        raise ValueError('Prior masks come from a different primary source')
    lock=root/'run.lock';descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(descriptor,json.dumps(dict(pid=os.getpid(),plan_sha256=args.plan_sha)).encode());os.close(descriptor)
    started=time.time()
    def save(name,value):base.save(root/name,value)
    def checkpoint(state,**extra):
        record=dict(state=state,pid=os.getpid(),elapsed_seconds=time.time()-started,**extra)
        save('checkpoint.json',record);print(json.dumps(record),flush=True)
    try:
        save('frozen_plan.json',plan);checkpoint('READING_FIXED_SOURCES')
        frame=ordered(macro.read_frame(paths['input_panel']))
        if len(frame)!=plan['expected_primary_rows']or not macro.finite_rows(frame,['quantity_change_pct',CURRENT]+controls).all():
            raise AssertionError('Frozen primary rows/complete cases changed')
        lagmap=macro.read_frame(paths['lag_map']);frame=join_lag(frame,lagmap);del lagmap
        old_coefs=pd.read_csv(paths['focus_coefficients']);old=old_metrics(previous,old_coefs)
        old.to_csv(root/'old_metrics.csv',index=False,encoding='utf-8-sig')
        saved_support=pd.read_csv(paths['focus_support'],dtype={'fund_id':str})
        source_support=focus.focus_support_table(frame)
        joined_support=saved_support.merge(source_support,on=['fund_focus','fund_id'],how='outer',suffixes=('_saved','_source'),indicator=True,validate='one_to_one')
        if not joined_support['_merge'].eq('both').all():raise AssertionError('Frozen focus-support fund keys changed')
        for c in ['fund_rows','fund_intervals','fund_end_dates']:
            if not joined_support[c+'_saved'].eq(joined_support[c+'_source']).all():raise AssertionError('Support was not fixed on the old focus baseline')
        masks=focus.model_samples(frame,saved_support)
        lag_columns=[c for c in frame if c.startswith('lag_')or c in ['row_id','fund_id','security_id','company_id','interval_id','start_date','end_date',LAG,'complete_current_lag1']]
        frame[lag_columns].to_parquet(root/'lag_inclusion_decisions.parquet',index=False,compression='zstd')
        global_flow=dict(source_rows=len(frame),eligible_lag_rows=int(frame.complete_current_lag1.sum()),
            missing_lag_rows=int((~frame.complete_current_lag1).sum()),lag_reason_rows=frame.lag_reason.value_counts(dropna=False).to_dict(),
            source_current_values_unchanged=True,lag_map_one_to_one=True,old_support_qualifiers_unchanged=True,
            lag_is_previous_actual_report_interval=True,no_surviving_row_shift=True,no_missing_score_fill=True)
        save('lag_sample_flow.json',global_flow)
        records=[];coefs=[];samples={};native={};tests={};catalog=[];flows=[];representatives={};sample_files={};fits=0
        (root/'samples').mkdir(exist_ok=True)
        for oldname,old_sample,group,minrows,minintervals in masks:
            old_meta=previous['samples'][oldname]
            if len(old_sample)!=old_meta['rows']or row_hash(old_sample)!=old_meta['row_hash']:
                raise AssertionError('Old row mask changed: '+oldname)
            common=ordered(old_sample.loc[old_sample.complete_current_lag1])
            common_hash=row_hash(common)
            achieved=focus.support_table(common)
            achieved_support=dict(
                achieved_min_fund_rows=int(achieved.fund_rows.min())if len(achieved)else None,
                achieved_median_fund_rows=float(achieved.fund_rows.median())if len(achieved)else None,
                achieved_min_fund_intervals=int(achieved.fund_intervals.min())if len(achieved)else None,
                achieved_median_fund_intervals=float(achieved.fund_intervals.median())if len(achieved)else None,
                achieved_min_fund_end_dates=int(achieved.fund_end_dates.min())if len(achieved)else None)
            if common_hash not in sample_files:
                path=root/'samples'/(oldname+'__COMMON.parquet');common.to_parquet(path,index=False,compression='zstd')
                sample_files[common_hash]=dict(parquet=str(path),rows=len(common),row_hash=common_hash,exported_columns=list(common.columns),representative_base_model=oldname)
            flow=dict(base_model=oldname,fund_focus=group,old_rows=len(old_sample),common_rows=len(common),
                dropped_rows=len(old_sample)-len(common),old_funds=int(old_sample.fund_id.nunique()),common_funds=int(common.fund_id.nunique()),
                old_companies=int(old_sample.company_id.nunique()),common_companies=int(common.company_id.nunique()),
                old_row_hash=old_meta['row_hash'],common_row_hash=common_hash,minimum_support_not_reapplied=True,
                fixed_min_fund_rows=minrows,fixed_min_fund_intervals=minintervals,
                first_start_date=str(common.start_date.min())if len(common)else None,
                last_end_date=str(common.end_date.max())if len(common)else None,**achieved_support,
                dropped_reason_counts=old_sample.loc[~old_sample.complete_current_lag1,'lag_reason'].value_counts(dropna=False).to_dict())
            flows.append(flow)
            for variant in VARIANTS:
                name=oldname+'__'+variant;distributed=variant=='CURRENT_PLUS_LAG1';key=(variant,common_hash)
                samples[name]=copy.deepcopy(sample_files[common_hash])
                checkpoint('PROCESSING_MODEL',model=name,rows=len(common),funds=int(common.fund_id.nunique()))
                alias=representatives.get(key,{}).get('model')
                if alias:
                    stored=representatives[key];record=copy.deepcopy(stored['record']);coefficients=copy.deepcopy(stored['coefs'])
                    text=stored['text'];htmltext=stored['html'];test=copy.deepcopy(stored['tests']);origin='CURRENT_EXACT_COMMON_SAMPLE_ALIAS'
                else:
                    issue=macro.model_status(common,plan['minimum_model_rows'])
                    if issue:
                        record=dict(status='NOT_ESTIMABLE',reason=issue,nobs=len(common),funds=int(common.fund_id.nunique()),companies=int(common.company_id.nunique()))
                        coefficients=[];text='NOT_ESTIMABLE: '+issue;htmltext='<pre>'+html.escape(text)+'</pre>';test=None;origin='NOT_ESTIMABLE'
                    else:
                        fits+=1
                        try:
                            fitted,result,terms,dropped=fit_native(common,controls,base,distributed)
                            record,raw=base.describe_result(fitted,result,name,'quantity_change_pct',terms,dropped)
                            record['covariance_options']['cluster_time']=False
                            coefficients=[{field:r.get(field)for field in COEF_FIELDS}for r in raw]
                            test=distributed_tests(result)if distributed else None
                            joint=result.f_statistic_robust
                            record['overall_robust_F_inference']=dict(statistic=float(joint.stat),native_p=float(joint.pval),
                                status='VALID'if np.isfinite(joint.stat)and joint.stat>=0 and np.isfinite(joint.pval)else'UNDEFINED_COVARIANCE')
                            text=str(result.summary);htmltext=result.summary.as_html();origin='NEW_UNIQUE_COMMON_SAMPLE_FIT'
                        except Exception as error:
                            record=dict(status='NOT_ESTIMABLE',reason=f'{type(error).__name__}: {error}',nobs=len(common),
                                funds=int(common.fund_id.nunique()),companies=int(common.company_id.nunique()))
                            coefficients=[];text=record['reason'];htmltext='<pre>'+html.escape(text)+'</pre>';test=None;origin='NOT_ESTIMABLE'
                    representatives[key]=dict(model=name,record=copy.deepcopy(record),coefs=copy.deepcopy(coefficients),text=text,html=htmltext,tests=copy.deepcopy(test))
                record.update(model=name,base_model=oldname,variant=variant,fund_focus=group,result_origin=origin,exact_common_sample_alias_of=alias,
                    net_inference_status=target_status(coefficients,CURRENT),lag_inference_status=target_status(coefficients,LAG)if distributed else'NOT_INCLUDED',
                    support_qualifiers_fixed_before_lag_drop=True,old_nobs=len(old_sample),n_lag_missing=len(old_sample)-len(common),
                    fixed_min_fund_rows=minrows,fixed_min_fund_intervals=minintervals,**achieved_support,
                    first_start_date=str(common.start_date.min())if len(common)else None,last_end_date=str(common.end_date.max())if len(common)else None)
                for row in coefficients:row['model']=name
                if distributed:
                    tests[name]=test if test is not None else dict(status='NOT_ESTIMABLE',reason=record.get('reason'))
                records.append(record);coefs.extend(coefficients);native[name]=text
                catalog.append(dict(model=name,base_model=oldname,variant=variant,old_row_hash=old_meta['row_hash'],
                    common_row_hash=common_hash,old_rows=len(old_sample),rows=len(common),alias_of=alias,status=record['status']))
                (root/(name+'.summary.txt')).write_text(text,encoding='utf-8')
                (root/(name+'.summary.html')).write_text(htmltext,encoding='utf-8')
            if samples[oldname+'__CURRENT_COMMON']['row_hash']!=samples[oldname+'__CURRENT_PLUS_LAG1']['row_hash']:
                raise AssertionError('Paired variants changed sample')
        pd.DataFrame(coefs).to_csv(root/'coefficients.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame([{k:v for k,v in flow.items()if k!='dropped_reason_counts'}for flow in flows]).to_csv(root/'sample_flow.csv',index=False,encoding='utf-8-sig')
        save('sample_flow.json',flows);save('distributed_tests.json',tests);save('mask_catalog.json',catalog)
        comparison=[];test_rows=[]
        for record in records:
            item={k:record.get(k)for k in ['model','base_model','variant','fund_focus','status','nobs','funds','companies','rsquared_within',
                'result_origin','exact_common_sample_alias_of','net_inference_status','lag_inference_status',
                'first_start_date','last_end_date','old_nobs','n_lag_missing','fixed_min_fund_rows','fixed_min_fund_intervals',
                'achieved_min_fund_rows','achieved_median_fund_rows','achieved_min_fund_intervals','achieved_median_fund_intervals','achieved_min_fund_end_dates']}
            for term,prefix in [(CURRENT,'current'),(LAG,'lag1')]:
                target=next((r for r in coefs if r['model']==record['model']and r['term']==term),{})
                item.update({prefix+'_'+k:target.get(k)for k in ['beta','se_native','ci95_native_low','ci95_native_high','p_native']})
            test=tests.get(record['model'],{});joint=test.get('joint_current_lag_zero',{});total=test.get('sum_current_lag',{})
            item.update(joint_status=joint.get('status'),joint_p=joint.get('p_value'),sum_status=total.get('status'),
                sum_beta=total.get('beta'),sum_se=total.get('se_native'),sum_ci95_low=total.get('ci95_native_low'),
                sum_ci95_high=total.get('ci95_native_high'),sum_p=total.get('p_native'))
            comparison.append(item)
            if record['variant']=='CURRENT_PLUS_LAG1':
                test_rows.append(dict(model=record['model'],joint_status=joint.get('status'),joint_statistic=joint.get('statistic'),
                    joint_p=joint.get('p_value'),sum_status=total.get('status'),sum_beta=total.get('beta'),sum_se=total.get('se_native'),
                    sum_ci95_low=total.get('ci95_native_low'),sum_ci95_high=total.get('ci95_native_high'),sum_p=total.get('p_native'),
                    var_current=test.get('var_current'),var_lag=test.get('var_lag'),cov_current_lag=test.get('cov_current_lag')))
        comparison=pd.DataFrame(comparison);comparison.to_csv(root/'comparison.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(test_rows).to_csv(root/'distributed_tests.csv',index=False,encoding='utf-8-sig')
        preview=[c for c in ['row_id','fund_name','fund_id','fund_focus','asset_name','security_id','company','company_id','start_date','end_date',
            'lag_start_date','lag_end_date','net_sentiment',LAG,'lag_reason','lag_source_kind','quantity_open','quantity_close','quantity_change_pct']if c in frame]
        frame.loc[frame.complete_current_lag1,preview].head(200).to_csv(root/'regression_preview.csv',index=False,encoding='utf-8-sig')
        unchanged={key:sha(path)==sources[key]['sha256']for key,path in paths.items()}
        if not all(unchanged.values()):raise AssertionError('A frozen input changed during execution')
        save('source_manifest.json',dict(sources=sources,source_hashes_unchanged_after_run=unchanged,plan_sha256=args.plan_sha,runner_sha256=sha(__file__)))
        summary=dict(status='COMPLETE',specification=plan,plan_sha256=args.plan_sha,models=records,samples=samples,
            distributed_tests=tests,mask_catalog=catalog,lag_sample_flow=global_flow,model_sample_flow=flows,
            new_unique_variant_fit_attempts=fits,unique_variant_masks=len(representatives),unique_common_samples=len(sample_files),
            elapsed_seconds=time.time()-started,independent_QA='PENDING_SEPARATE_REVIEW',
            inference='Unadjusted two-sided native coefficient tests; joint chi-square2 and sum with full covariance',
            support_qualification_fixed_before_lag_drop=True,old_metrics_file='old_metrics.csv')
        save('summary.json',summary);render(root,summary,comparison,old,native)
        checkpoint('COMPLETE',new_unique_variant_fit_attempts=fits,labels=len(records))
        save('completion.json',dict(status='COMPLETE',plan_sha256=args.plan_sha,elapsed_seconds=time.time()-started,
            finished_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        return 0
    except Exception as error:
        save('failure.json',dict(status='FAILED',message=str(error),traceback=traceback.format_exc(),plan_sha256=args.plan_sha));raise
    finally:lock.unlink(missing_ok=True)


def fixture(args):
    base=load_module(Path(__file__).with_name('run_h1_net_fund_fe_20261030.py'),'h1_lag_fixture_base')
    rng=np.random.default_rng(20261103);rows=[]
    for fund in range(18):
        for interval in range(10):
            for company in range(6):
                net,lag,control,noise=rng.normal(size=4)
                rows.append(dict(fund_id=str(fund),company_id=str(company),security_id=str(company),
                    interval_id=f'i{fund}-{interval}',row_id=f'r{fund}-{interval}-{company}',
                    start_date=pd.Timestamp('2020-01-31')+pd.offsets.MonthEnd(interval),
                    end_date=pd.Timestamp('2020-01-31')+pd.offsets.MonthEnd(interval+1),
                    opening_report_id=f'r{fund}-{interval}',net_sentiment=net,net_sentiment_lag1=lag,control=control,absorbed=float(fund),
                    quantity_change_pct=.7*net+1.2*lag+.3*control+.2*fund+noise,
                    opening_passive_explicit=False,closing_passive_explicit=False,closing_full_portfolio=True))
    sample,result,terms,dropped=fit_native(pd.DataFrame(rows),['control','absorbed'],base,True)
    within=sample[['quantity_change_pct']+terms]-sample[['quantity_change_pct']+terms].groupby(sample.fund_id).transform('mean')
    independent=np.linalg.lstsq(within[terms],within.quantity_change_pct,rcond=None)[0]
    covariance=base.cluster_covariance_fixture(sample,within[terms].to_numpy(),result.resids.to_numpy(),result.df_resid)
    assert np.allclose(result.params[terms],independent,rtol=1e-10,atol=1e-10)
    assert np.allclose(result.cov.loc[terms,terms],covariance,rtol=1e-10,atol=1e-10)
    tests=distributed_tests(result);block=np.asarray(tests['target_covariance']);b=result.params.loc[[CURRENT,LAG]].to_numpy()
    assert tests['joint_current_lag_zero']['status']=='ESTIMATED'
    assert np.isclose(tests['joint_current_lag_zero']['statistic'],b@np.linalg.inv(block)@b,rtol=1e-10)
    assert np.isclose(tests['sum_current_lag']['variance'],block[0,0]+block[1,1]+2*block[0,1])
    assert np.isclose(tests['sum_current_lag']['beta'],b.sum())
    baseline,baseline_result,baseline_terms,_=fit_native(sample,['control','absorbed'],base,False)
    assert row_hash(baseline)==row_hash(sample)and LAG not in baseline_terms
    singular=sample.copy();singular[LAG]=singular[CURRENT]
    try:fit_native(singular,['control','absorbed'],base,True)
    except ValueError as error:assert'Required sentiment target'in str(error)
    else:raise AssertionError('Required collinear lag was silently removed')
    source=sample.head(2).drop(columns=[LAG]).copy().reset_index(drop=True)
    source['row_id']=source.row_id.astype('string')
    mapping=pd.DataFrame(dict(row_id=source.row_id,current_interval_id=source.interval_id,fund_id=source.fund_id,company_id=source.company_id,
        current_start_date=source.start_date,current_end_date=source.end_date,lag_interval_id=['previous','previous'],
        lag_start_date=[pd.Timestamp('2019-12-31')]*2,lag_end_date=source.start_date,
        lag_closing_report_id=source.opening_report_id,lag_company_id=source.company_id,lag_company_period_id=['w0','w1'],
        net_sentiment_lag1=[np.nan,-.4],lag_eligible_primary=[False,True],lag_reason=['no_relevant','eligible'],lag_source_kind=['actual','actual'],lag_n_relevant=[0,5]))
    mapping['row_id']=mapping.row_id.astype(object)
    joined=join_lag(source,mapping.iloc[::-1].reset_index(drop=True))
    assert len(joined)==2 and joined.loc[1,LAG]==-.4 and pd.isna(joined.loc[0,LAG])
    assert joined.complete_current_lag1.tolist()==[False,True]
    first=mapping.copy()
    for column in ['lag_interval_id','lag_start_date','lag_end_date','lag_closing_report_id','lag_company_id','lag_company_period_id']:
        first.loc[0,column]=None
    assert not join_lag(source,first).loc[0,'complete_current_lag1']
    neutral=mapping.copy();neutral.loc[1,LAG]=0.
    assert join_lag(source,neutral).loc[1,LAG]==0.
    filled=mapping.copy();filled.loc[0,LAG]=0.
    try:join_lag(source,filled)
    except AssertionError:pass
    else:raise AssertionError('Ineligible missing lag was accepted as neutral zero')
    changed=mapping.copy();changed.loc[1,'lag_end_date']=pd.Timestamp('2019-11-30')
    try:join_lag(source,changed)
    except AssertionError:pass
    else:raise AssertionError('A skipped interval was accepted')
    print(json.dumps(dict(status='PASS',scope='synthetic_only',rows=len(sample),paired_samples=True,
        native_within_OLS_and_covariance=True,mandatory_current_lag=True,joint_crosscov_sum=True,
        one_to_one_lag_source_preservation=True,no_missing_fill=True)))
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan');parser.add_argument('--plan-sha')
    parser.add_argument('--fixture-only',action='store_true');args=parser.parse_args()
    if args.fixture_only:return fixture(args)
    if not args.plan or not args.plan_sha:parser.error('--plan and --plan-sha are required')
    return execute(args)


if __name__=='__main__':raise SystemExit(main())
