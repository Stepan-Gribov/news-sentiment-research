"""Supplementary quantity-direction cash-excess future-price panel models.

This position-level proxy is not fund alpha or the original fund NTS model.
Actual research execution requires a final frozen plan on research-host.
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


OUTCOME='dar_q_cash_price_pp'
CURRENT='align_current'
LAG='align_lag1'
VARIANTS=['CURRENT_COMMON','CURRENT_PLUS_LAG1']
MICRO=['opening_weight','closing_log_NAV','closing_HHI','log_news_count_interval','log_interval_days','bond_indicator']
MACRO=['Imoex','imoex_ret','Ruonia','USD_RUB','RVI','Flow_billion_RUB','unique_funds','dummy_mar2014','dummy_feb2022','dummy_sep2018']
SOURCE_KEYS=['input_panel','lag_map','lag_helper','focus_summary','focus_support','focus_helper','cutoff_reference_sample',
             'base_helper','h2_helper','future_outcomes']
COEF_FIELDS=['model','outcome','term','beta','se_native','t_native','p_native','ci95_native_low','ci95_native_high']


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,str(path));module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def ordered(frame):return frame.sort_values(['fund_id','end_date','security_id','interval_id'],kind='stable').reset_index(drop=True)


def row_hash(frame):return hashlib.sha256('\n'.join(frame.row_id.astype(str)).encode()).hexdigest()


def truth(series):return series.eq(True).fillna(False).astype(bool)


def join_future(source,future,horizon):
    required=['row_id','horizon','trade_sign','p0_price_rub','p0_quote_date','p0_quote_age_days','future_price_rub',
        'future_quote_date','future_target_date','future_quote_age_days','future_price_return','cash_return',OUTCOME,
        'outcome_eligible','outcome_reason',CURRENT,LAG]
    if set(required)-set(future):raise ValueError('Future-outcome source missing: '+str(sorted(set(required)-set(future))))
    part=future.loc[future.horizon.eq(horizon)]
    if part.row_id.duplicated().any()or set(part.row_id)!=set(source.row_id):raise AssertionError('Future outcomes do not cover source exactly once per horizon')
    aligned=part.set_index('row_id').reindex(source.row_id)
    for column in set(source)&set(part)-{'row_id'}:
        original=source[column].reset_index(drop=True);other=aligned[column].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(original):
            equal=np.array_equal(pd.to_numeric(original,errors='coerce').to_numpy(dtype=float,na_value=np.nan),
                                 pd.to_numeric(other,errors='coerce').to_numpy(dtype=float,na_value=np.nan),equal_nan=True)
        elif pd.api.types.is_datetime64_any_dtype(original):
            equal=np.array_equal(pd.to_datetime(original).to_numpy(dtype='datetime64[ns]'),
                                 pd.to_datetime(other).to_numpy(dtype='datetime64[ns]'),equal_nan=True)
        else:equal=original.astype('string').fillna('<NA>').equals(other.astype('string').fillna('<NA>'))
        if not equal:raise AssertionError('Future source changed an original column: '+column)
    d=source.copy()
    for column in part:
        if column in source or column in [CURRENT,LAG]:continue
        values=aligned[column].reset_index(drop=True);values.index=source.index;d[column]=values
    for column in source:
        if not d[column].equals(source[column]):raise AssertionError('Future join changed original values or dtype: '+column)
    delta=pd.to_numeric(source.delta_quantity_units,errors='coerce');sign=np.sign(delta)
    if not np.array_equal(pd.to_numeric(d.trade_sign,errors='coerce'),sign,equal_nan=True):raise AssertionError('Trade direction differs from frozen estimated quantity change')
    known_nonzero=pd.Series(np.isfinite(delta),index=source.index)&delta.ne(0)
    d[CURRENT]=(sign*pd.to_numeric(source.net_sentiment,errors='coerce')).where(known_nonzero)
    d[LAG]=(sign*pd.to_numeric(source.net_sentiment_lag1,errors='coerce')).where(known_nonzero&truth(source.complete_current_lag1))
    for column in [CURRENT,LAG]:
        if not np.allclose(d[column],pd.to_numeric(aligned[column],errors='coerce').to_numpy(float),rtol=0,atol=0,equal_nan=True):
            raise AssertionError('Alignment proxy differs from sign(deltaQ) times frozen news score')
    eligible=truth(d.outcome_eligible)
    values=d[['p0_price_rub','future_price_rub','future_price_return','cash_return',OUTCOME,'trade_sign']].apply(pd.to_numeric,errors='coerce')
    if not np.isfinite(values.loc[eligible]).all().all()or not values.loc[eligible,['p0_price_rub','future_price_rub']].gt(0).all().all():
        raise AssertionError('Eligible future outcome contains invalid numeric data')
    if not delta.loc[eligible].ne(0).all()or not np.isfinite(delta.loc[eligible]).all():raise AssertionError('Eligible outcome has no nonzero known quantity change')
    expected_target=pd.to_datetime(source.end_date)+pd.DateOffset(months=horizon)
    target=pd.to_datetime(d.future_target_date);quote=pd.to_datetime(d.future_quote_date);origin=pd.to_datetime(source.end_date)
    if not np.array_equal(target.to_numpy(dtype='datetime64[ns]'),expected_target.to_numpy(dtype='datetime64[ns]')):
        raise AssertionError('Future targets are not calendar DateOffset horizons')
    age=(target-quote).dt.days
    p0date=pd.to_datetime(d.p0_quote_date);p0age=(origin-p0date).dt.days
    if not(quote.loc[eligible].gt(origin.loc[eligible])&quote.loc[eligible].le(target.loc[eligible])&age.loc[eligible].between(0,7)).all():
        raise AssertionError('Eligible future quote violates the forward as-of/age rule')
    if not(p0date.loc[eligible].le(origin.loc[eligible])&p0age.loc[eligible].between(0,7)).all():
        raise AssertionError('Eligible origin quote violates as-of/age rule')
    if not np.array_equal(age.loc[eligible].to_numpy(float),pd.to_numeric(d.loc[eligible,'future_quote_age_days']).to_numpy(float)):
        raise AssertionError('Stored future quote ages disagree with dates')
    if not np.array_equal(p0age.loc[eligible].to_numpy(float),pd.to_numeric(d.loc[eligible,'p0_quote_age_days']).to_numpy(float)):
        raise AssertionError('Stored origin quote ages disagree with dates')
    calculated_return=values.future_price_rub/values.p0_price_rub-1
    if not np.allclose(calculated_return.loc[eligible],values.loc[eligible,'future_price_return'],rtol=1e-11,atol=1e-12):
        raise AssertionError('Stored price return differs from normalized price ratio')
    calculated=100*values.trade_sign*(values.future_price_return-values.cash_return)
    if not np.allclose(calculated.loc[eligible],values.loc[eligible,OUTCOME],rtol=1e-11,atol=1e-10):
        raise AssertionError('Supplementary outcome formula disagrees with inputs')
    d['complete_dar_and_lag']=eligible&truth(d.complete_current_lag1)&np.isfinite(d[CURRENT])&np.isfinite(d[LAG])
    return d


def fit_native(sample,controls,h2,distributed):
    from linearmodels.panel import PanelOLS
    sample=ordered(sample);targets=[CURRENT,LAG]if distributed else[CURRENT]
    terms,dropped=h2.select_regressors(sample,targets,controls)
    index=pd.MultiIndex.from_arrays([sample.fund_id.astype(str),pd.to_datetime(sample.end_date)],names=['fund_id','report_date'])
    x=sample[terms].astype(float).copy();x.index=index;x.insert(0,'const',1.)
    y=pd.Series(sample[OUTCOME].to_numpy(float),index=index,name=OUTCOME)
    clusters=pd.DataFrame({'fund_cluster':pd.Categorical(sample.fund_id.astype(str)).codes,
        'company_cluster':pd.Categorical(sample.company_id.astype(str)).codes},index=index)
    result=PanelOLS(y,x,entity_effects=True,time_effects=False,other_effects=None,singletons=True,
        drop_absorbed=False,check_rank=True).fit(cov_type='clustered',clusters=clusters,cluster_time=False,
        cluster_entity=False,group_debias=True,debiased=True,auto_df=False,count_effects=True)
    if result.nobs!=len(sample)or not result.resids.index.equals(index):raise AssertionError('Estimator changed rows/order')
    return sample,result,terms,dropped


def outcome_scenario_mask(frame,scenario,horizon):
    quantile=scenario.get('outcome_abs_quantile')
    if quantile is None:return pd.Series(True,index=frame.index)
    value=float(scenario['outcome_abs_cutoffs_by_horizon'][str(horizon)])
    if not np.isfinite(value)or value<0:raise ValueError('Invalid fixed absolute-DAR threshold')
    return pd.to_numeric(frame[OUTCOME],errors='coerce').abs().le(value)


def distributed_tests(result,h2):
    targets=[CURRENT,LAG];beta=result.params.loc[targets].to_numpy(float);cov=result.cov.loc[targets,targets].to_numpy(float)
    total=h2.linear_contrast(beta,cov,[1.,1.],result.df_resid)
    joint=dict(status='UNDEFINED_COVARIANCE',statistic=None,p_value=None,df=2,distribution='chi2(2)',target_covariance_eigenvalues=None)
    if np.isfinite(cov).all():
        eigenvalues=np.linalg.eigvalsh((cov+cov.T)/2);joint['target_covariance_eigenvalues']=eigenvalues.tolist()
        if eigenvalues.min()>0:
            restriction=np.zeros((2,len(result.params)))
            for j,name in enumerate(targets):restriction[j,result.params.index.get_loc(name)]=1.
            test=result.wald_test(restriction=restriction,value=np.zeros(2))
            if np.isfinite(test.stat)and test.stat>=0 and np.isfinite(test.pval):
                joint.update(status='ESTIMATED',statistic=float(test.stat),p_value=float(test.pval),df=int(test.df),distribution=str(test.dist_name))
    return dict(target_terms=targets,target_covariance=cov.tolist(),var_current=float(cov[0,0]),var_lag=float(cov[1,1]),
        cov_current_lag=float(cov[0,1]),joint_current_lag_zero=joint,sum_current_lag=total,covariance_repair=False,
        interpretation='Supplementary price-mechanism alignment association; not fund alpha or structural long-run performance.')


def render(root,summary,comparison,native):
    parts=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Дополнительный DAR по направлению позиции</title>',
        '<style>body{font:16px system-ui;max-width:1600px;margin:32px auto;padding:0 20px;color:#17212d}.scroll{overflow:auto}table{border-collapse:collapse;font-size:12px}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}pre{background:#f5f6f8;padding:16px;overflow:auto;font:12px monospace}</style>',
        '<h1>Дополнительный DAR: направление изменения позиции и будущая цена</h1>',
        '<p>Основная H3 с fund alpha остаётся неоценимой на подтверждённых данных. Эта отдельная позиционная проверка её не заменяет. Y=100×sign(ΔQ)×(будущая ценовая доходность−доходность денежного рынка). X — sign(ΔQ)×NetSentiment текущего и точного предыдущего отчётного интервала; это proxy согласованности новостей и направления позиции, а не фондовый NTS.</p>',
        '<p>Горизонты — 1/2/3 календарных месяца. Будущая котировка строго после текущего отчёта, не позже целевой даты и не старше 7 дней. Денежный benchmark — отношение официального накопленного индекса RUONIA на целевую и текущую даты, shift=0. Цены чистые: без купонов, дивидендов, реинвестирования и транзакционных расходов. Результат не является полной доходностью или факторной alpha.</p>',
        '<p>Все прежние fund-focus/support маски сохранены до потерь данных. Зафиксированы пять сценариев: Q99_RAW как диагностика без отсечения DAR; Q99_D99; Q975_D975; Q95_D95; Q90_D90. Q относится к прежнему изменению количества, D — к абсолютному будущему DAR. Пороги D отдельно по горизонту рассчитаны один раз по уникальным ценовым окнам в существующей Q99 основе до дополнительных групповых, лаговых и пороговых фильтров. Равные порогу строки сохраняются. Отсечение будущего outcome меняет исследуемую выборку и не доказывает исправления всех корпоративных ценовых разрывов.</p>',
        '<p>Обе модели пары используют одинаковые строки с доступным лагом и будущим outcome. Fund FE и кластеры фонд/компания сохранены, временных эффектов нет. Приведены обычные двусторонние p-values и все модели; значимый отрицательный коэффициент не переименовывается в поддержку положительного механизма. Порог Q сохраняет входы/выходы, но отдельный D-фильтр может удалять их по будущему outcome.</p>',
        '<div class="scroll">'+comparison.to_html(index=False,border=0,float_format=lambda x:f'{x:.6g}')+'</div>',
        '<p><a href="coefficients.csv">Все коэффициенты</a> · <a href="distributed_tests.json">Сумма и совместный тест</a> · <a href="sample_flow.csv">Выборки</a> · <a href="outcome_audit.json">Данные и экстремумы</a> · <a href="thresholds.json">Пороги количества</a> · <a href="summary.json">Все метрики</a></p>']
    for name,text in native.items():parts.append('<details><summary>'+html.escape(name)+'</summary><pre>'+html.escape(text)+'</pre></details>')
    parts.append('</html>');(root/'report.html').write_text('\n'.join(parts),encoding='utf-8')


def execute(args):
    if sys.platform!='win32':raise RuntimeError('Actual research execution must run on research-host')
    planpath=Path(args.plan)
    if sha(planpath)!=args.plan_sha:raise ValueError('Frozen plan SHA256 mismatch')
    plan=json.loads(planpath.read_text(encoding='utf-8'))
    if not plan['status'].startswith('FIXED'):raise ValueError('Supplementary DAR plan is not fixed before coefficients')
    controls=plan['baseline_controls']+plan['macro_controls']
    if controls!=MICRO+MACRO or plan['variants']!=VARIANTS or plan['horizons']!=[1,2,3]:raise ValueError('Frozen model grid changed')
    if plan['dependent']!=OUTCOME or plan['current_target']!=CURRENT or plan['lag_target']!=LAG:raise ValueError('Supplementary variable definitions changed')
    if plan.get('multiple_testing_adjustments')is not False or plan.get('time_FE')or plan.get('cluster_time')or plan.get('clusters')!=['fund_id','company_id']:
        raise ValueError('Frozen covariance/inference conventions changed')
    if plan.get('rf_index_shift_days')!=0 or plan.get('not_a_replacement_for_core_H3')is not True:raise ValueError('Supplement status or cash convention changed')
    if [c['tag']for c in plan['cutoffs']]!=['Q99_RAW','Q99_D99','Q975_D975','Q95_D95','Q90_D90']:
        raise ValueError('The five fixed quantity/outcome scenarios changed')
    if [c.get('outcome_abs_quantile')for c in plan['cutoffs']]!=[None,.99,.975,.95,.9]:
        raise ValueError('Fixed absolute-outcome quantiles changed')
    root=Path(plan['output_root'])
    if any(p.lower()=='databaseresources'for p in root.parts):raise ValueError('Protected output location')
    root.mkdir(parents=True,exist_ok=True)
    if(root/'completion.json').exists():raise FileExistsError('Completed results may not be overwritten')
    paths={key:plan[key]for key in SOURCE_KEYS}
    for key in plan['expected_source_sha256']:
        if key not in paths and isinstance(plan.get(key),str):paths[key]=plan[key]
    sources={}
    for key,path in paths.items():
        actual=sha(path)
        if actual!=plan['expected_source_sha256'].get(key):raise ValueError('Frozen source mismatch/missing hash: '+key)
        sources[key]=dict(path=path,sha256=actual,size_bytes=Path(path).stat().st_size)
    base=load_module(paths['base_helper'],'dar_source_base');lag=load_module(paths['lag_helper'],'dar_source_lag')
    focus=load_module(paths['focus_helper'],'dar_source_focus');h2=load_module(paths['h2_helper'],'dar_source_h2_utilities')
    expected=[cut['tag']+'__M'+str(h)+'__'+name+'__'+variant for cut in plan['cutoffs']for h in plan['horizons']for name in plan['base_models']for variant in VARIANTS]
    if plan['planned_models']!=expected or plan['base_models']!=focus.planned_models():raise ValueError('Frozen model IDs changed')
    previous=json.loads(Path(paths['focus_summary']).read_text(encoding='utf-8'))
    if previous['specification']['expected_source_sha256']['input_panel']!=plan['expected_source_sha256']['input_panel']:
        raise ValueError('Old support masks use a different primary source')
    lock=root/'run.lock';descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(descriptor,json.dumps(dict(pid=os.getpid(),plan_sha256=args.plan_sha)).encode());os.close(descriptor)
    started=time.time()
    def save(name,value):base.save(root/name,value)
    def checkpoint(state,**extra):
        record=dict(state=state,pid=os.getpid(),elapsed_seconds=time.time()-started,**extra)
        save('checkpoint.json',record);print(json.dumps(record),flush=True)
    try:
        save('frozen_plan.json',plan);checkpoint('READING_FROZEN_SOURCES')
        source=ordered(pd.read_parquet(paths['input_panel']))
        if len(source)!=plan.get('expected_primary_rows',221899):raise AssertionError('Frozen primary row count changed')
        source=lag.join_lag(source,pd.read_parquet(paths['lag_map']))
        reference=ordered(pd.read_parquet(paths['cutoff_reference_sample']))
        cut_masks,thresholds=h2.cutoff_masks(source,reference,plan['cutoffs']);save('thresholds.json',thresholds)
        support=pd.read_csv(paths['focus_support'],dtype={'fund_id':str})
        outcomes=pd.read_parquet(paths['future_outcomes'])
        if outcomes.duplicated(['row_id','horizon']).any()or set(outcomes.horizon)!=set(plan['horizons']):raise AssertionError('Future row/horizon keys invalid')
        records=[];coefficients=[];tests={};samples={};catalog=[];native={};representatives={};sample_files={};flows=[];audits=[];attempts=0
        (root/'samples').mkdir(exist_ok=True)
        for horizon in plan['horizons']:
            frame=join_future(source,outcomes,horizon)
            values=pd.to_numeric(frame.loc[truth(frame.outcome_eligible),OUTCOME],errors='coerce')
            audits.append(dict(horizon=horizon,rows=len(frame),eligible_future_outcomes=int(truth(frame.outcome_eligible).sum()),
                eligible_with_lag=int(frame.complete_dar_and_lag.sum()),outcome_min=float(values.min())if len(values)else None,
                outcome_max=float(values.max())if len(values)else None,abs_outcome_over_1000pp=int(values.abs().gt(1000).sum()),
                outcome_reason_rows=frame.outcome_reason.value_counts(dropna=False).to_dict(),raw_future_outcomes_preserved=True))
            old_masks=focus.model_samples(frame,support)
            for name,part,_,_,_ in old_masks:
                prior=previous['samples'][name]
                if len(part)!=prior['rows']or row_hash(part)!=prior['row_hash']:raise AssertionError('Old support mask changed: '+name)
            for cutoff in plan['cutoffs']:
                tag=cutoff['tag'];keep_ids=set(source.loc[cut_masks[tag],'row_id'])
                outcome_keep=outcome_scenario_mask(frame,cutoff,horizon)
                outcome_keep_ids=set(frame.loc[outcome_keep,'row_id'])
                for base_name,original,group,minrows,minintervals in old_masks:
                    trimmed=ordered(original.loc[original.row_id.isin(keep_ids)])
                    untrimmed_common=ordered(trimmed.loc[trimmed.complete_dar_and_lag])
                    common=ordered(untrimmed_common.loc[untrimmed_common.row_id.isin(outcome_keep_ids)])
                    key_hash=row_hash(common);sample_key=(horizon,key_hash)
                    achieved=focus.support_table(common)
                    if sample_key not in sample_files:
                        path=root/'samples'/(tag+'__M'+str(horizon)+'__'+base_name+'__COMMON.parquet')
                        common.to_parquet(path,index=False,compression='zstd')
                        sample_files[sample_key]=dict(parquet=str(path),rows=len(common),row_hash=key_hash,horizon=horizon,exported_columns=list(common.columns))
                    for variant in VARIANTS:
                        name=tag+'__M'+str(horizon)+'__'+base_name+'__'+variant;distributed=variant=='CURRENT_PLUS_LAG1'
                        key=(variant,horizon,key_hash);alias=representatives.get(key,{}).get('model')
                        samples[name]=copy.deepcopy(sample_files[sample_key])
                        details=dict(model=name,base_model=base_name,horizon=horizon,cutoff_tag=tag,cutoff_percent=cutoff['cutoff_percent'],
                            variant=variant,fund_focus=group,old_rows=len(original),trimmed_rows=len(trimmed),n_cutoff_removed=len(original)-len(trimmed),
                            n_future_or_lag_dropped=len(trimmed)-len(untrimmed_common),n_abs_dar_removed=len(untrimmed_common)-len(common),
                            n_abs_dar_removed_entries=int(untrimmed_common.entry_zero_to_positive.sum()-common.entry_zero_to_positive.sum()),
                            n_abs_dar_removed_exits=int(untrimmed_common.complete_exit.sum()-common.complete_exit.sum()),
                            outcome_abs_quantile=cutoff.get('outcome_abs_quantile'),
                            outcome_abs_cutoff=cutoff.get('outcome_abs_cutoffs_by_horizon',{}).get(str(horizon)),
                            fixed_min_fund_rows=minrows,fixed_min_fund_intervals=minintervals,
                            achieved_min_fund_rows=int(achieved.fund_rows.min())if len(achieved)else None,
                            achieved_min_fund_intervals=int(achieved.fund_intervals.min())if len(achieved)else None,
                            first_start_date=str(common.start_date.min())if len(common)else None,last_end_date=str(common.end_date.max())if len(common)else None,
                            row_hash=key_hash,exact_sample_alias_of=alias)
                        checkpoint('PROCESSING_MODEL',model=name,rows=len(common),alias_of=alias)
                        if alias:
                            stored=representatives[key];record=copy.deepcopy(stored['record']);coefs=copy.deepcopy(stored['coefs'])
                            text=stored['text'];htm=stored['html'];test=copy.deepcopy(stored['test']);origin='CURRENT_EXACT_HORIZON_SAMPLE_ALIAS'
                        else:
                            issue=('Fewer than minimum rows'if len(common)<plan.get('minimum_model_rows',30)else
                                'Fewer than two fund/company clusters'if common.fund_id.nunique()<2 or common.company_id.nunique()<2 else None)
                            if issue:
                                record=dict(status='NOT_ESTIMABLE',reason=issue,nobs=len(common),funds=int(common.fund_id.nunique()),companies=int(common.company_id.nunique()))
                                coefs=[];text='NOT_ESTIMABLE: '+issue;htm='<pre>'+html.escape(text)+'</pre>';test=None;origin='NOT_ESTIMABLE'
                            else:
                                attempts+=1
                                try:
                                    fitted,result,terms,dropped=fit_native(common,controls,h2,distributed)
                                    record,raw=base.describe_result(fitted,result,name,OUTCOME,terms,dropped)
                                    record['covariance_options']['cluster_time']=False
                                    coefs=[{field:r.get(field)for field in COEF_FIELDS}for r in raw]
                                    test=distributed_tests(result,h2)if distributed else None
                                    overall=result.f_statistic_robust
                                    record['overall_robust_F_inference']=dict(statistic=float(overall.stat),native_p=float(overall.pval),
                                        status='VALID'if np.isfinite(overall.stat)and overall.stat>=0 and np.isfinite(overall.pval)else'UNDEFINED_COVARIANCE')
                                    text=str(result.summary);htm=result.summary.as_html();origin='NEW_UNIQUE_HORIZON_SAMPLE_FIT'
                                except Exception as error:
                                    record=dict(status='NOT_ESTIMABLE',reason=f'{type(error).__name__}: {error}',nobs=len(common),
                                        funds=int(common.fund_id.nunique()),companies=int(common.company_id.nunique()))
                                    coefs=[];text=record['reason'];htm='<pre>'+html.escape(text)+'</pre>';test=None;origin='NOT_ESTIMABLE'
                            representatives[key]=dict(model=name,record=copy.deepcopy(record),coefs=copy.deepcopy(coefs),text=text,html=htm,test=copy.deepcopy(test))
                        record.update(details,result_origin=origin,analysis_role='SUPPLEMENTARY_DAR_QCASH_PRICE_NOT_FUND_ALPHA',
                            current_inference_status=lag.target_status(coefs,CURRENT),lag_inference_status=lag.target_status(coefs,LAG)if distributed else'NOT_INCLUDED')
                        for row in coefs:row['model']=name
                        if distributed:tests[name]=test if test is not None else dict(status='NOT_ESTIMABLE',reason=record.get('reason'))
                        records.append(record);coefficients.extend(coefs);native[name]=text;catalog.append(details|dict(status=record['status'],result_origin=origin))
                        flows.append(details|dict(nobs=len(common),funds=int(common.fund_id.nunique()),companies=int(common.company_id.nunique())))
                        (root/(name+'.summary.txt')).write_text(text,encoding='utf-8')
                        (root/(name+'.summary.html')).write_text(htm,encoding='utf-8')
        pd.DataFrame(coefficients).to_csv(root/'coefficients.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(flows).to_csv(root/'sample_flow.csv',index=False,encoding='utf-8-sig')
        save('distributed_tests.json',tests);save('mask_catalog.json',catalog);save('outcome_audit.json',audits)
        comparison=[]
        for record in records:
            item={k:record.get(k)for k in ['model','base_model','horizon','cutoff_tag','cutoff_percent','fund_focus','variant','status','nobs','funds','companies',
                'rsquared_within','first_start_date','last_end_date','n_cutoff_removed','n_future_or_lag_dropped','fixed_min_fund_rows',
                'n_abs_dar_removed','n_abs_dar_removed_entries','n_abs_dar_removed_exits','outcome_abs_quantile','outcome_abs_cutoff',
                'fixed_min_fund_intervals','achieved_min_fund_rows','achieved_min_fund_intervals','current_inference_status','lag_inference_status',
                'result_origin','exact_sample_alias_of']}
            for term,prefix in [(CURRENT,'current'),(LAG,'lag1')]:
                target=next((r for r in coefficients if r['model']==record['model']and r['term']==term),{})
                item.update({prefix+'_'+k:target.get(k)for k in ['beta','se_native','ci95_native_low','ci95_native_high','p_native']})
            test=tests.get(record['model'],{});joint=test.get('joint_current_lag_zero',{});total=test.get('sum_current_lag',{})
            item.update(joint_status=joint.get('status'),joint_p=joint.get('p_value'),sum_status=total.get('status'),sum_beta=total.get('beta'),
                sum_se=total.get('se_native'),sum_ci95_low=total.get('ci95_native_low'),sum_ci95_high=total.get('ci95_native_high'),sum_p=total.get('p_native'))
            comparison.append(item)
        comparison=pd.DataFrame(comparison);comparison.to_csv(root/'comparison.csv',index=False,encoding='utf-8-sig')
        unchanged={key:sha(path)==sources[key]['sha256']for key,path in paths.items()}
        if not all(unchanged.values()):raise AssertionError('A frozen source changed during execution')
        save('source_manifest.json',dict(sources=sources,source_hashes_unchanged_after_run=unchanged,plan_sha256=args.plan_sha,runner_sha256=sha(__file__)))
        summary=dict(status='COMPLETE',specification=plan,plan_sha256=args.plan_sha,models=records,samples=samples,
            distributed_tests=tests,thresholds=thresholds,outcome_audit=audits,mask_catalog=catalog,
            unique_horizon_sample_fit_attempts=attempts,unique_horizon_sample_designs=len(representatives),
            unique_horizon_sample_files=len(sample_files),elapsed_seconds=time.time()-started,independent_QA='PENDING_SEPARATE_REVIEW',
            analysis_role='SUPPLEMENTARY_DAR_QCASH_PRICE_NOT_FUND_ALPHA',core_H3_alpha_status=plan['core_H3_alpha_status'],
            not_a_replacement_for_core_H3=True,raw_future_outcomes_preserved=True,
            predeclared_absolute_outcome_trim_scenarios=True,no_model_selected_by_significance=True)
        save('summary.json',summary);render(root,summary,comparison,native)
        checkpoint('COMPLETE',labels=len(records),unique_horizon_sample_fit_attempts=attempts)
        save('completion.json',dict(status='COMPLETE',plan_sha256=args.plan_sha,elapsed_seconds=time.time()-started,
            finished_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        return 0
    except Exception as error:
        save('failure.json',dict(status='FAILED',message=str(error),traceback=traceback.format_exc(),plan_sha256=args.plan_sha));raise
    finally:lock.unlink(missing_ok=True)


def fixture():
    h2=load_module(Path(__file__).with_name('run_h2_sentiment_lag_20261103.py'),'dar_fixture_h2')
    base=load_module(Path(__file__).with_name('run_h1_net_fund_fe_20261030.py'),'dar_fixture_base')
    source=pd.DataFrame(dict(row_id=['a','b'],fund_id=['f','f'],company_id=['c','c'],security_id=['s1','s2'],
        end_date=pd.to_datetime(['2020-01-31','2020-01-31']),delta_quantity_units=[2.,-3.],
        quantity_change_pct=[20.,-30.],net_sentiment=[.4,.4],net_sentiment_lag1=[.2,np.nan],complete_current_lag1=[True,False]))
    future=pd.DataFrame(dict(row_id=['b','a'],horizon=[1,1],trade_sign=[-1.,1.],p0_price_rub=[100.,100.],
        p0_quote_date=pd.to_datetime(['2020-01-31']*2),p0_quote_age_days=[0,0],future_price_rub=[103.,105.],
        future_quote_date=pd.to_datetime(['2020-02-28']*2),future_target_date=pd.to_datetime(['2020-02-29']*2),
        future_quote_age_days=[1,1],future_price_return=[.03,.05],cash_return=[.01,.01],
        dar_q_cash_price_pp=[-2.,4.],outcome_eligible=[True,True],outcome_reason=['eligible','eligible'],
        align_current=[-.4,.4],align_lag1=[np.nan,.2]))
    source['lag_example_date']=pd.to_datetime(['2019-12-31',None])
    source['example_nullable_flag']=pd.Series([True,pd.NA],dtype='boolean')
    future['lag_example_date']=pd.to_datetime([None,'2019-12-31'])
    future['example_nullable_flag']=pd.Series([pd.NA,True],dtype='boolean')
    joined=join_future(source,future,1)
    assert joined[OUTCOME].tolist()==[4.,-2.]and joined.complete_dar_and_lag.tolist()==[True,False]
    assert joined.quantity_change_pct.equals(source.quantity_change_pct)
    scenario=dict(outcome_abs_quantile=.99,outcome_abs_cutoffs_by_horizon={'1':2.})
    assert outcome_scenario_mask(joined,scenario,1).tolist()==[False,True]
    assert outcome_scenario_mask(joined,dict(outcome_abs_quantile=None),1).all()
    zero_source=pd.concat([source,pd.DataFrame([dict(row_id='c',fund_id='f',company_id='c',security_id='s3',end_date=pd.Timestamp('2020-01-31'),
        delta_quantity_units=0.,quantity_change_pct=0.,net_sentiment=.4,net_sentiment_lag1=.2,complete_current_lag1=True)])],ignore_index=True)
    zero_row=future.iloc[1].copy();zero_row['row_id']='c';zero_row['trade_sign']=0.
    zero_row['lag_example_date']=pd.NaT;zero_row['example_nullable_flag']=pd.NA
    zero_row[CURRENT]=np.nan;zero_row[LAG]=np.nan;zero_row[OUTCOME]=np.nan
    zero_row['outcome_eligible']=False;zero_row['outcome_reason']='no_trade'
    zero_future=pd.concat([future,pd.DataFrame([zero_row])],ignore_index=True)
    zero_joined=join_future(zero_source,zero_future,1)
    assert zero_joined.loc[2,[CURRENT,LAG,OUTCOME]].isna().all()and not zero_joined.loc[2,'complete_dar_and_lag']
    bad=future.copy();bad.loc[0,'future_quote_date']=pd.Timestamp('2020-01-30')
    try:join_future(source,bad,1)
    except AssertionError:pass
    else:raise AssertionError('A nonfuture quote was accepted')
    bad=future.copy();bad.loc[0,OUTCOME]=99.
    try:join_future(source,bad,1)
    except AssertionError:pass
    else:raise AssertionError('Incorrect DAR formula was accepted')
    rng=np.random.default_rng(420261103);rows=[]
    for fund in range(18):
        for period in range(10):
            for company in range(6):
                current,lag,control,noise=rng.normal(size=4)
                rows.append(dict(row_id=f'r{fund}-{period}-{company}',fund_id=str(fund),company_id=str(company),security_id=str(company),
                    interval_id=f'i{fund}-{period}',end_date=pd.Timestamp('2020-01-31')+pd.offsets.MonthEnd(period),
                    align_current=current,align_lag1=lag,control=control,absorbed=float(fund),
                    dar_q_cash_price_pp=.7*current+.4*lag+.3*control+.2*fund+noise,quantity_change_pct=777.))
    frame=pd.DataFrame(rows);sample,result,terms,dropped=fit_native(frame,['control','absorbed'],h2,True)
    within=sample[[OUTCOME]+terms]-sample[[OUTCOME]+terms].groupby(sample.fund_id).transform('mean')
    beta=np.linalg.lstsq(within[terms],within[OUTCOME],rcond=None)[0]
    covariance=base.cluster_covariance_fixture(sample,within[terms].to_numpy(),result.resids.to_numpy(),result.df_resid)
    assert np.allclose(result.params[terms],beta,rtol=1e-10,atol=1e-10)
    assert np.allclose(result.cov.loc[terms,terms],covariance,rtol=1e-10,atol=1e-10)
    assert sample.quantity_change_pct.eq(777.).all()and result.model.dependent.vars==[OUTCOME]
    test=distributed_tests(result,h2);block=np.asarray(test['target_covariance'])
    assert np.isclose(test['sum_current_lag']['variance'],block.sum())
    assert test['joint_current_lag_zero']['status']=='ESTIMATED'
    print(json.dumps(dict(status='PASS',scope='synthetic_only',rows=len(sample),source_quantity_Y_preserved=True,
        DAR_formula_and_trade_direction=True,forward_price_horizon_guard=True,alignment_verified=True,
        native_within_and_covariance=True,joint_and_sum_crosscovariance=True,no_trade_alignment_NA=True,not_fund_alpha=True)))
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan');parser.add_argument('--plan-sha')
    parser.add_argument('--fixture-only',action='store_true');args=parser.parse_args()
    if args.fixture_only:return fixture()
    if not args.plan or not args.plan_sha:parser.error('--plan and --plan-sha are required')
    return execute(args)


if __name__=='__main__':raise SystemExit(main())
