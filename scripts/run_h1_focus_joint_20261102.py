"""Fixed fund-focus H1 suite with exact-sample cache and explicit signal level.

Actual execution is Windows-only. Outcomes, sentiment, controls, fund effects
and fund/company covariance are inherited unchanged from the frozen source.
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


REFERENCE='REFERENCE_AUGMENTED_P99'
FOCUS=['equity','bond','mixed','money_market','other']
MICRO=['opening_weight','closing_log_NAV','closing_HHI','log_news_count_interval','log_interval_days','bond_indicator']
MACRO=['Imoex','imoex_ret','Ruonia','USD_RUB','RVI','Flow_billion_RUB','unique_funds','dummy_mar2014','dummy_feb2022','dummy_sep2018']


def planned_models():
    names=[REFERENCE]
    for group in FOCUS:
        names.append('FOCUS_'+group.upper()+'_BASELINE')
        names.extend(f'FOCUS_{group.upper()}_ROWS_{rows}_INTERVALS_{intervals}' for rows in [30,60] for intervals in [12,24,36])
    return names


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,str(path));module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def row_hash(frame):return hashlib.sha256('\n'.join(frame.row_id.astype(str)).encode()).hexdigest()


def ordered(frame):return frame.sort_values(['fund_id','end_date','security_id','interval_id'],kind='stable').reset_index(drop=True)


def support_table(frame):
    return frame.groupby('fund_id',sort=True,observed=True).agg(fund_rows=('row_id','size'),
        fund_intervals=('interval_id','nunique'),fund_end_dates=('end_date','nunique')).reset_index()


def focus_support_table(frame):
    parts=[]
    for group,part in frame.groupby('fund_focus',sort=True,dropna=False,observed=True):
        support=support_table(part);support.insert(0,'fund_focus',group);parts.append(support)
    return pd.concat(parts,ignore_index=True)


def model_samples(frame,focus_support):
    result=[(REFERENCE,ordered(frame),None,None,None)]
    for group in FOCUS:
        focus=frame.fund_focus.eq(group)
        support=focus_support.loc[focus_support.fund_focus.eq(group)]
        result.append(('FOCUS_'+group.upper()+'_BASELINE',ordered(frame.loc[focus]),group,None,None))
        for rows in [30,60]:
            for intervals in [12,24,36]:
                funds=set(support.loc[support.fund_rows.ge(rows)&support.fund_intervals.ge(intervals),'fund_id'])
                result.append((f'FOCUS_{group.upper()}_ROWS_{rows}_INTERVALS_{intervals}',
                    ordered(frame.loc[focus&frame.fund_id.isin(funds)]),group,rows,intervals))
    return result


def holm(values):
    items=[]
    for name,p in values:
        valid=p is not None and np.isfinite(p) and 0<=p<=1
        items.append((name,float(p) if valid else 1.,valid))
    maximum=0.;result={}
    for rank,(name,p,valid) in enumerate(sorted(items,key=lambda x:x[1])):
        maximum=max(maximum,min(1.,(len(items)-rank)*p));result[name]=maximum if valid else None
    return result


def focus_adjustment(coefficients,catalog):
    by_hash={};name_hash={}
    for item in catalog:
        name=item['model']
        if name==REFERENCE:continue
        key=item['row_hash'];name_hash[name]=key
        target=next((r for r in coefficients if r['model']==name and r['term']=='net_sentiment'),{})
        p=target.get('p_native')
        if target.get('se_native')is not None and (not np.isfinite(target['se_native'])or target['se_native']<=0):p=None
        if key not in by_hash:by_hash[key]=dict(representative=name,p_native=p,aliases=[name],rows=item['rows'])
        else:
            old=by_hash[key]['p_native']
            missing=(old is None or not np.isfinite(old)) and (p is None or not np.isfinite(p))
            if not missing and not np.isclose(old,p,rtol=1e-10,atol=1e-12):raise AssertionError('Identical focus mask has unequal p: '+name)
            by_hash[key]['aliases'].append(name)
    adjusted=holm([(key,item['p_native'])for key,item in by_hash.items()])
    by_name={name:adjusted[key]for name,key in name_hash.items()}
    for row in coefficients:
        row['p_native_holm_focus_unique']=by_name.get(row['model']) if row['term']=='net_sentiment' else None
        row['unique_focus_family_size']=len(by_hash) if row['term']=='net_sentiment' and row['model']!=REFERENCE else None
    return dict(method='Holm step-down; two-sided native p on unique CURRENT nonreference focus masks',
        family_size=len(by_hash),deduplication_key='Exact source-row hash with unchanged estimator/test',
        members=[dict(row_hash=key,**item,p_native_holm=adjusted[key])for key,item in by_hash.items()],
        adjusted_pvalues_by_model=by_name,empty_or_unestimable_unique_masks_keep_p1_slots=True,
        scope='Current frozen focus suite only; not all historical H1 experimentation',
        known_prior_group_baselines_not_new_discoveries=True)


def validate_estimator(record,controls):
    needed=dict(cov_type='clustered',clusters=['fund_id','company_id'],debiased=True,
                auto_df=False,count_effects=True,group_debias=True,cluster_time=False)
    if any(record.get('covariance_options',{}).get(key)!=value for key,value in needed.items()):return False
    if record.get('included_effects')!=['Entity'] or record.get('time_effects') is not False or record.get('other_effects') is not False:return False
    if record.get('weights') is not None:return False
    retained=record.get('retained_terms',[]);dropped=[r['term']for r in record.get('dropped_controls',[])]
    return 'net_sentiment' in retained and set(retained+dropped)==set(['net_sentiment']+controls)


def cache_match(sample,key,caches,controls,track,checks):
    for family in caches:
        for record in family['summary']['models']:
            name=record['model'];meta=family['summary'].get('samples',{}).get(name,{})
            if record.get('status')!='ESTIMATED' or meta.get('row_hash')!=key or meta.get('rows')!=len(sample):continue
            if record.get('source_row_hash')!=key or not validate_estimator(record,controls):continue
            path=Path(meta['parquet']);numeric=['quantity_change_pct','quantity_open','quantity_close','net_sentiment']+controls
            prior=pd.read_parquet(path,columns=['row_id']+numeric)
            same=np.array_equal(prior.row_id.astype(str),sample.row_id.astype(str))
            same_values=all(np.array_equal(prior[c].to_numpy(float),sample[c].to_numpy(float),equal_nan=True)for c in numeric)
            if not same or row_hash(prior)!=key or not same_values:raise AssertionError('Cache metadata falsely matches a different sample: '+name)
            coef=family['coefficients'].loc[family['coefficients'].model.eq(name)].copy()
            if coef.term.duplicated().any() or set(coef.term)!=set(['const']+record['retained_terms']):raise AssertionError('Cache coefficient schema invalid')
            txt=family['root']/(name+'.summary.txt');htm=family['root']/(name+'.summary.html')
            for item in [path,txt,htm]:track(item)
            provenance=dict(cache_family=family['name'],source_model=name,source_summary=family['summary_path'],
                source_sample=str(path),row_hash=key,rows=len(sample),exact_ordered_ids=True,exact_Y_X_quantities=True,
                same_helpers_and_estimator=True)
            checks.append(provenance)
            return copy.deepcopy(record),coef.to_dict('records'),txt.read_text(encoding='utf-8'),htm.read_text(encoding='utf-8'),provenance
    return None


def signal_level_audit(frame):
    required=['company_period_id','company_id','start_date','end_date','net_sentiment','n_relevant','fund_id','security_id','interval_id','fund_focus']
    if set(required)-set(frame):raise ValueError('Signal audit lacks mandatory fields')
    if frame.company_period_id.isna().any():raise AssertionError('Eligible row has no company-period key')
    groups=frame.groupby('company_period_id',sort=False,observed=True)
    signal=groups.agg(rows=('row_id','size'),companies=('company_id','nunique'),starts=('start_date','nunique'),
        ends=('end_date','nunique'),net_values=('net_sentiment','nunique'),news_counts=('n_relevant','nunique'),
        funds=('fund_id','nunique'),securities=('security_id','nunique')).reset_index()
    checks={column:bool(signal[column].eq(1).all())for column in ['companies','starts','ends','net_values','news_counts']}
    checks['reverse_company_window_key']=bool(frame.groupby(['company_id','start_date','end_date'],observed=True).company_period_id.nunique().eq(1).all())
    if not all(checks.values()):raise AssertionError('Company-period score is not a single company/window signal')
    count_columns=['n_positive','n_negative','n_neutral']
    if all(column in frame for column in count_columns):
        partition=np.isclose(frame.n_positive+frame.n_negative+frame.n_neutral,frame.n_relevant,rtol=0,atol=0)
        calculated=(frame.n_positive-frame.n_negative)/frame.n_relevant
        formula=np.isclose(frame.net_sentiment,calculated,rtol=1e-12,atol=1e-12)
        if not partition.all()or not formula.all():raise AssertionError('Stored news counts disagree with the frozen Net formula')
        formula_audit=dict(status='PASS',formula='(n_positive-n_negative)/(n_positive+n_neutral+n_negative)',
            count_partition_equal=True,stored_net_matches_counts=True,rows_checked=len(frame))
    else:formula_audit=dict(status='COUNTS_NOT_IN_SOURCE',consistency_only=True,missing_columns=[c for c in count_columns if c not in frame])
    if frame.groupby(['fund_id','interval_id'],observed=True).fund_focus.nunique().gt(1).any():
        raise AssertionError('Opening-report focus changes within one fund interval')
    issuer_position=frame.groupby(['fund_id','company_id','interval_id'],sort=False,observed=True).security_id.nunique()
    candidates=signal.loc[signal.funds.gt(1)&signal.securities.gt(1)]
    if candidates.empty:candidates=signal
    example_key=candidates.sort_values(['funds','securities','rows'],ascending=False,kind='stable').company_period_id.iloc[0]
    columns=[c for c in ['row_id','company_period_id','company','company_id','fund_name','fund_id','fund_focus','asset_name','security_id',
        'asset_type','start_date','end_date','net_sentiment','n_relevant','n_positive','n_negative','n_neutral','quantity_open','quantity_close','quantity_change_pct']if c in frame]
    example=frame.loc[frame.company_period_id.eq(example_key),columns].copy()
    per_fund_focus=frame.groupby('fund_id',observed=True).fund_focus.nunique()
    result=dict(status='PASS',regression_rows=len(frame),unique_company_period_signals=len(signal),
        unique_companies=int(frame.company_id.nunique()),unique_securities=int(frame.security_id.nunique()),
        company_period_unique_checks=checks,net_formula_audit=formula_audit,company_periods_repeated_across_funds=int(signal.funds.gt(1).sum()),
        company_periods_repeated_across_securities=int(signal.securities.gt(1).sum()),
        fund_company_intervals_with_multiple_securities=int(issuer_position.gt(1).sum()),
        max_securities_in_fund_company_interval=int(issuer_position.max()),
        funds_with_multiple_focus_labels_over_time=int(per_fund_focus.gt(1).sum()),
        example_company_period_id=str(example_key),example_rows=len(example),
        sentiment_level='Company × exact actual reporting window; same score reused across matching funds and securities',
        outcome_level='Fund × security × actual reporting interval; unchanged estimated-quantity growth, conditional zero-entry code',
        focus_level='Fund asset focus from existing opening-report classification, not security asset class',
        aggregation_changed=False,
        interpretation='Repeated signals are not independent company-news draws. Fund/company clustering is unchanged; rows, securities and unique signals must be distinguished. Fund focus may change across intervals.')
    return result,signal,example


def type_coverage(frame):
    rows=[]
    for focus,part in frame.groupby('fund_focus',dropna=False,sort=True,observed=True):
        support=support_table(part)
        label=str(focus)if pd.notna(focus)else'(missing)'
        rows.append(dict(fund_focus=label,rows=len(part),funds=int(part.fund_id.nunique()),companies=int(part.company_id.nunique()),
            securities=int(part.security_id.nunique()),company_period_signals=int(part.company_period_id.nunique()),
            fund_intervals=int(part[['fund_id','interval_id']].drop_duplicates().shape[0]),
            within_group_min_fund_rows=int(support.fund_rows.min()),within_group_min_fund_intervals=int(support.fund_intervals.min()),
            included_prespecified_focus=label in FOCUS))
    return pd.DataFrame(rows)


def render(root,summary,comparison,native):
    parts=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>H1: типы фондов и совместные ограничения</title>',
        '<style>body{font:16px system-ui;max-width:1550px;margin:32px auto;padding:0 20px;color:#17212d}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}pre{padding:16px;background:#f5f6f8;overflow:auto;font:12px monospace}a{color:#075ba5}</style>',
        '<h1>H1: типы фондов и совместные ограничения</h1>',
        '<p>Пять заранее заданных типов: акции, облигации, смешанные, денежный рынок, прочие. Для каждого показаны исходная группа и шесть сочетаний ≥30/≥60 строк И ≥12/≥24/≥36 интервалов. Support посчитан один раз отдельно внутри каждой исходной группы до ограничений, без итеративного пересчёта. Полный support по всем типам сохранён для сравнения. Это доступные estimation-интервалы, не обязательно календарные месяцы и не вся история фонда. Тип задаётся на начало интервала, может меняться; фонды разных групп могут пересекаться во времени.</p>',
        '<p>Y остаётся изменением расчётного количества конкретной бумаги в конкретном фонде. NetSentiment уже агрегирован по компании и точному отчётному окну: один score повторяется у подходящих бумаг и фондов. Уровень зависимой переменной не менялся. Порог P99, +100% для подтверждённых входов, контроли, fund FE и кластеры фонд/компания сохранены. Временных эффектов/кластеров нет.</p>',
        '<p>Все 36 labels сохранены, включая пустые или неоцениваемые. Точные совпадения используют один результат и не являются независимыми подтверждениями. Номинальное p&lt;0,05 отдельной группы не доказывает H1 для всей совокупности или различий между типами. Ранее известные результаты не считаются новыми открытиями. Поправка Holm относится к уникальным маскам текущей серии, а не ко всем историческим решениям.</p>',
        comparison.to_html(index=False,border=0,float_format=lambda x:f'{x:.6g}'),
        '<p><a href="type_coverage.csv">Покрытие типов</a> · <a href="news_signal_level_audit.json">Уровень новостного сигнала</a> · <a href="issuer_period_example.csv">Пример одного сигнала у разных бумаг/фондов</a> · <a href="focus_fund_support.csv">Support внутри исходных типов</a> · <a href="fund_support.csv">Полный support фондов</a> · <a href="mask_catalog.json">Маски и кэш</a> · <a href="multiplicity.json">Holm</a> · <a href="coefficients.csv">Все коэффициенты</a> · <a href="summary.json">Все метрики</a></p>']
    for name,text in native.items():parts.append('<h2>'+html.escape(name)+'</h2><pre>'+html.escape(text)+'</pre>')
    parts.append('</html>');(root/'report.html').write_text('\n'.join(parts),encoding='utf-8')


def execute(args):
    if sys.platform!='win32':raise RuntimeError('Actual research execution must run on research-host')
    planpath=Path(args.plan)
    if sha(planpath)!=args.plan_sha:raise ValueError('Frozen plan SHA256 mismatch')
    plan=json.loads(planpath.read_text(encoding='utf-8'))
    if plan['planned_models']!=planned_models() or plan['focus_groups']!=FOCUS:raise ValueError('Frozen focus suite changed')
    controls=plan['baseline_controls']+plan['macro_controls']
    if controls!=MICRO+MACRO or plan.get('time_FE') or plan.get('cluster_time') or plan.get('clusters')!=['fund_id','company_id']:
        raise ValueError('Frozen estimator/control conventions changed')
    root=Path(plan['output_root'])
    if any(part.lower()=='databaseresources'for part in root.parts):raise ValueError('Protected output location')
    root.mkdir(parents=True,exist_ok=True)
    if (root/'completion.json').exists():raise FileExistsError('Completed version may not be overwritten')
    keys=['input_panel','base_helper','macro_runner_helper','entries_summary','entries_coefficients','joint_summary','joint_coefficients']
    paths={key:plan[key]for key in keys};sources={}
    for key,path in paths.items():
        actual=sha(path)
        if actual!=plan['expected_source_sha256'][key]:raise ValueError('Frozen source hash mismatch: '+key)
        sources[key]=dict(path=path,sha256=actual,size_bytes=Path(path).stat().st_size)
    used_cache_files={}
    def track(path):
        key=str(path)
        if key not in used_cache_files:used_cache_files[key]=dict(path=key,sha256=sha(path),size_bytes=Path(path).stat().st_size)
    base=load_module(paths['base_helper'],'h1_focus_base');macro=load_module(paths['macro_runner_helper'],'h1_focus_macro')
    caches=[]
    for family in ['entries','joint']:
        summarypath=Path(paths[family+'_summary']);summary=json.loads(summarypath.read_text(encoding='utf-8'))
        for helper in ['base_helper','macro_runner_helper']:
            if summary['specification']['expected_source_sha256'][helper]!=plan['expected_source_sha256'][helper]:
                raise ValueError('Cached helper differs: '+family+'/'+helper)
        if summary['specification']['baseline_controls']+summary['specification']['macro_controls']!=controls:raise ValueError('Cache controls changed')
        caches.append(dict(name=family,summary=summary,summary_path=str(summarypath),root=summarypath.parent,
            coefficients=pd.read_csv(paths[family+'_coefficients'])))
    lock=root/'run.lock';descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(descriptor,json.dumps(dict(pid=os.getpid(),plan_sha256=args.plan_sha)).encode());os.close(descriptor)
    started=time.time()
    def save(name,value):base.save(root/name,value)
    def checkpoint(state,**extra):
        record=dict(state=state,pid=os.getpid(),elapsed_seconds=time.time()-started,**extra)
        save('checkpoint.json',record);print(json.dumps(record),flush=True)
    try:
        save('frozen_plan.json',plan);checkpoint('READING_FIXED_PRIMARY')
        frame=ordered(macro.read_frame(paths['input_panel']))
        if len(frame)!=plan.get('expected_primary_rows',221899) or frame.row_id.duplicated().any():raise AssertionError('Frozen primary changed')
        if not macro.finite_rows(frame,['quantity_change_pct','net_sentiment']+controls).all():raise ValueError('Incomplete frozen source')
        support=support_table(frame);support.to_csv(root/'fund_support.csv',index=False,encoding='utf-8-sig')
        focus_support=focus_support_table(frame)
        focus_support.to_csv(root/'focus_fund_support.csv',index=False,encoding='utf-8-sig')
        coverage=type_coverage(frame);coverage.to_csv(root/'type_coverage.csv',index=False,encoding='utf-8-sig')
        signal,signal_table,example=signal_level_audit(frame)
        save('news_signal_level_audit.json',signal)
        signal_table.to_csv(root/'company_period_signal_counts.csv',index=False,encoding='utf-8-sig')
        example.to_csv(root/'issuer_period_example.csv',index=False,encoding='utf-8-sig')
        samples=model_samples(frame,focus_support)
        coefficients=[];records=[];metadata={};catalog=[];native={};representatives={};cache_checks=[];newfits=0;cachehits=0
        (root/'samples').mkdir(exist_ok=True)
        for name,sample,group,minrows,minintervals in samples:
            key=row_hash(sample);alias=representatives.get(key,{}).get('model')
            checkpoint('PROCESSING_MODEL',model=name,rows=len(sample),funds=int(sample.fund_id.nunique()),alias_of=alias)
            path=root/'samples'/(name+'.parquet');sample.to_parquet(path,index=False,compression='zstd')
            metadata[name]=dict(parquet=str(path),rows=len(sample),row_hash=key,exported_columns=list(sample.columns))
            subsupport=support_table(sample)
            item=dict(model=name,fund_focus=group,min_focus_fund_rows=minrows,min_focus_fund_intervals=minintervals,
                operator='AND',rows=len(sample),funds=int(sample.fund_id.nunique()),companies=int(sample.company_id.nunique()),
                row_hash=key,alias_of=alias,min_within_group_fund_rows=int(subsupport.fund_rows.min())if len(subsupport)else None,
                min_within_group_fund_intervals=int(subsupport.fund_intervals.min())if len(subsupport)else None)
            if alias:
                stored=representatives[key];record=copy.deepcopy(stored['record']);coefs=copy.deepcopy(stored['coefs'])
                text=stored['text'];htmltext=stored['html'];provenance=copy.deepcopy(stored['provenance']);origin='CURRENT_EXACT_SAMPLE_ALIAS'
            else:
                cached=cache_match(sample,key,caches,controls,track,cache_checks)
                if cached:
                    record,coefs,text,htmltext,provenance=cached;origin='VALIDATED_PRIOR_EXACT_SAMPLE_CACHE';cachehits+=1
                else:
                    issue=macro.model_status(sample,plan.get('minimum_model_rows',30))
                    if issue:
                        record=dict(status='NOT_ESTIMABLE',reason=issue,nobs=len(sample),funds=int(sample.fund_id.nunique()),companies=int(sample.company_id.nunique()))
                        coefs=[];text='NOT_ESTIMABLE: '+issue;htmltext='<pre>'+html.escape(text)+'</pre>';provenance=None;origin='NOT_ESTIMABLE'
                    else:
                        newfits+=1
                        try:
                            fitted,result,terms,dropped=macro.fit(sample,controls,base)
                            record,coefs=base.describe_result(fitted,result,name,'quantity_change_pct',terms,dropped)
                            record['covariance_options']['cluster_time']=False
                            joint=result.f_statistic_robust
                            record['joint_robust_F_inference']=dict(statistic=float(joint.stat),native_p=float(joint.pval),
                                status='VALID'if np.isfinite(joint.stat)and joint.stat>=0 and np.isfinite(joint.pval)else'UNDEFINED_COVARIANCE')
                            text=str(result.summary);htmltext=result.summary.as_html();provenance=None;origin='NEW_UNIQUE_SAMPLE_FIT'
                        except Exception as error:
                            if name==REFERENCE:raise
                            record=dict(status='NOT_ESTIMABLE',reason=f'{type(error).__name__}: {error}',nobs=len(sample),funds=int(sample.fund_id.nunique()),companies=int(sample.company_id.nunique()))
                            coefs=[];text=record['reason'];htmltext='<pre>'+html.escape(text)+'</pre>';provenance=None;origin='NOT_ESTIMABLE'
                representatives[key]=dict(model=name,record=copy.deepcopy(record),coefs=copy.deepcopy(coefs),text=text,html=htmltext,provenance=copy.deepcopy(provenance))
            record.update(model=name,result_origin=origin,exact_sample_alias_of=alias,cache_provenance=provenance,
                fund_focus=group,conditional_entry_rows=int(sample.entry_zero_to_positive.sum()),complete_exit_rows=int(sample.complete_exit.sum()))
            for row in coefs:
                row['model']=name
                for old in list(row):
                    if 'holm' in old.lower()or old in ['unique_family_size','unique_focus_family_size']:row.pop(old,None)
            target=next((r for r in coefs if r['term']=='net_sentiment'),{})
            infer_fields=['beta','se_native','p_native','ci95_native_low','ci95_native_high']
            target_valid=bool(target)and all(target.get(field)is not None and np.isfinite(target[field])for field in infer_fields)
            target_valid=target_valid and target['se_native']>0 and 0<=target['p_native']<=1
            record['net_inference_status']=('FINITE'if target_valid else'UNDEFINED')if coefs else'NOT_ESTIMATED'
            if name==REFERENCE:
                for field,expected in [('beta',.5946851064715204),('se_native',.3821674238484671),('p_native',.11968924263465253)]:
                    if not np.isclose(target.get(field,np.nan),expected,rtol=1e-7,atol=1e-8):raise AssertionError('Reference changed: '+field)
            item.update(result_origin=origin,cache_provenance=provenance,status=record['status'])
            catalog.append(item);records.append(record);coefficients.extend(coefs);native[name]=text
            (root/(name+'.summary.txt')).write_text(text,encoding='utf-8')
            (root/(name+'.summary.html')).write_text(htmltext,encoding='utf-8')
        adjustment=focus_adjustment(coefficients,catalog)
        save('multiplicity.json',adjustment);save('mask_catalog.json',catalog)
        pd.DataFrame(coefficients).to_csv(root/'coefficients.csv',index=False,encoding='utf-8-sig')
        comparison=[]
        for record in records:
            item={k:record.get(k)for k in ['model','fund_focus','status','nobs','funds','companies','rsquared_within','result_origin',
                'exact_sample_alias_of','conditional_entry_rows','complete_exit_rows','net_inference_status']}
            target=next((r for r in coefficients if r['model']==record['model']and r['term']=='net_sentiment'),{})
            item.update({k:target.get(k)for k in ['beta','se_native','ci95_native_low','ci95_native_high','p_native','p_native_holm_focus_unique','unique_focus_family_size']})
            comparison.append(item)
        comparison=pd.DataFrame(comparison);comparison.to_csv(root/'comparison.csv',index=False,encoding='utf-8-sig')
        unchanged={key:sha(path)==sources[key]['sha256']for key,path in paths.items()}
        cache_unchanged={path:sha(path)==info['sha256']for path,info in used_cache_files.items()}
        if not all(unchanged.values())or not all(cache_unchanged.values()):raise AssertionError('Source/cache changed during execution')
        save('source_manifest.json',dict(sources=sources,source_hashes_unchanged_after_run=unchanged,used_cache_files=used_cache_files,
            cache_hashes_unchanged_after_run=cache_unchanged,plan_sha256=args.plan_sha,runner_sha256=sha(__file__)))
        summary=dict(status='COMPLETE',specification=plan,plan_sha256=args.plan_sha,models=records,samples=metadata,
            multiplicity=adjustment,mask_catalog=catalog,cache_checks=cache_checks,new_unique_sample_fits=newfits,
            validated_prior_cache_reuses=cachehits,unique_samples=len(representatives),elapsed_seconds=time.time()-started,
            type_coverage=coverage.to_dict('records'),news_signal_level_audit=signal,
            support_counted_once_on_unfiltered_focus_baseline=True,full_source_support_descriptive_only=True,
            iterative_support_recounting=False,filter_support_scope='Each unfiltered focus baseline, once',
            independent_QA='PENDING_SEPARATE_REVIEW',interpretation='Fixed focus sensitivity; all labels reported. Known prior baselines retained with provenance, not new discoveries. No outcome, sentiment aggregation, estimator or winning-model selection change.')
        save('summary.json',summary);render(root,summary,comparison,native)
        checkpoint('COMPLETE',new_unique_sample_fits=newfits,cache_reuses=cachehits)
        save('completion.json',dict(status='COMPLETE',plan_sha256=args.plan_sha,elapsed_seconds=time.time()-started,
            finished_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        return 0
    except Exception as error:
        save('failure.json',dict(status='FAILED',message=str(error),traceback=traceback.format_exc(),plan_sha256=args.plan_sha));raise
    finally:lock.unlink(missing_ok=True)


def fixture(args):
    records=[]
    for fund,(intervals,securities)in enumerate([(1,1),(12,3),(24,1),(24,3),(36,2),(40,4)]):
        for period in range(intervals):
            for company in range(securities):
                start=pd.Timestamp('2020-01-31')+pd.offsets.MonthEnd(period)
                records.append(dict(fund_id=str(fund),company_id=str(company),security_id=str(company),row_id=f'r{fund}-{period}-{company}',
                    interval_id=f'i{fund}-{period}',company_period_id=f'cp{company}-{period}',start_date=start,end_date=start+pd.offsets.MonthEnd(1),
                    fund_focus='bond'if period%2 else'equity',net_sentiment=company/10,n_relevant=period+1))
    frame=ordered(pd.DataFrame(records));support=focus_support_table(frame);samples=model_samples(frame,support)
    assert len(samples)==36 and [item[0]for item in samples]==planned_models()
    for name,sample,group,rows,intervals in samples[1:]:
        mask=frame.fund_focus.eq(group)
        if rows is not None:
            original_focus=frame.loc[frame.fund_focus.eq(group)]
            funds={fund for fund,part in original_focus.groupby('fund_id')if len(part)>=rows and part.interval_id.nunique()>=intervals}
            mask&=frame.fund_id.isin(funds)
        assert np.array_equal(sample.row_id,ordered(frame.loc[mask]).row_id)
    audit,signals,example=signal_level_audit(frame)
    assert audit['company_period_unique_checks']['net_values']and audit['funds_with_multiple_focus_labels_over_time']>0
    corrupt=frame.copy();corrupt.loc[0,'net_sentiment']=.9
    try:signal_level_audit(corrupt)
    except AssertionError:pass
    else:raise AssertionError('Signal-level corruption was accepted')
    coefficients=[];catalog=[];lookup={};number=0
    for name,sample,_,_,_ in samples:
        key=row_hash(sample)
        if key not in lookup:lookup[key]=.01+number*.01;number+=1
        catalog.append(dict(model=name,row_hash=key,rows=len(sample)))
        if len(sample):coefficients.append(dict(model=name,term='net_sentiment',p_native=lookup[key]))
    adjustment=focus_adjustment(coefficients,catalog)
    assert adjustment['family_size']==len({r['row_hash']for r in catalog if r['model']!=REFERENCE})
    aliases=[item for item in adjustment['members']if len(item['aliases'])>1]
    assert aliases and any(item['rows']==0for item in adjustment['members'])
    print(json.dumps(dict(status='PASS',scope='synthetic_only',models=len(samples),fixed_unfiltered_focus_support=True,
        focus_can_vary_over_time=True,signal_level_checks=True,unique_mask_Holm=True,empty_labels_preserved=True)))
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan');parser.add_argument('--plan-sha')
    parser.add_argument('--fixture-only',action='store_true');args=parser.parse_args()
    if args.fixture_only:return fixture(args)
    if not args.plan or not args.plan_sha:parser.error('--plan and --plan-sha are required')
    return execute(args)


if __name__=='__main__':raise SystemExit(main())
