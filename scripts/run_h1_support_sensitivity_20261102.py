"""Finite, frozen H1 fund-support robustness suite; actual runs on research-host only.

All eight new restrictions are reported and form one Holm family. Thresholds
are counted once on the original frozen primary sample, without iteration.
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


MODELS = ['REFERENCE_AUGMENTED_P99', 'MIN_FUND_ROWS_30', 'MIN_FUND_ROWS_60',
          'MIN_FUND_INTERVALS_12', 'MIN_FUND_INTERVALS_24', 'MIN_FUND_INTERVALS_36',
          'NEWS_MIN_3', 'NEWS_MIN_5', 'BOUNDED_GROSS_CONTROLS']
RULES = [('REFERENCE_AUGMENTED_P99', None, 0),
         ('MIN_FUND_ROWS_30', 'fund_rows', 30),
         ('MIN_FUND_ROWS_60', 'fund_rows', 60),
         ('MIN_FUND_INTERVALS_12', 'fund_intervals', 12),
         ('MIN_FUND_INTERVALS_24', 'fund_intervals', 24),
         ('MIN_FUND_INTERVALS_36', 'fund_intervals', 36)]
MICRO = ['opening_weight', 'closing_log_NAV', 'closing_HHI',
         'log_news_count_interval', 'log_interval_days', 'bond_indicator']
MACRO = ['Imoex', 'imoex_ret', 'Ruonia', 'USD_RUB', 'RVI',
         'Flow_billion_RUB', 'unique_funds', 'dummy_mar2014',
         'dummy_feb2022', 'dummy_sep2018']


def load_module(path, name):
    spec=importlib.util.spec_from_file_location(name,str(path))
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def row_hash(frame):
    return hashlib.sha256('\n'.join(frame.row_id.astype(str)).encode()).hexdigest()


def ordered(frame):
    return frame.sort_values(['fund_id','end_date','security_id','interval_id'],kind='stable').reset_index(drop=True)


def support_table(frame):
    grouped=frame.groupby('fund_id',sort=True,observed=True)
    result=grouped.agg(fund_rows=('row_id','size'),fund_intervals=('interval_id','nunique'),
        fund_end_dates=('end_date','nunique'),fund_securities=('security_id','nunique'),
        fund_companies=('company_id','nunique'),first_start_date=('start_date','min'),
        last_end_date=('end_date','max')).reset_index()
    if 'fund_name' in frame:
        names=grouped.fund_name.agg(lambda s:next((str(v) for v in s if pd.notna(v)),None)).rename('fund_name').reset_index()
        result=result.merge(names,on='fund_id',how='left',validate='one_to_one')
    return result


def samples_from_support(frame, support):
    if support.fund_id.duplicated().any():raise ValueError('Duplicate support fund keys')
    samples=[];removed=[]
    for name,column,minimum in RULES:
        keep=pd.Series(True,index=support.index) if column is None else support[column].ge(minimum)
        funds=set(support.loc[keep,'fund_id'])
        sample=ordered(frame.loc[frame.fund_id.isin(funds)])
        samples.append((name,sample))
        missing=support.loc[~keep].copy()
        missing.insert(0,'model',name)
        missing['restriction_column']=column
        missing['restriction_minimum']=minimum
        removed.append(missing)
    for name in MODELS[6:]:
        if name.startswith('NEWS_MIN_'):
            minimum=int(name.rsplit('_',1)[1])
            mask=pd.to_numeric(frame.n_relevant,errors='coerce').ge(minimum)
        else:
            minimum=None
            values=frame[['opening_weight','closing_weight','closing_HHI']].apply(pd.to_numeric,errors='coerce')
            mask=pd.Series(np.isfinite(values).all(axis=1),index=frame.index)&values.ge(0).all(axis=1)&values.le(1.001).all(axis=1)
        sample=ordered(frame.loc[mask])
        samples.append((name,sample))
        missing=support.loc[~support.fund_id.isin(set(sample.fund_id))].copy()
        missing.insert(0,'model',name)
        missing['restriction_column']='n_relevant' if minimum is not None else 'bounded_gross_controls'
        missing['restriction_minimum']=minimum
        removed.append(missing)
    return samples,pd.concat(removed,ignore_index=True)


def holm_adjust(coefficients):
    family=MODELS[1:];result={}
    for field,outfield in [('p_native','p_native_holm_support_family'),
                           ('p_H1_positive_one_sided','p_H1_positive_one_sided_holm_support_family')]:
        items=[]
        for name in family:
            found=next((r for r in coefficients if r['model']==name and r['term']=='net_sentiment'),{})
            p=found.get(field)
            valid=p is not None and np.isfinite(p) and 0<=p<=1
            items.append((name,float(p) if valid else 1.,valid))
        maximum=0.;adjusted={}
        for rank,(name,p,valid) in enumerate(sorted(items,key=lambda x:x[1])):
            maximum=max(maximum,min(1.,(len(family)-rank)*p))
            adjusted[name]=maximum if valid else None
        for row in coefficients:row[outfield]=adjusted.get(row['model']) if row['term']=='net_sentiment' else None
        result[field]=dict(method='Holm step-down on all eight new tests',planned_family=family,
            reference_excluded=True,undefined_or_unestimated_tests_keep_p1_slots=True,adjusted_pvalues=adjusted)
    return result


def information_table(frame, residual_net, grouping, group_name):
    """Share of squared residual Net: a design-information diagnostic, not Y fit."""
    data=pd.DataFrame({'group':pd.Series(grouping,index=frame.index).astype('string').fillna('(missing)'),
                       'net_information':np.square(residual_net),
                       'fund_id':frame.fund_id,'company_id':frame.company_id})
    table=data.groupby('group',sort=False,dropna=False,observed=True).agg(
        rows=('net_information','size'),information_sum=('net_information','sum'),
        funds=('fund_id','nunique'),companies=('company_id','nunique')).reset_index()
    total=float(data.net_information.sum())
    table['information_share']=table.information_sum/total if total>0 else np.nan
    table.insert(0,'dimension',group_name)
    return table.sort_values('information_sum',ascending=False,kind='stable').reset_index(drop=True)


def fwl_information(frame,support,controls,base):
    terms,dropped=base.select_regressors(frame,controls)
    control_terms=[term for term in terms if term!='net_sentiment']
    values=frame[['net_sentiment']+control_terms].astype(float)
    within=values-values.groupby(frame.fund_id,sort=False).transform('mean')
    net=within.net_sentiment.to_numpy(float)
    if control_terms:
        x=within[control_terms].to_numpy(float)
        norms=np.linalg.norm(x,axis=0)
        if (norms<=0).any():raise ValueError('Retained control has no within variation')
        z=x/norms
        beta,_,rank,singular=np.linalg.lstsq(z,net,rcond=None)
        if rank!=len(control_terms):raise ValueError('FWL controls are rank deficient')
        residual=net-z@beta
        max_orthogonality=float(np.max(np.abs(z.T@residual)))
    else:
        residual=net.copy();rank=0;max_orthogonality=0.
    total=float(residual@residual)
    if not np.isfinite(total) or total<=0:raise ValueError('No residual Net information')
    expanded=frame[['fund_id']].merge(support[['fund_id','fund_rows','fund_intervals','fund_end_dates']],
        on='fund_id',how='left',validate='many_to_one',sort=False)
    rowbins=pd.cut(expanded.fund_rows,[0,1,29,59,np.inf],labels=['1 row','2–29 rows','30–59 rows','60+ rows'])
    intervalbins=pd.cut(expanded.fund_intervals,[0,1,11,23,35,np.inf],labels=['1 interval','2–11 intervals','12–23 intervals','24–35 intervals','36+ intervals'])
    tables={
        'fund_rows':information_table(frame,residual,rowbins,'original fund row support'),
        'fund_intervals':information_table(frame,residual,intervalbins,'original distinct actual intervals'),
        'fund':information_table(frame,residual,frame.fund_id,'fund_id'),
        'company':information_table(frame,residual,frame.company_id,'company_id')}
    if 'n_relevant' in frame:
        bins=pd.cut(pd.to_numeric(frame.n_relevant,errors='coerce'),[0,1,4,9,29,np.inf],
                    labels=['1 relevant','2–4 relevant','5–9 relevant','10–29 relevant','30+ relevant'])
        tables['news_count']=information_table(frame,residual,bins,'relevant article count in exact company interval')
    for name in ['source_path','fund_focus','index_status']:
        if name in frame:tables[name]=information_table(frame,residual,frame[name],name)
    metadata=dict(status='COMPUTED',scope='Original frozen full primary only',
        definition='Fund-demeaned NetSentiment residualized on the same deterministic retained controls; row information is residual Net squared / its total sum',
        outcome_not_used=True,does_not_select_or_weight_outcome_regressions=True,
        retained_control_terms=control_terms,dropped_controls=dropped,control_rank=int(rank),
        total_residual_net_SS=total,within_net_SS=float(net@net),
        max_control_orthogonality=max_orthogonality,max_row_information_share=float(np.square(residual).max()/total),
        top_10_funds_information_share=float(tables['fund'].information_share.head(10).sum()),
        top_10_companies_information_share=float(tables['company'].information_share.head(10).sum()),
        notes='Information shares describe regressor support, not causal identification or outcome leverage. Relevant-news counts refer to matched dataset coverage, not all possible company news.')
    return metadata,tables


def render(root,summary,comparison,native):
    parts=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>H1: устойчивость к числу наблюдений фонда</title>',
      '<style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 20px;color:#17212d}table{border-collapse:collapse;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}pre{background:#f5f6f8;padding:16px;overflow:auto;font:13px monospace}a{color:#075ba5}</style>',
      '<h1>H1: устойчивость к числу наблюдений и качеству данных</h1>',
      '<p>До новых оценок зафиксированы восемь ограничений и исходная модель. Число строк и число различных фактических интервалов рассчитаны один раз на полной исходной очищенной панели. Строка — фонд × бумага × интервал; несколько бумаг за один отчётный период дают несколько строк. Min Obs в стандартном выводе не равен числу месяцев. Дополнительно отдельно проверены окна с ≥3/≥5 релевантными новостями и ограниченная подвыборка opening/closing weights и HHI в [0;1,001]. Выход за эти границы сам по себе не доказывает ошибку: gross exposure и обязательства могут менять отношение к NAV. Фильтры не комбинируются.</p>',
      '<p>Порог P99, условные +100% для подтверждённых входов, −100% для выходов, NetSentiment и все контроли не изменены. Fund FE, кластеризация фонд/компания; временных эффектов и кластеров нет. Ни одна спецификация не выбирается по p-value. Для всех восьми новых тестов показана поправка Holm отдельно для двустороннего native p и ранее сохранявшегося направленного H1 p с conservative cluster-t. Эта серия — исследовательская проверка устойчивости после прежних результатов.</p>',
      comparison.to_html(index=False,border=0,float_format=lambda x:f'{x:.7g}'),
      '<p><a href="fund_support.csv">Наблюдения и интервалы каждого фонда</a> · <a href="removed_funds.csv">Исключённые фонды по моделям</a> · <a href="coefficients.csv">Все коэффициенты</a> · <a href="FWL_information.json">Информация в Net</a> · <a href="summary.json">Все метрики</a> · <a href="source_manifest.json">Исходные файлы</a></p>',
      '<p>FWL-диагностика описывает, какие наблюдения дают вариацию сентимента после учёта контролей и fund FE. Она не использует зависимую переменную и не доказывает причинность. Значимость всей модели по F-тесту не равна значимости сентимента.</p>']
    for name,text in native.items():parts.append('<h2>'+html.escape(name)+'</h2><pre>'+html.escape(text)+'</pre>')
    parts.append('</html>');(root/'report.html').write_text('\n'.join(parts),encoding='utf-8')


def execute(args):
    if sys.platform!='win32':raise RuntimeError('Actual research-data execution must run on research-host')
    plan_path=Path(args.plan)
    if sha(plan_path)!=args.plan_sha:raise ValueError('Frozen plan SHA256 mismatch')
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    if plan['planned_models']!=MODELS:raise ValueError('Unexpected frozen model suite')
    controls=plan['baseline_controls']+plan['macro_controls']
    if controls!=MICRO+MACRO:raise ValueError('Control set changed')
    if plan.get('time_FE') or plan.get('cluster_time') or plan.get('clusters')!=['fund_id','company_id']:
        raise ValueError('FE/clustering differs from fixed specification')
    root=Path(plan['output_root'])
    if any(p.lower()=='databaseresources' for p in root.parts):raise ValueError('Protected output location')
    root.mkdir(parents=True,exist_ok=True)
    if (root/'completion.json').exists():raise FileExistsError('Completed version may not be overwritten')
    basepath=args.base_helper or plan['base_helper']
    macropath=args.macro_runner_helper or plan['macro_runner_helper']
    paths={'input_panel':plan['input_panel'],'base_helper':basepath,'macro_runner_helper':macropath}
    sources={}
    for key,path in paths.items():
        actual=sha(path)
        if actual!=plan['expected_source_sha256'][key]:raise ValueError('Frozen source mismatch: '+key)
        sources[key]=dict(path=path,sha256=actual,size_bytes=Path(path).stat().st_size)
    base=load_module(basepath,'h1_support_base')
    macro=load_module(macropath,'h1_support_macro')
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
        for c in ['start_date','end_date']:frame[c]=pd.to_datetime(frame[c],errors='raise')
        if frame.row_id.duplicated().any() or frame.row_id.isna().any():raise ValueError('Source row IDs invalid')
        if len(frame)!=plan.get('expected_primary_rows',221899):raise AssertionError('Frozen primary row count changed')
        if not macro.finite_rows(frame,['quantity_change_pct','net_sentiment']+controls).all():raise ValueError('Frozen source incomplete')
        support=support_table(frame)
        support.to_csv(root/'fund_support.csv',index=False,encoding='utf-8-sig')
        samples,removed=samples_from_support(frame,support)
        removed.to_csv(root/'removed_funds.csv',index=False,encoding='utf-8-sig')
        flow=dict(original_rows=len(frame),original_funds=int(frame.fund_id.nunique()),
            original_companies=int(frame.company_id.nunique()),
            support_counted_once_on_original_primary=True,iterative_filtering=False,
            fund_row_support=support.fund_rows.describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).to_dict(),
            fund_interval_support=support.fund_intervals.describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).to_dict(),
            single_row_funds=int(support.fund_rows.eq(1).sum()),
            single_interval_funds=int(support.fund_intervals.eq(1).sum()),
            distinction='Rows are fund-security-interval observations; interval support is unique interval_id, end dates separately saved.')
        flow['gross_control_rows_outside_declared_bounds_nonexclusive']={column:int((~pd.Series(np.isfinite(pd.to_numeric(frame[column],errors='coerce')),index=frame.index)
            | ~pd.to_numeric(frame[column],errors='coerce').between(0,1.001,inclusive='both')).sum())
            for column in ['opening_weight','closing_weight','closing_HHI']}
        flow['source_news_count_rows']={str(n):int(pd.to_numeric(frame.n_relevant,errors='coerce').lt(n).sum()) for n in [3,5]}
        save('sample_flow.json',flow)
        checkpoint('FWL_NET_INFORMATION')
        try:
            info,tables=fwl_information(frame,support,controls,base)
            for name,table in tables.items():table.to_csv(root/('FWL_information_'+name+'.csv'),index=False,encoding='utf-8-sig')
            info['table_files']={name:'FWL_information_'+name+'.csv' for name in tables}
        except Exception as error:
            info=dict(status='NOT_COMPUTED',reason=f'{type(error).__name__}: {error}',outcome_not_used=True)
        save('FWL_information.json',info)
        models=[];coefficients=[];exported={};native={}
        (root/'samples').mkdir(exist_ok=True)
        for name,sample in samples:
            checkpoint('FITTING_MODEL',model=name,rows=len(sample),funds=int(sample.fund_id.nunique()))
            path=root/'samples'/(name+'.parquet')
            sample.to_parquet(path,index=False,compression='zstd')
            exported[name]=dict(parquet=str(path),rows=len(sample),row_hash=row_hash(sample),exported_columns=list(sample.columns))
            issue=macro.model_status(sample,plan.get('minimum_model_rows',30))
            if issue:
                models.append(dict(model=name,status='NOT_ESTIMABLE',reason=issue,nobs=len(sample),
                    funds=int(sample.fund_id.nunique()),companies=int(sample.company_id.nunique())));continue
            try:
                fitted,result,terms,dropped=macro.fit(sample,controls,base)
                record,coefs=base.describe_result(fitted,result,name,'quantity_change_pct',terms,dropped)
                record['covariance_options']['cluster_time']=False
                record['conditional_entry_rows']=int(sample.entry_zero_to_positive.sum())
                record['complete_exit_rows']=int(sample.complete_exit.sum())
                record['distinct_fund_interval_pairs']=int(sample[['fund_id','interval_id']].drop_duplicates().shape[0])
                record['distinct_fund_end_date_pairs']=int(sample[['fund_id','end_date']].drop_duplicates().shape[0])
                f=result.f_statistic_robust
                record['joint_robust_F_inference']=dict(statistic=float(f.stat),native_p=float(f.pval),
                    status='VALID' if np.isfinite(f.stat) and f.stat>=0 and np.isfinite(f.pval) else 'UNDEFINED_COVARIANCE')
                if name==MODELS[0]:
                    if not np.isclose(result.params.net_sentiment,.5946851064715204,rtol=1e-7,atol=1e-8):raise AssertionError('Reference beta changed')
                    if not np.isclose(result.std_errors.net_sentiment,.3821674238484671,rtol=1e-7,atol=1e-8):raise AssertionError('Reference SE changed')
                    if not np.isclose(result.pvalues.net_sentiment,.11968924263465253,rtol=1e-7,atol=1e-8):raise AssertionError('Reference p changed')
            except Exception as error:
                if name==MODELS[0]:raise
                models.append(dict(model=name,status='NOT_ESTIMABLE',reason=f'{type(error).__name__}: {error}',
                    nobs=len(sample),funds=int(sample.fund_id.nunique()),companies=int(sample.company_id.nunique())));continue
            text=str(result.summary)
            (root/(name+'.summary.txt')).write_text(text,encoding='utf-8')
            (root/(name+'.summary.html')).write_text(result.summary.as_html(),encoding='utf-8')
            models.append(record);coefficients.extend(coefs);native[name]=text
        holm=holm_adjust(coefficients)
        pd.DataFrame(coefficients).to_csv(root/'coefficients.csv',index=False,encoding='utf-8-sig')
        comparison=[]
        for record in models:
            item={k:record.get(k) for k in ['model','status','nobs','funds','companies','rsquared_within',
                'distinct_fund_interval_pairs','distinct_fund_end_date_pairs','conditional_entry_rows','complete_exit_rows']}
            target=next((r for r in coefficients if r['model']==record['model'] and r['term']=='net_sentiment'),{})
            item.update({k:target.get(k) for k in ['beta','se_native','ci95_native_low','ci95_native_high','p_native','p_native_holm_support_family',
                'p_H1_positive_one_sided','p_H1_positive_one_sided_holm_support_family']})
            comparison.append(item)
        comparison=pd.DataFrame(comparison)
        comparison.to_csv(root/'comparison.csv',index=False,encoding='utf-8-sig')
        unchanged={key:sha(path)==sources[key]['sha256'] for key,path in paths.items()}
        if not all(unchanged.values()):raise AssertionError('A frozen source changed during execution')
        save('source_manifest.json',dict(sources=sources,source_hashes_unchanged_after_run=unchanged,
            runner_sha256=sha(__file__),plan_sha256=args.plan_sha))
        summary=dict(status='COMPLETE',specification=plan,plan_sha256=args.plan_sha,models=models,
            samples=exported,sample_flow=flow,holm_support_adjustment=holm,FWL_information=info,
            elapsed_seconds=time.time()-started,independent_QA='PENDING_SEPARATE_REVIEW',
            interpretation='Finite exploratory support/data-quality sensitivity. All eight new tests reported; no threshold selected by significance. Original threshold/outcome/controls/estimator unchanged.')
        save('summary.json',summary);render(root,summary,comparison,native)
        checkpoint('COMPLETE',models=len(models))
        save('completion.json',dict(status='COMPLETE',plan_sha256=args.plan_sha,
            elapsed_seconds=time.time()-started,finished_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        return 0
    except Exception as error:
        save('failure.json',dict(status='FAILED',message=str(error),traceback=traceback.format_exc(),plan_sha256=args.plan_sha));raise
    finally:lock.unlink(missing_ok=True)


def fixture(args):
    rng=np.random.default_rng(70112026);records=[]
    for fund,ninterval in enumerate([1,8,12,18,24,36,40,42]):
        for month in range(ninterval):
            for company in range(3):
                net,control,noise=rng.normal(size=3)
                records.append(dict(fund_id=str(fund),company_id=str(company),security_id=str(company),
                    row_id=f'r{fund}-{month}-{company}',interval_id=f'i{fund}-{month}',
                    start_date=pd.Timestamp('2020-01-31')+pd.offsets.MonthEnd(month),
                    end_date=pd.Timestamp('2020-01-31')+pd.offsets.MonthEnd(month+1),
                    net_sentiment=net,control=control,absorbed=float(fund),n_relevant=1+month%10,
                    opening_weight=.1,closing_weight=.11,closing_HHI=.2,
                    quantity_change_pct=2.5*net+.4*control+.6*fund+noise,
                    opening_passive_explicit=False,closing_passive_explicit=False,closing_full_portfolio=True))
    frame=ordered(pd.DataFrame(records));support=support_table(frame)
    frame.loc[0,'closing_weight']=1.002
    samples,removed=samples_from_support(frame,support)
    assert len(samples)==9 and support.loc[support.fund_id.eq('0'),'fund_rows'].iloc[0]==3
    assert support.loc[support.fund_id.eq('0'),'fund_intervals'].iloc[0]==1
    for name,column,minimum in RULES:
        sample=dict(samples)[name]
        selected=support if column is None else support.loc[support[column].ge(minimum)]
        assert set(sample.fund_id)==set(selected.fund_id)
        assert set(sample.row_id).issubset(set(frame.row_id))
    assert len(dict(samples)['BOUNDED_GROSS_CONTROLS'])==len(frame)-1
    assert dict(samples)['NEWS_MIN_3'].n_relevant.ge(3).all()
    assert dict(samples)['NEWS_MIN_5'].n_relevant.ge(5).all()
    rows=[dict(model=name,term='net_sentiment',p_native=p,p_H1_positive_one_sided=p/2) for name,p in zip(MODELS[1:],[.01,.02,.04,.2,.7,.8,.9,.99])]
    h=holm_adjust(rows)
    assert np.isclose(h['p_native']['adjusted_pvalues'][MODELS[1]],.08)
    assert np.isclose(h['p_native']['adjusted_pvalues'][MODELS[2]],.14)
    assert np.isclose(h['p_H1_positive_one_sided']['adjusted_pvalues'][MODELS[1]],.04)
    base=load_module(args.base_helper or Path(__file__).with_name('run_h1_net_fund_fe_20261030.py'),'h1_support_fixture_base')
    macro=load_module(args.macro_runner_helper or Path(__file__).with_name('run_h1_macro_no_time_20261031.py'),'h1_support_fixture_macro')
    info,tables=fwl_information(frame,support,['control','absorbed'],base)
    assert info['outcome_not_used'] and info['total_residual_net_SS']>0
    assert info['retained_control_terms']==['control']
    for table in tables.values():assert np.isclose(table.information_share.sum(),1)
    changed=frame.copy();changed['quantity_change_pct']=rng.normal(size=len(changed))*100000
    other,others=fwl_information(changed,support,['control','absorbed'],base)
    assert info['total_residual_net_SS']==other['total_residual_net_SS']
    sample,result,terms,dropped=macro.fit(frame,['control','absorbed'],base)
    within=sample[['quantity_change_pct']+terms]-sample[['quantity_change_pct']+terms].groupby(sample.fund_id).transform('mean')
    independent=np.linalg.lstsq(within[terms],within.quantity_change_pct,rcond=None)[0]
    independent_cov=base.cluster_covariance_fixture(sample,within[terms].to_numpy(),result.resids.to_numpy(),result.df_resid)
    assert np.allclose(result.params[terms],independent,rtol=1e-10,atol=1e-10)
    assert np.allclose(result.cov.loc[terms,terms],independent_cov,rtol=1e-10,atol=1e-10)
    assert result.included_effects==['Entity']
    print(json.dumps(dict(status='PASS',scope='synthetic_only',rows=len(frame),models=len(samples),
        fixed_support_masks=True,Holm_8_tests=True,FWL_outcome_independent=True,native_covariance_verified=True)))
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan');parser.add_argument('--plan-sha')
    parser.add_argument('--base-helper');parser.add_argument('--macro-runner-helper')
    parser.add_argument('--fixture-only',action='store_true')
    args=parser.parse_args()
    if args.fixture_only:return fixture(args)
    if not args.plan or not args.plan_sha:parser.error('--plan and --plan-sha are required')
    return execute(args)


if __name__=='__main__':raise SystemExit(main())
