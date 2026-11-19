"""Finite AND-combinations of H1 fund support; actual execution on research-host only.

Every combination is reported. Identical samples may reuse exactly validated
prior estimates; cache reuse is disclosed and is not independent replication.
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
JOINT=[f'JOINT_ROWS_{rows}_INTERVALS_{intervals}' for rows in [30,60] for intervals in [12,24,36]]
MODELS=[REFERENCE]+JOINT
MICRO=['opening_weight','closing_log_NAV','closing_HHI','log_news_count_interval','log_interval_days','bond_indicator']
MACRO=['Imoex','imoex_ret','Ruonia','USD_RUB','RVI','Flow_billion_RUB','unique_funds','dummy_mar2014','dummy_feb2022','dummy_sep2018']


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


def model_samples(frame,support):
    samples=[(REFERENCE,ordered(frame),None,None)]
    for rows in [30,60]:
        for intervals in [12,24,36]:
            keep=support.fund_rows.ge(rows)&support.fund_intervals.ge(intervals)
            funds=set(support.loc[keep,'fund_id'])
            samples.append((f'JOINT_ROWS_{rows}_INTERVALS_{intervals}',ordered(frame.loc[frame.fund_id.isin(funds)]),rows,intervals))
    return samples


def holm(values):
    """Preserve every frozen hypothesis slot, including duplicate sample masks."""
    items=[]
    for name,p in values:
        valid=p is not None and np.isfinite(p) and 0<=p<=1
        items.append((name,float(p) if valid else 1.,valid))
    maximum=0.;result={}
    for rank,(name,p,valid) in enumerate(sorted(items,key=lambda x:x[1])):
        maximum=max(maximum,min(1.,(len(items)-rank)*p));result[name]=maximum if valid else None
    return result


def adjust_families(coefficients,previous_coefficients,previous_names,previous_hashes,current_hashes):
    values=[]
    for name in JOINT:
        target=next((r for r in coefficients if r['model']==name and r['term']=='net_sentiment'),{})
        values.append((name,target.get('p_native')))
    previous=[]
    for name in previous_names:
        target=previous_coefficients.loc[previous_coefficients.model.eq(name)&previous_coefficients.term.eq('net_sentiment')]
        previous.append(('PREVIOUS::'+name,float(target.p_native.iloc[0]) if len(target)==1 else None))
    if len(previous)!=8:raise ValueError('Append family requires exactly eight prior tests')
    within=holm(values);appended=holm(previous+values)
    by_hash={};name_hash={}
    for name,p in previous+values:
        key=previous_hashes[name.removeprefix('PREVIOUS::')] if name.startswith('PREVIOUS::') else current_hashes[name]
        name_hash[name]=key
        if key not in by_hash:by_hash[key]=dict(representative=name,p_native=p,aliases=[name])
        else:
            item=by_hash[key]
            old=item['p_native']
            both_missing=(old is None or not np.isfinite(old)) and (p is None or not np.isfinite(p))
            if not both_missing and not np.isclose(old,p,rtol=1e-10,atol=1e-12):
                raise AssertionError('Identical sample/test has different p-values: '+name)
            item['aliases'].append(name)
    unique_adjusted=holm([(key,item['p_native']) for key,item in by_hash.items()])
    unique_by_name={name:unique_adjusted[key] for name,key in name_hash.items()}
    for row in coefficients:
        row['p_native_holm_joint6']=within.get(row['model']) if row['term']=='net_sentiment' else None
        row['p_native_holm_append14']=appended.get(row['model']) if row['term']=='net_sentiment' else None
        row['p_native_holm_unique_pooled']=unique_by_name.get(row['model']) if row['term']=='net_sentiment' else None
        row['unique_family_size']=len(by_hash) if row['term']=='net_sentiment' and row['model'] in JOINT else None
    return dict(method='Holm step-down on two-sided native p; no reference hypotheses',
        primary_pooled_unique=dict(family_size=len(by_hash),deduplication_key='Exact source-row hash; same outcome/controls/estimator/test',
            members=[dict(row_hash=key,**item,p_native_holm=unique_adjusted[key]) for key,item in by_hash.items()],
            adjusted_pvalues_by_model=unique_by_name),
        within_joint6=dict(family=[name for name,_ in values],adjusted_pvalues=within),
        appended14=dict(family=[name for name,_ in previous+values],adjusted_pvalues=appended),
        duplicate_sample_masks_keep_all_frozen_slots=True,
        caveat='Exploratory append correction for these 14 recorded sensitivity hypotheses; not a correction for every earlier research choice.')


def validate_estimator(record,controls):
    options=record.get('covariance_options',{})
    needed=dict(cov_type='clustered',clusters=['fund_id','company_id'],debiased=True,
                auto_df=False,count_effects=True,group_debias=True,cluster_time=False)
    if any(options.get(key)!=value for key,value in needed.items()):return False
    if record.get('included_effects')!=['Entity'] or record.get('time_effects') is not False:return False
    if record.get('other_effects') is not False or record.get('weights') is not None:return False
    retained=record.get('retained_terms',[])
    dropped=[item['term'] for item in record.get('dropped_controls',[])]
    return 'net_sentiment' in retained and set(retained+ dropped)==set(['net_sentiment']+controls)


def read_cache(sample,key,previous,previous_coefficients,prior_root,controls,track,cache_checks):
    """Validate IDs, all used Y/X values and estimator provenance before reuse."""
    for record in previous['models']:
        name=record['model'];metadata=previous.get('samples',{}).get(name,{})
        if record.get('status')!='ESTIMATED' or metadata.get('row_hash')!=key or metadata.get('rows')!=len(sample):continue
        if record.get('source_row_hash')!=key or not validate_estimator(record,controls):continue
        cachepath=Path(metadata['parquet']);numeric=['quantity_change_pct','quantity_open','quantity_close','net_sentiment']+controls
        cached=pd.read_parquet(cachepath,columns=['row_id']+numeric)
        same_ids=np.array_equal(cached.row_id.astype(str),sample.row_id.astype(str))
        same_values=all(np.array_equal(cached[column].to_numpy(float),sample[column].to_numpy(float),equal_nan=True) for column in numeric)
        if not same_ids or row_hash(cached)!=key or not same_values:
            raise AssertionError('Prior cache metadata falsely matches an unequal sample: '+name)
        subset=previous_coefficients.loc[previous_coefficients.model.eq(name)].copy()
        if set(subset.term)!=set(['const']+record['retained_terms']) or subset.term.duplicated().any():
            raise AssertionError('Prior cache coefficient schema is inconsistent')
        native_txt=prior_root/(name+'.summary.txt');native_html=prior_root/(name+'.summary.html')
        for path in [cachepath,native_txt,native_html]:track(path)
        cache_checks.append(dict(source_model=name,row_hash=key,rows=len(sample),
            exact_row_ids=True,exact_Y_X_quantities=True,estimator_options_equal=True,
            cached_sample=str(cachepath),cached_native_text=str(native_txt)))
        return copy.deepcopy(record),subset.to_dict('records'),native_txt.read_text(encoding='utf-8'),native_html.read_text(encoding='utf-8'),name
    return None


def render(root,summary,comparison,native):
    parts=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>H1: совместные ограничения наблюдений</title>',
        '<style>body{font:16px system-ui;max-width:1450px;margin:32px auto;padding:0 20px;color:#17212d}table{border-collapse:collapse;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}pre{padding:16px;background:#f5f6f8;overflow:auto;font:13px monospace}a{color:#075ba5}</style>',
        '<h1>H1: число строк и отчётных интервалов одновременно</h1>',
        '<p>Зафиксированы шесть комбинаций: ≥30/≥60 строк фонда И ≥12/≥24/≥36 различных фактических интервалов. Все количества посчитаны один раз на прежней полной основной панели; фильтры применены независимо к ней. Порог P99, условные +100% для входов, −100% для выходов, все контроли, fund FE и кластеризация фонд/компания сохранены. Временных эффектов/кластеров нет.</p>',
        '<p>Показаны все шесть результатов; модель с минимальным p не назначается основной. Равные row hashes означают одну и ту же выборку; повторное использование её результата — кэш, а не независимая репликация. Основная поправка Holm охватывает 11 уникальных выборок прежних и новых проверок. Дополнительно сохранены поправки для шести текущих и консервативные 14 слотов без удаления повторов. Эта конечная серия остаётся исследовательской.</p>',
        comparison.to_html(index=False,border=0,float_format=lambda x:f'{x:.7g}'),
        '<p><a href="summary.json">Все метрики</a> · <a href="coefficients.csv">Все коэффициенты</a> · <a href="fund_support.csv">Наблюдения фондов</a> · <a href="mask_catalog.json">Маски, совпадения и кэш</a> · <a href="multiplicity.json">Holm 6/14</a> · <a href="source_manifest.json">Источники</a></p>']
    for name,text in native.items():parts.append('<h2>'+html.escape(name)+'</h2><pre>'+html.escape(text)+'</pre>')
    parts.append('</html>');(root/'report.html').write_text('\n'.join(parts),encoding='utf-8')


def execute(args):
    if sys.platform!='win32':raise RuntimeError('Actual research-data execution must run on research-host')
    planpath=Path(args.plan)
    if sha(planpath)!=args.plan_sha:raise ValueError('Frozen plan SHA256 mismatch')
    plan=json.loads(planpath.read_text(encoding='utf-8'))
    if plan['planned_models']!=MODELS:raise ValueError('Frozen six-combination suite changed')
    controls=plan['baseline_controls']+plan['macro_controls']
    if controls!=MICRO+MACRO or plan.get('time_FE') or plan.get('cluster_time') or plan.get('clusters')!=['fund_id','company_id']:
        raise ValueError('Frozen estimator/control conventions changed')
    root=Path(plan['output_root'])
    if any(part.lower()=='databaseresources' for part in root.parts):raise ValueError('Protected output location')
    root.mkdir(parents=True,exist_ok=True)
    if (root/'completion.json').exists():raise FileExistsError('Completed results may not be overwritten')
    paths={key:plan[key] for key in ['input_panel','base_helper','macro_runner_helper','prior_support_summary','prior_support_coefficients']}
    if args.base_helper:paths['base_helper']=args.base_helper
    if args.macro_runner_helper:paths['macro_runner_helper']=args.macro_runner_helper
    sources={}
    for key,path in paths.items():
        actual=sha(path)
        if actual!=plan['expected_source_sha256'][key]:raise ValueError('Frozen source mismatch: '+key)
        sources[key]=dict(path=path,sha256=actual,size_bytes=Path(path).stat().st_size)
    used_cache_files={}
    def track(path):
        path=Path(path);key=str(path)
        if key not in used_cache_files:used_cache_files[key]=dict(path=key,sha256=sha(path),size_bytes=path.stat().st_size)
    base=load_module(paths['base_helper'],'h1_joint_base');macro=load_module(paths['macro_runner_helper'],'h1_joint_macro')
    previous=json.loads(Path(paths['prior_support_summary']).read_text(encoding='utf-8'))
    previous_coefficients=pd.read_csv(paths['prior_support_coefficients'])
    prior_root=Path(paths['prior_support_summary']).parent
    previous_plan=previous['specification']
    for key in ['input_panel','base_helper','macro_runner_helper']:
        if previous_plan['expected_source_sha256'][key]!=plan['expected_source_sha256'][key]:
            raise ValueError('Cache comes from different source/helper version: '+key)
    if previous_plan['baseline_controls']+previous_plan['macro_controls']!=controls:raise ValueError('Cached controls changed')
    previous_names=previous_plan['planned_models'][1:]
    if len(previous_names)!=8:raise ValueError('Previous family must contain eight new tests')
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
        if len(frame)!=plan.get('expected_primary_rows',221899) or frame.row_id.duplicated().any():raise AssertionError('Frozen primary universe changed')
        if not macro.finite_rows(frame,['quantity_change_pct','net_sentiment']+controls).all():raise ValueError('Incomplete source sample')
        support=support_table(frame);support.to_csv(root/'fund_support.csv',index=False,encoding='utf-8-sig')
        samples=model_samples(frame,support)
        records=[];coefficients=[];metadata={};native={};catalog=[];cache_checks=[];representatives={};fit_count=0;cache_count=0
        (root/'samples').mkdir(exist_ok=True)
        for name,sample,minrows,minintervals in samples:
            key=row_hash(sample);alias=representatives.get(key,{}).get('model')
            if name in plan.get('prefit_sample_counts',{}):
                expected=plan['prefit_sample_counts'][name]
                if len(sample)!=expected['rows'] or sample.fund_id.nunique()!=expected['funds']:
                    raise AssertionError('AND support sample differs from independent prefit counts: '+name)
            checkpoint('PROCESSING_MODEL',model=name,rows=len(sample),funds=int(sample.fund_id.nunique()),alias_of=alias)
            samplepath=root/'samples'/(name+'.parquet');sample.to_parquet(samplepath,index=False,compression='zstd')
            metadata[name]=dict(parquet=str(samplepath),rows=len(sample),row_hash=key,exported_columns=list(sample.columns))
            item=dict(model=name,min_fund_rows=minrows,min_fund_intervals=minintervals,operator='AND',
                rows=len(sample),funds=int(sample.fund_id.nunique()),companies=int(sample.company_id.nunique()),row_hash=key,alias_of=alias)
            if alias:
                cached=representatives[key]
                record=copy.deepcopy(cached['record']);coefs=copy.deepcopy(cached['coefficients'])
                text=cached['text'];htmltext=cached['html'];cache_source=cached['cache_source']
                origin='CURRENT_EXACT_SAMPLE_ALIAS'
            else:
                cached=read_cache(sample,key,previous,previous_coefficients,prior_root,controls,track,cache_checks)
                if cached:
                    record,coefs,text,htmltext,cache_source=cached;cache_count+=1;origin='VALIDATED_PRIOR_EXACT_SAMPLE_CACHE'
                else:
                    issue=macro.model_status(sample,plan.get('minimum_model_rows',30))
                    if issue:
                        record=dict(model=name,status='NOT_ESTIMABLE',reason=issue,nobs=len(sample),funds=int(sample.fund_id.nunique()),companies=int(sample.company_id.nunique()))
                        coefs=[];text='NOT_ESTIMABLE: '+issue;htmltext='<pre>'+html.escape(text)+'</pre>';cache_source=None;origin='NOT_ESTIMABLE'
                    else:
                        fit_count+=1
                        try:
                            fitted,result,terms,dropped=macro.fit(sample,controls,base)
                            record,coefs=base.describe_result(fitted,result,name,'quantity_change_pct',terms,dropped)
                            record['covariance_options']['cluster_time']=False
                            text=str(result.summary);htmltext=result.summary.as_html();cache_source=None;origin='NEW_UNIQUE_SAMPLE_FIT'
                        except Exception as error:
                            if name==REFERENCE:raise
                            record=dict(model=name,status='NOT_ESTIMABLE',reason=f'{type(error).__name__}: {error}',nobs=len(sample),funds=int(sample.fund_id.nunique()),companies=int(sample.company_id.nunique()))
                            coefs=[];text=record['reason'];htmltext='<pre>'+html.escape(text)+'</pre>';cache_source=None;origin='NOT_ESTIMABLE'
                representatives[key]=dict(model=name,record=copy.deepcopy(record),coefficients=copy.deepcopy(coefs),text=text,html=htmltext,cache_source=cache_source)
            record['model']=name
            record['result_origin']=origin;record['exact_sample_alias_of']=alias
            record['prior_cached_model']=cache_source
            record['conditional_entry_rows']=int(sample.entry_zero_to_positive.sum())
            record['complete_exit_rows']=int(sample.complete_exit.sum())
            record['distinct_fund_interval_pairs']=int(sample[['fund_id','interval_id']].drop_duplicates().shape[0])
            record['distinct_fund_end_date_pairs']=int(sample[['fund_id','end_date']].drop_duplicates().shape[0])
            for row in coefs:
                row['model']=name
                for column in ['p_native_holm_support_family','p_H1_positive_one_sided_holm_support_family']:
                    row.pop(column,None)
            if name==REFERENCE:
                target=next((row for row in coefs if row['term']=='net_sentiment'),{})
                for field,expected in [('beta',.5946851064715204),('se_native',.3821674238484671),('p_native',.11968924263465253)]:
                    if not np.isclose(target.get(field,np.nan),expected,rtol=1e-7,atol=1e-8):raise AssertionError('Reference changed: '+field)
            item.update(result_origin=origin,prior_cached_model=cache_source)
            catalog.append(item);records.append(record);coefficients.extend(coefs);native[name]=text
            (root/(name+'.summary.txt')).write_text(text,encoding='utf-8')
            (root/(name+'.summary.html')).write_text(htmltext,encoding='utf-8')
        previous_hashes={name:previous['samples'][name]['row_hash'] for name in previous_names}
        current_hashes={item['model']:item['row_hash'] for item in catalog if item['model']!=REFERENCE}
        adjustment=adjust_families(coefficients,previous_coefficients,previous_names,previous_hashes,current_hashes)
        if adjustment['primary_pooled_unique']['family_size']!=plan['multiple_testing']['expected_unique_pooled_tests']:
            raise AssertionError('Unique pooled family differs from prefit deduplication')
        if fit_count!=plan['prefit_new_unique_masks']:
            raise AssertionError('Unexpected number of new unique masks fitted; exact cache checks did not match frozen preflight')
        save('multiplicity.json',adjustment);save('mask_catalog.json',catalog)
        pd.DataFrame(coefficients).to_csv(root/'coefficients.csv',index=False,encoding='utf-8-sig')
        comparison=[]
        for record in records:
            item={key:record.get(key) for key in ['model','status','nobs','funds','companies','rsquared_within',
                'result_origin','exact_sample_alias_of','prior_cached_model','conditional_entry_rows','complete_exit_rows']}
            target=next((row for row in coefficients if row['model']==record['model'] and row['term']=='net_sentiment'),{})
            item.update({key:target.get(key) for key in ['beta','se_native','ci95_native_low','ci95_native_high','p_native','p_native_holm_unique_pooled','unique_family_size','p_native_holm_joint6','p_native_holm_append14']})
            comparison.append(item)
        comparison=pd.DataFrame(comparison);comparison.to_csv(root/'comparison.csv',index=False,encoding='utf-8-sig')
        unchanged={key:sha(path)==sources[key]['sha256'] for key,path in paths.items()}
        cache_unchanged={path:sha(path)==info['sha256'] for path,info in used_cache_files.items()}
        if not all(unchanged.values()) or not all(cache_unchanged.values()):raise AssertionError('A source/cache changed during execution')
        save('source_manifest.json',dict(sources=sources,source_hashes_unchanged_after_run=unchanged,
            used_cache_files=used_cache_files,cache_hashes_unchanged_after_run=cache_unchanged,
            plan_sha256=args.plan_sha,runner_sha256=sha(__file__)))
        summary=dict(status='COMPLETE',specification=plan,plan_sha256=args.plan_sha,models=records,samples=metadata,
            multiplicity=adjustment,mask_catalog=catalog,cache_checks=cache_checks,new_unique_sample_fits=fit_count,
            validated_prior_cache_reuses=cache_count,unique_samples=len(representatives),elapsed_seconds=time.time()-started,
            support_counted_once_on_original_primary=True,iterative_filtering=False,independent_QA='PENDING_SEPARATE_REVIEW',
            interpretation='All six frozen AND combinations reported. Identical masks/cache are not independent replication. No winning model selected; no significance guarantee.')
        save('summary.json',summary);render(root,summary,comparison,native)
        checkpoint('COMPLETE',new_unique_sample_fits=fit_count,validated_prior_cache_reuses=cache_count)
        save('completion.json',dict(status='COMPLETE',plan_sha256=args.plan_sha,elapsed_seconds=time.time()-started,
            finished_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        return 0
    except Exception as error:
        save('failure.json',dict(status='FAILED',message=str(error),traceback=traceback.format_exc(),plan_sha256=args.plan_sha));raise
    finally:lock.unlink(missing_ok=True)


def fixture(args):
    records=[]
    for fund,(intervals,securities) in enumerate([(1,1),(12,3),(24,1),(24,3),(36,2),(40,4)]):
        for interval in range(intervals):
            for company in range(securities):
                records.append(dict(fund_id=str(fund),company_id=str(company),security_id=str(company),
                    row_id=f'r{fund}-{interval}-{company}',interval_id=f'i{fund}-{interval}',
                    end_date=pd.Timestamp('2020-01-31')+pd.offsets.MonthEnd(interval)))
    frame=ordered(pd.DataFrame(records));support=support_table(frame);samples=model_samples(frame,support)
    assert len(samples)==7
    for name,sample,rows,intervals in samples[1:]:
        independent={fund for fund,group in frame.groupby('fund_id') if len(group)>=rows and group.interval_id.nunique()>=intervals}
        assert set(sample.fund_id)==independent
    assert row_hash(dict((name,sample)for name,sample,_,_ in samples)['JOINT_ROWS_30_INTERVALS_36'])==row_hash(dict((name,sample)for name,sample,_,_ in samples)['JOINT_ROWS_60_INTERVALS_36'])
    p=holm([('a',.01),('b',.01),('c',.2)])
    assert p['a']==p['b']==.03 and p['c']==.2
    coefs=[dict(model=name,term='net_sentiment',p_native=p)for name,p in zip(JOINT,[.01,.11,.12,.04,.05,.12])]
    prior=pd.DataFrame([dict(model=f'P{i}',term='net_sentiment',p_native=.1+i*.01)for i in range(8)])
    previous_hashes={f'P{i}':f'p{i}'for i in range(8)}
    current_hashes=dict(zip(JOINT,['q0','p1','p2','q3','q4','p2']))
    adjusted=adjust_families(coefs,prior,[f'P{i}'for i in range(8)],previous_hashes,current_hashes)
    assert len(adjusted['within_joint6']['family'])==6 and len(adjusted['appended14']['family'])==14
    assert np.isclose(coefs[0]['p_native_holm_joint6'],.06) and np.isclose(coefs[0]['p_native_holm_append14'],.14)
    assert adjusted['primary_pooled_unique']['family_size']==11
    assert np.isclose(coefs[0]['p_native_holm_unique_pooled'],.11)
    assert coefs[2]['p_native_holm_unique_pooled']==coefs[5]['p_native_holm_unique_pooled']
    record=dict(covariance_options=dict(cov_type='clustered',clusters=['fund_id','company_id'],debiased=True,auto_df=False,count_effects=True,group_debias=True,cluster_time=False),
        included_effects=['Entity'],time_effects=False,other_effects=False,weights=None,retained_terms=['net_sentiment','x'],dropped_controls=[])
    assert validate_estimator(record,['x'])
    record['covariance_options']['cluster_time']=True
    assert not validate_estimator(record,['x'])
    import tempfile
    record['covariance_options']['cluster_time']=False
    with tempfile.TemporaryDirectory(prefix='joint-h1-cache-fixture-') as temporary:
        directory=Path(temporary)
        cached=pd.DataFrame(dict(row_id=['a','b'],quantity_change_pct=[10.,20.],quantity_open=[10.,10.],
            quantity_close=[11.,12.],net_sentiment=[-.2,.4],x=[1.,2.]))
        key=row_hash(cached);samplepath=directory/'prior.parquet';cached.to_parquet(samplepath,index=False)
        record.update(model='P',status='ESTIMATED',source_row_hash=key)
        previous=dict(models=[record],samples={'P':dict(parquet=str(samplepath),rows=2,row_hash=key)})
        previous_coefficients=pd.DataFrame([dict(model='P',term=term,beta=0.,se_native=1.,p_native=.5)for term in ['const','net_sentiment','x']])
        (directory/'P.summary.txt').write_text('synthetic native summary',encoding='utf-8')
        (directory/'P.summary.html').write_text('<pre>synthetic native summary</pre>',encoding='utf-8')
        tracked=[];checks=[]
        hit=read_cache(cached,key,previous,previous_coefficients,directory,['x'],lambda path:tracked.append(str(path)),checks)
        assert hit is not None and hit[-1]=='P' and len(tracked)==3 and checks[0]['exact_Y_X_quantities']
        altered=cached.copy();altered.loc[0,'quantity_change_pct']=99.
        try:read_cache(altered,key,previous,previous_coefficients,directory,['x'],lambda path:None,[])
        except AssertionError:pass
        else:raise AssertionError('Cache accepted unequal Y despite identical row IDs')
    print(json.dumps(dict(status='PASS',scope='synthetic_only',models=len(samples),AND_masks=True,identical_hash_alias=True,
        Holm_unique11_and_slots6_14=True,cache_estimator_guard=True,exact_cache_Y_X_guard=True)))
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan');parser.add_argument('--plan-sha')
    parser.add_argument('--base-helper');parser.add_argument('--macro-runner-helper');parser.add_argument('--fixture-only',action='store_true')
    args=parser.parse_args()
    if args.fixture_only:return fixture(args)
    if not args.plan or not args.plan_sha:parser.error('--plan and --plan-sha are required')
    return execute(args)


if __name__=='__main__':raise SystemExit(main())
