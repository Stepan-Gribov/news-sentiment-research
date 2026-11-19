"""H2 signed-part panel models and absolute-asymmetry intersection-union tests.

Real research execution requires a final frozen plan on research-host. The fixture
branch only constructs synthetic observations and does not inspect real data.
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


CURRENT='net_sentiment'
LAG='net_sentiment_lag1'
PARTS=['net_positive','net_negative','net_positive_lag1','net_negative_lag1']
MICRO=['opening_weight','closing_log_NAV','closing_HHI','log_news_count_interval','log_interval_days','bond_indicator']
MACRO=['Imoex','imoex_ret','Ruonia','USD_RUB','RVI','Flow_billion_RUB','unique_funds','dummy_mar2014','dummy_feb2022','dummy_sep2018']
VARIANTS=['CURRENT_FULL','CURRENT_COMMON','CURRENT_PLUS_LAG1']
SOURCE_KEYS=['input_panel','lag_map','lag_helper','focus_summary','focus_support','focus_helper','cutoff_reference_sample','base_helper']
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


def add_signed_parts(frame):
    d=frame.copy()
    for source,positive,negative in [(CURRENT,PARTS[0],PARTS[1]),(LAG,PARTS[2],PARTS[3])]:
        values=pd.to_numeric(d[source],errors='coerce')
        d[positive]=values.clip(lower=0)
        d[negative]=(-values).clip(lower=0)
        valid=np.isfinite(values)
        if not np.array_equal((d.loc[valid,positive]-d.loc[valid,negative]).to_numpy(),values.loc[valid].to_numpy()):
            raise AssertionError('Signed parts no longer reconstruct their original score')
        if not d.loc[~valid,[positive,negative]].isna().all().all():
            raise AssertionError('Missing sentiment was changed to zero')
    for column in frame:
        if not d[column].equals(frame[column]):raise AssertionError('Signed transformation changed source column: '+column)
    return d


def select_regressors(sample,targets,controls):
    """Same declared-order fund-within rank rule, with every sign target required."""
    names=list(targets)+list(controls)
    if len(set(names))!=len(names):raise ValueError('Duplicate declared regressors')
    values=sample[names].astype(float)
    within=values-values.groupby(sample.fund_id.astype(str),sort=False).transform('mean')
    tolerance=16*np.finfo(float).eps*max(len(sample),len(names))
    kept=[];dropped=[];basis=[]
    for name in names:
        raw=within[name].to_numpy(float);norm=float(np.linalg.norm(raw))
        absorbed=sample.groupby('fund_id',observed=True)[name].nunique().max()<=1
        if absorbed or norm==0:
            if name in targets:raise ValueError('Required H2 sign target is absorbed by fund effects: '+name)
            dropped.append(dict(term=name,reason='constant_within_every_fund'));continue
        vector=raw/norm
        residual=vector if not basis else vector-np.column_stack(basis)@np.linalg.lstsq(np.column_stack(basis),vector,rcond=None)[0]
        residual_norm=float(np.linalg.norm(residual))
        if residual_norm<=tolerance:
            if name in targets:raise ValueError('Required H2 sign target is collinear: '+name)
            dropped.append(dict(term=name,reason='deterministic_within_linear_dependence',relative_residual_norm=residual_norm,
                tolerance=tolerance,retained_predecessors=kept.copy()));continue
        kept.append(name);basis.append(vector)
    return kept,dropped


def fit_native(sample,controls,distributed):
    from linearmodels.panel import PanelOLS
    sample=ordered(sample);targets=PARTS if distributed else PARTS[:2]
    terms,dropped=select_regressors(sample,targets,controls)
    index=pd.MultiIndex.from_arrays([sample.fund_id.astype(str),pd.to_datetime(sample.end_date)],names=['fund_id','report_date'])
    x=sample[terms].astype(float).copy();x.index=index;x.insert(0,'const',1.)
    y=pd.Series(sample.quantity_change_pct.to_numpy(float),index=index,name='quantity_change_pct')
    clusters=pd.DataFrame({'fund_cluster':pd.Categorical(sample.fund_id.astype(str)).codes,
        'company_cluster':pd.Categorical(sample.company_id.astype(str)).codes},index=index)
    result=PanelOLS(y,x,entity_effects=True,time_effects=False,other_effects=None,singletons=True,
        drop_absorbed=False,check_rank=True).fit(cov_type='clustered',clusters=clusters,
        cluster_time=False,cluster_entity=False,group_debias=True,debiased=True,auto_df=False,count_effects=True)
    if result.nobs!=len(sample)or not result.resids.index.equals(index):raise AssertionError('Estimator changed rows/order')
    return sample,result,terms,dropped


def linear_contrast(beta,covariance,vector,df):
    from scipy import stats
    beta=np.asarray(beta,float);covariance=np.asarray(covariance,float);vector=np.asarray(vector,float)
    estimate=float(vector@beta);variance=float(vector@covariance@vector)
    record=dict(status='UNDEFINED_COVARIANCE',beta=estimate,variance=variance,se_native=None,
        t_native=None,p_native=None,ci95_native_low=None,ci95_native_high=None,native_df_resid=int(df),
        contrast=vector.tolist(),covariance_repair=False)
    if np.isfinite(estimate)and np.isfinite(variance)and variance>0 and df>0:
        se=float(np.sqrt(variance));t=estimate/se;critical=float(stats.t.ppf(.975,df))
        record.update(status='ESTIMATED',se_native=se,t_native=t,p_native=float(2*stats.t.sf(abs(t),df)),
            ci95_native_low=estimate-critical*se,ci95_native_high=estimate+critical*se)
    return record


def absolute_asymmetry(beta,covariance,df):
    """H0 |b+|=|b-| is the union b+-b-=0 OR b++b-=0.

    Reject the union only when both component two-sided tests reject. The max
    of their p-values is the intersection-union p-value, without sign choice.
    """
    beta=np.asarray(beta,float);covariance=np.asarray(covariance,float)
    diff=linear_contrast(beta,covariance,[1.,-1.],df)
    summed=linear_contrast(beta,covariance,[1.,1.],df)
    valid=diff['status']=='ESTIMATED'and summed['status']=='ESTIMATED'
    magnitude=float(abs(beta[1])-abs(beta[0]))
    expected=bool(beta[0]>=0 and beta[1]<=0)
    return dict(status='ESTIMATED'if valid else'UNDEFINED_COVARIANCE',
        beta_positive=float(beta[0]),beta_negative=float(beta[1]),target_covariance=covariance.tolist(),
        expected_directional_signs=expected,negative_absolute_minus_positive_absolute=magnitude,
        negative_absolute_stronger=bool(magnitude>0),
        difference_zero=diff,sum_zero=summed,p_abs=max(diff['p_native'],summed['p_native'])if valid else None,
        null='abs(beta_positive)=abs(beta_negative); union of difference_zero and sum_zero',
        test='Intersection-union: p_abs=max(two-sided p_difference,two-sided p_sum)',
        zero_or_negative_variance_not_repaired=True,
        interpretation='Rejection means unequal absolute slopes, not automatically stronger negative-news response or the expected signs.')


def h2_tests(result,distributed):
    terms=PARTS if distributed else PARTS[:2]
    beta=result.params.loc[terms].to_numpy(float);covariance=result.cov.loc[terms,terms].to_numpy(float)
    matrices={'current':np.asarray([[1.,0.],[0.,1.]])}
    if distributed:
        matrices={'current':np.asarray([[1.,0.,0.,0.],[0.,1.,0.,0.]]),
            'lag1':np.asarray([[0.,0.,1.,0.],[0.,0.,0.,1.]]),
            'total_current_plus_lag1':np.asarray([[1.,0.,1.,0.],[0.,1.,0.,1.]])}
    tests={}
    for name,matrix in matrices.items():
        b=matrix@beta;v=matrix@covariance@matrix.T
        test=absolute_asymmetry(b,v,result.df_resid)
        test['positive_coefficient']=linear_contrast(b,v,[1.,0.],result.df_resid)
        test['negative_coefficient']=linear_contrast(b,v,[0.,1.],result.df_resid)
        test['transformation_from_full_target_block']=matrix.tolist();tests[name]=test
    return dict(target_terms=terms,full_target_covariance=covariance.tolist(),tests=tests,
        native_df_resid=int(result.df_resid),signs_always_reported=True,covariance_repair=False)


def cutoff_masks(frame,reference,cutoffs,expected_population_rows=204152):
    population=reference.loc[reference.quantity_open.gt(0)&reference.quantity_close.gt(0)]
    if len(population)!=expected_population_rows:raise AssertionError('Frozen pre-entry continuing quantile population changed')
    q0=pd.to_numeric(frame.quantity_open,errors='coerce');q1=pd.to_numeric(frame.quantity_close,errors='coerce')
    continuing=q0.gt(0)&q1.gt(0);entries=q0.eq(0)&q1.gt(0);exits=q0.gt(0)&q1.eq(0)
    if not(continuing|entries|exits).all():raise AssertionError('Unexpected quantity state in frozen primary')
    if not frame.loc[entries,'quantity_change_pct'].eq(100).all()or not frame.loc[exits,'quantity_change_pct'].eq(-100).all():
        raise AssertionError('Accepted entry/exit outcome convention changed')
    result={};audit=[]
    for specification in cutoffs:
        cutoff=float(population.quantity_change_pct.quantile(specification['quantile'],interpolation='linear'))
        if not np.isclose(cutoff,specification['cutoff_percent'],rtol=1e-12,atol=1e-10):raise AssertionError('Frozen cutoff mismatch: '+specification['tag'])
        keep=~continuing|frame.quantity_change_pct.le(cutoff)
        if not keep.loc[entries|exits].all():raise AssertionError('Stricter threshold removed entries or exits')
        result[specification['tag']]=keep
        audit.append(dict(**specification,reference_population_rows=len(population),reference_population_row_hash=row_hash(population),
            kept_from_P99_primary=int(keep.sum()),removed_from_P99_primary=int((~keep).sum()),
            entries_kept=int(entries.sum()),exits_kept=int(exits.sum()),ties_kept=True,trim_only_continuing=True))
    return result,audit


def render(root,summary,comparison,native):
    parts=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>H2: асимметрия реакции на новости</title>',
        '<style>body{font:16px system-ui;max-width:1600px;margin:32px auto;padding:0 20px;color:#17212d}.scroll{overflow:auto}table{border-collapse:collapse;font-size:12px}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}pre{background:#f5f6f8;padding:16px;overflow:auto;font:12px monospace}a{color:#075ba5}</style>',
        '<h1>H2: положительный и отрицательный NetSentiment</h1>',
        '<p>Положительная часть = max(Net,0), отрицательная = max(−Net,0). Отрицательная часть имеет неотрицательные значения, поэтому ожидаемый коэффициент сокращения позиции имеет отрицательный знак. Сам raw Net в этих моделях не включён дополнительно.</p>',
        '<p>Для каждого прежнего типа/support показаны CURRENT_FULL на всех допустимых строках, CURRENT_COMMON на строках с точным лагом и CURRENT_PLUS_LAG1 на той же общей выборке. Пороговые выборки P99/P97,5/P95/P90 строятся из одной прежней основы. Отсекаются только продолжающиеся позиции строго выше порога; условные входы +100% и выходы −100% сохранены. Support после отсечения и потерь по лагу не пересчитывается.</p>',
        '<p>H2 проверяет равенство абсолютных коэффициентов, а не простую разность знаковых коэффициентов. Нулевая гипотеза |β+|=|β−| означает β+−β−=0 ИЛИ β++β−=0. Поэтому p_abs=max(p_diff,p_sum): отвергнуть равенство можно только при отклонении обеих составляющих. Все знаки и направление различия величин показаны; само отклонение равенства не означает более сильную отрицательную реакцию.</p>',
        '<p>Fund FE, 16 прежних контролей, кластеры фонд/компания; временных эффектов/кластеров нет. Показаны обычные двусторонние p-values, все заданные модели и все четыре порога. Неопределённая ковариация не исправляется и не подменяется p=1.</p>',
        '<div class="scroll">'+comparison.to_html(index=False,border=0,float_format=lambda x:f'{x:.6g}')+'</div>',
        '<p><a href="coefficients.csv">Все коэффициенты</a> · <a href="h2_tests.json">Контрасты и ковариация</a> · <a href="thresholds.json">Пороги и популяция</a> · <a href="sample_flow.csv">Выборки</a> · <a href="summary.json">Все метрики</a> · <a href="source_manifest.json">Источники</a></p>']
    for name,text in native.items():parts.append('<details><summary>'+html.escape(name)+'</summary><pre>'+html.escape(text)+'</pre></details>')
    parts.append('</html>');(root/'report.html').write_text('\n'.join(parts),encoding='utf-8')


def execute(args):
    if sys.platform!='win32':raise RuntimeError('Actual research execution must run on research-host')
    planpath=Path(args.plan)
    if sha(planpath)!=args.plan_sha:raise ValueError('Frozen plan SHA256 mismatch')
    plan=json.loads(planpath.read_text(encoding='utf-8'))
    if not plan['status'].startswith('FIXED'):raise ValueError('H2 plan is not fixed before new coefficients')
    controls=plan['baseline_controls']+plan['macro_controls']
    if controls!=MICRO+MACRO or plan['variants']!=VARIANTS or plan.get('multiple_testing_adjustments')is not False:
        raise ValueError('Frozen controls, variants or requested inference changed')
    if plan['required_targets_by_variant']!={variant:(PARTS if variant=='CURRENT_PLUS_LAG1'else PARTS[:2])for variant in VARIANTS}:
        raise ValueError('Frozen mandatory sign target names changed')
    if plan.get('time_FE')or plan.get('cluster_time')or plan.get('clusters')!=['fund_id','company_id']:
        raise ValueError('Frozen covariance convention changed')
    root=Path(plan['output_root'])
    if any(p.lower()=='databaseresources'for p in root.parts):raise ValueError('Protected output location')
    root.mkdir(parents=True,exist_ok=True)
    if(root/'completion.json').exists():raise FileExistsError('Completed results may not be overwritten')
    paths={key:plan[key]for key in SOURCE_KEYS};sources={}
    for key,path in paths.items():
        actual=sha(path)
        if actual!=plan['expected_source_sha256'].get(key):raise ValueError('Frozen source mismatch/missing hash: '+key)
        sources[key]=dict(path=path,sha256=actual,size_bytes=Path(path).stat().st_size)
    base=load_module(paths['base_helper'],'h2_source_base');lag=load_module(paths['lag_helper'],'h2_source_lag')
    focus=load_module(paths['focus_helper'],'h2_source_focus')
    expected=[cut['tag']+'__'+name+'__'+variant for cut in plan['cutoffs']for name in plan['base_models']for variant in VARIANTS]
    if plan['base_models']!=focus.planned_models()or plan['planned_models']!=expected:raise ValueError('Frozen 432-label definitions changed')
    previous=json.loads(Path(paths['focus_summary']).read_text(encoding='utf-8'))
    if previous['specification']['expected_source_sha256']['input_panel']!=plan['expected_source_sha256']['input_panel']:
        raise ValueError('Prior support masks used another primary source')
    lock=root/'run.lock';descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(descriptor,json.dumps(dict(pid=os.getpid(),plan_sha256=args.plan_sha)).encode());os.close(descriptor)
    started=time.time()
    def save(name,value):base.save(root/name,value)
    def checkpoint(state,**extra):
        record=dict(state=state,pid=os.getpid(),elapsed_seconds=time.time()-started,**extra)
        save('checkpoint.json',record);print(json.dumps(record),flush=True)
    try:
        save('frozen_plan.json',plan);checkpoint('READING_FROZEN_SOURCES')
        frame=ordered(pd.read_parquet(paths['input_panel']))
        if len(frame)!=plan.get('expected_primary_rows',221899):raise AssertionError('Frozen primary row count changed')
        frame=lag.join_lag(frame,pd.read_parquet(paths['lag_map']));frame=add_signed_parts(frame)
        reference=ordered(pd.read_parquet(paths['cutoff_reference_sample']))
        cut_masks,thresholds=cutoff_masks(frame,reference,plan['cutoffs']);save('thresholds.json',thresholds)
        support=pd.read_csv(paths['focus_support'],dtype={'fund_id':str})
        old_masks=focus.model_samples(frame,support)
        for name,part,_,_,_ in old_masks:
            prior=previous['samples'][name]
            if len(part)!=prior['rows']or row_hash(part)!=prior['row_hash']:raise AssertionError('Old support row mask changed: '+name)
        records=[];coefficients=[];tests={};samples={};catalog=[];native={};representatives={};sample_files={};flows=[];attempts=0
        (root/'samples').mkdir(exist_ok=True)
        for cutoff in plan['cutoffs']:
            tag=cutoff['tag'];keep_ids=set(frame.loc[cut_masks[tag],'row_id'])
            for base_name,original,group,minrows,minintervals in old_masks:
                full=ordered(original.loc[original.row_id.isin(keep_ids)])
                common=ordered(full.loc[full.complete_current_lag1])
                for variant in VARIANTS:
                    name=tag+'__'+base_name+'__'+variant;distributed=variant=='CURRENT_PLUS_LAG1'
                    sample=full if variant=='CURRENT_FULL'else common;key_hash=row_hash(sample)
                    design='CURRENT_LAG_PARTS'if distributed else'CURRENT_PARTS';key=(design,key_hash)
                    if key_hash not in sample_files:
                        path=root/'samples'/(name+'.parquet');sample.to_parquet(path,index=False,compression='zstd')
                        sample_files[key_hash]=dict(parquet=str(path),rows=len(sample),row_hash=key_hash,exported_columns=list(sample.columns))
                    samples[name]=copy.deepcopy(sample_files[key_hash]);alias=representatives.get(key,{}).get('model')
                    achieved=focus.support_table(sample)
                    details=dict(model=name,base_model=base_name,cutoff_tag=tag,cutoff_percent=cutoff['cutoff_percent'],variant=variant,
                        fund_focus=group,old_rows=len(original),trimmed_full_rows=len(full),common_rows=len(common),
                        n_cutoff_removed=len(original)-len(full),n_lag_missing=len(full)-len(common),
                        n_lag_dropped_from_this_model=0 if variant=='CURRENT_FULL'else len(full)-len(common),
                        fixed_min_fund_rows=minrows,fixed_min_fund_intervals=minintervals,
                        achieved_min_fund_rows=int(achieved.fund_rows.min())if len(achieved)else None,
                        achieved_min_fund_intervals=int(achieved.fund_intervals.min())if len(achieved)else None,
                        first_start_date=str(sample.start_date.min())if len(sample)else None,last_end_date=str(sample.end_date.max())if len(sample)else None,
                        row_hash=key_hash,design=design,exact_sample_alias_of=alias)
                    checkpoint('PROCESSING_MODEL',model=name,rows=len(sample),alias_of=alias)
                    if alias:
                        stored=representatives[key];record=copy.deepcopy(stored['record']);coefs=copy.deepcopy(stored['coefs'])
                        text=stored['text'];htm=stored['html'];test=copy.deepcopy(stored['test']);origin='CURRENT_EXACT_SAMPLE_DESIGN_ALIAS'
                    else:
                        issue=('Fewer than minimum rows'if len(sample)<plan.get('minimum_model_rows',30)else
                            'Fewer than two fund/company clusters'if sample.fund_id.nunique()<2 or sample.company_id.nunique()<2 else None)
                        if issue:
                            record=dict(status='NOT_ESTIMABLE',reason=issue,nobs=len(sample),funds=int(sample.fund_id.nunique()),companies=int(sample.company_id.nunique()))
                            coefs=[];text='NOT_ESTIMABLE: '+issue;htm='<pre>'+html.escape(text)+'</pre>';test=None;origin='NOT_ESTIMABLE'
                        else:
                            attempts+=1
                            try:
                                fitted,result,terms,dropped=fit_native(sample,controls,distributed)
                                record,raw=base.describe_result(fitted,result,name,'quantity_change_pct',terms,dropped)
                                record['covariance_options']['cluster_time']=False
                                coefs=[{field:r.get(field)for field in COEF_FIELDS}for r in raw];test=h2_tests(result,distributed)
                                overall=result.f_statistic_robust
                                record['overall_robust_F_inference']=dict(statistic=float(overall.stat),native_p=float(overall.pval),
                                    status='VALID'if np.isfinite(overall.stat)and overall.stat>=0 and np.isfinite(overall.pval)else'UNDEFINED_COVARIANCE')
                                text=str(result.summary);htm=result.summary.as_html();origin='NEW_UNIQUE_SAMPLE_DESIGN_FIT'
                            except Exception as error:
                                record=dict(status='NOT_ESTIMABLE',reason=f'{type(error).__name__}: {error}',nobs=len(sample),
                                    funds=int(sample.fund_id.nunique()),companies=int(sample.company_id.nunique()))
                                coefs=[];text=record['reason'];htm='<pre>'+html.escape(text)+'</pre>';test=None;origin='NOT_ESTIMABLE'
                        representatives[key]=dict(model=name,record=copy.deepcopy(record),coefs=copy.deepcopy(coefs),text=text,html=htm,test=copy.deepcopy(test))
                    record.update(details,result_origin=origin,support_qualification_fixed_before_cutoff_and_lag_drop=True)
                    for row in coefs:row['model']=name
                    tests[name]=test if test is not None else dict(status='NOT_ESTIMABLE',reason=record.get('reason'))
                    records.append(record);coefficients.extend(coefs);native[name]=text;catalog.append(details|dict(status=record['status'],result_origin=origin))
                    flows.append(details|dict(nobs=len(sample),funds=int(sample.fund_id.nunique()),companies=int(sample.company_id.nunique())))
                    (root/(name+'.summary.txt')).write_text(text,encoding='utf-8')
                    (root/(name+'.summary.html')).write_text(htm,encoding='utf-8')
        pd.DataFrame(coefficients).to_csv(root/'coefficients.csv',index=False,encoding='utf-8-sig')
        save('h2_tests.json',tests);save('mask_catalog.json',catalog)
        pd.DataFrame(flows).to_csv(root/'sample_flow.csv',index=False,encoding='utf-8-sig')
        comparison=[];longtests=[]
        for record in records:
            item={k:record.get(k)for k in ['model','base_model','fund_focus','cutoff_tag','cutoff_percent','variant','status','nobs','funds','companies',
                'rsquared_within','first_start_date','last_end_date','n_cutoff_removed','n_lag_missing','fixed_min_fund_rows','fixed_min_fund_intervals',
                'achieved_min_fund_rows','achieved_min_fund_intervals','result_origin','exact_sample_alias_of']}
            modeltests=tests[record['model']].get('tests',{})
            for scope in ['current','lag1','total_current_plus_lag1']:
                test=modeltests.get(scope,{})
                for field in ['beta_positive','beta_negative','p_abs','negative_absolute_minus_positive_absolute','expected_directional_signs','status']:
                    item[scope+'_'+field]=test.get(field)
                if test:
                    longtests.append(dict(model=record['model'],scope=scope,**{k:test.get(k)for k in ['status','beta_positive','beta_negative','p_abs',
                        'expected_directional_signs','negative_absolute_minus_positive_absolute','negative_absolute_stronger']},
                        p_difference=test['difference_zero']['p_native'],p_sum=test['sum_zero']['p_native'],
                        difference_beta=test['difference_zero']['beta'],sum_beta=test['sum_zero']['beta']))
            comparison.append(item)
        comparison=pd.DataFrame(comparison);comparison.to_csv(root/'comparison.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(longtests).to_csv(root/'h2_tests.csv',index=False,encoding='utf-8-sig')
        unchanged={key:sha(path)==sources[key]['sha256']for key,path in paths.items()}
        if not all(unchanged.values()):raise AssertionError('A frozen source changed during execution')
        save('source_manifest.json',dict(sources=sources,source_hashes_unchanged_after_run=unchanged,plan_sha256=args.plan_sha,runner_sha256=sha(__file__)))
        summary=dict(status='COMPLETE',specification=plan,plan_sha256=args.plan_sha,models=records,samples=samples,h2_tests=tests,
            thresholds=thresholds,mask_catalog=catalog,unique_sample_design_fit_attempts=attempts,unique_sample_designs=len(representatives),
            unique_sample_files=len(sample_files),elapsed_seconds=time.time()-started,independent_QA='PENDING_SEPARATE_REVIEW',
            inference='Native two-sided scalar contrasts and unconditional absolute-equality intersection-union p_abs; no sign relabeling',
            no_cutoff_or_model_selected_by_significance=True)
        save('summary.json',summary);render(root,summary,comparison,native)
        checkpoint('COMPLETE',labels=len(records),unique_sample_design_fit_attempts=attempts)
        save('completion.json',dict(status='COMPLETE',plan_sha256=args.plan_sha,elapsed_seconds=time.time()-started,
            finished_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        return 0
    except Exception as error:
        save('failure.json',dict(status='FAILED',message=str(error),traceback=traceback.format_exc(),plan_sha256=args.plan_sha));raise
    finally:lock.unlink(missing_ok=True)


def fixture():
    base=load_module(Path(__file__).with_name('run_h1_net_fund_fe_20261030.py'),'h2_fixture_previous_base')
    variance=np.diag([.01,.01]);same=absolute_asymmetry([1.,1.],variance,100)
    mirrored=absolute_asymmetry([1.,-1.],variance,100)
    stronger_negative=absolute_asymmetry([1.,-2.],variance,100)
    stronger_positive=absolute_asymmetry([2.,-1.],variance,100)
    assert same['p_abs']==1 and same['sum_zero']['p_native']<.05
    assert mirrored['p_abs']==1 and mirrored['difference_zero']['p_native']<.05
    assert stronger_negative['p_abs']<.05 and stronger_negative['negative_absolute_stronger']
    assert stronger_positive['p_abs']<.05 and not stronger_positive['negative_absolute_stronger']
    undefined=absolute_asymmetry([1.,-2.],[[1.,2.],[2.,1.]],100)
    assert undefined['p_abs']is None and undefined['status']=='UNDEFINED_COVARIANCE'
    rng=np.random.default_rng(320261103);rows=[]
    for fund in range(18):
        for period in range(10):
            for company in range(6):
                net,lag=rng.uniform(-1,1,size=2);control,noise=rng.normal(size=2)
                y=1.2*max(net,0)-2.*max(-net,0)+.4*max(lag,0)-.6*max(-lag,0)+.3*control+.2*fund+noise
                rows.append(dict(row_id=f'r{fund}-{period}-{company}',fund_id=str(fund),company_id=str(company),security_id=str(company),
                    interval_id=f'i{fund}-{period}',end_date=pd.Timestamp('2020-01-31')+pd.offsets.MonthEnd(period),
                    net_sentiment=net,net_sentiment_lag1=lag,control=control,absorbed=float(fund),quantity_change_pct=y))
    source=pd.DataFrame(rows);frame=add_signed_parts(source)
    sample,result,terms,dropped=fit_native(frame,['control','absorbed'],True)
    within=sample[['quantity_change_pct']+terms]-sample[['quantity_change_pct']+terms].groupby(sample.fund_id).transform('mean')
    beta=np.linalg.lstsq(within[terms],within.quantity_change_pct,rcond=None)[0]
    covariance=base.cluster_covariance_fixture(sample,within[terms].to_numpy(),result.resids.to_numpy(),result.df_resid)
    assert np.allclose(result.params[terms],beta,rtol=1e-10,atol=1e-10)
    assert np.allclose(result.cov.loc[terms,terms],covariance,rtol=1e-10,atol=1e-10)
    assert CURRENT not in terms and LAG not in terms and set(PARTS).issubset(terms)
    tests=h2_tests(result,True)
    total=tests['tests']['total_current_plus_lag1']
    full=np.asarray(tests['full_target_covariance']);matrix=np.asarray([[1,0,1,0],[0,1,0,1]])
    assert np.allclose(total['target_covariance'],matrix@full@matrix.T)
    one_side=frame.copy();one_side['net_negative']=0.
    try:fit_native(one_side,['control','absorbed'],False)
    except ValueError as error:assert'Required H2 sign target'in str(error)
    else:raise AssertionError('Unidentified sign target was silently removed')
    missing=source.head(2).copy();missing.loc[0,CURRENT]=np.nan
    assert add_signed_parts(missing).loc[0,PARTS[:2]].isna().all()
    y=np.asarray([-50.,0.,10.,10.,10.,20.,100.,1000.])
    reference=pd.DataFrame(dict(row_id=[str(i)for i in range(len(y))],quantity_open=np.ones(len(y)),quantity_close=1+y/100,quantity_change_pct=y))
    primary=reference.iloc[:-1].copy()
    primary=pd.concat([primary,pd.DataFrame([dict(row_id='entry',quantity_open=0.,quantity_close=1.,quantity_change_pct=100.),
        dict(row_id='exit',quantity_open=1.,quantity_close=0.,quantity_change_pct=-100.)])],ignore_index=True)
    cutoffs=[dict(tag='P99',quantile=.99,cutoff_percent=float(np.quantile(y,.99))),dict(tag='P50',quantile=.5,cutoff_percent=10.)]
    masks,thresholds=cutoff_masks(primary,reference,cutoffs,expected_population_rows=8)
    assert masks['P99'].all()and masks['P50'].iloc[-2:].all()
    assert masks['P50'].loc[primary.quantity_change_pct.eq(10)].all()
    assert not masks['P50'].loc[primary.quantity_open.gt(0)&primary.quantity_change_pct.gt(10)].any()
    print(json.dumps(dict(status='PASS',scope='synthetic_only',rows=len(sample),signed_parts=True,
        union_null_IUT=True,signs_and_magnitude_direction=True,undefined_covariance_preserved=True,
        native_within_and_covariance=True,total_full_crosscovariance=True,mandatory_sign_targets=True,
        fixed_cutoff_entry_exit_tie_preservation=True)))
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan');parser.add_argument('--plan-sha')
    parser.add_argument('--fixture-only',action='store_true');args=parser.parse_args()
    if args.fixture_only:return fixture()
    if not args.plan or not args.plan_sha:parser.error('--plan and --plan-sha are required')
    return execute(args)


if __name__=='__main__':raise SystemExit(main())
