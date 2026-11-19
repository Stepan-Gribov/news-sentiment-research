"""Nine fixed post-result shared-sign diagnostics on existing DAR samples.

The original models are cached, never refitted. Only trade_sign is added.
This diagnostic is not a new primary model, fund alpha or causal correction.
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
SIGN='trade_sign'
REQUIRED=[CURRENT,LAG,SIGN]
SOURCE_KEYS=['baseline_summary','baseline_coefficients','h3_helper','h2_helper','base_helper']
COEF_FIELDS=['model','outcome','term','beta','se_native','t_native','p_native','ci95_native_low','ci95_native_high']


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


def sign_design_status(sample,h2):
    sign=pd.to_numeric(sample[SIGN],errors='coerce')
    if not np.isfinite(sign).all()or not sign.isin([-1.,1.]).all()or not sign.pow(2).eq(1).all():
        raise AssertionError('Diagnostic requires nonzero direction s in {-1,+1}')
    if 'delta_quantity_units'in sample:
        delta=pd.to_numeric(sample.delta_quantity_units,errors='coerce')
        if not np.isfinite(delta).all()or delta.eq(0).any()or not np.array_equal(np.sign(delta),sign):
            raise AssertionError('Trade sign is inconsistent with nonzero known quantity change')
    if sample.groupby('fund_id',observed=True)[SIGN].nunique().max()<=1:
        return'already_absorbed_equivalent'
    try:h2.select_regressors(sample,REQUIRED,[])
    except ValueError as error:
        if SIGN in str(error):return'not_estimable_sign_collinear_with_targets'
        raise
    return'estimable'


def fit_native(sample,controls,h2):
    from linearmodels.panel import PanelOLS
    sample=ordered(sample);state=sign_design_status(sample,h2)
    if state!='estimable':raise ValueError(state)
    terms,dropped=h2.select_regressors(sample,REQUIRED,controls)
    if not set(REQUIRED).issubset(terms):raise AssertionError('A required diagnostic regressor was silently removed')
    index=pd.MultiIndex.from_arrays([sample.fund_id.astype(str),pd.to_datetime(sample.end_date)],names=['fund_id','report_date'])
    x=sample[terms].astype(float).copy();x.index=index;x.insert(0,'const',1.)
    y=pd.Series(sample[OUTCOME].to_numpy(float),index=index,name=OUTCOME)
    clusters=pd.DataFrame({'fund_cluster':pd.Categorical(sample.fund_id.astype(str)).codes,
        'company_cluster':pd.Categorical(sample.company_id.astype(str)).codes},index=index)
    result=PanelOLS(y,x,entity_effects=True,time_effects=False,other_effects=None,singletons=True,
        drop_absorbed=False,check_rank=True).fit(cov_type='clustered',clusters=clusters,cluster_time=False,
        cluster_entity=False,group_debias=True,debiased=True,auto_df=False,count_effects=True)
    if result.nobs!=len(sample)or not result.resids.index.equals(index):raise AssertionError('Diagnostic estimator changed rows/order')
    return sample,result,terms,dropped


def render(root,summary,comparison,native):
    parts=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Диагностика общего знака сделки</title>',
        '<style>body{font:16px system-ui;max-width:1600px;margin:32px auto;padding:0 20px;color:#17212d}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}pre{padding:16px;background:#f5f6f8;overflow:auto;font:12px monospace}</style>',
        '<h1>POST-RESULT: диагностика общего знака сделки</h1>',
        '<p>В DAR знак изменения позиции s входит и в Y=100×s×excess price return, и в X=s×Net. При отрицательной средней избыточной ценовой доходности и положительном среднем Net может возникать механическая отрицательная связь. Эта ограниченная проверка добавляет ровно один контроль s.</p>',
        '<p>Ровно девять прежних Q99_D99/core60/24 моделей на горизонтах 1/2/3 месяца. Строки, Y, все прежние X и 16 контролей, пороги, fund FE и кластеры фонд/компания сохранены. Старые оценки прочитаны из кэша без повторного расчёта. Изменение β описательное: отдельный межмодельный тест не проводился.</p>',
        '<p>Если s поглощён fund FE, фиксируется already_absorbed_equivalent с прежними коэффициентами; коэффициент s не подменяется нулём. Если s коллинеарен с целевыми переменными, модель не оценивается. Показаны все девять результатов независимо от знака и p.</p>',
        '<div class="scroll">'+comparison.to_html(index=False,border=0,float_format=lambda x:f'{x:.6g}')+'</div>',
        '<p>Это диагностика после основного результата, не новая primary и не fund alpha. Она не исправляет меняющийся во времени carry, купоны, корпоративные действия, риск или ограничения исходного proxy. Положительный коэффициент не является целью отбора.</p>',
        '<p><a href="coefficients.csv">Новые коэффициенты</a> · <a href="baseline_cache.json">Прежние оценки</a> · <a href="distributed_tests.json">Совместный тест и сумма</a> · <a href="summary.json">Все метрики</a> · <a href="source_manifest.json">Неизменные источники</a></p>']
    for name,text in native.items():parts.append('<h2>'+html.escape(name)+'</h2><pre>'+html.escape(text)+'</pre>')
    parts.append('</html>');(root/'report.html').write_text('\n'.join(parts),encoding='utf-8')


def execute(args):
    if sys.platform!='win32':raise RuntimeError('Actual research execution must run on research-host')
    planpath=Path(args.plan)
    if sha(planpath)!=args.plan_sha:raise ValueError('Frozen diagnostic plan SHA mismatch')
    plan=json.loads(planpath.read_text(encoding='utf-8'))
    if not plan['status'].startswith('FIXED'):raise ValueError('Diagnostic plan is not fixed before diagnostic coefficients')
    if plan['required_design']!=REQUIRED or plan['dependent']!=OUTCOME:raise ValueError('Fixed diagnostic design changed')
    if plan.get('multiple_testing_adjustments')is not False or plan.get('time_FE')or plan.get('cluster_time')or plan.get('clusters')!=['fund_id','company_id']:
        raise ValueError('Fixed covariance/inference changed')
    if len(plan['baseline_models'])!=9 or plan['planned_models']!=[name+'__WITH_TRADE_SIGN'for name in plan['baseline_models']]:
        raise ValueError('Exactly nine preselected diagnostics are required')
    controls=plan['baseline_controls']+plan['macro_controls']
    root=Path(plan['output_root'])
    if any(p.lower()=='databaseresources'for p in root.parts):raise ValueError('Protected output location')
    root.mkdir(parents=True,exist_ok=True)
    if(root/'completion.json').exists():raise FileExistsError('Completed diagnostic may not be overwritten')
    paths={key:plan[key]for key in SOURCE_KEYS};sources={}
    for key,path in paths.items():
        actual=sha(path)
        if actual!=plan['expected_source_sha256'].get(key):raise ValueError('Frozen source mismatch: '+key)
        sources[key]=dict(path=path,sha256=actual,size_bytes=Path(path).stat().st_size)
    baseline=json.loads(Path(paths['baseline_summary']).read_text(encoding='utf-8'))
    baseline_coefs=pd.read_csv(paths['baseline_coefficients'])
    base=load_module(paths['base_helper'],'sign_diag_base');h2=load_module(paths['h2_helper'],'sign_diag_h2')
    h3=load_module(paths['h3_helper'],'sign_diag_h3')
    if controls!=h3.MICRO+h3.MACRO:raise ValueError('Sixteen prior controls changed')
    baseline_records={m['model']:m for m in baseline['models']};sample_sources={}
    for oldname in plan['baseline_models']:
        frozen=plan['samples_frozen'][oldname];metadata=baseline['samples'][oldname]
        if frozen['rows']!=metadata['rows']or frozen['row_hash']!=metadata['row_hash']or frozen['horizon']!=metadata['horizon']:
            raise AssertionError('Frozen sample reference differs from original summary')
        if sha(frozen['parquet'])!=frozen['sha256']:raise ValueError('Frozen exact sample checksum mismatch')
        record=baseline_records[oldname]
        if record['status']!='ESTIMATED'or record['dependent']!=OUTCOME or record['source_row_hash']!=frozen['row_hash']:
            raise ValueError('Original cached model is not the declared estimated baseline')
        options=record.get('covariance_options',{})
        if any(options.get(k)!=v for k,v in dict(cov_type='clustered',clusters=['fund_id','company_id'],cluster_time=False,
            debiased=True,auto_df=False,count_effects=True,group_debias=True).items()):raise ValueError('Original covariance convention differs')
        sample_sources[oldname]=copy.deepcopy(frozen)
    lock=root/'run.lock';descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(descriptor,json.dumps(dict(pid=os.getpid(),plan_sha256=args.plan_sha)).encode());os.close(descriptor)
    started=time.time()
    def save(name,value):base.save(root/name,value)
    def checkpoint(state,**extra):
        record=dict(state=state,pid=os.getpid(),elapsed_seconds=time.time()-started,**extra)
        save('checkpoint.json',record);print(json.dumps(record),flush=True)
    try:
        save('frozen_plan.json',plan)
        records=[];coefficients=[];samples={};tests={};cache={};native={};comparison=[];fit_count=0
        for oldname,name in zip(plan['baseline_models'],plan['planned_models']):
            info=sample_sources[oldname];sample=ordered(pd.read_parquet(info['parquet']))
            if len(sample)!=info['rows']or row_hash(sample)!=info['row_hash']:raise AssertionError('Frozen diagnostic sample row IDs changed')
            if sample.horizon.nunique()!=1 or sample.horizon.iloc[0]!=info['horizon']:raise AssertionError('Frozen diagnostic horizon mismatch')
            old_record=copy.deepcopy(baseline_records[oldname])
            if old_record['nobs']!=len(sample)or old_record['funds']!=sample.fund_id.nunique()or old_record['companies']!=sample.company_id.nunique():
                raise AssertionError('Original cached model counts differ from the frozen sample')
            oldcoef=baseline_coefs.loc[baseline_coefs.model.eq(oldname)]
            if oldcoef.term.duplicated().any()or SIGN in set(oldcoef.term)or not{CURRENT,LAG}.issubset(set(oldcoef.term)):
                raise ValueError('Original cached coefficient design is inconsistent')
            cache[oldname]=dict(record=old_record,coefficients=oldcoef.to_dict('records'),
                distributed_tests=baseline['distributed_tests'].get(oldname),baseline_refit=False)
            before_values=sample[[OUTCOME,CURRENT,LAG,SIGN]+controls].copy()
            diagnostic_status=sign_design_status(sample,h2)
            checkpoint('PROCESSING_DIAGNOSTIC',model=name,rows=len(sample),diagnostic_status=diagnostic_status)
            if diagnostic_status=='already_absorbed_equivalent':
                record=copy.deepcopy(old_record);record['status']='ALREADY_ABSORBED_EQUIVALENT'
                coefs=[{k:r.get(k)for k in COEF_FIELDS}for r in oldcoef.to_dict('records')]
                coefs.append(dict(model=name,outcome=OUTCOME,term=SIGN,beta=None,se_native=None,t_native=None,p_native=None,
                    ci95_native_low=None,ci95_native_high=None))
                test=copy.deepcopy(baseline['distributed_tests'].get(oldname));text='already_absorbed_equivalent: trade_sign is constant within every fund. Cached baseline coefficients retained without refit.'
                htm='<pre>'+html.escape(text)+'</pre>'
            elif diagnostic_status!='estimable':
                record=dict(status='NOT_ESTIMABLE',reason=diagnostic_status,nobs=len(sample),funds=int(sample.fund_id.nunique()),companies=int(sample.company_id.nunique()))
                coefs=[];test=None;text=diagnostic_status;htm='<pre>'+html.escape(text)+'</pre>'
            else:
                fit_count+=1
                try:
                    fitted,result,terms,dropped=fit_native(sample,controls,h2)
                    record,raw=base.describe_result(fitted,result,name,OUTCOME,terms,dropped)
                    record['covariance_options']['cluster_time']=False
                    coefs=[{k:r.get(k)for k in COEF_FIELDS}for r in raw]
                    test=h3.distributed_tests(result,h2)
                    test['all_required_target_terms']=REQUIRED
                    test['all_required_target_covariance']=result.cov.loc[REQUIRED,REQUIRED].to_numpy(float).tolist()
                    overall=result.f_statistic_robust
                    record['overall_robust_F_inference']=dict(statistic=float(overall.stat),native_p=float(overall.pval),
                        status='VALID'if np.isfinite(overall.stat)and overall.stat>=0 and np.isfinite(overall.pval)else'UNDEFINED_COVARIANCE')
                    text=str(result.summary);htm=result.summary.as_html()
                except Exception as error:
                    record=dict(status='NOT_ESTIMABLE',reason=f'{type(error).__name__}: {error}',nobs=len(sample),funds=int(sample.fund_id.nunique()),companies=int(sample.company_id.nunique()))
                    coefs=[];test=None;text=record['reason'];htm='<pre>'+html.escape(text)+'</pre>'
            if not sample[[OUTCOME,CURRENT,LAG,SIGN]+controls].equals(before_values):raise AssertionError('Diagnostic changed original Y or X')
            signrow=next((r for r in coefs if r['term']==SIGN),{})
            sign_inference=('ABSORBED_FUND_FE'if diagnostic_status=='already_absorbed_equivalent'else
                'FINITE'if signrow and all(signrow.get(k)is not None and np.isfinite(signrow[k])for k in ['beta','se_native','p_native','ci95_native_low','ci95_native_high'])and signrow['se_native']>0 else
                'UNDEFINED'if signrow else'NOT_ESTIMATED')
            record.update(model=name,baseline_model=oldname,diagnostic_status=diagnostic_status,horizon=info['horizon'],
                fund_focus=old_record.get('fund_focus'),analysis_role='POST_RESULT_SHARED_SIGN_DIAGNOSTIC_NOT_PRIMARY_NOT_ALPHA',
                same_ordered_rows_as_baseline=True,baseline_refit=False,baseline_sample_sha256=info['sha256'],trade_sign_inference_status=sign_inference)
            for row in coefs:row['model']=name
            tests[name]=test if test is not None else dict(status='NOT_ESTIMABLE',reason=record.get('reason'))
            records.append(record);coefficients.extend(coefs);samples[name]=copy.deepcopy(info);native[name]=text
            (root/(name+'.summary.txt')).write_text(text,encoding='utf-8');(root/(name+'.summary.html')).write_text(htm,encoding='utf-8')
            item={k:record.get(k)for k in ['model','baseline_model','horizon','fund_focus','status','diagnostic_status','nobs','funds','companies','rsquared_within','trade_sign_inference_status']}
            item['baseline_rsquared_within']=old_record.get('rsquared_within')
            for term,prefix in [(CURRENT,'current'),(LAG,'lag1'),(SIGN,'trade_sign')]:
                controlled=next((r for r in coefs if r['term']==term),{})
                original=oldcoef.loc[oldcoef.term.eq(term)].iloc[0].to_dict()if term in set(oldcoef.term)else{}
                for field in ['beta','se_native','ci95_native_low','ci95_native_high','p_native']:
                    item['controlled_'+prefix+'_'+field]=controlled.get(field)
                    if term!=SIGN:item['baseline_'+prefix+'_'+field]=original.get(field)
                if term!=SIGN:
                    a=controlled.get('beta');b=original.get('beta')
                    item['descriptive_delta_'+prefix+'_beta']=float(a-b)if a is not None and b is not None else None
            item['mean_cash_excess_price_return']=float((sample.future_price_return-sample.cash_return).mean())
            item['mean_current_net']=float(sample.net_sentiment.mean());item['mean_lag_net']=float(sample.net_sentiment_lag1.mean())
            item['positive_trade_rows']=int(sample[SIGN].eq(1).sum());item['negative_trade_rows']=int(sample[SIGN].eq(-1).sum())
            joint=tests[name].get('joint_current_lag_zero',{});total=tests[name].get('sum_current_lag',{})
            item.update(joint_status=joint.get('status'),joint_p=joint.get('p_value'),sum_status=total.get('status'),sum_beta=total.get('beta'),
                sum_se=total.get('se_native'),sum_ci95_low=total.get('ci95_native_low'),sum_ci95_high=total.get('ci95_native_high'),sum_p=total.get('p_native'))
            comparison.append(item)
        table=pd.DataFrame(comparison);table.to_csv(root/'comparison.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(coefficients).to_csv(root/'coefficients.csv',index=False,encoding='utf-8-sig')
        save('baseline_cache.json',cache);save('distributed_tests.json',tests)
        unchanged={key:sha(path)==sources[key]['sha256']for key,path in paths.items()}
        samples_unchanged={name:sha(info['parquet'])==info['sha256']for name,info in sample_sources.items()}
        if not all(unchanged.values())or not all(samples_unchanged.values()):raise AssertionError('A frozen source/sample changed')
        save('source_manifest.json',dict(sources=sources,samples_frozen=sample_sources,source_hashes_unchanged_after_run=unchanged,
            sample_hashes_unchanged_after_run=samples_unchanged,plan_sha256=args.plan_sha,runner_sha256=sha(__file__)))
        summary=dict(status='COMPLETE',specification=plan,plan_sha256=args.plan_sha,models=records,samples=samples,
            distributed_tests=tests,baseline_cache=cache,actual_new_fit_attempts=fit_count,elapsed_seconds=time.time()-started,
            baseline_refit=False,analysis_role='POST_RESULT_SHARED_SIGN_DIAGNOSTIC_NOT_PRIMARY_NOT_ALPHA',
            crossmodel_coefficient_change_inference='Descriptive only; no independent-crossmodel significance test',
            independent_QA='PENDING_SEPARATE_REVIEW')
        save('summary.json',summary);render(root,summary,table,native)
        checkpoint('COMPLETE',models=len(records),actual_new_fit_attempts=fit_count)
        save('completion.json',dict(status='COMPLETE',plan_sha256=args.plan_sha,elapsed_seconds=time.time()-started,
            finished_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        return 0
    except Exception as error:
        save('failure.json',dict(status='FAILED',message=str(error),traceback=traceback.format_exc(),plan_sha256=args.plan_sha));raise
    finally:lock.unlink(missing_ok=True)


def fixture():
    h2=load_module(Path(__file__).with_name('run_h2_sentiment_lag_20261103.py'),'sign_fixture_h2')
    base=load_module(Path(__file__).with_name('run_h1_net_fund_fe_20261030.py'),'sign_fixture_base')
    rng=np.random.default_rng(520261103);rows=[]
    for fund in range(18):
        for period in range(10):
            for company in range(6):
                sign=float(rng.choice([-1,1]));net,lag=rng.uniform(.1,.7,size=2);control,noise=rng.normal(size=2)
                rows.append(dict(row_id=f'r{fund}-{period}-{company}',fund_id=str(fund),company_id=str(company),security_id=str(company),
                    interval_id=f'i{fund}-{period}',end_date=pd.Timestamp('2020-01-31')+pd.offsets.MonthEnd(period),
                    trade_sign=sign,delta_quantity_units=sign,align_current=sign*net,align_lag1=sign*lag,control=control,
                    absorbed=float(fund),dar_q_cash_price_pp=-2*sign+.1*noise,constant_excess_y=-2*sign))
    frame=ordered(pd.DataFrame(rows))
    values=frame[['constant_excess_y',CURRENT,LAG,SIGN]]
    within=values-values.groupby(frame.fund_id).transform('mean')
    old=np.linalg.lstsq(within[[CURRENT,LAG]],within.constant_excess_y,rcond=None)[0]
    corrected=np.linalg.lstsq(within[REQUIRED],within.constant_excess_y,rcond=None)[0]
    assert(old<0).all()and np.max(np.abs(corrected[:2]))<1e-12 and np.isclose(corrected[2],-2)
    sample,result,terms,dropped=fit_native(frame,['control','absorbed'],h2)
    w=sample[[OUTCOME]+terms]-sample[[OUTCOME]+terms].groupby(sample.fund_id).transform('mean')
    beta=np.linalg.lstsq(w[terms],w[OUTCOME],rcond=None)[0]
    covariance=base.cluster_covariance_fixture(sample,w[terms].to_numpy(),result.resids.to_numpy(),result.df_resid)
    assert np.allclose(result.params[terms],beta,rtol=1e-10,atol=1e-10)
    assert np.allclose(result.cov.loc[terms,terms],covariance,rtol=1e-10,atol=1e-10)
    assert set(REQUIRED).issubset(terms)
    absorbed=frame.copy();absorbed[SIGN]=np.where(absorbed.fund_id.astype(int)%2,1.,-1.);absorbed.delta_quantity_units=absorbed[SIGN]
    assert sign_design_status(absorbed,h2)=='already_absorbed_equivalent'
    collinear=frame.copy();collinear[CURRENT]=collinear[SIGN]
    assert sign_design_status(collinear,h2)=='not_estimable_sign_collinear_with_targets'
    zero=frame.copy();zero.loc[0,SIGN]=0.;zero.loc[0,'delta_quantity_units']=0.
    try:sign_design_status(zero,h2)
    except AssertionError:pass
    else:raise AssertionError('Zero trade was accepted')
    print(json.dumps(dict(status='PASS',scope='synthetic_only',rows=len(sample),constant_negative_excess_positive_mean_net=True,
        shared_sign_mean_component_removed=True,native_within_and_covariance=True,
        absorbed_equivalent_and_nullspace_guards=True,no_zero_trade=True)))
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan');parser.add_argument('--plan-sha')
    parser.add_argument('--fixture-only',action='store_true');args=parser.parse_args()
    if args.fixture_only:return fixture()
    if not args.plan or not args.plan_sha:parser.error('--plan and --plan-sha are required')
    return execute(args)


if __name__=='__main__':raise SystemExit(main())
