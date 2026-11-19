"""Frozen24-cell H3 fund-minus-index run, executed from an immutable job."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import zipfile
import numpy as np
import pandas as pd
from h3_alpha_io_20261104 import sha,file_record,save_json,seal_directory,utc
from h3_index_excess_core_20261104 import index_excess_outcomes
from fit_h3_index_excess_fe_20261104 import fit_fund_fe

ROOT=Path('./analysis_outputs/h3_index_excess_20261104')
SPEC_SHA='b0d982e5bf57866a2ea6d0fca5e57186a54076d4359f5ebe42ec752224dbd74b'
OLD=ROOT.parent/'h3_relaxed_news_20261104/outputs/63d2bda2bad14bec'


def checkpoint(stage,**kw):
    r=dict(stage=stage,pid=os.getpid(),updated_utc=utc(),**kw)
    save_json(ROOT/'runtime/checkpoint.json',r);print(json.dumps(r,default=str),flush=True)


def normalize_benchmarks(frame,columns):
    d=frame.copy()
    for source,target in columns.items():
        if source not in d:raise ValueError('Missing declared benchmark source column '+source)
        if target in d:
            if not d[source].equals(d[target]):raise ValueError('Conflicting benchmark aliases '+source+' / '+target)
        else:d[target]=d[source]
    required={'benchmark_id','month','benchmark_return','benchmark_verified','start_valuation_date','end_valuation_date'}
    if required-set(d):raise ValueError('Benchmark adapter missing '+str(required-set(d)))
    d['month']=pd.PeriodIndex(d.month,freq='M').astype(str)
    d['benchmark_id']=d.benchmark_id.astype(str)
    if set(d.benchmark_id)!={'IMOEX','RGBITR'}:raise ValueError('Exact approved benchmark IDs required')
    if d.duplicated(['benchmark_id','month']).any():raise ValueError('Duplicate benchmark month')
    for col in ['start_valuation_date','end_valuation_date']:d[col]=pd.to_datetime(d[col])
    if 'return_period_end' not in d:d['return_period_end']=pd.PeriodIndex(d.month,freq='M').to_timestamp(how='end').normalize()
    else:d['return_period_end']=pd.to_datetime(d.return_period_end)
    return d


def render_tables(results):
    lookup={(r['group'],r['horizon'],r['variant']):r for r in results}
    labels={'pooled':'Все фонды','equity':'Акции','bond':'Облигации','mixed':'Смешанные'}
    out=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>H3: доходность минус индекс</title>',
      '<style>body{font:18px/1.5 system-ui;max-width:1450px;margin:32px auto;padding:0 24px;color:#20242a}table{border-collapse:collapse;width:100%;font-size:16px}th,td{border:1px solid #ccd4dc;padding:10px;vertical-align:top;text-align:left}th{background:#edf2f7}h2{margin-top:38px}.wrap{overflow:auto}small{color:#56606b}</style>',
      '<h1>H3: будущая доходность фонда минус индекс</h1>',
      '<p>Основной горизонт — следующий полный календарный месяц (h1). Акции и смешанные фонды сравниваются с IMOEX, облигационные — с RGBITR. Общая выборка использует индекс типа фонда на отчётную дату. Это простая индексная разница доходностей, не факторная alpha и не доказательство навыка.</p>',
      '<p>24 заранее зафиксированных варианта. Только fund FE и последний доступный прошлый месячный результат фонда минус тот же индекс; без требования36/60месяцев, временных эффектов и новых отсечек. Все p двусторонние t(G−1), fund CR1, без поправок. Текущая и лаговая модели могут использовать разные выборки.</p>']
    for h in [1,2,3]:
        out.append('<h2>'+('Основной горизонт' if h==1 else 'Дополнительный горизонт')+': h='+str(h)+'</h2>')
        for variant,label in [('CURRENT_ONLY_OWN_AVAILABLE','Только текущий сигнал'),('CURRENT_PLUS_LAGNEWS','Текущий + предыдущие новости')]:
            rows={}
            for group in ['pooled','equity','bond','mixed']:
                r=lookup[(group,h,variant)];coef={c['term']:c for c in r.get('coefficients',[])}
                cell={'Индекс':{'pooled':'По типу фонда','equity':'IMOEX','mixed':'IMOEX','bond':'RGBITR'}[group],
                    'Статус':r['status'],'N / фондов':str(r['nobs'])+' / '+str(r['funds']),
                    'Fund FE':'Да','Within R²':f"{r['within_R2']:.5f}" if r['status']=='ESTIMATED' else 'NA',
                    'Период origins':r.get('first_origin','NA')+' — '+r.get('last_origin','NA')}
                for term,name in [('NTS_Q_current','Текущий сигнал'),('NTS_Q_lagnews','Предыдущие новости'),('past_index_excess_return','Прошлая индексная доходность')]:
                    v=coef.get(term)
                    cell[name]=f"β={v['beta']:.6g}; SE={v['SE']:.6g}; p={v['p']:.6g}" if v else ('Не включён' if term=='NTS_Q_lagnews' and variant.startswith('CURRENT_ONLY') else 'NA')
                cell['Мало кластеров']='Да, выводы ненадёжны' if r['weak_cluster_inference'] else 'Нет'
                rows[labels[group]]=cell
            out+=['<h3>'+label+'</h3><div class="wrap">',pd.DataFrame(rows).to_html(escape=True),'</div>']
    out+=['<h2>Ограничения и чтение результатов</h2><p>Используется рост стоимости пая с известными корректировками; полнота выплат не подтверждена. IMOEX — ценовой индекс без дивидендов, RGBITR — индекс полной доходности государственных облигаций; сравнение не симметрично по выплатам. IMOEX для смешанных фондов — единый ориентир, не сбалансированный портфель. Разница с индексом предполагает экспозицию1, а не оценивает beta.</p>',
      '<p>Последний прошлый контроль может быть старше одного месяца; его возраст сохранён, ограничение давности не добавлено. Известные публикации после origin исключены из прошлого контроля, неизвестная публикационная давность остаётся ограничением. Нули NTS относятся к наблюдаемому корпоративному подпортуфелю, неизвестные позиции не заменены нулями. Кластеризация по фондам не покрывает общие шоки разных фондов. 24 обычных p не контролируют общую ошибку семейства.</p>',
      '<p>Все коэффициенты, 95%CI, полная covariance, суммы и joint-тесты, native summaries, поток выборки и датировка доступны рядом. Большие used samples и исходные месячные legs остаются наresearch-host. Независимая численная проверка и методический review — отдельные этапы.</p></html>']
    return ''.join(out)


def run(job_path):
    start=time.monotonic();job_path=Path(job_path);job=json.loads(job_path.read_text(encoding='utf-8'))
    code=Path(__file__).parent;spec=json.loads((code/'h3_index_excess_spec_20261104.json').read_text(encoding='utf-8'))
    file_record(code/'h3_index_excess_spec_20261104.json',SPEC_SHA)
    if spec['model_grid']['total_cells']!=24:raise ValueError('Only frozen24 cells supported')
    for r in job['inputs'].values():file_record(r['path'],r['sha256'])
    for r in job['code'].values():file_record(r['path'],r['sha256'])
    audit=json.loads(Path(job['inputs']['benchmark_audit']['path']).read_text(encoding='utf-8'))
    if audit['status']!=job['approved_benchmark_status']:raise ValueError('Benchmark approval status changed')
    out=ROOT/'outputs'/job['version'];out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():
        checkpoint('ALREADY_COMPLETED_NO_DUPLICATE',output=str(out));return
    input_manifest=dict(job=record_job(job_path),inputs=job['inputs'],code=job['code'],spec_sha256=SPEC_SHA,
        frozen_before_coefficients=True,created_utc=utc(),outcome='mean_monthly_index_excess_return',control='past_index_excess_return')
    save_json(out/'input_manifest.json',input_manifest)
    scores=pd.read_parquet(job['inputs']['scores']['path'])
    returns=pd.read_parquet(job['inputs']['fund_returns']['path'])
    benchmark=normalize_benchmarks(pd.read_parquet(job['inputs']['benchmark_monthly']['path']),job['benchmark_columns'])
    benchmark.to_parquet(out/'benchmark_monthly_adapter.parquet',index=False,compression='zstd')
    checkpoint('BUILDING_INDEX_RELATIVE_OUTCOMES_ALL_ORIGINS',output=str(out),origins=len(scores))
    outcomes,legs=index_excess_outcomes(scores,returns,benchmark,
        focus_benchmarks={g:spec['benchmark_assignment'][g]['series_id'] for g in ['equity','bond','mixed']},
        past_month_rule='latest_eligible_complete_calendar_month_at_or_before_origin',require_benchmark_future_anchor=False)
    outcomes.to_parquet(out/'origin_outcome_control_ledger.parquet',index=False,compression='zstd')
    legs.to_parquet(out/'monthly_outcome_control_legs.parquet',index=False,compression='zstd')
    scores=scores.drop(columns=['origin_month'],errors='ignore')
    panel=outcomes.merge(scores,on=['fund_id','interval_id'],how='left',validate='many_to_one')
    if not pd.to_datetime(panel.origin_date).eq(pd.to_datetime(panel.end_date)).all():raise AssertionError('Origin changed')
    panel['NTS_Q_current']=panel.NTS_Q_current_relaxed;panel['NTS_Q_lagnews']=panel.NTS_Q_lagnews_relaxed
    panel.to_parquet(out/'index_relative_analysis_panel.parquet',index=False,compression='zstd')
    save_json(out/'data_ready_before_coefficients.json',dict(ready_utc=utc(),spec_sha256=SPEC_SHA,
        input_manifest=file_record(out/'input_manifest.json'),outcomes=file_record(out/'origin_outcome_control_ledger.parquet'),
        legs=file_record(out/'monthly_outcome_control_legs.parquet'),benchmark=file_record(out/'benchmark_monthly_adapter.parquet'),
        analysis_panel=file_record(out/'index_relative_analysis_panel.parquet'),no_alpha_history_or_factor_gate=True))
    results=[];coefficients=[];tests=[];flow=[];overlap=[];dates=[]
    for group in ['pooled','equity','bond','mixed']:
        for h in [1,2,3]:
            base=panel.horizon.eq(h)&panel.universe_eligible
            if group!='pooled':base&=panel.fund_focus.eq(group)
            assigned=base&panel.benchmark_id.notna()
            fund_complete=assigned&panel.fund_future_months_eligible.eq(h)&panel.future_fund_anchor_verified
            outcome=fund_complete&panel.outcome_known
            controlled=outcome&panel.past_control_known
            current=controlled&panel.current_eligible;both=current&panel.lagnews_eligible
            ids_by_variant={}
            for distributed,variant in [(False,'CURRENT_ONLY_OWN_AVAILABLE'),(True,'CURRENT_PLUS_LAGNEWS')]:
                mask=both if distributed else current;sample=panel.loc[mask].copy()
                model_id=f'INDEX_EXCESS__{group}__h{h}__{variant}';model_out=out/'regression'/model_id
                done=model_out/'cell_complete.json'
                if done.exists():
                    marker=json.loads(done.read_text(encoding='utf-8'))
                    file_record(model_out/'fit.json',marker['fit_sha256']);file_record(model_out/'used_sample.parquet',marker['used_sha256'])
                    info=json.loads((model_out/'fit.json').read_text(encoding='utf-8'));used=pd.read_parquet(model_out/'used_sample.parquet')
                else:
                    info,used=fit_fund_fe(sample,distributed=distributed,output=model_out)
                    info.update(model_id=model_id,group=group,horizon=h,variant=variant,
                        outcome='mean_monthly_index_excess_return',control='past_index_excess_return',benchmark='origin_focus_specific' if group=='pooled' else ('RGBITR' if group=='bond' else 'IMOEX'))
                    if len(used):info.update(first_origin=str(pd.to_datetime(used.end_date).min().date()),last_origin=str(pd.to_datetime(used.end_date).max().date()),
                        first_future_month=used.future_first_month.min(),last_future_month=used.future_last_month.max())
                    save_json(model_out/'fit.json',info)
                    save_json(done,dict(fit_sha256=sha(model_out/'fit.json'),used_sha256=sha(model_out/'used_sample.parquet')))
                ids=set(used.row_id.astype(str));ids_by_variant[variant]=ids
                old_id=f'UNIT_VALUE_KNOWN_ADJUSTMENTS__{group}__h{h}__{variant}'
                old=pd.read_parquet(OLD/'regression'/old_id/'used_sample.parquet',columns=['row_id'])
                old_ids=set(old.row_id.astype(str))
                tags=dict(model_id=model_id,group=group,horizon=h,variant=variant)
                results.append(info)
                coefficients += [dict(tags,**c,nobs=info['nobs'],funds=info['funds']) for c in info.get('coefficients',[])]
                tests += [dict(tags,**t,nobs=info['nobs'],funds=info['funds']) for t in info.get('target_tests',[])]
                flow.append(dict(tags,inherited_universe=int(base.sum()),assigned_benchmark=int(assigned.sum()),
                    complete_future_fund_with_anchor=int(fund_complete.sum()),complete_future_fund_and_index=int(outcome.sum()),
                    past_control=int(controlled.sum()),current_known=int(current.sum()),both_known=int(both.sum()),
                    before_singletons=len(sample),final_N=info['nobs'],funds=info['funds'],status=info['status'],
                    current_zeros=int(used.NTS_Q_current.eq(0).sum()),current_known_lag_unknown=int((used.current_eligible&~used.lagnews_eligible).sum()),
                    past_control_older_available=int(used.control_uses_older_available_month.sum()),
                    old_relaxed_alpha_N=len(old_ids),overlap_old=len(ids&old_ids),added_vs_old=len(ids-old_ids),removed_vs_old=len(old_ids-ids)))
                if len(used):dates.append(dict(tags,N=len(used),funds=info['funds'],first_origin=info['first_origin'],last_origin=info['last_origin'],
                    first_future_month=info['first_future_month'],last_future_month=info['last_future_month'],
                    earliest_past_month=used.past_reference_month.min(),latest_past_month=used.past_reference_month.max(),
                    min_control_age_months=used.control_age_months.min(),max_control_age_months=used.control_age_months.max()))
                checkpoint('INDEX_EXCESS_MODEL_CELL',output=str(out),model=model_id,status=info['status'],N=info['nobs'],funds=info['funds'])
            own=ids_by_variant['CURRENT_ONLY_OWN_AVAILABLE'];dl=ids_by_variant['CURRENT_PLUS_LAGNEWS']
            overlap.append(dict(group=group,horizon=h,current_only=len(own),distributed=len(dl),overlap=len(own&dl),current_only_not_distributed=len(own-dl),distributed_not_current=len(dl-own)))
    if len(results)!=24:raise AssertionError('Exactly24 frozen cells')
    for name,data in [('coefficients',coefficients),('tests',tests),('sample_flow',flow),('current_vs_lag_overlap',overlap),('used_sample_date_coverage',dates)]:
        pd.DataFrame(data).to_csv(out/(name+'.csv'),index=False,encoding='utf-8-sig')
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in results]).to_csv(out/'models.csv',index=False,encoding='utf-8-sig')
    save_json(out/'all_model_details.json',results)
    (out/'review.html').write_text(render_tables(results),encoding='utf-8')
    panel.groupby(['horizon','fund_focus','outcome_status','past_status'],dropna=False).size().rename('origins').reset_index().to_csv(out/'outcome_NA_reason_counts.csv',index=False,encoding='utf-8-sig')
    summary=dict(status='INDEX_EXCESS24_COMPUTED_EXECUTOR_CHECKED_INDEPENDENT_QA_PENDING',output=str(out),spec_sha256=SPEC_SHA,
        model_cells=24,estimated_models=sum(r['status']=='ESTIMATED' for r in results),elapsed_seconds=time.monotonic()-start,
        no_alpha_history_or_factor_gate=True,primary_horizon=1,old_sources_unchanged=True,new_LLM=False,
        meaning='Mean monthly fund-minus-origin-focus-index return; not factor alpha or payout-complete verified total return.')
    save_json(out/'summary.json',summary)
    (out/'README_Data_Dictionary.txt').write_text('Outcome mean_monthly_index_excess_return (alias index_excess_return), decimal arithmetic mean of exactlyh future calendar months. Control past_index_excess_return is latest eligible observed past matched month, with calendar month-end, all endpoint dates and known publication <=origin; age/gaps saved without cutoff. No alpha/beta/RF/history gate. NTS values/zeroNA are unchanged. IMOEX for equity/mixed; RGBITR for bond; pooled by origin focus. Missing future month ->NA, not shorter mean. Fund future anchor>=actualorigin. Unit-value payouts not proven complete; benchmark conventions differ. Full covariance/native residual-df statistics separate from main t(G-1). G<30fragile/no stars. All24cells preserved. Raw monthly inputs retained at pinned source paths; all source fields for benchmark copied into its adapter; outcome/control legs retain source hashes/point IDs. Exact used samples stayPC. IndependentQA pending.\n',encoding='utf-8')
    for r in job['inputs'].values():file_record(r['path'],r['sha256'])
    for r in job['code'].values():file_record(r['path'],r['sha256'])
    seal_directory(out,dict(status=summary['status'],inputs=job['inputs'],code=job['code'],spec_sha256=SPEC_SHA,summary='summary.json'))
    export=ROOT/'exports'/job['version'];export.mkdir(parents=True,exist_ok=True);bundle=export/'h3_index_excess_reports.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file() and p.suffix in ['.json','.csv','.txt','.html']:z.write(p,str(p.relative_to(out)).replace('\\','/'))
    save_json(ROOT/'runtime/result_ready.json',dict(output=str(out),manifest_sha256=sha(out/'manifest.json'),bundle=file_record(bundle),summary=summary))
    checkpoint('INDEX_EXCESS24_DONE_QA_PENDING',output=str(out),estimated_models=summary['estimated_models'])


def record_job(path):return file_record(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);run(p.parse_args().job)
