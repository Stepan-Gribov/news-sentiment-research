"""Frozen48-model relaxed-news H3, separate new output and cached alpha reuse."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile
import numpy as np
import pandas as pd

from h3_alpha_core_20261104 import rolling_future_alpha,numeric,truth,unique
from h3_alpha_io_20261104 import sha,file_record,save_json,seal_directory,utc
from fit_h3_alpha_fe_20261104 import fit_fund_fe
from build_h3_relaxed_panel_20261104 import ROOT,OLD,SPEC_SHA,checkpoint

DIAG='UNIT_VALUE_KNOWN_ADJUSTMENTS';TR='TR_PROVIDER_ACCRUED_INCOME_CONVENTION'
V3=OLD/'outputs/run_d0440fad412ee5e9';TR4=OLD/'outputs/provider_TR4_b00ab703609fb58f'

def extend_alpha(scores,returns,factors,cached,out):
    """Evaluate eligible fund origins before any news gate for honest flow.

    This includes the entire own-current/distributed union and reports the
    alpha-only->current->both flow without reclassifying unevaluated returns.
    Existing NA risk/history/future outcomes are also reused verbatim.
    """
    candidate=scores.loc[truth(scores.universe_eligible)].copy()
    cached_keys=set(zip(cached.fund_id.astype(str),cached.interval_id.astype(str)))
    needs=candidate.loc[[(str(f),str(i)) not in cached_keys for f,i in zip(candidate.fund_id,candidate.interval_id)]]
    save_json(out/'alpha_extension_plan.json',dict(requested_universe_origins=len(candidate),requested_current_origins=int(candidate.current_eligible.sum()),
        cached_origins=len(cached_keys),new_origins=len(needs),lag_requirement=False))
    portions=[cached];funds=sorted(needs.fund_id.astype(str).unique())
    chunks=[funds[i:i+20] for i in range(0,len(funds),20)]
    for k,ids in enumerate(chunks):
        stem=out/f'new_alpha_chunk_{k:04d}';done=stem.with_suffix('.done.json')
        if done.exists():
            meta=json.loads(done.read_text(encoding='utf-8'));file_record(stem.with_suffix('.parquet'),meta['sha256'])
            a=pd.read_parquet(stem.with_suffix('.parquet'))
        else:
            origins=needs.loc[needs.fund_id.astype(str).isin(ids),['fund_id','interval_id','end_date','identity_origin_verified']]
            a,m=rolling_future_alpha(origins,returns.loc[returns.fund_id.astype(str).isin(ids)],factors,
                factor_definitions={'equity_return':'minus_rf','bond_return':'minus_rf'},
                lookback_months=60,minimum_history=36,minimum_recent=9,recent_months=12,horizons=(1,2,3),condition_max=10000)
            a.to_parquet(stem.with_suffix('.parquet'),index=False,compression='zstd')
            m.to_parquet(out/f'new_training_chunk_{k:04d}.parquet',index=False,compression='zstd')
            save_json(done,dict(sha256=sha(stem.with_suffix('.parquet')),funds=ids))
        portions.append(a)
        checkpoint('EXTENDING_DIAGNOSTIC_ALPHA_CURRENT_OWN_SAMPLE',completed_chunks=k+1,total_chunks=len(chunks),
                   additional_origins=len(needs),output=str(out))
    combined=pd.concat(portions,ignore_index=True)
    unique(combined,['fund_id','interval_id','horizon'],'cached plus new alpha')
    # No old outcome, beta or NA status may change in this news-only extension.
    old=combined.merge(cached,on=['fund_id','interval_id','horizon'],how='inner',validate='one_to_one',suffixes=('_new','_old'))
    for c in ['alpha','past_alpha','beta_equity_return','beta_bond_return','n_past','n_past_recent']:
        if not np.allclose(numeric(old[c+'_new']),numeric(old[c+'_old']),rtol=0,atol=0,equal_nan=True):raise AssertionError('Cached outcome changed '+c)
    combined.to_parquet(out/'alpha_all_reused_and_extended.parquet',index=False,compression='zstd')
    return combined,dict(requested_universe_origins=len(candidate),requested_current_origins=int(candidate.current_eligible.sum()),cached_origins=len(cached_keys),new_origins=len(needs))

def old_sample(branch,group,h,distributed):
    variant='CURRENT_PLUS_LAGNEWS' if distributed else 'CURRENT_COMMON'
    identifier=f'{branch}__PRIMARY__{group}__h{h}__{variant}'
    path=(V3/'regression'/branch/'PRIMARY'/identifier/'used_sample.parquet') if branch==DIAG else (TR4/'regression/PRIMARY'/identifier/'used_sample.parquet')
    d=pd.read_parquet(path)
    return set(d.row_id.astype(str)) if len(d) else set()

def run():
    start=time.monotonic();code=Path(__file__).parent
    spec_path=code/'h3_relaxed_news_spec_20261104.json';file_record(spec_path,SPEC_SHA)
    ready_path=ROOT/'runtime/panel_ready.json'
    if not ready_path.exists():raise RuntimeError('Panel must be sealed before model task starts')
    ready=json.loads(ready_path.read_text(encoding='utf-8'));panel_root=Path(ready['output'])
    file_record(panel_root/'manifest.json',ready['manifest_sha256'])
    panel_manifest=json.loads((panel_root/'manifest.json').read_text(encoding='utf-8'))
    for rec in panel_manifest['files'].values():file_record(rec['path'],rec['sha256'])
    inputs=dict(spec=file_record(spec_path,SPEC_SHA),panel_manifest=file_record(panel_root/'manifest.json'),
        v3_manifest=file_record(V3/'manifest.json','d61349f8275762eee3c224a8118d69eec81fb08107d48a6fc00f63b324a09628'),
        TR4_manifest=file_record(TR4/'manifest.json','af25eb9dc57288ef54d2b0839477f0ced6b9539708bd3a88a8cbd2257b8e98c3'),
        scores=file_record(panel_root/'relaxed_fund_period_scores.parquet'),
        diagnostic_returns=file_record(V3/'monthly_returns_adapter.parquet'),factors=file_record(V3/'monthly_factors_adapter.parquet'),
        diagnostic_cached_alpha=file_record(V3/'alpha/UNIT_VALUE_KNOWN_ADJUSTMENTS/HISTORY60_MIN36/alpha.parquet'),
        TR4_cached_alpha=file_record(TR4/'lookback60_min36_alpha.parquet'))
    for name in ['run_h3_relaxed_news_20261104.py','h3_relaxed_news_core_20261104.py','fit_h3_alpha_fe_20261104.py','h3_alpha_core_20261104.py']:
        inputs['code:'+name]=file_record(code/name)
    version=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()[:16]
    out=ROOT/'outputs'/version;out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise RuntimeError('Already completed; no duplicate')
    for key,rec in list(inputs.items()):
        if not key.startswith('code:'):continue
        dest=out/'code'/Path(rec['path']).name;dest.parent.mkdir(exist_ok=True)
        if not dest.exists():shutil.copy2(rec['path'],dest)
        inputs[key]=file_record(dest,rec['sha256'])
    save_json(out/'input_manifest.json',dict(inputs=inputs,version=version,spec_sha256=SPEC_SHA,frozen_before_new_coefficients=True))
    scores=pd.read_parquet(inputs['scores']['path'])
    # Expose only the new independent scores to the estimator's fixed term names.
    scores['NTS_Q_current']=numeric(scores.NTS_Q_current_relaxed)
    scores['NTS_Q_lagnews']=numeric(scores.NTS_Q_lagnews_relaxed)
    returns=pd.read_parquet(inputs['diagnostic_returns']['path'])
    returns['return_verified']=truth(returns.diagnostic_eligible)
    factors=pd.read_parquet(inputs['factors']['path'])
    cached=pd.read_parquet(inputs['diagnostic_cached_alpha']['path'])
    checkpoint('REUSING_ALPHA_AND_EXTENDING_ONLY_ADDITIONAL_CURRENT_ORIGINS',output=str(out),new_coefficients_estimated=False)
    alpha_dir=out/'alpha';alpha_dir.mkdir(exist_ok=True)
    diag,cache_counts=extend_alpha(scores,returns,factors,cached,alpha_dir)
    tr=pd.read_parquet(inputs['TR4_cached_alpha']['path'])
    tr_funds=set(tr.fund_id.astype(str))
    required=scores.loc[scores.fund_id.astype(str).isin(tr_funds)&scores.universe_eligible&scores.current_eligible]
    missing=set(zip(required.fund_id.astype(str),required.interval_id.astype(str)))-set(zip(tr.fund_id.astype(str),tr.interval_id.astype(str)))
    if missing:raise AssertionError('Old TR4 cache does not cover all new current-score origins; explicit extension needed')
    save_json(alpha_dir/'reuse_TR4.json',dict(path=inputs['TR4_cached_alpha']['path'],sha256=inputs['TR4_cached_alpha']['sha256'],
        all_original_397_origins_reused=True,new_TR4_alpha_computed=0,required_current_origins=len(required)))
    results=[];flow=[];coef=[];tests=[];fitted_dates=[];own_overlap=[]
    source_score=scores.drop(columns=['origin_month'],errors='ignore')
    for branch,alpha in [(DIAG,diag),(TR,tr)]:
        panel=alpha.merge(source_score,on=['fund_id','interval_id'],how='left',validate='many_to_one')
        if not pd.to_datetime(panel.origin_date).eq(pd.to_datetime(panel.end_date)).all():raise AssertionError('Origin dates differ')
        panel['alpha_known']=np.isfinite(numeric(panel.alpha))&np.isfinite(numeric(panel.past_alpha))
        panel.to_parquet(out/('outcome_score_panel_'+branch+'.parquet'),index=False,compression='zstd')
        for group in ['pooled','equity','bond','mixed']:
            for h in [1,2,3]:
                base=panel.horizon.eq(h)&truth(panel.universe_eligible)
                if group!='pooled':base&=panel.fund_focus.eq(group)
                selected_masks={}
                for distributed,variant in [(False,'CURRENT_ONLY_OWN_AVAILABLE'),(True,'CURRENT_PLUS_LAGNEWS')]:
                    mask=base&panel.alpha_known&panel.current_eligible
                    if distributed:mask&=panel.lagnews_eligible
                    sample=panel.loc[mask].copy()
                    selected_masks[variant]=set(sample.row_id.astype(str))
                    model_id=f'{branch}__{group}__h{h}__{variant}'
                    info,used=fit_fund_fe(sample,distributed=distributed,output=out/'regression'/model_id)
                    prior_ids=old_sample(branch,group,h,distributed)
                    ids=set(used.row_id.astype(str)) if len(used) else set()
                    tags=dict(model_id=model_id,branch=branch,group=group,horizon=h,variant=variant)
                    info.update(tags)
                    info.update(overlap_old_primary=len(ids&prior_ids),added_vs_old=len(ids-prior_ids),removed_vs_old=len(prior_ids-ids),
                        own_sample_does_not_require_lag=not distributed,
                        current_zero_rows=int(used.NTS_Q_current.eq(0).sum()) if len(used) else 0,
                        current_known_lag_unknown_rows=int((used.current_eligible&~used.lagnews_eligible).sum()) if len(used) else 0)
                    results.append(info);save_json(out/'regression'/model_id/'fit.json',info)
                    coef += [dict(tags,**c,nobs=info['nobs'],funds=info['funds']) for c in info.get('coefficients',[])]
                    tests += [dict(tags,**t,nobs=info['nobs'],funds=info['funds']) for t in info.get('target_tests',[])]
                    group_origins=scores.loc[truth(scores.universe_eligible)&(scores.fund_focus.eq(group) if group!='pooled' else True)]
                    flow.append(dict(tags,source_origins=len(group_origins),source_funds=int(group_origins.fund_id.nunique()),
                        alpha_defined=int((base&panel.alpha_known).sum()),current_known=int((base&panel.alpha_known&panel.current_eligible).sum()),
                        both_known=int((base&panel.alpha_known&panel.current_eligible&panel.lagnews_eligible).sum()),
                        before_singletons=len(sample),final_N=info['nobs'],final_funds=info['funds'],status=info['status'],
                        added_vs_old=info['added_vs_old'],overlap_old=info['overlap_old_primary'],removed_vs_old=info['removed_vs_old'],
                        current_zero=info['current_zero_rows'],known_current_unknown_lag=info['current_known_lag_unknown_rows']))
                    if len(used):
                        fitted_dates.append(dict(tags,status=info['status'],N=len(used),funds=int(used.fund_id.nunique()),
                            first_origin=str(pd.to_datetime(used.end_date).min().date()),last_origin=str(pd.to_datetime(used.end_date).max().date()),
                            first_future_month=used.future_first_month.min(),last_future_month=used.future_last_month.max()))
                    if len(used):
                        flags=[c for c in used if c.startswith(('current_zero_','lagnews_zero_'))]
                        save_json(out/'regression'/model_id/'zero_reason_counts.json',{c:int(truth(used[c]).sum()) for c in flags})
                    checkpoint('RELAXED_MODEL_CELL',output=str(out),model=model_id,status=info['status'],N=info['nobs'],funds=info['funds'])
                current=selected_masks['CURRENT_ONLY_OWN_AVAILABLE'];both=selected_masks['CURRENT_PLUS_LAGNEWS']
                if not both.issubset(current):raise AssertionError('Distributed sample not a subset of own current before singleton gates')
                own_overlap.append(dict(branch=branch,group=group,horizon=h,current_only=len(current),distributed=len(both),overlap=len(current&both),current_without_lag=len(current-both)))
    if len(results)!=48:raise AssertionError('Exactly48 fixed cells required')
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in results]).to_csv(out/'models.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(coef).to_csv(out/'coefficients.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(tests).to_csv(out/'tests.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(flow).to_csv(out/'sample_flow_old_new.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(own_overlap).to_csv(out/'current_own_vs_distributed_overlap.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(fitted_dates).to_csv(out/'used_sample_date_coverage.csv',index=False,encoding='utf-8-sig')
    save_json(out/'all_model_details.json',results)
    comparison=pd.read_parquet(panel_root/'old_new_score_comparison.parquet')
    comparison.groupby(['fund_focus','current_status','lagnews_status'],dropna=False).size().rename('origins').reset_index().to_csv(out/'score_zero_NA_status_counts.csv',index=False,encoding='utf-8-sig')
    scope=truth(comparison.universe_eligible)
    ratio={}
    for label in ['current','lagnews']:
        oldcol='old_NTS_'+label;newcol='NTS_Q_'+label+'_relaxed'
        shared=scope&np.isfinite(numeric(comparison[oldcol]))&np.isfinite(numeric(comparison[newcol]))
        ratio[label]=dict(shared_known_origins=int(shared.sum()),unchanged_ratio=int((shared&comparison[label+'_same_numeric_ratio']).sum()),
            changed_ratio=int((shared&~comparison[label+'_same_numeric_ratio']).sum()),new_known_zero_origins=int((scope&comparison[newcol].eq(0)&comparison[oldcol].isna()).sum()))
    save_json(out/'old_new_ratio_equivalence.json',ratio)
    summary=dict(status='RELAXED_NEWS48_COMPUTED_EXECUTOR_CHECKED_INDEPENDENT_QA_PENDING',output=str(out),spec_sha256=SPEC_SHA,
        panel=str(panel_root),panel_counts=ready['counts'],alpha_cache_reuse=cache_counts,TR4_alpha_recomputed=0,
        models=48,estimated_models=sum(r['status']=='ESTIMATED' for r in results),ratio_comparison=ratio,
        elapsed_seconds=time.monotonic()-start,new_LLM_or_news_collection=False,old_outputs_unchanged=True,
        inference='Fund CR1,t(G-1); G<30 fragile/no stars; no time FE/clusters; no Holm',
        interpretation='Observed corporate-subportfolio alignment. Known zero mixes inactivity, no-news, neutral/balanced and offsetting cases; unknowns remain NA outside E and do not veto a supported observed-subset zero. Own-current/distributed samples differ; no causal attribution to adding lag.')
    save_json(out/'summary.json',summary)
    display=pd.DataFrame(flow)[['branch','group','horizon','variant','status','final_N','final_funds','current_zero','added_vs_old','overlap_old']]
    html='<!doctype html><meta charset="utf-8"><style>body{font:18px system-ui;margin:32px}td,th{padding:7px;border:1px solid #ddd}table{border-collapse:collapse}</style><h1>H3: ослабленные ограничения новостей, отдельный прогон</h1><p>48 заранее зафиксированных ячеек. Текущий сигнал и лаг независимы; подтверждённые нули не означают нейтральную LLM-разметку или отсутствие торговли во всём фонде. Прежние outcomes и alpha60/36 сохранены. TR: только четыре фонда одной УК, мало кластеров.</p>'+display.to_html(index=False)
    if coef:html+='<h2>Коэффициенты: основные p по t(G−1), без поправки на множественные проверки</h2>'+pd.DataFrame(coef).to_html(index=False)
    html+='<h2>Тесты, включая совместные и суммы</h2>'+pd.DataFrame(tests).to_html(index=False)
    (out/'review.html').write_text(html,encoding='utf-8')
    (out/'README_Data_Dictionary.txt').write_text('Zero/NA: see sealed panel company_window_status_ledger and relaxed_security_window_ledger. Missing values are not set0. Scores use independent current/lag E; B_Q unchanged and P99 fixed. Old known Net preserved despite undated audit flags; new absence0 requires date evidence. current-only own available rows do not require lag. Outcomes alpha are mean monthly decimal abnormal returns, not cumulative; beta*10000 is bps per+1 score. Fund CR1 and t(G-1) main inference; native residual-df output separate; no stars atG<30. Exact used samples and ledgers stayPC. Adding zero products need not change a normalized ratio: old_new_ratio_equivalence reports this separately from sample expansion. No new LLM, H1/H2 change or source overwrite. Independent QA pending.\n',encoding='utf-8')
    for rec in inputs.values():file_record(rec['path'],rec['sha256'])
    seal_directory(out,dict(status=summary['status'],inputs=inputs,summary='summary.json',version=version))
    export=ROOT/'exports'/version;export.mkdir(parents=True,exist_ok=True)
    bundle=export/'h3_relaxed_news_reports.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in out.rglob('*'):
            if p.is_file() and p.suffix in ['.json','.csv','.txt','.html'] and 'alpha' not in p.relative_to(out).parts:
                archive.write(p,str(p.relative_to(out)).replace('\\','/'))
    save_json(export/'export_manifest.json',dict(bundle=file_record(bundle),source_manifest_sha256=sha(out/'manifest.json'),output=str(out)))
    save_json(ROOT/'runtime/result_ready.json',dict(output=str(out),manifest_sha256=sha(out/'manifest.json'),bundle=str(bundle),summary=summary))
    checkpoint('RELAXED_NEWS_DONE_QA_PENDING',output=str(out),models=48,estimated_models=summary['estimated_models'])

if __name__=='__main__':run()
