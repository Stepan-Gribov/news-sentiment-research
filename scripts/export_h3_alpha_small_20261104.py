"""Package only small H3 reports. No holdings, article or used-sample export."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import zipfile

import pandas as pd

from h3_alpha_io_20261104 import ROOT, save_json, sha, utc, file_record


def run(run_root):
    source=Path(run_root);version=source.name.removeprefix('run_')
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    destination=ROOT/'exports'/version;destination.mkdir(parents=True,exist_ok=True)
    summary=json.loads((source/'summary.json').read_text(encoding='utf-8'))
    models=pd.read_csv(source/'models.csv')
    coverage=[];fund_coverage=[]
    origins=pd.read_parquet(source/'origin_eligibility.parquet')
    for row in models.loc[models.sensitivity.eq('PRIMARY')&models.variant.eq('CURRENT_PLUS_LAGNEWS')].itertuples(index=False):
        directory=source/'regression'/row.branch/'PRIMARY'/row.model_id
        sample=pd.read_parquet(directory/'used_sample.parquet')
        data=dict(branch=row.branch,group=row.group,horizon=row.horizon,status=row.status,nobs=len(sample),
            funds=int(sample.fund_id.nunique()),first_origin=None,last_origin=None,
            first_future_month=None,last_future_month=None,first_training_month_allowed=None,last_training_month_observed=None)
        if len(sample):
            data.update(first_origin=str(pd.to_datetime(sample.end_date).min().date()),
                last_origin=str(pd.to_datetime(sample.end_date).max().date()),
                first_future_month=str(sample.future_first_month.min()),last_future_month=str(sample.future_last_month.max()),
                first_training_month_allowed=str(sample.training_first_allowed.min()),last_training_month_observed=str(sample.training_last_observed.max()))
        coverage.append(data)
        if row.branch=='UNIT_VALUE_KNOWN_ADJUSTMENTS' and row.group=='pooled':
            counts=sample.groupby('fund_id',observed=True).interval_id.nunique().rename(f'used_h{row.horizon}')
            fund_coverage.append(counts)
    pd.DataFrame(coverage).to_csv(destination/'coverage_by_group_horizon.csv',index=False,encoding='utf-8-sig')
    # Per-fund availability stays on PC. Only aggregated group/year coverage is
    # included in the portable archive below.
    fund=origins.groupby('fund_id',observed=True).agg(source_origins=('interval_id','nunique'),
        NTS_eligible=('nts_pair_eligible','sum'),mandate_eligible=('universe_eligible','sum'),
        dated_origin_identity_verified=('identity_origin_verified','sum'))
    for counts in fund_coverage:fund=fund.join(counts,how='left')
    fund.to_parquet(destination/'fund_level_availability_PC_ONLY.parquet')
    selection=[]
    for group in ['pooled','equity','bond','mixed']:
        d=origins if group=='pooled' else origins.loc[origins.fund_focus.eq(group)]
        selection.append(dict(group=group,source_funds=int(d.fund_id.nunique()),source_origins=len(d),
            universe_origins=int(d.universe_eligible.sum()),NTS_eligible_origins=int(d.nts_pair_eligible.sum()),
            eligible_NTS_universe=int((d.universe_eligible&d.nts_pair_eligible).sum()),
            origin_identity_verified=int(d.identity_origin_verified.sum())))
    pd.DataFrame(selection).to_csv(destination/'selection_flow_by_group.csv',index=False,encoding='utf-8-sig')
    dictionary='''H3 — first sealed subset, 2026-11-04

STATUS
The numerical output requires independent numerical and methodological review.
TR_VERIFIED primary H3 has no verified complete return series in this release.
UNIT_VALUE_KNOWN_ADJUSTMENTS is a separate diagnostic of factor-adjusted unit
value growth. It is not a verified total-return alpha or proof of fund skill.

OBSERVATION AND SAMPLES
One fund × actual closing-report interval, at most one origin per closing month.
The latest closing date is selected before gates; there is no earlier fallback.
All 603 inherited funds and all 28,976 calendar origins remain in origin audits
on PC. Main NTS uses the frozen inherited 221,899 security-row set, while its
coverage denominator includes all measurable union-held corporate positions.
Single-observation funds are dropped separately within each regression group.
Consequently fitted group counts need not sum exactly to the fitted pooled N.
Historical fund-focus classifications can vary over time.

NTS_Q_CURRENT, NTS_Q_LAGNEWS
B_Q=(Q1_adjusted_to_opening_units-Q0)*P0 for continuing positions/exits;
confirmed entries use Q1*P1. Both are signed RUB value proxies, not raw share
counts, not H1 artificial entry +100%, and not observed discretionary trades.
Current=sum_E(B_Q*Net_current)/sum_E(abs(B_Q)*abs(Net_current)).
Lagnews uses the SAME CURRENT trades and PREVIOUS interval company news;
it is not a lagged fund trading score. Both scores use the identical common E.
Observed Net=0 is valid; unknown news or quantity is NA, not zero.
Primary requires >=2 contributing companies and >=0.5 measurable news-trade
coverage for each score. Broad NTS is only a predeclared TR sensitivity.
Asset-value coverage uses all positive disclosed position values, not a claim
of independently reconciled NAV. The missing-position-value slot count in the
NTS ledger also includes securities absent at that endpoint of the union.
Documented domestic and unknown geography are separate; no ISIN inference.
Known scheduled redemptions are excluded from E but retained in measured U.
Three verified quoted-share unit changes are adjusted; unknown action-history
completeness is retained as a limitation. Bond V/P uses the clean-price proxy.

FUND RETURNS AND ALPHA
Source: observed Investfunds per-unit values, not growth of total fund NAV.
Only adjacent observed months, positive values, dated identity, RUB and source
freshness <=3 trading days. Known unresolved cash-event fund series are excluded
from this first diagnostic; unknown payout histories are not called empty.
Fund214's known 100:1 split is already back-adjusted in the source; not applied
again. Source disagreements with the official chart are preserved.
Primary factors: MCFTR gross total return, RGBITR total return, CBR RUONIA.
RUONIA monthly RF uses I_close(d)=officialIndex(d+1 calendar day), consistently
for both training and future months. No RUCBITR/RUCBTRNS splice.
Past betas: 60 calendar months ending at last calendar EOM<=actual origin,
>=36 complete observations, >=9 of the latest12, full rank and condition<=10000.
Even a final trading day before calendar EOM cannot use that month's full return.
Future months: the next full calendar months, h=1,2,3; actual first unit-value
anchor must be >=origin. All h legs required; no partial averages or imputation.
Alpha_h is MEAN MONTHLY fund excess less fixed past-beta factor exposures;
the past intercept is NOT subtracted. It is retained as past_alpha control.
Not annualized and not cumulative h-month performance.

REGRESSIONS AND INFERENCE
Fund FE only, no time FE. No security-row replication of a fund-level outcome.
Distributed: alpha~NTS_current+NTS_lagnews+past_alpha+fundFE.
Current-only comparison uses the exact SAME observations.
Unweighted OLS; one-way fund-cluster CR1 correction:
G/(G-1)*(N-1)/(N-K_full), K_full including absorbed fund effects.
Main p and95%CI: ordinary two-sided Student t with G-1df.
Native TXT/HTML retain the package's residual-df inference, not silently replaced.
Sum and joint tests use the complete target covariance. Undefined/singular tests
are retained; no PSD clipping or arbitrary p=1. <30 fund clusters is fragile.
Beta unit: decimal average monthly return per +1 score; *10,000 gives basis
points. -1 to+1 contrast is *20,000; observed IQR contrasts are separate.
R2_within and overall_R2_native retain their distinct package definitions.
All114 planned labels retained:24 diagnostic baseline fits,90 unestimable TR
labels in this first release. No Holm, outcome cuts, control search or new LLM.

FILES
summary.json: source hashes, sample counts, limitations and actual run status.
models.csv: every planned cell, status, N, G, reference df, R2 and sample hash.
coefficients.csv: coefficient, SE, t, p, CI, basis-point and IQR contrasts.
tests.csv: current, lagnews, sum and joint tests; covariance-based inference.
coverage_by_group_horizon.csv: actual fitted origin and future date limits.
selection_flow_by_group.csv: available/eligible source selection counts.
regression/*/native.summary.txt|html: original package output.
regression/*/coefficients_main_tGminus1.csv: main cluster-df inference.
regression/*/coefficients_native.csv: original residual-df p values.
regression/*/full_covariance.csv: full covariance including intercept/control.
All large contributions, observed monthly inputs, training/future memberships,
origin decisions, fund-level availability and exact used samples stay on research-host.
Input resources, prior H1/H2/H3 outputs, and live Gemma were not modified.
'''
    (destination/'README_Data_Dictionary.txt').write_text(dictionary,encoding='utf-8')
    release=summary.get('fund_release','v1_existing_cache')
    if release=='v3_action_aware':
        dictionary=dictionary.replace('first sealed subset','v3 action-aware extension')
        dictionary=dictionary.replace('Known unresolved cash-event fund series are excluded\nfrom this first diagnostic; unknown payout histories are not called empty.',
            'Known announced record/payment calendar months are excluded in v3;\nraw event-month flags are independently reconstructed. No ex-date is invented\nand no payout is added on an undocumented date. Remaining payout uncertainty\nis not asserted absent. The v1 fund-wide cash-event exclusion is not retained.')
        dictionary+='\nV1/V3 COMPARISON\nThe previous v1 result is preserved. comparison_v1_v3_models.csv reports N/fund\nchanges for every label; comparison_v1_v3_coverage_and_overlap.csv gives actual\nperiod limits, row overlap, added and removed observations by group/horizon.\nReturn exclusion tables retain the changed source/action eligibility policies.\nAll frozen sensitivity families are listed in frozen_sensitivity_audit.csv;\nTR-only sensitivity cells are not estimable when no verified TR months exist,\neven though factors and broad NTS are available. They are not silently waived.\n'
        (destination/'README_Data_Dictionary.txt').write_text(dictionary,encoding='utf-8')
    coefficients=pd.read_csv(source/'coefficients.csv')
    current=coefficients.loc[coefficients.variant.eq('CURRENT_PLUS_LAGNEWS')&coefficients.term.eq('NTS_Q_current')].copy()
    labels={'pooled':'Все допустимые фонды','equity':'Фонды акций','bond':'Облигационные','mixed':'Смешанные'}
    compact=[]
    for row in current.itertuples(index=False):
        compact.append({'Группа':labels.get(row.group,row.group),'Горизонт, мес.':row.horizon,
            'Наблюдения':row.nobs,'Фонды':row.funds,'β текущего NTS, б.п.':round(row.beta*10000,2),
            '95% ДИ, б.п.':f'[{row.CI_low*10000:.2f}; {row.CI_high*10000:.2f}]','p':f'{row.p:.4f}'})
    page='''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>H3 — первый расчёт</title>
<style>body{font:18px/1.5 system-ui;margin:32px auto;padding:0 22px;max-width:1350px;color:#172235}h1{line-height:1.2}table{border-collapse:collapse;width:100%;font-size:16px}th,td{padding:9px 11px;border-bottom:1px solid #ccd3dc;text-align:right}th:first-child,td:first-child{text-align:left}.notice{padding:16px 20px;border-left:5px solid #cb731e;background:#fff5e8}.muted{color:#536070}details{margin:18px 0}</style>
<h1>H3: первый зафиксированный расчёт</h1>
<div class="notice"><strong>Основная H3 пока не оценивается.</strong> Нет подтверждённой полноты выплат по паям. Таблица ниже — отдельная диагностика факторно-скорректированного роста стоимости пая, а не проверенная полная доходность и не доказательство навыка управляющего.</div>
<p>603 исходных фонда; 28 976 отчётных дат; 7 850 дат с допустимым основным NTS и мандатом. Оценены все 24 базовые диагностические модели. Все 90 недоступных ячеек основной ветки сохранены как NOT_ESTIMABLE.</p>
<h2>Текущая торговля, согласованная с новостями</h2>
<p>Модель с текущим NTS, NTS по предыдущим новостям, прошлой alpha и эффектами фондов. β — средняя месячная доходность в базисных пунктах при росте текущего NTS на 1.</p>'''
    page+=pd.DataFrame(compact).to_html(index=False,escape=True)
    if release=='v3_action_aware':
        page=page.replace('первый зафиксированный расчёт','расширение v3 с учётом известных выплат')
        comparison_file=source/'comparison_v1_v3_coverage_and_overlap.csv'
        if comparison_file.exists():
            comparison=pd.read_csv(comparison_file)
            comparison=comparison.loc[comparison.branch.eq('UNIT_VALUE_KNOWN_ADJUSTMENTS'),[
                'group','horizon','N_v1','N_v3','common_rows','added_rows','removed_rows','last_origin_v1','last_origin_v3',
                'last_future_month_v1','last_future_month_v3']]
            page+='<h2>Расширение относительно v1</h2><div style="overflow:auto">'+comparison.to_html(index=False,escape=True)+'</div>'
    positive=int((current.beta.gt(0)&current.p.lt(.05)).sum())
    if positive:
        page+=f'<p>Положительный коэффициент текущего NTS имеет обычное p&lt;0,05 в {positive} из {len(current)} диагностических ячеек. Это не подтверждение основной H3 по полной доходности и не скорректированный за множественные проверки результат.</p>'
    else:
        page+='<p>Положительная связь текущего NTS не достигает 5% ни в одной из этих диагностических ячеек. Это не заменяет основную H3 и не доказывает отсутствие эффекта.</p>'
    page+='''<p>Все p — обычные двусторонние, без поправки за множественные проверки.</p>
<details><summary>Расчёт, неопределённость и ограничения</summary><p>Прошлые экспозиции оценены на 60 месяцах, минимум 36 наблюдений и 9 из последних 12. Обучение не использует данные после отчётной даты. Горизонты 1–3 — полные будущие месяцы. Fund FE, один кластер по фонду, CR1, p и ДИ по t(G−1); без временных эффектов и кластеров. Полнота выплат/дроблений и публичность архивных оценок на дату сделки не гарантированы. NTS основан на V/P-количествах, не на наблюдаемых сделках.</p></details>
<p class="muted">Внутренняя сверка вторым численным способом: отдельный QA-файл. Независимая проверка и исследовательская интерпретация ещё требуются. Полные native TXT/HTML, коэффициенты, ковариации и все ячейки находятся в ZIP; большие исходники и точные выборки — только на домашнем ПК.</p></html>'''
    (destination/'quick_review.html').write_text(page,encoding='utf-8')
    whitelist=['summary.json','models.csv','coefficients.csv','tests.csv','review.html','input_manifest.json',
               'manifest.json','all_model_details.json','fund_return_exclusion_counts.csv',
               'origin_selection_flow_by_group_year.csv','synthetic_FE_QA.json',
               'alpha_gate_counts_TR_VERIFIED.csv','alpha_gate_counts_UNIT_VALUE_KNOWN_ADJUSTMENTS.csv',
               'frozen_sensitivity_audit.json','frozen_sensitivity_audit.csv','comparison_v1_v3.json',
               'comparison_v1_v3_models.csv','comparison_v1_v3_coverage_and_overlap.csv',
               'comparison_v1_v3_return_exclusions.csv','source_return_exclusion_counts.csv']
    selected=[p for name in whitelist if (p:=source/name).exists()]
    for file in (source/'regression').rglob('*'):
        if not file.is_file():continue
        if file.name.endswith(('.summary.txt','.summary.html')) or file.suffix=='.csv' or file.name=='fit.json':selected.append(file)
    qa=[p for p in (ROOT/'qa').glob(source.name+'_*/qa.json')]
    bundle=destination/'h3_results_small.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for file in selected:archive.write(file,str(file.relative_to(source)).replace('\\','/'))
        for file in destination.glob('*.csv'):archive.write(file,file.name)
        archive.write(destination/'README_Data_Dictionary.txt','README_Data_Dictionary.txt')
        archive.write(destination/'quick_review.html','quick_review.html')
        for file in qa:archive.write(file,'qa/'+file.parent.name+'/qa.json')
    record=dict(status='SMALL_REPORTS_EXPORTED_NO_BULK',created_utc=utc(),run_root=str(source),
        source_manifest_sha256=sha(source/'manifest.json'),bundle=file_record(bundle),
        parquet_exported=False,private_used_sample_exported=False,source_files_unchanged=True)
    save_json(destination/'export_manifest.json',record)
    print(json.dumps(record,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root');a=p.parse_args()
    root=a.run_root or json.loads((ROOT/'runtime/alpha_ready.json').read_text(encoding='utf-8'))['output']
    run(root)
