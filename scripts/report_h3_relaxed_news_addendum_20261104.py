"""Separate reporting-only addendum; no regressions or source/old-output edits."""
from pathlib import Path
import hashlib
import html
import json
import shutil
import zipfile
import pandas as pd

ROOT=Path('./analysis_outputs/h3_relaxed_news_20261104')
RUN=ROOT/'outputs/63d2bda2bad14bec'
OUT=ROOT/'exports/63d2bda2bad14bec/reporting_addendum_v2'
DIAG='UNIT_VALUE_KNOWN_ADJUSTMENTS'
TR='TR_PROVIDER_ACCRUED_INCOME_CONVENTION'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def record(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')


def main():
    if (OUT/'manifest.json').exists():raise RuntimeError('Reporting addendum already sealed')
    OUT.mkdir(parents=True,exist_ok=True)
    code=OUT/'code'/Path(__file__).name;code.parent.mkdir(exist_ok=True)
    shutil.copy2(__file__,code)
    before=sha(RUN/'manifest.json')
    assert before=='43085f4b2992939dce794dabbed1992fe70d039f664aee7c67cb5a23ad5c1f0e'
    old=pd.read_csv(RUN/'sample_flow_old_new.csv')
    panels={b:pd.read_parquet(RUN/('outcome_score_panel_'+b+'.parquet')) for b in [DIAG,TR]}
    flow=[]
    for _,r in old.iterrows():
        p=panels[r.branch]
        if r['group']!='pooled':p=p.loc[p.fund_focus.eq(r['group'])]
        origins=p.drop_duplicates(['fund_id','interval_id'])
        eligible=origins.loc[origins.universe_eligible]
        row=r.to_dict()
        row['all_fund_universe_origins']=row.pop('source_origins')
        row['all_fund_universe_funds']=row.pop('source_funds')
        row.update(branch_origins_before_universe=len(origins),branch_funds_before_universe=origins.fund_id.nunique(),
            branch_universe_origins=len(eligible),branch_universe_funds=eligible.fund_id.nunique())
        assert row['alpha_defined']<=row['branch_universe_origins']
        assert row['both_known']<=row['current_known']<=row['alpha_defined']
        flow.append(row)
    flow=pd.DataFrame(flow)
    flow.to_csv(OUT/'sample_flow_branch_specific.csv',index=False,encoding='utf-8-sig')
    coef=pd.read_csv(RUN/'coefficients.csv');tests=pd.read_csv(RUN/'tests.csv')
    target=coef.loc[coef.term.isin(['NTS_Q_current','NTS_Q_lagnews'])].copy()
    target.to_csv(OUT/'all_target_coefficients.csv',index=False,encoding='utf-8-sig')
    signals=tests.loc[tests.p.lt(.05)].copy()
    signals.to_csv(OUT/'all_nominal_target_tests_below_05.csv',index=False,encoding='utf-8-sig')
    pooled=target.loc[target['group'].eq('pooled')]
    qa=json.loads((ROOT/'qa/executor_63d2bda2bad14bec_20261104_v1/qa.json').read_text(encoding='utf-8'))
    summary=json.loads((RUN/'summary.json').read_text(encoding='utf-8'))
    facts=dict(status='COMPUTED_EXECUTOR_QA_PASS_EXTERNAL48_QA_PENDING',models=48,estimated=36,not_estimable=12,
        elapsed_seconds=summary['elapsed_seconds'],executor_QA={k:qa[k] for k in ['passed','failed','numerical_replicas']},
        old_sealed_run_manifest_sha256=before,frozen_spec_sha256=summary['spec_sha256'],
        actual_panel=summary['panel'],actual_output=str(RUN),
        pooled_diagnostic_current_only_p=pooled.loc[pooled.branch.eq(DIAG)&pooled.variant.eq('CURRENT_ONLY_OWN_AVAILABLE'),'p'].tolist(),
        nominal_target_tests_below_05=signals.where(pd.notna(signals),None).to_dict(orient='records'),
        score_change='Independent E changes4853 current ratios among8663 shared known eligible-universe origins; not just adding zeros.',
        historical_QA_issue='The reported builder/core mismatch concerns superseded panel5adc30709c6f4cc2, not actual panel0a4d9c7365f0faa1 used by sealed run63d2bda2bad14bec. No old hash was replaced.',
        reporting_correction='Original sample_flow source_origins/source_funds describe the global all-fund universe, even in TR rows. This separate addendum renames those columns explicitly and adds true branch-specific origin/fund counts. All alpha/current/both/used counts and fits are unchanged.',
        limitations=['Broad outcome is factor-adjusted unit-value growth, not verified total return.',
            'TR4 covers four funds of one manager; equity models have only two fund clusters and no identified two-target joint test.',
            'Ordinary unadjusted p-values; post-baseline specification extension; no positive H3 confirmation.',
            'Observed-subportfolio zeros are not whole-fund inactivity, no-news evidence or an LLM-neutral class.',
            'Executor checks are not the external independent reviewer verdict.'])
    save(OUT/'summary.json',facts)
    nice=flow[['branch','group','horizon','variant','branch_universe_origins','alpha_defined','current_known','both_known','final_N','final_funds','status']]
    ptable=pooled[['branch','horizon','variant','term','beta','SE','p','nobs','funds']]
    parts=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>H3 — обновлённые результаты</title>',
        '<style>body{font:18px/1.5 system-ui;max-width:1550px;margin:36px auto;padding:0 24px;color:#20242a}table{border-collapse:collapse;font-size:15px}td,th{border:1px solid #d4d9de;padding:7px;text-align:left}th{background:#f2f4f7}.wrap{overflow:auto}li{margin:8px 0}code{font-size:15px}</style>',
        '<h1>H3: независимые текущий и предыдущий новостные сигналы</h1>',
        '<p>Отдельное расширение прежней спецификации, зафиксированное до новых коэффициентов. 48 ячеек, 36 оценённых моделей, 12 неоценимых. Расчёт — 2 мин 54 с. Проверка исполнителя: 1615 PASS / 0 FAIL, 36 численных реплик. Независимая проверка итоговых 48 ячеек ещё не завершена.</p>',
        '<h2>Главный результат</h2><p>В широкой диагностической выборке ни текущий сигнал, ни предыдущий, ни их сумма или совместный тест не проходят уровень 5% — ни в общей выборке, ни по типам фондов. На горизонте одного месяца текущая модель использует 12 795 наблюдений / 405 фондов; модель с предыдущими новостями — 12 788 / 405. Прежние исходные наблюдения сохранены.</p>',
        '<p>В TR-поднаборе общая модель имеет 68 наблюдений / 4 фонда. Два номинальных отрицательных результата ниже5% относятся лишь к двум акционным фондам: текущий сигнал h2 (p=0,027128) и сумма текущего/предыдущего h1 (p=0,042850). Это крайне слабая кластерная опора; положительную H3 не подтверждает. Для облигационного и смешанного TR-поднаборов всего по17 наблюдений /1 фонду: оценки и выводы не выдумываются.</p>',
        '<h2>Как читать скоры</h2><p>Текущий и предыдущий сигнал строятся на независимых наборах измеримых позиций. Подтверждённые нули относятся к наблюдаемому корпоративному подпортуфелю; неизвестные позиции остаются NA. Убраны пороги50% покрытия и двух компаний. 119 235 позиций восстановлены только из прежних новостных исключений; прежние неновостные правила и P99 сохранены. Это меняет и выборку, и часть ненулевых отношений, а не просто добавляет нули.</p>',
        '<h2>Общие выборки: все коэффициенты интереса</h2><div class="wrap">',ptable.to_html(index=False,float_format=lambda x:f'{x:.6g}'),'</div>',
        '<h2>Все номинальные тесты ниже5% (без поправок)</h2><div class="wrap">',signals[['branch','group','horizon','variant','test','beta','p','nobs','funds']].to_html(index=False,float_format=lambda x:f'{x:.6g}'),'</div>',
        '<h2>Поток выборки по ветвям</h2><p>В исходном sealed CSV столбцы source_origins/source_funds описывали глобальную universe, в том числе в строках TR. Здесь отдельно указаны реальные начальные размеры каждой ветви; все следующие численности и регрессии неизменны.</p><div class="wrap">',nice.to_html(index=False),'</div>',
        '<h2>Даты и ограничения</h2><p>Диагностические использованные origins: 31.12.2013—29.05.2026; TR: 31.01.2025—29.05.2026. Последние будущие месяцы: июнь / июль / август2026 для h1/h2/h3. Диагностический рост пая не выдаётся за подтверждённую TR; TR взята из228 одобренных месячных доходностей четырёх фондов однойУК. Контроль — только past_alpha, fundFE, CR1 по фондам, t(G−1), без временных эффектов и поправок. Изменение коэффициентов между текущей и лаговой моделью нельзя приписывать одному добавлению лага: выборки различаются.</p>',
        '<h2>Сохранность и версии</h2><p>Прежние результаты, H1/H2, protected inputs и Gemma не изменялись. Новые корпуса или LLM-прогоны не запускались. Расхождение версий кода относилось к неиспользованной ранней панели5adc; использованная0a4d совпадает с записанными хешами. Точные копии кода закреплены отдельным manifest без правки старых хешей.</p>',
        '<p>Большие панели и exact used samples остаются на research-host: <code>'+html.escape(str(RUN))+'</code>. Малые таблицы, native summaries, covariance и исходный обзор находятся в соседней папке отчёта.</p></html>']
    (OUT/'practical_summary.html').write_text(''.join(parts),encoding='utf-8')
    assert sha(RUN/'manifest.json')==before
    save(OUT/'manifest.json',dict(status='REPORTING_ONLY_NO_REFIT',source_model_manifest_sha256=before,
        source_panel_manifest_sha256='2fb48b5ae9be0063e330181df925cb8afabbec2b604cbd3c878d6979b7e0fa92',
        frozen_spec_sha256=summary['spec_sha256'],code=record(code),
        supersedes_reporting_only='reporting_addendum_v1; numerical results unchanged; utility code is now an immutable embedded snapshot',
        files={str(p.relative_to(OUT)).replace('\\','/'):record(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='manifest.json'}))
    bundle=OUT.parent/'reporting_addendum_v2.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file():z.write(p,str(p.relative_to(OUT)).replace('\\','/'))
    print(json.dumps(dict(output=str(OUT),manifest=record(OUT/'manifest.json'),bundle=record(bundle)),default=str),flush=True)


if __name__=='__main__':main()
