"""Readonly executor checks; external numerical review remains separate."""
from pathlib import Path
import hashlib
import itertools
import json
import time
import duckdb
import numpy as np
import pandas as pd
from scipy import stats

ROOT=Path('./analysis_outputs/h3_index_excess_20261104')
OUT=ROOT/'outputs/21bd79859fefaa8e'
QA=ROOT/'qa/executor_21bd79859fefaa8e_v1'
EXPECTED='d0721c8a87067d5788ca12552212834daa5d7a2b4b18aa7417a5d10636b1cc51'
checks=[]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ck(name,ok,**kw):checks.append(dict(check=name,status='PASS' if bool(ok) else 'FAIL',**kw))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
def record(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))


def main():
    start=time.monotonic();QA.mkdir(parents=True,exist_ok=True)
    if (QA/'manifest.json').exists():raise RuntimeError('Already sealed, no overwrite')
    ck('run_manifest_pinned',sha(OUT/'manifest.json')==EXPECTED)
    m=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    for phase in ['before']:
        for group in ['inputs','code','files']:
            for name,r in m[group].items():ck(phase+':'+group+':'+name,Path(r['path']).stat().st_size==r['bytes'] and sha(r['path'])==r['sha256'])
    panel=pd.read_parquet(OUT/'index_relative_analysis_panel.parquet')
    scores=pd.read_parquet(m['inputs']['scores']['path']);funds=pd.read_parquet(m['inputs']['fund_returns']['path'])
    legs=pd.read_parquet(OUT/'monthly_outcome_control_legs.parquet')
    bench=pd.read_parquet(OUT/'benchmark_monthly_adapter.parquet')
    ck('all_selected_origins_all3_horizons',len(panel)==len(scores)*3 and not panel.duplicated(['fund_id','interval_id','horizon']).any())
    common=[c for c in scores if c in panel]
    one=panel.loc[panel.horizon.eq(1),common].sort_values(['fund_id','interval_id']).reset_index(drop=True)
    original=scores[common].sort_values(['fund_id','interval_id']).reset_index(drop=True)
    ck('all_reused_score_origin_universe_focus_fields_exact',one.equals(original),columns=len(common))
    expected_benchmark=panel.fund_focus.map({'equity':'IMOEX','mixed':'IMOEX','bond':'RGBITR'})
    ck('origin_focus_benchmark_assignment',(panel.benchmark_id.fillna('')==expected_benchmark.fillna('')).all())
    valid=bench.benchmark_verified
    ck('benchmark_ratios_positive_endpoints',bench.loc[valid,'index_start_value'].gt(0).all() and bench.loc[valid,'index_end_value'].gt(0).all()
        and np.allclose(bench.loc[valid,'benchmark_return'],bench.loc[valid,'index_end_value']/bench.loc[valid,'index_start_value']-1,rtol=1e-12,atol=1e-13))
    ck('no_alpha_or_history_columns_in_new_panel',not {'alpha','past_alpha','n_past','beta_equity_return','factor_observation_eligible'}&set(panel))
    c=duckdb.connect(':memory:',config={'threads':2,'memory_limit':'2GB'})
    for name,d in [('p',panel),('s',scores),('f',funds),('b',bench),('l',legs)]:c.register(name,d)
    def sqlck(name,sql):
        violations=int(c.execute(sql).fetchone()[0]);ck(name,violations==0,violations=violations)
    sqlck('exact_future_months_and_means',"""WITH a AS (SELECT p.fund_id,p.interval_id,p.horizon,
      count(*) FILTER(WHERE l.component_verified) known,avg(l.index_excess_component) expected
      FROM p JOIN l ON l.fund_id=p.fund_id AND l.interval_id=p.interval_id AND l.kind='FUTURE'
      AND l.month>=p.future_first_month AND l.month<=p.future_last_month GROUP BY ALL)
      SELECT count(*) FROM p JOIN a USING(fund_id,interval_id,horizon)
      WHERE p.outcome_known AND (known<>p.horizon OR abs(p.index_excess_return-a.expected)>1e-12
      OR NOT p.future_fund_anchor_verified OR NOT p.identity_origin_verified)""")
    sqlck('past_selected_month_and_all_endpoints_before_origin',"""SELECT count(*) FROM l WHERE kind='PAST' AND component_verified
      AND (last_day(CAST(month||'-01' AS DATE))>origin_date OR fund_return_period_end>origin_date
      OR benchmark_return_period_end>origin_date OR fund_start_valuation_date>origin_date OR fund_end_valuation_date>origin_date
      OR benchmark_start_valuation_date>origin_date OR benchmark_end_valuation_date>origin_date)""")
    sqlck('all_known_leg_differences_exact',"SELECT count(*) FROM l WHERE component_verified AND abs(index_excess_component-(fund_return-benchmark_return))>1e-12")
    c.execute("""CREATE TEMP TABLE latest AS SELECT s.fund_id,s.interval_id,max(f.month) expected_month
      FROM s JOIN f ON s.fund_id=f.fund_id JOIN b ON b.month=f.month AND b.benchmark_id=CASE WHEN s.fund_focus='bond' THEN 'RGBITR' WHEN s.fund_focus IN ('equity','mixed') THEN 'IMOEX' END
      WHERE f.diagnostic_eligible AND f.identity_verified AND isfinite(f.fund_return) AND f.fund_return>=-1
      AND b.benchmark_verified AND isfinite(b.benchmark_return) AND b.benchmark_return>-1
      AND last_day(CAST(f.month||'-01' AS DATE))<=s.end_date AND f.return_period_end<=s.end_date
      AND f.start_valuation_date<=s.end_date AND f.end_valuation_date<=s.end_date AND f.start_valuation_date<=f.end_valuation_date
      AND b.return_period_end<=s.end_date AND b.start_valuation_date<=s.end_date AND b.end_valuation_date<=s.end_date
      AND b.start_valuation_date<=b.end_valuation_date AND (f.publication_timestamp IS NULL OR f.publication_timestamp<=s.end_date)
      GROUP BY s.fund_id,s.interval_id""")
    sqlck('latest_available_past_pair_no_fixed_month_or_age_cut',"SELECT count(*) FROM p LEFT JOIN latest USING(fund_id,interval_id) WHERE past_reference_month IS DISTINCT FROM expected_month")
    sqlck('past_age_saved_no_false_lag1_label',"SELECT count(*) FROM p WHERE past_reference_month IS NOT NULL AND control_age_months<>date_diff('month',CAST(past_reference_month||'-01' AS DATE),origin_date)")
    history=c.execute("""SELECT s.fund_id,s.interval_id,count(*) n_history FROM s JOIN f ON s.fund_id=f.fund_id
      WHERE f.diagnostic_eligible AND f.identity_verified AND isfinite(f.fund_return) AND f.return_period_end<=s.end_date AND f.end_valuation_date<=s.end_date GROUP BY ALL""").fetchdf()
    models=json.loads((OUT/'all_model_details.json').read_text(encoding='utf-8'))
    expected=set(itertools.product(['pooled','equity','bond','mixed'],[1,2,3],['CURRENT_ONLY_OWN_AVAILABLE','CURRENT_PLUS_LAGNEWS']))
    ck('exact24_cells',len(models)==24 and {(r['group'],r['horizon'],r['variant']) for r in models}==expected)
    short_history={}
    for r in models:
        mid=r['model_id'];base=panel.horizon.eq(r['horizon'])&panel.universe_eligible&panel.outcome_known&panel.past_control_known&panel.current_eligible
        if r['group']!='pooled':base&=panel.fund_focus.eq(r['group'])
        if r['variant']=='CURRENT_PLUS_LAGNEWS':base&=panel.lagnews_eligible
        wanted=panel.loc[base];wanted=wanted.loc[wanted.groupby('fund_id').fund_id.transform('size').ge(2)]
        used=pd.read_parquet(OUT/'regression'/mid/'used_sample.parquet')
        ck(mid+':exact_own_sample',len(wanted)==len(used) and set(wanted.row_id)==set(used.row_id))
        ck(mid+':N_G',len(used)==r['nobs'] and used.fund_id.nunique()==r['funds'])
        h=used[['fund_id','interval_id']].merge(history,on=['fund_id','interval_id'],how='left',validate='one_to_one')
        short_history[mid]=int(h.n_history.lt(36).sum())
        if r['status']!='ESTIMATED':continue
        ck(mid+':native_vs_within_and_CR1',r['QA']['status']=='PASS' and r['QA']['native_within_slope_max_abs_error']<1e-8 and r['QA']['native_CR1_max_abs_error']<1e-8)
        coeff=pd.DataFrame(r['coefficients']).set_index('term');cov=pd.read_csv(OUT/'regression'/mid/'full_covariance.csv',index_col=0)
        g=r['funds'];q=stats.t.ppf(.975,g-1)
        for term in r['terms']:
            beta=coeff.loc[term,'beta'];se=np.sqrt(cov.loc[term,term]);pvalue=2*stats.t.sf(abs(beta/se),g-1)
            ck(mid+':tG_CI:'+term,np.isclose(coeff.loc[term,'SE'],se) and np.isclose(coeff.loc[term,'p'],pvalue,rtol=1e-7,atol=1e-10)
                and np.isclose(coeff.loc[term,'CI_low'],beta-q*se) and np.isclose(coeff.loc[term,'CI_high'],beta+q*se))
        if r['distributed']:
            targets=['NTS_Q_current','NTS_Q_lagnews'];block=cov.loc[targets,targets].to_numpy();betas=coeff.loc[targets,'beta'].to_numpy()
            summ=next(x for x in r['target_tests'] if x['test']=='current+lagnews=0')
            ck(mid+':sum_uses_full_cov',np.isclose(summ['p'],2*stats.t.sf(abs(betas.sum()/np.sqrt(block.sum())),g-1),rtol=1e-7,atol=1e-10))
    ck('short_history_origins_really_admitted',sum(short_history.values())>0)
    c.close()
    for group in ['inputs','code']:
        for name,r in m[group].items():ck('after:'+group+':'+name,sha(r['path'])==r['sha256'])
    ck('sealed_manifest_unchanged',sha(OUT/'manifest.json')==EXPECTED)
    result=dict(status='PASS' if all(r['status']=='PASS' for r in checks) else 'FAIL',role='EXECUTOR_CHECKS_NOT_EXTERNAL_QA',
        passed=sum(r['status']=='PASS' for r in checks),failed=sum(r['status']=='FAIL' for r in checks),
        output=str(OUT),manifest_sha256=EXPECTED,models=24,elapsed_seconds=time.monotonic()-start,
        used_rows_with_fewer_than36_verified_past_months_by_model=short_history,checks=checks)
    save(QA/'qa.json',result)
    save(QA/'manifest.json',dict(status=result['status'],source_manifest_sha256=EXPECTED,
        files={'qa.json':record(QA/'qa.json')}))
    save(ROOT/'runtime/executor_qa_ready.json',dict(status=result['status'],passed=result['passed'],failed=result['failed'],output=str(QA),manifest_sha256=sha(QA/'manifest.json')))
    print(json.dumps({k:v for k,v in result.items() if k not in ['checks','used_rows_with_fewer_than36_verified_past_months_by_model']},default=str),flush=True)


if __name__=='__main__':main()
