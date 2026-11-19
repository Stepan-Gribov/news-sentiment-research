"""Read-only executor QA and exact-code provenance addendum for relaxed H3.

Not the independent reviewer's verdict. No production imports or model search;
only the frozen 48 cells are checked, using saved exact samples. Old manifests
are never repaired or resealed. New evidence has its own immutable directory.
"""
from pathlib import Path
import datetime as dt
import hashlib
import itertools
import json
import os
import shutil
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import duckdb
import numpy as np
import pandas as pd
from scipy import stats

ROOT=Path('./analysis_outputs/h3_relaxed_news_20261104')
RUN=ROOT/'outputs/63d2bda2bad14bec'
PANEL=ROOT/'panels/0a4d9c7365f0faa1'
OUT=ROOT/'qa/executor_63d2bda2bad14bec_20261104_v1'
SNAP=ROOT/'provenance/panel_0a4d9c7365f0faa1_20261104_v1'
SPEC='36a64070c5dfcba749d7baa1f15b639f9801a39fcac12142e372a94a2554699f'
RUN_SHA='43085f4b2992939dce794dabbed1992fe70d039f664aee7c67cb5a23ad5c1f0e'
PANEL_SHA='2fb48b5ae9be0063e330181df925cb8afabbec2b604cbd3c878d6979b7e0fa92'
DIAG='UNIT_VALUE_KNOWN_ADJUSTMENTS'
TR='TR_PROVIDER_ACCRUED_INCOME_CONVENTION'
checks=[]


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def record(p):
    p=Path(p)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))


def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')


def ck(name,ok,**kw):
    checks.append(dict(check=name,status='PASS' if bool(ok) else 'FAIL',**kw))


def seal(path,metadata):
    path=Path(path)
    if (path/'manifest.json').exists():raise RuntimeError('Do not overwrite an existing evidence seal')
    metadata=dict(metadata,sealed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        files={str(p.relative_to(path)).replace('\\','/'):record(p)
               for p in sorted(path.rglob('*')) if p.is_file() and p.name!='manifest.json'})
    save(path/'manifest.json',metadata)


def check_records(manifest,label):
    for group in ['inputs','files']:
        for name,r in manifest.get(group,{}).items():
            p=Path(r['path'])
            ck(label+':'+group+':'+name,p.is_file() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'])


def main():
    start=time.monotonic()
    if (OUT/'manifest.json').exists():raise RuntimeError('Executor QA already sealed; no overwrite')
    OUT.mkdir(parents=True,exist_ok=True)
    pm=json.loads((PANEL/'manifest.json').read_text(encoding='utf-8'))
    rm=json.loads((RUN/'manifest.json').read_text(encoding='utf-8'))
    ck('actual_panel_manifest',sha(PANEL/'manifest.json')==PANEL_SHA)
    ck('actual_model_manifest',sha(RUN/'manifest.json')==RUN_SHA)
    ck('actual_model_uses_actual_panel',rm['inputs']['panel_manifest']['sha256']==PANEL_SHA)
    ck('spec_V3',pm['spec_sha256']==SPEC and rm['inputs']['spec']['sha256']==SPEC)
    check_records(pm,'panel_before');check_records(rm,'run_before')
    # Preserve exact recorded bytes under new, never-reused snapshot paths.
    # The old5adc panel's mutable-code mismatch remains historical evidence;
    # it is not the panel referenced by the new model input manifest.
    SNAP.mkdir(parents=True,exist_ok=True)
    snapshots={}
    items={'builder':pm['inputs']['builder'],'scoring_core':pm['inputs']['core'],
        'frozen_spec':pm['inputs']['spec']}
    for k,r in rm['inputs'].items():
        if k.startswith('code:'):items[k]=r
    for key,r in items.items():
        source=Path(r['path']);dest=SNAP/source.name
        if sha(source)!=r['sha256']:raise AssertionError('Recorded executable bytes unavailable: '+key)
        if not dest.exists():shutil.copy2(source,dest)
        if sha(dest)!=r['sha256']:raise AssertionError('Snapshot conflict: '+key)
        snapshots[key]=record(dest)
    if not (SNAP/'manifest.json').exists():
        save(SNAP/'original_code_hash_evidence.json',dict(
            original_panel=record(PANEL/'manifest.json'),original_run=record(RUN/'manifest.json'),
            exact_original_hashes_verified=True,old_manifests_modified=False,
            earlier_unused_panel='5adc30709c6f4cc2',
            earlier_panel_note='Predates R1/R2 corrections; old mutable paths no longer identify its original code. Not an input to63d2 model run. Preserved, not repaired.',
            snapshots=snapshots))
        seal(SNAP,dict(status='EXACT_RECORDED_CODE_SNAPSHOTS',snapshots=snapshots,
            source_panel_manifest_sha256=PANEL_SHA,source_model_manifest_sha256=RUN_SHA))
    ck('snapshot_seal_exists',(SNAP/'manifest.json').is_file())
    c=duckdb.connect(':memory:',config={'threads':2,'memory_limit':'2GB'})
    def view(name,path):
        path=str(path).replace('\\','/').replace("'","''")
        c.execute('CREATE VIEW '+name+" AS SELECT * FROM read_parquet('"+path+"')")
    def sqlck(name,sql):
        n=int(c.execute(sql).fetchone()[0]);ck(name,n==0,violations=n)
    view('n',PANEL/'relaxed_security_window_ledger.parquet')
    view('s',PANEL/'relaxed_fund_period_scores.parquet')
    view('w',PANEL/'company_window_status_ledger.parquet')
    view('d',pm['inputs']['decisions']['path'])
    threshold=json.loads(Path(pm['inputs']['threshold']['path']).read_text(encoding='utf-8'))['cutoff_percent']
    ck('inherited_P99',abs(threshold-322.7771642245575)<1e-10)
    sqlck('physical_mask_reconstructed_from_source_decisions',"""SELECT count(*) FROM n JOIN d USING(fund_id,interval_id,security_id)
      WHERE n.quantity_in_scope IS DISTINCT FROM
      (coalesce(array_to_string(list_filter(string_split(coalesce(d.exclusion_reason_augmented,''),';'),x->x<>'' AND x NOT IN
       ('sentiment_not_primary','missing_net_sentiment','missing_control_log_news_count_interval')),';'),'')=''
       AND NOT(coalesce(d.trimmed_upper_p99,false) OR coalesce(d.quantity_open>0 AND d.quantity_close>0
       AND d.quantity_change_pct>322.7771642245575,false)))""")
    sqlck('old_finite_Net_preserved_even_undated',"SELECT count(*) FROM w WHERE eligible_primary AND isfinite(net_sentiment) AND window_valid AND mapping_in_scope AND value IS DISTINCT FROM net_sentiment")
    sqlck('absence0_has_no_undated_or_errors',"SELECT count(*) FROM w WHERE news_status IN ('ZERO_NO_CANDIDATES_IN_FROZEN_CORPUS','ZERO_ALL_CANDIDATES_IRRELEVANT') AND (n_undated_company_candidates<>0 OR n_errors_or_unattempted<>0 OR n_relevant<>0)")
    for label in ['current','lagnews']:
        eligible=('quantity_in_scope AND quantity_action_eligible AND coalesce(isfinite(B_Q),false) AND mapping_in_scope '
            'AND company_id IS NOT NULL AND '+label+'_window_valid AND '
            '((coalesce('+label+'_news_known,false) AND coalesce(isfinite('+label+'_news_value),false) AND '+label+'_source_present) OR B_Q=0)')
        c.execute('CREATE TEMP TABLE e_'+label+' AS SELECT *,coalesce(('+eligible+'),false) qa_E FROM n')
        sqlck(label+':independent_E','SELECT count(*) FROM e_'+label+' WHERE '+label+'_usable_E IS DISTINCT FROM qa_E')
        sqlck(label+':unknown_products_remain_NA','SELECT count(*) FROM e_'+label+' WHERE NOT qa_E AND ('+label+'_product IS NOT NULL OR '+label+'_weight IS NOT NULL)')
        c.execute('CREATE TEMP TABLE a_'+label+' AS SELECT fund_id,interval_id,count(*) FILTER(WHERE qa_E) n_E,'
            'sum(CASE WHEN qa_E THEN CASE WHEN B_Q=0 THEN 0. ELSE B_Q*'+label+'_news_value END ELSE 0. END) num,'
            'sum(CASE WHEN qa_E THEN CASE WHEN B_Q=0 THEN 0. ELSE abs(B_Q)*abs('+label+'_news_value) END ELSE 0. END) den,'
            'sum(CASE WHEN qa_E THEN abs(B_Q) ELSE 0. END) gross FROM e_'+label+' GROUP BY fund_id,interval_id')
        calc='CASE WHEN a.n_E>0 THEN CASE WHEN a.den<=1e-12*greatest(1.,a.gross) THEN 0. ELSE a.num/a.den END END'
        sqlck(label+':all_independent_ratios_and_zeros','SELECT count(*) FROM s LEFT JOIN a_'+label+' a USING(fund_id,interval_id) WHERE '
            'coalesce(isfinite(s.NTS_Q_'+label+'_relaxed),false) IS DISTINCT FROM coalesce(isfinite('+calc+'),false) OR '
            '(isfinite('+calc+') AND abs(s.NTS_Q_'+label+'_relaxed-('+calc+'))>1e-10)')
        sqlck(label+':outside_or_missing_window_never_usable','SELECT count(*) FROM n WHERE '+label+'_usable_E AND '
            '(NOT mapping_in_scope OR company_id IS NULL OR NOT '+label+'_window_valid OR NOT quantity_in_scope OR NOT quantity_action_eligible)')
    expanded=pd.read_parquet(RUN/'alpha/alpha_all_reused_and_extended.parquet')
    cached=pd.read_parquet(rm['inputs']['diagnostic_cached_alpha']['path'])
    keys=['fund_id','interval_id','horizon']
    shared=expanded.merge(cached,on=keys,how='inner',validate='one_to_one',suffixes=('_new','_old'))
    ck('all_cached_alpha_rows_reused',len(shared)==len(cached))
    for col in ['alpha','past_alpha','beta_equity_return','beta_bond_return','n_past','n_past_recent']:
        ck('cached_alpha_exact:'+col,np.allclose(shared[col+'_new'],shared[col+'_old'],rtol=0,atol=0,equal_nan=True))
    scores=pd.read_parquet(PANEL/'relaxed_fund_period_scores.parquet')
    wanted=set(map(tuple,scores.loc[scores.universe_eligible,['fund_id','interval_id']].astype(str).to_numpy()))
    got=set(map(tuple,expanded[['fund_id','interval_id']].astype(str).drop_duplicates().to_numpy()))
    ck('alpha_covers_all_universe_origins_before_news',wanted.issubset(got),expected=len(wanted),actual=len(got))
    branches={b:pd.read_parquet(RUN/('outcome_score_panel_'+b+'.parquet')) for b in [DIAG,TR]}
    models=json.loads((RUN/'all_model_details.json').read_text(encoding='utf-8'))
    expected=set(itertools.product([DIAG,TR],['pooled','equity','bond','mixed'],[1,2,3],['CURRENT_ONLY_OWN_AVAILABLE','CURRENT_PLUS_LAGNEWS']))
    ck('exact48_fixed_cells',len(models)==48 and {(r['branch'],r['group'],r['horizon'],r['variant']) for r in models}==expected)
    numeric_replicas=[]
    for r in models:
        mid=r['model_id'];p=branches[r['branch']]
        mask=p.horizon.eq(r['horizon'])&p.universe_eligible&p.current_eligible&np.isfinite(p.alpha)&np.isfinite(p.past_alpha)
        distributed=r['variant']=='CURRENT_PLUS_LAGNEWS'
        if distributed:mask&=p.lagnews_eligible
        if r['group']!='pooled':mask&=p.fund_focus.eq(r['group'])
        expected_sample=p.loc[mask].copy()
        expected_sample=expected_sample.loc[expected_sample.groupby('fund_id').fund_id.transform('size').ge(2)]
        used=pd.read_parquet(RUN/'regression'/mid/'used_sample.parquet')
        ck(mid+':exact_sample_no_hidden_lag_gate',set(expected_sample.row_id.astype(str))==set(used.row_id.astype(str)) and len(expected_sample)==len(used))
        ck(mid+':dimensions',len(used)==r['nobs'] and used.fund_id.nunique()==r['funds'])
        if r['status']!='ESTIMATED':
            ck(mid+':honest_not_estimable',len(used)<30 or used.fund_id.nunique()<2 or bool(r.get('absorbed_columns')) or 'collinear' in r.get('reason',''))
            continue
        n=len(used);g=used.fund_id.nunique();terms=r['terms']
        x=(used[terms]-used[terms].groupby(used.fund_id).transform('mean')).to_numpy(float)
        y=(used.alpha-used.alpha.groupby(used.fund_id).transform('mean')).to_numpy(float)
        b=np.linalg.lstsq(x,y,rcond=None)[0];res=y-x@b
        cluster=pd.Categorical(used.fund_id).codes;meat=np.zeros((g,len(terms)))
        np.add.at(meat,cluster,x*res[:,None]);bread=np.linalg.inv(x.T@x)
        cov=g/(g-1)*(n-1)/(n-g-len(terms))*bread@(meat.T@meat)@bread
        reported=pd.DataFrame(r['coefficients']).set_index('term')
        full_cov=pd.read_csv(RUN/'regression'/mid/'full_covariance.csv',index_col=0)
        ck(mid+':replicated_slopes',np.allclose(b,reported.loc[terms,'beta'],rtol=1e-7,atol=1e-10))
        ck(mid+':replicated_CR1',np.allclose(cov,full_cov.loc[terms,terms],rtol=1e-6,atol=1e-11))
        se=np.sqrt(np.diag(cov));pvalue=2*stats.t.sf(abs(b/se),g-1);critical=stats.t.ppf(.975,g-1)
        ck(mid+':replicated_tGminus1_CI',np.allclose(pvalue,reported.loc[terms,'p'],rtol=1e-6,atol=1e-9)
            and np.allclose(b-critical*se,reported.loc[terms,'CI_low'],rtol=1e-6,atol=1e-10)
            and np.allclose(b+critical*se,reported.loc[terms,'CI_high'],rtol=1e-6,atol=1e-10))
        ck(mid+':fragile_clusters_flag',r['weak_cluster_inference']==(g<30) and r['significance_stars_suppressed']==(g<30))
        if distributed:
            block=cov[:2,:2];total_beta=float(b[:2].sum());total_var=float(block.sum())
            sum_test=next(t for t in r['target_tests'] if t['test']=='current+lagnews=0')
            joint=next(t for t in r['target_tests'] if t['test']=='joint_current_lagnews=0')
            ck(mid+':full_covariance_sum_test',np.isclose(total_beta,sum_test['beta'],rtol=1e-7,atol=1e-10)
                and np.isclose(2*stats.t.sf(abs(total_beta/np.sqrt(total_var)),g-1),sum_test['p'],rtol=1e-6,atol=1e-9))
            identified=g>=3 and np.linalg.matrix_rank(block)==2 and np.linalg.eigvalsh(block).min()>0
            if identified:
                f=float(b[:2]@np.linalg.solve(block,b[:2])/2)
                ck(mid+':joint_test',joint['status']=='defined' and np.isclose(stats.f.sf(f,2,g-1),joint['p'],rtol=1e-6,atol=1e-9))
            else:ck(mid+':unidentified_joint_is_NA',not np.isfinite(joint.get('p',np.nan)))
        numeric_replicas.append(dict(model_id=mid,nobs=n,funds=g,
            slope_error=float(np.max(abs(b-reported.loc[terms,'beta'].to_numpy()))),
            covariance_error=float(np.max(abs(cov-full_cov.loc[terms,terms].to_numpy())))))
    c.close()
    check_records(pm,'panel_after');check_records(rm,'run_after')
    ck('original_manifest_bytes_unchanged',sha(PANEL/'manifest.json')==PANEL_SHA and sha(RUN/'manifest.json')==RUN_SHA)
    pd.DataFrame(numeric_replicas).to_csv(OUT/'numerical_replicas.csv',index=False)
    result=dict(status='PASS' if all(r['status']=='PASS' for r in checks) else 'FAIL',
        role='EXECUTOR_QA_NOT_EXTERNAL_REVIEW',passed=sum(r['status']=='PASS' for r in checks),failed=sum(r['status']=='FAIL' for r in checks),
        model_cells=48,numerical_replicas=len(numeric_replicas),elapsed_seconds=time.monotonic()-start,
        frozen_spec_sha256=SPEC,run=record(RUN/'manifest.json'),panel=record(PANEL/'manifest.json'),
        provenance=record(SNAP/'manifest.json'),checks=checks)
    save(OUT/'qa.json',result);seal(OUT,dict(status=result['status'],role=result['role'],summary='qa.json'))
    save(ROOT/'runtime/executor_qa_ready.json',dict(output=str(OUT),manifest_sha256=sha(OUT/'manifest.json'),
        status=result['status'],passed=result['passed'],failed=result['failed'],provenance=str(SNAP)))
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},default=str),flush=True)


if __name__=='__main__':main()
