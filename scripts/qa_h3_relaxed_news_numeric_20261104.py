"""External numerical QA of the sealed48-cell run, never a model executor.
Imports only the auditor's own prepared LSDV/CR1 implementation, not production.
Provenance acceptance remains separate until the panel reissue is resolved.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib,datetime as dt,time,traceback
import numpy as np,pandas as pd,duckdb
from scipy import linalg,stats
from qa_h3_relaxed_news_independent_20261104 import independent_full_dummy_cr1,ordered_sample_sha256,expected_model_sample,planned_cells,RELAXED_SPEC_SHA256

BASE=Path('./analysis_outputs');OLD=BASE/'h3_alpha_20261104'
RUN=BASE/'h3_relaxed_news_20261104/outputs/63d2bda2bad14bec'
OUT=OLD/'qa/h3_relaxed_news_numeric_63d2bda2bad14bec_20261104';OUT.mkdir(parents=True,exist_ok=True)
RUN_SHA='43085f4b2992939dce794dabbed1992fe70d039f664aee7c67cb5a23ad5c1f0e'
DIAG='UNIT_VALUE_KNOWN_ADJUSTMENTS';TR='TR_PROVIDER_ACCRUED_INCOME_CONVENTION'
START=time.monotonic();checks=[];details={};hashcache={}


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()


def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,np.bool_):return bool(v)
    if isinstance(v,(Path,pd.Timestamp,dt.date)):return str(v)
    return v


def save(name,value):
    p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)


def stage(name,**kw):
    save('checkpoint.json',dict(stage=name,pid=os.getpid(),elapsed_seconds=time.monotonic()-START,**kw));print(name,kw,flush=True)


def ck(name,ok,**kw):checks.append(dict(check=name,status='PASS' if bool(ok) else 'FAIL',**kw))


def eq(name,a,b,rtol=1e-8,atol=1e-11):
    a=np.asarray(a,float);b=np.asarray(b,float);same=a.shape==b.shape and np.allclose(a,b,rtol=rtol,atol=atol,equal_nan=True)
    error=np.abs(a-b) if a.shape==b.shape else np.array([])
    ck(name,same,max_absolute_difference=float(np.nanmax(error)) if np.isfinite(error).any() else 0.,rtol=rtol,atol=atol)


def records(m,prefix):
    for kind in ['inputs','files']:
        for name,z in m[kind].items():
            p=Path(z['path']);key=str(p)
            if key not in hashcache:hashcache[key]=(p.stat().st_size,sha(p))
            size,h=hashcache[key];ck(prefix+':'+kind+':'+name,size==z['bytes'] and h==z['sha256'],actual_sha256=h if h!=z['sha256'] else None)


def q(p):return "'"+str(p).replace('\\','/').replace("'","''")+"'"


def alpha_replica(label,a,r,f,scores,c,training_paths,cached):
    """Array-based, independently fitted past windows plus all future legs."""
    ck(label+':unique_alpha',not a.duplicated(['fund_id','interval_id','horizon']).any())
    ck(label+':all_three_horizons',a.groupby(['fund_id','interval_id']).horizon.apply(lambda x:set(x)=={1,2,3}).all())
    d=r.merge(f[['month','risk_free','equity_return','bond_return','factors_verified']],on='month',validate='many_to_one')
    d['ord']=pd.PeriodIndex(d.month,freq='M').asi8
    d['ok']=d.return_verified.eq(True)&d.factors_verified.eq(True)&np.isfinite(d[['fund_return','risk_free','equity_return','bond_return']]).all(axis=1)
    groups={}
    for fund,g in d.groupby('fund_id',sort=False):
        g=g.sort_values('ord')
        groups[str(fund)]=dict(month=g.month.to_numpy(),ord=g.ord.to_numpy(),ok=g.ok.to_numpy(bool),start=pd.to_datetime(g.start_valuation_date).to_numpy(),end=pd.to_datetime(g.end_valuation_date).to_numpy(),period_end=pd.to_datetime(g.return_period_end).to_numpy(),y=(g.fund_return-g.risk_free).to_numpy(float),x=np.column_stack([g.equity_return-g.risk_free,g.bond_return-g.risk_free]),lookup={int(k):j for j,k in enumerate(g.ord)})
    trainmap={}
    if training_paths:
        c.execute('CREATE OR REPLACE TEMP VIEW training AS FROM read_parquet(['+','.join(q(p) for p in training_paths)+'])')
        ck(label+':unique_training_members',c.execute('SELECT count(*) FROM (SELECT training_id,month,count(*) n FROM training GROUP BY ALL HAVING n>1)').fetchone()[0]==0)
        trainmap={tid:(int(n),h) for tid,n,h in c.execute("SELECT training_id,count(*),sha256(string_agg(month,chr(10) ORDER BY month)) FROM training GROUP BY training_id").fetchall()}
        c.register('raw_monthly_design',d[['fund_id','month','fund_return','risk_free','equity_return','bond_return']])
        ck(label+':all_training_design_from_source',c.execute('''SELECT count(*) FROM training t LEFT JOIN raw_monthly_design r USING(fund_id,month)
            WHERE r.month IS NULL OR abs(t.fund_excess-(r.fund_return-r.risk_free))>1e-12 OR abs(t.x_equity_return-(r.equity_return-r.risk_free))>1e-12 OR abs(t.x_bond_return-(r.bond_return-r.risk_free))>1e-12''').fetchone()[0]==0)
    identities=scores.set_index('interval_id').identity_origin_verified
    expected={};past={};examples=[];fitcount=0;outcomes=0;maxerr=0.;newfits=0
    oldkeys=set(zip(cached.fund_id.astype(str),cached.interval_id.astype(str)))
    for index,((fund,interval),rows) in enumerate(a.groupby(['fund_id','interval_id'],sort=False)):
        z=rows.iloc[0];date=pd.Timestamp(z.origin_date);month=date.to_period('M');last=month if month.end_time.normalize()<=date else month-1;first=last-59
        g=groups.get(str(fund));b=None;condition=np.nan;n_past=0;recent=0;state='no_verified_fund_return_series';idx=np.array([],int)
        if g is not None:
            idx=np.flatnonzero(g['ok']&(g['ord']>=first.ordinal)&(g['ord']<=last.ordinal)&(g['start']<=date.to_datetime64())&(g['end']<=date.to_datetime64())&(g['period_end']<=date.to_datetime64()))
            n_past=len(idx);recent=int((g['ord'][idx]>=last.ordinal-11).sum());state='insufficient_past_history'
            if n_past>=36 and recent>=9:
                xx=g['x'][idx];sd=xx.std(0,ddof=1);state='factor_rank_failure'
                if np.isfinite(sd).all() and (sd>0).all():
                    X=np.column_stack([np.ones(n_past),xx]);condition=float(np.linalg.cond(np.column_stack([np.ones(n_past),(xx-xx.mean(0))/sd])));state='factor_rank_or_condition_failure'
                    if np.linalg.matrix_rank(X)==3 and condition<=10000:b=linalg.lstsq(X,g['y'][idx],lapack_driver='gelsy')[0];state='eligible'
        if identities.loc[interval]!=True:b=None;state='unverified_dated_identity_at_origin'
        ck(label+':history_members_and_bounds:'+interval,n_past==z.n_past and recent==z.n_past_recent and str(first)==z.training_first_allowed and str(last)==z.training_last_allowed and (rows.n_past==n_past).all() and (rows.n_past_recent==recent).all())
        past[(str(fund),interval)]=float(b[0]) if b is not None else np.nan
        eq(label+':past_beta_or_NA:'+interval,[z.past_alpha,z.beta_equity_return,z.beta_bond_return],b if b is not None else [np.nan]*3)
        if b is not None:
            fitcount+=1;newfits+=int((str(fund),interval) not in oldkeys)
            if trainmap:
                expected_hash=hashlib.sha256('\n'.join(g['month'][idx]).encode()).hexdigest()
                ck(label+':exact_training_months:'+interval,trainmap.get(z.training_id)==(n_past,expected_hash))
            eq(label+':condition:'+interval,[z.training_condition_number],[condition])
        for row in rows.itertuples(index=False):
            value=np.nan;status=state
            if b is not None:
                locations=[g['lookup'].get(m) for m in range(month.ordinal+1,month.ordinal+int(row.horizon)+1)]
                full=all(j is not None and g['ok'][j] for j in locations)
                anchor=g['start'][locations[0]] if locations[0] is not None else np.datetime64('NaT')
                anchorok=not np.isnat(anchor) and anchor>=date.to_datetime64()
                if not full:status='incomplete_future_horizon'
                elif not anchorok:status='future_anchor_precedes_origin_or_missing'
                else:
                    abnormal=g['y'][locations]-g['x'][locations]@b[1:];value=float(abnormal.mean());status='eligible';outcomes+=1;maxerr=max(maxerr,abs(value-row.alpha))
                    if len(examples)<6 and int(row.horizon)==3 and (str(fund),interval) not in oldkeys:
                        examples.append(dict(fund_id=str(fund),interval_id=interval,origin_date=str(date.date()),training_months=g['month'][idx].tolist(),betas=b.tolist(),future_months=[str(pd.Period(ordinal=month.ordinal+h,freq='M')) for h in range(1,4)],future_abnormal=abnormal.tolist(),alpha_independent=value,alpha_saved=row.alpha))
            expected[(str(fund),interval,int(row.horizon))]=value
            eq(label+':future_alpha:'+interval+':h'+str(row.horizon),[row.alpha],[value],rtol=1e-9,atol=1e-11)
            ck(label+':future_status:'+interval+':h'+str(row.horizon),row.status==status)
        if index and index%5000==0:stage('ALPHA_REPLICATION_PROGRESS',branch=label,origins_done=index)
    ck(label+':every_numeric_alpha_reproduced',outcomes==int(a.alpha.notna().sum()))
    return expected,past,dict(origins=a.interval_id.nunique(),past_fits=fitcount,new_past_fits=newfits,numeric_alpha=outcomes,max_absolute_difference=maxerr,new_origin_examples=examples)


def main():
    stage('PIN_NEW_RUN_OLD_RELEASES_AND_REISSUED_PANEL')
    m=json.loads((RUN/'manifest.json').read_text());ck('exact_ready_run',sha(RUN/'manifest.json')==RUN_SHA)
    ck('exact_V3_spec',m['inputs']['spec']['sha256']==RELAXED_SPEC_SHA256)
    panel_path=Path(m['inputs']['panel_manifest']['path']).parent;pm=json.loads((panel_path/'manifest.json').read_text())
    vm=json.loads(Path(m['inputs']['v3_manifest']['path']).read_text());tm=json.loads(Path(m['inputs']['TR4_manifest']['path']).read_text())
    nm=json.loads((OLD/'panels/nts_ee5a8bf9b450137a/manifest.json').read_text())
    manifests=[('new_run',m),('new_panel',pm),('old_v3',vm),('old_TR4',tm),('old_NTS',nm)]
    for label,manifest in manifests:records(manifest,'before:'+label)
    oldpanel=BASE/'h3_relaxed_news_20261104/panels/5adc30709c6f4cc2';opm=json.loads((oldpanel/'manifest.json').read_text())
    panel_qa_path=OLD/('qa/h3_relaxed_news_panel_'+panel_path.name+'_20261104/independent_panel_qa.json')
    panel_qa=json.loads(panel_qa_path.read_text())
    ck('actual_production_panel_independent_source_semantics_PASS',panel_qa['status']=='PASS' and panel_qa['checks_fail']==0 and panel_qa['panel_manifest_sha256']==m['inputs']['panel_manifest']['sha256'] and Path(panel_qa['panel'])==panel_path)
    identity=[]
    for key,z in pm['files'].items():
        if key=='input_manifest.json':continue
        ok=z['sha256']==opm['files'][key]['sha256'];identity.append(dict(file=key,sha256=z['sha256'],identical_to_audited_first_panel=ok))
    details['panel_reissue_data_identity']=identity
    details['panel_acceptance_rule']='Actual production panel must pass its own independent source/semantic audit. Cross-version Parquet byte equality is not a V3 requirement. Initial three FAIL from that extra criterion remain preserved in the original audit artifact; final acceptance is in a separate receipt.'
    details['provenance_acceptance']='Pending separate executor reissue evidence; first FAIL artifact is retained. Numeric results do not alone close provenance.'
    scores=pd.read_parquet(m['inputs']['scores']['path']);ck('full28976_origin_score_source',len(scores)==28976)
    ck('new_scores_identical_to_independently_checked_panel',m['inputs']['scores']['sha256']=='b91618a88d33e3ed0a705ac5e30c7bd34542a4266a8097b99791c7d09a8b2f12')
    factors=pd.read_parquet(m['inputs']['factors']['path']);returns=pd.read_parquet(m['inputs']['diagnostic_returns']['path']);returns['return_verified']=returns.diagnostic_eligible.eq(True)
    diag=pd.read_parquet(RUN/'alpha/alpha_all_reused_and_extended.parquet');tr=pd.read_parquet(m['inputs']['TR4_cached_alpha']['path']);cache=pd.read_parquet(m['inputs']['diagnostic_cached_alpha']['path'])
    required=scores.loc[scores.universe_eligible.eq(True),['fund_id','interval_id']]
    ck('alpha_all27156_universe_origins_before_news',len(required)==27156 and set(map(tuple,diag[['fund_id','interval_id']].drop_duplicates().to_numpy()))==set(map(tuple,required.to_numpy())) and len(diag)==27156*3)
    common=diag.merge(cache,on=['fund_id','interval_id','horizon'],suffixes=('_new','_old'),validate='one_to_one')
    ck('all7850_cached_origins_preserved',len(common)==len(cache)==7850*3)
    for col in ['alpha','past_alpha','beta_equity_return','beta_bond_return','n_past','n_past_recent']:
        eq('cached_exact:'+col,common[col+'_new'],common[col+'_old'],rtol=0,atol=0)
    ck('cached_status_preserved',common.status_new.equals(common.status_old))
    ck('all19306_additional_origins',diag.interval_id.nunique()-cache.interval_id.nunique()==19306)
    c=duckdb.connect(config={'threads':2,'memory_limit':'2GB','temp_directory':str(OUT/'spill')})
    old_training=list((Path(m['inputs']['diagnostic_cached_alpha']['path']).parent).glob('chunk_*.training.parquet'))
    new_training=list((RUN/'alpha').glob('new_training_chunk_*.parquet'))
    stage('ALL_EXPANDED_DIAGNOSTIC_PAST_FITS_AND_FORWARD_LEGS')
    de,dp,ds=alpha_replica(DIAG,diag,returns,factors,scores,c,old_training+new_training,cache)
    tr_returns=pd.read_parquet(Path(m['inputs']['TR4_cached_alpha']['path']).parent/'approved_monthly_provider_returns.parquet')
    te,tp,ts=alpha_replica(TR,tr,tr_returns,factors,scores,c,[Path(m['inputs']['TR4_cached_alpha']['path']).parent/'lookback60_min36_training.parquet'],tr)
    details['alpha_reconstruction']={DIAG:ds,TR:ts}
    models=json.loads((RUN/'all_model_details.json').read_text());grid=planned_cells([DIAG,TR])
    ck('exact48_fixed_cells',len(models)==48 and {(x['branch'],x['group'],x['horizon'],x['variant']) for x in models}==set(grid))
    flow=pd.read_csv(RUN/'sample_flow_old_new.csv');replicas=[];samples=[];singular=0
    stage('ALL48_EXACT_SAMPLES_AND36_INDEPENDENT_LSDV_CR1_FITS')
    for model in models:
        mid=model['model_id'];isdiag=model['branch']==DIAG;alpha=diag if isdiag else tr;expected_alpha=de if isdiag else te;expected_past=dp if isdiag else tp
        pool=alpha.merge(scores.drop(columns=['origin_month'],errors='ignore'),on=['fund_id','interval_id'],validate='many_to_one')
        pool['NTS_Q_current']=pool.NTS_Q_current_relaxed;pool['NTS_Q_lagnews']=pool.NTS_Q_lagnews_relaxed
        pool=pool.loc[pool.horizon.eq(model['horizon'])].copy()
        expected,removed=expected_model_sample(pool,outcome='alpha',current='NTS_Q_current',lag='NTS_Q_lagnews',distributed=model['distributed'],universe_column='universe_eligible',group=model['group'])
        path=RUN/'regression'/mid;used=pd.read_parquet(path/'used_sample.parquet').sort_values(['fund_id','end_date','interval_id']).reset_index(drop=True)
        ck('exact_model_rows:'+mid,used[['fund_id','interval_id']].equals(expected[['fund_id','interval_id']]) and not used.duplicated(['fund_id','interval_id']).any())
        ck('N_G_hash_singletons:'+mid,len(used)==model['nobs'] and used.fund_id.nunique()==model['funds'] and ordered_sample_sha256(used)==model['sample_hash'] and len(removed)==model['singleton_rows_removed'])
        eq('source_score_and_outcome_rows:'+mid,used[['alpha','past_alpha','NTS_Q_current','NTS_Q_lagnews']],expected[['alpha','past_alpha','NTS_Q_current','NTS_Q_lagnews']],rtol=0,atol=0)
        eq('used_outcomes_from_source_legs:'+mid,used.alpha,[expected_alpha[(str(x.fund_id),x.interval_id,int(x.horizon))] for x in used.itertuples(index=False)],rtol=1e-9,atol=1e-11)
        eq('used_past_alpha_from_source:'+mid,used.past_alpha,[expected_past[(str(x.fund_id),x.interval_id)] for x in used.itertuples(index=False)],rtol=1e-9,atol=1e-11)
        n=len(used);G=used.fund_id.nunique();ck('FE_clustering_and_fragility:'+mid,model['fund_FE'] and not model['time_FE'] and model['clusters']==['fund_id'] and model['reference_df']==G-1 and model['weak_cluster_inference']==(G<30) and model['significance_stars_suppressed']==(G<30) and not model['multiple_testing_adjustments'])
        ck('own_sample_lag_requirement_and_zero_counts:'+mid,model['own_sample_does_not_require_lag']==(not model['distributed']) and model['current_zero_rows']==int(used.NTS_Q_current.eq(0).sum()) and model['current_known_lag_unknown_rows']==int((used.current_eligible&~used.lagnews_eligible).sum()))
        prior_variant='CURRENT_PLUS_LAGNEWS' if model['distributed'] else 'CURRENT_COMMON';oldmid=f"{model['branch']}__PRIMARY__{model['group']}__h{model['horizon']}__{prior_variant}"
        priorpath=(Path(m['inputs']['v3_manifest']['path']).parent/'regression'/DIAG/'PRIMARY'/oldmid if isdiag else Path(m['inputs']['TR4_manifest']['path']).parent/'regression/PRIMARY'/oldmid)
        oldused=pd.read_parquet(priorpath/'used_sample.parquet');oldids=set(oldused.row_id.astype(str));ids=set(used.row_id.astype(str))
        ck('old_new_exact_overlap:'+mid,model['overlap_old_primary']==len(ids&oldids) and model['added_vs_old']==len(ids-oldids) and model['removed_vs_old']==len(oldids-ids))
        base=pool.universe_eligible.eq(True)
        if model['group']!='pooled':base&=pool.fund_focus.eq(model['group'])
        alphaknown=np.isfinite(pool[['alpha','past_alpha']]).all(axis=1)
        nativeflow=flow.loc[flow.model_id.eq(mid)].iloc[0]
        ck('dataflow:'+mid,int(nativeflow.alpha_defined)==int((base&alphaknown).sum()) and int(nativeflow.current_known)==int((base&alphaknown&pool.current_eligible).sum()) and int(nativeflow.both_known)==int((base&alphaknown&pool.current_eligible&pool.lagnews_eligible).sum()) and int(nativeflow.before_singletons)==n+len(removed) and int(nativeflow.final_N)==n)
        samples.append(dict(model_id=mid,n=n,funds=int(G),status=model['status'],sample_hash=model['sample_hash'],old_overlap=len(ids&oldids),new_rows=len(ids-oldids),removed_old_rows=len(oldids-ids),current_zero_rows=model['current_zero_rows'],current_known_lag_unknown_rows=model['current_known_lag_unknown_rows']))
        terms=['NTS_Q_current']+(['NTS_Q_lagnews'] if model['distributed'] else [])+['past_alpha']
        fit=independent_full_dummy_cr1(used,outcome='alpha',terms=terms)
        ck('independent_estimability:'+mid,fit['status']==model['status'])
        if model['status']!='ESTIMATED':
            ck('honest_NA_no_inference:'+mid,(n<30 or G<2) and model['reason']=='fewer_than_30_rows_or_two_funds_after_singletons' and not model.get('coefficients') and not model.get('target_tests'));continue
        names=fit['terms'];native=pd.DataFrame(model['coefficients']).set_index('term').loc[names];cov=pd.read_csv(path/'full_covariance.csv',index_col=0,encoding='utf-8-sig').loc[names,names]
        eq('LSDV_beta:'+mid,fit['beta'],native.beta,rtol=1e-7,atol=1e-10);eq('LSDV_CR1_full_covariance:'+mid,fit['covariance'],cov,rtol=1e-6,atol=1e-11)
        ck('rank_and_df:'+mid,fit['rank']==model['K_full'] and fit['residual_df']==model['residual_df'] and fit['absorbed_columns']==model['absorbed_columns'])
        eq('CR1_correction:'+mid,[fit['CR1_correction']],[model['CR1_correction']]);eq('within_R2:'+mid,[fit['within_R2']],[model['within_R2']])
        for field in ['SE','t','p','CI_low','CI_high']:eq('tGminus1_'+field+':'+mid,fit[field],native[field],rtol=1e-6,atol=1e-9)
        tests={z['test']:z for z in model['target_tests']}
        for j,term in enumerate([x for x in terms if x!='past_alpha'],start=1):
            z=tests[term+'=0'];eq('target:'+mid+':'+term,[z[x] for x in ['beta','SE','t','p','CI_low','CI_high']],[fit['beta'][j]]+[fit[x][j] for x in ['SE','t','p','CI_low','CI_high']],rtol=1e-6,atol=1e-9)
        if model['distributed']:
            total=fit['sum_test'];z=tests['current+lagnews=0'];eq('sum_test:'+mid,[z[x] for x in ['beta','SE','t','p','CI_low','CI_high']],[total[x] for x in ['beta','SE','t','p','CI_low','CI_high']],rtol=1e-6,atol=1e-9)
            joint=fit['joint_test'];z=tests['joint_current_lagnews=0'];ck('joint_status:'+mid,z['status']==joint['status'])
            if joint['status']=='defined':eq('joint_F_p:'+mid,[z['F'],z['p']],[joint['F'],joint['p']],rtol=1e-6,atol=1e-9)
            else:singular+=1;ck('singular_joint_honest_NA:'+mid,not np.isfinite([z['F'],z['p']]).any())
        replicas.append(dict(model_id=mid,n=n,funds=int(G),rank=fit['rank'],residual_df=fit['residual_df'],reference_df=G-1,sample_hash=model['sample_hash'],max_beta_difference=float(np.max(abs(fit['beta']-native.beta.to_numpy()))),max_cov_difference=float(np.max(abs(fit['covariance']-cov.to_numpy()))),independent_full_dummy_fit=True))
        stage('NUMERICAL_MODEL_REPLICATED',completed=len(replicas),model_id=mid)
    ck('all36_estimated_models_replicated',len(replicas)==36);ck('all12_NA_cells_accounted',len(models)-len(replicas)==12);ck('three_two_fund_distributed_joint_tests_NA',singular==3)
    details.update(regression_replicas=replicas,all48_samples=samples,singular_joint_tests_NA=singular)
    stage('REHASH_NEW_AND_ALL_OLD_SEALED_OUTPUTS');hashcache.clear()
    for label,manifest in manifests:records(manifest,'after:'+label)
    ck('run_manifest_unchanged',sha(RUN/'manifest.json')==RUN_SHA)
    details['run_manifest_sha256']=RUN_SHA;details['panel_manifest_sha256']=m['inputs']['panel_manifest']['sha256'];details['spec_sha256']=RELAXED_SPEC_SHA256


try:
    main()
    report=dict(status='PASS' if all(x['status']=='PASS' for x in checks) else 'FAIL',audit_type='EXTERNAL_INDEPENDENT_EXPANDED_ALPHA_AND48_CELLS',overall_acceptance='PENDING_SEPARATE_PANEL_PROVENANCE_RESOLUTION',run=str(RUN),audit_code_sha256=sha(__file__),independent_helper_sha256=sha(Path(__file__).with_name('qa_h3_relaxed_news_independent_20261104.py')),production_modules_imported=False,executor_checks_counted_as_external=False,checks_pass=sum(x['status']=='PASS' for x in checks),checks_fail=sum(x['status']=='FAIL' for x in checks),checks=checks,details=details,completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),runtime_seconds=time.monotonic()-START,limitations=['Outcome/source-gate formulas and immutable inputs are inherited from separately independently audited v3/TR4; every new/cached past fit and future outcome is numerically rechecked here.','Panel semantics were independently checked on first-panel bytes; all seven data files must be byte-identical in the reissue. Code provenance resolution remains separate.','TR4 has four funds of one manager; two-fund covariance is singular for a two-target joint test, few clusters remain fragile/no stars.','Current-only/distributed use their own available samples; differences do not identify a causal effect of adding lag.'])
    save('independent_numeric_qa.json',report);save('summary.json',{k:v for k,v in report.items() if k!='checks'})
    stage('COMPLETED',status=report['status'],checks_pass=report['checks_pass'],checks_fail=report['checks_fail'])
    if report['checks_fail']:print('FAILURES',json.dumps(clean([x for x in checks if x['status']=='FAIL']),ensure_ascii=False));raise SystemExit(1)
except Exception:
    save('execution_failure.json',dict(traceback=traceback.format_exc(),checks=checks,details=details));stage('EXECUTION_FAILED');raise
