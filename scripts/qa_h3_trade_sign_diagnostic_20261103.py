"""Independent QA of the fixed post-result shared-sign diagnostic on nine frozen samples.

Research data are read on research-host only. No PanelOLS or regression runner is
imported. The only research write is the separately named QA JSON report.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def row_hash(frame):
    return hashlib.sha256('\n'.join(frame.row_id.astype(str)).encode()).hexdigest()


def truth(values):
    return values.astype(str).str.lower().str.strip().isin(['true', '1', 'yes'])


def finite(frame, columns):
    return np.isfinite(frame[columns].apply(pd.to_numeric, errors='coerce').to_numpy(float)).all(axis=1)


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (np.integer, np.bool_)): return value.item()
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value, (Path, pd.Timestamp)): return str(value)
    return value


def within_ols(frame, terms, outcome='dar_q_cash_price_pp'):
    """Fund-demeaned OLS and independent two-way CGM sandwich."""
    groups = frame.fund_id.astype(str)
    x = frame[terms].astype(float)
    y = frame[outcome].astype(float)
    xw = (x-x.groupby(groups, sort=False).transform('mean')).to_numpy(float)
    yw = (y-y.groupby(groups, sort=False).transform('mean')).to_numpy(float)
    scale = np.linalg.norm(xw, axis=0)
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Retained term lacks finite within variation')
    z = xw/scale
    beta_z, _, rank, singular = np.linalg.lstsq(z, yw, rcond=None)
    if rank != len(terms): raise ValueError('Within design is rank deficient')
    residual = yw-z@beta_z
    bread = np.linalg.inv(z.T@z)
    scores = z*residual[:, None]

    def meat(keys):
        codes, labels = pd.factorize(keys, sort=False)
        n, g = len(codes), len(labels)
        if g < 2: raise ValueError('Fewer than two clusters')
        sums = np.zeros((g, scores.shape[1]))
        np.add.at(sums, codes, scores)
        return sums.T@sums * (g/(g-1)) * ((n-1)/n), g

    fund_meat, funds = meat(groups)
    companies = frame.company_id.astype(str)
    company_meat, company_count = meat(companies)
    joint_meat, joint_count = meat(pd.MultiIndex.from_arrays([groups, companies]))
    n, k = z.shape
    df = n-funds-k
    if df <= 0: raise ValueError('Nonpositive residual degrees of freedom')
    covariance = bread@(fund_meat+company_meat-joint_meat)@bread * n/df
    covariance = covariance/scale[:, None]/scale[None, :]
    ssr, tss = float(residual@residual), float(yw@yw)
    return dict(beta=beta_z/scale, covariance=(covariance+covariance.T)/2,
        nobs=n, funds=funds, companies=company_count, joint_clusters=joint_count,
        df_resid=df, rank=rank, condition_number=float(singular.max()/singular.min()),
        within_SSR=ssr, within_TSS=tss, within_R2=1-ssr/tss if tss else np.nan)


def independently_retained_terms(sample, required, controls):
    """Audit the declared-order within-rank rule without importing an estimator."""
    names = list(required) + list(controls)
    values = sample[names].astype(float)
    within = values-values.groupby(sample.fund_id.astype(str), sort=False).transform('mean')
    tolerance = 16*np.finfo(float).eps*max(len(sample), len(names))
    kept, basis = [], []
    for term in names:
        raw = within[term].to_numpy(float)
        norm = np.linalg.norm(raw)
        if sample.groupby('fund_id', observed=True)[term].nunique().max() <= 1 or norm == 0:
            if term in required: raise ValueError('Required term is fund-absorbed: '+term)
            continue
        vector = raw/norm
        residual = vector if not basis else vector-np.column_stack(basis)@np.linalg.lstsq(np.column_stack(basis), vector, rcond=None)[0]
        if np.linalg.norm(residual) <= tolerance:
            if term in required: raise ValueError('Required term is collinear: '+term)
            continue
        kept.append(term)
        basis.append(vector)
    return kept


def audit(args):
    if os.name!='nt':raise RuntimeError('Actual-data QA must run on research-host')
    from scipy import stats
    planpath=Path(args.plan);plan=json.loads(planpath.read_text(encoding='utf-8'));root=Path(args.root or plan['output_root'])
    output=Path(args.output or root/'qa_independent.json')
    if output.exists():raise FileExistsError('Refusing to overwrite an existing QA report')
    if 'databaseresources'in[p.lower()for p in output.parts]:raise ValueError('Protected output')
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'));coef=pd.read_csv(root/'coefficients.csv');comparison=pd.read_csv(root/'comparison.csv')
    baseline=json.loads(Path(plan['baseline_summary']).read_text(encoding='utf-8'));oldcoef=pd.read_csv(plan['baseline_coefficients']);oldrecords={r['model']:r for r in baseline['models']}
    checks,warnings,details=[],[],{}
    def check(name,passed,**evidence):checks.append(dict(name=name,status='PASS'if bool(passed)else'FAIL',**clean(evidence)))
    def close(a,b,atol=1e-7):
        if pd.isna(a) or pd.isna(b):return bool(pd.isna(a) and pd.isna(b))
        return bool(np.isclose(float(a),float(b),rtol=args.rtol,atol=atol))
    def same(a,b,atol=1e-10):return bool(np.allclose(np.asarray(a,float),np.asarray(b,float),rtol=1e-12,atol=atol,equal_nan=True))
    check('frozen_plan_hash',sha(planpath)==summary['plan_sha256'])
    check('same_frozen_specification',summary.get('specification')==plan)
    check('exactly_nine_fixed_diagnostics',len(plan['baseline_models'])==len(plan['planned_models'])==9 and plan['planned_models']==[name+'__WITH_TRADE_SIGN'for name in plan['baseline_models']])
    check('native_inference_only',plan['multiple_testing_adjustments']is False and set(coef.columns)=={'model','outcome','term','beta','se_native','t_native','p_native','ci95_native_low','ci95_native_high'})
    check('no_baseline_refit',summary['baseline_refit']is False)
    check('no_independent_crossmodel_delta_inference',not any('delta'in c.lower()and any(v in c.lower()for v in ['_se','p_native','pvalue','p_value'])for c in comparison.columns))
    before={}
    for key,expected in plan['expected_source_sha256'].items():
        before[key]=sha(plan[key]);check('frozen_source:'+key,before[key]==expected,sha256=before[key])
    records={r['model']:r for r in summary['models']};required=plan['required_design'];controls=plan['baseline_controls']+plan['macro_controls']
    check('all_nine_output_models',set(records)==set(plan['planned_models'])and set(comparison.model)==set(records)and not comparison.model.duplicated().any())
    check('unique_new_coefficient_keys',not coef.duplicated(['model','term']).any())
    native_fields=['beta','se_native','t_native','p_native','ci95_native_low','ci95_native_high'];sample_before={};actual_numeric=[];inference={}
    def native_arithmetic(label,table,df):
        finite=np.isfinite(table[native_fields].to_numpy(float)).all(axis=1)&table.se_native.gt(0).to_numpy();valid=table.loc[finite]
        t=valid.beta/valid.se_native;critical=stats.t.ppf(.975,df)
        check('defined_native_t_p_CI:'+label,same(t,valid.t_native,1e-7)and np.allclose(2*stats.t.sf(abs(t),df),valid.p_native,rtol=args.rtol,atol=1e-10)and np.allclose(valid.beta-critical*valid.se_native,valid.ci95_native_low,rtol=args.rtol,atol=1e-7)and np.allclose(valid.beta+critical*valid.se_native,valid.ci95_native_high,rtol=args.rtol,atol=1e-7))
        return finite
    def distributed_arithmetic(label,beta,cov,df,test,prefix):
        beta=np.asarray(beta,float);cov=np.asarray(cov,float);joint=test['joint_current_lag_zero'];total=test['sum_current_lag']
        check(prefix+'target_names_and_no_repair:'+label,test['target_terms']==required[:2]and test['covariance_repair']is False)
        if np.isfinite(cov).all()and np.isfinite(beta).all():
            eigenvalues=np.linalg.eigvalsh((cov+cov.T)/2)
            check(prefix+'target_eigenvalues:'+label,same(joint['target_covariance_eigenvalues'],eigenvalues))
            if eigenvalues.min()>0:
                statistic=float(beta@np.linalg.solve(cov,beta));p=stats.chi2.sf(statistic,2)
                check(prefix+'joint_chi_square_two:'+label,joint['status']=='ESTIMATED'and joint['df']==2 and 'chi2'in joint['distribution'].lower()and close(joint['statistic'],statistic)and close(joint['p_value'],p,1e-10))
            else:check(prefix+'undefined_joint_not_p_one:'+label,joint['status']!='ESTIMATED'and joint['p_value']is None)
            value=float(beta.sum());variance=float(cov.sum());check(prefix+'sum_with_crosscovariance:'+label,close(total['beta'],value)and close(total['variance'],variance)and close(test['cov_current_lag'],cov[0,1]))
            if variance>0 and df>0:
                se=float(np.sqrt(variance));critical=stats.t.ppf(.975,df);p=2*stats.t.sf(abs(value/se),df)
                check(prefix+'sum_native_t_CI:'+label,total['status']=='ESTIMATED'and total['native_df_resid']==df and close(total['se_native'],se)and close(total['p_native'],p,1e-10)and close(total['ci95_native_low'],value-critical*se)and close(total['ci95_native_high'],value+critical*se))
            else:check(prefix+'undefined_sum_not_p_one:'+label,total['status']!='ESTIMATED'and total['p_native']is None)
        else:check(prefix+'nonfinite_covariance_undefined:'+label,joint['status']!='ESTIMATED'and total['status']!='ESTIMATED')
    for oldname,name in zip(plan['baseline_models'],plan['planned_models']):
        frozen=plan['samples_frozen'][oldname];path=Path(frozen['parquet']);sample_before[oldname]=sha(path);sample=pd.read_parquet(path)
        check('frozen_exact_sample_sha:'+name,sample_before[oldname]==frozen['sha256'])
        check('frozen_exact_row_ids_N_horizon:'+name,row_hash(sample)==frozen['row_hash']and len(sample)==frozen['rows']and not sample.row_id.duplicated().any()and sample.horizon.eq(frozen['horizon']).all())
        metadata=summary['samples'][name];oldmeta=baseline['samples'][oldname]
        check('same_old_new_sample_metadata:'+name,metadata==frozen and oldmeta['rows']==frozen['rows']and oldmeta['row_hash']==frozen['row_hash']and oldmeta['horizon']==frozen['horizon'])
        s=pd.to_numeric(sample.trade_sign,errors='coerce');check('nonzero_known_trade_sign_square_one:'+name,np.isfinite(s).all()and s.isin([-1.,1.]).all()and s.pow(2).eq(1).all()and sample.delta_quantity_units.ne(0).all()and same(s,np.sign(sample.delta_quantity_units),0))
        check('shared_sign_algebra_exact_source:'+name,np.allclose(s*sample.align_current,sample.net_sentiment,rtol=0,atol=0)and np.allclose(s*sample.align_lag1,sample.net_sentiment_lag1,rtol=0,atol=0)and np.allclose(s*sample[plan['dependent']],100*(sample.future_price_return-sample.cash_return),rtol=1e-12,atol=1e-10))
        cache=summary['baseline_cache'][oldname];oldrecord=oldrecords[oldname];original=oldcoef.loc[oldcoef.model.eq(oldname)].set_index('term');cached=pd.DataFrame(cache['coefficients']).set_index('term').reindex(original.index)
        check('baseline_cache_not_refit_metadata:'+name,cache['baseline_refit']is False and cache['record']==oldrecord and cache['distributed_tests']==baseline['distributed_tests'][oldname])
        check('baseline_cached_coefficients_literal:'+name,set(cached.index)==set(original.index)and same(cached[native_fields],original[native_fields]))
        check('baseline_sample_counts_and_design:'+name,oldrecord['nobs']==len(sample)and oldrecord['funds']==sample.fund_id.nunique()and oldrecord['companies']==sample.company_id.nunique()and required[2]not in original.index and set(required[:2]).issubset(original.index))
        native_arithmetic('BASELINE:'+oldname,original,oldrecord['native_df_resid'])
        oldtest=cache['distributed_tests'];distributed_arithmetic('BASELINE:'+oldname,original.loc[required[:2],'beta'],oldtest['target_covariance'],oldrecord['native_df_resid'],oldtest,'mechanical_')
        values=sample[required].astype(float);within=values-values.groupby(sample.fund_id.astype(str),sort=False).transform('mean');norms=np.linalg.norm(within.to_numpy(float),axis=0)
        absorbed=sample.groupby('fund_id',observed=True).trade_sign.nunique().max()<=1
        if absorbed:expected_status='already_absorbed_equivalent';rank=2
        else:
            rank=int(np.linalg.matrix_rank(within.to_numpy(float)/norms,tol=16*np.finfo(float).eps*max(len(sample),len(required))))
            expected_status='estimable'if rank==3 else'not_estimable_sign_collinear_with_targets'
        record=records[name];new=coef.loc[coef.model.eq(name)].set_index('term');display=comparison.loc[comparison.model.eq(name)].iloc[0]
        check('independent_required_design_absorption_rank:'+name,record['diagnostic_status']==expected_status,independent_rank=rank,sign_absorbed=bool(absorbed))
        check('same_model_counts_frozen_rows:'+name,record['nobs']==len(sample)and record['funds']==sample.fund_id.nunique()and record['companies']==sample.company_id.nunique()and record['same_ordered_rows_as_baseline']is True and record['baseline_refit']is False and record['baseline_sample_sha256']==frozen['sha256'])
        if expected_status=='not_estimable_sign_collinear_with_targets':
            check('required_sign_not_silently_removed:'+name,record['status']=='NOT_ESTIMABLE'and len(new)==0);warnings.append(dict(model=name,kind='REQUIRED_SIGN_DESIGN_UNIDENTIFIED'));continue
        if record['status']=='NOT_ESTIMABLE':warnings.append(dict(model=name,kind='NOT_ESTIMABLE_DESPITE_REQUIRED_RANK',reason=record.get('reason')));continue
        terms=record['retained_terms'];df=record['native_df_resid'];options=record['covariance_options'];current_targets=required[:2]if absorbed else required
        check('same_declared_sixteen_controls_only_sign_added:'+name,set(terms).issubset(required+controls)and (set(required).issubset(terms)if not absorbed else set(required[:2]).issubset(terms)))
        check('required_regressors_before_ordinary_controls:'+name,terms[:len(current_targets)]==current_targets)
        if not absorbed:
            check('all_declared_controls_selected_by_fixed_within_rank_rule:'+name,terms==independently_retained_terms(sample,required,controls))
        expected_coef_terms=set(['const']+terms)|({required[2]}if absorbed else set());check('coefficient_terms_and_df:'+name,set(new.index)==expected_coef_terms and df==len(sample)-sample.fund_id.nunique()-len(terms))
        check('fund_FE_and_original_two_way_covariance:'+name,record['included_effects']==['Entity']and record['time_effects']is False and record['other_effects']is False and record['weights']is None and all(options.get(k)==v for k,v in dict(cov_type='clustered',clusters=['fund_id','company_id'],debiased=True,auto_df=False,count_effects=True,group_debias=True,cluster_time=False).items()))
        if absorbed:
            check('absorbed_sign_not_zero_filled:'+name,record['status']=='ALREADY_ABSORBED_EQUIVALENT'and record['trade_sign_inference_status']=='ABSORBED_FUND_FE'and new.loc[required[2],native_fields].isna().all()and same(new.reindex(original.index)[native_fields],original[native_fields]))
        finite=native_arithmetic(name,new,df)
        if not absorbed and (~finite).any():warnings.append(dict(model=name,kind='UNDEFINED_COEFFICIENT_INFERENCE',terms=new.index[~finite].tolist()))
        if min(record['funds'],record['companies'])<30:warnings.append(dict(model=name,kind='LOW_CLUSTER_COUNT',funds=record['funds'],companies=record['companies']))
        if record.get('overall_robust_F_inference',{}).get('status')=='UNDEFINED_COVARIANCE':warnings.append(dict(model=name,kind='UNDEFINED_OVERALL_ROBUST_F',native=record['overall_robust_F_inference']))
        inference[name]={term:'ABSORBED'if absorbed and term==required[2]else'FINITE'if term in new.index[finite]else'UNDEFINED'for term in required}
        for term,prefix in [(required[0],'current'),(required[1],'lag1')]:
            check('baseline_comparison_values:'+name+':'+term,all(close(display['baseline_'+prefix+'_'+field],original.loc[term,field],1e-10 if field=='p_native' else 1e-7)for field in ['beta','se_native','ci95_native_low','ci95_native_high','p_native']))
            check('controlled_comparison_values:'+name+':'+term,all(close(display['controlled_'+prefix+'_'+field],new.loc[term,field],1e-10 if field=='p_native' else 1e-7)for field in ['beta','se_native','ci95_native_low','ci95_native_high','p_native']))
            check('descriptive_coefficient_change_only:'+name+':'+term,close(display['descriptive_delta_'+prefix+'_beta'],new.loc[term,'beta']-original.loc[term,'beta']))
        result=within_ols(sample,terms,outcome=plan['dependent']);begin=len(checks)
        check('independent_df:'+name,result['df_resid']==df);check('independent_within_R2:'+name,close(result['within_R2'],record['rsquared_within'],1e-10))
        for term in current_targets:
            j=terms.index(term);beta=float(result['beta'][j]);variance=float(result['covariance'][j,j]);se=float(np.sqrt(variance))if variance>0 else np.nan;found=new.loc[term]
            check('independent_beta:'+name+':'+term,close(beta,found.beta,1e-8),independent=beta,native=found.beta)
            if np.isfinite(se):
                t=beta/se;p=2*stats.t.sf(abs(t),df);critical=stats.t.ppf(.975,df)
                check('independent_CGM_SE:'+name+':'+term,close(se,found.se_native),independent=se,native=found.se_native)
                check('independent_native_t_p_CI:'+name+':'+term,close(t,found.t_native)and close(p,found.p_native,1e-10)and close(beta-critical*se,found.ci95_native_low)and close(beta+critical*se,found.ci95_native_high))
            else:check('undefined_required_variance_preserved:'+name+':'+term,not np.isfinite(found.se_native)or found.se_native<=0)
        test=summary['distributed_tests'][name];indices=[terms.index(term)for term in required[:2]];betas=result['beta'][indices];block=result['covariance'][np.ix_(indices,indices)]
        check('independent_alignment_covariance_after_sign_control:'+name,np.allclose(block,np.asarray(test['target_covariance'],float),rtol=args.rtol,atol=1e-8,equal_nan=True));distributed_arithmetic(name,betas,block,df,test,'independent_')
        check('joint_and_sum_comparison_values:'+name,display.joint_status==test['joint_current_lag_zero']['status'] and close(display.joint_p,test['joint_current_lag_zero']['p_value'],1e-10) and display.sum_status==test['sum_current_lag']['status'] and all(close(display[c],test['sum_current_lag'][key],1e-10 if key=='p_native' else 1e-7) for c,key in [('sum_beta','beta'),('sum_se','se_native'),('sum_ci95_low','ci95_native_low'),('sum_ci95_high','ci95_native_high'),('sum_p','p_native')]))
        if not absorbed:
            indices=[terms.index(term)for term in required];allcov=result['covariance'][np.ix_(indices,indices)]
            check('independent_full_three_required_covariance:'+name,test['all_required_target_terms']==required and np.allclose(allcov,np.asarray(test['all_required_target_covariance'],float),rtol=args.rtol,atol=1e-8,equal_nan=True))
            check('raw_sign_coefficient_display:'+name,all(close(display['controlled_trade_sign_'+field],new.loc[required[2],field],1e-10 if field=='p_native' else 1e-7)for field in ['beta','se_native','ci95_native_low','ci95_native_high','p_native']))
        details[name]={key:value for key,value in result.items()if key not in['beta','covariance']};details[name].update(coefficients=dict(zip(terms,result['beta'])),required_design_rank=rank,sign_absorbed=bool(absorbed),numerical_replication_status='PASS'if all(c['status']=='PASS'for c in checks[begin:])else'FAIL');actual_numeric.append(name)
    for key,value in before.items():check('source_preserved_after_QA:'+key,sha(plan[key])==value)
    for oldname,value in sample_before.items():check('frozen_sample_preserved_after_QA:'+oldname,sha(plan['samples_frozen'][oldname]['parquet'])==value)
    failed=sum(c['status']=='FAIL'for c in checks)
    result=dict(status='FAIL'if failed else('PASS_WITH_WARNINGS'if warnings else'PASS'),created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),plan_sha256=sha(planpath),pass_count=len(checks)-failed,fail_count=failed,checks=checks,warnings=warnings,native_inference_status=inference,details=details,independent_replication_models=actual_numeric,baseline_verification='Frozen source/sample/cache and native arithmetic only; no baseline model refitted',limitations=['Post-result mechanical diagnostic, not a new primary model, factor alpha or causal fund-skill estimate.','One global trade-sign control removes a common linear level component; it does not remove all fund-specific or time-varying carry.','Coupons, distributions, corporate actions and original quantity/price approximation limits remain.','Coefficient changes across the nested models are descriptive; no independent-cross-model standard error or significance test is used.'])
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(dict(status=result['status'],pass_count=result['pass_count'],fail_count=failed)))


def fixture():
    rng=np.random.default_rng(3991);n=4800;fund=np.repeat(np.arange(24),200);s=rng.choice([-1.,1.],n);net=rng.uniform(.1,.7,n);lag=rng.uniform(.1,.7,n)
    frame=pd.DataFrame(dict(fund_id=fund,company_id=np.tile(np.arange(12),400),trade_sign=s,align_current=s*net,align_lag1=s*lag,dar_q_cash_price_pp=-2*s+fund*.03+rng.normal(scale=.02,size=n)))
    old=within_ols(frame,['align_current','align_lag1']);new=within_ols(frame,['align_current','align_lag1','trade_sign'])
    assert (old['beta']<0).all()and np.allclose(new['beta'],[0,0,-2],atol=.01)
    assert new['df_resid']==4773
    absorbed=frame.copy();absorbed.trade_sign=np.where(absorbed.fund_id%2,1.,-1.);assert absorbed.groupby('fund_id').trade_sign.nunique().max()==1
    collinear=frame.copy();collinear.align_current=collinear.trade_sign;values=collinear[['align_current','align_lag1','trade_sign']];within=values-values.groupby(collinear.fund_id).transform('mean');assert np.linalg.matrix_rank(within)==2
    control_frame=frame.assign(control=np.arange(n)%7, absorbed=np.repeat(np.arange(24),200))
    assert independently_retained_terms(control_frame,['align_current','align_lag1','trade_sign'],['control','absorbed'])==['align_current','align_lag1','trade_sign','control']
    assert np.isfinite(new['covariance']).all() and np.array_equal(new['covariance'],new['covariance'].T)
    print(json.dumps(dict(status='PASS',scope='synthetic_only',checks=6)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan');parser.add_argument('--root');parser.add_argument('--output');parser.add_argument('--rtol',type=float,default=1e-5);parser.add_argument('--fixture-only',action='store_true');args=parser.parse_args();fixture()if args.fixture_only else audit(args)


