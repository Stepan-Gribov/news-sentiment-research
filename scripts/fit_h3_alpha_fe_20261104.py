"""Frozen H3 fund-FE OLS, one-way fund CR1 and t(G-1) inference."""
from __future__ import annotations
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from h3_alpha_core_20261104 import unique, digest_lines, numeric
from h3_alpha_io_20261104 import save_json

CURRENT='NTS_Q_current'
LAG='NTS_Q_lagnews'


def _test(beta, variance, df):
    if not np.isfinite([beta,variance]).all() or variance<=0 or df<=0:
        return dict(beta=float(beta),SE=np.nan,t=np.nan,p=np.nan,CI_low=np.nan,CI_high=np.nan,
                    status='undefined_nonpositive_or_nonfinite_variance')
    se=float(np.sqrt(variance));t=float(beta/se);q=float(stats.t.ppf(.975,df))
    return dict(beta=float(beta),SE=se,t=t,p=float(2*stats.t.sf(abs(t),df)),
                CI_low=float(beta-q*se),CI_high=float(beta+q*se),status='defined')


def fit_fund_fe(frame, *, distributed=True, output=None):
    targets=[CURRENT,LAG] if distributed else [CURRENT]
    needed=['fund_id','interval_id','end_date','alpha','past_alpha']+targets
    missing=set(needed)-set(frame)
    if missing:raise ValueError('H3 fit missing: '+str(missing))
    sample=frame.copy().sort_values(['fund_id','end_date','interval_id'],kind='stable').reset_index(drop=True)
    unique(sample,['fund_id','interval_id'],'one-horizon regression sample')
    if not np.isfinite(sample[['alpha','past_alpha']+targets].apply(numeric).to_numpy(float)).all():
        raise ValueError('Nonfinite H3 input; caller must save explicit inclusion decisions')
    singleton=sample.groupby('fund_id',observed=True).interval_id.transform('size').lt(2)
    removed=sample.loc[singleton,['fund_id','interval_id']].copy()
    sample=sample.loc[~singleton].reset_index(drop=True)
    n=len(sample);g=sample.fund_id.nunique()
    result=dict(status='NOT_ESTIMABLE',nobs=n,funds=int(g),singleton_rows_removed=len(removed),
        singleton_funds_removed=int(removed.fund_id.nunique()),distributed=distributed,
        p_reference='Student t(G-1), ordinary two-sided',reference_df=int(g-1),
        weak_cluster_inference=g<30,significance_stars_suppressed=g<30,
        time_FE=False,fund_FE=True,clusters=['fund_id'],multiple_testing_adjustments=False,
        sample_hash=digest_lines(sample.fund_id.astype(str)+'\t'+sample.interval_id.astype(str)))
    def finish():
        if output:
            root=Path(output);root.mkdir(parents=True,exist_ok=True)
            sample.to_parquet(root/'used_sample.parquet',index=False,compression='zstd')
            removed.to_parquet(root/'singleton_removals.parquet',index=False,compression='zstd')
            save_json(root/'fit.json',result)
            if result['status']=='NOT_ESTIMABLE':
                text='NOT_ESTIMABLE\n'+str(result.get('reason'))+'\n'
                (root/'native.summary.txt').write_text(text,encoding='utf-8')
                (root/'native.summary.html').write_text('<pre>'+html.escape(text)+'</pre>',encoding='utf-8')
        return result,sample
    if n<30 or g<2:
        result['reason']='fewer_than_30_rows_or_two_funds_after_singletons'
        return finish()
    terms=targets+['past_alpha']
    original=sample[terms].astype(float)
    within=original-original.groupby(sample.fund_id,observed=True).transform('mean')
    absorbed=[c for c in terms if np.max(np.abs(within[c]))<=1e-12*max(1.,float(np.max(np.abs(original[c]))))]
    result['absorbed_columns']=absorbed
    if set(absorbed)&set(targets):
        result['reason']='required_NTS_target_absorbed_by_fund_FE'
        return finish()
    terms=[c for c in terms if c not in absorbed]
    wx=within[terms].to_numpy(float)
    scaled=wx/np.sqrt(np.square(wx).sum(axis=0))
    if np.linalg.matrix_rank(scaled)<len(terms):
        result['reason']='unabsorbed_regressors_collinear_no_silent_target_or_control_drop'
        return finish()
    full_rank=int(g+len(terms));df_resid=n-full_rank
    if df_resid<=0:
        result['reason']='nonpositive_residual_df';return finish()
    from linearmodels.panel import PanelOLS
    index=pd.MultiIndex.from_arrays([sample.fund_id.astype(str),pd.to_datetime(sample.end_date)],
                                    names=['fund_id','report_date'])
    x=sample[terms].astype(float).copy();x.index=index;x.insert(0,'const',1.)
    y=pd.Series(sample.alpha.to_numpy(float),index=index,name='mean_monthly_factor_adjusted_return')
    clusters=pd.DataFrame({'fund_cluster':pd.Categorical(sample.fund_id.astype(str)).codes},index=index)
    model=PanelOLS(y,x,entity_effects=True,time_effects=False,singletons=True,drop_absorbed=False,check_rank=True)
    fitted=model.fit(cov_type='clustered',clusters=clusters,debiased=True,auto_df=False,
                     count_effects=True,group_debias=True,cluster_entity=False,cluster_time=False)
    if fitted.nobs!=n or not fitted.resids.index.equals(index) or fitted.df_resid!=df_resid:
        raise AssertionError('Native fund-FE estimator changed sample or rank')
    wy=(sample.alpha-sample.alpha.groupby(sample.fund_id,observed=True).transform('mean')).to_numpy(float)
    b=np.linalg.lstsq(wx,wy,rcond=None)[0];residual=wy-wx@b
    codes=pd.Categorical(sample.fund_id.astype(str)).codes
    scores=np.zeros((g,len(terms)));np.add.at(scores,codes,wx*residual[:,None])
    bread=np.linalg.inv(wx.T@wx)
    correction=float(g/(g-1)*(n-1)/(n-full_rank))
    independent_cov=correction*(bread@(scores.T@scores)@bread)
    actual_cov=fitted.cov.loc[terms,terms].to_numpy(float)
    slope_error=float(np.max(np.abs(b-fitted.params[terms].to_numpy(float))))
    covariance_error=float(np.max(np.abs(independent_cov-actual_cov)))
    if not np.allclose(b,fitted.params[terms],rtol=1e-7,atol=1e-10):
        raise AssertionError('Native slopes do not match independent within-FE OLS')
    if not np.allclose(independent_cov,actual_cov,rtol=1e-6,atol=1e-11):
        raise AssertionError('Native covariance does not match frozen one-way CR1')
    coefficients=[]
    for c in fitted.params.index:
        test=_test(float(fitted.params[c]),float(fitted.cov.loc[c,c]),g-1)
        iqr=float(sample[c].quantile(.75)-sample[c].quantile(.25)) if c in sample else np.nan
        coefficients.append(dict(term=c,**test,reference_df=int(g-1),IQR=iqr,
            beta_bps_per_score=10000*test['beta'],minus1_to_plus1_bps=20000*test['beta'] if c in targets else np.nan,
            IQR_effect_bps=10000*test['beta']*iqr,native_residual_df_p=float(fitted.pvalues[c])))
    result.update(status='ESTIMATED',terms=terms,K_full=full_rank,residual_df=df_resid,CR1_correction=correction,
        within_R2=float(fitted.rsquared_within),overall_R2_native=float(fitted.rsquared_overall),
        R2_native=float(fitted.rsquared),coefficients=coefficients,
        QA=dict(native_within_slope_max_abs_error=slope_error,native_CR1_max_abs_error=covariance_error,
                status='PASS'))
    tests=[]
    for term in targets:
        tests.append(dict(test=term+'=0',**_test(float(fitted.params[term]),float(fitted.cov.loc[term,term]),g-1)))
    if distributed:
        block=fitted.cov.loc[targets,targets].to_numpy(float);betas=fitted.params[targets].to_numpy(float)
        tests.append(dict(test='current+lagnews=0',**_test(float(betas.sum()),float(block.sum()),g-1)))
        # The cluster-score rank is at most G-1. Tiny positive eigenvalues from
        # rounding cannot identify a two-restriction Wald test with G=2.
        if g>=3 and np.isfinite(block).all() and np.linalg.matrix_rank(block)==2 and np.linalg.eigvalsh(block).min()>0:
            statistic=float(betas@np.linalg.solve(block,betas)/2)
            tests.append(dict(test='joint_current_lagnews=0',F=statistic,df_num=2,df_denom=int(g-1),
                              p=float(stats.f.sf(statistic,2,g-1)),status='defined'))
        else:
            tests.append(dict(test='joint_current_lagnews=0',F=np.nan,p=np.nan,
                              status='undefined_target_covariance_not_positive_definite'))
    result['target_tests']=tests
    if output:
        root=Path(output);root.mkdir(parents=True,exist_ok=True)
        try:
            if g-1<len(terms):raise np.linalg.LinAlgError('Native overall robust-F block cannot be identified with G-1 below slope count')
            native_text=fitted.summary.as_text()+'\n';native_html=fitted.summary.as_html()+'\n'
        except (np.linalg.LinAlgError,ValueError,ZeroDivisionError) as error:
            result['native_summary_warning']=str(error)
            native_table=pd.DataFrame({'beta':fitted.params,'SE':fitted.std_errors,'t':fitted.tstats,'p_native_residual_df':fitted.pvalues})
            native_text='Native PanelOLS coefficient output; aggregate robust-F undefined/unrenderable.\n'+str(error)+'\n'+native_table.to_string()+'\n'
            native_html='<pre>'+html.escape(native_text)+'</pre>'
        (root/'native.summary.txt').write_text(native_text,encoding='utf-8')
        (root/'native.summary.html').write_text(native_html,encoding='utf-8')
        pd.DataFrame(coefficients).to_csv(root/'coefficients_main_tGminus1.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame({'beta':fitted.params,'SE':fitted.std_errors,'t':fitted.tstats,'p':fitted.pvalues,
                      'residual_df':df_resid}).to_csv(root/'coefficients_native.csv',encoding='utf-8-sig')
        fitted.cov.to_csv(root/'full_covariance.csv',encoding='utf-8-sig')
        pd.DataFrame(independent_cov,index=terms,columns=terms).to_csv(root/'independent_CR1_covariance.csv',encoding='utf-8-sig')
        pd.DataFrame(tests).to_csv(root/'target_tests.csv',index=False,encoding='utf-8-sig')
        sample.assign(residual_native=fitted.resids.to_numpy(float)).to_parquet(
            root/'used_sample_with_residuals.parquet',index=False,compression='zstd')
    return finish()


def synthetic_tests():
    rng=np.random.default_rng(2026110402);g=35;t=12;n=g*t
    ids=np.repeat(np.arange(g).astype(str),t)
    current=rng.uniform(-1,1,n);lag=rng.uniform(-1,1,n);past=rng.normal(.002,.003,n)
    fund_effect=np.repeat(rng.normal(0,.008,g),t)
    frame=pd.DataFrame(dict(fund_id=ids,interval_id=['i'+str(i) for i in range(n)],
        end_date=np.tile(pd.date_range('2020-01-31',periods=t,freq=pd.offsets.MonthEnd()),g),
        alpha=.002*current+.001*lag+.3*past+fund_effect+rng.normal(0,.01,n),
        NTS_Q_current=current,NTS_Q_lagnews=lag,past_alpha=past))
    result,sample=fit_fund_fe(frame)
    assert result['status']=='ESTIMATED' and result['reference_df']==g-1
    assert result['residual_df']==n-g-3 and result['QA']['native_CR1_max_abs_error']<1e-10
    assert len(result['target_tests'])==4 and result['nobs']==n
    comparison,_=fit_fund_fe(frame,distributed=False)
    assert comparison['sample_hash']==result['sample_hash']
    absorbed=frame.copy();absorbed[CURRENT]=absorbed.fund_id.astype(float)
    failure,_=fit_fund_fe(absorbed)
    assert failure['status']=='NOT_ESTIMABLE' and 'absorbed' in failure['reason']
    small,_=fit_fund_fe(frame.iloc[:20]);assert small['status']=='NOT_ESTIMABLE'
    two=frame.iloc[:48].copy();two['fund_id']=np.repeat(['a','b'],24)
    two_result,_=fit_fund_fe(two)
    assert two_result['status']=='ESTIMATED' and two_result['reference_df']==1
    assert two_result['target_tests'][-1]['status']=='undefined_target_covariance_not_positive_definite'
    return dict(status='PASS',synthetic_only=True,checks_pass=9,checks_fail=0,
                slope_error=result['QA']['native_within_slope_max_abs_error'],
                covariance_error=result['QA']['native_CR1_max_abs_error'])


if __name__=='__main__':
    print(json.dumps(synthetic_tests(),indent=2))
