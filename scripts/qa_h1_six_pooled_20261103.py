"""One independent rowwise OLS/CGM audit of the pooled H1 SIX model."""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def audit(args):
    if os.name != 'nt': raise RuntimeError('Research-data QA runs on research-host only')
    from scipy import stats
    spec=importlib.util.spec_from_file_location('independent_h1_qa_utilities',args.qa_helper)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    planpath=Path(args.plan);plan=json.loads(planpath.read_text(encoding='utf-8'))
    root=Path(args.root or plan['output_root']);output=Path(args.output or root/'qa_independent.json')
    if output.exists():raise FileExistsError('Refusing to overwrite existing QA')
    if 'databaseresources'in [p.lower()for p in output.parts]:raise ValueError('Protected output path')
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    manifest=json.loads((root/'input_manifest.json').read_text(encoding='utf-8'))
    covariance=json.loads((root/'full_covariance.json').read_text(encoding='utf-8'))
    tests=json.loads((root/'targetcovariance.json').read_text(encoding='utf-8'))
    co=pd.read_csv(root/'coefficients.csv').set_index('term')
    checks=[]
    def check(name,passed,**evidence):checks.append(dict(name=name,status='PASS'if bool(passed)else'FAIL',**helper.clean(evidence)))
    def same(a,b,atol=1e-8):return bool(np.allclose(np.asarray(a,float),np.asarray(b,float),rtol=args.rtol,atol=atol,equal_nan=True))
    before=helper.sha(plan['input_panel'])
    check('frozen_plan_and_source_byte_hash',summary['plan']==plan and summary['manifest']==manifest
        and manifest['source_sha256']==before and manifest['plan_sha256']==helper.sha(planpath))
    helper_hashes={key:helper.sha(plan[key])for key in ['base_helper','lag_helper']}
    check('original_FE_CGM_helper_hashes',all(manifest[key+'_sha256']==value for key,value in helper_hashes.items()))
    frame=pd.read_parquet(plan['input_panel']).sort_values(['fund_id','end_date','security_id','interval_id'],kind='stable').reset_index(drop=True)
    n,funds,companies=len(frame),frame.fund_id.nunique(),frame.company_id.nunique()
    check('exact_unrestricted_common_reference_sample',
        (n,funds,companies)==(194041,591,269)==(plan['expected_rows'],plan['expected_funds'],plan['expected_companies'])
        and helper.row_hash(frame)==plan['expected_row_hash']==manifest['source_row_hash']
        and not frame.row_id.duplicated().any() and plan['minimum_history_filter']is None and summary['minimum_history_filter']is None)
    controls=['closing_HHI','bond_indicator','imoex_ret','RVI','dummy_mar2014','dummy_sep2018']
    terms=[helper.CURRENT,helper.LAG]+controls
    check('same_fixed_eight_slopes_and_covariance_policy',plan['controls']==controls and plan['targets']==terms[:2]
        and plan['fixed_effects']==['fund_id']and plan['clusters']==['fund_id','company_id']
        and plan['time_FE']is False and plan['cluster_time']is False and plan['outcome']==helper.OUTCOME
        and covariance['terms']==['const']+terms and list(co.index)==['const']+terms
        and helper.selected_terms(frame,terms[:2],controls)==terms)
    independent=helper.independent_rowwise(frame,terms)
    block=np.asarray(covariance['matrix'],float);beta=independent['beta'];cov=independent['covariance'];df=independent['df_resid']
    scale=np.sqrt(np.maximum(np.abs(np.diag(block)),np.abs(np.diag(cov))))
    denominator=np.outer(np.maximum(scale,1e-150),np.maximum(scale,1e-150))
    check('independent_full_native_intercept_and_slopes',same(co.beta,beta))
    check('independent_full_CGM_covariance',same(block/denominator,cov/denominator,1e-7))
    se=np.sqrt(np.where(np.diag(cov)>0,np.diag(cov),np.nan));critical=stats.t.ppf(.975,df);p=2*stats.t.sf(abs(beta/se),df)
    check('independent_all_SE_two_sided_p_and_CI',same(co.se,se)and same(co.p,p,1e-10)
        and same(co.ci_low,beta-critical*se)and same(co.ci_high,beta+critical*se))
    check('native_df_counts_and_within_R2',summary['df_resid']==df==n-funds-len(terms)
        and (summary['nobs'],summary['funds'],summary['companies'])==(n,funds,companies)
        and same(summary['rsquared_within'],independent['within_R2'],1e-10)
        and summary['securities']==frame.security_id.nunique()
        and summary['singleton_funds']==int(frame.groupby('fund_id').size().eq(1).sum())
        and summary['fund_focus_counts']==frame.fund_focus.value_counts().to_dict())
    indices=[independent['terms'].index(t)for t in terms[:2]];target_beta=beta[indices];target_cov=cov[np.ix_(indices,indices)]
    check('independent_target_2x2_crosscovariance',tests==summary['distributed_tests']and tests['target_terms']==terms[:2]
        and tests['covariance_repair']is False and same(tests['target_covariance'],target_cov)
        and same(tests['var_current'],target_cov[0,0])and same(tests['var_lag'],target_cov[1,1])
        and same(tests['cov_current_lag'],target_cov[0,1]))
    joint=tests['joint_current_lag_zero'];total=tests['sum_current_lag'];eigen=np.linalg.eigvalsh(target_cov)
    check('target_covariance_eigenvalues',same(joint['target_covariance_eigenvalues'],eigen))
    if eigen.min()>0:
        statistic=float(target_beta@np.linalg.solve(target_cov,target_beta))
        check('independent_joint_chi_square_two',joint['status']=='ESTIMATED'and joint['df']==2
            and same(joint['statistic'],statistic)and same(joint['p_value'],stats.chi2.sf(statistic,2),1e-10))
    else:check('undefined_joint_not_repaired',joint['status']!='ESTIMATED'and joint['p_value']is None)
    sum_beta=float(target_beta.sum());sum_variance=float(target_cov.sum())
    check('sum_uses_full_crosscovariance',same(total['beta'],sum_beta)and same(total['variance'],sum_variance))
    if sum_variance>0:
        sum_se=np.sqrt(sum_variance);sum_p=2*stats.t.sf(abs(sum_beta/sum_se),df)
        check('independent_sum_SE_t_CI',total['status']=='ESTIMATED'and same(total['se_native'],sum_se)
            and same(total['p_native'],sum_p,1e-10)and same(total['ci95_native_low'],sum_beta-critical*sum_se)
            and same(total['ci95_native_high'],sum_beta+critical*sum_se))
    else:check('undefined_sum_not_repaired',total['status']!='ESTIMATED'and total['p_native']is None)
    check('source_and_helper_bytes_preserved',helper.sha(plan['input_panel'])==before
        and all(helper.sha(plan[key])==value for key,value in helper_hashes.items()))
    failed=sum(c['status']=='FAIL'for c in checks)
    warnings=[]
    if not np.isfinite(se).all():warnings.append(dict(kind='UNDEFINED_NATIVE_VARIANCE',terms=[independent['terms'][i]for i in np.where(~np.isfinite(se))[0]]))
    result=dict(status='FAIL'if failed else('PASS_WITH_WARNINGS'if warnings else'PASS'),created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        checks=checks,pass_count=len(checks)-failed,fail_count=failed,warnings=warnings,
        independent_model_count=1,independent_method='Rowwise fund-demeaned OLS and fund/company/intersection CGM; no other model fitted or grid repeated.',
        nobs=n,funds=funds,companies=companies,source_sha256=before,source_row_hash=helper.row_hash(frame),
        df_resid=df,within_R2=independent['within_R2'],target_beta=target_beta,target_covariance=target_cov,
        sum_beta=sum_beta,sum_variance=sum_variance,
        comparison_footnote='Pooled reference includes all591 eligible funds without the60/24 history filter. Previously reported equity/bond/mixed groups retain that eligibility filter and are not exhaustive partitions of the pooled reference or of all Russian funds.')
    output.write_text(json.dumps(helper.clean(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],pass_count=result['pass_count'],fail_count=failed)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True)
    parser.add_argument('--qa-helper',required=True);parser.add_argument('--root');parser.add_argument('--output')
    parser.add_argument('--rtol',type=float,default=2e-6);audit(parser.parse_args())
