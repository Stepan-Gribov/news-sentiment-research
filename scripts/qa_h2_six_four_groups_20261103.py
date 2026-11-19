"""Independent rowwise QA of four fixed H2 SIX/current-and-lag panels."""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PARTS = ['net_positive', 'net_negative', 'net_positive_lag1', 'net_negative_lag1']
CONTROLS = ['closing_HHI', 'bond_indicator', 'imoex_ret', 'RVI', 'dummy_mar2014', 'dummy_sep2018']
MATRICES = dict(current=np.array([[1.,0.,0.,0.],[0.,1.,0.,0.]]),
    lag1=np.array([[0.,0.,1.,0.],[0.,0.,0.,1.]]),
    total_current_plus_lag1=np.array([[1.,0.,1.,0.],[0.,1.,0.,1.]]))


def contrast(beta, covariance, vector, df):
    from scipy import stats
    beta, covariance, vector = np.asarray(beta,float), np.asarray(covariance,float), np.asarray(vector,float)
    value, variance = float(vector@beta), float(vector@covariance@vector)
    result = dict(beta=value, variance=variance, status='UNDEFINED_COVARIANCE', se=None, t=None, p=None, low=None, high=None)
    if np.isfinite(value) and np.isfinite(variance) and variance > 0 and df > 0:
        se = np.sqrt(variance); t = value/se; critical = stats.t.ppf(.975,df)
        result.update(status='ESTIMATED',se=se,t=t,p=2*stats.t.sf(abs(t),df),low=value-critical*se,high=value+critical*se)
    return result


def audit(args):
    if os.name != 'nt': raise RuntimeError('Actual research QA runs on research-host only')
    from scipy import stats
    spec = importlib.util.spec_from_file_location('independent_h1_qa_utilities',args.qa_helper)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    planpath = Path(args.plan); plan = json.loads(planpath.read_text(encoding='utf-8'))
    root = Path(args.root or plan['output_root']); output = Path(args.output or root/'qa_independent.json')
    if output.exists(): raise FileExistsError('Refusing to overwrite an existing QA report')
    if 'databaseresources' in [p.lower() for p in output.parts]: raise ValueError('Protected output')
    summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    covariance = json.loads((root/'full_covariance.json').read_text(encoding='utf-8'))
    tests_file = json.loads((root/'h2_tests.json').read_text(encoding='utf-8'))
    manifest = json.loads((root/'source_manifest.json').read_text(encoding='utf-8'))
    samples = json.loads((root/'samples_manifest.json').read_text(encoding='utf-8'))
    coefs = pd.read_csv(root/'coefficients.csv')
    records = {r['group']:r for r in summary['models']}
    checks, warnings, details, before, used_before = [], [], {}, {}, {}

    def check(name, passed, **evidence):
        checks.append(dict(name=name,status='PASS' if bool(passed) else 'FAIL',**helper.clean(evidence)))

    def same(a,b,atol=1e-8):
        return bool(np.allclose(np.asarray(a,float),np.asarray(b,float),rtol=args.rtol,atol=atol,equal_nan=True))

    def covsame(a,b):
        a,b = np.asarray(a,float),np.asarray(b,float)
        s = np.sqrt(np.maximum(np.abs(np.diag(a)),np.abs(np.diag(b))))
        denominator = np.outer(np.maximum(s,1e-150),np.maximum(s,1e-150))
        return same(a/denominator,b/denominator,1e-7)

    check('four_frozen_models_and_plan', summary['specification']==plan and manifest['plan_sha256']==helper.sha(planpath)
        and set(records)==set(covariance)==set(samples)==set(tests_file)=={'pooled','equity','bond','mixed'}
        and summary['unique_native_fits']==4 and len(summary['models'])==4)
    check('required_four_parts_six_controls_and_no_new_selection', plan['targets']==PARTS and plan['controls']==CONTROLS
        and plan['h2_scopes']==list(MATRICES) and plan['multiple_testing_adjustments'] is False
        and plan['covariance_repair'] is False and summary['no_selection_by_significance'] is True
        and plan['existing_upper_tail_rule']['new_trimming'] is False
        and not any('holm' in c.lower() or 'adjusted_p' in c.lower() for c in coefs.columns))
    check('same_FE_and_two_way_CGM', plan['fixed_effects']==['fund_id'] and plan['clusters']==['fund_id','company_id']
        and plan['time_FE'] is False and plan['cluster_time'] is False
        and plan['covariance_options']==dict(cov_type='clustered',cluster_entity=False,cluster_time=False,
            group_debias=True,debiased=True,auto_df=False,count_effects=True)
        and plan['panel_options']==dict(entity_effects=True,time_effects=False,singletons=True,drop_absorbed=False,check_rank=True))
    check('unique_coefficient_keys_and_four_groups', not coefs.duplicated(['group','term']).any() and set(coefs.group)==set(records))
    check('frozen_original_H2_helper', helper.sha(plan['helper_path'])==plan['helper_sha256']==manifest['helper_sha256'])
    helper_before = helper.sha(plan['helper_path'])
    for definition in plan['groups']:
        group = definition['group']; record = records[group]; meta = samples[group]
        before[group] = helper.sha(definition['input_panel']); used_before[group] = helper.sha(meta['path'])
        original = pd.read_parquet(definition['input_panel']).sort_values(['fund_id','end_date','security_id','interval_id'],kind='stable').reset_index(drop=True)
        used = pd.read_parquet(meta['path'])
        n,funds,companies = len(original),original.fund_id.nunique(),original.company_id.nunique()
        check('exact_prior_H1_sample_and_frozen_bytes:'+group,
            before[group]==definition['expected_source_sha256']==meta['source_sha256']==manifest['sources'][group]['sha256_before']
            and used_before[group]==meta['sha256']
            and (n,funds,companies)==tuple(definition[k] for k in ['expected_rows','expected_funds','expected_companies'])
            and helper.row_hash(original)==helper.row_hash(used)==definition['expected_ordered_row_hash']==meta['ordered_row_hash']
            and not original.row_id.duplicated().any())
        check('all_original_columns_values_preserved_literal:'+group, set(used.columns)==set(original.columns)|set(PARTS)
            and original.equals(used[original.columns]),
            mismatched_columns=[c for c in original if c not in used or not original[c].equals(used[c])])
        derived = original.copy()
        for source,positive,negative in [('net_sentiment',PARTS[0],PARTS[1]),('net_sentiment_lag1',PARTS[2],PARTS[3])]:
            values = pd.to_numeric(original[source],errors='coerce')
            derived[positive] = np.maximum(values,0)
            derived[negative] = np.maximum(-values,0)
            check('independent_exact_parts_and_no_NA_imputation:'+group+':'+source,
                np.array_equal(used[positive].to_numpy(float),derived[positive].to_numpy(float),equal_nan=True)
                and np.array_equal(used[negative].to_numpy(float),derived[negative].to_numpy(float),equal_nan=True)
                and np.array_equal((used[positive]-used[negative]).to_numpy(float),values.to_numpy(float),equal_nan=True))
        terms = helper.selected_terms(derived,PARTS,CONTROLS)
        order = ['const']+PARTS+CONTROLS
        full = covariance[group]; table = coefs.loc[coefs.group.eq(group)].set_index('term').reindex(order)
        check('exact_required_terms_rows_and_native_df:'+group, terms==PARTS+CONTROLS
            and record['used_terms']==full['terms']==order and record['dropped_terms']==[]
            and len(table)==11 and set(coefs.loc[coefs.group.eq(group),'term'])==set(order)
            and (record['nobs'],record['funds'],record['companies'])==(n,funds,companies)
            and record['df_resid']==full['df_resid']==n-funds-10
            and record['minimum_history_filter']==definition['minimum_history_filter'])
        result = helper.independent_rowwise(derived,terms)
        beta,cov,df = result['beta'],result['covariance'],result['df_resid']
        check('independent_all_eleven_coefficients_and_full_CGM_covariance:'+group,
            same(beta,table.beta,1e-8) and covsame(cov,full['matrix']) and df==record['df_resid'])
        diagonal = np.diag(cov); se = np.sqrt(np.where(diagonal>0,diagonal,np.nan))
        t = beta/se; p = 2*stats.t.sf(abs(t),df); critical = stats.t.ppf(.975,df)
        check('independent_all_native_SE_t_p_CI:'+group,same(se,table.se_native)
            and same(t,table.t_native) and same(p,table.p_native,1e-10)
            and same(beta-critical*se,table.ci95_native_low) and same(beta+critical*se,table.ci95_native_high))
        check('independent_within_R2_and_same_metadata:'+group,same(result['within_R2'],record['rsquared_within'],1e-10)
            and record['securities']==original.security_id.nunique()
            and record['singleton_funds']==int(original.groupby('fund_id').size().eq(1).sum()))
        if not np.isfinite(se).all():warnings.append(dict(group=group,kind='UNDEFINED_COEFFICIENT_VARIANCE',terms=[order[i] for i in np.where(~np.isfinite(se))[0]]))
        block = cov[1:5,1:5]; target_beta = beta[1:5]
        h2 = tests_file[group]
        check('independent_full_four_target_covariance:'+group,h2==record['h2_tests'] and h2['target_terms']==PARTS
            and h2['covariance_repair'] is False and h2['native_df_resid']==df and same(h2['full_target_covariance'],block))
        scope_details = {}
        for scope,matrix in MATRICES.items():
            test = h2['tests'][scope]
            b = matrix@target_beta; v = matrix@block@matrix.T
            check('independent_scope_transformation_and_full_crosscov:'+group+':'+scope,
                same(test['transformation_from_full_target_block'],matrix)
                and same([test['beta_positive'],test['beta_negative']],b) and same(test['target_covariance'],v))
            calculated = {}
            for name,vector in [('difference_zero',[1.,-1.]),('sum_zero',[1.,1.]),('positive_coefficient',[1.,0.]),('negative_coefficient',[0.,1.])]:
                independent = contrast(b,v,vector,df); native = test[name]; calculated[name] = independent
                check('independent_contrast_native_t_CI:'+group+':'+scope+':'+name,
                    native['status']==independent['status'] and native['native_df_resid']==df
                    and native['covariance_repair'] is False and same(native['contrast'],vector)
                    and same(native['beta'],independent['beta']) and same(native['variance'],independent['variance'])
                    and same(native['se_native'],independent['se']) and same(native['t_native'],independent['t'])
                    and same(native['p_native'],independent['p'],1e-10)
                    and same(native['ci95_native_low'],independent['low']) and same(native['ci95_native_high'],independent['high']))
            valid = all(calculated[name]['status']=='ESTIMATED' for name in ['difference_zero','sum_zero'])
            p_abs = max(calculated['difference_zero']['p'],calculated['sum_zero']['p']) if valid else None
            difference = float(abs(b[1])-abs(b[0]))
            check('independent_absolute_IUT_and_directional_labels:'+group+':'+scope,
                test['status']==('ESTIMATED' if valid else 'UNDEFINED_COVARIANCE')
                and same(test['p_abs'],p_abs,1e-10) and same(test['negative_absolute_minus_positive_absolute'],difference)
                and test['negative_absolute_stronger']==bool(difference>0)
                and test['expected_directional_signs']==bool(b[0]>=0 and b[1]<=0))
            # Signed NetMinus=min(Net,0) is a reparameterization only: the
            # component tests swap and the absolute IUT stays unchanged.
            sign_flip = np.diag([1.,-1.]); signed_beta=sign_flip@b; signed_cov=sign_flip@v@sign_flip
            signed_diff=contrast(signed_beta,signed_cov,[1.,-1.],df)
            signed_sum=contrast(signed_beta,signed_cov,[1.,1.],df)
            check('negative_signed_convention_invariance:'+group+':'+scope,
                same(signed_diff['p'],calculated['sum_zero']['p'],1e-10)
                and same(signed_sum['p'],calculated['difference_zero']['p'],1e-10))
            scope_details[scope]=dict(beta_positive=float(b[0]),beta_negative=float(b[1]),
                p_difference=calculated['difference_zero']['p'],p_sum=calculated['sum_zero']['p'],p_abs=p_abs,
                negative_absolute_minus_positive_absolute=difference,expected_directional_signs=bool(b[0]>=0 and b[1]<=0))
            if not valid:warnings.append(dict(group=group,scope=scope,kind='UNDEFINED_H2_CONTRAST_VARIANCE'))
        details[group]=dict(nobs=n,funds=funds,companies=companies,df_resid=df,within_R2=result['within_R2'],
            source_sha256=before[group],source_row_hash=helper.row_hash(original),target_coefficients=target_beta,
            full_target_covariance=block,h2_tests=scope_details)
        print(json.dumps(dict(status='INDEPENDENT_H2_MODEL_COMPLETE',group=group)),flush=True)
    for definition in plan['groups']:
        group=definition['group']
        check('source_and_used_sample_bytes_preserved:'+group,helper.sha(definition['input_panel'])==before[group]
            and helper.sha(samples[group]['path'])==used_before[group])
    check('original_helper_preserved',helper.sha(plan['helper_path'])==helper_before)
    failed=sum(c['status']=='FAIL' for c in checks)
    result=dict(status='FAIL' if failed else ('PASS_WITH_WARNINGS' if warnings else 'PASS'),
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),pass_count=len(checks)-failed,fail_count=failed,
        checks=checks,warnings=warnings,independent_numerical_models=['pooled','equity','bond','mixed'],details=details,
        scope='Four independent rowwise fund-within OLS/CGM replicas, all11 coefficients/full covariance/native inference/current-lag-total H2 IUT, exact sample/source preservation. No native refits or other models run.',
        reporting='Current H2 is primary; lag and total-current-plus-lag are separately reported. Absolute asymmetry does not itself prove stronger negative response or the expected H1 signs. Unadjusted p; fixed exploratory SIX transfer.')
    output.write_text(json.dumps(helper.clean(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],pass_count=result['pass_count'],fail_count=failed)))


def fixture():
    covariance=np.diag([.01,.01]);df=100
    for beta,expected_reject in [([1.,-1.],False),([1.,1.],False),([1.,-2.],True),([2.,-1.],True)]:
        p=max(contrast(beta,covariance,[1.,-1.],df)['p'],contrast(beta,covariance,[1.,1.],df)['p'])
        assert (p<.05)==expected_reject
    invalid=contrast([1.,-2.],[[1.,2.],[2.,1.]],[1.,-1.],df)
    assert invalid['status']=='UNDEFINED_COVARIANCE' and invalid['p'] is None
    full=np.array([[1.,.1,.4,.2],[.1,2.,.3,.5],[.4,.3,3.,.2],[.2,.5,.2,4.]])
    total=MATRICES['total_current_plus_lag1']@full@MATRICES['total_current_plus_lag1'].T
    assert np.allclose(total,[[4.8,.8],[.8,7.]])
    print(json.dumps(dict(status='PASS',scope='synthetic_only',checks=6)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan');parser.add_argument('--qa-helper')
    parser.add_argument('--root');parser.add_argument('--output');parser.add_argument('--rtol',type=float,default=2e-6)
    parser.add_argument('--fixture-only',action='store_true');args=parser.parse_args()
    fixture() if args.fixture_only else audit(args)
