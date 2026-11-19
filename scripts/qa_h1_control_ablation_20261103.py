"""Independent selected-model and intercept-representation QA for H1 ablations."""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def load(path):
    spec = importlib.util.spec_from_file_location('independent_h1_qa_utilities', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit(args):
    if os.name != 'nt': raise RuntimeError('Research-data QA runs on research-host only')
    from scipy import stats
    helper = load(args.qa_helper)
    planpath = Path(args.plan)
    plan = json.loads(planpath.read_text(encoding='utf-8'))
    root = Path(args.root or plan['output_root'])
    output = Path(args.output or root/'qa_independent.json')
    if output.exists(): raise FileExistsError('QA output already exists')
    if 'databaseresources' in [p.lower() for p in output.parts]: raise ValueError('Protected output')
    summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    manifest = json.loads((root/'samples_manifest.json').read_text(encoding='utf-8'))
    fullcov = json.loads((root/'all_model_covariances.json').read_text(encoding='utf-8'))
    records = {r['model']: r for r in summary['models']+summary['no_const_models']}
    coef = pd.concat([pd.read_csv(root/'coefficients.csv'), pd.read_csv(root/'no_const_coefficients.csv')], ignore_index=True)
    checks, warnings, details = [], [], {}

    def check(name, passed, **evidence):
        checks.append(dict(name=name, status='PASS' if bool(passed) else 'FAIL', **helper.clean(evidence)))

    def same(a, b, atol=1e-8, rtol=None):
        return bool(np.allclose(np.asarray(a, float), np.asarray(b, float),
            rtol=args.rtol if rtol is None else rtol, atol=atol, equal_nan=True))

    def covsame(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        s = np.sqrt(np.maximum(np.abs(np.diag(a)), np.abs(np.diag(b))))
        denominator = np.outer(np.maximum(s, 1e-150), np.maximum(s, 1e-150))
        return same(a/denominator, b/denominator, 1e-7)

    expected_main = {focus.upper()+'__'+variant+'__'+controlset
        for focus in ['equity','bond','mixed'] for variant in ['CURRENT_ONLY','CURRENT_PLUS_LAG1']
        for controlset in ['ALL16','DROP_OPENING_WEIGHT','DROP_NEWS_COUNT','DROP_BOTH','EQUITY_SELECTED_SIX']}
    check('frozen_plan_and_exact_30_plus_3_models', summary['plan'] == plan
        and summary['plan_sha256'] == helper.sha(planpath) == manifest['plan_sha256']
        and {r['model'] for r in summary['models']} == expected_main and len(summary['models']) == 30
        and len(summary['no_const_models']) == 3 and len(records) == 33
        and set(records) == set(fullcov['models']))
    expected_sets = dict(ALL16=helper.CONTROLS,
        DROP_OPENING_WEIGHT=[c for c in helper.CONTROLS if c != 'opening_weight'],
        DROP_NEWS_COUNT=[c for c in helper.CONTROLS if c != 'log_news_count_interval'],
        DROP_BOTH=[c for c in helper.CONTROLS if c not in ['opening_weight','log_news_count_interval']],
        EQUITY_SELECTED_SIX=['closing_HHI','bond_indicator','imoex_ret','RVI','dummy_mar2014','dummy_sep2018'])
    check('fixed_five_control_sets_and_FE_CGM_policy', plan['control_sets'] == expected_sets
        and plan['fixed_effects'] == ['fund_id'] and plan['time_FE'] is False
        and plan['clusters'] == ['fund_id','company_id'] and plan['cluster_time'] is False
        and plan['covariance_options'] == dict(cov_type='clustered',debiased=True,auto_df=False,count_effects=True,group_debias=True,PSD_repair=False))
    check('unique_coefficient_keys', not coef.duplicated(['model','term']).any() and set(coef.model) == set(records))
    frames, hashes = {}, {}
    for focus, source in plan['sources'].items():
        hashes[focus] = helper.sha(source['path'])
        frame = pd.read_parquet(source['path']).sort_values(['fund_id','end_date','security_id','interval_id'], kind='stable').reset_index(drop=True)
        frames[focus] = frame
        check('exact_frozen_source:'+focus, hashes[focus] == source['sha256'] == manifest['samples'][focus]['sha256']
            and helper.row_hash(frame) == source['row_hash'] == manifest['samples'][focus]['row_hash']
            and (len(frame),frame.fund_id.nunique(),frame.company_id.nunique()) == (source['rows'],source['funds'],source['companies'])
            and not frame.row_id.duplicated().any() and frame.fund_focus.eq(focus).all())
        check('full_design_finite_and_log_news_definition:'+focus, np.isfinite(frame[[helper.OUTCOME,helper.CURRENT,helper.LAG]+helper.CONTROLS].astype(float)).all().all()
            and same(frame.log_news_count_interval, np.log1p(frame.n_relevant), 1e-12,1e-12))
        entry = frame.quantity_open.eq(0)&frame.quantity_close.gt(0)
        check('entry_outcome_and_initial_weight_convention:'+focus, frame.loc[entry,'quantity_change_pct'].eq(100).all()
            and frame.loc[entry,'opening_weight'].eq(0).all())
    for name, record in records.items():
        frame = frames[record['focus']]
        source = plan['sources'][record['focus']]
        required = [helper.CURRENT,helper.LAG] if record['variant'] == 'CURRENT_PLUS_LAG1' else [helper.CURRENT]
        controls = expected_sets[record['control_set']]
        terms = helper.selected_terms(frame, required, controls)
        exported = fullcov['models'][name]
        order = exported['term_order']
        covariance = np.asarray(exported['covariance'],float)
        beta = np.asarray(exported['params'],float)
        table = coef.loc[coef.model.eq(name)].set_index('term').reindex(order)
        check('all_models_same_rows_exact_terms_df:'+name, record['retained_terms'] == terms == exported['retained_terms']
            and exported['controls'] == controls and exported['source'] == source
            and exported['nobs'] == record['nobs'] == len(frame)
            and record['funds'] == source['funds'] and record['companies'] == source['companies']
            and record['df_resid'] == exported['df_resid'] == len(frame)-source['funds']-len(terms)
            and order == (['const'] if record['explicit_const'] else [])+terms)
        variance = np.diag(covariance)
        se = np.sqrt(np.where(variance>0,variance,np.nan))
        p = 2*stats.t.sf(abs(beta/se), record['df_resid'])
        critical = stats.t.ppf(.975, record['df_resid'])
        check('all_native_coefficients_covariance_t_CI:'+name, set(table.index) == set(order)
            and same(table.beta,beta) and same(table.se,se) and same(table.p,p,1e-10)
            and same(table.ci_low,beta-critical*se) and same(table.ci_high,beta+critical*se)
            and same(exported['std_errors'],se) and same(exported['p_values'],p,1e-10))
        if (variance<=0).any(): warnings.append(dict(model=name,kind='UNDEFINED_NATIVE_COEFFICIENT_VARIANCE',terms=[order[i] for i in np.where(variance<=0)[0]]))
        if len(required)==2:
            indices=[order.index(t) for t in required]
            block=covariance[np.ix_(indices,indices)];targets=beta[indices]
            check('distributed_target_full_covariance:'+name,same(record['target_covariance'],block))
            total_var=float(block.sum());total_beta=float(targets.sum())
            if total_var>0:
                total_se=np.sqrt(total_var);total_p=2*stats.t.sf(abs(total_beta/total_se),record['df_resid'])
                check('distributed_sum_crosscovariance_and_CI:'+name,record['sum_status']=='ESTIMATED'
                    and same(record['sum_beta'],total_beta) and same(record['sum_se'],total_se)
                    and same(record['sum_p'],total_p,1e-10)
                    and same(record['sum_ci_low'],total_beta-critical*total_se) and same(record['sum_ci_high'],total_beta+critical*total_se))
            else: check('undefined_sum_remains_undefined:'+name,record['sum_status']!='ESTIMATED' and record['sum_p'] is None)
            eig=np.linalg.eigvalsh(block)
            if eig.min()>0:
                statistic=float(targets@np.linalg.solve(block,targets))
                check('distributed_joint_chi_square:'+name,record['joint_status']=='ESTIMATED'
                    and same(record['joint_stat'],statistic) and same(record['joint_p'],stats.chi2.sf(statistic,2),1e-10))
            else: check('undefined_joint_remains_undefined:'+name,record['joint_status']!='ESTIMATED' and record['joint_p'] is None)

    selected = [name for name,r in records.items() if r['explicit_const'] and r['control_set'] in ['ALL16','DROP_BOTH']]
    selected.append('EQUITY__CURRENT_PLUS_LAG1__EQUITY_SELECTED_SIX')
    independent_cache = {}
    for name in selected:
        record=records[name];frame=frames[record['focus']];terms=record['retained_terms']
        independent=helper.independent_rowwise(frame,terms);native=helper.native_fit(frame,terms)
        exported=fullcov['models'][name];table=coef.loc[coef.model.eq(name)].set_index('term').reindex(independent['terms'])
        check('independent_full_coefficients_and_CGM_covariance:'+name,
            same(independent['beta'],exported['params']) and covsame(independent['covariance'],exported['covariance'])
            and same(independent['beta'],native.params) and covsame(independent['covariance'],native.cov)
            and same(table.beta,native.params) and same(table.se,native.std_errors)
            and same(table.p,native.pvalues,1e-10)
            and independent['df_resid']==native.df_resid==record['df_resid']
            and same(independent['within_R2'],native.rsquared_within,1e-10)
            and same(independent['within_R2'],record['rsquared_within'],1e-10))
        independent_cache[name]=independent
        details[name]=dict(nobs=len(frame),funds=record['funds'],companies=record['companies'],df_resid=independent['df_resid'],retained_terms=terms,
            within_R2=independent['within_R2'],target_coefficients={t:float(native.params[t]) for t in [helper.CURRENT,helper.LAG] if t in terms})
        print(json.dumps(dict(status='INDEPENDENT_REPLICA_COMPLETE',model=name)),flush=True)
    for spec in plan['representation_checks']:
        basename=spec['focus'].upper()+'__'+spec['variant']+'__'+spec['controls']
        name=basename+'__NO_EXPLICIT_CONST';record=records[name];frame=frames[spec['focus']]
        independent=independent_cache[basename];no=fullcov['models'][name]
        check('no_const_independent_slopes_covariance_df:'+name,
            same(no['params'],independent['beta'][1:]) and covsame(no['covariance'],independent['covariance'][1:,1:])
            and no['df_resid']==independent['df_resid'] and no['explicit_const'] is False)
        raw=pd.read_parquet(root/'representation_checks'/(name+'.parquet'))
        check('representation_exact_row_ids_and_source_keys:'+name,
            raw[['row_id','fund_id','company_id','security_id','start_date','end_date']].equals(frame[['row_id','fund_id','company_id','security_id','start_date','end_date']]))
        terms=record['retained_terms'];beta=independent['beta'][1:];const=independent['beta'][0]
        xb=frame[terms].to_numpy(float)@beta
        remainder=pd.Series(frame[helper.OUTCOME].to_numpy(float)-xb)
        fe=remainder.groupby(frame.fund_id.astype(str),sort=False).transform('mean').to_numpy(float)
        total=xb+fe;residual=frame[helper.OUTCOME].to_numpy(float)-total
        expected=dict(with_const_fitted_excluding_FE=xb+const,without_const_fitted_excluding_FE=xb,
            with_const_FE=fe-const,without_const_FE=fe,with_const_total_prediction=total,without_const_total_prediction=total,
            with_const_residual=residual,without_const_residual=residual)
        check('independent_all_row_predictions_residuals_and_FE_normalization:'+name,
            all(same(raw[column],values,3e-8) for column,values in expected.items())
            and same(raw.with_const_fitted_excluding_FE-raw.without_const_fitted_excluding_FE,const,3e-8)
            and same(raw.with_const_FE-raw.without_const_FE,-const,3e-8)
            and frame.fund_id.groupby(frame.fund_id).size().size==record['funds'])
    for focus,source in plan['sources'].items():
        check('frozen_source_preserved_after_QA:'+focus,helper.sha(source['path'])==hashes[focus])
    failed=sum(c['status']=='FAIL' for c in checks)
    result=dict(status='FAIL' if failed else ('PASS_WITH_WARNINGS' if warnings else 'PASS'),
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),plan_sha256=helper.sha(planpath),
        checks=checks,pass_count=len(checks)-failed,fail_count=failed,warnings=warnings,
        independent_numerical_models=selected,independent_numerical_model_count=len(selected),
        details=details,const_representation_checks=3,source_hashes_preserved=True,
        scope='All33 model metadata/native coefficient arithmetic/full covariance exports checked;13 unique models independently refit native+rowwise CGM;3 no-const rowwise representations independently reconstructed.',
        methodological_notes=['With fund FE, removing explicit const changes intercept normalization, not unrestricted fund intercepts or slopes.',
            'The p-value of const or of a control does not determine whether a control belongs in the model.',
            'Initial weight relates to the positive-base outcome denominator and is zero for conditional entries. Dropping it changes size/entry adjustment.',
            'log_news_count_interval=log1p(n_relevant); retaining it estimates tone conditional on news intensity. Omitting it changes the conditional association.',
            'SIX and DROP_BOTH are different control sets. Control-sensitive results remain exploratory; no independent cross-model p-value for coefficient changes is used.'])
    output.write_text(json.dumps(helper.clean(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],pass_count=result['pass_count'],fail_count=failed,numerical_replicas=len(selected))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--qa-helper',required=True)
    parser.add_argument('--root');parser.add_argument('--output');parser.add_argument('--rtol',type=float,default=2e-6)
    audit(parser.parse_args())
