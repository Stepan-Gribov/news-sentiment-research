"""Second-path numerical QA on PC; writes a separate QA version only.

This is an executor cross-check, not a substitute for independent research
review. DuckDB aggregation and full-dummy OLS do not call the production kernels.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from h3_alpha_io_20261104 import ROOT, file_record, save_json, sha, utc


def run(run_root):
    run_root=Path(run_root)
    seal=json.loads((run_root/'manifest.json').read_text(encoding='utf-8'))
    for record in seal['inputs'].values():file_record(record['path'],record['sha256'])
    for record in seal['files'].values():file_record(record['path'],record['sha256'])
    summary=json.loads((run_root/'summary.json').read_text(encoding='utf-8'))
    qa_root=ROOT/'qa'/(run_root.name+'_'+sha(__file__)[:12]);qa_root.mkdir(parents=True,exist_ok=True)
    checks=[]
    def check(name,condition,**detail):
        checks.append(dict(check=name,passed=bool(condition),**detail))
    import duckdb
    con=duckdb.connect(':memory:');con.execute('SET threads=2')
    nts_manifest=Path(seal['inputs']['nts_manifest']['path'])
    nts_root=nts_manifest.parent
    contributions=nts_root/'quantity_news_contributions.parquet'
    con.execute('CREATE VIEW contributions AS SELECT * FROM read_parquet(\''+str(contributions).replace("'","''")+'\')')
    check('unique_fund_security_interval_contributions',con.execute('SELECT count(*) FROM (SELECT fund_id,interval_id,security_id FROM contributions GROUP BY ALL HAVING count(*)>1)').fetchone()[0]==0)
    reconstructed=con.execute('''SELECT fund_id,interval_id,
        sum(abs(B_Q)) AS gross_u,
        sum(B_Q*net_sentiment) FILTER(WHERE NTS_common_E) AS num0,
        sum(abs(B_Q*net_sentiment)) FILTER(WHERE NTS_common_E) AS den0,
        sum(B_Q*net_sentiment_lag1) FILTER(WHERE NTS_common_E) AS numL,
        sum(abs(B_Q*net_sentiment_lag1)) FILTER(WHERE NTS_common_E) AS denL,
        sum(abs(B_Q)) FILTER(WHERE NTS_common_E) AS gross_e,
        sum(abs(B_Q)) FILTER(WHERE NTS_common_E AND net_sentiment<>0) AS covered0,
        sum(abs(B_Q)) FILTER(WHERE NTS_common_E AND net_sentiment_lag1<>0) AS coveredL,
        count(DISTINCT company_id) FILTER(WHERE NTS_common_E AND abs(B_Q*net_sentiment)>0) AS companies0,
        count(DISTINCT company_id) FILTER(WHERE NTS_common_E AND abs(B_Q*net_sentiment_lag1)>0) AS companiesL
        FROM contributions GROUP BY fund_id,interval_id''').fetchdf()
    nts=pd.read_parquet(nts_root/'fund_period_nts_primary.parquet')
    d=nts.merge(reconstructed,on=['fund_id','interval_id'],how='left',validate='one_to_one')
    for actual,expected in [('gross_B_Q_all_measurable_corporate','gross_u'),('current_numerator','num0'),
        ('current_denominator','den0'),('lagnews_numerator','numL'),('lagnews_denominator','denL')]:
        # SQL empty aggregate NULL and explicit empty-set mass zero differ only
        # for missing-score intervals; compare the defined common contribution set.
        mask=d.common_E_positions.gt(0)
        check('SQL_NTS_'+actual,np.allclose(d.loc[mask,actual],d.loc[mask,expected],rtol=1e-10,atol=1e-6,equal_nan=True))
    eligible=d.nts_pair_eligible.eq(True)
    check('SQL_NTS_current_score',np.allclose(d.loc[eligible,'NTS_Q_current'],(d.num0/d.den0).loc[eligible],rtol=1e-10,atol=1e-11))
    check('SQL_NTS_lagnews_score',np.allclose(d.loc[eligible,'NTS_Q_lagnews'],(d.numL/d.denL).loc[eligible],rtol=1e-10,atol=1e-11))
    check('SQL_NTS_coverage_current',np.allclose(d.loc[eligible,'news_trade_coverage_current'],(d.covered0/d.gross_u).loc[eligible],rtol=1e-10,atol=1e-11))
    check('SQL_NTS_coverage_lagnews',np.allclose(d.loc[eligible,'news_trade_coverage_lagnews'],(d.coveredL/d.gross_u).loc[eligible],rtol=1e-10,atol=1e-11))
    check('two_contributing_companies_and_coverage',((d.loc[eligible,'companies0']>=2)&(d.loc[eligible,'companiesL']>=2)
        &(d.loc[eligible,'news_trade_coverage_current']>=.5)&(d.loc[eligible,'news_trade_coverage_lagnews']>=.5)).all())
    origins=pd.read_parquet(run_root/'origin_eligibility.parquet')
    check('unique_fund_closing_month',not origins.duplicated(['fund_id','origin_month']).any())
    monthly=pd.read_parquet(run_root/'monthly_returns_adapter.parquet')
    factors=pd.read_parquet(run_root/'monthly_factors_adapter.parquet')
    alpha_root=run_root/'alpha/UNIT_VALUE_KNOWN_ADJUSTMENTS/HISTORY60_MIN36'
    alpha=pd.read_parquet(alpha_root/'alpha.parquet')
    check('unique_fund_interval_horizon_alpha',not alpha.duplicated(['fund_id','interval_id','horizon']).any())
    good=alpha.alpha.notna()
    check('alpha_future_horizon_complete',alpha.loc[good,'future_months_complete'].eq(alpha.loc[good,'horizon']).all())
    check('alpha_future_anchor_after_origin',(pd.to_datetime(alpha.loc[good,'future_anchor_date'])>=pd.to_datetime(alpha.loc[good,'origin_date'])).all())
    check('alpha_history_36_and_recent9',alpha.loc[good,'n_past'].ge(36).all() and alpha.loc[good,'n_past_recent'].ge(9).all())
    one=alpha.loc[alpha.horizon.eq(1),['fund_id','interval_id','origin_date','training_id']].drop_duplicates()
    past_violations=0;past_rows=0
    for file in sorted(alpha_root.glob('chunk_*.training.parquet')):
        m=pd.read_parquet(file)
        if not len(m):continue
        m=m.merge(one,on=['fund_id','interval_id','training_id'],how='left',validate='many_to_one')
        m=m.merge(monthly[['fund_id','month','start_valuation_date','end_valuation_date']],on=['fund_id','month'],how='left',validate='many_to_one')
        month_end=pd.PeriodIndex(m.month,freq='M').end_time.normalize()
        date=pd.to_datetime(m.origin_date)
        violation=(month_end>date)|(pd.to_datetime(m.end_valuation_date)>date)|(pd.to_datetime(m.start_valuation_date)>date)
        past_violations+=int(violation.sum());past_rows+=len(m)
    check('every_training_source_date_at_or_before_origin',past_violations==0,training_rows=past_rows,violations=past_violations)
    models=pd.read_csv(run_root/'models.csv')
    check('all_predeclared_model_labels',len(models)==114 and not models.model_id.duplicated().any(),labels=len(models))
    for group in ['pooled','equity','bond','mixed']:
        for horizon in [1,2,3]:
            pair=models.loc[models.branch.eq('UNIT_VALUE_KNOWN_ADJUSTMENTS')&models.sensitivity.eq('PRIMARY')
                &models.group.eq(group)&models.horizon.eq(horizon)]
            check(f'identical_current_and_distributed_rows_{group}_h{horizon}',len(pair)==2 and pair.sample_hash.nunique()==1)
    audit_path=run_root/'frozen_sensitivity_audit.json'
    if audit_path.exists():
        audit=json.loads(audit_path.read_text(encoding='utf-8'))
        check('frozen_sensitivity_families_complete',audit['status']=='COMPLETE_ALL_FROZEN_LABELS_ACCOUNTED_FOR'
            and len(audit['families'])==6 and sum(x['actual_labels'] for x in audit['families'])==114)
        check('TR_only_inapplicable_cells_have_concrete_reason',all(
            x['actual_labels']==x['expected_labels'] and (x['family']=='PRIMARY' or x['estimated_labels']>0
            or 'No TR_VERIFIED' in x['reason_not_estimated']) for x in audit['families']))
    if summary.get('fund_release')=='v3_action_aware':
        known=monthly.known_unresolved_cash_event.eq(True)
        check('no_known_unresolved_event_month_used',not monthly.loc[known,'diagnostic_eligible'].any())
        check('known_event_month_flags_retained',monthly.known_unresolved_cash_event.equals(monthly.known_distribution_date_unresolved.astype(bool)))
        previous=Path(summary['previous_run'])
        old=json.loads((previous/'summary.json').read_text(encoding='utf-8'))
        check('v1_and_v3_identical_NTS_input',old['input_manifest']['nts_primary']['sha256']==summary['input_manifest']['nts_primary']['sha256'])
        check('v1_and_v3_identical_frozen_spec',old['spec_sha256']==summary['spec_sha256'])
        check('v1_seal_preserved',sha(previous/'manifest.json')=='cef0c99b76becb44964b9492226cdefeb25481af2ae95527da87a5b2325bf5c7')
        comparison=pd.read_csv(run_root/'comparison_v1_v3_models.csv')
        check('all_comparison_cells_accounted_with_phase',len(comparison)==114 and comparison.model_id_v1.notna().all()
              and comparison.model_id_v3.notna().all())
    tr=models.branch.eq('TR_VERIFIED')
    check('no_UNIT_diagnostic_relabelled_TR',models.loc[tr,'status'].eq('NOT_ESTIMABLE').all() if summary['monthly_returns']['TR_verified']==0 else True)
    replications=[]
    for group in ['pooled','equity','bond','mixed']:
        for h in [1,3]:
            match=models.loc[models.branch.eq('UNIT_VALUE_KNOWN_ADJUSTMENTS')&models.sensitivity.eq('PRIMARY')
                &models.group.eq(group)&models.horizon.eq(h)&models.variant.eq('CURRENT_PLUS_LAGNEWS')]
            if len(match)!=1:raise AssertionError('Missing prescribed group/horizon model')
            model=match.iloc[0];directory=run_root/'regression/UNIT_VALUE_KNOWN_ADJUSTMENTS/PRIMARY'/model.model_id
            info=json.loads((directory/'fit.json').read_text(encoding='utf-8'))
            if info['status']!='ESTIMATED':
                replications.append(dict(group=group,horizon=h,status='NOT_ESTIMABLE',reason=info.get('reason')));continue
            sample=pd.read_parquet(directory/'used_sample.parquet')
            terms=info['terms'];funds=pd.Categorical(sample.fund_id.astype(str));g=len(funds.categories);n=len(sample)
            # Full set of fund intercepts, without an extra intercept, provides
            # a separate full-dummy design rather than production demeaning.
            dummy=np.zeros((n,g));dummy[np.arange(n),funds.codes]=1.
            x=np.column_stack([sample[terms].to_numpy(float),dummy]);y=sample.alpha.to_numpy(float)
            beta=np.linalg.lstsq(x,y,rcond=None)[0];resid=y-x@beta
            bread=np.linalg.inv(x.T@x);scores=np.zeros((g,x.shape[1]));np.add.at(scores,funds.codes,x*resid[:,None])
            correction=g/(g-1)*(n-1)/(n-x.shape[1]);cov=correction*(bread@scores.T@scores@bread)
            coeff=pd.read_csv(directory/'coefficients_main_tGminus1.csv').set_index('term')
            actual=pd.read_csv(directory/'full_covariance.csv',index_col=0).loc[terms,terms].to_numpy(float)
            slope_error=float(np.max(np.abs(beta[:len(terms)]-coeff.loc[terms,'beta'].to_numpy(float))))
            cov_error=float(np.max(np.abs(cov[:len(terms),:len(terms)]-actual)))
            check(f'full_dummy_OLS_{group}_h{h}',np.allclose(beta[:len(terms)],coeff.loc[terms,'beta'],rtol=1e-6,atol=1e-9),max_error=slope_error)
            check(f'full_dummy_CR1_{group}_h{h}',np.allclose(cov[:len(terms),:len(terms)],actual,rtol=1e-5,atol=1e-10),max_error=cov_error)
            p=2*stats.t.sf(np.abs(coeff.loc[terms,'beta']/coeff.loc[terms,'SE']),g-1)
            check(f'fund_cluster_t_reference_{group}_h{h}',np.allclose(p,coeff.loc[terms,'p'],rtol=1e-8,atol=1e-10))
            replications.append(dict(group=group,horizon=h,status='REPLICATED',slope_max_abs_error=slope_error,
                                     covariance_max_abs_error=cov_error,nobs=n,funds=g))
    result=dict(status='PASS' if all(x['passed'] for x in checks) else 'FAIL',scope='Executor second-path numerical QA; independent reviewer still required',
        checked_utc=utc(),input_root=str(run_root),run_manifest_sha256=sha(run_root/'manifest.json'),
        checks_pass=sum(x['passed'] for x in checks),checks_fail=sum(not x['passed'] for x in checks),checks=checks,
        numerical_replications=replications,code_sha256=sha(__file__))
    save_json(qa_root/'qa.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['checks','numerical_replications']},ensure_ascii=False))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root');a=p.parse_args()
    root=a.run_root or json.loads((ROOT/'runtime/alpha_ready.json').read_text(encoding='utf-8'))['output']
    result=run(root)
    if result['checks_fail']:raise SystemExit(2)
