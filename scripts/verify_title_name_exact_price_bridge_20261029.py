"""Read-only completion QA for the finite exact price bridge; never refits."""
from pathlib import Path
import argparse
import hashlib,json
import numpy as np
import pandas as pd
from scipy import stats

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('./analysis_outputs/title_name_exact_price_bridge_20261029'))
ROOT=p.parse_args().output
BASE=Path('./analysis_outputs/title_name_regressions_20261026_w2_v2')
KEYS=['fund_id','security_id','start_date','end_date']
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
def frame_hash(frame,columns):return hashlib.sha256(pd.util.hash_pandas_object(frame[columns],index=False).values.tobytes()).hexdigest()
out=json.loads((ROOT/'summary.json').read_text());DATA=Path(out.get('data_root',str(ROOT)));manifest=out['manifest'];checks={};errors=[]
checks['completed']=out['status']=='EXACT_PRICE_BRIDGE_COMPLETED'
checks['twenty_cells_seventy_two_slots']=len(out['records'])==20 and len(out['tests'])==72
checks['plan_and_gate_hashes']=sha(DATA/'config/plan.json')==manifest['plan_sha256'] and sha(DATA/'config/method_review.json')==manifest['method_gate_sha256']
checks['frozen_code_unchanged']=all(sha(DATA/'code'/name)==digest for name,digest in manifest['code_hashes'].items())
checks['input_contents_unchanged']=all(sha(path)==digest for path,digest in manifest['source_hashes'].items())
checks['baseline_modules_unchanged']=all(sha(BASE/'code'/name)==digest for name,digest in manifest['base_code_hashes'].items())
quote_sources=json.loads((DATA/'quote_source_hashes.json').read_text())
checks['quote_source_hashes']=all(sha(path)==digest for path,digest in quote_sources.items())
checks['no_S_or_LLM_rerun']=out['no_new_S_fit'] and out['no_new_LLM_run'] and not out['new_external_data']
source=pd.read_parquet('./analysis_outputs/title_name_hypothesis_research_20261029/exact_actual_window_panel.parquet',columns=KEYS+['S'])
frames={'full_portfolio_AT':pd.read_parquet(DATA/'full_AT_common_final_sample.parquet'),
        'relative_position_growth':pd.read_parquet(DATA/'relative_common_final_sample.parquet')}
for family,frame in frames.items():
    info=out['group_validation'][family]['common_sample']
    checks[family+'_saved_sample_hash']=frame_hash(frame,KEYS)==info['sample_hash']
    checks[family+'_unique_keys']=not frame.duplicated(KEYS).any()
    bind=frame[KEYS+['S']].merge(source,on=KEYS,validate='one_to_one',suffixes=('_actual','_sealed'))
    checks[family+'_sealed_S_identical']=len(bind)==len(frame) and np.array_equal(bind.S_actual.to_numpy(),bind.S_sealed.to_numpy())
    selected=[r for r in out['records'] if r['family']==family or (family=='relative_position_growth' and r['family']=='price_component_diagnostic')]
    checks[family+'_all_outcomes_final_sample_same']=all(r['N']==len(frame) and r['sample_hash']==info['sample_hash'] for r in selected)
    estimated=[r for r in selected if r['status'].startswith('ESTIMATED')]
    checks[family+'_same_X_and_FE']=not estimated or (len({r['common_X_hash'] for r in estimated})==1 and len({tuple(r['regressors']) for r in estimated})==1 and len({tuple(r['FE']) for r in estimated})==1)
logs=frames['relative_position_growth']
checks['log_outcome_decomposition']=bool(np.allclose(logs.log_value_growth,logs.price_adjusted_value_growth+logs.log_price_growth,atol=1e-12,rtol=1e-12))
checks['log_value_and_price_formula']=bool(np.allclose(logs.log_value_growth,np.log(logs.reported_value_close/logs.reported_value_open),atol=1e-12) and np.allclose(logs.log_price_growth,np.log(logs.price_close/logs.price_open),atol=1e-12))
checks['continuing_positive_only']=bool(logs.reported_value_open.gt(0).all() and logs.reported_value_close.gt(0).all())
pairs=pd.read_parquet(DATA/'accepted_price_pairs.parquet')
checks['no_future_or_stale_prices']=bool(pairs.quote_open_date.le(pairs.start_date).all() and pairs.quote_close_date.le(pairs.end_date).all() and pairs.quote_open_age_days.between(0,7).all() and pairs.quote_close_age_days.between(0,7).all())
checks['unique_price_pairs']=not pairs.duplicated(['security_id','start_date','end_date']).any()
cov=pd.read_csv(DATA/'price_coverage.csv')
eligible=cov.loc[cov.eligible_full_price_AT]
checks['full_denominator_coverage_gate']=bool(eligible.uncovered_weight.le(1e-12).all() and eligible.opening_weight.sub(1).abs().le(1e-8).all() and eligible.closing_weight.sub(1).abs().le(1e-8).all() and eligible.bh_denominator.gt(0).all())
full=frames['full_portfolio_AT']
if len(full):
    joined=full.merge(pairs[['security_id','start_date','end_date','price_return']],on=['security_id','start_date','end_date'],how='left',validate='many_to_one').merge(cov[['interval_id','bh_denominator']],on='interval_id',validate='many_to_one')
    growth=np.where(joined.weight_open.gt(1e-12),joined.weight_open*(1+joined.price_return),0)
    checks['E1_target_full_denominator_formula']=bool(np.allclose(joined.AT_PRICE,joined.weight_close-growth/joined.bh_denominator,atol=1e-12,rtol=1e-12))
for r in out['records']:
    if not r['status'].startswith('ESTIMATED'):continue
    df=r['inference_df'];critical=stats.t.ppf(.975,df)
    for row in r['coefficients']:
        if row['inference_status']!='ESTIMABLE':
            checks[r['cell']+'_'+row['term']+'_missing_invalid_inference']=all(row[k] is None for k in ['se','p','ci95_low','ci95_high']);continue
        errors += [abs(row['p']-2*stats.t.sf(abs(row['beta']/row['se']),df)),abs(row['ci95_low']-row['beta']+critical*row['se']),abs(row['ci95_high']-row['beta']-critical*row['se'])]
        scale=100 if r['family']!='full_portfolio_AT' else 10000
        errors.append(abs(row['effect_display']-scale*row['beta']))
    if r['H1'] and r['H1']['status']=='ESTIMATED':errors.append(abs(r['H1']['intersection_union_p']-max(r['H1']['one_sided'].values())))
    if r['H2'] and r['H2']['status']=='ESTIMATED':
        h=r['H2'];errors.append(abs(h['p_abs']-max(h['sum_restriction']['p'],h['difference_restriction']['p'])))
checks['coefficient_inference_and_units_arithmetic']=not errors or max(errors)<1e-10
decomp=out['group_validation']['relative_position_growth'].get('coefficient_decomposition')
checks['all_regressor_beta_decomposition']=decomp is not None and decomp['status']=='PASS' and decomp['maximum_absolute_coefficient_error']<1e-10
def check_holm(tests,key):
    bound=0.;ok=True
    for rank,test in enumerate(sorted(tests,key=lambda t:1 if t['p'] is None else t['p'])):
        bound=max(bound,min(1.,(len(tests)-rank)*(1 if test['p'] is None else test['p'])))
        ok &= test[key] is None if test['p'] is None else abs(bound-test[key])<1e-12
    return bool(ok)
for family,size in [('full_portfolio_AT',32),('relative_position_growth',32),('price_component_diagnostic',8)]:
    tests=[t for t in out['tests'] if t['family']==family]
    checks[family+'_fixed_family_Holm']=len(tests)==size and check_holm(tests,'p_holm_fixed_family')
checks['all_72_Holm']=check_holm(out['tests'],'p_holm_all_72_locked_tests')
checks['cell_Holm4']=all(check_holm(r['tests'],'p_holm_four_tests_in_cell') for r in out['records'] if r['family']!='price_component_diagnostic' and r['status'].startswith('ESTIMATED'))
for kind,keys,fields in [('coefficients',['cell','term'],['beta','se','ci95_low','ci95_high','p']),('tests',['cell','test'],['p','p_holm_fixed_family','p_holm_all_72_locked_tests'])]:
    rows=([v for r in out['records'] for v in r['coefficients']] if kind=='coefficients' else out['tests'])
    a=pd.DataFrame(rows);b=pd.read_csv(ROOT/(kind+'.csv'))
    m=a.merge(b,on=keys,validate='one_to_one',suffixes=('_json','_csv'))
    checks[kind+'_CSV_matches_JSON']=len(a)==len(b)==len(m) and all(np.allclose(m[f+'_json'].astype(float),m[f+'_csv'].astype(float),atol=1e-12,rtol=1e-12,equal_nan=True) for f in fields)
result=dict(status='PASS' if all(checks.values()) else 'FAIL',checks={k:bool(v) for k,v in checks.items()},check_count=len(checks),
    max_inference_arithmetic_error=max(errors) if errors else None,source_files_verified=len(manifest['source_hashes']),quote_files_verified=len(quote_sources),
    full_AT_common_rows=len(full),relative_common_rows=len(logs),summary_sha256=sha(ROOT/'summary.json'),
    csv_sha256={name:sha((ROOT if name in ['coefficients.csv','tests.csv'] else DATA)/name) for name in ['coefficients.csv','tests.csv','sample_flow.csv','price_coverage.csv']},
    coefficient_decomposition=decomp,only_read_checks=True,no_new_fits=True)
print(json.dumps(result,allow_nan=False))
raise SystemExit(0 if all(checks.values()) else 1)
