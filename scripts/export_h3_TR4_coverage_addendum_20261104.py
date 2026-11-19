"""Read-only extraction of actual fitted-sample coverage and existing NTS gates.
No fitting, score changes, new eligibility rules or sealed-run edits.
"""
import json
from pathlib import Path
import zipfile
import pandas as pd
from h3_alpha_io_20261104 import ROOT,sha,file_record,save_json,utc

source=ROOT/'outputs/provider_TR4_b00ab703609fb58f'
expected='af25eb9dc57288ef54d2b0839477f0ced6b9539708bd3a88a8cbd2257b8e98c3'
file_record(source/'manifest.json',expected)
destination=ROOT/'exports'/source.name/('addendum_'+sha(__file__)[:12]);destination.mkdir(parents=True,exist_ok=True)
coverage=[]
monthly=pd.read_parquet(source/'approved_monthly_provider_returns.parquet')
monthly['fund_id']=monthly.fund_id.astype(str)
for path in (source/'regression').rglob('fit.json'):
    info=json.loads(path.read_text(encoding='utf-8'))
    if info['status']!='ESTIMATED':continue
    sample=pd.read_parquet(path.parent/'used_sample.parquet')
    last=sample[['fund_id','future_last_month']].merge(
        monthly[['fund_id','month','end_valuation_date']],left_on=['fund_id','future_last_month'],right_on=['fund_id','month'],
        how='left',validate='many_to_one')
    coverage.append(dict(model_id=info['model_id'],sensitivity=info['sensitivity'],group=info['group'],horizon=info['horizon'],
        N=len(sample),funds=int(sample.fund_id.nunique()),reference_df=info['reference_df'],
        first_origin=str(pd.to_datetime(sample.end_date).min().date()),last_origin=str(pd.to_datetime(sample.end_date).max().date()),
        first_future_month=sample.future_first_month.min(),last_future_month=sample.future_last_month.max(),
        first_actual_future_anchor_date=str(pd.to_datetime(sample.future_anchor_date).min().date()),
        last_actual_future_end_date=str(pd.to_datetime(last.end_valuation_date).max().date()),
        sample_sha256=info['sample_hash'],used_sample_PC_path=str(path.parent/'used_sample.parquet'),fragile_inference=True))
coverage=pd.DataFrame(coverage).sort_values(['sensitivity','group','horizon'])
if len(coverage)!=9:raise AssertionError('Expected exactly nine saved fitted samples')
coverage.to_csv(destination/'fitted_samples_actual_date_coverage.csv',index=False,encoding='utf-8-sig')
panel_path=source/'regression/PRIMARY/fund_period_selection_panel.parquet'
panel=pd.read_parquet(panel_path)
ledger=panel.loc[panel.alpha.notna()].copy()
required=['fund_id','interval_id','horizon','end_date','alpha','past_alpha','universe_eligible','nts_pair_eligible',
    'finite_inputs','included_before_singletons','nts_current_status','nts_lagnews_status','common_E_positions',
    'company_count_current','company_count_lagnews','news_trade_coverage_current','news_trade_coverage_lagnews',
    'NTS_Q_current','NTS_Q_lagnews','common_E_hash']
missing=set(required)-set(ledger)
if missing:raise AssertionError('Missing preserved selection/gate fields '+str(missing))
ledger=ledger[required]
ledger['existing_NTS_reason']='current:'+ledger.nts_current_status.fillna('NA_UNAVAILABLE')+'|lagnews:'+ledger.nts_lagnews_status.fillna('NA_UNAVAILABLE')
ledger.to_parquet(destination/'primary68_to16_exact_ledger_PC_ONLY.parquet',index=False,compression='zstd')
flow=[]
for h,g in ledger.groupby('horizon',observed=True):
    row=dict(horizon=int(h),alpha_defined=len(g),universe_eligible=int(g.universe_eligible.eq(True).sum()),
        paired_primary_NTS_eligible=int(g.nts_pair_eligible.eq(True).sum()),
        included_before_singletons=int(g.included_before_singletons.eq(True).sum()),excluded_primary_NTS=int(g.nts_pair_eligible.ne(True).sum()))
    sample_path=next(p for p in (source/'regression/PRIMARY').rglob('fit.json')
        if (lambda d:d['group']=='pooled' and d['horizon']==h and d['variant']=='CURRENT_PLUS_LAGNEWS')(json.loads(p.read_text(encoding='utf-8'))))
    used=pd.read_parquet(sample_path.parent/'used_sample.parquet')
    row['after_existing_singleton_gate']=len(used)
    if (row['alpha_defined'],row['paired_primary_NTS_eligible'],len(used))!=(68,16,16):
        raise AssertionError('Preserved68->16 ledger does not reproduce saved sample')
    flow.append(row)
pd.DataFrame(flow).to_csv(destination/'primary68_to16_flow.csv',index=False,encoding='utf-8-sig')
ledger.groupby(['horizon','fund_id','nts_pair_eligible','existing_NTS_reason'],dropna=False).size().rename('fund_periods').reset_index().to_csv(
    destination/'primary_NTS_exact_reason_counts.csv',index=False,encoding='utf-8-sig')
ledger.groupby(['horizon','nts_current_status','nts_lagnews_status'],dropna=False).size().rename('fund_periods').reset_index().to_csv(
    destination/'primary_NTS_combined_reason_counts.csv',index=False,encoding='utf-8-sig')
tests=pd.read_csv(source/'tests.csv')
tests.to_csv(destination/'all_fitted_target_tests.csv',index=False,encoding='utf-8-sig')
coefficients=pd.read_csv(source/'coefficients.csv')
targets=coefficients.loc[coefficients.term.isin(['NTS_Q_current','NTS_Q_lagnews'])]
nominal=tests.loc[tests.p.lt(.05)].copy()
nominal.to_csv(destination/'nominal_tests_below005_no_multiplicity_adjustment.csv',index=False,encoding='utf-8-sig')
notes='''TR4 coverage addendum: extraction only, no new fit or eligibility rule.
fitted_samples_actual_date_coverage.csv describes the nine exact USED SAMPLES,
not every available alpha origin. Dates come from used_sample.parquet and the
approved published-return endpoints. Coefficients/p values are not recomputed.
primary68_to16_flow.csv and reason counts use the preserved PRIMARY selection
panel. Each horizon has68 alpha-defined fund-periods,16 paired-primary-NTS
eligible periods, and16 after existing singleton removal. The main model fails
the frozen N>=30 gate; no primary coefficient is estimated.
NTS reasons are copied verbatim from the existing current/lagnews gate fields.
Exact keyed fund-period ledger stays PC-only; its path is in addendum_manifest.
Nominal significant joint/sum tests are retained, not hidden. They coexist with
individually nonsignificant current/lag coefficients; use full covariance.
All funds share one manager. G=4 or2 and df=3 or1 are extremely fragile; no stars,
population claims or multiplicity-corrected significance are implied. With G=2
the2restriction joint covariance is not identified and joint p remains missing.
'''
(destination/'README.txt').write_text(notes,encoding='utf-8')
record=dict(status='EXTRACTED_NO_REFIT_SEALED_HASH_UNCHANGED',created_utc=utc(),source=str(source),
    source_manifest_sha256=expected,fitted_samples=9,all_current_and_lag_individual_p_ge005=bool(targets.p.ge(.05).all()),
    nominal_significant_tests=nominal.to_dict('records'),source_primary_ledger=str(panel_path),
    exact_68_to16_ledger_PC_only=str(destination/'primary68_to16_exact_ledger_PC_ONLY.parquet'),code_sha256=sha(__file__))
save_json(destination/'addendum_manifest.json',record)
bundle=destination/'TR4_coverage_addendum.zip'
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as archive:
    for p in destination.iterdir():
        if p.is_file() and p.suffix in ['.csv','.json','.txt']:archive.write(p,p.name)
file_record(source/'manifest.json',expected)
print(json.dumps(dict(output=str(destination),bundle=file_record(bundle),source_manifest_sha256=expected,flow=flow),ensure_ascii=False))
