"""Bounded research-host H3 source availability audit; no estimators or LLM calls."""
from pathlib import Path
import datetime as dt
import hashlib
import json

import duckdb
import numpy as np
import pandas as pd

B = Path('.')
F = B/'databaseresources/funds/mac-20261008'
OLD = B/'analysis_outputs/title_name_existing_fund_history_audit_20261029/v20261029T081055Z'
OUT = B/'analysis_outputs/h23_h3_data_readiness_20261103'
PANEL = B/'analysis_outputs/h1_entries_trim_20261102/samples/AUGMENTED_P99.parquet'
RF = B/'analysis_outputs/money_market_ruonia_20261027'
IMOEX = B/'analysis_outputs/h1_macro_no_time_20261031/inputs/raw_external_2026-09-30/moex/indices/IMOEX_daily_history_2012-01-01_2026-08-31.csv'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as s:
        for z in iter(lambda:s.read(4*1024*1024),b''):h.update(z)
    return h.hexdigest()


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if x is pd.NA or x is pd.NaT:return None
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,(pd.Timestamp,dt.date,dt.datetime,Path)):return str(x)
    if isinstance(x,float) and not np.isfinite(x):return None
    return x


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    paths={'returns':F/'fund_returns_append_available.csv','raw_archive':F/'fund_holdings_raw_20261008.tar.gz',
        'unit_observations':OLD/'existing_unit_value_observations.parquet',
        'strict_returns':OLD/'strict_adjacent_month_unit_returns.parquet',
        'series_links':OLD/'fund_series_links.csv','history_coverage':OLD/'fund_history_coverage.csv',
        'nav_coverage':OLD/'nav_report_coverage.csv','rf_manifest':RF/'download_manifest.json',
        'rf_monthly':RF/'ruonia_monthly_rf.csv','panel':PANEL,'imoex_price_index':IMOEX}
    before={k:dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for k,p in paths.items()}
    assert before['returns']['sha256']=='73595dd25aafff4682f14b50c94cc55af94010590b51d6937eb1f67e36ed4c7c'
    assert before['raw_archive']['sha256']=='5c4abcaf647333788951b0583764c267a250bffee191978f702148f83d996e43'
    assert before['panel']['sha256']=='8bd9cfd2eef8668036c8c6c95c9863d8edbc1923d2c4d627d5b1306d03cfd43c'
    df=pd.read_csv(paths['returns']);df['date']=pd.to_datetime(df.date)
    df=df.sort_values(['fund_key','date']).reset_index(drop=True)
    assert not df.duplicated(['fund_key','date']).any()
    df['unit']=pd.to_numeric(df['Пай'],errors='coerce')
    df['previous_unit']=df.groupby('fund_key').unit.shift()
    df['previous_date']=df.groupby('fund_key').date.shift()
    month=df.date.dt.year*12+df.date.dt.month
    prev=df.previous_date.dt.year*12+df.previous_date.dt.month
    df['strict']=df.unit.gt(0)&df.previous_unit.gt(0)&month.sub(prev).eq(1)
    df['strict_return']=(df.unit/df.previous_unit-1).where(df.strict)
    df['existing_ret']=pd.to_numeric(df.fund_ret,errors='coerce')
    strict=df.loc[df.strict].copy();observed=df.loc[df.unit.gt(0)].copy()
    oldstrict=pd.read_parquet(paths['strict_returns'])
    parity=strict.merge(oldstrict,on=['fund_key','date'],validate='one_to_one')
    assert len(strict)==len(oldstrict)==len(parity)==1173
    assert np.allclose(parity.strict_return,parity.strict_simple_return,rtol=0,atol=1e-12)
    maximum_strict=int(strict.groupby('fund_key').size().max())
    maximum_raw=int(df.loc[df.existing_ret.notna()].groupby('fund_key').size().max())
    history=pd.read_csv(paths['history_coverage']);links=pd.read_csv(paths['series_links'])
    nav=pd.read_csv(paths['nav_coverage']);panel=pd.read_parquet(PANEL,columns=['fund_id'])
    current_funds=set(panel.fund_id.astype(str))
    links['fund_id']=pd.to_numeric(links.static_unique_fund_id,errors='coerce').astype('Int64').astype('string')
    linked=links.loc[links.fund_id.notna()]
    linked_price=linked.loc[linked.fund_key.isin(observed.fund_key)]
    linked_strict=linked.loc[linked.fund_key.isin(strict.fund_key)]
    monthly=[]
    for date,g in df.groupby(df.date.dt.to_period('M')):
        monthly.append(dict(month=str(date),rows=len(g),positive_unit_values=int(g.unit.gt(0).sum()),
            strict_observed_returns=int(g.strict.sum()),existing_nonmissing_returns=int(g.existing_ret.notna().sum())))
    manifest=json.loads((B/'databaseresources/manifest.json').read_text())
    selected=[x for x in manifest['links'] if any(t in json.dumps(x).lower() for t in
        ['fund_returns','funds_0101','fund_price','unit_value','mcftr','rgbitr','factor','ruonia'])]
    c=duckdb.connect('./local/research-data-service/research.duckdb',read_only=True,
        config={'threads':2,'memory_limit':'1GB'})
    catalog_sources=c.execute("select * from meta.sources where source_id like 'funds.%' or source_id like 'market.%'").fetchdf().to_dict('records')
    tables=c.execute("select table_schema,table_name from information_schema.tables where table_schema in ('funds','market') order by 1,2").fetchdf().to_dict('records')
    schemas={t:c.execute('describe '+t).fetchdf()[['column_name','column_type']].to_dict('records') for t in
        ['market.fund_returns','funds.reports','funds.registry','funds.portfolio_units']}
    catalog_return=c.execute('select count(*) n_rows,count(distinct fund_key) series_keys,min(try_cast(date as date)) first_date,max(try_cast(date as date)) last_date from market.fund_returns').fetchdf().to_dict('records')[0]
    quote_coverage=c.execute('select count(*) n_rows,count(distinct REQUESTED_ISIN) requested_isins,min(try_cast(TRADEDATE as date)) first_date,max(try_cast(TRADEDATE as date)) last_date from market.security_quotes').fetchdf().to_dict('records')[0]
    c.close()
    rf_manifest=json.loads(paths['rf_manifest'].read_text());rf=pd.read_csv(paths['rf_monthly'])
    assert sha(paths['rf_monthly'])==rf_manifest['files']['ruonia_monthly_rf.csv']['sha256']
    price_index=pd.read_csv(IMOEX)
    legacy_roots=['data/reference','funds_data/Data/Y','_legacy_raw/funds/Data/Y','code/dissertation_pipeline/data/reference',
        'funds_data','research_v2/data/reference','code/dissertation_pipeline/research_v2/data/reference','_legacy_raw/research_v2/data/reference']
    legacy=[dict(path=str(B/r/'funds_01012015_17052025.csv'),exists=(B/r/'funds_01012015_17052025.csv').exists()) for r in legacy_roots]
    after={k:sha(p) for k,p in paths.items()}
    assert all(after[k]==v['sha256'] for k,v in before.items())
    result=dict(status='NOT_ESTIMABLE_H3_ALPHA_ON_VERIFIED_REGISTERED_INPUTS',observed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        scope='Read-only registered catalog, resource manifest, exact registered fund CSV and previously audited immutable source artifacts; no whole-disk search or new raw workbook scan.',
        execution_host='research-host',regressions_run=False,new_LLM=False,source_inputs_changed=False,source_fingerprints=before,
        accepted_user_assumptions=dict(source='config/title_name_assumptions_20261027.json',
            security_returns='User accepts omitted unobserved distributions/coupons and price-based adjustment; this does not supply missing fund price history.',
            cash='Identified idle cash and deposits without disclosed income may have zero interest; missing security quotes, unknown noncash and FX are not zero.',
            controls='Additional full control requirements dropped; available reduced/macro controls retained.',
            RUONIA='Available money-market risk-free reference, not automatically earned by fund cash.',
            history_requirement_not_waived='36 past monthly returns within60 and9 within12; complete future1-3 months; dated fund-series identity.'),
        observed_fund_source=dict(path=str(paths['returns']),rows=len(df),series_keys=int(df.fund_key.nunique()),
            first_date=df.date.min(),last_date=df.date.max(),positive_unit_value_rows=len(observed),
            positive_unit_value_series=int(observed.fund_key.nunique()),first_positive_unit_date=observed.date.min(),last_positive_unit_date=observed.date.max(),
            strict_observed_adjacent_returns=len(strict),strict_return_series=int(strict.fund_key.nunique()),
            strict_return_first_date=strict.date.min(),strict_return_last_date=strict.date.max(),
            max_observed_unit_levels_per_series=int(observed.groupby('fund_key').size().max()),
            max_observed_returns_within60=maximum_strict,max_observed_returns_within12=maximum_strict,
            existing_nonmissing_returns=int(df.existing_ret.notna().sum()),max_existing_nonmissing_returns_per_series=maximum_raw,
            existing_returns_without_adjacent_observed_prices=int((df.existing_ret.notna()&~df.strict).sum()),
            max_raw_return_history_even_if_all_undocumented_returns_granted=maximum_raw,
            fund_series_passing36in60_and9in12=0,fund_series_passing24in36_sensitivity=0,
            strict_return_parity_with_frozen_audit=True,monthly=monthly),
        current_universe_linkage=dict(current_panel_rows=len(panel),current_funds=len(current_funds),
            source_series_with_static_unique_id=len(linked),source_static_unique_fund_ids=int(linked.fund_id.nunique()),
            source_series_dated_identity_verified=int(links.dated_series_identity_verified.eq(True).sum()),
            current_funds_with_provisional_static_return_series=len(current_funds&set(linked.fund_id)),
            current_funds_with_provisional_static_positive_unit_levels=len(current_funds&set(linked_price.fund_id)),
            current_funds_with_provisional_static_strict_returns=len(current_funds&set(linked_strict.fund_id)),
            current_funds_with_eligible_alpha_history=0,warning='Static registration suffix is a candidate identity, not a verified dated return-series crosswalk.'),
        nav_is_separate=dict(funds_with_positive_nav=int(nav.positive_nav_months.gt(0).sum()),
            funds_with36positive_nav_months=int(nav.positive_nav_months.ge(36).sum()),
            funds_with36nav_months_within60=int(nav.max_nav_months_last60.ge(36).sum()),
            can_recover_unit_value_without_outstanding_fund_units=False,
            portfolio_units_meaning='quantity_est and unit_price_rub refer to securities held by a fund; they are not fund units outstanding or NAV per fund unit.'),
        risk_free=dict(status=rf_manifest['status'],rows=len(rf),columns=rf.columns.tolist(),period=rf_manifest['monthly_period'],
            source=str(paths['rf_monthly']),hash_verified=True,convention=rf_manifest['monthly_convention']),
        factor_readiness=dict(required=['MCFTR monthly equity return','RGBITR monthly bond return','RUONIA monthly RF'],
            MCFTR_and_RGBITR_verified_sources_in_inspected_manifest_or_catalog=False,
            IMOEX_price_control_available=True,IMOEX_rows=len(price_index),IMOEX_first_date=price_index.TRADEDATE.min(),
            IMOEX_last_date=price_index.TRADEDATE.max(),IMOEX_is_not_MCFTR_total_return=True),
        registered_metadata=dict(catalog_return_counts=catalog_return,table_inventory=tables,catalog_sources=catalog_sources,
            relevant_manifest_matches=selected,relevant_schemas=schemas,exact_legacy_path_probes=legacy),
        additional_mechanism=dict(status='SEPARATE_PRICE_ONLY_DESIGN_POSSIBLE_NOT_ESTIMATED_NOT_H3_ALPHA',
            existing_accepted_DAR_formula='sign(AT)*(future security return-matched benchmark return); future window begins after closing holding date.',
            source_quote_coverage=quote_coverage,available_equity_price_benchmark=str(IMOEX),
            limitations=['Future endpoint pairs and age/board/currency validity not yet constructed for this branch.',
                'Price-only quantities and returns omit distributions; label any simplified mechanism explicitly.',
                'Suitable duration/credit-matched bond benchmark not established; do not use equity benchmark for bonds.',
                'Existing quote_pairs helper hardcodes dates2017-01-01..2026-05-29, so it needs a separately frozen future-window adaptation.',
                'No ready computed DAR outcome or accepted fund-alpha substitute has been established.']),
        exact_missing_dependencies=['A dated per-fund unit-value/return history long enough for at least36 past monthly observations in60 (currently observed maximum3; even undocumented stored returns maximum10).',
            'A verified dated fund_id-to-return-series identity.',
            'Verified compatible monthly equity and bond factor series and complete future1-3-month fund/factor/RF windows.'],
        existing_code=dict(alpha_helper='scripts/title_name_regressions_alpha.py::future_fund_alpha',
            required_returns=['fund_id','month','fund_return','return_verified'],required_factors=['month','risk_free','equity_return','bond_return','factors_verified'],
            existing_runner='scripts/run_title_name_regressions.py reports H3 unavailable; it is not a ready completed H3 fit.',
            quote_helper='scripts/hypothesis_extended_20261029.py::quote_pairs'),
        conclusion_ru='H3 о будущей alpha фонда пока нельзя оценить: в наблюдаемых данных максимум3 месячных доходности на серию вместо36. ИзменениеNAV, доходность отдельной бумаги и новый price-only DAR не заменяют alpha. Это ограниченный аудит зарегистрированных источников, а не утверждение об отсутствии файлов во всём ПК.')
    target=OUT/'data_readiness.json'
    target.write_text(json.dumps(clean(result),ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(clean(dict(status=result['status'],report=str(target),report_sha256=sha(target),observed=result['observed_fund_source'],
        current_universe=result['current_universe_linkage'],risk_free=result['risk_free'],quote_coverage=quote_coverage)),ensure_ascii=True,allow_nan=False))


if __name__=='__main__':main()
