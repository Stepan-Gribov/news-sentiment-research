"""Build frozen H3 paired NTS on research-host; inputs remain immutable.

All bulk joins and contribution files stay on the PC. The stdout contains only
aggregate QA/progress. Does not calculate a security-level alpha or news labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from h3_alpha_core_20261104 import paired_nts_q, numeric, truth, unique, digest_lines
from h3_alpha_io_20261104 import ROOT, sha, save_json, checkpoint, file_record, seal_directory

BASE = ROOT.parent
SPEC_SHA = '565a726fd3858b94b2bfb3dbf43442d8cab4812c79749fd2ea24db806c44e811'


def select_origins(calendar):
    """Latest genuine closing date, before any outcome or NTS gate; no fallback."""
    c = calendar.copy()
    unique(c, ['fund_id', 'interval_id'], 'full calendar')
    c['end_date'] = pd.to_datetime(c.end_date)
    c['start_date'] = pd.to_datetime(c.start_date)
    c['origin_month'] = c.end_date.dt.to_period('M').astype(str)
    latest = c.groupby(['fund_id', 'origin_month'], observed=True).end_date.transform('max')
    c['latest_date_in_month'] = c.end_date.eq(latest)
    conflicts = c.loc[c.latest_date_in_month].groupby(['fund_id', 'origin_month'], observed=True).interval_id.transform('size')
    c['same_date_conflict'] = False
    c.loc[conflicts.index, 'same_date_conflict'] = conflicts.gt(1)
    c['selected_origin'] = c.latest_date_in_month & ~c.same_date_conflict
    c['origin_selection_reason'] = np.select([c.same_date_conflict, ~c.latest_date_in_month],
        ['conflicting_latest_same_date_reports', 'earlier_report_same_month'], default='selected_latest')
    return c.loc[c.selected_origin].copy(), c


def join_lag(frame, mapping):
    unique(frame, ['row_id'], 'inherited panel')
    unique(mapping, ['row_id'], 'frozen lag map')
    if set(frame.row_id) != set(mapping.row_id):
        raise AssertionError('Frozen lag map does not cover the exact inherited row set')
    m = mapping.set_index('row_id').reindex(frame.row_id).reset_index()
    for a, b in [('fund_id', 'fund_id'), ('company_id', 'company_id'), ('interval_id', 'current_interval_id')]:
        if not np.array_equal(frame[a].astype(str), m[b].astype(str)):
            raise AssertionError('Frozen current/lag identity mismatch: ' + a)
    for a, b in [('start_date', 'current_start_date'), ('end_date', 'current_end_date')]:
        if not np.array_equal(pd.to_datetime(frame[a]).to_numpy(), pd.to_datetime(m[b]).to_numpy()):
            raise AssertionError('Frozen current/lag dates mismatch: ' + a)
    prev = m.lag_interval_id.notna()
    if not np.array_equal(pd.to_datetime(m.loc[prev, 'lag_end_date']).to_numpy(dtype='datetime64[ns]'),
            pd.to_datetime(frame.loc[prev.to_numpy(), 'start_date']).to_numpy(dtype='datetime64[ns]')):
        raise AssertionError('Previous company-news interval does not end at current opening')
    if not m.loc[prev, 'lag_closing_report_id'].astype(str).reset_index(drop=True).equals(
            frame.loc[prev.to_numpy(), 'opening_report_id'].astype(str).reset_index(drop=True)):
        raise AssertionError('Previous news report chain mismatch')
    d = frame.copy()
    d['net_sentiment_lag1'] = numeric(m.net_sentiment_lag1).to_numpy()
    d['lag_eligible_primary'] = truth(m.lag_eligible_primary).to_numpy()
    d['lag_reason'] = m.lag_reason.to_numpy()
    for name in ['lag_interval_id', 'lag_company_period_id', 'lag_start_date', 'lag_end_date']:
        d[name] = m[name].to_numpy()
    if (d.net_sentiment_lag1.notna() & ~d.lag_eligible_primary).any():
        raise AssertionError('Ineligible lag score is numeric')
    return d


def quantity_contributions(u, actions, maturities):
    u = u.copy()
    for c in ['Q0', 'Q1', 'P0', 'P1', 'V0', 'V1', 'age0', 'age1']:
        u[c] = numeric(u[c])
    for c in ['quote0', 'quote1', 'start_date', 'end_date']:
        u[c] = pd.to_datetime(u[c])
    for suffix, full in [('0', 'opening_full_portfolio'), ('1', 'closing_full_portfolio')]:
        observed = truth(u['observed' + suffix])
        absent_zero = ~observed & truth(u[full])
        u['absence_zero' + suffix] = absent_zero
        u.loc[absent_zero, 'Q' + suffix] = 0.
        u.loc[absent_zero, 'V' + suffix] = 0.
        q = u['Q' + suffix]; p = u['P' + suffix]; v = u['V' + suffix]
        quote_ok = p.gt(0) & np.isfinite(p) & u['age' + suffix].between(0, 7)
        quote_ok &= u['quote' + suffix].le(u['start_date' if suffix == '0' else 'end_date'])
        quote_ok &= u['currency' + suffix].fillna('').str.upper().isin(['RUB', 'RUR', 'SUR'])
        value_ok = np.isfinite(v) & (v - q * p).abs().le(1e-7 + 1e-10 * v.abs())
        u['endpoint_valid' + suffix] = np.isfinite(q) & q.ge(0) & ((q.gt(0) & quote_ok & value_ok)
            | (q.eq(0) & (absent_zero | (observed & v.eq(0)))))
    u['closing_to_opening_quantity_ratio'] = 1.
    u['known_split_adjusted'] = False
    u['known_action_ambiguous_transition'] = False
    u['known_action_source'] = ''
    for event in actions['events']:
        match = u['isin'].eq(event['isin'])
        threshold = pd.Timestamp(event['post_quote_from'])
        crosses = match & u.quote0.lt(threshold) & u.quote1.ge(threshold) & u.Q0.gt(0) & u.Q1.gt(0)
        u.loc[crosses, 'closing_to_opening_quantity_ratio'] *= event['closing_to_opening_quantity_ratio']
        u.loc[crosses, 'known_split_adjusted'] = True
        u.loc[crosses, 'known_action_source'] = event['source_url']
        if event.get('event_date'):
            # A report in the halt/legal-transition gap uses a stale old-unit
            # quote. Do not assert that its derived share count is comparable.
            effective = pd.Timestamp(event['event_date'])
            ambiguous = match & (u.start_date.between(effective, threshold, inclusive='left')
                                 | u.end_date.between(effective, threshold, inclusive='left'))
            u.loc[ambiguous, 'known_action_ambiguous_transition'] = True
    u['Q1_opening_units'] = u.Q1 * u.closing_to_opening_quantity_ratio
    valid = truth(u.endpoint_valid0) & truth(u.endpoint_valid1)
    valid &= ~truth(u.identity_conflict) & ~truth(u.class_conflict) & ~u.known_action_ambiguous_transition
    u['confirmed_entry'] = u.Q0.eq(0) & u.Q1.gt(0) & truth(u.opening_full_portfolio)
    u['confirmed_exit'] = u.Q0.gt(0) & u.Q1.eq(0) & truth(u.closing_full_portfolio)
    continuing = valid & u.Q0.gt(0) & u.Q1.gt(0)
    exits = valid & u.confirmed_exit
    entries = valid & u.confirmed_entry
    zeros = valid & u.Q0.eq(0) & u.Q1.eq(0)
    u['B_Q'] = np.nan
    u.loc[continuing | exits, 'B_Q'] = ((u.Q1_opening_units - u.Q0) * u.P0).loc[continuing | exits]
    u.loc[entries, 'B_Q'] = (u.Q1 * u.P1).loc[entries]
    u.loc[zeros, 'B_Q'] = 0.
    u['quantity_missing_reason'] = np.select([u.identity_conflict, u.class_conflict,
        u.known_action_ambiguous_transition, ~truth(u.endpoint_valid0), ~truth(u.endpoint_valid1),
        u.B_Q.isna()], ['issuer_identity_conflict', 'asset_class_conflict', 'known_split_transition_ambiguous',
        'opening_quantity_price_value_or_absence_unverified', 'closing_quantity_price_value_or_absence_unverified',
        'entry_or_exit_requires_complete_report'], default='measurable')
    u['scheduled_redemption_in_interval'] = False
    events = maturities[['security_id', 'scheduled_maturity_date']].copy()
    events['scheduled_maturity_date'] = pd.to_datetime(events.scheduled_maturity_date)
    candidates = u[['security_id','start_date','end_date']].assign(_position_index=u.index).merge(
        events, on='security_id', how='inner', validate='many_to_many')
    affected = candidates.loc[candidates.start_date.lt(candidates.scheduled_maturity_date)
        & candidates.end_date.ge(candidates.scheduled_maturity_date), '_position_index'].unique()
    u.loc[affected, 'scheduled_redemption_in_interval'] = True
    u['eligible_for_score'] = ~u.scheduled_redemption_in_interval
    u['corporate_action_history_complete'] = False
    return u


def run(spec_path, lag_plan_path, actions_path):
    spec_path = Path(spec_path)
    spec_record = file_record(spec_path, SPEC_SHA)
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    lag_plan = json.loads(Path(lag_plan_path).read_text(encoding='utf-8'))
    paths = {k: Path(spec['input_contract'][k]) for k in
        ['quantity_panel', 'portfolio_units', 'full_report_calendar', 'current_net', 'lag_map', 'lag_windows']}
    paths['focus_map'] = BASE / 'h1_macro_no_time_20261031/fund_interval_classification.parquet'
    paths['scheduled_events'] = BASE / 'title_name_regressions_20261026_w2_v2/scheduled_corporate_events.parquet'
    expected = lag_plan['expected_source_sha256']
    alias = dict(quantity_panel='input_panel', full_report_calendar='interval_source', current_net='net_source')
    checkpoint('PINNING_NTS_INPUT_HASHES')
    inputs = {k: file_record(p, expected.get(alias.get(k, k))) for k, p in paths.items()}
    inputs.update(spec=spec_record, actions=file_record(actions_path), code=file_record(__file__),
                  core=file_record(Path(__file__).with_name('h3_alpha_core_20261104.py')))
    version = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:16]
    out = ROOT / 'panels' / ('nts_' + version)
    if (out / 'manifest.json').exists():
        manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
        for r in manifest['files'].values():
            file_record(r['path'], r['sha256'])
        checkpoint('NTS_RESUMED_SEALED', output=str(out), counts=manifest['counts'])
        return out
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / 'input_manifest.json', dict(version=version, inputs=inputs))
    checkpoint('NTS_LOADING_FROZEN_TARGETS', output=str(out))
    wanted = ['row_id', 'fund_id', 'security_id', 'company_id', 'interval_id', 'company_period_id',
              'opening_report_id', 'closing_report_id', 'start_date', 'end_date', 'quantity_open',
              'quantity_close', 'net_sentiment', 'eligible_primary', 'issuer_identity_conflict', 'asset_class_conflict']
    target = pd.read_parquet(paths['quantity_panel'], columns=wanted)
    if len(target) != spec['input_contract']['quantity_panel_expected_rows_before_H3_filters']:
        raise AssertionError('Unexpected inherited source size')
    unique(target, ['fund_id', 'interval_id', 'security_id'], 'inherited target security keys')
    target = join_lag(target, pd.read_parquet(paths['lag_map']))
    # Verify the copied current scores against their separately pinned original.
    current = pd.read_parquet(paths['current_net'], columns=['company_period_id', 'net_sentiment'])
    unique(current, ['company_period_id'], 'frozen current company scores')
    verify = target[['company_period_id', 'net_sentiment']].merge(current, on='company_period_id',
        how='left', validate='many_to_one', suffixes=('', '_source'))
    if not np.allclose(numeric(verify.net_sentiment), numeric(verify.net_sentiment_source), atol=0, rtol=0, equal_nan=True):
        raise AssertionError('Current news copied into inherited panel differs from frozen source')
    calendar = pd.read_parquet(paths['full_report_calendar'])
    calendar = calendar.loc[calendar.fund_id.astype(str).isin(set(target.fund_id.astype(str)))].copy()
    calendar['fund_id'] = calendar.fund_id.astype(str)
    selected, decisions = select_origins(calendar)
    decisions.to_parquet(out / 'origin_selection_decisions.parquet', index=False, compression='zstd')
    focus = pd.read_parquet(paths['focus_map'])
    unique(focus, ['fund_id', 'interval_id'], 'report-dated focus map')
    focus['fund_id'] = focus.fund_id.astype(str)
    selected = selected.merge(focus, on=['fund_id', 'interval_id'], how='left', validate='one_to_one')
    selected['universe_eligible'] = selected.fund_focus.isin(['equity', 'bond', 'mixed']) & ~truth(selected.exclude_index_fund)
    selected['original_AT_branch_status'] = 'NOT_AVAILABLE_NO_VERIFIED_WHOLE_PORTFOLIO_AT'
    selected.to_parquet(out / 'selected_origins.parquet', index=False, compression='zstd')
    import duckdb
    con = duckdb.connect(':memory:')
    con.execute("SET threads=2; SET memory_limit='2GB'")
    con.register('selected_origins', selected)
    portfolio_literal = "'" + str(paths['portfolio_units']).replace("'", "''") + "'"
    con.execute('CREATE VIEW p AS SELECT * FROM read_parquet('+portfolio_literal+') WHERE is_reported_position=true')
    if con.execute('SELECT count(*) FROM (SELECT report_id,security_id FROM p GROUP BY ALL HAVING count(*)>1)').fetchone()[0]:
        raise AssertionError('Portfolio has duplicate original report-security positions')
    fields = ['quantity_est', 'unit_price_rub', 'position_value_rub', 'quote_date', 'quote_age_days',
              'currency', 'company_id', 'asset_class', 'market_type', 'asset_name', 'usable_weight',
              'residence_documented_by_disclosure', 'bucket_kind', 'price_isin']
    aliases = dict(quantity_est='Q', unit_price_rub='P', position_value_rub='V', quote_date='quote',
                   quote_age_days='age', currency='currency')
    endpoints = ','.join(f'{p}.{c} AS {aliases.get(c,c)}{s}' for p,s in [('a','0'),('b','1')] for c in fields)
    checkpoint('NTS_RECONSTRUCTING_ALL_UNION_HELD_POSITIONS', selected_origins=len(selected))
    query = '''WITH keys AS (
        SELECT c.interval_id,p.security_id FROM selected_origins c JOIN p ON p.report_id=c.opening_report_id
        UNION SELECT c.interval_id,p.security_id FROM selected_origins c JOIN p ON p.report_id=c.closing_report_id)
        SELECT c.fund_id,c.interval_id,c.opening_report_id,c.closing_report_id,c.start_date,c.end_date,
        c.opening_full_portfolio,c.closing_full_portfolio,k.security_id,
        a.report_id IS NOT NULL AS observed0,b.report_id IS NOT NULL AS observed1,
        coalesce(a.company_id,b.company_id) company_id,coalesce(a.asset_class,b.asset_class) asset_class,
        coalesce(a.market_type,b.market_type) market_type,coalesce(a.price_isin,b.price_isin) isin,
        a.company_id IS NOT NULL AND b.company_id IS NOT NULL AND a.company_id<>b.company_id identity_conflict,
        a.asset_class IS NOT NULL AND b.asset_class IS NOT NULL AND a.asset_class<>b.asset_class class_conflict,
        ''' + endpoints + ''' FROM keys k JOIN selected_origins c USING(interval_id)
        LEFT JOIN p a ON a.report_id=c.opening_report_id AND a.security_id=k.security_id
        LEFT JOIN p b ON b.report_id=c.closing_report_id AND b.security_id=k.security_id'''
    all_positions = con.execute(query).fetchdf()
    unique(all_positions, ['fund_id', 'interval_id', 'security_id'], 'union-held positions')
    all_positions['corporate_candidate'] = all_positions.asset_class.isin(['equity', 'bond']) & ~all_positions.market_type.isin(
        ['ofz_bond', 'subfederal_bond', 'municipal_bond', 'state_bond', 'ifi_bond', 'etf_ppif', 'public_ppif'])
    all_positions['unknown_market_classification'] = all_positions.market_type.isna()
    # Value-coverage denominator is all nonnegative disclosed positions, including
    # unknown/unallocated/noncorporate buckets. It is NOT silently renamed NAV.
    totals = []
    for (fund, interval), g in all_positions.groupby(['fund_id', 'interval_id'], observed=True, sort=False):
        row = dict(fund_id=fund, interval_id=interval)
        for s, label in [('0', 'opening'), ('1', 'closing')]:
            value = numeric(g['V'+s]).where(numeric(g['V'+s]).ge(0))
            total = value.sum(min_count=1)
            row[label+'_positive_disclosed_asset_value_RUB'] = total
            row[label+'_missing_position_values'] = int(value.isna().sum())
            row[label+'_noncorporate_asset_share'] = float(value.where(~g.corporate_candidate).sum()/total) if total>0 else np.nan
            domestic = truth(g['residence_documented_by_disclosure'+s])
            row[label+'_documented_domestic_asset_share'] = float(value.where(domestic).sum()/total) if total>0 else np.nan
            row[label+'_foreign_asset_share'] = np.nan  # Not inferable from an ISIN or a false domestic flag.
            row[label+'_geography_unknown_asset_share'] = float(value.where(~domestic).sum()/total) if total>0 else np.nan
        totals.append(row)
    u = all_positions.loc[all_positions.corporate_candidate].copy()
    del all_positions
    checkpoint('NTS_CHECKING_QUANTITY_PRICE_ACTIONS', corporate_positions=len(u))
    u = quantity_contributions(u, json.loads(Path(actions_path).read_text(encoding='utf-8')),
                               pd.read_parquet(paths['scheduled_events']))
    t = target[['row_id','fund_id','interval_id','security_id','company_id','net_sentiment','net_sentiment_lag1',
        'eligible_primary','lag_eligible_primary','lag_reason','lag_interval_id','lag_company_period_id',
        'lag_start_date','lag_end_date','issuer_identity_conflict','asset_class_conflict']].rename(columns={'company_id':'target_company_id'})
    t['fund_id'] = t.fund_id.astype(str)
    t['inherited_target'] = True
    u = u.merge(t, on=['fund_id','interval_id','security_id'], how='left', validate='one_to_one')
    inherited = truth(u.inherited_target)
    expected_target_rows = target.loc[target.interval_id.isin(set(selected.interval_id)), 'row_id'].astype(str)
    if set(u.loc[inherited,'row_id'].astype(str)) != set(expected_target_rows):
        raise AssertionError('Selected inherited target rows missing from the all-corporate universe')
    mismatch = inherited & u.target_company_id.astype(str).ne(u.company_id.astype(str))
    if mismatch.any():
        raise AssertionError('Inherited and union-position company IDs differ')
    u['row_id'] = [str(old) if pd.notna(old) else 'H3U:'+hashlib.sha256((str(i)+'\t'+str(s)).encode()).hexdigest()
                   for old,i,s in zip(u.row_id,u.interval_id,u.security_id)]
    u['inherited_target'] = inherited
    u['eligible_for_score'] &= truth(u.eligible_primary) & truth(u.lag_eligible_primary)
    u['eligible_for_score'] &= ~truth(u.issuer_identity_conflict) & ~truth(u.asset_class_conflict)
    primary, u = paired_nts_q(u, selected, minimum_companies=2, minimum_coverage=.5)
    broad, _ = paired_nts_q(u, selected, minimum_companies=1, minimum_coverage=0.)
    detail = []
    for (fund, interval), g in u.groupby(['fund_id','interval_id'], observed=True, sort=False):
        measurable = np.isfinite(g.B_Q); e = truth(g.NTS_common_E)
        gross = float(g.loc[e, 'B_Q'].abs().sum())
        row = dict(fund_id=fund, interval_id=interval,
            entry_RUB_share=float(g.loc[e & g.confirmed_entry,'B_Q'].abs().sum()/gross) if gross>0 else np.nan,
            exit_RUB_share=float(g.loc[e & g.confirmed_exit,'B_Q'].abs().sum()/gross) if gross>0 else np.nan,
            bond_RUB_share=float(g.loc[e & g.asset_class.eq('bond'),'B_Q'].abs().sum()/gross) if gross>0 else np.nan,
            known_split_adjusted_positions=int(g.known_split_adjusted.sum()),
            known_redemption_excluded_positions=int(g.scheduled_redemption_in_interval.sum()),
            unknown_action_history=True)
        for s,label in [('0','opening'),('1','closing')]:
            row[label+'_measurable_Q_asset_value_RUB'] = float(numeric(g['V'+s]).where(measurable).sum())
            row[label+'_E_asset_value_RUB'] = float(numeric(g['V'+s]).where(e).sum())
        detail.append(row)
    detail = pd.DataFrame(detail)
    totals = pd.DataFrame(totals)
    for d in [primary,broad]:
        for addition in [detail,totals]:
            merged = d.merge(addition, on=['fund_id','interval_id'], how='left', validate='one_to_one')
            for c in addition.columns.difference(['fund_id','interval_id']):
                d[c] = merged[c].to_numpy()
        for label in ['opening','closing']:
            denom = d[label+'_positive_disclosed_asset_value_RUB']
            d[label+'_asset_coverage'] = (d[label+'_measurable_Q_asset_value_RUB']/denom).where(denom.gt(0))
            d[label+'_E_asset_coverage'] = (d[label+'_E_asset_value_RUB']/denom).where(denom.gt(0))
    u.to_parquet(out / 'quantity_news_contributions.parquet', index=False, compression='zstd')
    primary.to_parquet(out / 'fund_period_nts_primary.parquet', index=False, compression='zstd')
    broad.to_parquet(out / 'fund_period_nts_broad.parquet', index=False, compression='zstd')
    primary.groupby(['fund_focus','nts_pair_eligible'], dropna=False).size().rename('origins').reset_index().to_csv(
        out / 'nts_coverage_by_group.csv', index=False, encoding='utf-8-sig')
    u.groupby(['quantity_missing_reason','inherited_target'], dropna=False).size().rename('positions').reset_index().to_csv(
        out / 'quantity_gate_counts.csv', index=False, encoding='utf-8-sig')
    counts = dict(inherited_target_rows=len(target), source_funds=int(target.fund_id.nunique()),
        calendar_intervals=len(calendar), selected_origins=len(selected), discarded_origins=int((~decisions.selected_origin).sum()),
        corporate_positions=len(u), measurable_B_Q=int(u.B_Q.notna().sum()),
        common_E_positions=int(u.NTS_common_E.sum()), primary_eligible_origins=int(primary.nts_pair_eligible.sum()),
        primary_eligible_in_universe=int((primary.nts_pair_eligible & primary.universe_eligible).sum()),
        broad_eligible_origins=int(broad.nts_pair_eligible.sum()), known_split_adjustments=int(u.known_split_adjusted.sum()))
    qa = dict(status='PASS', unique_contributions=True, unique_origins=True, current_net_exact_parity=True,
              lag_report_chain_exact=True, scalar_rebasing_test='synthetic fixture', all_inputs_unchanged=True)
    for r in inputs.values():
        file_record(r['path'], r['sha256'])
    save_json(out / 'qa.json', qa)
    seal_directory(out, dict(status='SEALED_NTS_Q', inputs=inputs, counts=counts, spec_id=spec['spec_id'],
        limitations=['Unknown action histories; quantity V/P rather than observed executions.',
                     'Known domestic/unknown geography not inferred from ISIN; foreign shares unavailable.',
                     'Asset coverage is relative to positive disclosed positions, not an independently reconciled NAV.']))
    save_json(ROOT / 'runtime/nts_ready.json', dict(ready=True, output=str(out), manifest_sha256=sha(out/'manifest.json'), counts=counts))
    checkpoint('NTS_SEALED_READY', output=str(out), counts=counts)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--spec', default=str(ROOT/'code/h3_alpha_spec_20261104.json'))
    p.add_argument('--lag-plan', default=str(ROOT/'code/h1_sentiment_lag_plan_20261103.json'))
    p.add_argument('--actions', default=str(ROOT/'code/h3_security_actions_20261104.json'))
    a = p.parse_args()
    run(a.spec, a.lag_plan, a.actions)
