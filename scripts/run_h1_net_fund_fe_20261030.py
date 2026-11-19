"""Local-file H1 implementation. No network, database writes or news aggregation.

Execution/access to research inputs is a separate authorization decision.
--fixture-only uses generated data exclusively and never opens the research plan.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd

PLAN_SHA256 = '2e685b1ea18c37c649ae1845c9c9a02103d09b45ef2840596141884cdc4ed575'
CONTROLS = ['opening_weight', 'closing_log_NAV', 'closing_HHI',
            'log_news_count_interval', 'log_interval_days', 'bond_indicator']
MODELS = [
    ('NET_ONLY', 'quantity_change_pct', []),
    ('NET_CONTROLS', 'quantity_change_pct', CONTROLS),
    ('ABS_DELTA_Q_CONTROLS', 'delta_quantity_units', CONTROLS),
    ('DELTA_SHARE_PP_CONTROLS', 'delta_share_pp', CONTROLS),
]
KEYS = ['fund_id', 'security_id', 'interval_id']
POSITION_FIELDS = ['report_id', 'security_id', 'fund_id', 'report_date',
    'company_id', 'target_security_candidate', 'is_countable', 'is_reported_position',
    'is_exit', 'quantity_est', 'usable_weight', 'asset_type', 'asset_name', 'fund_name',
    'quantity_status', 'unit_price_rub', 'quote_date', 'w2_static_link_used',
    'residence_documented_by_disclosure', 'isin', 'price_isin', 'source_path']
REQUIRED_POS = set(POSITION_FIELDS[:13]) | {'w2_static_link_used', 'residence_documented_by_disclosure'}
REQUIRED_INTERVAL = {'interval_id', 'fund_id', 'opening_report_id', 'closing_report_id',
    'start_date', 'end_date', 'opening_full_portfolio', 'closing_full_portfolio',
    'closing_NAV', 'closing_HHI', 'opening_passive_explicit', 'closing_passive_explicit',
    'closing_HHI_has_documented_or_unallocated_buckets'}
REQUIRED_NET = {'company_period_id', 'company_id', 'period_kind', 'period_start',
    'period_end', 'net_sentiment', 'n_relevant', 'eligible_primary', 'state'}
REQUIRED_BRIDGE = {'interval_id', 'company_id', 'company_period_id', 'fund_id',
                   'opening_report_id', 'closing_report_id'}


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, (float, np.floating)): return float(value) if math.isfinite(value) else None
    if value is pd.NA or value is pd.NaT: return None
    if isinstance(value, (Path, dt.date, dt.datetime, pd.Timestamp)): return str(value)
    return value


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def require(frame, columns, label):
    missing = sorted(set(columns) - set(frame.columns))
    if missing: raise ValueError(f'{label}: missing columns {missing}')


def unique(frame, columns, label):
    if frame.duplicated(columns).any(): raise ValueError(f'{label}: duplicate {columns}')


def truth(series):
    return series.eq(True).fillna(False).astype(bool)


def finite(series):
    return np.isfinite(pd.to_numeric(series, errors='coerce'))


def coalesce(left, right):
    return left.where(left.notna(), right)


def prepare_chunk(positions, intervals, news, bridge):
    """Union of actual endpoint securities; retain unknown closing absences."""
    require(positions, REQUIRED_POS, 'positions')
    require(intervals, REQUIRED_INTERVAL, 'intervals')
    require(news, REQUIRED_NET, 'news')
    require(bridge, REQUIRED_BRIDGE, 'bridge')
    unique(positions, ['report_id', 'security_id'], 'ledger')
    unique(intervals, ['interval_id'], 'intervals')
    unique(bridge, ['interval_id', 'company_id'], 'company/window bridge')
    unique(news, ['company_period_id'], 'news company periods')
    reported = positions.loc[truth(positions.is_reported_position)]
    opening_keys = intervals[['interval_id', 'opening_report_id']].merge(
        reported[['report_id', 'security_id']], left_on='opening_report_id',
        right_on='report_id', how='inner', validate='one_to_many')
    closing_keys = intervals[['interval_id', 'closing_report_id']].merge(
        positions[['report_id', 'security_id']], left_on='closing_report_id',
        right_on='report_id', how='inner', validate='one_to_many')
    keys = pd.concat([opening_keys[['interval_id', 'security_id']],
                      closing_keys[['interval_id', 'security_id']]], ignore_index=True).drop_duplicates()
    d = keys.merge(intervals, on='interval_id', how='left', validate='many_to_one')
    for prefix, frame, report_key in [('open', reported, 'opening_report_id'),
                                       ('close', positions, 'closing_report_id')]:
        renamed = frame.rename(columns={c: prefix + '__' + c for c in frame.columns if c != 'security_id'})
        d = d.merge(renamed, left_on=[report_key, 'security_id'],
                    right_on=[prefix + '__report_id', 'security_id'],
                    how='left', validate='many_to_one')
        observed = d[prefix + '__report_id'].notna()
        if not d.loc[observed, prefix + '__fund_id'].astype(str).eq(d.loc[observed, 'fund_id'].astype(str)).all():
            raise ValueError('Endpoint report belongs to another fund')
        endpoint_date = 'start_date' if prefix == 'open' else 'end_date'
        if not pd.to_datetime(d.loc[observed, prefix + '__report_date']).eq(
                pd.to_datetime(d.loc[observed, endpoint_date])).all():
            raise ValueError('Endpoint report date does not match actual interval')
    d['opening_position_observed'] = d.open__report_id.notna()
    d['closing_position_present'] = d.close__report_id.notna()
    d['closing_position_reported'] = truth(d.close__is_reported_position)
    d['certified_exit'] = truth(d.close__is_exit)
    if (d.certified_exit & ~truth(d.closing_full_portfolio)).any():
        raise ValueError('An inferred zero exit lacks a complete closing report')
    d['unknown_opening_absence'] = ~d.opening_position_observed & ~truth(d.opening_full_portfolio)
    d['unknown_closing_absence'] = ~d.closing_position_present & ~truth(d.closing_full_portfolio)
    d['issuer_identity_conflict'] = (d.open__company_id.notna() & d.close__company_id.notna()
        & d.open__company_id.astype(str).ne(d.close__company_id.astype(str)))
    d['company_id'] = coalesce(d.close__company_id, d.open__company_id)
    d.loc[d.issuer_identity_conflict, 'company_id'] = None
    d['target_security_candidate'] = (
        (~d.opening_position_observed | truth(d.open__target_security_candidate))
        & (~d.closing_position_present | truth(d.close__target_security_candidate)))
    d['asset_type'] = coalesce(d.close__asset_type, d.open__asset_type)
    d['asset_class_conflict'] = (d.open__asset_type.notna() & d.close__asset_type.notna()
                                & d.open__asset_type.ne(d.close__asset_type))
    for field in ['fund_name', 'asset_name', 'isin', 'price_isin']:
        if 'open__' + field in d and 'close__' + field in d:
            d[field] = coalesce(d['close__' + field], d['open__' + field])
    d['quantity_open'] = pd.to_numeric(d.open__quantity_est, errors='coerce')
    d['quantity_close'] = pd.to_numeric(d.close__quantity_est, errors='coerce')
    d['opening_weight'] = pd.to_numeric(d.open__usable_weight, errors='coerce')
    d['closing_weight'] = pd.to_numeric(d.close__usable_weight, errors='coerce')
    known_open_absence = (~d.opening_position_observed & truth(d.opening_full_portfolio)
                          & truth(d.close__is_countable))
    known_close_absence = (~d.closing_position_present & truth(d.closing_full_portfolio)
                           & truth(d.open__is_countable))
    d.loc[known_open_absence, ['quantity_open', 'opening_weight']] = 0.0
    d.loc[known_close_absence, ['quantity_close', 'closing_weight']] = 0.0
    d['known_opening_absence_zero'] = known_open_absence
    d['known_closing_absence_zero'] = known_close_absence | d.certified_exit
    d['w2_static_link_used'] = truth(d.open__w2_static_link_used) | truth(d.close__w2_static_link_used)
    d['issuer_valid_time_verified'] = False  # The supplied crosswalk is static, not a new validity claim.
    d['bond_indicator'] = d.asset_type.eq('bond').astype(float)
    bridge_cols = ['interval_id', 'company_id', 'company_period_id']
    d = d.merge(bridge[bridge_cols], on=['interval_id', 'company_id'], how='left', validate='many_to_one')
    actual = news.loc[news.period_kind.eq('actual_report_interval')].copy()
    unique(actual, ['company_id', 'period_start', 'period_end'], 'actual news windows')
    actual = actual.rename(columns={'company_id': 'news_company_id'})
    d = d.merge(actual, on='company_period_id', how='left', validate='many_to_one')
    joined = d.company_period_id.notna()
    if not d.loc[joined, 'company_id'].astype(str).eq(d.loc[joined, 'news_company_id'].astype(str)).all():
        raise ValueError('News company identity mismatch')
    if not (pd.to_datetime(d.loc[joined, 'start_date']).eq(pd.to_datetime(d.loc[joined, 'period_start']))
            & pd.to_datetime(d.loc[joined, 'end_date']).eq(pd.to_datetime(d.loc[joined, 'period_end']))).all():
        raise ValueError('News window is not the exact fund report interval')
    q0, q1 = d.quantity_open, d.quantity_close
    d['quantity_change_pct'] = np.nan
    relative = finite(q0) & finite(q1) & q0.gt(0) & q1.ge(0)
    d.loc[relative, 'quantity_change_pct'] = 100 * (q1.loc[relative] / q0.loc[relative] - 1)
    valid_q = finite(q0) & finite(q1) & q0.ge(0) & q1.ge(0)
    d['delta_quantity_units'] = (q1 - q0).where(valid_q)
    valid_w = finite(d.opening_weight) & finite(d.closing_weight) & d.opening_weight.ge(0) & d.closing_weight.ge(0)
    d['delta_share_pp'] = (100 * (d.closing_weight - d.opening_weight)).where(valid_w)
    d['closing_log_NAV'] = np.log(pd.to_numeric(d.closing_NAV, errors='coerce').where(d.closing_NAV.gt(0)))
    d['interval_days'] = (pd.to_datetime(d.end_date) - pd.to_datetime(d.start_date)).dt.days
    d['log_interval_days'] = np.log(d.interval_days.where(d.interval_days.gt(0)))
    d['log_news_count_interval'] = np.log1p(pd.to_numeric(d.n_relevant, errors='coerce').where(d.n_relevant.ge(0)))
    # Missing classification is never changed into zero sentiment.
    d['net_sentiment'] = pd.to_numeric(d.net_sentiment, errors='coerce').where(truth(d.eligible_primary))
    d['row_id'] = [hashlib.sha256(f'{f}\t{s}\t{i}'.encode()).hexdigest()
                   for f, s, i in zip(d.fund_id, d.security_id, d.interval_id)]
    unique(d, KEYS, 'candidate panel')
    return annotate_samples(d)


def annotate_samples(d):
    d = d.copy()
    for name, outcome, controls in MODELS:
        reason = pd.Series('', index=d.index, dtype='string')
        def reject(mask, label):
            nonlocal reason
            reason = reason.mask(mask, reason + label + ';')
        reject(~truth(d.target_security_candidate), 'not_target_security')
        reject(d.company_id.isna(), 'company_link_missing')
        reject(truth(d.issuer_identity_conflict), 'issuer_identity_conflict')
        reject(truth(d.asset_class_conflict), 'asset_class_conflict')
        reject(truth(d.unknown_opening_absence), 'unknown_opening_absence')
        reject(truth(d.unknown_closing_absence), 'unknown_closing_absence')
        reject(~truth(d.eligible_primary), 'sentiment_not_primary')
        reject(~finite(d.net_sentiment), 'missing_net_sentiment')
        if outcome == 'quantity_change_pct': reject(d.quantity_open.eq(0), 'zero_opening_quantity')
        reject(~finite(d[outcome]), 'missing_or_nonfinite_outcome')
        for control in controls: reject(~finite(d[control]), 'missing_control_' + control)
        d['complete_' + name] = reason.eq('')
        d['reason_' + name] = reason.str.rstrip(';')
    d['complete_cases'] = d.complete_NET_CONTROLS
    d['exclusion_reason'] = d.reason_NET_CONTROLS
    return d


def select_regressors(sample, controls):
    """Declared-order within-rank check, independent of coefficients/p-values."""
    names = ['net_sentiment'] + list(controls)
    x = sample[names].astype(float)
    within = x - x.groupby(sample.fund_id.astype(str), sort=False).transform('mean')
    n = len(sample)
    tolerance = 16 * np.finfo(float).eps * max(n, len(names))
    keep, dropped, columns = [], [], []
    for name in names:
        raw = within[name].to_numpy()
        norm = float(np.linalg.norm(raw))
        exactly_absorbed = sample.groupby('fund_id', observed=True)[name].nunique().max() <= 1
        if exactly_absorbed or norm == 0:
            if name == 'net_sentiment': raise ValueError('Net sentiment is absorbed by fund effects')
            dropped.append({'term': name, 'reason': 'constant_within_every_fund'}); continue
        vector = raw / norm
        residual = vector if not columns else vector - np.column_stack(columns) @ np.linalg.lstsq(
            np.column_stack(columns), vector, rcond=None)[0]
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm <= tolerance:
            if name == 'net_sentiment': raise ValueError('Net sentiment cannot be silently removed')
            dropped.append({'term': name, 'reason': 'deterministic_within_linear_dependence',
                            'relative_residual_norm': residual_norm, 'tolerance': tolerance,
                            'retained_predecessors': keep.copy()}); continue
        keep.append(name); columns.append(vector)
    return keep, dropped


def fit_native(sample, outcome, controls):
    from linearmodels.panel import PanelOLS
    sample = sample.sort_values(['fund_id', 'end_date', 'security_id', 'interval_id'], kind='stable').copy()
    terms, dropped = select_regressors(sample, controls)
    index = pd.MultiIndex.from_arrays([sample.fund_id.astype(str), pd.to_datetime(sample.end_date)],
                                      names=['fund_id', 'report_date'])
    x = sample[terms].astype(float).copy(); x.index = index
    x.insert(0, 'const', 1.0)
    y = pd.Series(sample[outcome].to_numpy(float), index=index, name=outcome)
    clusters = pd.DataFrame({'fund_cluster': pd.Categorical(sample.fund_id.astype(str)).codes,
                             'company_cluster': pd.Categorical(sample.company_id.astype(str)).codes}, index=index)
    model = PanelOLS(y, x, entity_effects=True, time_effects=False,
                     other_effects=None, singletons=True, drop_absorbed=False, check_rank=True)
    options = dict(cov_type='clustered', clusters=clusters, debiased=True,
                   auto_df=False, count_effects=True, group_debias=True)
    result = model.fit(**options)
    if result.nobs != len(sample): raise ValueError('Native estimator changed the complete-case row set')
    if not result.resids.index.equals(index): raise ValueError('Native estimator changed row order/index')
    return sample, result, terms, dropped


def cluster_covariance_fixture(sample, x, residual, df_resid):
    """Independent slope-only two-way sandwich for native-package fixture QA."""
    scores = x * residual[:, None]
    n = len(sample); meat = np.zeros((x.shape[1], x.shape[1]))
    groups = [sample.fund_id, sample.company_id,
              pd.MultiIndex.from_frame(sample[['fund_id', 'company_id']])]
    for sign, group in zip([1, 1, -1], groups):
        codes, labels = pd.factorize(group, sort=False)
        sums = np.zeros((len(labels), x.shape[1])); np.add.at(sums, codes, scores)
        g = len(labels)
        meat += sign * g / (g - 1) * (n - 1) / n * (sums.T @ sums)
    bread = np.linalg.inv(x.T @ x)
    return n / df_resid * bread @ meat @ bread


def fixture_tests():
    """Synthetic data only; explicitly permits repeated fund/report-date indices."""
    rng = np.random.default_rng(30102026)
    rows = []
    for f in range(16):
        for t in range(8):
            for company in range(5):
                net, control, noise = rng.normal(size=3)
                rows.append(dict(fund_id=str(f), company_id=str(company), security_id=f'S{company}',
                    interval_id=f'I{f}_{t}', end_date=pd.Timestamp('2020-01-31') + pd.offsets.MonthEnd(t),
                    net_sentiment=net, control=control, absorbed=float(f), duplicate=2*net,
                    y=3.2*net - .7*control + f*.8 + noise))
    sample = pd.DataFrame(rows)
    sample, result, terms, dropped = fit_native(sample, 'y', ['control', 'absorbed', 'duplicate'])
    within = sample[['y'] + terms] - sample[['y'] + terms].groupby(sample.fund_id).transform('mean')
    beta_within = np.linalg.lstsq(within[terms], within.y, rcond=None)[0]
    dummies = pd.get_dummies(sample.fund_id, drop_first=True, dtype=float)
    lsdv_x = np.column_stack([np.ones(len(sample)), sample[terms], dummies])
    beta_lsdv = np.linalg.lstsq(lsdv_x, sample.y, rcond=None)[0][1:1+len(terms)]
    beta_native = result.params[terms].to_numpy()
    within_error = float(np.max(abs(beta_native-beta_within)))
    lsdv_error = float(np.max(abs(beta_native-beta_lsdv)))
    independent_cov = cluster_covariance_fixture(sample, within[terms].to_numpy(),
                                                  result.resids.to_numpy(), result.df_resid)
    covariance_error = float(np.max(abs(independent_cov-result.cov.loc[terms, terms].to_numpy())))
    assert result.model.entity_effects and not result.model.time_effects and not result.model.other_effects
    assert result.resids.index.has_duplicates
    assert within_error < 1e-10 and lsdv_error < 1e-10 and covariance_error < 1e-10
    assert {r['term'] for r in dropped} == {'absorbed', 'duplicate'}
    # Endpoint/absence fixture exercises the SAME union and eligibility functions.
    dates = [pd.Timestamp('2020-01-31'), pd.Timestamp('2020-03-30')]
    pos = []
    for report, when, security, q, w in [
        ('r0', dates[0], 'unknown_exit', 12., .1), ('r0', dates[0], 'observed', 10., .2),
        ('r1', dates[1], 'observed', 12., .24), ('r1', dates[1], 'entry', 3., .06)]:
        pos.append(dict(report_id=report, report_date=when, fund_id='f', security_id=security,
            company_id='c', target_security_candidate=True, is_countable=True,
            is_reported_position=True, is_exit=False, quantity_est=q, usable_weight=w,
            asset_type='equity', asset_name=security, fund_name='Synthetic',
            w2_static_link_used=False, residence_documented_by_disclosure=True))
    interval = pd.DataFrame([dict(interval_id='i', fund_id='f', opening_report_id='r0',
        closing_report_id='r1', start_date=dates[0], end_date=dates[1], opening_full_portfolio=True,
        closing_full_portfolio=False, closing_NAV=1000., closing_HHI=.25,
        opening_passive_explicit=True, closing_passive_explicit=True,
        closing_HHI_has_documented_or_unallocated_buckets=True)])
    news = pd.DataFrame([dict(company_period_id='cp', company_id='c',
        period_kind='actual_report_interval', period_start=dates[0], period_end=dates[1],
        net_sentiment=0., n_relevant=2, eligible_primary=True, state='OBSERVED')])
    bridge = pd.DataFrame([dict(interval_id='i', company_id='c', company_period_id='cp',
        fund_id='f', opening_report_id='r0', closing_report_id='r1')])
    panel = prepare_chunk(pd.DataFrame(pos), interval, news, bridge).set_index('security_id')
    assert len(panel)==3 and pd.isna(panel.loc['unknown_exit', 'quantity_close'])
    assert 'unknown_closing_absence' in panel.loc['unknown_exit', 'exclusion_reason']
    assert panel.loc['observed', 'complete_NET_CONTROLS']
    assert panel.loc['observed', 'opening_passive_explicit']
    assert abs(panel.loc['observed', 'quantity_change_pct']-20.) < 1e-12
    assert not panel.loc['entry', 'complete_NET_CONTROLS']
    assert panel.loc['entry', 'complete_ABS_DELTA_Q_CONTROLS']
    assert panel.loc['entry', 'complete_DELTA_SHARE_PP_CONTROLS']
    assert panel.loc['observed', 'net_sentiment']==0
    incomplete = news.copy(); incomplete['eligible_primary']=False; incomplete['net_sentiment']=np.nan
    assert not prepare_chunk(pd.DataFrame(pos), interval, incomplete, bridge).complete_NET_ONLY.any()
    complete_interval=interval.copy(); complete_interval['closing_full_portfolio']=True
    exits=prepare_chunk(pd.DataFrame(pos), complete_interval, news, bridge).set_index('security_id')
    assert exits.loc['unknown_exit', 'quantity_close']==0
    assert exits.loc['unknown_exit', 'quantity_change_pct']==-100
    # Exercise actual Parquet streaming across TWO fund batches, including an
    # entirely unmatched news company in the last batch. No research files used.
    import tempfile
    import pyarrow.parquet as pq
    with tempfile.TemporaryDirectory(prefix='h1-net-synthetic-') as temporary:
        directory=Path(temporary);many_pos=[];many_intervals=[];many_bridge=[]
        for f in range(33):
            fund=f'f{f:03d}'
            p=pd.DataFrame(pos).copy();p['fund_id']=fund
            p['report_id']=p.report_id.map({'r0':fund+'_r0','r1':fund+'_r1'})
            i=interval.copy();i['fund_id']=fund;i['interval_id']=fund+'_i'
            i['opening_report_id']=fund+'_r0';i['closing_report_id']=fund+'_r1'
            if f==32:p['company_id']='unmatched'
            else:
                b=bridge.copy();b['fund_id']=fund;b['interval_id']=fund+'_i'
                b['opening_report_id']=fund+'_r0';b['closing_report_id']=fund+'_r1';many_bridge.append(b)
            many_pos.append(p);many_intervals.append(i)
        paths={k:directory/(k+'.parquet') for k in ['portfolio','intervals','news','bridge']}
        for name,frame in [('portfolio',pd.concat(many_pos,ignore_index=True)),
                           ('intervals',pd.concat(many_intervals,ignore_index=True)),
                           ('news',news),('bridge',pd.concat(many_bridge,ignore_index=True))]:
            frame.to_parquet(paths[name],index=False)
        candidate=directory/'all_candidate_panel.parquet'
        flow=build_candidate_file(paths,candidate,lambda *a,**kw:None)
        saved=pd.read_parquet(candidate)
        assert len(saved)==99 and flow['diagnostics']['unknown_closing_absence']==33
        assert saved.complete_NET_CONTROLS.sum()==32
        assert str(pq.ParquetFile(candidate).schema_arrow.field('eligible_primary').type)=='bool'
    return dict(passed=True, synthetic_only=True, observations=len(sample), duplicate_fund_date_index=True,
        fund_only_FE=True, within_max_error=within_error, LSDV_max_error=lsdv_error,
        native_two_way_covariance_max_error=covariance_error,
        deterministic_collinearity_drops=dropped, endpoint_union=True,
        unknown_absence_preserved=True, partial_report_observed_position_retained=True,
        passive_flag_not_filtered=True, zero_base_entry_excluded_from_relative_only=True,
        true_zero_vs_missing_sentiment=True, certified_complete_absence_minus100=True,
        two_batch_parquet_roundtrip_rows=99,nullable_news_boolean_preserved=True)


def arrow_ready(frame):
    d=frame.copy()
    for column in d:
        if column in ['eligible_primary','strict_complete','eligible_partial95_secondary']:
            d[column]=d[column].astype('boolean')
        elif d[column].dtype == object: d[column]=d[column].astype('string')
    return d


def build_candidate_file(paths, output, checkpoint):
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema_names = set(pq.ParquetFile(paths['portfolio']).schema_arrow.names)
    if REQUIRED_POS - schema_names: raise ValueError('Portfolio schema is missing required fields')
    columns=[c for c in POSITION_FIELDS if c in schema_names]
    positions=pd.read_parquet(paths['portfolio'], columns=columns)
    lookup=pd.read_parquet(paths['company_lookup']) if 'company_lookup' in paths else None
    if lookup is not None:
        require(lookup,{'company_id','company'},'company lookup');unique(lookup,['company_id'],'company lookup')
    for field in ['report_date', 'quote_date']:
        if field in positions: positions[field]=pd.to_datetime(positions[field])
    intervals=pd.read_parquet(paths['intervals'])
    intervals['start_date']=pd.to_datetime(intervals.start_date);intervals['end_date']=pd.to_datetime(intervals.end_date)
    news=pd.read_parquet(paths['news'])
    bridge=pd.read_parquet(paths['bridge'])
    require(intervals,REQUIRED_INTERVAL,'intervals');require(news,REQUIRED_NET,'news');require(bridge,REQUIRED_BRIDGE,'bridge')
    unique(positions,['report_id','security_id'],'portfolio')
    unique(intervals,['interval_id'],'intervals');unique(bridge,['interval_id','company_id'],'bridge')
    unique(news,['company_period_id'],'news')
    funds=sorted(intervals.fund_id.astype(str).unique())
    writer=None; schema=None; count=0; totals={m[0]:0 for m in MODELS}; reasons={m[0]:{} for m in MODELS}
    diagnostics={'all_actual_intervals':len(intervals),'funds_with_intervals':len(funds),'unknown_closing_absence':0,
                 'known_complete_exit':0,'zero_opening_quantity':0,'observed_partial_closing_positions':0}
    try:
        for offset in range(0,len(funds),32):
            selected=set(funds[offset:offset+32])
            block_i=intervals.loc[intervals.fund_id.astype(str).isin(selected)]
            report_ids=set(block_i.opening_report_id)|set(block_i.closing_report_id)
            block_p=positions.loc[positions.report_id.isin(report_ids)].copy()
            block_b=bridge.loc[bridge.interval_id.isin(block_i.interval_id)]
            block=prepare_chunk(block_p,block_i,news,block_b)
            if lookup is not None:
                block=block.merge(lookup[['company_id','company']],on='company_id',how='left',validate='many_to_one')
            if block.empty: continue
            count+=len(block)
            for name,_,_ in MODELS:
                totals[name]+=int(block['complete_'+name].sum())
                for reason,n in block.loc[~block['complete_'+name],'reason_'+name].value_counts().items():
                    reasons[name][str(reason)]=reasons[name].get(str(reason),0)+int(n)
            diagnostics['unknown_closing_absence']+=int(block.unknown_closing_absence.sum())
            diagnostics['known_complete_exit']+=int(block.known_closing_absence_zero.sum())
            diagnostics['zero_opening_quantity']+=int(block.quantity_open.eq(0).sum())
            diagnostics['observed_partial_closing_positions']+=int((block.closing_position_reported & ~truth(block.closing_full_portfolio)).sum())
            # Preserve nullable flags consistently even in an all-absent endpoint block.
            for c in block:
                if c.startswith(('open__','close__')) and any(c.endswith(x) for x in [
                    'target_security_candidate','is_countable','is_reported_position','is_exit',
                    'w2_static_link_used','residence_documented_by_disclosure']): block[c]=block[c].astype('boolean')
            table=pa.Table.from_pandas(arrow_ready(block),preserve_index=False)
            if writer is None:
                schema=table.schema;writer=pq.ParquetWriter(output,schema,compression='zstd')
            else: table=table.cast(schema)
            writer.write_table(table)
            checkpoint('BUILDING_CANDIDATE_PANEL',completed_funds=min(offset+32,len(funds)),candidate_rows=count)
    finally:
        if writer is not None: writer.close()
    if not count: raise ValueError('No candidate rows')
    return dict(candidate_rows=count,complete_cases_by_model=totals,exclusion_reason_combinations=reasons,
                diagnostics=diagnostics,sample_policy='Each declared model uses its own fixed complete cases; no trimming/winsorization')


def describe_result(sample,result,identifier,outcome,terms,dropped):
    from scipy import stats
    groups={'funds':int(sample.fund_id.nunique()),'companies':int(sample.company_id.nunique())}
    conservative_df=min(groups.values())-1
    native_ci=result.conf_int()
    rows=[]
    for term in result.params.index:
        beta=float(result.params[term]);se=float(result.std_errors[term]);t=beta/se if se>0 else np.nan
        critical=stats.t.ppf(.975,conservative_df) if conservative_df>0 else np.nan
        rows.append(dict(model=identifier,outcome=outcome,term=term,beta=beta,se_native=se,
            t_native=float(result.tstats[term]),p_native=float(result.pvalues[term]),
            ci95_native_low=float(native_ci.loc[term].iloc[0]),ci95_native_high=float(native_ci.loc[term].iloc[1]),
            conservative_cluster_df=conservative_df,
            p_cluster_t_two_sided=2*stats.t.sf(abs(t),conservative_df) if conservative_df>0 else np.nan,
            p_H1_positive_one_sided=stats.t.sf(t,conservative_df) if conservative_df>0 and term=='net_sentiment' else np.nan,
            ci95_cluster_t_low=beta-critical*se,ci95_cluster_t_high=beta+critical*se))
    def f_record(stat):
        try:return dict(stat=float(stat.stat),p=float(stat.pval),df=stat.df,df_denom=stat.df_denom)
        except Exception:return dict(description=str(stat))
    negative_diagonal=[str(term) for term,value in zip(result.params.index,np.diag(result.cov)) if value<0]
    diagnostics=dict(model=identifier,status='ESTIMATED',dependent=outcome,nobs=int(result.nobs),**groups,
        negative_covariance_diagonal_terms=negative_diagonal,covariance_PSD_clipping=False,
        fund_date_index_duplicates=int(sample.duplicated(['fund_id','end_date']).sum()),
        rsquared=float(result.rsquared),rsquared_within=float(result.rsquared_within),
        rsquared_between=float(result.rsquared_between),rsquared_overall=float(result.rsquared_overall),
        native_df_model=int(result.df_model),native_df_resid=int(result.df_resid),
        native_F=f_record(result.f_statistic),native_robust_F=f_record(result.f_statistic_robust),
        included_effects=list(result.included_effects),entity_definition='fund_id only',
        time_effects=False,other_effects=False,weights=None,retained_terms=terms,
        dropped_controls=dropped,conservative_cluster_df=conservative_df,
        first_start_date=str(sample.start_date.min()),last_end_date=str(sample.end_date.max()),
        explicitly_passive_rows=int((truth(sample.opening_passive_explicit)|truth(sample.closing_passive_explicit)).sum()),
        partial_closing_report_rows=int((~truth(sample.closing_full_portfolio)).sum()),
        singleton_funds=int(sample.groupby('fund_id',observed=True).size().eq(1).sum()),
        dependent_min=float(sample[outcome].min()),dependent_max=float(sample[outcome].max()),
        dependent_quantiles=sample[outcome].quantile([0,.01,.5,.99,1]).to_dict(),
        covariance_options=dict(cov_type='clustered',clusters=['fund_id','company_id'],debiased=True,
                                auto_df=False,count_effects=True,group_debias=True),
        native_summary_pvalues_unchanged=True,source_row_hash=hashlib.sha256('\n'.join(sample.row_id).encode()).hexdigest())
    return diagnostics,rows


def viewer(frame,root):
    available=['fund_name','fund_id','asset_name','security_id','company','company_id','start_date','end_date',
        'quantity_open','quantity_close','quantity_change_pct','net_sentiment','n_relevant',
        'opening_weight','closing_log_NAV','closing_HHI','log_news_count_interval',
        'log_interval_days','bond_indicator','closing_full_portfolio',
        'closing_HHI_has_documented_or_unallocated_buckets','w2_static_link_used',
        'opening_passive_explicit','closing_passive_explicit','row_id']
    columns=[c for c in available if c in frame]
    sample=frame.groupby('fund_id',sort=True,observed=True).head(2).head(200)
    paths='\n'.join(str(root/p) for p in ['all_candidate_panel.parquet','regression_panel.parquet',
        'regression_panel.csv.gz','sample_flow.json','variable_dictionary.json','source_manifest.json'])
    table=sample[columns].to_html(index=False,na_rep='',border=0,escape=True,table_id='panel')
    content='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>H1: использованные строки</title>
<style>body{font:14px Arial;margin:24px;color:#202b36}h1{font-size:22px}input{font:inherit;padding:8px;width:540px}
.scroll{overflow:auto;max-height:76vh}table{border-collapse:collapse}th{background:#24445c;color:white;position:sticky;top:0}
th,td{padding:7px 10px;border-bottom:1px solid #e6ebef;text-align:left;white-space:nowrap}tbody tr:nth-child(even){background:#f5f7f9}
pre{white-space:pre-wrap;user-select:text}p{max-width:1000px}</style>
<h1>H1: строки основной модели NET_CONTROLS</h1><p>Показаны до 200 фактически использованных строк.
Полная панель сохранена на ПК по путям ниже. Количества расчетные V/P, не наблюдаемые сделки.
Y — процентное изменение количества между фактическими раскрытиями. Пропуски не заменялись нулями.</p>
<input id="q" placeholder="Поиск по фонду, компании, бумаге или дате"><div class="scroll">'''+table+'''</div>
<h2>Полные файлы</h2><pre>'''+html.escape(paths)+'''</pre><script>
document.getElementById('q').addEventListener('input',e=>{const q=e.target.value.toLowerCase();
document.querySelectorAll('#panel tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));});
</script></html>'''
    (root/'regression_data_view.html').write_text(content,encoding='utf-8')


def run(args):
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    import linearmodels
    plan_path=Path(args.plan);plan=json.loads(plan_path.read_text(encoding='utf-8'))
    if sha(plan_path)!=PLAN_SHA256: raise ValueError('Frozen analysis plan hash mismatch')
    if plan['fixed_effects']!=['fund_id'] or plan['main_model']!='NET_CONTROLS': raise ValueError('Unexpected model plan')
    input_root=Path(plan['output_root'])
    root=Path(args.output) if args.output else input_root/'regression'
    if root.resolve()==input_root.resolve():raise ValueError('Regression outputs must use a separate subdirectory')
    root.mkdir(parents=True,exist_ok=True)
    if (root/'regression_summary.json').exists(): raise FileExistsError('Completed results must not be overwritten')
    with (root/'estimation_owner.lock').open('x',encoding='utf-8') as lock:lock.write(str(os.getpid()))
    started=time.monotonic()
    def checkpoint(stage,**extra):
        record=dict(stage=stage,pid=os.getpid(),at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    elapsed_seconds=time.monotonic()-started,**extra)
        save(root/'estimation_checkpoint.json',record);print(json.dumps(clean(record)),flush=True)
    checkpoint('CHECKING_READY_INPUTS')
    ready=json.loads((input_root/'ready_manifest.json').read_text(encoding='utf-8'))
    if ready.get('ready') is not True: raise ValueError('News dataset is not ready')
    news_qa=json.loads((input_root/'qa.json').read_text(encoding='utf-8'))
    if news_qa.get('passed') is not True: raise ValueError('News QA is not passed')
    paths=dict(portfolio=Path(plan['portfolio_source']),intervals=Path(plan['interval_source']),
        news=input_root/'net_sentiment_company_period.parquet',bridge=input_root/'fund_interval_company_windows.parquet')
    if (input_root/'control_company_lookup.parquet').exists():paths['company_lookup']=input_root/'control_company_lookup.parquet'
    sources={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in paths.values()}
    save(root/'source_manifest.json',dict(inputs=sources,plan_sha256=sha(plan_path),code_sha256=sha(__file__),
        executable=sys.executable,python=sys.version,linearmodels=linearmodels.__version__,
        no_news_reaggregation=True,no_new_LLM=True,protected_inputs_read_only=True))
    save(root/'estimation_plan_before_fit.json',dict(frozen_user_plan=plan,models=MODELS,
        sample_comparison='Each model uses its declared complete cases; N may differ',
        estimator='Native unweighted PanelOLS with fund entity effects only',
        native_covariance_options=dict(cov_type='clustered',clusters=['fund_id','company_id'],
            debiased=True,auto_df=False,count_effects=True,group_debias=True),
        no_outlier_filter=True,no_residualized_signal=True,no_hidden_extra_effects=True))
    checkpoint('SYNTHETIC_NATIVE_ESTIMATOR_QA')
    fixture=fixture_tests();save(root/'estimation_fixture_qa.json',fixture)
    checkpoint('BUILDING_CANDIDATE_PANEL')
    flow=build_candidate_file(paths,root/'all_candidate_panel.parquet',checkpoint)
    save(root/'sample_flow.json',flow)
    dictionary=dict(definitions=plan['control_definitions'],primary_y='100*(Q1/Q0-1), Q0>0',
        quantity_est='V/P from sealed source; approximate units, not observed fund transactions',
        bond_prices='Clean price times historical face value /100; accrued interest not verified',
        delta_quantity_units='Q1-Q0; may include known zero-base entries',
        delta_share_pp='100*(gross-asset weight1-weight0), percentage points',
        missing='Unknown absence is never zero. No-news/incomplete classification is never neutral.',
        passive='Explicit passive flags are retained, not an exclusion criterion',
        HHI='Existing closing inventory HHI, with documented/unallocated-bucket and partial-report flags',
        issuer_identity='Existing static issuer links/target masks; no expanded or newly time-validated link',
        timing='Actual consecutive disclosure dates; calendar/EOM restrictions not applied',
        interpretation='Association in estimated position changes, not directly observed trades or causal news effects',
        closing_controls='End-date controls are contemporaneous and can themselves respond to news; specification follows user request')
    save(root/'variable_dictionary.json',dictionary)
    all_coefficients=[];results=[]
    dataset=ds.dataset(root/'all_candidate_panel.parquet',format='parquet')
    for identifier,outcome,controls in MODELS:
        checkpoint('FITTING_NATIVE_PANELOLS',model=identifier)
        sample=dataset.to_table(filter=ds.field('complete_'+identifier)==True).to_pandas()
        if len(sample)!=flow['complete_cases_by_model'][identifier]: raise ValueError('Saved sample flag/count mismatch')
        if sample.empty:
            results.append(dict(model=identifier,status='NOT_ESTIMABLE',reason='No declared complete cases'));continue
        if min(sample.fund_id.nunique(),sample.company_id.nunique())<2:
            results.append(dict(model=identifier,status='NOT_ESTIMABLE',reason='Fewer than two clusters in a dimension'));continue
        sample,result,terms,dropped=fit_native(sample,outcome,controls)
        # Native summaries are exported verbatim: native p-values/F statistics are not rewritten.
        (root/(identifier+'.summary.txt')).write_text(result.summary.as_text()+'\n',encoding='utf-8')
        (root/(identifier+'.summary.html')).write_text(result.summary.as_html(),encoding='utf-8')
        record,coefficients=describe_result(sample,result,identifier,outcome,terms,dropped)
        results.append(record);all_coefficients.extend(coefficients)
        stem='regression_panel' if identifier=='NET_CONTROLS' else identifier+'.used_rows'
        sample.to_parquet(root/(stem+'.parquet'),index=False,compression='zstd')
        if identifier=='NET_CONTROLS':
            sample.to_csv(root/'regression_panel.csv.gz',index=False,compression='gzip',encoding='utf-8')
            preview=sample.groupby('fund_id',sort=True,observed=True).head(2).head(20)
            preview.to_csv(root/'regression_rows_preview.csv',index=False,encoding='utf-8-sig')
            viewer(sample,root)
        checkpoint('MODEL_EXPORTED',model=identifier,nobs=int(result.nobs))
        del sample,result
    pd.DataFrame(all_coefficients).to_csv(root/'regression_coefficients.csv',index=False,encoding='utf-8-sig')
    for path,record in sources.items():
        if sha(path)!=record['sha256']:raise ValueError('Input changed during the run: '+path)
    save(root/'regression_summary.json',dict(status='COMPLETE',models=results,
        coefficients=all_coefficients,sample_flow=flow,fixture_QA=fixture,output_root=str(root),
        primary='NET_CONTROLS',native_pvalues_untouched=True,extra_cluster_t_inference_separate=True,
        plan_sha256=PLAN_SHA256,no_new_LLM=True,no_news_aggregation=True))
    checkpoint('COMPLETED',models=len(results),primary='NET_CONTROLS')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',help='Explicit path to the frozen user-approved plan')
    parser.add_argument('--output',help='Optional separate new output directory')
    parser.add_argument('--fixture-only',action='store_true',help='Generated data only; no research input access')
    parser.add_argument('--fixture-report',help='Save small synthetic QA JSON')
    args=parser.parse_args()
    if args.fixture_only:
        record=fixture_tests()
        if args.fixture_report:save(args.fixture_report,record)
        print(json.dumps(clean(record),indent=2));return
    if not args.plan:parser.error('--plan is required for actual data; fixtures do not need it')
    try:run(args)
    except Exception as error:
        if args.output:
            root=Path(args.output)
        else:
            try:root=Path(json.loads(Path(args.plan).read_text(encoding='utf-8'))['output_root'])/'regression'
            except Exception:raise
        if root.exists():save(root/'estimation_failure.json',dict(error=repr(error),traceback=traceback.format_exc(),pid=os.getpid()))
        raise


if __name__=='__main__':main()
