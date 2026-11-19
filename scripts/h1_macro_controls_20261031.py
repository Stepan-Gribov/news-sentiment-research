"""Build auditable macro controls for actual fund-report intervals.

This module only reads the supplied small raw-source files. It never opens the
news database, executes regressions, downloads inputs, or changes source files.
Actual research-data aggregation is intended to run on research-host. --self-test uses
temporary synthetic inputs only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


SERIES = {
    'USD_RUB': dict(path='central_bank/USD_RUB_official_daily.csv', date='Date',
        value='VunitRate', date_format='%d.%m.%Y',
        sha256='45f5ec8b5d7d805cff8ef8249f16411489ef82d0e5ffd9c7e6938400dff2b36b'),
    'Ruonia': dict(path='central_bank/RUONIA_daily.csv', date='DT', value='ruo',
        date_format='%Y-%m-%d',
        sha256='9f5bf404dcb7dc07b957508ae5c47abeb431a643a3e7717f50825cb09dcda499'),
    'Imoex': dict(path='moex/indices/IMOEX_daily_history_2012-01-01_2026-08-31.csv',
        date='TRADEDATE', value='CLOSE', date_format='%Y-%m-%d',
        sha256='96e83ceb46c314f5f79900ec6361e77afe49ed28c71a4dbd2c9ec530eddafed3'),
    'RVI': dict(path='moex/indices/RVI_daily_history_2012-01-01_2026-08-31.csv',
        date='TRADEDATE', value='CLOSE', date_format='%Y-%m-%d',
        sha256='c63c1e55e934f07c150fa3ba436d30c2035247ff560f54156fe1daa3e961d2a6'),
}
MARKET_STATS_PATH = 'investfunds/public_market_statistics_raw/fund_count_and_market_net_flows_public_source_points.csv'
MARKET_STATS_SHA256 = '96cb4f72cc57cf19c45462a6e79e3d2cbf224cfa6b5d1f89800dd414357d27aa'
MACRO_CONTROL_COLUMNS = [
    'Imoex', 'imoex_ret', 'Ruonia', 'USD_RUB', 'RVI', 'Flow_billion_RUB',
    'unique_funds', 'dummy_mar2014', 'dummy_feb2022', 'dummy_sep2018',
]


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _numeric(values):
    return pd.to_numeric(values.astype(str).str.replace('\u00a0', '', regex=False)
        .str.replace(' ', '', regex=False).str.replace(',', '.', regex=False),
        errors='coerce')


def _dates(values):
    parsed = pd.to_datetime(values, errors='raise')
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_convert('Europe/Moscow').dt.tz_localize(None)
    return parsed.dt.normalize()


def _load_series(source_root, name, *, validate_hashes):
    spec = SERIES[name]
    path = Path(source_root) / spec['path']
    checksum = file_sha256(path)
    if validate_hashes and checksum != spec['sha256']:
        raise ValueError(f'Unexpected source checksum: {spec["path"]}')
    source = pd.read_csv(path, encoding='utf-8-sig', dtype=str)
    out = pd.DataFrame({
        'source_date': pd.to_datetime(source[spec['date']], format=spec['date_format'], errors='raise').dt.normalize(),
        'value': _numeric(source[spec['value']]),
    })
    if name == 'Ruonia' and 'DateUpdate' in source:
        out['source_update_timestamp'] = source.DateUpdate.fillna('')
    else:
        out['source_update_timestamp'] = ''
    if out.source_date.isna().any() or (~np.isfinite(out.value)).any() or out.value.le(0).any():
        raise ValueError(f'Invalid date or nonpositive/nonfinite level in {spec["path"]}')
    if out.groupby('source_date').value.nunique().gt(1).any():
        raise ValueError(f'Ambiguous duplicate dates in {spec["path"]}')
    duplicate_rows = int(out.duplicated('source_date').sum())
    out = out.drop_duplicates('source_date', keep='last').sort_values('source_date').reset_index(drop=True)
    provenance = dict(relative_path=spec['path'], path=str(path), sha256=checksum,
        rows=int(len(source)), unique_dates=int(len(out)), identical_duplicate_dates_removed=duplicate_rows,
        first_date=str(out.source_date.min().date()), last_date=str(out.source_date.max().date()),
        date_column=spec['date'], value_column=spec['value'])
    return out, provenance


def _asof(source, target_dates, name, max_age_days):
    """Last dated observation <= report date, never a future observation."""
    target_dates = pd.Series(target_dates).reset_index(drop=True)
    source_dates = source.source_date.to_numpy(dtype='datetime64[ns]')
    targets = target_dates.to_numpy(dtype='datetime64[ns]')
    loc = np.searchsorted(source_dates, targets, side='right') - 1
    matched = (loc >= 0) & target_dates.notna().to_numpy()
    selected = np.full(len(targets), np.datetime64('NaT'), dtype='datetime64[ns]')
    values = np.full(len(targets), np.nan)
    updates = np.full(len(targets), '', dtype=object)
    selected[matched] = source_dates[loc[matched]]
    values[matched] = source.value.to_numpy(float)[loc[matched]]
    updates[matched] = source.source_update_timestamp.to_numpy(object)[loc[matched]]
    selected = pd.Series(selected)
    age = (target_dates - selected).dt.days
    reason = np.full(len(targets), '', dtype=object)
    reason[~matched] = 'no_prior_source_observation'
    reason[target_dates.isna().to_numpy()] = 'missing_report_date'
    stale = matched & age.gt(max_age_days).to_numpy()
    reason[stale] = 'source_observation_older_than_' + str(max_age_days) + '_days'
    valid = matched & ~stale
    values[~valid] = np.nan
    if ((selected[matched].to_numpy() > targets[matched])).any():
        raise AssertionError('Future observation selected')
    return pd.DataFrame({name: values, name + '_source_date': selected,
        name + '_source_age_days': age, name + '_source_update_timestamp': updates,
        name + '_missing_reason': reason})


def _load_flows(source_root, *, flow_month_label, validate_hashes):
    if flow_month_label not in {'same_month', 'previous_month'}:
        raise ValueError('flow_month_label must explicitly be same_month or previous_month')
    path = Path(source_root) / MARKET_STATS_PATH
    checksum = file_sha256(path)
    if validate_hashes and checksum != MARKET_STATS_SHA256:
        raise ValueError('Unexpected source checksum: ' + MARKET_STATS_PATH)
    raw = pd.read_csv(path, encoding='utf-8-sig', dtype=str)
    chosen = raw[raw.data_key_source.eq('inflow') & raw.series_id_source.eq('4')].copy()
    if chosen.empty:
        raise ValueError('Total market inflow series ID4 missing')
    timestamp = pd.to_numeric(chosen.timestamp_ms_source, errors='raise')
    date = pd.to_datetime(timestamp, unit='ms', utc=True).dt.tz_convert('Europe/Moscow').dt.tz_localize(None).dt.normalize()
    out = pd.DataFrame({'source_date': date, 'source_timestamp_ms': timestamp,
        'raw_RUB': _numeric(chosen.value_source)})
    out['economic_month'] = out.source_date.dt.to_period('M')
    if flow_month_label == 'previous_month':
        out['economic_month'] = out.economic_month - 1
    if out.raw_RUB.isna().any() or (~np.isfinite(out.raw_RUB)).any():
        raise ValueError('Nonfinite flow source values')
    if out.groupby('economic_month').raw_RUB.nunique().gt(1).any():
        raise ValueError('Conflicting monthly market flow observations')
    out = out.drop_duplicates('economic_month', keep='last').sort_values('economic_month').reset_index(drop=True)
    meta = dict(relative_path=MARKET_STATS_PATH, path=str(path), sha256=checksum,
        source_series='data_key_source=inflow, series_id_source=4; total only, no constituent-category sum',
        rows=int(len(out)), flow_month_label=flow_month_label, raw_unit='RUB', output_unit='billion RUB',
        divisor=1_000_000_000, first_economic_month=str(out.economic_month.min()),
        last_economic_month=str(out.economic_month.max()),
        source_dates_are_api_labels_not_verified_first_publication_times=True)
    return out, meta


def _flow_controls(intervals, flows, mode):
    if mode not in {'window_complete', 'latest_completed'}:
        raise ValueError('flow_mode must explicitly be window_complete or latest_completed')
    lookup = {r.economic_month: r for r in flows.itertuples(index=False)}
    records = []
    for r in intervals.itertuples(index=False):
        start, end = r.start_date, r.end_date
        complete = bool(start.is_month_end and end.is_month_end and end > start)
        expected = pd.period_range((start + pd.Timedelta(days=1)).to_period('M'), end.to_period('M'), freq='M') if complete else pd.PeriodIndex([], freq='M')
        missing = [str(m) for m in expected if m not in lookup]
        valid_window = complete and len(expected) > 0 and not missing
        full_rows = [lookup[m] for m in expected if m in lookup]
        window = sum(v.raw_RUB for v in full_rows) / 1e9 if valid_window else np.nan
        latest_month = end.to_period('M') if end.is_month_end else end.to_period('M') - 1
        latest = lookup.get(latest_month)
        endpoint = float(latest.raw_RUB) / 1e9 if latest is not None else np.nan
        window_reason = '' if valid_window else ('partial_calendar_month_edges' if not complete else 'missing_flow_months:' + ','.join(missing))
        endpoint_reason = '' if latest is not None else 'missing_latest_completed_economic_month:' + str(latest_month)
        chosen_rows = full_rows if mode == 'window_complete' and valid_window else ([latest] if mode == 'latest_completed' and latest is not None else [])
        records.append({
            'Flow_billion_RUB': window if mode == 'window_complete' else endpoint,
            'Flow_missing_reason': window_reason if mode == 'window_complete' else endpoint_reason,
            'Flow_window_complete_billion_RUB': window,
            'Flow_window_complete_missing_reason': window_reason,
            'Flow_latest_completed_billion_RUB': endpoint,
            'Flow_latest_completed_missing_reason': endpoint_reason,
            'Flow_latest_completed_economic_month': str(latest_month),
            'Flow_window_calendar_aligned': complete,
            'Flow_window_expected_months': int(len(expected)),
            'Flow_window_missing_months': ','.join(missing),
            'Flow_economic_first_month': str(expected[0]) if mode == 'window_complete' and len(expected) else (str(latest_month) if mode == 'latest_completed' else ''),
            'Flow_economic_last_month': str(expected[-1]) if mode == 'window_complete' and len(expected) else (str(latest_month) if mode == 'latest_completed' else ''),
            'Flow_source_first_date': min(v.source_date for v in chosen_rows) if chosen_rows else pd.NaT,
            'Flow_source_last_date': max(v.source_date for v in chosen_rows) if chosen_rows else pd.NaT,
            'Flow_source_first_timestamp_ms': min(v.source_timestamp_ms for v in chosen_rows) if chosen_rows else np.nan,
            'Flow_source_last_timestamp_ms': max(v.source_timestamp_ms for v in chosen_rows) if chosen_rows else np.nan,
        })
    return pd.DataFrame.from_records(records)


def build_macro_controls(intervals, source_root, *, flow_month_label,
                         flow_mode='window_complete', unique_funds_by_date=None,
                         max_age_days=7, validate_hashes=True):
    """Return (one-row-per-interval DataFrame, JSON-serializable audit metadata).

    unique_funds_by_date, if supplied, must have date/end_date and unique_funds
    and must describe the full baseline observed-report universe, not a model's
    complete-case subsample. Otherwise the supplied interval universe is used.
    Flow's label convention must be explicitly fixed before coefficients.
    """
    required = ['interval_id', 'fund_id', 'start_date', 'end_date']
    if any(c not in intervals for c in required):
        raise ValueError('Missing interval keys/dates: ' + ','.join(c for c in required if c not in intervals))
    if intervals.interval_id.isna().any() or intervals.interval_id.duplicated().any():
        raise ValueError('interval_id must be unique and nonmissing')
    if max_age_days < 0:
        raise ValueError('max_age_days must be nonnegative')
    d = intervals[required].copy().reset_index(drop=True)
    d['start_date'], d['end_date'] = _dates(d.start_date), _dates(d.end_date)
    if d[required].isna().any().any() or d.end_date.le(d.start_date).any():
        raise ValueError('Invalid or nonpositive actual interval')
    source_meta = {}
    loaded = {}
    pieces = [d]
    for name in ['Imoex', 'Ruonia', 'USD_RUB', 'RVI']:
        loaded[name], source_meta[name] = _load_series(source_root, name, validate_hashes=validate_hashes)
        pieces.append(_asof(loaded[name], d.end_date, name, max_age_days))
    opening = _asof(loaded['Imoex'], d.start_date, 'Imoex_open', max_age_days)
    pieces.append(opening)
    result = pd.concat(pieces, axis=1)
    result['imoex_ret'] = result.Imoex.div(result.Imoex_open) - 1
    result['imoex_ret_missing_reason'] = np.where(result.Imoex.isna(), 'missing_closing_index',
        np.where(result.Imoex_open.isna(), 'missing_opening_index', ''))
    flows, source_meta['Flow'] = _load_flows(source_root, flow_month_label=flow_month_label, validate_hashes=validate_hashes)
    result = pd.concat([result, _flow_controls(d, flows, flow_mode)], axis=1)
    if unique_funds_by_date is None:
        counts = d.groupby('end_date', as_index=False).fund_id.nunique().rename(columns={'fund_id': 'unique_funds'})
        count_scope = 'number of distinct observed fund_id at each exact closing date in supplied unfiltered interval universe'
    else:
        counts = unique_funds_by_date.copy()
        if 'end_date' not in counts and 'date' in counts:
            counts = counts.rename(columns={'date': 'end_date'})
        if not {'end_date', 'unique_funds'}.issubset(counts):
            raise ValueError('unique_funds_by_date requires date/end_date and unique_funds')
        counts = counts[['end_date', 'unique_funds']].copy()
        counts['end_date'] = _dates(counts.end_date)
        counts['unique_funds'] = pd.to_numeric(counts.unique_funds, errors='raise')
        count_scope = 'caller-supplied full-baseline observed fund universe at each exact closing date'
    if counts.end_date.duplicated().any() or counts.unique_funds.isna().any() or counts.unique_funds.le(0).any() or counts.unique_funds.mod(1).ne(0).any():
        raise ValueError('Invalid or duplicate closing-date fund counts')
    result = result.merge(counts, how='left', on='end_date', validate='many_to_one', sort=False)
    result['unique_funds_missing_reason'] = np.where(result.unique_funds.isna(), 'missing_observed_fund_count_at_exact_end_date', '')
    for term, cutoff in [('dummy_mar2014', '2014-03-01'), ('dummy_feb2022', '2022-02-01'), ('dummy_sep2018', '2018-09-01')]:
        result[term] = result.end_date.ge(pd.Timestamp(cutoff)).astype(int)
    if len(result) != len(d) or result.interval_id.duplicated().any() or not result.interval_id.equals(d.interval_id):
        raise AssertionError('Macro join changed row keys/order')
    availability = {c: dict(nonmissing=int(result[c].notna().sum()), missing=int(result[c].isna().sum())) for c in MACRO_CONTROL_COLUMNS}
    reasons = {c: result[c].value_counts(dropna=False).to_dict() for c in result if c.endswith('_missing_reason')}
    dictionary = {
        'Imoex': 'Last positive IMOEX daily CLOSE on/before actual closing report date; no more than max_age_days old; index points.',
        'imoex_ret': 'Imoex(end)/Imoex(start)-1 over the exact actual rebalance interval; fraction, not percentage points.',
        'Ruonia': 'Last RUONIA observation DT on/before actual closing report date; percent annual rate. DateUpdate retained separately, not asserted to be original publication date.',
        'USD_RUB': 'Last official CBR VunitRate effective date on/before actual closing report date; RUB per USD.',
        'RVI': 'Last positive RVI daily CLOSE on/before actual closing report date; index points; no pre-launch extrapolation.',
        'Flow_billion_RUB': ('Sum of all economic calendar-month ID4 total inflows across a fully calendar-aligned actual report interval; incomplete edges/missing months remain NA.' if flow_mode == 'window_complete' else 'End-date control: total ID4 flow for the latest fully completed economic calendar month at/before the actual report date. It is not a sum over the full irregular rebalance interval.'),
        'Flow_window_complete_billion_RUB': 'Strict actual-window sum; finite only when both endpoints are calendar month ends and every economic month exists; no prorating.',
        'Flow_latest_completed_billion_RUB': 'Latest complete economic calendar-month end-date control; not a daily flow and not an interval-total substitute.',
        'unique_funds': count_scope + '; not the external count of all Russian funds.',
        'dummy_mar2014': '1 iff actual closing report date>=2014-03-01.',
        'dummy_feb2022': '1 iff actual closing report date>=2022-02-01.',
        'dummy_sep2018': '1 iff actual closing report date>=2018-09-01.',
    }
    metadata = dict(builder='h1_macro_controls_20261031.py', interval_rows=int(len(d)),
        source_root=str(source_root), source_files=source_meta, controls=MACRO_CONTROL_COLUMNS,
        flow_mode=flow_mode, flow_month_label=flow_month_label, max_endpoint_age_days=int(max_age_days),
        availability=availability, missing_reasons=reasons, variable_dictionary=dictionary,
        no_imputation=True, no_prorating=True, no_future_dated_daily_observations=True,
        historical_revised_sources_not_realtime_vintages=True,
        timing_note='Daily endpoint selection constrains the economic observation/effective date. Historical DateUpdate and monthly API labels are preserved and do not establish first-publication availability. These are contemporaneous/descriptive controls, not pre-decision causal controls.',
        flow_legacy_scaling_note='Raw source RUB/1e9 is billion RUB. Legacy displayed controls appear scaled by1e12; do not silently duplicate that inconsistent unit.',
        unique_funds_scope=count_scope)
    return result, metadata


def self_test():
    """Synthetic-only checks: endpoint direction, quote age, months and joins."""
    with tempfile.TemporaryDirectory(prefix='h1_macro_fixture_') as folder:
        root = Path(folder)
        daily_dates = ['2020-01-31', '2020-02-27', '2020-02-28', '2020-03-02']
        for name, spec in SERIES.items():
            p = root / spec['path']; p.parent.mkdir(parents=True, exist_ok=True)
            dates = [pd.Timestamp(x).strftime('%d.%m.%Y') for x in daily_dates] if name == 'USD_RUB' else daily_dates
            pd.DataFrame({spec['date']: dates, spec['value']: [100, 109, 110, 120]}).to_csv(p, index=False)
        p = root / MARKET_STATS_PATH; p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'data_key_source': ['inflow']*3, 'series_id_source': ['4']*3,
            'timestamp_ms_source': [int(pd.Timestamp(x, tz='Europe/Moscow').timestamp()*1000) for x in ['2020-02-01', '2020-03-01', '2020-04-01']],
            'value_source': [1e9, 2e9, -3e9]}).to_csv(p, index=False)
        intervals = pd.DataFrame({'interval_id': ['a', 'b', 'c'], 'fund_id': ['f1', 'f2', 'f1'],
            'start_date': ['2020-01-31', '2020-01-31', '2019-12-31'],
            'end_date': ['2020-02-29', '2020-02-28', '2020-01-31']})
        controls, meta = build_macro_controls(intervals, root, flow_month_label='previous_month',
            unique_funds_by_date=pd.DataFrame({'date': ['2020-02-29','2020-02-28','2020-01-31'], 'unique_funds': [22,21,20]}),
            validate_hashes=False)
        assert controls.Imoex.iloc[0] == 110 and controls.Imoex_source_date.iloc[0] == pd.Timestamp('2020-02-28')
        assert np.isclose(controls.imoex_ret.iloc[0], .1)
        assert controls.Flow_billion_RUB.iloc[0] == 2
        assert pd.isna(controls.Flow_billion_RUB.iloc[1]) and controls.Flow_latest_completed_billion_RUB.iloc[1] == 1
        assert controls.unique_funds.tolist() == [22,21,20]
        assert pd.isna(controls.imoex_ret.iloc[2]) and controls.imoex_ret_missing_reason.iloc[2] == 'missing_opening_index'
        late = pd.DataFrame({'interval_id':['late'], 'fund_id':['f1'], 'start_date':['2020-02-29'], 'end_date':['2020-03-31']})
        stale, _ = build_macro_controls(late, root, flow_month_label='previous_month', validate_hashes=False)
        assert pd.isna(stale.Imoex.iloc[0]) and stale.Imoex_source_age_days.iloc[0] == 29
        endpoint, _ = build_macro_controls(intervals, root, flow_month_label='previous_month', flow_mode='latest_completed', validate_hashes=False)
        assert endpoint.Flow_billion_RUB.tolist() == [2,1,1]
        assert meta['no_future_dated_daily_observations']
        try:
            build_macro_controls(pd.concat([intervals,intervals.iloc[:1]]), root, flow_month_label='previous_month', validate_hashes=False)
        except ValueError:
            pass
        else:
            raise AssertionError('Duplicate interval key not rejected')
    return {'status':'PASS','scope':'synthetic_only','checks':8}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(self_test(), sort_keys=True))
