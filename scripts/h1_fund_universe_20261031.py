"""Auditable fund-level index exclusions, dated asset focus, and sentiment NA audit.

Input data are supplied by the caller; this module opens no database or network.
Run actual portfolio classification on research-host. --self-test uses synthetic data only.
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd


FOCUS_LABELS = ['equity', 'bond', 'mixed', 'money_market', 'other', 'unclassified']
POSITION_COLUMNS = ['report_id', 'fund_id', 'usable_weight', 'asset_class',
    'asset_type', 'bucket_kind', 'asset_name', 'is_reported_position', 'is_exit',
    'is_countable', 'full_report', 'security_id']
REPORT_COLUMNS = ['report_id', 'fund_id', 'report_date', 'fund_name_xls',
    'fund_type_category']
INDEX_PATTERN = re.compile(r'индекс|\bindex(?:ed)?\b|index[- ]tracking', re.I)
ACTIVE_PATTERN = re.compile(r'активн\w*\s+управлен|actively\s+managed|active\s+management', re.I)
NEGATIVE_INDEX_PATTERN = re.compile(r'не\s+(?:является\s+)?индексн\w*|non[- ]index|not\s+(?:an?\s+)?index', re.I)
STRATEGY_FIELDS = ['fund_type_category', 'fund_category', 'investment_strategy',
    'strategy', 'investment_focus', 'fund_asset_focus', 'asset_focus']
NAME_FIELDS = ['fund_name_xls', 'fund_name', 'latest_fund_name', 'fund_name_file']


def _truth(v):
    if pd.isna(v):
        return False
    return bool(v is True or isinstance(v, (bool, np.bool_)) and bool(v) or
        isinstance(v, (int, float, np.integer, np.floating)) and v == 1 or
        isinstance(v, str) and v.strip().lower() in {'true', '1', 'yes', 'да'})


def _true(s):
    return s.map(_truth).astype(bool)


def _text(value):
    return '' if pd.isna(value) else str(value).strip()


def _evidence(row):
    """Positive evidence and explicit active proof; a missing index word is not proof."""
    positive, active = [], []
    for col in NAME_FIELDS + STRATEGY_FIELDS:
        value = _text(row.get(col))
        if not value:
            continue
        without_negative = NEGATIVE_INDEX_PATTERN.sub('', value)
        if INDEX_PATTERN.search(without_negative):
            positive.append(f'{col}:index_text:{value}')
        if col in STRATEGY_FIELDS and (ACTIVE_PATTERN.search(value) or NEGATIVE_INDEX_PATTERN.search(value)):
            active.append(f'{col}:explicit_nonindex_strategy:{value}')
    for col in ['is_index', 'is_index_fund', 'index_fund']:
        value = row.get(col)
        if _truth(value):
            positive.append(f'{col}:explicit_index_true')
        elif (value is False or isinstance(value, (bool, np.bool_)) and not value) and \
                _truth(row.get('index_classification_verified')):
            active.append(f'{col}:verified_index_false')
    # These inherited flags were generated from index words in report metadata.
    for col in ['passive_explicit', 'opening_passive_explicit', 'closing_passive_explicit']:
        if _truth(row.get(col)):
            positive.append(f'{col}:inherited_index_name_or_category_flag')
    if _truth(row.get('active_management_verified')):
        active.append('active_management_verified:explicit_true')
    return positive, active


def _official_focus(row):
    found, evidence = set(), []
    for col in STRATEGY_FIELDS:
        value = _text(row.get(col)).lower().replace('ё', 'е')
        if not value:
            continue
        if re.search(r'смешан|сбалансирован|\bmixed\b|\bbalanced\b', value):
            focus = 'mixed'
        elif re.search(r'денежн|money[_ -]?market|\bcash\b', value):
            focus = 'money_market'
        else:
            equity = bool(re.search(r'\bакци(?:й|и|я|онн\w*)\b|\bequit(?:y|ies)\b|\bstocks?\b', value))
            bond = bool(re.search(r'облигац|\bbonds?\b|fixed[_ -]?income', value))
            if equity and bond:
                focus = 'mixed'
            elif equity:
                focus = 'equity'
            elif bond:
                focus = 'bond'
            elif re.search(r'недвижим|товарн|драгоцен|драгметалл|фондов|венчур|хедж|\breal[_ -]?estate\b|\bcommodit|funds?\s+of\s+funds?', value):
                focus = 'other'
            else:
                continue
        found.add(focus); evidence.append(f'{col}={row.get(col)}')
    if len(found) > 1:
        return 'unclassified', 'conflicting_official_focus', '; '.join(evidence)
    if found:
        return found.pop(), 'official_opening_report_category', '; '.join(evidence)
    return None, None, None


def _opening_portfolio_profiles(positions, full_flags, threshold, weight_tolerance):
    if positions is None or not len(positions):
        return {}
    if 'report_id' not in positions or not {'asset_class', 'asset_type'}.intersection(positions):
        raise ValueError('Positions require report_id and asset_class or asset_type')
    weight_col = 'usable_weight' if 'usable_weight' in positions else 'weight'
    if weight_col not in positions:
        raise ValueError('Positions require gross-asset usable_weight or weight')
    d = positions
    if 'is_reported_position' in d:
        d = d.loc[_true(d.is_reported_position)]
    if 'is_exit' in d:
        d = d.loc[~_true(d.is_exit)]
    if 'security_id' in d and d.duplicated(['report_id', 'security_id']).any():
        raise ValueError('Repeated report/security rows would double-count portfolio focus')
    result = {}
    for report_id, group in d.groupby('report_id', observed=True, sort=False):
        rid = str(report_id)
        full = full_flags.get(rid, False)
        if 'full_report' in group:
            full = full and bool(_true(group.full_report).all())
        weights = pd.to_numeric(group[weight_col], errors='coerce')
        valid = bool(np.isfinite(weights).all() and weights.ge(0).all() and weights.le(1+weight_tolerance).all())
        total = float(weights.sum())
        profile = dict(fund_focus='unclassified', fund_focus_source='opening_portfolio_unusable',
            fund_focus_evidence='incomplete_or_invalid_opening_portfolio',
            focus_equity_share=np.nan, focus_bond_share=np.nan, focus_cash_share=np.nan,
            focus_other_known_share=np.nan, focus_unclassified_share=np.nan,
            focus_weight_sum=total, focus_opening_full=bool(full), focus_weights_valid=valid)
        if not (full and valid and abs(total-1) <= weight_tolerance):
            result[rid] = profile; continue
        asset = group['asset_class'] if 'asset_class' in group else group['asset_type']
        asset = asset.fillna('').astype(str).str.lower()
        eq, bond = asset.isin(['equity','share','shares','stock']), asset.isin(['bond','bonds'])
        cash = asset.isin(['cash','deposit','deposits','cash_deposit','money_market'])
        # Never turn every unidentified/other position into cash. A labeled
        # documented nonsecurity bucket can identify cash/deposits conservatively.
        if {'bucket_kind','asset_name','is_countable'} <= set(group):
            documented = group.bucket_kind.fillna('').ne('instrument_leaf') & ~_true(group.is_countable)
            cash = cash | (documented & group.asset_name.fillna('').astype(str).str.contains(
                r'денежные\s+средства|депозит|\bcash\b|\bdeposits?\b', case=False, regex=True))
        known_other = ~(eq | bond | cash)
        if 'is_countable' in group:
            known_other &= _true(group.is_countable)
        else:
            known_other &= False
        e, b, c, o = [float(weights.loc[mask].sum()) for mask in [eq,bond,cash,known_other]]
        unknown = max(0., total-e-b-c-o)
        focus = 'unclassified'
        if e >= threshold: focus = 'equity'
        elif b >= threshold: focus = 'bond'
        elif c >= threshold: focus = 'money_market'
        elif e >= .2 and b >= .2 and e+b >= threshold: focus = 'mixed'
        elif o >= threshold: focus = 'other'
        profile.update(fund_focus=focus, fund_focus_source='derived_opening_gross_portfolio',
            fund_focus_evidence=f'equity={e:.12g};bond={b:.12g};cash_deposit={c:.12g};known_other={o:.12g};unclassified={unknown:.12g};threshold={threshold}',
            focus_equity_share=e, focus_bond_share=b, focus_cash_share=c,
            focus_other_known_share=o, focus_unclassified_share=unknown)
        result[rid] = profile
    return result


def classify_funds(panel, registry=None, positions=None, *, focus_threshold=.8, weight_tolerance=.001):
    """Return one row per fund/interval and JSON-safe metadata.

    registry is preferably ALL funds.reports metadata, not merely the latest fund
    registry. Index evidence is propagated to the entire fund_id (EVER_INDEX).
    Fund focus uses only the exact opening report and full opening portfolio.
    """
    if not .5 < focus_threshold < 1:
        raise ValueError('focus_threshold must be between .5 and1')
    required = {'interval_id','fund_id','start_date','end_date','opening_report_id'}
    if not required <= set(panel):
        raise ValueError(f'Missing interval fields: {sorted(required-set(panel))}')
    columns = [c for c in ['interval_id','fund_id','start_date','end_date','opening_report_id',
        'closing_report_id','opening_full_portfolio','fund_name','fund_name_xls',
        'opening_passive_explicit','closing_passive_explicit'] if c in panel]
    d = panel[columns].drop_duplicates().copy()
    if d.interval_id.duplicated().any():
        raise ValueError('Conflicting fund/interval metadata')
    d['fund_id'] = d.fund_id.astype(str)
    d['start_date'] = pd.to_datetime(d.start_date, errors='raise')
    d['end_date'] = pd.to_datetime(d.end_date, errors='raise')
    reports = pd.DataFrame() if registry is None else registry.copy()
    if len(reports):
        if 'fund_id' not in reports:
            raise ValueError('Registry metadata requires fund_id')
        reports['fund_id'] = reports.fund_id.astype(str)
        if 'report_id' in reports:
            reports = reports.drop_duplicates()
            if reports.report_id.duplicated().any():
                raise ValueError('Conflicting report_id metadata revisions')
    evidence_rows = pd.concat([reports,d], ignore_index=True, sort=False)
    evidence_columns = [c for c in ['fund_id'] + NAME_FIELDS + STRATEGY_FIELDS +
        ['is_index','is_index_fund','index_fund','index_classification_verified',
         'active_management_verified','passive_explicit','opening_passive_explicit',
         'closing_passive_explicit'] if c in evidence_rows]
    evidence_rows = evidence_rows.drop_duplicates(evidence_columns)
    official_cache = {}
    def official(row):
        key = tuple(_text(row.get(c)) for c in STRATEGY_FIELDS)
        if key not in official_cache:
            official_cache[key] = _official_focus(row)
        return official_cache[key]
    by_fund = {}
    for fid, group in evidence_rows.groupby('fund_id', observed=True, sort=False):
        positive, active, names, source_nonindex = [], [], [], []
        for row in group.to_dict('records'):
            pos, act = _evidence(row)
            origin = 'report_id=' + _text(row.get('report_id')) + ';interval_id=' + _text(row.get('interval_id'))
            origin += ';date=' + _text(row.get('report_date',row.get('start_date')))
            positive.extend(f'{origin};{p}' for p in pos)
            active.extend(f'{origin};{p}' for p in act)
            if _text(row.get('fund_type_category')) and official(row)[1] == 'official_opening_report_category':
                source_nonindex.append(f'{origin};fund_type_category={row["fund_type_category"]}')
            names.extend(_text(row.get(c)) for c in NAME_FIELDS if _text(row.get(c)))
        positive, active = sorted(set(positive)), sorted(set(active))
        status = 'index' if positive else 'nonindex_verified' if active else 'source_classified_nonindex' if source_nonindex else 'unknown'
        by_fund[str(fid)] = dict(index_status=status, exclude_index_fund=bool(positive),
            index_evidence=' | '.join(positive), nonindex_verified_evidence=' | '.join(active),
            source_nonindex_category_evidence=' | '.join(sorted(set(source_nonindex))),
            classification_fund_name=' | '.join(sorted(set(names))),
            index_history_scope='all supplied report history; any positive index evidence excludes every interval')
    exact_reports = {}
    if len(reports) and {'report_id','report_date'} <= set(reports):
        reports['report_date'] = pd.to_datetime(reports.report_date, errors='raise')
        exact_reports = {str(r['report_id']):r for r in reports.to_dict('records')}
    full_flags = {}
    for row in d.to_dict('records'):
        rid = str(row['opening_report_id'])
        value = _truth(row.get('opening_full_portfolio'))
        full_flags[rid] = full_flags.get(rid, True) and bool(value)
    if 'opening_full_portfolio' not in d and positions is not None and 'full_report' in positions:
        full_flags = {str(k):bool(_true(g.full_report).all()) for k,g in positions.groupby('report_id', observed=True)}
    interval_records = d.to_dict('records')
    opening_official, need_profiles = {}, set()
    for row in interval_records:
        fid, rid = str(row['fund_id']), str(row['opening_report_id'])
        report = exact_reports.get(rid)
        classification = (None,None,None)
        if report is not None and str(report['fund_id']) == fid and report['report_date'] == row['start_date']:
            classification = official(report)
        opening_official[row['interval_id']] = classification
        if classification[0] is None:
            need_profiles.add(rid)
    subset = None if positions is None else positions.loc[positions.report_id.astype(str).isin(need_profiles)]
    profiles = _opening_portfolio_profiles(subset, full_flags, focus_threshold, weight_tolerance)
    records = []
    for row in interval_records:
        fid, rid = str(row['fund_id']), str(row['opening_report_id'])
        result = dict(interval_id=row['interval_id'], fund_id=fid, **by_fund[fid],
            fund_focus='unclassified', fund_focus_source='opening_focus_unavailable',
            fund_focus_evidence='no usable dated opening classification or full opening portfolio',
            focus_equity_share=np.nan, focus_bond_share=np.nan, focus_cash_share=np.nan,
            focus_other_known_share=np.nan, focus_unclassified_share=np.nan,
            focus_weight_sum=np.nan, focus_opening_full=full_flags.get(rid, False),
            focus_weights_valid=False)
        focus, source, proof = opening_official[row['interval_id']]
        if focus is not None:
            result.update(fund_focus=focus, fund_focus_source=source, fund_focus_evidence=proof)
        elif rid in profiles:
            result.update(profiles[rid])
        records.append(result)
    mapping = pd.DataFrame(records)
    funds = mapping.drop_duplicates('fund_id')
    metadata = dict(index_policy='EVER_INDEX across all supplied history, not a dated mandate switch policy',
        index_status_counts_funds={str(k):int(v) for k,v in funds.index_status.value_counts().items()},
        index_status_counts_intervals={str(k):int(v) for k,v in mapping.index_status.value_counts().items()},
        focus_counts_intervals={str(k):int(v) for k,v in mapping.fund_focus.value_counts().items()},
        focus_labels=FOCUS_LABELS, focus_threshold=float(focus_threshold), mixed_min_each_share=.2,
        gross_weight_tolerance=float(weight_tolerance), fund_focus_uses_only_opening_data=True,
        focus_method='Exact dated opening report category first; complete valid gross-portfolio fallback; names never determine asset focus',
        report_metadata_supplied=bool(len(reports)), report_metadata_rows=int(len(reports)),
        full_index_identification_guaranteed=bool(funds.index_status.isin(['index','nonindex_verified']).all()) if len(funds) else False,
        source_category_identification_complete=bool(funds.index_status.ne('unknown').all()) if len(funds) else False,
        opening_portfolio_fallback_requested_reports=len(need_profiles),
        source_classified_nonindex_is_provisional_not_verified_active=True,
        index_unknown_is_not_verified_active=True, ETF_is_not_automatically_index=True,
        generic_passive_is_not_automatically_index=True,
        coverage_note='Absence of an index word is unknown, not verified nonindex. All-positive detection is limited to supplied metadata/flags. The caller must state how unknown funds enter a model.',
        source_report_columns=REPORT_COLUMNS, source_position_columns=POSITION_COLUMNS)
    return mapping, metadata


def sentiment_exception_reasons(frame, news_company_ids=None):
    """Disjoint observable news/issuer states; preserve every missing score as NA."""
    n = len(frame)
    company = frame.get('company_id', pd.Series(pd.NA,index=frame.index))
    score = pd.to_numeric(frame.get('net_sentiment',pd.Series(np.nan,index=frame.index)),errors='coerce')
    candidate = pd.to_numeric(frame.get('n_candidates',pd.Series(np.nan,index=frame.index)),errors='coerce')
    relevant = pd.to_numeric(frame.get('n_relevant',pd.Series(np.nan,index=frame.index)),errors='coerce')
    state = frame.get('state',pd.Series('',index=frame.index)).fillna('').astype(str)
    eligible = _true(frame.get('eligible_primary',pd.Series(False,index=frame.index)))
    reason = pd.Series('missing_or_invalid_sentiment', index=frame.index, dtype='string')
    finite = pd.Series(np.isfinite(score),index=frame.index)
    reason.loc[eligible & finite & score.ne(0)] = 'observed_nonzero_sentiment'
    reason.loc[eligible & finite & score.eq(0)] = 'observed_zero_sentiment'
    reason.loc[state.eq('partial_or_error_coverage')] = 'incomplete_or_error_classification'
    reason.loc[state.eq('no_relevant_news') | (candidate.gt(0)&relevant.eq(0)&state.ne('partial_or_error_coverage'))] = 'all_candidates_irrelevant'
    reason.loc[state.eq('no_title_candidates') | candidate.eq(0)] = 'no_news_in_period'
    undated = pd.to_numeric(frame.get('n_undated_company_candidates',pd.Series(0,index=frame.index)),errors='coerce')
    reason.loc[reason.eq('no_news_in_period') & undated.gt(0)] = 'no_dated_news_in_period_undated_company_candidates'
    no_join = frame.get('company_period_id',pd.Series(pd.NA,index=frame.index)).isna()
    reason.loc[no_join & company.notna()] = 'coverage_unknown_or_company_period_join_missing'
    if news_company_ids is not None:
        universe = set(str(v) for v in news_company_ids if pd.notna(v))
        outside = company.notna() & ~company.astype(str).isin(universe)
        reason.loc[outside] = 'issuer_outside_declared_news_universe'
    reason.loc[company.isna()] = 'issuer_unmapped_or_ambiguous'
    impossible_zero = reason.isin(['no_news_in_period','no_dated_news_in_period_undated_company_candidates','all_candidates_irrelevant',
        'incomplete_or_error_classification','issuer_outside_declared_news_universe',
        'issuer_unmapped_or_ambiguous']) & finite
    if impossible_zero.any():
        raise ValueError('A missing/error/no-news state has a finite sentiment score')
    assert len(reason) == n
    return reason


def exception_summary(frame, news_company_ids=None):
    reasons = sentiment_exception_reasons(frame, news_company_ids)
    rows = []
    for reason in pd.unique(reasons):
        d = frame.loc[reasons.eq(reason)]
        rows.append(dict(reason=str(reason), rows=int(len(d)),
            funds=int(d.fund_id.nunique()) if 'fund_id' in d else None,
            companies=int(d.company_id.nunique()) if 'company_id' in d else None))
    return dict(rows=int(len(frame)), disjoint_reason_counts=rows,
        no_score_imputation=True, declared_news_universe_supplied=news_company_ids is not None,
        candidate_scope='All supplied candidate rows; not automatically every raw portfolio line',
        handling='All rows retained in audit/portfolio. Only eligible finite sentiment enters sentiment regressions. Actual zero requires observed eligible relevant news.',
        target_security_counts={str(k):int(v) for k,v in frame.target_security_candidate.value_counts(dropna=False).items()} if 'target_security_candidate' in frame else {})


def self_test():
    panel = pd.DataFrame([
        dict(interval_id='i1',fund_id='f1',opening_report_id='r1',start_date='2020-01-31',end_date='2020-02-29',opening_full_portfolio=True),
        dict(interval_id='i2',fund_id='f1',opening_report_id='r2',start_date='2020-02-29',end_date='2020-03-31',opening_full_portfolio=True),
        dict(interval_id='i3',fund_id='f2',opening_report_id='r3',start_date='2020-01-31',end_date='2020-02-29',opening_full_portfolio=True),
        dict(interval_id='i4',fund_id='f3',opening_report_id='r4',start_date='2020-01-31',end_date='2020-02-29',opening_full_portfolio=False)])
    reports = pd.DataFrame([
        dict(report_id='r1',fund_id='f1',report_date='2020-01-31',fund_name_xls='Earlier name',fund_type_category='открытый фонд акций'),
        dict(report_id='r2',fund_id='f1',report_date='2020-02-29',fund_name_xls='Index ETF',fund_type_category='фонд облигаций'),
        dict(report_id='r3',fund_id='f2',report_date='2020-01-31',fund_name_xls='ETF акций',fund_type_category='биржевой'),
        dict(report_id='r4',fund_id='f3',report_date='2020-01-31',fund_name_xls='Ordinary fund',fund_type_category='рыночных финансовых инструментов')])
    positions = pd.DataFrame([
        dict(report_id='r3',security_id='a',usable_weight=.45,asset_class='equity',full_report=True),
        dict(report_id='r3',security_id='b',usable_weight=.4,asset_class='bond',full_report=True),
        dict(report_id='r3',security_id='c',usable_weight=.15,asset_class='other',full_report=True),
        dict(report_id='r4',security_id='a',usable_weight=.99,asset_class='equity',full_report=False)])
    mapping, meta = classify_funds(panel,reports,positions)
    assert mapping.loc[mapping.fund_id.eq('f1'),'exclude_index_fund'].all()
    assert mapping.set_index('interval_id').loc['i1','fund_focus'] == 'equity'
    assert mapping.set_index('interval_id').loc['i2','fund_focus'] == 'bond'
    assert mapping.set_index('interval_id').loc['i3','fund_focus'] == 'mixed'
    assert mapping.set_index('interval_id').loc['i3','index_status'] == 'unknown'
    assert mapping.set_index('interval_id').loc['i4','fund_focus'] == 'unclassified'
    assert not meta['full_index_identification_guaranteed']
    examples=pd.DataFrame(dict(company_id=['c1','c1','c1','c2',None,'c1'],
        company_period_id=['a','b','c',None,None,'d'],net_sentiment=[np.nan,np.nan,0,np.nan,np.nan,np.nan],
        n_candidates=[0,1,1,np.nan,np.nan,1],n_relevant=[0,0,1,np.nan,np.nan,0],
        state=['no_title_candidates','no_relevant_news','complete_relevant_news','','','partial_or_error_coverage'],
        eligible_primary=[False,False,True,False,False,False]))
    reasons=sentiment_exception_reasons(examples,['c1'])
    assert reasons.tolist()==['no_news_in_period','all_candidates_irrelevant','observed_zero_sentiment',
        'issuer_outside_declared_news_universe','issuer_unmapped_or_ambiguous','incomplete_or_error_classification']
    bad=examples.copy();bad.loc[0,'net_sentiment']=0
    try: sentiment_exception_reasons(bad,['c1'])
    except ValueError: pass
    else: raise AssertionError('No-news zero was accepted')
    return dict(status='PASS',scope='synthetic_only',checks=9)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true',required=True)
    p.parse_args()
    print(json.dumps(self_test(),sort_keys=True))
