"""Plot saved H1 quantity changes and partition a fixed extreme-growth rule.

Actual input processing is Windows-only. No regressions, SSH, news queries,
source mutations, winsorization, or percentile-selected cutoffs are performed.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


SOURCE = './analysis_outputs/h1_macro_no_time_20261031/samples/NET_MACRO_COMMON.parquet'
OUTPUT = './analysis_outputs/h1_quantity_outliers_20261102'
SOURCE_SHA256 = '79d54cb284a476f0cec1bdf7b222a2c1f3f9aea09817a455693c270aa44b4e1a'
EXCLUDE_ABOVE_PCT = 10000.0
CENTRAL_LOW, CENTRAL_HIGH = -100.0, 500.0
QUANTILES = [0, .001, .01, .05, .25, .5, .75, .95, .99, .999, 1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
        allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def row_hash(frame):
    return hashlib.sha256('\n'.join(frame.row_id.astype(str)).encode()).hexdigest()


def signed_log(value):
    value = np.asarray(value, dtype=float)
    return np.sign(value) * np.log10(1 + np.abs(value))


def inverse_signed_log(value):
    value = np.asarray(value, dtype=float)
    return np.sign(value) * np.expm1(np.log(10) * np.abs(value))


def partition(frame):
    required = {'row_id', 'fund_id', 'security_id', 'start_date', 'end_date',
        'quantity_open', 'quantity_close', 'quantity_change_pct', 'net_sentiment'}
    if required - set(frame):
        raise ValueError(f'Missing columns: {sorted(required-set(frame))}')
    if frame.row_id.isna().any() or frame.row_id.duplicated().any():
        raise ValueError('Source row IDs must be present and unique')
    values = pd.to_numeric(frame.quantity_change_pct, errors='coerce')
    if not np.isfinite(values.to_numpy(float)).all():
        raise ValueError('Source primary sample contains nonfinite outcomes')
    opening=pd.to_numeric(frame.quantity_open,errors='coerce').to_numpy(float)
    closing=pd.to_numeric(frame.quantity_close,errors='coerce').to_numpy(float)
    if not np.isfinite(opening).all() or not np.isfinite(closing).all() or \
            (opening<=0).any() or (closing<0).any():
        raise ValueError('Source quantities require finite Q0>0 and finite Q1>=0')
    calculated=100*(closing/opening-1)
    if not np.isclose(values.to_numpy(float),calculated,rtol=1e-9,atol=1e-7).all():
        raise ValueError('Stored outcome differs from 100*(Q1/Q0-1)')
    if values.lt(-100-1e-8).any():
        raise ValueError('A primary relative-quantity outcome is below the valid -100% bound')
    start = pd.to_datetime(frame.start_date, errors='raise')
    end = pd.to_datetime(frame.end_date, errors='raise')
    days = (end-start).dt.days
    if start.isna().any() or end.isna().any() or days.le(0).any():
        raise ValueError('Invalid actual reporting interval dates')
    month_difference = (end.dt.year-start.dt.year)*12 + end.dt.month-start.dt.month
    monthly = month_difference.eq(1) & days.between(20,40,inclusive='both')
    flagged = frame.copy()
    flagged['actual_interval_days'] = days.to_numpy()
    flagged['consecutive_calendar_months'] = month_difference.eq(1).to_numpy()
    flagged['is_monthly_20_40_days'] = monthly.to_numpy()
    flagged['extreme_quantity_growth_excluded'] = values.gt(EXCLUDE_ABOVE_PCT).to_numpy()
    flagged['outlier_exclusion_reason'] = np.where(values.gt(EXCLUDE_ABOVE_PCT),
        'quantity_change_pct > 10000; fixed before this cleanup', '')
    if not flagged[list(frame)].equals(frame):
        raise AssertionError('Original source values changed while adding flags')
    retained = flagged.loc[~flagged.extreme_quantity_growth_excluded].copy()
    excluded = flagged.loc[flagged.extreme_quantity_growth_excluded].copy()
    if set(retained.row_id) & set(excluded.row_id):
        raise AssertionError('Clean and excluded sets overlap')
    if set(retained.row_id) | set(excluded.row_id) != set(frame.row_id):
        raise AssertionError('Partition does not cover the source')
    if len(retained)+len(excluded) != len(frame):
        raise AssertionError('Partition row counts changed')
    if not flagged.quantity_change_pct.eq(-100).equals(
            flagged.quantity_change_pct.eq(-100) & ~flagged.extreme_quantity_growth_excluded):
        raise AssertionError('A valid -100% exit was excluded')
    return flagged, retained, excluded


def describe(frame):
    values = frame.quantity_change_pct.astype(float)
    def finite(value):
        return float(value) if pd.notna(value) and math.isfinite(float(value)) else None
    record=dict(rows=len(frame), funds=int(frame.fund_id.nunique()),
        companies=int(frame.company_id.nunique()) if 'company_id' in frame else None,
        quantiles={str(q):finite(values.quantile(q)) for q in QUANTILES},
        mean=finite(values.mean()),std=finite(values.std()),
        negative_rows=int(values.lt(0).sum()),zero_rows=int(values.eq(0).sum()),
        positive_rows=int(values.gt(0).sum()),complete_exit_minus100_rows=int(values.eq(-100).sum()),
        greater_than_percent={str(int(cutoff)):int(values.gt(cutoff).sum())
            for cutoff in [100,500,1000,10000]},
        note='Threshold counts are descriptive; only the fixed >10000% rule partitions the data.')
    if 'actual_interval_days' in frame:
        days=frame.actual_interval_days
        record['actual_interval_days']=dict(min=finite(days.min()),median=finite(days.median()),
            p25=finite(days.quantile(.25)),p75=finite(days.quantile(.75)),max=finite(days.max()))
    if 'opening_weight' in frame:
        weights=pd.to_numeric(frame.opening_weight,errors='coerce')
        record['positive_opening_gross_asset_weight_below']={str(cutoff):
            int((weights.gt(0)&weights.lt(cutoff)).sum()) for cutoff in [1e-6,1e-4,.001]}
        record['tiny_weight_counts_are_descriptive_not_exclusion_rules']=True
    return record


def histogram(values, scale='linear', edges=None, bins=80):
    values = np.asarray(values, dtype=float)
    transformed = signed_log(values) if scale=='signed_log10_1p' else values
    if edges is None:
        low=float(transformed.min()) if len(values) else -1
        high=float(transformed.max()) if len(values) else 1
        if low==high:
            low-=.5;high+=.5
        edges=np.linspace(np.nextafter(low,-np.inf),np.nextafter(high,np.inf),bins+1)
    edges=np.asarray(edges,float)
    counts,_=np.histogram(transformed,bins=edges)
    below=int((transformed<edges[0]).sum())
    above=int((transformed>edges[-1]).sum())
    if int(counts.sum())+below+above!=len(values):
        raise AssertionError('Histogram accounting failed')
    raw_edges=inverse_signed_log(edges) if scale=='signed_log10_1p' else edges
    return dict(scale=scale, rows=len(values), counts=counts.astype(int).tolist(),
        edges_transformed=edges.tolist(),edges_raw_pct=raw_edges.tolist(),
        below_visible_range=below,above_visible_range=above,
        histogram_rows=int(counts.sum()),
        density=False,scale_note=('Uniform bins in sign(y)*log10(1+abs(y)); raw-percent tick labels. Bin widths differ on the raw percentage scale.'
            if scale=='signed_log10_1p' else 'Uniform raw-percentage bins; rightmost bin includes its right edge.'))


def export_bins(root, name, record):
    with (root/(name+'.bins.csv')).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['bin','left_raw_pct','right_raw_pct','left_transformed',
            'right_transformed','count','fraction_of_population'])
        for i,count in enumerate(record['counts']):
            writer.writerow([i,record['edges_raw_pct'][i],record['edges_raw_pct'][i+1],
                record['edges_transformed'][i],record['edges_transformed'][i+1],
                count,count/record['rows'] if record['rows'] else 0])


def plotting(root, populations):
    os.environ.setdefault('MPLCONFIGDIR',str(root/'matplotlib_config'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    def values(name):
        return populations[name].quantity_change_pct.to_numpy(float)
    central_edges=np.arange(CENTRAL_LOW,CENTRAL_HIGH+1,10)
    bins={
        'monthly_full_central':histogram(values('monthly_full'),'linear',central_edges),
        'monthly_full_signed_log':histogram(values('monthly_full'),'signed_log10_1p'),
        'monthly_retained_signed_log':histogram(values('monthly_retained'),'signed_log10_1p'),
        'all_full_signed_log':histogram(values('all_full'),'signed_log10_1p'),
    }
    bins['all_retained_signed_log']=histogram(values('all_retained'),'signed_log10_1p',
        bins['all_full_signed_log']['edges_transformed'])
    bins['all_full_central']=histogram(values('all_full'),'linear',central_edges)
    bins['all_retained_central']=histogram(values('all_retained'),'linear',central_edges)
    def label_percent(value):
        if abs(value)>=1e6:
            return f'{value/1e6:g} млн'
        return f'{value:g}'
    def draw(axis, record, title, *, color='#27688c',log_frequency=False):
        edge=np.array(record['edges_transformed'])
        axis.bar(edge[:-1],record['counts'],width=np.diff(edge),align='edge',
            color=color,edgecolor='white',linewidth=.35)
        axis.set_xlim(edge[0],edge[-1])
        axis.set_title(title,pad=12,fontweight='bold')
        axis.set_ylabel('Число наблюдений'+(' (лог. шкала)' if log_frequency else ''))
        axis.set_xlabel('Изменение расчётного количества, %')
        if log_frequency:
            axis.set_yscale('log');axis.set_ylim(bottom=.8)
        axis.grid(axis='y',alpha=.18)
        if record['scale']=='signed_log10_1p':
            ticks=np.array([-100,-10,0,10,100,1000,10000,100000,1e6,1e7,1e8])
            loc=signed_log(ticks)
            keep=(loc>=edge[0])&(loc<=edge[-1])
            axis.set_xticks(loc[keep],[label_percent(v) for v in ticks[keep]],rotation=35,ha='right')
        axis.text(.98,.97,f'N = {record["rows"]:,}'.replace(',',' '),transform=axis.transAxes,
            ha='right',va='top',bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
    fig,axes=plt.subplots(1,3,figsize=(17,5.8))
    draw(axes[0],bins['monthly_full_central'],'Все месячные: центральный участок')
    draw(axes[1],bins['monthly_full_signed_log'],'Все месячные: весь диапазон',log_frequency=True)
    draw(axes[2],bins['monthly_retained_signed_log'],'Месячные после исключения >10 000%',
        color='#2a7c63',log_frequency=True)
    tail=bins['monthly_full_central']
    excluded_monthly=len(populations['monthly_full'])-len(populations['monthly_retained'])
    exits=int(populations['monthly_full'].quantity_change_pct.eq(-100).sum())
    fig.suptitle('Изменение количества бумаг между соседними отчётными месяцами',fontsize=16,fontweight='bold',y=.99)
    fig.text(.02,.02,'Месячные = соседние календарные месяцы и 20–40 дней. Не годовая доходность и не пересчёт квартальных интервалов.\n'
        f'Левая панель — только масштаб: за её границами {tail["below_visible_range"]} строк слева и {tail["above_visible_range"]} справа. '
        f'Действительно исключено месячных: {excluded_monthly}. Полные выходы −100% сохранены: {exits}.\n'
        'Средняя и правая панели: signed-log по оси X, подписи в исходных %; частоты по логарифмической оси Y.\n'
        'Исключение диагностическое: эти строки не признаны доказанными ошибками; регрессия здесь не переоценивается.',fontsize=9)
    fig.tight_layout(rect=[0,.17,1,.94])
    for suffix in ['png','svg']:
        fig.savefig(root/('monthly_quantity_distribution.'+suffix),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5.8))
    before=bins['all_full_signed_log'];after=bins['all_retained_signed_log']
    edge=np.array(before['edges_transformed'])
    axes[0].stairs(before['counts'],edge,fill=True,color='#a4b2bf',alpha=.65,label='Все исходные')
    axes[0].stairs(after['counts'],edge,color='#246b53',linewidth=1.5,label='После фиксированного исключения')
    axes[0].set_yscale('log');axes[0].set_ylim(bottom=.8)
    axes[0].axvline(float(signed_log(EXCLUDE_ABOVE_PCT)),color='#b34335',ls='--',lw=1,label='Граница 10 000%')
    ticks=np.array([-100,-10,0,10,100,1000,10000,100000,1e6,1e7,1e8])
    tickloc=signed_log(ticks);keep=(tickloc>=edge[0])&(tickloc<=edge[-1])
    axes[0].set_xticks(tickloc[keep],[label_percent(v) for v in ticks[keep]],rotation=35,ha='right')
    axes[0].set_xlabel('Изменение количества, %; signed-log шкала')
    axes[0].set_ylabel('Число наблюдений (лог. шкала)')
    axes[0].set_title('Все фактические интервалы: общий диапазон',fontweight='bold')
    axes[0].legend(fontsize=8)
    draw(axes[1],bins['all_full_central'],'Центральный участок до и после совпадает')
    axes[1].stairs(bins['all_retained_central']['counts'],central_edges,color='#246b53',linewidth=1.4)
    fig.suptitle('Проверка фиксированного исключения экстремальных приростов',fontsize=15,fontweight='bold')
    fig.text(.02,.02,f'Все интервалы: {len(populations["all_full"]):,}; сохранено {len(populations["all_retained"]):,}; '
        f'исключено {len(populations["all_full"])-len(populations["all_retained"]):,}. '.replace(',',' ')+
        'Здесь совместно показаны месячные, квартальные и прочие реальные интервалы; они не приведены к месячной длине.\n'
        'Исключение: только Y > 10 000%. Значения не обрезаются до границы; Y = −100% и Y = 10 000% остаются.\n'
        'Диагностический порог: исключённые строки не признаны доказанными ошибками; новых регрессий здесь нет.',fontsize=9)
    fig.tight_layout(rect=[0,.16,1,.94])
    for suffix in ['png','svg']:
        fig.savefig(root/('all_intervals_before_after.'+suffix),dpi=180)
    plt.close(fig)
    for name,record in bins.items():
        export_bins(root,name,record)
    save(root/'histogram_bins.json',bins)
    return bins


def execute(args):
    if os.name!='nt':
        raise RuntimeError('Actual research-data processing must execute on research-host, not Mac')
    source=Path(args.source);root=Path(args.output)
    if 'databaseresources' in [part.lower() for part in root.parts]:
        raise ValueError('Output path is within protected inputs')
    root.mkdir(parents=True,exist_ok=True)
    if (root/'completion.json').exists():
        raise FileExistsError('A completed output version cannot be overwritten')
    before=sha(source)
    if before!=args.source_sha:
        raise ValueError('Frozen source SHA256 mismatch')
    if source.resolve().parent==root.resolve():
        raise ValueError('Source must be outside the new output root')
    import importlib.util
    if importlib.util.find_spec('matplotlib') is None:
        raise ImportError('matplotlib is required in the selected plotting runtime')
    frame=pd.read_parquet(source)
    if len(frame)!=args.expected_rows:
        raise ValueError('Frozen source row count mismatch')
    full,retained,excluded=partition(frame)
    populations={'all_full':full,'all_retained':retained,
        'monthly_full':full.loc[full.is_monthly_20_40_days],
        'monthly_retained':retained.loc[retained.is_monthly_20_40_days]}
    if populations['monthly_full'].empty:
        raise ValueError('There are no intervals meeting the frozen monthly definition')
    full.to_parquet(root/'full_main_with_flags.parquet',index=False,compression='zstd')
    retained.to_parquet(root/'cleaned_main.parquet',index=False,compression='zstd')
    retained.to_csv(root/'cleaned_main.csv.gz',index=False,encoding='utf-8-sig',compression='gzip')
    detail=[c for c in ['fund_name','fund_id','asset_name','security_id','isin','company',
        'company_id','start_date','end_date','actual_interval_days','is_monthly_20_40_days',
        'quantity_open','quantity_close','quantity_change_pct','opening_weight','closing_weight',
        'net_sentiment','fund_focus','index_status','outlier_exclusion_reason','row_id'] if c in excluded]
    excluded[detail].to_csv(root/'excluded_rows.csv',index=False,encoding='utf-8-sig')
    excluded.to_parquet(root/'excluded_rows.parquet',index=False,compression='zstd')
    statistics={name:describe(population) for name,population in populations.items()}
    save(root/'distribution_statistics.json',statistics)
    with (root/'quantiles.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['population','rows','quantile','quantity_change_pct'])
        for name,record in statistics.items():
            for q,value in record['quantiles'].items():writer.writerow([name,record['rows'],q,value])
    bins=plotting(root,populations)
    if sha(source)!=before:
        raise AssertionError('Source file changed during processing')
    report=dict(status='COMPLETE',created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source=dict(path=str(source),sha256=before,rows=len(frame),row_hash=row_hash(frame)),
        fixed_rule=dict(exclude='quantity_change_pct > 10000',cutoff_percent=EXCLUDE_ABOVE_PCT,
            cutoff_selected_from_existing_diagnostic_flag=True,p_values_used_to_choose_cutoff=False,
            exploratory_data_screening=True,excluded_rows_not_verified_errors=True,
            winsorization=False,lower_bound_filter=False,valid_minus100_exits_retained=True),
        monthly_definition='Calendar month number differs by exactly1 AND actual interval length is20–40days inclusive; no annualization',
        counts=dict(source=len(full),retained=len(retained),excluded=len(excluded),
            monthly=len(populations['monthly_full']),monthly_retained=len(populations['monthly_retained']),
            monthly_excluded=len(populations['monthly_full'])-len(populations['monthly_retained']),
            other_actual_intervals=len(full)-len(populations['monthly_full'])),
        partition=dict(disjoint=True,source_row_ids_preserved=True,
            retained_row_hash=row_hash(retained),excluded_row_hash=row_hash(excluded)),
        statistics=statistics,source_unchanged=True,new_fits=False,
        figures=['monthly_quantity_distribution.png','monthly_quantity_distribution.svg',
            'all_intervals_before_after.png','all_intervals_before_after.svg'],
        histogram_populations={k:dict(rows=v['rows'],histogram_rows=v['histogram_rows'],
            below=v['below_visible_range'],above=v['above_visible_range']) for k,v in bins.items()})
    save(root/'outlier_partition_report.json',report)
    save(root/'completion.json',dict(status='COMPLETE',source_sha256=before,
        **report['counts'],figures=report['figures']))
    print(json.dumps(dict(status='COMPLETE',**report['counts']),sort_keys=True))


def fixture():
    rows=[]
    for number,(start,end,value) in enumerate([
        ('2020-01-31','2020-02-29',-100),('2020-01-31','2020-02-29',0),
        ('2020-01-31','2020-02-29',500),('2020-01-31','2020-02-29',10000),
        ('2020-01-31','2020-02-29',10001),('2020-01-31','2020-04-30',1e8),
        ('2020-01-01','2020-02-29',10)]):
        rows.append(dict(row_id=str(number),fund_id='f',security_id=str(number),
            start_date=start,end_date=end,quantity_open=1,quantity_close=1+value/100,
            quantity_change_pct=value,net_sentiment=0))
    source=pd.DataFrame(rows);full,retained,excluded=partition(source)
    assert len(retained)==5 and len(excluded)==2
    assert full.is_monthly_20_40_days.sum()==5
    assert excluded.is_monthly_20_40_days.sum()==1
    assert retained.quantity_change_pct.eq(-100).sum()==1
    assert retained.quantity_change_pct.eq(10000).sum()==1
    central=histogram(full.quantity_change_pct,'linear',np.arange(-100,501,10))
    assert central['histogram_rows']==4 and central['above_visible_range']==3
    record=histogram(full.quantity_change_pct,'signed_log10_1p')
    assert sum(record['counts'])==len(full)
    assert np.allclose(inverse_signed_log(signed_log([-100,0,1,10000,1e8])),[-100,0,1,10000,1e8])
    print(json.dumps(dict(status='PASS',scope='synthetic_only',checks=8)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default=SOURCE)
    parser.add_argument('--source-sha',default=SOURCE_SHA256)
    parser.add_argument('--expected-rows',type=int,default=213612)
    parser.add_argument('--output',default=OUTPUT)
    parser.add_argument('--fixture-only',action='store_true')
    args=parser.parse_args()
    fixture() if args.fixture_only else execute(args)
