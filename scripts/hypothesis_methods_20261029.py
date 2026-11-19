"""Small extensions of the verified v2 estimator; fixed analysis families."""
import json
from pathlib import Path

import numpy as np
from scipy import stats

from title_name_regressions_core import NotEstimable, IntegrityError
from title_name_regressions_estimation import fit_panel_fe, h1_directional_test, h2_absolute_reaction_test

FAMILY_SIZE = {'monthly_DW_reference':4, 'asset_heterogeneity':7, 'relative_allocation':12,
               'exact_disclosure_windows':4, 'full_portfolio_price_AT':8}
CONTROLS = ['opening_weight','lag_log_news_count','opening_log_NAV','opening_HHI']


def clean(value):
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)): return [clean(v) for v in value]
    if isinstance(value,(np.integer,)): return int(value)
    if isinstance(value,(np.bool_,)): return bool(value)
    if isinstance(value,(float,np.floating)): return float(value) if np.isfinite(value) else None
    if hasattr(value,'isoformat'): return value.isoformat()
    if isinstance(value,Path): return str(value)
    return value


def save(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(clean(value),ensure_ascii=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def holm(tests):
    result=[dict(t) for t in tests]
    for family,size in FAMILY_SIZE.items():
        indices=[i for i,t in enumerate(result) if t['family']==family]
        if len(indices)>size: raise IntegrityError('More tests than frozen family size: '+family)
        ranked=sorted(indices,key=lambda i:result[i].get('p') if result[i].get('p') is not None else 1)
        last=0.0
        for rank,index in enumerate(ranked):
            raw=result[index].get('p');value=1.0 if raw is None else raw
            last=max(last,min(1.0,(size-rank)*value))
            result[index]['p_holm']=last
            result[index]['family_size_including_unavailable']=size
    return result


def contrast(result,weights):
    g=np.array([weights.get(name,0.0) for name in result.names])
    beta=float(g@result.beta);variance=float(g@result.covariance@g)
    if not np.isfinite(variance) or variance<=0:
        return dict(beta=beta,status='NOT_ESTIMABLE_COVARIANCE',p=None,se=None,ci95_low=None,ci95_high=None)
    se=float(np.sqrt(variance));df=result.diagnostics['inference_df'];critical=stats.t.ppf(.975,df)
    return dict(beta=beta,se=se,ci95_low=beta-critical*se,ci95_high=beta+critical*se,
                p=float(2*stats.t.sf(abs(beta/se),df)),status='ESTIMABLE')


def joint_interactions(result):
    indices=[result.names.index(name) for name in ['S_positive_equity','S_negative_equity']]
    b=result.beta[indices];cov=result.covariance[np.ix_(indices,indices)]
    if not np.isfinite(cov).all() or np.linalg.eigvalsh(cov).min()<=0:
        return dict(status='NOT_ESTIMABLE_COVARIANCE',p=None)
    f=float(b@np.linalg.solve(cov,b)/2)
    return dict(status='ESTIMABLE',F=f,df_num=2,df_denom=result.diagnostics['inference_df'],
                p=float(stats.f.sf(f,2,result.diagnostics['inference_df'])))


def model(frame,*,identifier,family,outcome='DW',relative=False,interaction=False,extra_controls=(),signal='S'):
    d=frame.copy().reset_index(drop=True)
    if d.index.has_duplicates: raise IntegrityError('Model index is not unique')
    d['S_positive']=d[signal].clip(lower=0)
    d['S_negative']=(-d[signal]).clip(lower=0)
    targets=['S_positive','S_negative']
    if interaction:
        d['S_positive_equity']=d.S_positive*d.asset_class.eq('equity')
        d['S_negative_equity']=d.S_negative*d.asset_class.eq('equity')
        targets+=['S_positive_equity','S_negative_equity']
    controls=CONTROLS+list(extra_controls)
    record=dict(model=identifier,family=family,outcome=outcome,candidate_rows=len(d),
                status='NOT_ESTIMABLE',tests=[],coefficients=[])
    try:
        result=fit_panel_fe(d,outcome,targets+controls,time='interval_id' if relative else 'month',
                            allow_absorbed=controls,allow_collinear=controls)
    except NotEstimable as error:
        record['reason']=str(error)
        count=7 if interaction else 4
        record['tests']=[dict(model=identifier,family=family,test='unavailable_'+str(i),p=None) for i in range(count)]
        return record,None
    selected=d.loc[result.sample.index]
    diag=result.diagnostics
    diag.update(calendar_months=int(selected.month.nunique()),period_start=str(selected.month.min()),
                period_end=str(selected.month.max()),estimator_sample_rows=len(selected),
                conditional_generated_regressor_inference=True)
    record.update(status='ESTIMATED_POSTHOC_EXPLORATORY',diagnostics=diag)
    scale=100 if outcome=='price_adjusted_value_growth' else 10000
    unit='100 log points' if scale==100 else 'basis points of gross-asset weight'
    for row in result.table():
        row.update(model=identifier,family=family,outcome=outcome,display_scale=scale,display_units=unit,
                   effect_display=row['beta']*scale,se_display=row['se']*scale,
                   ci95_low_display=row['ci95_low']*scale,ci95_high_display=row['ci95_high']*scale)
        record['coefficients'].append(row)
    def test(name,p,**other):
        record['tests'].append(dict(model=identifier,family=family,test=name,p=p,**other))
    if interaction:
        joint=joint_interactions(result);record['joint_asset_interaction']=joint
        test('joint_zero_equity_interactions',joint.get('p'))
        for name in ['S_positive_equity','S_negative_equity']:
            value=contrast(result,{name:1});test(name,value['p'])
        record['asset_slopes']={}
        for asset in ['bond','equity']:
            for term in ['S_positive','S_negative']:
                weights={term:1}
                if asset=='equity': weights[term+'_equity']=1
                value=contrast(result,weights);record['asset_slopes'][asset+'_'+term]=value
                test(asset+'_'+term,value['p'])
    else:
        h1=h1_directional_test(result);h2=h2_absolute_reaction_test(result)
        record.update(H1_directional=h1,H2_absolute_asymmetry=h2)
        for name in ['S_positive','S_negative']:
            value=contrast(result,{name:1});test(name,value['p'])
        test('H1_directional_intersection_union',h1.get('intersection_union_p'))
        test('H2_absolute_union_null',h2.get('p_abs'))
    return record,result


def mde(result):
    if result is None: return dict(status='NOT_ESTIMABLE')
    df=result.diagnostics['inference_df'];out={}
    for name in ['S_positive','S_negative']:
        value=contrast(result,{name:1});se=value['se']
        out[name]={str(power):float((stats.t.ppf(.975,df)+stats.norm.ppf(power))*se*10000)
                   for power in [.8,.9]} if se is not None else None
    return dict(status='APPROXIMATE_DESIGN_SENSITIVITY',units='basis points per one unit standardized S',
                method='(t_0.975,cluster_df + z_power)*conditional clustered SE; not observed post-hoc power',
                cluster_df=df,minimum_detectable_effect=out)
