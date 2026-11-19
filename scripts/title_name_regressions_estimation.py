"""Absorbed OLS with explicit two-way fund/company cluster covariance."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from title_name_regressions_core import IntegrityError, NotEstimable, require_columns


def _codes(values):
    # pandas 3 no longer accepts a plain Python list here.
    result, levels = pd.factorize(pd.Index(values), sort=False)
    if (result < 0).any():
        raise IntegrityError("Null fixed-effect/cluster identifier")
    return result, len(levels)


def _demean(matrix, codes, count, weights):
    mass = np.bincount(codes, weights=weights, minlength=count)
    if (mass <= 0).any():
        raise IntegrityError("A fixed-effect group has no positive mass")
    for col in range(matrix.shape[1]):
        mean = np.bincount(codes, weights=matrix[:, col] * weights, minlength=count) / mass
        matrix[:, col] -= mean[codes]


def within_transform(matrix, effects, *, weights=None, tolerance=1e-11, max_iterations=5000):
    values = np.asarray(matrix, dtype=float).copy()
    if values.ndim == 1:
        values = values[:, None]
    weights = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise IntegrityError("Within transformation received nonfinite values/nonpositive weights")
    code_groups = [_codes(x) for x in effects]
    scale = max(float(np.max(np.abs(values))), 1.0)
    for iteration in range(1, max_iterations + 1):
        old = values.copy()
        for codes, count in code_groups:
            _demean(values, codes, count, weights)
        if np.max(np.abs(values - old)) <= tolerance * scale:
            return values, iteration
    raise NotEstimable("Fixed-effect projection did not converge")


def fixed_effect_rank(effect1, effect2):
    """Rank of two categorical FE sets, including disconnected components."""
    a, na = _codes(effect1)
    b, nb = _codes(effect2)
    parent = np.arange(na + nb)
    size = np.ones(na + nb, dtype=np.int64)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for x, y in zip(a, b + na):
        x, y = find(x), find(y)
        if x != y:
            if size[x] < size[y]: x, y = y, x
            parent[y] = x
            size[x] += size[y]
    components = len({find(x) for x in range(na + nb)})
    return na + nb - components, components


def prune_singletons(frame, effects):
    out = frame.copy()
    removed = 0
    rounds = 0
    while len(out):
        bad = np.zeros(len(out), dtype=bool)
        for key in effects:
            bad |= out.groupby(key, observed=True)[key].transform("size").to_numpy() <= 1
        if not bad.any(): break
        removed += int(bad.sum())
        rounds += 1
        out = out.loc[~bad].copy()
    return out, removed, rounds


def _cluster_meat(scores, cluster):
    codes, count = _codes(cluster)
    if count <= 1:
        raise NotEstimable("At least two clusters are needed in each cluster dimension")
    grouped = np.zeros((count, scores.shape[1]))
    np.add.at(grouped, codes, scores)
    return grouped.T @ grouped, count


@dataclass
class PanelResult:
    names: list
    beta: np.ndarray
    covariance: np.ndarray
    sample: pd.DataFrame
    residuals: np.ndarray
    diagnostics: dict

    def table(self):
        variance = np.diag(self.covariance)
        inference_ok=np.isfinite(variance)&(variance>0)
        se = np.sqrt(np.where(inference_ok, variance, np.nan))
        t = np.divide(self.beta, se, out=np.full_like(self.beta, np.nan), where=se > 0)
        df = self.diagnostics["inference_df"]
        critical = stats.t.ppf(0.975, df)
        return [dict(term=name, beta=float(b), se=float(s), ci95_low=float(b-critical*s),
                     ci95_high=float(b+critical*s), t=float(z),
                     p=float(2 * stats.t.sf(abs(z), df)),
                     inference_status="ESTIMABLE" if ok else "NOT_ESTIMABLE_COVARIANCE",
                     inference_reason=None if ok else "nonpositive_or_nonfinite_coefficient_variance")
                for name, b, s, z,ok in zip(self.names, self.beta, se, t,inference_ok)]


def fit_panel_fe(frame, outcome, regressors, *, entity="fund_security_id", time="month",
                 fund_cluster="fund_id", company_cluster="company_id", min_clusters=2,
                 allow_absorbed=(), allow_collinear=(), extra_cluster=None, weight=None, drop_singletons=True):
    required = list(dict.fromkeys([outcome] + list(regressors) + [entity, time, fund_cluster, company_cluster]
                                  + ([extra_cluster] if extra_cluster else [])+([weight] if weight else [])
                                  + (["security_id"] if "security_id" in frame else [])))
    require_columns(frame, required, "regression input")
    d = frame.loc[:, required].copy()
    numeric = [outcome] + list(regressors) + ([weight] if weight else [])
    d[numeric] = d[numeric].apply(pd.to_numeric, errors="raise")
    eligible = d[required].notna().all(axis=1) & np.isfinite(d[numeric]).all(axis=1)
    if weight:
        eligible &= d[weight].gt(0)
    d = d.loc[eligible].copy()
    n_complete = len(d)
    d, n_singleton, singleton_rounds = prune_singletons(d, [entity, time]) if drop_singletons else (d, 0, 0)
    if len(d) == 0:
        raise NotEstimable("No observations after complete-case/singleton gates")
    w = d[weight].to_numpy(float) if weight else np.ones(len(d))
    matrix = d[[outcome] + list(regressors)].to_numpy(float)
    projected, iterations = within_transform(matrix, [d[entity], d[time]], weights=w)
    y, x = projected[:, 0], projected[:, 1:]
    raw_scale = np.maximum(np.max(np.abs(matrix[:, 1:]), axis=0), 1.0)
    absorbed = np.max(np.abs(x), axis=0) <= 1e-9 * raw_scale
    absorbed_names = [name for name, flag in zip(regressors, absorbed) if flag]
    unexpected = set(absorbed_names) - set(allow_absorbed)
    if unexpected:
        raise NotEstimable(f"Regressors fully absorbed by fixed effects: {sorted(unexpected)}")
    keep = ~absorbed
    names = [name for name, yes in zip(regressors, keep) if yes]
    x = x[:, keep]
    if not names:
        raise NotEstimable("Every regressor is absorbed")
    xs = x * np.sqrt(w[:, None])
    ys = y * np.sqrt(w)
    collinear_names=[]
    if allow_collinear:
        protected=[j for j,name in enumerate(names) if name not in allow_collinear]
        def scaled_rank(columns):
            if not columns:return 0
            trial=xs[:,columns]
            return np.linalg.matrix_rank(trial/np.maximum(np.linalg.norm(trial,axis=0),1e-30),tol=1e-9)
        if scaled_rank(protected)!=len(protected):
            raise NotEstimable("Protected target regressors are collinear; never drop a target term")
        chosen=protected.copy()
        for j,name in enumerate(names):
            if j in protected:continue
            if scaled_rank(chosen+[j])==len(chosen)+1:chosen.append(j)
            else:collinear_names.append(name)
        chosen=sorted(chosen)
        x=x[:,chosen];xs=xs[:,chosen];names=[names[j] for j in chosen]
    rank = np.linalg.matrix_rank(xs)
    if rank != len(names):
        raise NotEstimable(f"Within design rank {rank} is smaller than {len(names)} regressors")
    fe_rank, components = fixed_effect_rank(d[entity], d[time])
    parameters = fe_rank + rank
    residual_df = len(d) - parameters
    if residual_df <= 0:
        raise NotEstimable("No residual degrees of freedom after fixed effects")
    beta = np.linalg.lstsq(xs, ys, rcond=None)[0]
    residual = y - x @ beta
    scores = x * (w * residual)[:, None]
    meat_f, nf = _cluster_meat(scores, d[fund_cluster])
    meat_c, nc = _cluster_meat(scores, d[company_cluster])
    intersection = pd.MultiIndex.from_frame(d[[fund_cluster, company_cluster]])
    meat_i, ni = _cluster_meat(scores, intersection)
    if min(nf, nc) < min_clusters:
        raise NotEstimable(f"Too few fund/company clusters: {nf}/{nc}, need {min_clusters}")
    correction = (len(d)-1) / residual_df
    meat = correction * (nf/(nf-1)*meat_f + nc/(nc-1)*meat_c - ni/(ni-1)*meat_i)
    extra_counts={}
    inference_df=min(nf,nc)-1
    cluster_dimensions=[fund_cluster,company_cluster]
    if extra_cluster:
        mt,nt=_cluster_meat(scores,d[extra_cluster])
        ft,nft=_cluster_meat(scores,pd.MultiIndex.from_frame(d[[fund_cluster,extra_cluster]]))
        ct,nct=_cluster_meat(scores,pd.MultiIndex.from_frame(d[[company_cluster,extra_cluster]]))
        fct,nfct=_cluster_meat(scores,pd.MultiIndex.from_frame(d[[fund_cluster,company_cluster,extra_cluster]]))
        if nt<min_clusters:raise NotEstimable("Too few clusters in extra cluster dimension")
        meat+=correction*(nt/(nt-1)*mt-nft/(nft-1)*ft-nct/(nct-1)*ct+nfct/(nfct-1)*fct)
        extra_counts=dict(extra_clusters=nt,fund_extra_intersections=nft,company_extra_intersections=nct,
                          triple_intersections=nfct)
        inference_df=min(nf,nc,nt)-1
        cluster_dimensions.append(extra_cluster)
    bread = np.linalg.inv(xs.T @ xs)
    covariance = bread @ meat @ bread
    covariance = (covariance + covariance.T) / 2
    total = float(np.dot(w, y*y))
    diag = dict(n_input=len(frame), n_complete_case=n_complete, nobs=len(d),
                dropped_missing_or_nonfinite=len(frame)-n_complete, singletons_removed=n_singleton,
                singleton_rounds=singleton_rounds, fixed_effects=[entity, time], fe_rank=fe_rank,
                fe_connected_components=components, regressors_rank=int(rank), absorbed=absorbed_names,
                collinear_nuisance_dropped=collinear_names,
                residual_df=residual_df, fund_clusters=nf, company_clusters=nc, intersection_clusters=ni,
                inference_df=inference_df, cluster_dimensions=cluster_dimensions,
                covariance="CGM "+("three-way" if extra_cluster else "two-way")+", CR1 with full absorbed parameter rank; no PSD clipping",
                covariance_min_eigenvalue=float(np.linalg.eigvalsh(covariance).min()),
                negative_coefficient_variances=int((np.diag(covariance) < 0).sum()),
                within_r2=1-float(np.dot(w,residual*residual))/total if total>0 else None,
                within_iterations=iterations, unit_effects=d[entity].nunique(), time_effects=d[time].nunique(),
                funds=d[fund_cluster].nunique(), companies=d[company_cluster].nunique(),
                securities=d.security_id.nunique() if "security_id" in d else None,
                within_regressor_sd={name:float(value) for name,value in zip(names,np.std(x,axis=0,ddof=1))},
                min_period=str(d[time].min()), max_period=str(d[time].max()),**extra_counts)
    return PanelResult(names, beta, covariance, d, residual, diag)


def h2_absolute_reaction_test(result, positive="S_positive", negative="S_negative"):
    """Frozen-spec conservative union-null test; no prespecified ordering."""
    i, j = result.names.index(positive), result.names.index(negative)
    a, b = float(result.beta[i]), float(result.beta[j])
    expected = a > 0 and b < 0
    df = result.diagnostics["inference_df"]
    restrictions = {}
    for label, sign in [("sum",1.0),("diff",-1.0)]:
        gradient = np.zeros(len(result.names))
        gradient[i],gradient[j]=1.0,sign
        contrast=a+sign*b
        variance=float(gradient @ result.covariance @ gradient)
        if not np.isfinite(variance) or variance<=0:
            restrictions[label] = dict(status="NOT_ESTIMABLE_COVARIANCE",contrast=contrast)
        else:
            statistic=contrast**2/variance
            restrictions[label]=dict(status="ESTIMATED",contrast=contrast,se=float(np.sqrt(variance)),
                                     wald_f=float(statistic),p=float(stats.f.sf(statistic,1,df)))
    complete=all(x["status"]=="ESTIMATED" for x in restrictions.values())
    difference=abs(a)-abs(b)
    return dict(status="ESTIMATED" if complete else "NOT_ESTIMABLE_COVARIANCE",
                expected_h1_signs=expected,absolute_effect_difference=difference,
                method="union-null Wald: p_abs=max(p_sum,p_diff); sum alone only interpreted under H1 signs",
                sum_restriction=restrictions["sum"],difference_restriction=restrictions["diff"],
                p_abs=max(x["p"] for x in restrictions.values()) if complete else None,
                df_num=1,df_denom=df,
                implied_larger_absolute_effect="positive" if difference>0 else "negative" if difference<0 else "equal")


def h1_directional_test(result, positive="S_positive", negative="S_negative"):
    df=result.diagnostics["inference_df"]
    values={}
    for name,direction in [(positive,1),(negative,-1)]:
        index=result.names.index(name)
        variance=float(result.covariance[index,index])
        if not np.isfinite(variance) or variance<=0:
            return dict(status="NOT_ESTIMABLE_COVARIANCE")
        t=float(result.beta[index]/np.sqrt(variance))
        values[name]=float(stats.t.sf(direction*t,df))
    return dict(status="ESTIMATED",one_sided=values,
                intersection_union_p=max(values.values()),
                expected_signs=bool(result.beta[result.names.index(positive)]>0 and
                                    result.beta[result.names.index(negative)]<0))


def signal_control_plan(frame,candidates,minimum_fraction=.9):
    require_columns(frame,["Tone","lag_Tone"],"signal construction")
    base=frame.loc[frame.Tone.notna() & frame.lag_Tone.notna()]
    coverage={}
    selected=[]
    for name in candidates:
        fraction=float(np.isfinite(pd.to_numeric(base[name],errors="coerce")).mean()) if name in base and len(base) else 0.0
        coverage[name]=dict(fraction=fraction,observed=int(round(fraction*len(base))),total=len(base),
                            retained=fraction>=minimum_fraction,
                            reason="coverage_pass" if fraction>=minimum_fraction else "below_fixed_90_percent_coverage_gate")
        if fraction>=minimum_fraction:selected.append(name)
    return dict(selected=selected,coverage=coverage,threshold=minimum_fraction,
                denominator="unique company-months with defined current and exact-calendar-lag Tone; before outcome expansion")


def fit_signal(frame,company_controls=(),*,minimum_company_months=6,minimum_companies_month=10,signal_name="S"):
    from title_name_regressions_core import unique_keys
    unique_keys(frame,["company_id","month"],"signal natural unit")
    regressors=["lag_Tone","log_news_count","source_share_rbc","source_share_prime","source_share_kommersant"]+list(company_controls)
    require_columns(frame,["Tone"]+regressors,"signal model")
    numeric=["Tone"]+regressors
    complete=frame[numeric].notna().all(axis=1)&np.isfinite(frame[numeric]).all(axis=1)
    d=frame.loc[complete].copy()
    n_complete=len(d)
    rounds=0
    while len(d):
        company_ok=d.groupby("company_id",observed=True).company_id.transform("size").ge(minimum_company_months)
        month_ok=d.groupby("month",observed=True).company_id.transform("nunique").ge(minimum_companies_month)
        eligible=company_ok&month_ok
        if eligible.all():break
        d=d.loc[eligible].copy()
        rounds+=1
    if not len(d):raise NotEstimable("No company-months after fixed signal-history/cross-section minima")
    values,iterations=within_transform(d[numeric].to_numpy(float),[d.company_id,d.month])
    y,x=values[:,0],values[:,1:]
    kept=[];dropped=[]
    for j,name in enumerate(regressors):
        scale=max(np.linalg.norm(x[:,j]),1.0)
        candidate=x[:,j]/scale
        if np.max(np.abs(candidate))<1e-10:
            dropped.append(dict(column=name,reason="absorbed_or_zero_variance"));continue
        trial=x[:,kept+[j]]
        trial=trial/np.maximum(np.linalg.norm(trial,axis=0),1e-30)
        if np.linalg.matrix_rank(trial,tol=1e-9)<len(kept)+1:
            dropped.append(dict(column=name,reason="collinear_in_fixed_listed_order"));continue
        kept.append(j)
    fe_rank,components=fixed_effect_rank(d.company_id,d.month)
    if len(d)<=fe_rank+len(kept):raise NotEstimable("No signal-model residual degrees of freedom")
    slopes=np.linalg.lstsq(x[:,kept],y,rcond=None)[0] if kept else np.array([])
    residual=y-x[:,kept]@slopes if kept else y.copy()
    sigma=float(np.std(residual,ddof=1))
    if not np.isfinite(sigma) or sigma<=1e-12:raise NotEstimable("Signal residual SD is zero or undefined")
    result=frame.copy()
    result[signal_name]=np.nan
    result.loc[d.index,signal_name]=residual/sigma
    diagnostics=dict(n_months_input=len(frame),n_complete_case=n_complete,n_final=len(d),
        company_groups=d.company_id.nunique(),month_groups=d.month.nunique(),
        minimum_filter_rounds=rounds,fe_rank=fe_rank,fe_components=components,
        regressors=[regressors[j] for j in kept],dropped_nuisance=dropped,
        residual_sigma_ddof1=sigma,within_iterations=iterations,
        standardization_unit="unique company-month, not fund-expanded",generated_regressor_se="conditional, uncertainty not resolved",
        full_sample_retrospective=True,min_period=str(d.month.min()),max_period=str(d.month.max()))
    return result,diagnostics
