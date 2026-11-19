"""Compile a methodological review from sealed small H3 output tables.

No estimation, data selection, external requests or changes to input releases.
All empirical values below are copied from the existing versioned tables.
"""
from pathlib import Path
import csv
import datetime as dt
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
SPEC_SHA = '565a726fd3858b94b2bfb3dbf43442d8cab4812c79749fd2ea24db806c44e811'
RUNS = ['run_bf23bf409aff708f', 'run_d0440fad412ee5e9']

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read_csv(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def number(x):
    if x in (None, '', 'nan', 'NaN'):
        return None
    y = float(x)
    return y if math.isfinite(y) else None

def coefficient(row):
    result = {k: number(row[k]) for k in
              ['beta','SE','p','CI_low','CI_high','IQR','IQR_effect_bps','native_residual_df_p']}
    result['bps_per_one_score'] = result['beta'] * 10000
    result['CI_bps_per_one_score'] = [result['CI_low'] * 10000, result['CI_high'] * 10000]
    result['IQR_CI_bps'] = [result['CI_low'] * result['IQR'] * 10000,
                            result['CI_high'] * result['IQR'] * 10000]
    return result

def review_run(name):
    p = ROOT / 'outputs/h3_alpha_20261104' / name
    summary = json.loads((p/'summary.json').read_text())
    manifest = json.loads((p/'manifest.json').read_text())
    assert summary['spec_sha256'] == SPEC_SHA
    sources = {}
    for filename in ['input_manifest.json','summary.json','models.csv','coefficients.csv','tests.csv']:
        digest = sha(p/filename)
        assert digest == manifest['files'][filename]['sha256'], filename
        sources[filename] = {'local_path':str(p/filename), 'sha256':digest}
    models = read_csv(p/'models.csv')
    coefficients = read_csv(p/'coefficients.csv')
    tests = read_csv(p/'tests.csv')
    assert len(models) == 114 and len({x['model_id'] for x in models}) == 114
    cells = []
    for model in models:
        if not (model['branch'] == 'UNIT_VALUE_KNOWN_ADJUSTMENTS'
                and model['sensitivity'] == 'PRIMARY' and model['variant'] == 'CURRENT_PLUS_LAGNEWS'):
            continue
        cid = model['model_id']
        comp = next(x for x in models if x['branch'] == model['branch']
                    and x['sensitivity'] == 'PRIMARY' and x['group'] == model['group']
                    and x['horizon'] == model['horizon'] and x['variant'] == 'CURRENT_COMMON')
        assert comp['sample_hash'] == model['sample_hash']
        cr = next(x for x in coefficients if x['model_id'] == cid and x['term'] == 'NTS_Q_current')
        lr = next(x for x in coefficients if x['model_id'] == cid and x['term'] == 'NTS_Q_lagnews')
        co = next(x for x in coefficients if x['model_id'] == comp['model_id'] and x['term'] == 'NTS_Q_current')
        su = next(x for x in tests if x['model_id'] == cid and x['test'] == 'current+lagnews=0')
        jo = next(x for x in tests if x['model_id'] == cid and x['test'] == 'joint_current_lagnews=0')
        assert int(model['reference_df']) == int(model['funds'])-1
        cells.append(dict(group=model['group'], horizon_months=int(model['horizon']),
                          model_id=cid, nobs=int(model['nobs']), funds=int(model['funds']),
                          reference_df=int(model['reference_df']), sample_hash=model['sample_hash'],
                          within_R2=number(model['within_R2']), current=coefficient(cr),
                          lagnews=coefficient(lr), current_only_same_sample=coefficient(co),
                          sum_test={k:number(su[k]) for k in ['beta','SE','p','CI_low','CI_high']},
                          joint_test={k:number(jo[k]) for k in ['F','df_num','df_denom','p']}))
    assert len(cells) == 12
    qa = []
    for f in p.glob('qa/*/qa.json'):
        d = json.loads(f.read_text())
        qa.append({k:d.get(k) for k in ['status','scope','run_manifest_sha256','checks_pass','checks_fail']}
                  | {'path':str(f),'sha256':sha(f)})
    result = dict(run_id=name, input_release=summary.get('fund_release','v1_existing_cache'),
                  manifest_path=str(p/'manifest.json'), manifest_sha256=sha(p/'manifest.json'),
                  spec_sha256=SPEC_SHA, input_manifest=summary['input_manifest'],
                  small_source_files=sources, source_funds=summary['source_funds'],
                  selected_origins=summary['selected_origins'],
                  primary_NTS_origins=summary['primary_NTS_origins'],
                  primary_NTS_origins_in_mandate_universe=summary['primary_NTS_origins_in_mandate_universe'],
                  monthly_returns=summary['monthly_returns'],
                  primary_TR_status=summary['primary_TR_H3_status'],
                  primary_TR_baseline_cells=[{k:r[k] for k in ['group','horizon','variant','status','nobs','funds','reason']}
                    for r in models if r['branch']=='TR_VERIFIED' and r['sensitivity']=='PRIMARY'],
                  registry_labels=len(models),
                  diagnostic_estimated_labels=sum(r['branch']=='UNIT_VALUE_KNOWN_ADJUSTMENTS' and r['status']=='ESTIMATED' for r in models),
                  nonestimable_TR_labels=sum(r['branch']=='TR_VERIFIED' and r['status']=='NOT_ESTIMABLE' for r in models),
                  diagnostic_4_by_3_distributed=cells,
                  executor_QA=qa,
                  external_numerical_QA='Separate worker task; not duplicated or certified by this methodological review.')
    result['nominal_5pct_counts_distributed'] = {
        'positive_current':sum(c['current']['beta']>0 and c['current']['p']<.05 for c in cells),
        'negative_current':sum(c['current']['beta']<0 and c['current']['p']<.05 for c in cells),
        'positive_lagnews':sum(c['lagnews']['beta']>0 and c['lagnews']['p']<.05 for c in cells),
        'sum_tests':sum(c['sum_test']['p']<.05 for c in cells),
        'joint_tests':sum(c['joint_test']['p']<.05 for c in cells)}
    for filename,key in [('coverage_by_group_horizon.csv','actual_coverage'),
                         ('comparison_v1_v3_coverage_and_overlap.csv','version_coverage_comparison')]:
        if (p/filename).exists():
            result[key]=read_csv(p/filename)
            result['small_source_files'][filename]={'local_path':str(p/filename),'sha256':sha(p/filename)}
    return result

def main():
    spec = ROOT/'config/h3_alpha_spec_20261104.json'
    assert sha(spec) == SPEC_SHA
    runs = [review_run(name) for name in RUNS]
    report = {
      'review_id':'H3_ALPHA_METHOD_AND_INTERPRETATION_20261104_V1_WITH_V3_UPDATE',
      'created_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
      'status':'METHOD_REVIEW_COMPLETED_DIAGNOSTIC_ONLY_EXTERNAL_NUMERICAL_QA_SEPARATE',
      'spec_changed':False,
      'new_regressions_run_by_reviewer':False,
      'review_basis':'Frozen specification, hash-pinned source code, sealed small output tables and documentary return-source review; not an independent re-estimation of the numerical pipeline.',
      'source_eligibility_decision_order':'Arsagera documentary eligibility was finalized before this reviewer opened new H3 coefficient tables. Criteria were not changed in response to either run.',
      'current_reviewed_run':RUNS[-1],
      'current_run_manifest_sha256':runs[-1]['manifest_sha256'],
      'verdict_ru':'Проверенные компоненты реализации соответствуют зафиксированной практической спецификации NTS_Q и past-only factor adjustment. В обоих прогонах рассчитана диагностическая доходность стоимости пая, а не подтверждённая полная доходность; основная TR H3 пока не оценена. В расширенной v3 все 12 текущих коэффициентов, 12 коэффициентов согласованности с предыдущими новостями, 12 сумм и 12 совместных тестов незначимы на обычном двустороннем 5% уровне. Это не доказательство отсутствия экономически значимой связи и не результат для verified total-return alpha.',
      'method_checks':[
        {'item':'Frozen specification and provenance','status':'PASS_REVIEWED',
         'evidence':'Same spec SHA in both manifests; NTS/core/FE kernels unchanged. Five small source files per run match the sealed manifest; 114 unique labels in each run. v3 changes the dated history/action adapter and comparison/audit output, not the hypothesis, gates, controls or inference.'},
        {'item':'NTS_Q meaning and units','status':'PASS_WITH_MEASUREMENT_LIMITS',
         'evidence':'Continuing positions/exits use estimated quantity change in compatible opening units times opening RUB price. Confirmed entries use closing RUB position value. No sum of heterogeneous raw share/bond counts and no artificial H1 +100% entry growth in the aggregation.',
         'limit':'Q is derived from value/price; changes can reflect fund flows, corporate actions, clean-bond-price/NKD conventions and measurement. It is not observed execution or the original whole-portfolio passive-revaluation-adjusted AT.'},
        {'item':'Current and previous-news scores','status':'PASS_REVIEWED',
         'evidence':'Both scores use the same current quantity-value changes and identical eligible security set E. Lagnews uses exact previous report-interval news, not previous trading and not a shift of surviving regression rows. Observed Net=0 remains valid; missing scores/quantities are not zero-filled.'},
        {'item':'Coverage and origin gates','status':'PASS_WITH_SELECTION_LIMITS',
         'evidence':'Coverage denominator U includes all measurable union-held corporate positions, including no-news and outside-target positions. Each primary score needs two contributing companies and >=0.5 coverage; scalar denominator tolerance is relative to gross E activity. Latest closing origin is chosen before eligibility; no fallback.',
         'limit':'Coverage is of measurable corporate changes, not all true fund trades. Asset-value coverage uses positive disclosed positions, not independently reconciled NAV. Foreign shares remain unknown when documentary classification is absent; inherited H1 P99 selection remains exploratory history.'},
        {'item':'Fund-return basis and events','status':'DIAGNOSTIC_ONLY',
         'evidence':'Observed adjacent positive RUB unit values, dated identity and source endpoint freshness are required; total fund NAV growth is not used. v1 excludes entire known-cash-event fund series; v3 uses the separately sealed source policy excluding announced record/payment months and retains unknown ex-date/completeness caveats. Known fund214 split is not adjusted twice.',
         'limit':'The narrower v3 event screen is not proof that all remaining months are distribution-free. Both runs correctly retain zero TR-verified months. Published-return sources require a separate basis-aware adapter, not promotion of raw NAV.'},
        {'item':'Past-only risk estimation','status':'PASS_REVIEWED',
         'evidence':'Training ends at the last calendar month-end <= actual report date, with 60-month lookback/min36/last12>=9. Source endpoint dates must also precede the origin. Intercept and full-rank/standardized-condition<=10000 checks are present. May29 cannot consume the complete May return.'},
        {'item':'Forward outcome','status':'PASS_REVIEWED',
         'evidence':'Horizons are the next h full calendar months, h=1/2/3, with complete return/factor rows and first actual unit-value anchor >=origin. Betas are fixed within a future window. Future alpha is average monthly excess return less factor exposures; the fitted past intercept is not subtracted and remains the past_alpha control.',
         'limit':'The origin-month tail is excluded. This is not cumulative h-month return, annual alpha or real-time investability given unknown publication/revision vintages.'},
        {'item':'Risk benchmarks','status':'PASS_WITH_MODEL_LIMITS',
         'evidence':'MCFTR and RGBITR total returns with the same RUONIA shift1 convention in training and future legs. Corporate spread is RC-RB and is not charged RF again. No discontinued-index splice.',
         'limit':'Two factors do not span credit, duration, currency, style or illiquidity risks. TR-only corporate-spread sensitivity is not estimable in these runs and cannot be advertised as passed robustness.'},
        {'item':'Fund-period regression and inference','status':'PASS_REVIEWED',
         'evidence':'One fund-origin row; fund FE plus current NTS, lagnews NTS and past_alpha; current-only comparison has identical sample hash in all 12 group-horizon pairs. Fund-only CR1 uses full rank including FE; main t/CI reference is G-1. Native residual-df output is separate. Sum/joint tests use full covariance; all planned cells retained.',
         'limit':'Fund clustering covers within-fund serial dependence, including overlapping horizons, but not contemporaneous dependence across funds. Main uncertainty conditions on estimated exposures, news classification and constructed scores. P-values are unadjusted, not family-wide evidence.'},
        {'item':'External numerical QA','status':'NOT_DUPLICATED_PENDING_SEPARATE_REVIEW',
         'evidence':'Executor second-path QA reports 43/0 for v1 and 63/0 for v3. These are not relabelled as independent external replication. The worker owns empirical external numerical QA.'}
      ],
      'reviewed_code':{
        'NTS_builder':{'path':'scripts/build_h3_nts_q_20261104.py','sha256':'01c457e167ba6b8186d2890d88c73c67352de5b7f0f1f56b85ca143a30360d50'},
        'core':{'path':'scripts/h3_alpha_core_20261104.py','sha256':'42f60c5bcdfa6e29fd501034d7dc087dbcc90193cdbd93ed3c45aac96665dcf0'},
        'FE':{'path':'scripts/fit_h3_alpha_fe_20261104.py','sha256':'c599593358a125c718a872f483da18dfed0fabf9c93521521456a8c78d4c6ee4'},
        'runner_v1_sha256':'f65e20013d3eb34cb53be31f9707f66d0ce6a6e52fce2141d09b1e7f57c4931c',
        'runner_v3_sha256':'602b1a47f9bb667d8101e129a21ce3f213a91ba50204e6db04c85c9a81ad4a59'
      },
      'runs':runs,
      'interpretation':{
        'primary_H3':'NOT_YET_TESTED_WITH_ESTIMABLE_VERIFIED_TR_SAMPLE, not rejected and not confirmed by these diagnostic runs.',
        'v1':'No positive current-score coefficient is significant at5% in the 12 distributed cells. Pooled h1 lagnews is nominally positive (p=.01305), while current coefficients are nominally negative for bond h2 and mixed h1. Equity sum tests at h2/h3 are nominally positive, but these are joint score shifts, not a long-run multiplier or primary current-score confirmation.',
        'v3':'No current, lagnews, sum or joint test in the 12 distributed cells reaches5%; current-only same-sample coefficients also do not reach5%. The pooled h1 current estimate is about -5.11 basis points per +1 score with95%CI [-21.10,+10.88] basis points per month. Both positive and negative economic magnitudes remain compatible with this diagnostic interval.',
        'version_comparison':'The v1 pooled lagnews association does not persist in expanded v3 (p=.78880). This is sensitivity to a revised input/sample release, not a formal test that two coefficients differ. Pooled h1 rises from4653/260 funds to6045/351 funds; 4603 origins overlap,1442 are added and50 removed. Never select the older release because it has smaller p.',
        'coverage':'v1 fitted future outcomes end2026-05. v3 fitted origins extend2013-12-31..2026-05-29; future windows reach2026-06/07/08 for h1/2/3. These are observed fit limits, not a claim that every fund covers the full2012–2026 archive.',
        'hypothesis_scope':'NTS_Q measures news alignment in the covered measurable subportfolio. It does not establish that managers read each news item, that the news preceded each trade within an interval, or that funds reallocated discretionary weights net of investor flows.',
        'not_permitted_claims':['There is no effect of news trading.','Fund managers are unskilled or irrational.','H3 is confirmed by the v1 lag coefficient.','The diagnostic branch estimates verified total-return alpha.','All robustness checks passed.','The sum of current and lagnews coefficients is a causal long-run effect.']
      },
      'publication_ready_wording_en':'In the provisional unit-value-return branch, we find no consistent positive association between contemporaneous news alignment and subsequent factor-adjusted unit-value growth. In the expanded data release, estimates for current and previous-news alignment are statistically inconclusive across the four fund groups and three horizons. These results do not establish the absence of an economically meaningful relationship. They are diagnostic: incomplete verification of distributions and unit adjustments prevents interpreting this branch as verified total-return fund alpha.',
      'documentary_updates':{
        'arsagera_review':'reports/h3_arsagera_tr_review_20261104.json',
        'arsagera_review_sha256':sha(ROOT/'reports/h3_arsagera_tr_review_20261104.json'),
        'arsagera_status':'55 official monthly provider-convention adjusted returns for fund403,2022-01..2026-07, source-approved but not consumed by either reviewed run. Keep raw NAV history UNKNOWN and other three funds unapproved until evidence arrives.',
        'fund214_status':'The additional published-return review is a separate DB task. No TR status is inferred from zero accrued-income rows alone, a price-only formula, or aggregate fund-asset growth including flows.',
        'next_provider_run':'A separate provider-source run is permitted under unchanged statistical criteria. It must receive its own input/source-basis and output hashes. One newly certified fund alone does not satisfy the >=2-fund inference gate.'
      },
      'next_actions_without_spec_search':[
        'Complete external numerical QA and attach its exact run/hash before treating either diagnostic release as fully checked.',
        'Have the sole code executor consume the sealed Arsagera source through a basis-aware published-return adapter; do not weaken the raw-NAV equality check globally or double-add income/splits.',
        'Retain the missing verified-TR cells and all planned sensitivity labels; source availability does not authorize a diagnostic Cartesian robustness search.',
        'In the paper, distinguish Q-proxy alignment from original AT, observed unit-value diagnostics from total-return alpha, and inconclusive confidence intervals from evidence of zero effect.'
      ],
      'preservation':'No changes to H1/H2, frozen H3 spec, original data, regression outputs, LLM labels or RESEARCH_NOTES. The reviewer did not rerun worker numerical regressions.'
    }
    out=ROOT/'reports/h3_alpha_interpretation_20261104.json'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'path':str(out),'sha256':sha(out),'runs':RUNS,'cells_per_run':[len(r['diagnostic_4_by_3_distributed']) for r in runs]},ensure_ascii=False))

if __name__ == '__main__':
    main()
