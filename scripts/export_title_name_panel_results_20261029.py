"""Format already estimated PC statistics; no data loading or model estimation."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
sens = json.loads((OUT / "title_name_cluster_sensitivity_20261029_summary.json").read_text())
panels = json.loads((OUT / "title_name_hypothesis_research_20261029_summary.json").read_text())
lines = [
    "TITLE-NAME / NO-GAZ: PANEL REGRESSIONS AND COVARIANCE SENSITIVITY",
    "2026-10-29. All panel fits/covariance estimates executed on research-host via SSH.",
    "Post-hoc exploratory research; original results retained; no new research data.",
    "Rows: fund x security x observed disclosure interval. Rows are not independent.",
    "DW = closing weight - opening weight; not price-adjusted active trading or fund alpha.",
    "Coefficients per one unit S; S_negative is the magnitude of the negative signal.",
    "One weight bp =0.01 percentage point. Within R2 refers to the whole model.",
    "", "A. FINITE PANEL EXPERIMENTS: FUND+ISSUER / FULL FE RANK",
    "Raw coefficient p-values are two-sided. Full controls included below.",
    "M3 outcome: log(V1/V0)-log(P1/P0); coefficients in log units, NOT weight bp.",
]
for model in panels["models"]:
    lines += ["", model["model"], "Status: " + model["status"]]
    if "diagnostics" not in model:
        lines += ["Reason: " + str(model.get("reason", "NA"))]
        continue
    dg = model["diagnostics"]
    lines += [
        "N={nobs:,}; funds={funds}; issuers={companies}; securities={securities}; "
        "within_R2={within_r2:.8f}; inference_df={inference_df}".format(**dg),
        "term                                  coefficient        SE           t       p             [95% CI]",
    ]
    for c in model["coefficients"]:
        if c.get("se") is None:
            lines += [f"{c['term']:<36} {c['beta']: .9f} NA"]
        else:
            lines += [
                f"{c['term']:<36} {c['beta']: .9f} {c['se']: .9f} "
                f"{c['t']: 9.4f} {c['p']: .6f} [{c['ci95_low']: .9f}, {c['ci95_high']: .9f}]"
            ]
    if "H1_directional" in model:
        lines += [
            "H1 directional IUT p=" + str(model["H1_directional"].get("intersection_union_p")),
            "H2 absolute-asymmetry p=" + str(model["H2_absolute_asymmetry"].get("p_abs")),
        ]
    if "joint_asset_interaction" in model:
        lines += ["Joint equity/bond interaction: " + json.dumps(model["joint_asset_interaction"])]

lines += [
    "", "B. FIXED COVARIANCE GRID: IDENTICAL SAMPLE/FE PROJECTION/RESIDUALS/BETAS",
    "20 cells: two panels x five cluster structures x two FE-DoF conventions.",
    "Substantive: fund+issuer; fund+issuer+month; fund+issuer+quarter.",
    "Fund-only/issuer-only: incomplete-dependence diagnostics, not preferred estimates.",
    "Sample multiplier (N-1)/(N-k-A): A=full FE rank or verified nonnested FE rank, including constant.",
    "CGM uses component-specific G_J/(G_J-1) before inclusion-exclusion; no PSD clipping.",
    "Nested FE correction is reghdfe-motivated; not exact reghdfe VCE, which uses common G_min correction.",
    "Reference t/F df=min marginal cluster count-1.",
    "H1 p=max(one-sided positive slope p, one-sided negative slope p).",
    "H2 p=max[p(beta_plus+beta_minus=0),p(beta_plus-beta_minus=0)].",
    "Holm4: two coefficient tests + H1 + H2 per cell. Holm80: conservative whole-grid search sensitivity.",
]
for record in sens["records"]:
    tests = {t["test"]: t for t in record["tests"]}
    cluster_text = ", ".join(f"{key}={record['cluster_counts'][key]}" for key in record["clusters"])
    lines += [
        "", record["cell"],
        f"Scope={record['dependence_scope']}; N={record['N']:,}; funds={record['funds']}; "
        f"issuers={record['companies']}; months={record['months']}; "
        f"period={record['min_month']}..{record['max_month']}",
        f"Clusters: {cluster_text}; df={record['inference_df']}; A={record['correction_fe_rank']}; "
        f"q_sample={record['small_sample_q']:.8f}; within_R2={record['within_R2']:.8f}",
        "term                    beta(bp)     SE(bp)       t       p           95% CI(bp)                Holm4    Holm80",
    ]
    for c in record["coefficients"]:
        t = tests[c["term"]]
        lines += [
            f"{c['term']:<22} {c['effect_bps']: 10.5f} {c['se_bps']: 10.5f} {c['t']: 8.4f} "
            f"{c['p']: .6f} [{c['ci95_low_bps']: 10.5f},{c['ci95_high_bps']: 10.5f}] "
            f"{t['p_holm_four_tests_in_cell']: .6f} {t['p_holm_all_locked_covariance_tests_sensitivity']: .6f}"
        ]
    for key in ["H1_directional_IUT", "H2_absolute_union_null"]:
        t = tests[key]
        lines += [
            f"{key:<27} p={t['p']:.9f}; Holm4={t['p_holm_four_tests_in_cell']:.9f}; "
            f"Holm80={t['p_holm_all_locked_covariance_tests_sensitivity']:.9f}"
        ]
    lines += [
        f"Min covariance eigenvalue={record['covariance_min_eigenvalue']:.9g}; "
        f"negative coefficient variances={record['negative_coefficient_variances']}"
    ]
lines += [
    "", "C. VALIDATION AND INTERPRETATION",
    "Both panels: maximum beta difference=0.0; original covariance reproduced within1e-18.",
    "All20 covariance matrices positive definite; no negative coefficient variances.",
    "M0 FE: fund x security + month. M4 FE: fund x security + closing month.",
    "M4 additionally controls log(actual interval days); intervals28..94days, median31days.",
    "M4 is not a pure increase in N: samples, actual news windows and generated S differ from M0.",
    "M4 positive slope nominally significant in all6 substantive cells; not after the fixed Holm4 family.",
    "M4 H1 reaches nominal5% only in fund+issuer/nested-FE among6 substantive cells (p=.045999846).",
    "Nested-FE +time clusters: H1 p=.058836876(month), .067534807(quarter).",
    "H1 nested two-way Holm4=.137999539; Holm80=1.0. Robust confirmatory H1 not established.",
    "H2 unsupported; a small p for beta_plus-beta_minus alone is NOT absolute-asymmetry evidence.",
    "Full price-adjusted AT:71 fully covered intervals,26 target rows before later gates; none final.",
    "Missing security returns/unidentified noncash were not replaced by zero.",
    "Latest bounded fund-history audit:1,601 positive unit prices Jan-Apr2025;1,173 strict adjacent monthly returns.",
    "Zero validated series with36 past returns. H3 fund alpha unavailable; NAV growth not substituted.",
    "Static security links/mandates provisional; SE conditional on LLM labels and generated S.",
    "Official methodology: https://scorreia.com/software/reghdfe/faq.html",
    "Method/numerical/interpretation reviews: reports/title_name_cluster_*_20261029.json.",
    "Authoritative PC output: " + sens["output_root"] + "/summary.json",
]
path = OUT / "title_name_cluster_sensitivity_20261029_tables.txt"
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(str(path))
print("Bytes:", path.stat().st_size, "SHA256:", hashlib.sha256(path.read_bytes()).hexdigest())
