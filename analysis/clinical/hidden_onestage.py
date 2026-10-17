#!/usr/bin/env python3
"""E5 hidden-risk, one-stage supplement S1 (Addendum A): one logistic model across cohorts with cohort-specific
intercept, age spline and sex and ONE set of cell effects common to all cohorts; standardised to the pooled
age-sex-cohort distribution. Respondent-cluster bootstrap. Death only, cessation classification. Aggregates only."""
import hashlib, json, time
import numpy as np, pandas as pd
from scipy.special import expit
from e5_common import *  # noqa
from hidden_risk import prep, CLS, SRHG

NBOOT, BOOT_SEED = 500, 20261005
outcome, cls = "mortality", "cess"
frames, covs = [], []
for coh in MORT_COHORTS:
    co = load_cohort(coh)
    d, A, _ = build_frame(co, outcome)
    h, Ah = prep(d, A)
    h = h[["person_id", "y", "srh_grp", cls, "loss_n"]].copy()
    h["person_id"] = coh + "_" + h["person_id"]
    h["cohort"] = coh
    frames.append(h)
    covs.append(Ah[["age_linear", "age_nonlinear_1", "age_nonlinear_2", "female_code2"]].astype(float))
    del co, d, A
h = pd.concat(frames, ignore_index=True)
cov = pd.concat(covs, ignore_index=True).to_numpy(float)
n = len(h)
y = h["y"].to_numpy(float)
cohort_ix = pd.Categorical(h["cohort"], categories=MORT_COHORTS).codes
parts = [np.ones((n, 1))]
for c in range(1, len(MORT_COHORTS)):
    parts.append((cohort_ix == c).astype(float)[:, None])
for c in range(len(MORT_COHORTS)):
    parts.append(cov * (cohort_ix == c)[:, None])
Xc = np.hstack(parts)
cells = [(g, c) for g in SRHG for c in CLS[cls]]
lab = (h["srh_grp"] + "|" + h[cls]).to_numpy()
ci = np.array([cells.index(tuple(v.split("|"))) for v in lab])
C = np.column_stack([(ci == i).astype(float) for i in range(1, len(cells))])
X = np.hstack([Xc, C])
pc = pd.factorize(h["person_id"])[0]
P = int(pc.max()) + 1
ncov = Xc.shape[1]


def estimate(w, beta0=None):
    beta = fit_logit_w(X, y, w, beta0=beta0)
    lp0 = Xc @ beta[:ncov]
    delta = np.concatenate([[0.0], beta[ncov:]])
    return beta, np.array([float(np.sum(w * expit(lp0 + dl)) / w.sum()) for dl in delta])


beta_pt, pt = estimate(np.ones(n))
rng = np.random.default_rng(BOOT_SEED)
boots = np.zeros((NBOOT, len(cells)))
t0 = time.time()
for b in range(NBOOT):
    w = np.bincount(rng.integers(0, P, P), minlength=P)[pc].astype(float)
    boots[b] = estimate(w, beta0=beta_pt)[1]
    if b % 50 == 0:
        print(f"boot {b} {time.time()-t0:.0f}s", flush=True)
rows = []
for i, (g, c) in enumerate(cells):
    sel = ci == i
    ev = int(y[sel].sum())
    rows.append({"row_type": "onestage_S1", "cohort": "ALL4_ONE_STAGE", "outcome": outcome, "classification": cls,
                 "srh_group": g, "cell": c, "n": int(sel.sum()), "people": int(h.loc[sel, "person_id"].nunique()),
                 "events": ev, "non_events": int(sel.sum()) - ev, "risk_std_model": float(pt[i]),
                 "risk_std_model_lo": float(np.percentile(boots[:, i], 2.5)), "risk_std_model_hi": float(np.percentile(boots[:, i], 97.5)),
                 "risk_crude_summed_counts": ev / sel.sum()})
hi, lo = cells.index(("good_or_better", "2plus")), cells.index(("fair_poor", "0"))
dd = boots[:, hi] - boots[:, lo]
rr = boots[:, hi] / boots[:, lo]
rows.append({"row_type": "onestage_S1_contrast", "cohort": "ALL4_ONE_STAGE", "outcome": outcome, "classification": cls,
             "srh_group": "good_or_better|2plus minus fair_poor|0", "cell": "contrast",
             "risk_diff": float(pt[hi] - pt[lo]), "risk_diff_lo": float(np.percentile(dd, 2.5)), "risk_diff_hi": float(np.percentile(dd, 97.5)),
             "risk_ratio": float(pt[hi] / pt[lo]), "risk_ratio_lo": float(np.percentile(rr, 2.5)), "risk_ratio_hi": float(np.percentile(rr, 97.5)),
             "n_boot": NBOOT, "model_columns": int(X.shape[1])})
write_csv(OUT / "e5_hidden_onestage.csv", pd.DataFrame(rows))


print(json.dumps({"status": "PASS", "rows": len(rows)}))
