#!/usr/bin/env python3
"""B: pool cohort rows by REML-HK (safeguarded CI; at least 3 cohorts) or an inverse-variance fixed-effect summary
(2 cohorts, labelled), and write b_delayed_landmark.csv (aggregate only)."""
import glob
import numpy as np, pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from x3_common import pool_log, fixed_effect_log, write_csv, OUT   # noqa

FINAL = OUT.parent; FINAL.mkdir(parents=True, exist_ok=True)
b = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(OUT / "b_fit_*.csv")))], ignore_index=True)
KEYS = ["analysis", "population", "landmark_months", "rule", "term"]
lm = b[b.row_type.eq("cohort") & b.analysis.eq("delayed_landmark")].copy()
lm["rule"] = lm["rule"].fillna("")
pooled = []
for key, g in lm.groupby(KEYS, sort=False, dropna=False):
    meta = dict(zip(KEYS, key))
    est = g[g.status.eq("ESTIMABLE") & g.rr.notna()].sort_values("cohort")
    base = {**meta, "cohort": "POOLED", "k": int(len(est)), "cohorts": ";".join(est.cohort),
            "cohorts_not_evaluable": ";".join(f"{c}:{s}" for c, s in zip(g.cohort, g.status) if s != "ESTIMABLE"),
            "per_cohort_rr": ";".join(f"{c}:{r:.3f}" for c, r in zip(est.cohort, est.rr)),
            "events_group": int(est.events_group.sum()) if len(est) else 0, "n_group": int(est.n_group.sum()) if len(est) else 0}
    if len(est) >= 3:
        pooled.append({**base, "row_type": "pooled", "status": "POOLED", **pool_log(est)})
    elif len(est) == 2:
        pooled.append({**base, "row_type": "fixed_effect_2_cohorts", "status": "FIXED_EFFECT_2_COHORTS", **fixed_effect_log(est)})
    else:
        pooled.append({**base, "row_type": "pooled", "status": "NOT_POOLED_fewer_than_2_evaluable_cohorts"})
# like-for-like comparator: inverse-variance fixed-effect summary of HRS and SHARE (the cohorts that keep support at 12 months)
for key, g in lm.groupby(KEYS, sort=False, dropna=False):
    meta = dict(zip(KEYS, key))
    est = g[g.status.eq("ESTIMABLE") & g.rr.notna() & g.cohort.isin(["hrs", "share"])].sort_values("cohort")
    if len(est) == 2:
        pooled.append({**meta, "cohort": "POOLED", "row_type": "fixed_effect_HRS_SHARE_comparator", "status": "FIXED_EFFECT_HRS_SHARE", "k": 2,
                       "cohorts": ";".join(est.cohort), "per_cohort_rr": ";".join(f"{c}:{r:.3f}" for c, r in zip(est.cohort, est.rr)),
                       "events_group": int(est.events_group.sum()), "n_group": int(est.n_group.sum()), **fixed_effect_log(est)})
allr = pd.concat([b, pd.DataFrame(pooled)], ignore_index=True)
# change in the RR relative to the no-landmark (dated deaths only) reference, same population / cohort / term
ref = allr[allr.analysis.eq("delayed_landmark") & allr.landmark_months.eq(0) & allr.rr.notna()].set_index(["row_type", "cohort", "population", "term"])["rr"]
def ratio(r):
    if r["analysis"] != "delayed_landmark" or pd.isna(r.get("rr")):
        return np.nan
    k = (r["row_type"], r["cohort"], r["population"], r["term"])
    return r["rr"] / ref[k] if k in ref.index else np.nan
allr["rr_ratio_to_L0"] = allr.apply(ratio, axis=1)
cov = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(OUT / "step0_*.csv")))], ignore_index=True)
cov = cov[cov.block.eq("B_dates")].rename(columns={"block": "analysis"})
cov["row_type"] = "dating_coverage"
allr = pd.concat([allr, cov], ignore_index=True)
write_csv(FINAL / "b_delayed_landmark.csv", allr)
print(allr[allr.row_type.isin(["pooled", "fixed_effect_2_cohorts", "fixed_effect_HRS_SHARE_comparator"])][["population", "landmark_months", "rule", "term", "status", "k", "cohorts", "rr", "ci_low", "ci_high", "rr_ratio_to_L0"]].to_string())
