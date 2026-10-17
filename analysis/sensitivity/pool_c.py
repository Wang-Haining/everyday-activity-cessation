#!/usr/bin/env python3
"""C: pool weighted and complete-case cohort RRs / standardised risks (REML-HK, safeguarded CI; at least 3 cohorts)
and write c_selection_weights.csv and c_weighted_models.csv (aggregate only)."""
import glob
import numpy as np, pandas as pd
from scipy.special import expit, logit
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from x3_common import pool_log, reml_hk, write_csv, OUT   # noqa

FINAL = OUT.parent; FINAL.mkdir(parents=True, exist_ok=True)
sel = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(OUT / "c_sel_*.csv")))], ignore_index=True)
mod = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(OUT / "c_mod_*.csv")))], ignore_index=True)
mod["row_type"] = "cohort"
pooled = []
rr = mod[mod.model.eq("rr_count_model") & mod.rr.notna()]
for (outc, est, term), g in rr.groupby(["outcome", "estimator", "term"], sort=False):
    g = g.sort_values("cohort")
    base = {"model": "rr_count_model", "outcome": outc, "estimator": est, "term": term, "cohort": "POOLED", "row_type": "pooled",
            "k": int(len(g)), "cohorts": ";".join(g.cohort), "per_cohort_rr": ";".join(f"{c}:{r:.3f}" for c, r in zip(g.cohort, g.rr))}
    if len(g) < 3:
        pooled.append({**base, "status": "NOT_POOLED_fewer_than_3_cohorts"}); continue
    pooled.append({**base, "status": "POOLED", **pool_log(g)})
pdf = pd.DataFrame(pooled)
# ratio of each pooled / cohort RR to the complete-case RR
cc = pd.concat([mod[mod.model.eq("rr_count_model")], pdf], ignore_index=True)
ref = cc[cc.estimator.eq("complete_case")].set_index(["row_type", "cohort", "outcome", "term"])["rr"]
sr = mod[mod.model.eq("standardised_risk_common_target_k_ge_2")]
for (outc, est, term), g in sr.groupby(["outcome", "estimator", "term"], sort=False):
    g = g.sort_values("cohort")
    p = g.risk.to_numpy(float); se = g.se_risk.to_numpy(float)
    base = {"model": "standardised_risk_common_target_k_ge_2", "outcome": outc, "estimator": est, "term": term, "cohort": "POOLED", "row_type": "pooled",
            "k": int(len(g)), "cohorts": ";".join(g.cohort)}
    if len(g) < 3:
        pooled.append({**base, "status": "NOT_POOLED_fewer_than_3_cohorts"}); continue
    f = reml_hk(logit(p), (se / (p * (1 - p))) ** 2)
    row = {**base, "status": "POOLED", "risk": float(expit(f["pooled"])), "ci_low": float(expit(f["ci_low_safeguarded"])), "ci_high": float(expit(f["ci_high_safeguarded"])),
           "pi_low": float(expit(f["prediction_low_safeguarded"])), "pi_high": float(expit(f["prediction_high_safeguarded"])), "i2": f["i2"]}
    if term != "withdrawal_0":
        fr = reml_hk(g.risk_difference_vs_0.to_numpy(float), g.se_risk_difference.to_numpy(float) ** 2)
        row.update({"risk_difference_vs_0": fr["pooled"], "rd_ci_low": fr["ci_low_safeguarded"], "rd_ci_high": fr["ci_high_safeguarded"]})
    pooled.append(row)
allm = pd.concat([mod, pd.DataFrame(pooled)], ignore_index=True)
allm = allm.drop_duplicates(subset=None)
def rat(r):
    if r["model"] != "rr_count_model" or pd.isna(r.get("rr")):
        return np.nan
    k = (r["row_type"], r["cohort"], r["outcome"], r["term"])
    return r["rr"] / ref[k] if k in ref.index else np.nan
allm["rr_ratio_to_complete_case"] = allm.apply(rat, axis=1)
write_csv(FINAL / "c_weighted_models.csv", allm)
write_csv(FINAL / "c_selection_weights.csv", sel)
pd.set_option("display.width", 250, "display.max_columns", 30)
print(allm[allm.row_type.eq("pooled")][["model", "outcome", "estimator", "term", "status", "k", "rr", "ci_low", "ci_high", "risk", "rr_ratio_to_complete_case"]].to_string())
