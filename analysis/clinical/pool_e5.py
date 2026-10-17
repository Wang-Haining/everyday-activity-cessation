#!/usr/bin/env python3
"""Pool E5 aggregates (REML + safeguarded Hartung-Knapp) and apply the pre-set decision rules."""
import glob, json
import numpy as np, pandas as pd
from scipy.special import expit, logit
from e5_common import OUT, FINAL, reml_hk, OUTCOMES

FINAL.mkdir(parents=True, exist_ok=True)
REQ = {"mortality": 4, "incident_any_adl": 3}
KEYS = ["pooled", "ci_low_safeguarded", "ci_high_safeguarded", "ci_low", "ci_high",
        "prediction_low_safeguarded", "prediction_high_safeguarded", "i2", "tau2", "hk_scale", "heterogeneity_p"]


def read_all(pattern):
    files = [f for f in sorted(glob.glob(str(OUT / pattern))) if "onestage" not in f]
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


# ------------------------------------------------------------------ discrimination / calibration / net benefit
coh = read_all("e5_cohort_[a-z]*.csv")
mod = read_all("e5_models_*.csv")
coh.to_csv(FINAL / "e5_cohort_metrics.csv", index=False)
mod.to_csv(FINAL / "e5_model_metrics.csv", index=False)

rows = []
for (outcome, comp), g in coh.groupby(["outcome", "comparison"], sort=False):
    g = g.sort_values("cohort")
    k = len(g)
    base = g.iloc[0]
    row = dict(row_type="comparison", outcome=outcome, comparison=comp, baseline=base.baseline, addition=base.addition,
               base_model=base.base_model, aug_model=base.aug_model, k=k, cohorts=";".join(g.cohort),
               n_total=int(g.n.sum()), events_total=int(g.events.sum()),
               per_cohort_dC=";".join(f"{c}:{d:.4f}" for c, d in zip(g.cohort, g.dC)),
               per_cohort_C_base=";".join(f"{c}:{d:.4f}" for c, d in zip(g.cohort, g.C_base)),
               per_cohort_C_aug=";".join(f"{c}:{d:.4f}" for c, d in zip(g.cohort, g.C_aug)))
    dnb = g[[f"dnb_{i}" for i in (1, 2, 3)]]
    row["cohorts_with_positive_dNB_at_any_threshold"] = int((dnb > 0).any(axis=1).sum())
    for i in (1, 2, 3):
        row[f"thr{i}"] = float(g[f"thr{i}"].iloc[0])
        row[f"cohorts_dNB_positive_thr{i}"] = int((g[f"dnb_{i}"] > 0).sum())
        row[f"mean_dNB_thr{i}"] = float(g[f"dnb_{i}"].mean())
        row[f"cohorts_aug_NB_above_treat_all_and_zero_thr{i}"] = int(((g[f"nb_aug_{i}"] > g[f"nb_all_{i}"]) & (g[f"nb_aug_{i}"] > 0)).sum())
    row["cohorts_aug_NB_above_treat_all_and_zero_any_thr"] = int(
        ((g[[f"nb_aug_{i}" for i in (1, 2, 3)]].to_numpy() > g[[f"nb_all_{i}" for i in (1, 2, 3)]].to_numpy())
         & (g[[f"nb_aug_{i}" for i in (1, 2, 3)]].to_numpy() > 0)).any(axis=1).sum())
    row["mean_dScaledBrier"] = float(g.dScaledBrier.mean())
    row["cal_slope_aug_range"] = f"{g.cal_slope_aug.min():.3f}-{g.cal_slope_aug.max():.3f}"
    row["cal_intercept_aug_range"] = f"{g.cal_intercept_aug.min():.4f} to {g.cal_intercept_aug.max():.4f}"
    if k >= 3:
        fit = reml_hk(g.dC.to_numpy(float), g.dC_se.to_numpy(float) ** 2)
        row.update({kk: float(fit[kk]) for kk in KEYS})
        fb = reml_hk(g.dBrier.to_numpy(float), g.dBrier_se.to_numpy(float) ** 2)
        row.update({"dBrier_pooled": fb["pooled"], "dBrier_ci_low_safeguarded": fb["ci_low_safeguarded"],
                    "dBrier_ci_high_safeguarded": fb["ci_high_safeguarded"]})
        pos_ci = row["pooled"] > 0 and row["ci_low_safeguarded"] > 0
        if row["addition"] == "iv_vs_iii":
            row["rule"] = "rule2_history_beyond_current_state"
            row["verdict"] = "history adds beyond current state" if pos_ci else "history does not add beyond current state"
        else:
            row["rule"] = "rule1_clinically_meaningful_increment"
            crit_c = bool(row["pooled"] >= 0.010 and row["ci_low_safeguarded"] > 0)
            crit_nb = bool(row["cohorts_with_positive_dNB_at_any_threshold"] >= REQ[outcome])
            row["criterion_dC_ge_0.010_and_ci_excludes_0"] = crit_c
            row["criterion_dNB_positive_in_required_cohorts"] = crit_nb
            row["required_cohorts_with_positive_dNB"] = REQ[outcome]
            row["verdict"] = "meaningful" if (crit_c and crit_nb) else ("small" if pos_ci else "none")
    rows.append(row)

# pooled model-level C
for (outcome, model), g in mod.groupby(["outcome", "model"], sort=False):
    g = g.sort_values("cohort")
    row = dict(row_type="model", outcome=outcome, comparison=model, base_model=model, k=len(g), cohorts=";".join(g.cohort),
               n_total=int(g.n.sum()), events_total=int(g.events.sum()),
               per_cohort_C=";".join(f"{c}:{d:.4f}" for c, d in zip(g.cohort, g.C)))
    if len(g) >= 3:
        fit = reml_hk(g.C.to_numpy(float), g.C_se.to_numpy(float) ** 2)
        row.update({kk: float(fit[kk]) for kk in KEYS})
    row["mean_scaled_brier"] = float(g.scaled_brier.mean())
    row["cal_slope_range"] = f"{g.cal_slope.min():.3f}-{g.cal_slope.max():.3f}"
    rows.append(row)
pooled_disc = pd.DataFrame(rows)

# ------------------------------------------------------------------ hidden risk
hid = read_all("e5_hidden_[a-z]*.csv")
out_h = [hid]
pool_rows = []
for (outcome, cls), g in hid[hid.row_type.eq("cohort")].groupby(["outcome", "classification"], sort=False):
    for (sg, cell), gc in g.groupby(["srh_group", "cell"], sort=False):
        ev = gc[gc.evaluable]
        base = dict(row_type="pooled", cohort="POOLED", outcome=outcome, classification=cls, srh_group=sg, cell=cell,
                    n=int(gc.n.sum()), people=int(gc.people.sum()), events=int(gc.events.sum()), non_events=int(gc.non_events.sum()),
                    cohorts_all=";".join(gc.cohort), cohorts_evaluable=";".join(ev.cohort), k=len(ev),
                    risk_crude_summed_counts=float(gc.events.sum() / gc.n.sum()))
        for nm in ["crude", "std_model", "std_direct"]:
            if len(ev) >= 3:
                fit = reml_hk(ev[f"logit_{nm}"].to_numpy(float), ev[f"logit_{nm}_se"].to_numpy(float) ** 2)
                base.update({f"risk_{nm}": float(expit(fit["pooled"])), f"risk_{nm}_lo": float(expit(fit["ci_low_safeguarded"])),
                             f"risk_{nm}_hi": float(expit(fit["ci_high_safeguarded"])),
                             f"risk_{nm}_pi_lo": float(expit(fit["prediction_low_safeguarded"])),
                             f"risk_{nm}_pi_hi": float(expit(fit["prediction_high_safeguarded"])), f"i2_{nm}": fit["i2"]})
        base["status"] = "POOLED" if len(ev) >= 3 else "NOT_POOLED_fewer_than_3_evaluable_cohorts"
        pool_rows.append(base)
pool_h = pd.DataFrame(pool_rows)

# contrast rows
crow = []
for outcome, g in hid[hid.row_type.eq("cohort_contrast")].groupby("outcome", sort=False):
    ev = g[g.evaluable]
    base = dict(row_type="pooled_contrast", cohort="POOLED", outcome=outcome, classification="cess",
                srh_group="good_or_better|2plus minus fair_poor|0", cell="contrast", cohorts_all=";".join(g.cohort),
                cohorts_evaluable=";".join(ev.cohort), k=len(ev))
    for nm in ["crude", "std_model", "std_direct"]:
        if len(ev) >= 3:
            fit = reml_hk(ev[f"logit_{nm}"].to_numpy(float), ev[f"logit_{nm}_se"].to_numpy(float) ** 2)
            base.update({f"logit_diff_{nm}": fit["pooled"], f"logit_diff_{nm}_lo": fit["ci_low_safeguarded"],
                         f"logit_diff_{nm}_hi": fit["ci_high_safeguarded"], f"or_{nm}": float(np.exp(fit["pooled"])),
                         f"or_{nm}_lo": float(np.exp(fit["ci_low_safeguarded"])), f"or_{nm}_hi": float(np.exp(fit["ci_high_safeguarded"]))})
    base["status"] = "POOLED" if len(ev) >= 3 else "NOT_POOLED_fewer_than_3_evaluable_cohorts"
    crow.append(base)
pool_c = pd.DataFrame(crow)

# share of events
srow = []
sh = hid[hid.row_type.eq("share_of_events")]
for outcome, g in sh.groupby("outcome", sort=False):
    base = dict(row_type="pooled_share_of_events", cohort="POOLED", outcome=outcome, classification="cess",
                srh_group="good_or_better", cell=">=1 cessation", n=int(g.n.sum()), events=int(g.events.sum()),
                total_events=int(g.total_events.sum()), total_n=int(g.total_n.sum()),
                share_of_events=float(g.events.sum() / g.total_events.sum()), share_of_intervals=float(g.n.sum() / g.total_n.sum()),
                cohorts_all=";".join(g.cohort), k=len(g))
    if len(g) >= 3:
        p = (g.events / g.total_events).to_numpy(float)
        v = 1.0 / (g.total_events.to_numpy(float) * p * (1 - p))
        fit = reml_hk(logit(p), v)
        base.update({"share_reml": float(expit(fit["pooled"])), "share_reml_lo": float(expit(fit["ci_low_safeguarded"])),
                     "share_reml_hi": float(expit(fit["ci_high_safeguarded"])), "i2": fit["i2"]})
    srow.append(base)
pool_s = pd.DataFrame(srow)

# Rule 3 verdict (death), read on the model-standardised pooled risks
verd = []
for outcome in OUTCOMES:
    ph = pool_h[(pool_h.outcome == outcome) & (pool_h.classification == "cess")]
    def get(sg, cell, col):
        r = ph[(ph.srh_group == sg) & (ph.cell == cell)]
        return float(r[col].iloc[0]) if len(r) and col in r and pd.notna(r[col].iloc[0]) else np.nan
    for nm in ["std_model", "std_direct", "crude"]:
        hi, lo = get("good_or_better", "2plus", f"risk_{nm}"), get("fair_poor", "0", f"risk_{nm}")
        verd.append(dict(row_type="rule3", cohort="POOLED", outcome=outcome, classification="cess", srh_group="rule3", cell=nm,
                         risk_good_or_better_2plus=hi, risk_fair_poor_0=lo,
                         status=("NOT_EVALUABLE" if np.isnan(hi) or np.isnan(lo) else "EVALUATED"),
                         verdict=("hidden high-risk group" if (not np.isnan(hi) and not np.isnan(lo) and hi >= lo) else
                                  ("not evaluable" if np.isnan(hi) or np.isnan(lo) else "no hidden high-risk group")),
                         primary_for_rule=bool(nm == "std_model" and outcome == "mortality")))
os1 = OUT / "e5_hidden_onestage.csv"
one = pd.read_csv(os1) if os1.exists() else pd.DataFrame()
if len(one):
    c = one[one.row_type.eq("onestage_S1")]
    hi = float(c[(c.srh_group == "good_or_better") & (c.cell == "2plus")].risk_std_model.iloc[0])
    lo = float(c[(c.srh_group == "fair_poor") & (c.cell == "0")].risk_std_model.iloc[0])
    verd.append(dict(row_type="rule3", cohort="ONE_STAGE_S1", outcome="mortality", classification="cess", srh_group="rule3",
                     cell="onestage_std_model", risk_good_or_better_2plus=hi, risk_fair_poor_0=lo, status="EVALUATED",
                     verdict="hidden high-risk group" if hi >= lo else "no hidden high-risk group", primary_for_rule=False))
pool_v = pd.DataFrame(verd)
pd.concat([hid, pool_h, pool_c, pool_s, pool_v, one], ignore_index=True).to_csv(FINAL / "e5_hidden_risk.csv", index=False)
pooled_disc.to_csv(FINAL / "e5_pooled.csv", index=False)
print(json.dumps({"status": "PASS", "pooled_rows": len(pooled_disc), "hidden_rows": len(hid) + len(pool_h)}))
