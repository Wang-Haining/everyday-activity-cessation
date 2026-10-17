#!/usr/bin/env python3
"""Pool aggregate cohort outputs (REML + safeguarded Hartung-Knapp) and apply the E1 decision rule."""
import glob, json
import numpy as np, pandas as pd
from pathlib import Path
from exp_stats import reml_hk

OUT = Path(__file__).resolve().parents[2] / "results" / "prediction" / "cohort"
FINAL = OUT.parent; FINAL.mkdir(parents=True, exist_ok=True)
PUBLISHED_POOLED_2PLUS = (3.41, 2.92, 3.98)


def pool_cols(fit, transform=lambda z: z):
    keys = ["pooled", "ci_low_safeguarded", "ci_high_safeguarded", "ci_low", "ci_high",
            "prediction_low_safeguarded", "prediction_high_safeguarded", "prediction_low", "prediction_high"]
    out = {k: float(transform(fit[k])) for k in keys}
    out.update({"i2": fit["i2"], "tau2": fit["tau2"], "hk_scale": fit["hk_scale"], "heterogeneity_p": fit["heterogeneity_p"]})
    return out


# ---------------------------------------------------------------- E1
e1 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(OUT / "e1_cohort_*.csv")))], ignore_index=True)
e1.to_csv(FINAL / "e1_cohort_metrics.csv", index=False)
req = {"mortality": 4, "incident_any_adl": 3}
rows = []
for (outcome, sample, comp), g in e1.groupby(["outcome", "sample", "comparison"], sort=False):
    g = g.sort_values("cohort")
    k = len(g)
    row = dict(outcome=outcome, sample=sample, comparison=comp, k=k, cohorts=";".join(g.cohort),
               n_total=int(g.n.sum()), events_total=int(g.events.sum()),
               per_cohort_dC=";".join(f"{c}:{d:.4f}" for c, d in zip(g.cohort, g.dC)),
               per_cohort_C_base=";".join(f"{c}:{d:.4f}" for c, d in zip(g.cohort, g.C_base)),
               per_cohort_C_aug=";".join(f"{c}:{d:.4f}" for c, d in zip(g.cohort, g.C_aug)))
    thr_cols = [i for i in (1, 2, 3)]
    pos = [(g[[f"dnb_{i}" for i in thr_cols]] > 0).any(axis=1).iloc[j] for j in range(k)]
    row["cohorts_with_positive_dNB_at_any_threshold"] = int(sum(pos))
    for i in thr_cols:
        row[f"cohorts_dNB_positive_thr{i}"] = int((g[f"dnb_{i}"] > 0).sum())
        row[f"pooled_simple_mean_dNB_thr{i}"] = float(g[f"dnb_{i}"].mean())
        row[f"thr{i}"] = float(g[f"thr{i}"].iloc[0])
    if k >= 3:
        fit = reml_hk(g.dC.to_numpy(float), (g.dC_se.to_numpy(float)) ** 2)
        row.update(pool_cols(fit))
    row["required_cohorts_with_positive_dNB"] = req[outcome]
    if comp == "SRH_vs_M1":
        row["verdict"] = "context only (not scored)"
    elif k >= 3:
        crit_c = row["pooled"] >= 0.010 and row["ci_low_safeguarded"] > 0
        crit_nb = row["cohorts_with_positive_dNB_at_any_threshold"] >= req[outcome]
        row["criterion_dC_ge_0.010_and_ci_excludes_0"] = bool(crit_c)
        row["criterion_dNB_positive_in_required_cohorts"] = bool(crit_nb)
        if crit_c and crit_nb:
            row["verdict"] = "meaningful"
        elif row["pooled"] > 0 and row["ci_low_safeguarded"] > 0:
            row["verdict"] = "small"
        else:
            row["verdict"] = "none"
    rows.append(row)
pd.DataFrame(rows).to_csv(FINAL / "e1_pooled.csv", index=False)

# ---------------------------------------------------------------- E2 pre-window
e2 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(OUT / "e2_cohort_*.csv")))], ignore_index=True)
e2["row_type"] = "cohort"
keys = ["measure", "form", "adjustment", "gap_rule", "term"]
pooled = []
est = e2[e2.status.eq("ESTIMABLE") & e2.estimate.notna()]
for key, g in est.groupby(keys, sort=False):
    meta = dict(zip(keys, key))
    k = len(g)
    base = {**meta, "row_type": "pooled", "cohort": "POOLED", "cohorts": ";".join(sorted(g.cohort)), "k": k,
            "n": int(g.n.sum()), "poolable": bool(g.poolable.iloc[0]), "primary_measure": bool(g.primary_measure.iloc[0]), "note": g.note.iloc[0]}
    if not g.poolable.iloc[0]:
        pooled.append({**base, "status": "NOT_POOLED_recall_windows_differ"}); continue
    if k < 3:
        pooled.append({**base, "status": "NOT_POOLED_fewer_than_3_cohorts"}); continue
    if meta["form"] == "binary":
        fit = reml_hk(np.log(g.estimate.to_numpy(float)), g.se.to_numpy(float) ** 2); tr = np.exp
    else:
        fit = reml_hk(g.estimate.to_numpy(float), g.se.to_numpy(float) ** 2); tr = lambda z: z
    p = pool_cols(fit, tr)
    pooled.append({**base, "status": "POOLED", "estimate": p["pooled"], "ci_low": p["ci_low_safeguarded"], "ci_high": p["ci_high_safeguarded"],
                   "ci_low_unmodified_hk": p["ci_low"], "ci_high_unmodified_hk": p["ci_high"],
                   "pi_low": p["prediction_low_safeguarded"], "pi_high": p["prediction_high_safeguarded"], "i2": p["i2"], "tau2": p["tau2"],
                   "scale": "risk_ratio" if meta["form"] == "binary" else "SD_of_change"})
e2_out = pd.concat([e2, pd.DataFrame(pooled)], ignore_index=True)
e2_out.to_csv(FINAL / "e2_pre_window.csv", index=False)

# ---------------------------------------------------------------- E2 stable subset
st = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(OUT / "e2_stable_*.csv")))], ignore_index=True)
st["row_type"] = "cohort"
sp = []
est = st[st.status.eq("ESTIMABLE") & st.rr.notna()]
for (pop, term), g in est.groupby(["population", "term"], sort=False):
    k = len(g)
    base = {"row_type": "pooled", "cohort": "POOLED", "population": pop, "term": term, "cohorts": ";".join(sorted(g.cohort)), "k": k,
            "n": int(g.n.sum()), "deaths": int(g.deaths.sum()), "people": int(g.people.sum()),
            "n_group_0": int(g.n_group_0.sum()), "n_group_1": int(g.n_group_1.sum()), "n_group_2plus": int(g.n_group_2plus.sum()),
            "deaths_group_0": int(g.deaths_group_0.sum()), "deaths_group_1": int(g.deaths_group_1.sum()), "deaths_group_2plus": int(g.deaths_group_2plus.sum())}
    if k < 3:
        sp.append({**base, "status": "NOT_POOLED_fewer_than_3_cohorts"}); continue
    fit = reml_hk(np.log(g.rr.to_numpy(float)), g.se_log_rr.to_numpy(float) ** 2)
    p = pool_cols(fit, np.exp)
    sp.append({**base, "status": "POOLED", "rr": p["pooled"], "ci_low": p["ci_low_safeguarded"], "ci_high": p["ci_high_safeguarded"],
               "ci_low_unmodified_hk": p["ci_low"], "ci_high_unmodified_hk": p["ci_high"],
               "pi_low": p["prediction_low_safeguarded"], "pi_high": p["prediction_high_safeguarded"], "i2": p["i2"], "tau2": p["tau2"],
               "all_cohort_point_rr_above_1": bool((g.rr > 1).all())})
# Post hoc (not in the design; see Deviations): the stable subset clears the support gates only in HRS
# and SHARE, so no random-effects pooling is possible (k < 3). For a like-for-like comparison, an
# inverse-variance fixed-effect combination of exactly those two cohorts is given for every population.
for pop in ["all_primary", "T_all", "T_stable", "T_not_stable", "T_all_gap22_30", "T_stable_gap22_30"]:
    for term in ["loss_1", "loss_2plus"]:
        g = est[(est.population == pop) & (est.term == term) & est.cohort.isin(["hrs", "share"])]
        if g.cohort.nunique() != 2:
            continue
        w = 1 / g.se_log_rr.to_numpy(float) ** 2
        mu = float((w * np.log(g.rr.to_numpy(float))).sum() / w.sum()); se = float(w.sum() ** -0.5)
        sp.append({"row_type": "pooled_post_hoc", "cohort": "HRS+SHARE fixed effect", "population": pop, "term": term,
                   "cohorts": "hrs;share", "k": 2, "n": int(g.n.sum()), "deaths": int(g.deaths.sum()), "people": int(g.people.sum()),
                   "n_group_0": int(g.n_group_0.sum()), "n_group_1": int(g.n_group_1.sum()), "n_group_2plus": int(g.n_group_2plus.sum()),
                   "deaths_group_0": int(g.deaths_group_0.sum()), "deaths_group_1": int(g.deaths_group_1.sum()),
                   "deaths_group_2plus": int(g.deaths_group_2plus.sum()), "status": "FIXED_EFFECT_2_COHORTS_post_hoc",
                   "rr": float(np.exp(mu)), "ci_low": float(np.exp(mu - 1.959964 * se)), "ci_high": float(np.exp(mu + 1.959964 * se))})
sp = pd.DataFrame(sp)
# attenuation of log RR relative to comparator (ii) T_all
for term in ["loss_1", "loss_2plus"]:
    ref = sp[(sp.population == "T_all") & (sp.term == term) & (sp.status == "POOLED")]
    for pop in ["T_stable", "T_not_stable"]:
        cur = sp[(sp.population == pop) & (sp.term == term) & (sp.status == "POOLED")]
        if len(ref) and len(cur):
            sp.loc[cur.index, "log_rr_change_vs_T_all_percent"] = (np.log(cur.rr.to_numpy()) - np.log(ref.rr.iloc[0])) / np.log(ref.rr.iloc[0]) * 100
pd.concat([st, sp], ignore_index=True).to_csv(FINAL / "e2_stable_subset.csv", index=False)
ap = sp[(sp.population == "all_primary") & (sp.term == "loss_2plus") & (sp.status == "POOLED")]
chk = {"pooled_2plus_all_primary": float(ap.rr.iloc[0]), "ci": [float(ap.ci_low.iloc[0]), float(ap.ci_high.iloc[0])],
       "published": PUBLISHED_POOLED_2PLUS}
assert abs(chk["pooled_2plus_all_primary"] - 3.41) < 0.006 and abs(chk["ci"][0] - 2.92) < 0.006 and abs(chk["ci"][1] - 3.98) < 0.006, chk
(FINAL / "validation.json").write_text(json.dumps(chk, indent=1))
print(json.dumps({"status": "PASS", **chk}))
