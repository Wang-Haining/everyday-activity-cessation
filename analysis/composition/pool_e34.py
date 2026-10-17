#!/usr/bin/env python3
"""Pool cohort rows (REML, safeguarded Hartung-Knapp primary, unmodified HK alongside), write the
final aggregate tables and apply the E3 decision rule (mortality)."""
import glob, json
import numpy as np, pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prediction"))
from exp_stats import reml_hk

OUT = Path(__file__).resolve().parents[2] / "results" / "composition" / "cohort"
FINAL = OUT.parent; FINAL.mkdir(parents=True, exist_ok=True)
KEYS = ["analysis", "population", "pair", "omitted", "stratum", "outcome", "term"]


def pool_frame(fit):
    f = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(OUT / "fit_*.csv")))], ignore_index=True)
    for k in KEYS:
        if k not in f.columns:
            f[k] = ""
    f[KEYS] = f[KEYS].fillna("")
    if "is_contrast" not in f.columns:
        f["is_contrast"] = False
    f["is_contrast"] = f["is_contrast"].fillna(False).astype(bool)
    f["row_type"] = "cohort"
    pooled = []
    for key, g in f.groupby(KEYS, sort=False):
        meta = dict(zip(KEYS, key))
        est = g[g.status.eq("ESTIMABLE") & g.rr.notna()].sort_values("cohort")
        fail = ";".join(f"{c}:{s}" for c, s in zip(g.cohort, g.status) if s != "ESTIMABLE")
        base = {**meta, "row_type": "pooled", "cohort": "POOLED", "is_contrast": bool(g.is_contrast.iloc[0]),
                "k": int(len(est)), "cohorts": ";".join(est.cohort), "cohorts_not_evaluable": fail,
                "per_cohort_rr": ";".join(f"{c}:{r:.3f}" for c, r in zip(est.cohort, est.rr))}
        if "n_group" in est and est.n_group.notna().any():
            base["n_group"] = int(est.n_group.sum()); base["events_group"] = int(est.events_group.sum())
        if len(est) < 3:
            pooled.append({**base, "status": "NOT_POOLED_fewer_than_3_evaluable_cohorts"}); continue
        fitr = reml_hk(np.log(est.rr.to_numpy(float)), est.se_log_rr.to_numpy(float) ** 2)
        pooled.append({**base, "status": "POOLED", "rr": float(np.exp(fitr["pooled"])),
                       "ci_low": float(np.exp(fitr["ci_low_safeguarded"])), "ci_high": float(np.exp(fitr["ci_high_safeguarded"])),
                       "ci_low_unmodified_hk": float(np.exp(fitr["ci_low"])), "ci_high_unmodified_hk": float(np.exp(fitr["ci_high"])),
                       "pi_low": float(np.exp(fitr["prediction_low_safeguarded"])), "pi_high": float(np.exp(fitr["prediction_high_safeguarded"])),
                       "pi_low_unmodified_hk": float(np.exp(fitr["prediction_low"])), "pi_high_unmodified_hk": float(np.exp(fitr["prediction_high"])),
                       "i2": fitr["i2"], "tau2": fitr["tau2"], "hk_scale": fitr["hk_scale"]})
    return pd.concat([f, pd.DataFrame(pooled)], ignore_index=True)


allrows = pool_frame(None)
# ---- S1 one-stage rows (Addendum A), labelled row_type = one_stage_S1
for s1p in [OUT / "onestage_S1.csv", OUT / "onestage_S1b_posthoc.csv"]:
    if not s1p.exists():
        continue
    s1 = pd.read_csv(s1p)
    for k in KEYS:
        if k not in s1.columns:
            s1[k] = ""
    s1[KEYS] = s1[KEYS].fillna("")
    s1["is_contrast"] = s1["is_contrast"].fillna(False).astype(bool) if "is_contrast" in s1 else False
    s1["k"] = s1["cohorts_included"].str.count(";") + 1
    s1.loc[s1.status.eq("ESTIMABLE"), "status"] = "ONE_STAGE_ESTIMABLE"
    allrows = pd.concat([allrows, s1], ignore_index=True)
order = ["analysis", "population", "pair", "omitted", "stratum", "outcome", "term", "row_type", "cohort"]
allrows = allrows.sort_values(order, kind="stable")

# ---- validation (refit of the published count model, pooled)
val = allrows[allrows.analysis.eq("validation_count_model") & allrows.row_type.eq("pooled")]
val.to_csv(FINAL / "validation_count_model.csv", index=False)
chk = {}
for outcome, tgt in [("mortality", (3.41, 2.92, 3.98)), ("incident_any_adl", (1.99, 1.57, 2.53))]:
    r = val[val.outcome.eq(outcome) & val.term.eq("loss_2plus")].iloc[0]
    chk[outcome] = {"pooled": [float(r.rr), float(r.ci_low), float(r.ci_high)], "published": list(tgt)}
    assert abs(r.rr - tgt[0]) < 0.006 and abs(r.ci_low - tgt[1]) < 0.006 and abs(r.ci_high - tgt[2]) < 0.006, chk[outcome]

def sel(name):
    return allrows[allrows.analysis.isin(name if isinstance(name, list) else [name])]
s1_only = allrows[allrows.row_type.eq("one_stage_S1")]
allrows_primary = allrows[~allrows.row_type.eq("one_stage_S1")]

sel("e3_pattern_model").to_csv(FINAL / "e3_pattern_models.csv", index=False)
sel("e3_restricted_pair").to_csv(FINAL / "e3_restricted_pairs.csv", index=False)
sel("e3_leave_one_out").to_csv(FINAL / "e3_leave_one_out.csv", index=False)
sel(["e4_four_group", "e4_gain_within_loss_stratum"]).to_csv(FINAL / "e4_models.csv", index=False)

# ---- count tables with pooled (summed) rows
pat = pd.concat([pd.read_csv(p, dtype={"pattern": str, "label_DAW": str}) for p in sorted(glob.glob(str(OUT / "tables_patterns_*.csv")))], ignore_index=True)
tot = []
for (o, p), g in pat.groupby(["outcome", "pattern"], sort=False):
    n, ev = int(g.n.sum()), int(g.events.sum())
    r = {"cohort": "ALL_COHORTS_SUMMED", "outcome": o, "pattern": p, "label_DAW": p, "n_losses": int(g.n_losses.iloc[0]), "n": n,
         "people": int(g.people.sum()), "events": ev, "crude_risk": ev / n if n else np.nan}
    for k in (1, 2, 3):
        r[f"initial_opps_{k}_n"] = int(g[f"initial_opps_{k}_n"].sum()); r[f"initial_opps_{k}_share"] = r[f"initial_opps_{k}_n"] / n if n else np.nan
    tot.append(r)
pd.concat([pat, pd.DataFrame(tot)], ignore_index=True).to_csv(FINAL / "e3_patterns.csv", index=False)

x = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(OUT / "tables_e4_crosstab_*.csv")))], ignore_index=True)
tot = []
for (rt, o, lc, gc, sc, io), g in x.fillna({"gain_cat": "", "scope": "", "initial_opps": -1}).groupby(["row_type", "outcome", "loss_cat", "gain_cat", "scope", "initial_opps"], sort=False):
    r = {"row_type": rt, "cohort": "ALL_COHORTS_SUMMED", "outcome": o, "loss_cat": lc, "gain_cat": gc or np.nan, "scope": sc or np.nan,
         "initial_opps": io if io != -1 else np.nan, "n": int(g.n.sum())}
    if rt == "crosstab":
        r["people"] = int(g.people.sum()); r["events"] = int(g.events.sum()); r["crude_risk"] = r["events"] / r["n"] if r["n"] else np.nan
    else:
        r["n_with_gain"] = int(g.n_with_gain.sum()); r["share_with_gain"] = r["n_with_gain"] / r["n"] if r["n"] else np.nan
    tot.append(r)
pd.concat([x, pd.DataFrame(tot)], ignore_index=True).to_csv(FINAL / "e4_crosstab.csv", index=False)
ver = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(OUT / "tables_verify_*.csv")))], ignore_index=True)
ver.to_csv(FINAL / "verify_risk_sets.csv", index=False)

# ---- E3 decision rule (mortality scored), based on pairwise risk estimates
def decide(row_type, ok_status):
    dec_rows = []
    for outcome in ["mortality", "incident_any_adl"]:
        rpo = allrows[allrows.analysis.eq("e3_restricted_pair") & allrows.row_type.eq(row_type) & allrows.outcome.eq(outcome) & ~allrows.is_contrast]
        for pair, (s1, s2) in {"DA": ("single_D", "single_A"), "DW": ("single_D", "single_W"), "AW": ("single_A", "single_W")}.items():
            def g_(term):
                r = rpo[(rpo.pair == pair) & (rpo.term == term)]
                return r.iloc[0] if len(r) and r.iloc[0].status == ok_status else None
            P, A, B = g_(f"pair_{pair}"), g_(s1), g_(s2)
            row = {"estimator": row_type, "outcome": outcome, "pair": pair, "pair_pooled": P is not None, "single_1_pooled": A is not None, "single_2_pooled": B is not None}
            if P is not None:
                row.update(pair_rr=P.rr, pair_ci_low=P.ci_low, pair_ci_high=P.ci_high)
            if A is not None:
                row.update(single1_rr=A.rr)
            if B is not None:
                row.update(single2_rr=B.rr)
            if P is not None and A is not None and B is not None:
                mx = max(A.rr, B.rr)
                row.update(max_single_rr=mx, pair_over_max_single=P.rr / mx, pair_exceeds_max_single=bool(P.rr > mx),
                           pair_ci_excludes_1_above=bool(P.ci_low > 1), criterion_breadth_met=bool(P.rr > mx and P.ci_low > 1))
            else:
                row["criterion_breadth_met"] = False
                row["note"] = "support failure: pair or a single not estimable/pooled"
            row["pair_ci_excludes_1_above_flag"] = bool(P is not None and P.ci_low > 1)
            dec_rows.append(row)
    dec = pd.DataFrame(dec_rows)
    verdicts = {}
    for outcome in ["mortality", "incident_any_adl"]:
        dd = dec[dec.outcome.eq(outcome)].set_index("pair")
        n_breadth = int(dd.criterion_breadth_met.sum())
        all_eval = bool(dd.pair_pooled.all())
        act_pairs = bool(dd.loc["DA", "pair_ci_excludes_1_above_flag"] and dd.loc["AW", "pair_ci_excludes_1_above_flag"])
        nonact = bool(dd.loc["DW", "pair_ci_excludes_1_above_flag"])
        breadth = n_breadth >= 2
        actdom = all_eval and act_pairs and not nonact
        verdicts[outcome] = {"verdict": "breadth generalises" if breadth else ("activity-dominated" if actdom else "inconclusive"),
                             "n_pairs_meeting_breadth_criterion": n_breadth, "breadth_met": breadth, "activity_dominated_condition_met": actdom,
                             "all_three_pairs_estimable": all_eval, "DA_ci_excludes_1_above": bool(dd.loc["DA", "pair_ci_excludes_1_above_flag"]),
                             "AW_ci_excludes_1_above": bool(dd.loc["AW", "pair_ci_excludes_1_above_flag"]), "DW_ci_excludes_1_above": nonact,
                             "scored": outcome == "mortality"}
    return dec, verdicts

dec1, v1 = decide("pooled", "POOLED")
dec2, v2 = decide("one_stage_S1", "ONE_STAGE_ESTIMABLE")
dec3, v3 = decide("one_stage_S1b_posthoc", "ONE_STAGE_ESTIMABLE") if (allrows.row_type == "one_stage_S1b_posthoc").any() else (pd.DataFrame(), {})
pd.concat([dec1, dec2, dec3], ignore_index=True).to_csv(FINAL / "e3_decision.csv", index=False)
(FINAL / "e3_verdict.json").write_text(json.dumps({"validation": chk, "primary_REML_HK": v1, "supplementary_one_stage_S1": v2, "posthoc_one_stage_S1b": v3}, indent=1, default=str))
print(json.dumps({"status": "PASS", "validation": chk, "primary_REML_HK": v1, "supplementary_one_stage_S1": v2, "posthoc_one_stage_S1b": v3}, indent=1, default=str))
