#!/usr/bin/env python3
"""A: validate against the E3/E4 outputs, pool cohort rows (REML-HK, safeguarded CI primary; at least 3 cohorts),
derive pair / larger-single rows and write a_composition_noHRS.csv and a_gain_noHRS.csv (aggregate only)."""
import glob, json
import numpy as np, pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from x3_common import pool_log, write_csv, OUT   # noqa

V2 = Path(__file__).resolve().parents[2] / "results" / "composition" / "cohort"
FINAL = OUT.parent; FINAL.mkdir(parents=True, exist_ok=True)
KEYS = ["analysis", "population", "pair", "stratum", "outcome", "term"]
coh = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(OUT / "a_fit_*.csv")))], ignore_index=True)
one = pd.read_csv(OUT / "a_onestage.csv")
for df in (coh, one):
    for k in KEYS:
        if k not in df.columns:
            df[k] = ""
    df[KEYS] = df[KEYS].fillna("")
    if "is_contrast" not in df.columns:
        df["is_contrast"] = False
    df["is_contrast"] = df["is_contrast"].fillna(False).astype(bool)
coh["hrs_in_estimate"] = coh["cohort"].eq("hrs")
one["hrs_in_estimate"] = one["row_type"].eq("one_stage_all_cohorts")

# ------------------------------------------------------------------ validation against v0.2
MAP_A = {"composition_pair": "e3_restricted_pair"}
def v2_key(r):
    if r["analysis"] == "composition_pair":
        return ("e3_restricted_pair", r["population"], r["pair"], "", r["outcome"], r["term"])
    if r["analysis"] == "gain_within_one_loss" and r["population"] == "attainable_unadjusted":
        return ("e4_gain_within_loss_stratum", "stratum_gain_attainable_initial_counts", "", "loss_1", r["outcome"], r["term"])
    return None

checks = []
def compare(mine, theirs, label):
    t = theirs.copy()
    for k in ["analysis", "population", "pair", "stratum", "outcome", "term"]:
        if k not in t.columns:
            t[k] = ""
    t = t.fillna({k: "" for k in ["pair", "stratum", "population"]})
    t["is_contrast"] = t["is_contrast"].fillna(False).astype(bool) if "is_contrast" in t else False
    tk = {(r["analysis"], r["population"], r["pair"], r["stratum"], r["outcome"], r["term"]): r for _, r in t.iterrows()}
    n_cmp = n_bad = n_status = 0
    for _, r in mine.iterrows():
        k = v2_key(r)
        if k is None or k not in tk or r["is_contrast"]:
            continue
        o = tk[k]
        s1, s2 = str(r["status"]) == "ESTIMABLE", str(o["status"]) == "ESTIMABLE"
        if s1 != s2:
            n_status += 1
            continue
        if s1:
            n_cmp += 1
            if not (abs(np.log(r["rr"]) - np.log(o["rr"])) < 1e-6 and abs(r["se_log_rr"] - o["se_log_rr"]) < 1e-6):
                n_bad += 1
    checks.append({"comparison": label, "rows_compared": n_cmp, "rows_mismatch_estimate": n_bad, "rows_status_differs": n_status})

v2s1b = pd.read_csv(V2 / "onestage_S1b_posthoc.csv")
compare(one[one.row_type.eq("one_stage_all_cohorts")], v2s1b, "one_stage_all_cohorts_vs_E3E4_one_stage")
for c in ["elsa", "hrs", "klosa", "share"]:
    compare(coh[coh.cohort.eq(c) & coh.row_type.eq("cohort")], pd.read_csv(V2 / f"fit_{c}.csv"), f"cohort_{c}_vs_E3E4_cohort_fit")
chk = pd.DataFrame(checks)
write_csv(FINAL / "a_validation_vs_E3E4.csv", chk)
assert (chk.rows_mismatch_estimate.eq(0).all() and chk.rows_status_differs.eq(0).all() and chk.rows_compared.sum() > 0), chk.to_string()

# ------------------------------------------------------------------ pooling
est_rows = coh[coh.row_type.eq("cohort") & coh.analysis.isin(["composition_pair", "gain_within_one_loss"])]
pooled = []
for key, g in est_rows.groupby(KEYS, sort=False):
    meta = dict(zip(KEYS, key))
    for label, sel in [("pooled_noHRS", g[~g.cohort.eq("hrs")]), ("pooled_all_cohorts_reference", g)]:
        est = sel[sel.status.eq("ESTIMABLE") & sel.rr.notna()].sort_values("cohort")
        fail = ";".join(f"{c}:{s}" for c, s in zip(sel.cohort, sel.status) if s != "ESTIMABLE")
        base = {**meta, "row_type": label, "cohort": "POOLED", "hrs_in_estimate": label != "pooled_noHRS", "k": int(len(est)),
                "cohorts": ";".join(est.cohort), "cohorts_not_evaluable": fail, "per_cohort_rr": ";".join(f"{c}:{r:.3f}" for c, r in zip(est.cohort, est.rr)),
                "is_contrast": bool(g.is_contrast.iloc[0])}
        if len(est) < 3:
            pooled.append({**base, "status": "NOT_POOLED_fewer_than_3_evaluable_cohorts"}); continue
        pooled.append({**base, "status": "POOLED", **pool_log(est)})
allr = pd.concat([coh[coh.analysis.isin(["composition_pair", "gain_within_one_loss"])], one, pd.DataFrame(pooled)], ignore_index=True)

# ------------------------------------------------------------------ pair / larger single (derived rows)
derived = []
comp = allr[allr.analysis.eq("composition_pair")]
for (rt, ch, outc, pair), g in comp.groupby(["row_type", "cohort", "outcome", "pair"], sort=False):
    ok = lambda r: str(r.status) in ("ESTIMABLE", "ONE_STAGE_ESTIMABLE", "POOLED")
    def get(term):
        x = g[g.term.eq(term)]
        return x.iloc[0] if len(x) else None
    pt, s1, s2 = f"pair_{pair}", f"single_{pair[0]}", f"single_{pair[1]}"
    P, A, B = get(pt), get(s1), get(s2)
    row = {"analysis": "composition_pair", "population": g.population.iloc[0], "pair": pair, "outcome": outc, "term": "pair_over_larger_single",
           "row_type": rt, "cohort": ch, "is_contrast": True, "hrs_in_estimate": bool(g.hrs_in_estimate.iloc[0]) if "hrs_in_estimate" in g else np.nan}
    singles = {n: r for n, r in [(s1, A), (s2, B)] if r is not None and r.get("rr") == r.get("rr") and r.get("rr") is not None and not pd.isna(r.get("rr"))}
    if P is None or pd.isna(P.get("rr")) or not singles:
        row["status"] = "NOT_EVALUABLE_pair_or_singles"
        derived.append(row); continue
    larger = max(singles, key=lambda n: singles[n].rr)
    c = g[g.term.eq(f"contrast_{pt}_vs_{larger}")]
    row.update({"larger_single": larger, "pair_rr": P.rr, "larger_single_rr": singles[larger].rr,
                "n_singles_evaluable": len(singles)})
    if len(singles) < 2:
        # the larger single cannot be identified when one single fails support: report the contrast with the
        # evaluable single under a different name and say so
        row["term"] = "pair_over_evaluable_single_only"
        row["note"] = f"other single not evaluable; ratio is against {larger}, which need not be the larger single"
    if len(c) and not pd.isna(c.iloc[0].get("rr")):
        cc = c.iloc[0]
        row.update({"status": "ESTIMABLE" if rt in ("cohort",) else cc["status"], "rr": cc.rr, "ci_low": cc.ci_low, "ci_high": cc.ci_high, "se_log_rr": cc.get("se_log_rr"),
                    "epp_ge10": cc.get("epp_ge10"), "events_per_parameter": cc.get("events_per_parameter"), "n_parameters": cc.get("n_parameters"),
                    "events_model": cc.get("events_model")})
        row["reading"] = "supported" if cc.ci_low > 1 else ("direction_only" if cc.rr > 1 else "not_supported")
    else:
        row["status"] = "NOT_EVALUABLE_contrast_row_missing"
    derived.append(row)
allr = pd.concat([allr, pd.DataFrame(derived)], ignore_index=True)

order = ["analysis", "outcome", "population", "pair", "stratum", "row_type", "cohort", "term"]
allr = allr.sort_values(order, kind="stable")
comp_out = allr[allr.analysis.eq("composition_pair")].drop(columns=[c for c in ["stratum"] if c in allr.columns])
gain_out = allr[allr.analysis.eq("gain_within_one_loss")].drop(columns=[c for c in ["pair"] if c in allr.columns])
write_csv(FINAL / "a_composition_noHRS.csv", comp_out)
write_csv(FINAL / "a_gain_noHRS.csv", gain_out)
print(chk.to_string())
