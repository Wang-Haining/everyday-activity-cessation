#!/usr/bin/env python3
"""Step 0 (counts only; no effect model): verify the published risk sets, document death-date
coverage for B and the exposure-eligible / retained flow for C. One cohort per task."""
import argparse, json
import numpy as np, pandas as pd
from x3_common import *   # noqa

ap = argparse.ArgumentParser(); ap.add_argument("--cohort", required=True); a = ap.parse_args()
co = load_cohort(a.cohort)
rows = []


def add(block, item, group, value, **kw):
    rows.append({"cohort": a.cohort, "block": block, "item": item, "group": group, "value": value, **kw})


# ---------------------------------------------------------------- risk set verification
frames = {}
for outcome in OUTCOMES:
    if a.cohort not in PUB[outcome]:
        continue
    d, X0, got = build_frame(co, outcome)          # asserts the published n / events / people
    frames[outcome] = d
    for k, v in got.items():
        add("verify", f"{outcome}_{k}", "", v, published=PUB[outcome][a.cohort][["n", "events", "people"].index(k)], match=True)

# ---------------------------------------------------------------- B: death-date coverage
d = frames["mortality"]
ep = co.episodes.loc[d.index]
dd, src = load_death_dates(co)
t1 = ep["t1__interview_month_index"]
add("B_dates", "source", src, np.nan)
add("B_dates", "t1_interview_month_index_missing", "risk set", int(t1.isna().sum()))
dead = d["y"].eq(1)
yr = d["person_id"].map(dd["yr"]); mo = d["person_id"].map(dd["mo"]); idx = d["person_id"].map(dd["idx"])
cons = d["person_id"].map(dd["consistent"]).fillna(True)
add("B_dates", "deaths_in_window", "", int(dead.sum()))
add("B_dates", "deaths_year_and_month", "", int((dead & idx.notna()).sum()))
add("B_dates", "deaths_year_only", "", int((dead & yr.notna() & mo.isna()).sum()))
add("B_dates", "deaths_no_date", "", int((dead & yr.isna()).sum()))
add("B_dates", "deaths_person_level_inconsistent_dates", "", int((dead & ~cons).sum()))
add("B_dates", "survivors_with_a_death_date_recorded", "", int((~dead & idx.notna()).sum()))
diff = (idx - t1)
for lab, lo, hi in [("<=0", -1e9, 0), ("1-6", 1, 6), ("7-12", 7, 12), ("13-24", 13, 24), (">24", 25, 1e9)]:
    add("B_dates", "dated_deaths_death_minus_t1_months", lab, int((dead & diff.between(lo, hi)).sum()))
for w, g in d.loc[dead].groupby("t1_wave"):
    add("B_dates", "wave_deaths_dated_share", f"t1_wave={int(w)}", float(idx.loc[g.index].notna().mean()), n=int(len(g)))
if a.cohort == "hrs":
    p = DATA / co.universe["cohorts"]["hrs"]["primary_source_relative_path"]
    w_ = pd.read_stata(p, columns=[co.universe["cohorts"]["hrs"]["id"], "radtimtdth"], convert_categoricals=False, preserve_dtypes=False)
    w_["person_id"] = normalize_id(w_[co.universe["cohorts"]["hrs"]["id"]])
    rt = d["person_id"].map(w_.set_index("person_id")["radtimtdth"])
    delta = (diff - rt)
    ok = dead & delta.notna()
    add("B_dates", "hrs_radtimtdth_available_deaths", "", int(ok.sum()))
    for lab, lo, hi in [("|diff|<=1", -1, 1), ("|diff| 2-3", None, None)]:
        if lo is not None:
            add("B_dates", "hrs_computed_gap_minus_radtimtdth", lab, int((ok & delta.between(lo, hi)).sum()))
    add("B_dates", "hrs_computed_gap_minus_radtimtdth", "|diff|>=2", int((ok & ~delta.between(-1, 1)).sum()))

# ---------------------------------------------------------------- C: eligible / retained flow
for outcome in OUTCOMES:
    if a.cohort not in PUB[outcome]:
        continue
    f, cov = selection_frame(co, outcome)
    ret_idx = set(frames[outcome].index)
    calc_idx = set(f.index[f["retained_calc"]])
    add("C_flow", f"{outcome}_retained_flags_equal_published_set", "", bool(ret_idx == calc_idx),
        n_published_not_flagged=len(ret_idx - calc_idx), n_flagged_not_published=len(calc_idx - ret_idx))
    el = f["eligible"]
    add("C_flow", f"{outcome}_eligible", "", int(el.sum()))
    add("C_flow", f"{outcome}_retained", "", len(ret_idx))
    add("C_flow", f"{outcome}_excluded_outcome_missing", "", int((el & f["miss_outcome"]).sum()))
    add("C_flow", f"{outcome}_excluded_covariates_only", "", int((el & ~f["miss_outcome"] & ~f["cov_ok"]).sum()))
    add("C_flow", f"{outcome}_excluded_both", "", int((el & f["miss_outcome"] & ~f["cov_ok"]).sum()))
    if outcome != "mortality":
        add("C_flow", "adl_free_t1_all", "", int(f["adl_free_t1"].sum()))
        add("C_flow", "adl_died_before_t2", "", int(f["died_before_t2"].sum()))
        add("C_flow", "adl_status_unknown", "", int(f["status_unknown"].sum()))
    ep_b = co.behavior
    lc = ep_b["loss_count_core"]
    for lab, m in [("loss0", lc.eq(0)), ("loss1", lc.eq(1)), ("loss2plus", lc.ge(2))]:
        e_ = el & m
        add("C_flow", f"{outcome}_eligible_by_loss", lab, int(e_.sum()), retained=int((e_ & f.index.isin(ret_idx)).sum()))
    add("C_flow", f"{outcome}_retained_not_in_eligible", "", int(sum(1 for i in ret_idx if not bool(el.loc[i]))))

out = pd.DataFrame(rows)
write_csv(OUT / f"step0_{a.cohort}.csv", out)
print(json.dumps({"status": "PASS", "cohort": a.cohort, "rows": len(out)}))
