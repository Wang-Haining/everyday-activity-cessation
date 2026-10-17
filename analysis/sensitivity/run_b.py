#!/usr/bin/env python3
"""B: delayed landmarks. Repeat the 0/1/2+ mortality comparison among people known alive L months after the
t1 interview (death month index > t1 index + L), counting deaths through the original outcome window.
Exposures stay fixed (t0 -> t1). One cohort per task. Aggregates only."""
import argparse, hashlib, json
import numpy as np, pandas as pd
from x3_common import *    # noqa

LANDMARKS = [0, 6, 12]
ap = argparse.ArgumentParser(); ap.add_argument("--cohort", required=True); a = ap.parse_args()
co = load_cohort(a.cohort); pilot = co.pilot
d, X0, got = build_frame(co, "mortality")
ep = co.episodes.loc[d.index]
dd, src = load_death_dates(co)
t1 = ep["t1__interview_month_index"]
idx = d["person_id"].map(dd["idx"]); mo = d["person_id"].map(dd["mo"]); yr = d["person_id"].map(dd["yr"])
cons = d["person_id"].map(dd["consistent"]).fillna(True)
dead = d["y"].eq(1)
diff = idx - t1                                  # months from the t1 interview month to the death month
dated = idx.notna() & cons
n_dead, n_dated = int(dead.sum()), int((dead & dated).sum())
share_dated = n_dated / n_dead
USABLE = share_dated >= 0.85                     # death-date coverage threshold
manifest = {"cohort": a.cohort, "verified": got, "death_date_source": src, "deaths": n_dead, "deaths_dated_month": n_dated,
            "share_dated": share_dated, "usable_for_landmarks": bool(USABLE), "aggregate_only": True, "respondent_rows_exported": 0}
rows = []

# stable-subset mask (E2 definition: t-1 interview present, self-rated health not worse, ADL-free at t0)
s_m, s_0 = srh(co, "tm1").loc[d.index], srh(co, "t0").loc[d.index]
a_m, a_0 = adl_count(co, "tm1").loc[d.index], adl_count(co, "t0").loc[d.index]
T = ep["tm1_present"] & s_m.notna() & s_0.notna() & a_m.notna() & a_0.notna()
stable = T & (s_0 <= s_m) & (a_0 == 0)
POPS = {"all_primary": pd.Series(True, index=d.index), "has_t_minus_1_valid": T, "stable_prior_health": stable}

if not USABLE:
    rows.append({"cohort": a.cohort, "row_type": "cohort", "population": "all_primary", "landmark_months": np.nan, "rule": "",
                 "term": "all", "status": f"NOT_USABLE_death_dates_{src}_dated_share_{share_dated:.3f}"})
else:
    g_all = {"loss_1": d["loss_n"].eq(1), "loss_2plus": d["loss_n"].ge(2)}
    for pname, pmask in POPS.items():
        for L in LANDMARKS:
            for rule in (["strict", "lenient", "strict_undated_retained"] if L > 0 else ["strict"]):
                if L == 0:
                    keep = pmask & (~dead | dated)                     # dated deaths only; no landmark
                elif rule == "strict_undated_retained":
                    keep = pmask & (~dead | ~dated | (dated & (diff > L)))   # undated deaths kept as events (assumed late)
                else:
                    later = (diff > L) if rule == "strict" else (diff >= L)
                    keep = pmask & (~dead | (dated & later))
                ds, Xs = d.loc[keep], X0.loc[keep]
                gs = {k: v.loc[keep] for k, v in g_all.items()}
                if len(ds) == 0:
                    continue
                res, fit, kept = fit_groups(ds, Xs, gs, ["loss_1", "loss_2plus"], pilot)
                removed = pmask & dead & ~keep
                for r in res:
                    r.update({"cohort": a.cohort, "row_type": "cohort", "analysis": "delayed_landmark", "population": pname,
                              "landmark_months": L, "rule": rule, "deaths_removed_vs_primary_population": int(removed.sum()),
                              "deaths_primary_population": int((pmask & dead).sum())})
                rows += res
                if pname == "all_primary" and L == 0:
                    pub = PUB_RR["mortality"][a.cohort]
                    manifest["dated_only_rr_vs_published"] = [[r.get("rr") for r in res], list(pub)]
    # timing of deaths by exposure group (dated deaths, all primary)
    for lab, m in [("0", d["loss_n"].eq(0)), ("1", d["loss_n"].eq(1)), ("2plus", d["loss_n"].ge(2))]:
        n_g = int(m.sum()); dg = dead & m
        r = {"cohort": a.cohort, "row_type": "timing", "analysis": "death_timing", "population": "all_primary", "term": f"loss_{lab}",
             "n_group": n_g, "events_group": int(dg.sum()), "deaths_dated": int((dg & dated).sum()), "deaths_undated": int((dg & ~dated).sum())}
        for L in (3, 6, 12):
            r[f"deaths_within_{L}_months_strict"] = int((dg & dated & (diff <= L)).sum())
        r["deaths_dated_before_t1_month"] = int((dg & dated & (diff < 0)).sum())
        rows.append(r)

out = pd.DataFrame(rows)
write_csv(OUT / f"b_fit_{a.cohort}.csv", out)


print(json.dumps({"status": "PASS", "cohort": a.cohort, "rows": len(out), "usable": bool(USABLE), "share_dated": share_dated}))
