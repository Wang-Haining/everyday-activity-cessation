#!/usr/bin/env python3
"""Count-only step: verify the risk sets, then tabulate the eight loss patterns (E3 step 1)
and the loss x gain cross-tabulation (E4). No effect model is fitted here."""
import argparse, hashlib, json
import numpy as np, pandas as pd
from e34_common import *   # noqa

ap = argparse.ArgumentParser(); ap.add_argument("--cohort", required=True); a = ap.parse_args()
co = load_cohort(a.cohort)
pat_rows, x_rows, verify = [], [], []
for outcome in OUTCOMES:
    if a.cohort not in PUB[outcome]:
        continue
    d, X0, got = build_frame(co, outcome)          # asserts the published n / events / people
    verify.append({"cohort": a.cohort, "outcome": outcome, **got, "published": list(PUB[outcome][a.cohort]), "match": True})
    # --- E3 step 1: eight patterns
    assert d["pattern"].value_counts().sum() == len(d)
    for p in PATTERNS:
        m = d["pattern"].eq(p)
        n, ev = int(m.sum()), int(d.loc[m, "y"].sum())
        row = {"cohort": a.cohort, "outcome": outcome, "pattern": p, "label_DAW": p, "n_losses": p.count("1"),
               "n": n, "people": int(d.loc[m, "person_id"].nunique()), "events": ev,
               "crude_risk": ev / n if n else np.nan}
        for k in (1, 2, 3):
            row[f"initial_opps_{k}_n"] = int((m & d["opp"].eq(k)).sum())
            row[f"initial_opps_{k}_share"] = row[f"initial_opps_{k}_n"] / n if n else np.nan
        pat_rows.append(row)
    assert sum(r["n"] for r in pat_rows if r["cohort"] == a.cohort and r["outcome"] == outcome) == len(d)
    # --- E4 crosstab
    d["loss_cat"] = np.where(d["loss_n"] == 0, "0", np.where(d["loss_n"] == 1, "1", "2plus"))
    d["gain_cat"] = np.where(d["gain_n"] == 0, "0", "1plus")
    for lc in ["0", "1", "2plus"]:
        for gc in ["0", "1plus"]:
            m = d["loss_cat"].eq(lc) & d["gain_cat"].eq(gc)
            n, ev = int(m.sum()), int(d.loc[m, "y"].sum())
            x_rows.append({"row_type": "crosstab", "cohort": a.cohort, "outcome": outcome, "loss_cat": lc, "gain_cat": gc,
                           "n": n, "people": int(d.loc[m, "person_id"].nunique()), "events": ev,
                           "crude_risk": ev / n if n else np.nan})
    # gain share by loss category (all rows, and rows where a gain is possible: initial count <= 2)
    for lc in ["0", "1", "2plus"]:
        m = d["loss_cat"].eq(lc)
        for scope, mm in [("all_primary_rows", m), ("gain_possible_initial_opps_le2", m & d["opp"].le(2))]:
            n = int(mm.sum()); g = int((mm & d["gain_n"].ge(1)).sum())
            x_rows.append({"row_type": "gain_share", "cohort": a.cohort, "outcome": outcome, "loss_cat": lc, "scope": scope,
                           "n": n, "n_with_gain": g, "share_with_gain": g / n if n else np.nan})
        for k in (1, 2, 3):
            mm = m & d["opp"].eq(k); n = int(mm.sum()); g = int((mm & d["gain_n"].ge(1)).sum())
            x_rows.append({"row_type": "gain_share_by_initial_opps", "cohort": a.cohort, "outcome": outcome, "loss_cat": lc,
                           "initial_opps": k, "n": n, "n_with_gain": g, "share_with_gain": g / n if n else np.nan})
write_csv(OUT / f"tables_patterns_{a.cohort}.csv", pd.DataFrame(pat_rows))
write_csv(OUT / f"tables_e4_crosstab_{a.cohort}.csv", pd.DataFrame(x_rows))
write_csv(OUT / f"tables_verify_{a.cohort}.csv", pd.DataFrame(verify))
print(json.dumps({"status": "PASS", "cohort": a.cohort, "verify": verify}, default=str))
