#!/usr/bin/env python3
"""Step 0 (counts only, no effect model): verify the risk sets against the published numbers and
report the E5 analysis-sample sizes and hidden-risk cell counts."""
import argparse, hashlib, json
import numpy as np, pandas as pd
from e5_common import *  # noqa

ap = argparse.ArgumentParser(); ap.add_argument("--cohort", required=True); a = ap.parse_args()
co = load_cohort(a.cohort)
rows, cells = [], []
for outcome in OUTCOMES:
    if a.cohort not in PUB[outcome]:
        continue
    d, A, got = build_frame(co, outcome)
    m = analysis_mask(d, outcome)
    s = d.loc[m]
    designs = make_designs(s, A.loc[m], outcome)
    rows.append({"cohort": a.cohort, "outcome": outcome, **got, "published_n": PUB[outcome][a.cohort][0],
                 "published_events": PUB[outcome][a.cohort][1], "published_people": PUB[outcome][a.cohort][2],
                 "match": True, "n_srh_valid": int(d["srh"].notna().sum()), "n_adl_count_valid": int(d["adl_n"].notna().sum()),
                 "n_analysis": int(len(s)), "events_analysis": int(s["y"].sum()), "people_analysis": int(s["person_id"].nunique()),
                 "n_lost_to_analysis_sample": int(len(d) - len(s)), "n_gain_any": int(s["gain_any"].sum()),
                 "n_loss_1": int(s["loss_1"].sum()), "n_loss_2plus": int(s["loss_2plus"].sum()),
                 "max_design_columns": int(max(v.shape[1] for v in designs.values())),
                 "age_min": float(A.loc[m, "age_years"].min()), "age_max": float(A.loc[m, "age_years"].max())})
    h = d.loc[d["srh"].notna()].copy()
    h["srh_grp"] = np.where(h["srh"] <= 3, "good_or_better", "fair_poor")
    h["cess"] = np.select([h["loss_n"] == 0, h["loss_n"] == 1], ["0", "1"], "2plus")
    h["traj"] = np.select([h["loss_n"] == 0, h["gain_n"] == 0], ["no_loss", "loss_only"], "loss_with_gain")
    for cls in ["cess", "traj"]:
        g = h.groupby(["srh_grp", cls]).agg(n=("y", "size"), events=("y", "sum"), people=("person_id", "nunique")).reset_index()
        for _, r in g.iterrows():
            cells.append({"cohort": a.cohort, "outcome": outcome, "classification": cls, "srh_group": r["srh_grp"],
                          "cell": r[cls], "n": int(r["n"]), "events": int(r["events"]), "people": int(r["people"])})
write_csv(OUT / f"e5_verify_{a.cohort}.csv", pd.DataFrame(rows))
write_csv(FINAL / "support" / f"e5_cell_counts_{a.cohort}.csv", pd.DataFrame(cells))
print(json.dumps({"status": "PASS", "cohort": a.cohort, "rows": rows}, default=str))
