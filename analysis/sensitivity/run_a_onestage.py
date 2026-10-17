#!/usr/bin/env python3
"""A: one-stage cohort-stratified common-coefficient fits for composition (pairs vs singles) and the
one-loss gain contrast, in two cohort sets: without HRS (primary for A) and with all cohorts (reference
used for the one-stage analysis). All cohorts are loaded in this process; aggregates only."""
import gc, hashlib, json
import numpy as np, pandas as pd
from x3_common import *    # noqa
from a_defs import *       # noqa

parts = {o: {} for o in OUTCOMES}
pilot = None
manifest = {"verified": {}, "aggregate_only": True, "respondent_rows_exported": 0}
for cohort in MORT_COHORTS:
    co = load_cohort(cohort)
    pilot = co.pilot
    for outcome in OUTCOMES:
        if cohort not in PUB[outcome]:
            continue
        d, X0, got = build_frame(co, outcome)
        d["person_id"] = cohort + "_" + d["person_id"].astype(str)
        parts[outcome][cohort] = (d, X0)
        manifest["verified"][f"{cohort}/{outcome}"] = got
    del co; gc.collect()

SETS = {"one_stage_noHRS": lambda pl: {c: v for c, v in pl.items() if c != "hrs"},
        "one_stage_all_cohorts": lambda pl: dict(pl)}
rows = []
for outcome in OUTCOMES:
    for rt, sel in SETS.items():
        pl = sel(parts[outcome])
        lab = dict(outcome=outcome, row_type=rt, cohort="ONE_STAGE")
        for d1, d2 in PAIRS:
            def fn(d, d1=d1, d2=d2):
                pair, names, m, groups = pair_groups(d, d1, d2)
                return m, groups
            pair, names, _m, _g = pair_groups(next(iter(pl.values()))[0], d1, d2)
            rows += stacked_fit(pl, fn, names[:3], {**lab, "analysis": "composition_pair", "population": f"both_{LETTER[d1]}_{LETTER[d2]}_present_at_t0", "pair": pair},
                                pilot, contrasts=[(f"contrast_{names[0]}_vs_{names[1]}", names[0], names[1]),
                                                  (f"contrast_{names[0]}_vs_{names[2]}", names[0], names[2])])
        for vname, efn in E4_VARIANTS.items():
            def fns(d):
                return e4_mask(d), {"gain": d["gain_n"].ge(1)}
            def efn_masked(d, efn=efn):
                return efn(d) if efn is not None else pd.DataFrame(index=d.index)
            rows += stacked_fit(pl, fns, ["gain"], {**lab, "analysis": "gain_within_one_loss", "population": vname, "stratum": "loss_1"},
                                pilot, extra_fn=efn_masked, drop_cols=E4_DROP.get(vname, []))
out = pd.DataFrame(rows)
write_csv(OUT / "a_onestage.csv", out)


print(json.dumps({"status": "PASS", "rows": len(out)}))
