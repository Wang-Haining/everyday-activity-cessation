#!/usr/bin/env python3
"""A: cohort-specific pair-versus-single models (E3.3 design) and the one-loss gain contrast (E4 design,
lost-domain adjusted). One cohort per task; HRS is run for reference only. Aggregates only."""
import argparse, hashlib, json
import numpy as np, pandas as pd
from x3_common import *    # noqa
from a_defs import *       # noqa

ap = argparse.ArgumentParser(); ap.add_argument("--cohort", required=True); a = ap.parse_args()
co = load_cohort(a.cohort); pilot = co.pilot
rows, manifest = [], {"cohort": a.cohort, "verified": {}, "aggregate_only": True, "respondent_rows_exported": 0}


def tag(res, **kw):
    for r in res:
        r.update({"cohort": a.cohort, "row_type": "cohort", **kw})
    return res


for outcome in OUTCOMES:
    if a.cohort not in PUB[outcome]:
        continue
    d, X0, got = build_frame(co, outcome)
    manifest["verified"][outcome] = got
    # validation: refit of the published 0/1/2+ model
    g = {"loss_1": d["loss_n"].eq(1), "loss_2plus": d["loss_n"].ge(2)}
    res, fit, kept = fit_groups(d, X0, g, ["loss_1", "loss_2plus"], pilot)
    assert np.allclose([r["rr"] for r in res], PUB_RR[outcome][a.cohort], rtol=2e-5), (a.cohort, outcome)
    rows += tag(res, analysis="validation_count_model", population="all_primary", outcome=outcome)
    # E3.3 restricted pairs
    for d1, d2 in PAIRS:
        pair, names, m, groups = pair_groups(d, d1, d2)
        ds, Xs = d.loc[m], X0.loc[m]
        gs = {k: v.loc[m] for k, v in groups.items()}
        res, fit, kept = fit_groups(ds, Xs, gs, names[:3], pilot)
        info = dict(analysis="composition_pair", population=f"both_{LETTER[d1]}_{LETTER[d2]}_present_at_t0", pair=pair, outcome=outcome)
        rows += tag(res, **info)
        base_info = {**info, "cohort": a.cohort, "row_type": "cohort", "n_population": int(len(ds)), "events_population": int(ds["y"].sum())}
        rows += contrast_rows(fit, kept, [(f"contrast_{names[0]}_vs_{names[1]}", names[0], names[1]),
                                           (f"contrast_{names[0]}_vs_{names[2]}", names[0], names[2])], base_info)
    # E4 one-loss gain contrast, three specifications
    m = e4_mask(d)
    for vname, fn in E4_VARIANTS.items():
        ds, Xs = d.loc[m], X0.loc[m]
        if fn is not None:
            Xs = pd.concat([Xs, fn(ds)], axis=1)
        Xs = Xs.drop(columns=E4_DROP.get(vname, []))
        g = {"gain": ds["gain_n"].ge(1)}
        res, fit, kept = fit_groups(ds, Xs, g, ["gain"], pilot)
        rows += tag(res, analysis="gain_within_one_loss", population=vname, stratum="loss_1", outcome=outcome)
    # descriptive cells: gain x lost domain within the stratum
    ds = d.loc[m]
    for s in DOMS:
        for gv, lab in [(0, "no_gain"), (1, "gain")]:
            c = ds[(ds[f"{s}_loss"].eq(1)) & (ds["gain_n"].ge(1).astype(int).eq(gv))]
            rows.append({"cohort": a.cohort, "row_type": "cell_counts", "analysis": "gain_within_one_loss", "outcome": outcome,
                         "population": "attainable", "term": f"lost_{LETTER[s]}_{lab}", "n_group": int(len(c)),
                         "people_group": int(c["person_id"].nunique()), "events_group": int(c["y"].sum()),
                         "risk_group": float(c["y"].mean()) if len(c) else np.nan, "status": "DESCRIPTIVE"})

out = pd.DataFrame(rows)
write_csv(OUT / f"a_fit_{a.cohort}.csv", out)


print(json.dumps({"status": "PASS", "cohort": a.cohort, "rows": len(out)}))
