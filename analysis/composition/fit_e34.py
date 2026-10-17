#!/usr/bin/env python3
"""E3 (pattern model, restricted pairs, leave-one-domain-out) and E4 (losses x gains models)
for one cohort. Verifies the published risk set before any fit. Writes aggregate rows only."""
import argparse, hashlib, json
import numpy as np, pandas as pd
from e34_common import *   # noqa

ap = argparse.ArgumentParser(); ap.add_argument("--cohort", required=True); a = ap.parse_args()
co = load_cohort(a.cohort)
pilot = co.pilot
rows = []          # all cohort-level model rows, tagged
manifest = {"cohort": a.cohort, "verified": {}, "aggregate_only": True, "respondent_rows_exported": 0}

PAIRS = [("alcohol", "activity"), ("alcohol", "work"), ("activity", "work")]


def tag(res, **kw):
    for r in res:
        r.update({"cohort": a.cohort, **kw})
    return res


for outcome in OUTCOMES:
    if a.cohort not in PUB[outcome]:
        continue
    d, X0, got = build_frame(co, outcome)
    manifest["verified"][outcome] = got
    y = d["y"]
    P = d["pattern"]

    # ---------------- validation: refit of the published 0/1/2+ model
    g = {"loss_1": d["loss_n"].eq(1), "loss_2plus": d["loss_n"].ge(2)}
    res, fit, kept = fit_groups(d, X0, g, ["loss_1", "loss_2plus"], pilot)
    got_rr = [r["rr"] for r in res]
    assert np.allclose(got_rr, PUB_RR[outcome][a.cohort], rtol=2e-5), (a.cohort, outcome, got_rr)
    rows += tag(res, analysis="validation_count_model", population="all_primary", outcome=outcome)

    # ---------------- E3 step 2: seven-pattern model, reference 000, full risk set
    g = {f"p{p}": P.eq(p) for p in PATTERNS[1:]}
    res, fit, kept = fit_groups(d, X0, g, list(g), pilot)
    rows += tag(res, analysis="e3_pattern_model", population="all_primary", outcome=outcome)

    # ---------------- E3 step 3: restricted pairs (both domains of the pair present at t0)
    for d1, d2 in PAIRS:
        third = [s for s in DOMS if s not in (d1, d2)][0]
        m = d[f"{d1}_base"].eq(1) & d[f"{d2}_base"].eq(1)
        ds, Xs = d.loc[m], X0.loc[m]
        l1, l2, l3 = ds[f"{d1}_loss"].eq(1), ds[f"{d2}_loss"].eq(1), ds[f"{third}_loss"].eq(1)
        pair = f"{LETTER[d1]}{LETTER[d2]}"
        gname = {"pair": f"pair_{pair}", "s1": f"single_{LETTER[d1]}", "s2": f"single_{LETTER[d2]}", "nuis": f"nuisance_{LETTER[third]}_lost"}
        g = {gname["pair"]: l1 & l2 & ~l3, gname["s1"]: l1 & ~l2 & ~l3, gname["s2"]: ~l1 & l2 & ~l3, gname["nuis"]: l3}
        focal = [gname["pair"], gname["s1"], gname["s2"]]
        res, fit, kept = fit_groups(ds, Xs, g, focal, pilot)
        info = dict(analysis="e3_restricted_pair", population=f"both_{LETTER[d1]}_{LETTER[d2]}_present_at_t0", pair=pair, outcome=outcome)
        rows += tag(res, **info)
        base_info = {**{k: v for k, v in info.items()}, "cohort": a.cohort,
                     "n_population": int(len(ds)), "events_population": int(ds["y"].sum())}
        rows += contrast_rows(fit, kept, [(f"contrast_{gname['pair']}_vs_{gname['s1']}", gname["pair"], gname["s1"]),
                                           (f"contrast_{gname['pair']}_vs_{gname['s2']}", gname["pair"], gname["s2"])], base_info)

    # ---------------- E3 step 4 (supplement): leave-one-domain-out counts
    for omit in DOMS:
        rem = [s for s in DOMS if s != omit]
        loss_rem = d[[f"{s}_loss" for s in rem]].sum(axis=1)
        opp_rem = d[[f"{s}_base" for s in rem]].sum(axis=1)
        for popname, mask in [("same_primary_rows", pd.Series(True, index=d.index)), ("ge1_remaining_opportunity", opp_rem.ge(1))]:
            ds, Xs = d.loc[mask], X0.loc[mask]
            lr = loss_rem.loc[mask]
            g = {"loss_1": lr.eq(1), "loss_2": lr.eq(2)}
            res, fit, kept = fit_groups(ds, Xs, g, ["loss_1", "loss_2"], pilot, count_override=opp_rem.loc[mask])
            rows += tag(res, analysis="e3_leave_one_out", population=popname, omitted=omit, outcome=outcome)
    # the primary three-domain count, for the side-by-side
    g = {"loss_1": d["loss_n"].eq(1), "loss_2": d["loss_n"].ge(2)}
    res, fit, kept = fit_groups(d, X0, g, ["loss_1", "loss_2"], pilot)
    rows += tag(res, analysis="e3_leave_one_out", population="same_primary_rows", omitted="none_primary_count", outcome=outcome)

    # ---------------- E4: four-group model (reference no loss / no gain)
    L, G = d["loss_n"].ge(1), d["gain_n"].ge(1)
    g = {"gain_only": ~L & G, "loss_only": L & ~G, "loss_and_gain": L & G}
    res, fit, kept = fit_groups(d, X0, g, list(g), pilot)
    rows += tag(res, analysis="e4_four_group", population="all_primary", outcome=outcome)
    base_info = dict(analysis="e4_four_group", population="all_primary", outcome=outcome, cohort=a.cohort,
                     n_population=int(len(d)), events_population=int(y.sum()))
    rows += contrast_rows(fit, kept, [("contrast_loss_and_gain_vs_loss_only", "loss_and_gain", "loss_only")], base_info)

    # ---------------- E4: with vs without a simultaneous gain within loss strata
    for lname, lmask, attain in [("loss_1", d["loss_n"].eq(1), d["opp"].isin([1, 2])),
                                 ("loss_2plus", d["loss_n"].ge(2), d["opp"].eq(2))]:
        for popname, m in [("stratum_all_initial_counts", lmask), ("stratum_gain_attainable_initial_counts", lmask & attain)]:
            ds, Xs = d.loc[m], X0.loc[m]
            g = {"gain": ds["gain_n"].ge(1)}
            res, fit, kept = fit_groups(ds, Xs, g, ["gain"], pilot)
            rows += tag(res, analysis="e4_gain_within_loss_stratum", population=popname, stratum=lname, outcome=outcome)

out = pd.DataFrame(rows)
write_csv(OUT / f"fit_{a.cohort}.csv", out)

manifest["n_rows"] = int(len(out))

print(json.dumps({"status": "PASS", "cohort": a.cohort, "rows": len(out)}))
