#!/usr/bin/env python3
"""S1 (Addendum A): one-stage cohort-stratified common-effect modified Poisson for E3.2, E3.3 and E4.
All cohorts are loaded sequentially in this process; only aggregate rows are written."""
import argparse, gc, hashlib, json
import numpy as np, pandas as pd
from e34_common import *   # noqa

ap = argparse.ArgumentParser(); ap.add_argument("--relax-epv", action="store_true"); A = ap.parse_args()
ROW_TYPE = "one_stage_S1b_posthoc" if A.relax_epv else "one_stage_S1"
PAIRS = [("alcohol", "activity"), ("alcohol", "work"), ("activity", "work")]
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


def stacked_fit(plist, fn, focal, label_info, contrasts=(), count_fn=None):
    pilot_use = {**pilot, "events_per_parameter": 0} if A.relax_epv else pilot
    sub = {}
    for c, (d, X0) in plist.items():
        mask, groups = fn(d)
        X0c = X0.copy()
        if count_fn is not None:
            X0c["baseline_engagement_count"] = count_fn(d).astype(float)
        sub[c] = (d.loc[mask], X0c.loc[mask], {k: v.loc[mask] for k, v in groups.items()})
    names = list(next(iter(sub.values()))[2])
    ev = {k: sum(int(s[0]["y"][s[2][k]].sum()) for s in sub.values()) for k in names}
    nn = {k: sum(int(s[2][k].sum()) for s in sub.values()) for k in names}
    ref_n = ref_ev = 0
    for dsub, _, g in sub.values():
        r = ~pd.concat(list(g.values()), axis=1).any(axis=1)
        ref_n += int(r.sum()); ref_ev += int(dsub["y"][r].sum())
    n_pop = sum(len(s[0]) for s in sub.values()); ev_pop = sum(int(s[0]["y"].sum()) for s in sub.values())
    base = {**label_info, "n_population": n_pop, "events_population": ev_pop, "n_reference": ref_n, "events_reference": ref_ev,
            "cohorts_included": ";".join(sub)}
    status = {k: ("ESTIMABLE" if ev[k] >= 20 and (nn[k] - ev[k]) >= 20 else "NOT_EVALUABLE_EVENTS" if ev[k] < 20 else "NOT_EVALUABLE_NONEVENTS") for k in names}
    def nofit(note):
        return [{**base, "term": k, "n_group": nn[k], "events_group": ev[k], "risk_group": ev[k] / nn[k] if nn[k] else np.nan,
                 "status": note if status[k] == "ESTIMABLE" else status[k]} for k in focal]
    if not (ref_ev >= 20 and ref_n - ref_ev >= 20):
        return nofit("NOT_EVALUABLE_REFERENCE")
    kept = [k for k in names if status[k] == "ESTIMABLE"]
    if not any(k in focal for k in kept):
        return nofit("NOT_EVALUABLE_NO_FOCAL")
    Xl, dl, yl, gl = [], [], [], {k: [] for k in kept}
    dropped = 0
    for c, (dsub, Xsub, g) in sub.items():
        bad = pd.concat([g[k] for k in names if k not in kept], axis=1).any(axis=1) if len(kept) < len(names) else pd.Series(False, index=dsub.index)
        keep = dsub.index[~bad.to_numpy()]
        dropped += int(len(dsub) - len(keep))
        ds, Xs = dsub.loc[keep], Xsub.loc[keep]
        Xs = Xs.drop(columns=[x for x in Xs.columns if x.startswith("t1_wave_")])
        wave = pd.get_dummies(ds["t1_wave"].astype(int).astype(str), prefix="t1_wave", drop_first=True, dtype=float)
        Xs = pd.concat([Xs, wave], axis=1)
        Xs = Xs.loc[:, [x == "intercept" or Xs[x].std() > 0 for x in Xs.columns]]
        Xs.columns = [f"{c}__{x}" for x in Xs.columns]
        Xl.append(Xs); dl.append(ds[["person_id", "y"]]);
        for k in kept:
            gl[k].append(g[k].loc[keep].astype(float))
    Xall = pd.concat(Xl, axis=0, ignore_index=True).fillna(0.0)
    dall = pd.concat(dl, axis=0, ignore_index=True)
    for k in kept:
        Xall[k] = pd.concat(gl[k], axis=0, ignore_index=True)
    gate, counts = support_gate(pd.DataFrame({"person_id": dall["person_id"], **{k: Xall[k] for k in kept}}), dall["y"], Xall, kept, pilot_use, True)
    base2 = {**base, "n_model": int(len(dall)), "events_model": int(dall["y"].sum()), "n_parameters": int(Xall.shape[1]),
             "n_dropped_rows_failed_groups": dropped}
    if gate != "ESTIMABLE":
        return [{**base2, "term": k, "n_group": nn[k], "events_group": ev[k], "risk_group": ev[k] / nn[k] if nn[k] else np.nan,
                 "status": gate if status[k] == "ESTIMABLE" else status[k]} for k in focal]
    fit, warn = fit_clustered(dall["y"], Xall, dall["person_id"], True)
    crow = {r["term"]: r for r in coefficient_rows(fit, [k for k in kept if k in focal], True)}
    out = []
    for k in focal:
        r = {**base2, "term": k, "n_group": nn[k], "events_group": ev[k], "risk_group": ev[k] / nn[k] if nn[k] else np.nan}
        if k in crow:
            c_ = crow[k]; r.update({"status": "ESTIMABLE", "rr": c_["estimate"], "se_log_rr": c_["standard_error"], "ci_low": c_["ci_low"],
                                    "ci_high": c_["ci_high"], "p_value": c_["p_value"], "warnings": warn})
        else:
            r["status"] = status[k]
        out.append(r)
    out += contrast_rows(fit, kept, list(contrasts), {**base2})
    return out


rows = []
for outcome in OUTCOMES:
    pl = parts[outcome]
    lab = dict(outcome=outcome, row_type=ROW_TYPE, cohort="ONE_STAGE_ALL_COHORTS")
    P = lambda d: d["pattern"]
    # validation (one-stage count model; informational)
    rows += stacked_fit(pl, lambda d: (pd.Series(True, index=d.index), {"loss_1": d["loss_n"].eq(1), "loss_2plus": d["loss_n"].ge(2)}),
                        ["loss_1", "loss_2plus"], {**lab, "analysis": "validation_count_model", "population": "all_primary"})
    # E3.2
    rows += stacked_fit(pl, lambda d: (pd.Series(True, index=d.index), {f"p{p}": P(d).eq(p) for p in PATTERNS[1:]}),
                        [f"p{p}" for p in PATTERNS[1:]], {**lab, "analysis": "e3_pattern_model", "population": "all_primary"})
    # E3.3
    for d1, d2 in PAIRS:
        third = [s for s in DOMS if s not in (d1, d2)][0]
        pair = f"{LETTER[d1]}{LETTER[d2]}"
        names = [f"pair_{pair}", f"single_{LETTER[d1]}", f"single_{LETTER[d2]}", f"nuisance_{LETTER[third]}_lost"]
        def fn(d, d1=d1, d2=d2, third=third, names=names):
            m = d[f"{d1}_base"].eq(1) & d[f"{d2}_base"].eq(1)
            l1, l2, l3 = d[f"{d1}_loss"].eq(1), d[f"{d2}_loss"].eq(1), d[f"{third}_loss"].eq(1)
            return m, {names[0]: l1 & l2 & ~l3, names[1]: l1 & ~l2 & ~l3, names[2]: ~l1 & l2 & ~l3, names[3]: l3}
        rows += stacked_fit(pl, fn, names[:3], {**lab, "analysis": "e3_restricted_pair", "population": f"both_{LETTER[d1]}_{LETTER[d2]}_present_at_t0", "pair": pair},
                            contrasts=[(f"contrast_{names[0]}_vs_{names[1]}", names[0], names[1]), (f"contrast_{names[0]}_vs_{names[2]}", names[0], names[2])])
    # E4
    def fn4(d):
        L, G = d["loss_n"].ge(1), d["gain_n"].ge(1)
        return pd.Series(True, index=d.index), {"gain_only": ~L & G, "loss_only": L & ~G, "loss_and_gain": L & G}
    rows += stacked_fit(pl, fn4, ["gain_only", "loss_only", "loss_and_gain"], {**lab, "analysis": "e4_four_group", "population": "all_primary"},
                        contrasts=[("contrast_loss_and_gain_vs_loss_only", "loss_and_gain", "loss_only")])
    for lname, lfun, afun in [("loss_1", lambda d: d["loss_n"].eq(1), lambda d: d["opp"].isin([1, 2])),
                              ("loss_2plus", lambda d: d["loss_n"].ge(2), lambda d: d["opp"].eq(2))]:
        for popname, use_att in [("stratum_all_initial_counts", False), ("stratum_gain_attainable_initial_counts", True)]:
            def fns(d, lfun=lfun, afun=afun, use_att=use_att):
                m = lfun(d) & (afun(d) if use_att else True)
                return m, {"gain": d["gain_n"].ge(1)}
            rows += stacked_fit(pl, fns, ["gain"], {**lab, "analysis": "e4_gain_within_loss_stratum", "population": popname, "stratum": lname})
    if A.relax_epv:
        for omit in DOMS:
            rem = [x for x in DOMS if x != omit]
            lr = lambda d, rem=rem: d[[f"{x}_loss" for x in rem]].sum(axis=1)
            orm = lambda d, rem=rem: d[[f"{x}_base" for x in rem]].sum(axis=1)
            for popname, popfn in [("same_primary_rows", lambda d, orm=orm: pd.Series(True, index=d.index)), ("ge1_remaining_opportunity", lambda d, orm=orm: orm(d).ge(1))]:
                def fnl(d, lr=lr, popfn=popfn):
                    return popfn(d), {"loss_1": lr(d).eq(1), "loss_2": lr(d).eq(2)}
                rows += stacked_fit(pl, fnl, ["loss_1", "loss_2"], {**lab, "analysis": "e3_leave_one_out", "population": popname, "omitted": omit}, count_fn=orm)
        rows += stacked_fit(pl, lambda d: (pd.Series(True, index=d.index), {"loss_1": d["loss_n"].eq(1), "loss_2": d["loss_n"].ge(2)}),
                            ["loss_1", "loss_2"], {**lab, "analysis": "e3_leave_one_out", "population": "same_primary_rows", "omitted": "none_primary_count"})

out = pd.DataFrame(rows)
write_csv(OUT / ("onestage_S1b_posthoc.csv" if A.relax_epv else "onestage_S1.csv"), out)


print(json.dumps({"status": "PASS", "rows": len(out)}))
