#!/usr/bin/env python3
"""E1: incremental prognostic value of the cessation count (per cohort, aggregate output only)."""
import argparse, hashlib, json, time
import numpy as np, pandas as pd
from exp_common import *  # noqa
from exp_stats import *   # noqa

THRESH = {"mortality": [0.02, 0.05, 0.10], "incident_any_adl": [0.05, 0.10, 0.20]}
K, REPEATS, NBOOT = 5, 5, 500
FOLD_SEEDS = [101, 102, 103, 104, 105]
BOOT_SEED = 20261003
COMPS_FULL = {"M0": ("M0", "M0c"), "M1": ("M1", "M1c")}
COMPS_S2 = {"M0": ("M0", "M0c"), "M1": ("M1", "M1c"), "M2": ("M2", "M2c"), "SRH_vs_M1": ("M1", "M1s")}


def prune(designs: dict[str, pd.DataFrame], reference: str) -> dict[str, np.ndarray]:
    keep = [c for c in designs[reference].columns if c == "intercept" or designs[reference][c].std() > 0]
    out = {}
    for name, d in designs.items():
        cols = [c for c in keep if c in d.columns]
        out[name] = d[cols].to_numpy(float)
        assert cols[0] == "intercept"
    return out


def build(co, outcome):
    data, y, X, terms = primary_set(co, outcome)
    got = assert_published(co, outcome, data, y)
    idx = data.index
    cess, X0 = X[terms], X.drop(columns=terms)
    st = pd.DataFrame({f"{s}_t1": state_t1(co, s).loc[idx] for s in ["alcohol", "activity", "work"]})
    assert st.notna().all().all()
    s1 = srh(co, "t1").loc[idx]
    cnt = (adl_count(co, "t1") if outcome == "mortality" else iadl_count(co, "t1")).loc[idx]
    srh_d = pd.DataFrame({f"srh_{k}": (s1 == k).astype(float).where(s1.notna()) for k in [2, 3, 4, 5]})
    cnt_d = pd.DataFrame({f"cnt_{k}": ((cnt == k) if k < 3 else (cnt >= 3)).astype(float).where(cnt.notna()) for k in [1, 2, 3]})
    ycol = y.astype(float)
    pid = data["person_id"].astype(str)
    full = {"M0": X0, "M0c": pd.concat([X0, cess], axis=1), "M1": pd.concat([X0, st], axis=1),
            "M1c": pd.concat([X0, st, cess], axis=1)}
    s2 = (s1.notna() & cnt.notna()).to_numpy()
    all_d = {"M0": X0, "M0c": pd.concat([X0, cess], axis=1), "M1": pd.concat([X0, st], axis=1),
             "M1c": pd.concat([X0, st, cess], axis=1), "M1s": pd.concat([X0, st, srh_d], axis=1),
             "M2": pd.concat([X0, st, srh_d, cnt_d], axis=1), "M2c": pd.concat([X0, st, srh_d, cnt_d, cess], axis=1)}
    s2d = {k: v.loc[s2] for k, v in all_d.items()}
    return {
        "full": dict(y=ycol.to_numpy(float), pid=pid.to_numpy(), designs=prune(full, "M1c"), comps=COMPS_FULL),
        "S2": dict(y=ycol.loc[s2].to_numpy(float), pid=pid.loc[s2].to_numpy(), designs=prune(s2d, "M2c"), comps=COMPS_S2),
    }, got


def run_sample(cohort, outcome, sample, spec):
    y, designs, comps = spec["y"], spec["designs"], spec["comps"]
    pc = pd.factorize(spec["pid"])[0]
    P = int(pc.max()) + 1
    thr = THRESH[outcome]
    needed = sorted({m for pair in comps.values() for m in pair})
    use = {m: designs[m] for m in needed}
    preds, preps = [], []
    for r in range(REPEATS):
        folds = make_folds(pc, K, FOLD_SEEDS[r])
        pr = oof_predict(use, y, folds, K)
        preds.append(pr)
        preps.append({m: Prep(pr[m], y) for m in needed})
    prev = prevalence(y, None)

    def metrics(r, m, w):
        p = preds[r][m]
        b = brier(p, y, w)
        pv = prevalence(y, w)
        return preps[r][m].auc(w), b, scaled_brier(b, pv), [net_benefit(p, y, w, t) for t in thr]

    # point estimates (mean over repeats)
    point = {}
    for r in range(REPEATS):
        for m in needed:
            point[(r, m)] = metrics(r, m, None)
    def avg_over_r(f):
        return float(np.mean([f(r) for r in range(REPEATS)]))
    # bootstrap over respondents on fixed OOF predictions
    rng = np.random.default_rng(BOOT_SEED)
    names = list(comps)
    bdC = np.zeros((NBOOT, len(names))); bdB = np.zeros_like(bdC); bdN = np.zeros((NBOOT, len(names), len(thr)))
    t0 = time.time()
    for b in range(NBOOT):
        w = np.bincount(rng.integers(0, P, P), minlength=P)[pc].astype(float)
        cache = {(r, m): metrics(r, m, w) for r in range(REPEATS) for m in needed}
        for j, c in enumerate(names):
            base, aug = comps[c]
            bdC[b, j] = np.mean([cache[(r, aug)][0] - cache[(r, base)][0] for r in range(REPEATS)])
            bdB[b, j] = np.mean([cache[(r, aug)][1] - cache[(r, base)][1] for r in range(REPEATS)])
            for k in range(len(thr)):
                bdN[b, j, k] = np.mean([cache[(r, aug)][3][k] - cache[(r, base)][3][k] for r in range(REPEATS)])
    rows = []
    for j, c in enumerate(names):
        base, aug = comps[c]
        pm = lambda m, i: float(np.mean([point[(r, m)][i] for r in range(REPEATS)]))
        cal = {m: np.mean([calibration(preds[r][m], y) for r in range(REPEATS)], axis=0) for m in (base, aug)}
        dC = pm(aug, 0) - pm(base, 0)
        row = dict(cohort=cohort, outcome=outcome, sample=sample, comparison=c, base_model=base, aug_model=aug,
                   n=int(len(y)), people=P, events=int(y.sum()), prevalence=prev,
                   n_params_base=int(designs[base].shape[1]), n_params_aug=int(designs[aug].shape[1]),
                   C_base=pm(base, 0), C_aug=pm(aug, 0), dC=dC, dC_se=float(bdC[:, j].std(ddof=1)),
                   dC_lo=float(np.percentile(bdC[:, j], 2.5)), dC_hi=float(np.percentile(bdC[:, j], 97.5)),
                   brier_base=pm(base, 1), brier_aug=pm(aug, 1), dBrier=pm(aug, 1) - pm(base, 1),
                   dBrier_lo=float(np.percentile(bdB[:, j], 2.5)), dBrier_hi=float(np.percentile(bdB[:, j], 97.5)),
                   scaled_brier_base=pm(base, 2), scaled_brier_aug=pm(aug, 2),
                   cal_intercept_base=float(cal[base][0]), cal_slope_base=float(cal[base][1]),
                   cal_intercept_aug=float(cal[aug][0]), cal_slope_aug=float(cal[aug][1]),
                   n_boot=NBOOT, cv_folds=K, cv_repeats=REPEATS)
        for k, t in enumerate(thr, start=1):
            nb_b = float(np.mean([point[(r, base)][3][k - 1] for r in range(REPEATS)]))
            nb_a = float(np.mean([point[(r, aug)][3][k - 1] for r in range(REPEATS)]))
            row.update({f"thr{k}": t, f"nb_base_{k}": nb_b, f"nb_aug_{k}": nb_a, f"dnb_{k}": nb_a - nb_b,
                        f"dnb_{k}_lo": float(np.percentile(bdN[:, j, k - 1], 2.5)),
                        f"dnb_{k}_hi": float(np.percentile(bdN[:, j, k - 1], 97.5)),
                        f"nb_all_{k}": nb_all(prev, t)})
        rows.append(row)
    print(f"{cohort} {outcome} {sample}: n={len(y)} done in {time.time()-t0:.0f}s bootstrap", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cohort", required=True); a = ap.parse_args()
    co = load_cohort(a.cohort)
    all_rows, ver = [], []
    for outcome in ["mortality", "incident_any_adl"]:
        if a.cohort not in PUB[outcome]:
            continue
        samples, got = build(co, outcome)
        ver.append({"cohort": a.cohort, "outcome": outcome, **got, "verified_against_published": True})
        for sample, spec in samples.items():
            all_rows += run_sample(a.cohort, outcome, sample, spec)
    write_csv(OUT / f"e1_cohort_{a.cohort}.csv", pd.DataFrame(all_rows))
    write_csv(OUT / f"e1_verify_{a.cohort}.csv", pd.DataFrame(ver))


    print(json.dumps({"status": "PASS", "cohort": a.cohort, "rows": len(all_rows)}))
