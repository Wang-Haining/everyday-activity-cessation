#!/usr/bin/env python3
"""E5: added value of the engagement-trajectory question over clinician baselines (per cohort; aggregates only)."""
import argparse, hashlib, json, time
import numpy as np, pandas as pd
from e5_common import *  # noqa

K, REPEATS, NBOOT = 5, 5, 500
FOLD_SEEDS = [101, 102, 103, 104, 105]
BOOT_SEED = 20261003


def run_outcome(cohort, outcome, d, A):
    m = analysis_mask(d, outcome)
    s = d.loc[m]
    designs_df = make_designs(s, A.loc[m], outcome)
    designs = {k: v.to_numpy(float) for k, v in designs_df.items()}
    comps = comparisons(outcome)
    y = s["y"].to_numpy(float)
    pc = pd.factorize(s["person_id"])[0]
    P = int(pc.max()) + 1
    thr = THRESH[outcome]
    models = sorted(designs)
    preds, preps = [], []
    t_start = time.time()
    for r in range(REPEATS):
        folds = make_folds(pc, K, FOLD_SEEDS[r])
        pr = oof_predict(designs, y, folds, K)
        preds.append(pr)
        preps.append({mm: Prep(pr[mm], y) for mm in models})
    print(f"{cohort} {outcome}: n={len(y)} OOF done in {time.time()-t_start:.0f}s", flush=True)
    prev = prevalence(y, None)

    def metrics(r, mm, w):
        p = preds[r][mm]
        b = brier(p, y, w)
        pv = prevalence(y, w)
        return preps[r][mm].auc(w), b, scaled_brier(b, pv), [net_benefit(p, y, w, t) for t in thr]

    point = {(r, mm): metrics(r, mm, None) for r in range(REPEATS) for mm in models}
    pm = lambda mm, i: float(np.mean([point[(r, mm)][i] for r in range(REPEATS)]))
    rng = np.random.default_rng(BOOT_SEED)
    names = list(comps)
    nm = len(models)
    mi = {mm: i for i, mm in enumerate(models)}
    bC = np.zeros((NBOOT, nm))
    bdC = np.zeros((NBOOT, len(names))); bdB = np.zeros_like(bdC); bdS = np.zeros_like(bdC)
    bdN = np.zeros((NBOOT, len(names), len(thr)))
    t_start = time.time()
    for b in range(NBOOT):
        w = np.bincount(rng.integers(0, P, P), minlength=P)[pc].astype(float)
        cache = {(r, mm): metrics(r, mm, w) for r in range(REPEATS) for mm in models}
        for mm in models:
            bC[b, mi[mm]] = np.mean([cache[(r, mm)][0] for r in range(REPEATS)])
        for j, c in enumerate(names):
            base, aug = comps[c][0], comps[c][1]
            bdC[b, j] = np.mean([cache[(r, aug)][0] - cache[(r, base)][0] for r in range(REPEATS)])
            bdB[b, j] = np.mean([cache[(r, aug)][1] - cache[(r, base)][1] for r in range(REPEATS)])
            bdS[b, j] = np.mean([cache[(r, aug)][2] - cache[(r, base)][2] for r in range(REPEATS)])
            for k in range(len(thr)):
                bdN[b, j, k] = np.mean([cache[(r, aug)][3][k] - cache[(r, base)][3][k] for r in range(REPEATS)])
    print(f"{cohort} {outcome}: bootstrap done in {time.time()-t_start:.0f}s", flush=True)

    cal = {mm: np.mean([calibration(preds[r][mm], y) for r in range(REPEATS)], axis=0) for mm in models}
    base_info = dict(cohort=cohort, outcome=outcome, n=int(len(y)), people=P, events=int(y.sum()), prevalence=prev)
    mrows = []
    for mm in models:
        j = mi[mm]
        row = dict(**base_info, model=mm, n_params=int(designs[mm].shape[1]),
                   C=pm(mm, 0), C_se=float(bC[:, j].std(ddof=1)), C_lo=float(np.percentile(bC[:, j], 2.5)),
                   C_hi=float(np.percentile(bC[:, j], 97.5)), brier=pm(mm, 1), scaled_brier=pm(mm, 2),
                   cal_intercept=float(cal[mm][0]), cal_slope=float(cal[mm][1]), cv_folds=K, cv_repeats=REPEATS)
        for k, t in enumerate(thr, start=1):
            row.update({f"thr{k}": t, f"nb_{k}": float(np.mean([point[(r, mm)][3][k - 1] for r in range(REPEATS)])),
                        f"nb_all_{k}": nb_all(prev, t)})
        mrows.append(row)
    crows = []
    for j, c in enumerate(names):
        base, aug, blabel, alabel = comps[c]
        dC = pm(aug, 0) - pm(base, 0)
        row = dict(**base_info, comparison=c, baseline=blabel, addition=alabel, base_model=base, aug_model=aug,
                   n_params_base=int(designs[base].shape[1]), n_params_aug=int(designs[aug].shape[1]),
                   C_base=pm(base, 0), C_aug=pm(aug, 0), dC=dC, dC_se=float(bdC[:, j].std(ddof=1)),
                   dC_lo=float(np.percentile(bdC[:, j], 2.5)), dC_hi=float(np.percentile(bdC[:, j], 97.5)),
                   brier_base=pm(base, 1), brier_aug=pm(aug, 1), dBrier=pm(aug, 1) - pm(base, 1),
                   dBrier_se=float(bdB[:, j].std(ddof=1)), dBrier_lo=float(np.percentile(bdB[:, j], 2.5)),
                   dBrier_hi=float(np.percentile(bdB[:, j], 97.5)),
                   scaled_brier_base=pm(base, 2), scaled_brier_aug=pm(aug, 2), dScaledBrier=pm(aug, 2) - pm(base, 2),
                   dScaledBrier_lo=float(np.percentile(bdS[:, j], 2.5)), dScaledBrier_hi=float(np.percentile(bdS[:, j], 97.5)),
                   cal_intercept_base=float(cal[base][0]), cal_slope_base=float(cal[base][1]),
                   cal_intercept_aug=float(cal[aug][0]), cal_slope_aug=float(cal[aug][1]),
                   n_boot=NBOOT, cv_folds=K, cv_repeats=REPEATS)
        for k, t in enumerate(thr, start=1):
            nb_b = float(np.mean([point[(r, base)][3][k - 1] for r in range(REPEATS)]))
            nb_a = float(np.mean([point[(r, aug)][3][k - 1] for r in range(REPEATS)]))
            row.update({f"thr{k}": t, f"nb_base_{k}": nb_b, f"nb_aug_{k}": nb_a, f"dnb_{k}": nb_a - nb_b,
                        f"dnb_{k}_lo": float(np.percentile(bdN[:, j, k - 1], 2.5)),
                        f"dnb_{k}_hi": float(np.percentile(bdN[:, j, k - 1], 97.5)), f"nb_all_{k}": nb_all(prev, t)})
        crows.append(row)
    return mrows, crows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cohort", required=True); a = ap.parse_args()
    co = load_cohort(a.cohort)
    mrows, crows, ver = [], [], []
    for outcome in OUTCOMES:
        if a.cohort not in PUB[outcome]:
            continue
        d, A, got = build_frame(co, outcome)
        ver.append({"cohort": a.cohort, "outcome": outcome, **got, "verified_against_published": True})
        mr, cr = run_outcome(a.cohort, outcome, d, A)
        mrows += mr; crows += cr
    write_csv(OUT / f"e5_models_{a.cohort}.csv", pd.DataFrame(mrows))
    write_csv(OUT / f"e5_cohort_{a.cohort}.csv", pd.DataFrame(crows))
    write_csv(OUT / f"e5_fit_verify_{a.cohort}.csv", pd.DataFrame(ver))


    print(json.dumps({"status": "PASS", "cohort": a.cohort, "model_rows": len(mrows), "comparison_rows": len(crows)}))
