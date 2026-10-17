#!/usr/bin/env python3
"""E5 hidden-risk analysis: self-rated health at t1 x cessation group (and x loss/gain group); absolute 2-year risks.

Per cohort: n, people, events, crude risk, model-based age-sex-standardised risk (primary), direct age-band x sex
standardised risk (secondary). Respondent-cluster bootstrap (logit-scale SE). Aggregates only.
"""
import argparse, hashlib, json, time
import numpy as np, pandas as pd
from scipy.special import expit, logit
from e5_common import *  # noqa

NBOOT, BOOT_SEED = 500, 20261004
CLS = {
    "cess": ["0", "1", "2plus"],
    "traj": ["no_loss", "loss_only", "loss_with_gain"],
}
SRHG = ["good_or_better", "fair_poor"]
CLIP = 1e-6
GATE = 5   # Addendum A: events and non-events needed for a cell to enter pooling (design section 5 said 20)


def prep(d, A):
    h = d.loc[d["srh"].notna()].copy()
    Ah = A.loc[h.index]
    h["srh_grp"] = np.where(h["srh"] <= 3, "good_or_better", "fair_poor")
    h["cess"] = np.select([h["loss_n"] == 0, h["loss_n"] == 1], ["0", "1"], "2plus")
    h["traj"] = np.select([h["loss_n"] == 0, h["gain_n"] == 0], ["no_loss", "loss_only"], "loss_with_gain")
    band = np.select([Ah["age_years"] < 65, Ah["age_years"] < 75, Ah["age_years"] < 85], [0, 1, 2], 3)
    h["stratum"] = band * 2 + Ah["female_code2"].to_numpy(int)
    return h, Ah


def estimate(h, Acov, cls, w):
    """Return dict cell -> (crude, model-standardised, direct, covered weight share, weighted n, weighted events)."""
    cells = [(g, c) for g in SRHG for c in CLS[cls]]
    lab = (h["srh_grp"] + "|" + h[cls]).to_numpy()
    y = h["y"].to_numpy(float)
    cell_ix = {f"{g}|{c}": i for i, (g, c) in enumerate(cells)}
    ci = np.array([cell_ix[v] for v in lab])
    C = np.zeros((len(y), len(cells) - 1))
    for i in range(1, len(cells)):
        C[:, i - 1] = (ci == i)
    cov = np.column_stack([np.ones(len(y)), Acov[["age_linear", "age_nonlinear_1", "age_nonlinear_2", "female_code2"]].to_numpy(float)])
    X = np.column_stack([cov, C])
    beta = fit_logit_w(X, y, w)
    lp0 = cov @ beta[:cov.shape[1]]
    delta = np.concatenate([[0.0], beta[cov.shape[1]:]])
    sw = h["stratum"].to_numpy(int)
    Wtot = w.sum()
    Ws = np.bincount(sw, weights=w, minlength=8) / Wtot
    out = {}
    for i, cell in enumerate(cells):
        sel = ci == i
        wn = float(np.sum(w[sel])); wd = float(np.sum((w * y)[sel]))
        crude = wd / wn if wn > 0 else np.nan
        std = float(np.sum(w * expit(lp0 + delta[i])) / Wtot)
        n_s = np.bincount(sw[sel], weights=w[sel], minlength=8)
        d_s = np.bincount(sw[sel], weights=(w * y)[sel], minlength=8)
        ok = n_s > 0
        covered = float(Ws[ok].sum())
        direct = float(np.sum(Ws[ok] * d_s[ok] / n_s[ok]) / covered) if covered > 0 else np.nan
        out[cell] = (crude, std, direct, covered, wn, wd)
    return cells, out


def lg(x):
    return float(logit(np.clip(x, CLIP, 1 - CLIP)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cohort", required=True); a = ap.parse_args()
    co = load_cohort(a.cohort)
    rows = []
    for outcome in OUTCOMES:
        if a.cohort not in PUB[outcome]:
            continue
        d, A, got = build_frame(co, outcome)
        h, Acov = prep(d, A)
        pid = pd.factorize(h["person_id"])[0]
        P = int(pid.max()) + 1
        n = len(h)
        w1 = np.ones(n)
        for cls in CLS:
            cells, point = estimate(h, Acov, cls, w1)
            rng = np.random.default_rng(BOOT_SEED)
            t0 = time.time()
            boots = {k: np.zeros((NBOOT, 3)) for k in cells}
            for b in range(NBOOT):
                w = np.bincount(rng.integers(0, P, P), minlength=P)[pid].astype(float)
                _, est = estimate(h, Acov, cls, w)
                for k in cells:
                    boots[k][b] = est[k][:3]
            print(f"{a.cohort} {outcome} {cls}: bootstrap {time.time()-t0:.0f}s", flush=True)
            y = h["y"].to_numpy(float)
            lab = (h["srh_grp"] + "|" + h[cls]).to_numpy()
            cls_rows = []
            for k in cells:
                g, c = k
                sel = lab == f"{g}|{c}"
                ev = int(y[sel].sum()); nn = int(sel.sum())
                pe = point[k]
                row = {"row_type": "cohort", "cohort": a.cohort, "outcome": outcome, "classification": cls,
                       "srh_group": g, "cell": c, "n": nn, "people": int(h.loc[sel, "person_id"].nunique()), "events": ev,
                       "non_events": nn - ev, "evaluable": bool(ev >= GATE and (nn - ev) >= GATE),
                       "evaluable_ge20": bool(ev >= 20 and (nn - ev) >= 20)}
                for j, nm in enumerate(["crude", "std_model", "std_direct"]):
                    bj = boots[k][:, j]
                    row[f"risk_{nm}"] = pe[j]
                    row[f"risk_{nm}_lo"] = float(np.nanpercentile(bj, 2.5)); row[f"risk_{nm}_hi"] = float(np.nanpercentile(bj, 97.5))
                    lb = logit(np.clip(bj, CLIP, 1 - CLIP))
                    row[f"logit_{nm}"] = lg(pe[j]); row[f"logit_{nm}_se"] = float(np.nanstd(lb, ddof=1))
                row["direct_covered_weight_share"] = pe[3]
                cls_rows.append(row)
            rows.extend(cls_rows)
            if cls == "cess":
                # Rule 3 contrast: good-or-better with 2+ cessations minus fair/poor with 0 cessations
                hi_cell, lo_cell = ("good_or_better", "2plus"), ("fair_poor", "0")
                ev_ok = all(r["evaluable"] for r in cls_rows if (r["srh_group"], r["cell"]) in {hi_cell, lo_cell})
                crow = {"row_type": "cohort_contrast", "cohort": a.cohort, "outcome": outcome, "classification": cls,
                        "srh_group": "good_or_better|2plus minus fair_poor|0", "cell": "contrast",
                        "evaluable": bool(ev_ok)}
                for j, nm in enumerate(["crude", "std_model", "std_direct"]):
                    bh, bl = boots[hi_cell][:, j], boots[lo_cell][:, j]
                    dl = logit(np.clip(bh, CLIP, 1 - CLIP)) - logit(np.clip(bl, CLIP, 1 - CLIP))
                    rd = bh - bl
                    crow[f"logit_{nm}"] = lg(point[hi_cell][j]) - lg(point[lo_cell][j])
                    crow[f"logit_{nm}_se"] = float(np.nanstd(dl, ddof=1))
                    crow[f"risk_{nm}"] = point[hi_cell][j] - point[lo_cell][j]
                    crow[f"risk_{nm}_lo"] = float(np.nanpercentile(rd, 2.5)); crow[f"risk_{nm}_hi"] = float(np.nanpercentile(rd, 97.5))
                rows.append(crow)
        # share of all events among good-or-better self-rated health with >= 1 cessation
        yv = h["y"].to_numpy(float)
        grp = ((h["srh_grp"] == "good_or_better") & (h["loss_n"] >= 1)).to_numpy()
        ev_all = float(yv.sum()); ev_grp = float(yv[grp].sum())
        rows.append({"row_type": "share_of_events", "cohort": a.cohort, "outcome": outcome, "classification": "cess",
                     "srh_group": "good_or_better", "cell": ">=1 cessation", "n": int(grp.sum()), "people": int(h.loc[grp, "person_id"].nunique()),
                     "events": int(ev_grp), "non_events": int(ev_all - ev_grp), "evaluable": True,
                     "share_of_events": ev_grp / ev_all, "share_of_intervals": float(grp.mean()), "total_events": int(ev_all), "total_n": int(n)})
    write_csv(OUT / f"e5_hidden_{a.cohort}.csv", pd.DataFrame(rows))


    print(json.dumps({"status": "PASS", "cohort": a.cohort, "rows": len(rows)}))
