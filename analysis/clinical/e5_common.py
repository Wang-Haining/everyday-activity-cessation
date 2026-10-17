#!/usr/bin/env python3
"""Shared building blocks for E5 (added value of the engagement-trajectory question).

Uses the shared harmonized-data loader. Respondent-level frames live only in memory; only aggregates are written.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "prediction"))
from exp_common import (  # noqa: E402
    PUB, MORT_COHORTS, ADL_COHORTS, assert_published, load_cohort, primary_set,
    srh, adl_count, state_t1,
)
from exp_stats import (  # noqa: E402
    Prep, brier, calibration, fit_logit, make_folds, net_benefit, nb_all, oof_predict, prevalence,
    reml_hk, scaled_brier,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "clinical" / "cohort"
FINAL = OUT.parent
DOMS = ["alcohol", "activity", "work"]
OUTCOMES = ["mortality", "incident_any_adl"]
THRESH = {"mortality": [0.02, 0.05, 0.10], "incident_any_adl": [0.05, 0.10, 0.20]}
AGE_KNOTS = [60, 70, 80, 90]


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not ({"person_id", "id", "respondent_id", "prediction"} & set(frame.columns))
    frame.to_csv(path, index=False, lineterminator="\n")
    print(f"{path.name}: {len(frame)} aggregate rows")


def build_frame(co, outcome: str):
    """Primary risk set (verified against the published n/events/people) with the E5 variables.

    Returns (frame d, age/sex covariates A, published-check dict). All columns are aligned to d.index.
    """
    data, y, X, terms = primary_set(co, outcome)
    got = assert_published(co, outcome, data, y)
    idx = data.index
    b = co.behavior.loc[idx]
    d = pd.DataFrame(index=idx)
    d["person_id"] = data["person_id"].astype(str)
    d["t1_wave"] = data["t1_wave"].astype(int)
    d["y"] = y.astype(float)
    for s in DOMS:
        d[f"{s}_t0"] = b[f"{s}_baseline"].astype(int)
        loss = b[f"{s}_loss"].astype(int)
        gain = b[f"{s}_gain"].astype(int)
        d[f"{s}_loss"], d[f"{s}_gain"] = loss, gain
        t1 = d[f"{s}_t0"] - loss + gain
        assert t1.isin([0, 1]).all()
        assert (loss <= d[f"{s}_t0"]).all() and (gain <= 1 - d[f"{s}_t0"]).all()
        assert (t1 == state_t1(co, s).loc[idx]).all()
        d[f"{s}_t1"] = t1
    d["opp"] = d[[f"{s}_t0" for s in DOMS]].sum(axis=1)
    d["loss_n"] = d[[f"{s}_loss" for s in DOMS]].sum(axis=1)
    d["gain_n"] = d[[f"{s}_gain" for s in DOMS]].sum(axis=1)
    assert (d["opp"] == data["baseline_engagement_count"]).all() and (d["opp"] >= 1).all()
    assert ((d["loss_n"] == 1) == data["loss_1"].eq(1)).all() and ((d["loss_n"] >= 2) == data["loss_2plus"].eq(1)).all()
    d["loss_1"] = (d["loss_n"] == 1).astype(float)
    d["loss_2plus"] = (d["loss_n"] >= 2).astype(float)
    d["gain_any"] = (d["gain_n"] >= 1).astype(float)
    d["srh"] = srh(co, "t1").loc[idx]
    d["adl_n"] = adl_count(co, "t1").loc[idx]
    X0 = X.drop(columns=terms)
    age_cols = [c for c in X0.columns if c.startswith("age_")]
    assert len(age_cols) == 3 and "female_code2" in X0.columns
    A = X0[age_cols + ["female_code2"]].astype(float)
    A["age_years"] = data["age"].astype(float)
    return d, A, got


def analysis_mask(d: pd.DataFrame, outcome: str) -> pd.Series:
    """One analysis sample per outcome: valid self-rated health (and ADL count for death)."""
    m = d["srh"].notna()
    if outcome == "mortality":
        m &= d["adl_n"].notna()
    return m


def make_designs(d: pd.DataFrame, A: pd.DataFrame, outcome: str) -> dict[str, np.ndarray]:
    """Design matrices (analysis-sample rows only) for every baseline and addition."""
    ones = pd.DataFrame({"intercept": 1.0}, index=d.index)
    wave = pd.get_dummies(d["t1_wave"].astype(str), prefix="t1_wave", drop_first=True, dtype=float)
    B0 = pd.concat([ones, A.drop(columns=["age_years"]), wave], axis=1)
    srh_d = pd.DataFrame({f"srh_{k}": (d["srh"] == k).astype(float) for k in [2, 3, 4, 5]})
    adl_d = pd.DataFrame({f"adl_{k}": ((d["adl_n"] == k) if k < 3 else (d["adl_n"] >= 3)).astype(float) for k in [1, 2, 3]})
    bases = {"B0": B0, "B1": pd.concat([B0, srh_d], axis=1)}
    if outcome == "mortality":
        bases["B2"] = pd.concat([B0, srh_d, adl_d], axis=1)
    cur = d[[f"{s}_t1" for s in DOMS]].astype(float)
    prev = d[[f"{s}_t0" for s in DOMS]].astype(float)
    adds = {
        "i": d[["loss_1", "loss_2plus"]],
        "ii": d[["loss_1", "loss_2plus", "gain_any"]],
        "iii": cur,
        "iv": pd.concat([cur, prev], axis=1),
    }
    out = {}
    for bname, B in bases.items():
        out[bname] = B
        for a, Z in adds.items():
            out[f"{bname}_{a}"] = pd.concat([B, Z], axis=1)
    pruned = {}
    for k, v in out.items():
        keep = [c for c in v.columns if c == "intercept" or v[c].std() > 0]
        assert keep[0] == "intercept"
        pruned[k] = v[keep]
    return pruned


def comparisons(outcome: str) -> dict[str, tuple[str, str, str, str]]:
    """name -> (base design, augmented design, baseline label, addition label)."""
    comps = {}
    for bname in (["B0", "B1", "B2"] if outcome == "mortality" else ["B0", "B1"]):
        for a in ["i", "ii", "iii", "iv"]:
            comps[f"{bname}+{a}_vs_{bname}"] = (bname, f"{bname}_{a}", bname, a)
        comps[f"{bname}_iv_vs_iii"] = (f"{bname}_iii", f"{bname}_iv", bname, "iv_vs_iii")
    return comps


def fit_logit_w(X: np.ndarray, y: np.ndarray, w: np.ndarray, ridge: float = 1e-8, maxit: int = 60, beta0=None) -> np.ndarray:
    """Weighted logistic regression (bootstrap multiplicities as weights); column 0 is the intercept."""
    n, p = X.shape
    pen = np.full(p, ridge)
    pen[0] = 0.0
    pbar = float(np.clip(np.sum(w * y) / np.sum(w), 1e-6, 1 - 1e-6))
    beta = np.zeros(p)
    beta[0] = np.log(pbar / (1 - pbar))
    if beta0 is not None:
        beta = np.array(beta0, dtype=float)

    def pll(b):
        eta = X @ b
        return float(np.sum(w * (y * eta - np.logaddexp(0.0, eta))) - 0.5 * np.sum(pen * b * b))

    cur = pll(beta)
    for _ in range(maxit):
        mu = expit(X @ beta)
        ww = w * np.maximum(mu * (1 - mu), 1e-10)
        grad = X.T @ (w * (y - mu)) - pen * beta
        hess = (X * ww[:, None]).T @ X + np.diag(pen)
        step = np.linalg.solve(hess, grad)
        scale = 1.0
        for _h in range(30):
            cand = beta + scale * step
            new = pll(cand)
            if new >= cur - 1e-12:
                break
            scale *= 0.5
        beta = cand
        if abs(new - cur) < 1e-10 * (1 + abs(cur)) and np.max(np.abs(scale * step)) < 1e-7:
            break
        cur = new
    return beta
