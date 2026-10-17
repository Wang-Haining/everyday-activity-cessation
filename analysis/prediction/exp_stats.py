#!/usr/bin/env python3
"""Pure numpy/scipy helpers for E1 and E2 (no respondent-level I/O).

Logistic fitting by penalised Newton-Raphson, person-grouped folds, weighted
discrimination/calibration/net-benefit metrics for cluster bootstrap over fixed
out-of-fold predictions, and REML + Hartung-Knapp pooling (the safeguarded
max(1, q) variance factor).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.stats import chi2, t


# --------------------------------------------------------------- logistic fit
def fit_logit(X: np.ndarray, y: np.ndarray, ridge: float = 1e-8, maxit: int = 60, tol: float = 1e-10) -> np.ndarray:
    """Logistic regression; column 0 must be the intercept (unpenalised)."""
    n, p = X.shape
    pen = np.full(p, ridge)
    pen[0] = 0.0
    beta = np.zeros(p)
    pbar = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    beta[0] = math.log(pbar / (1 - pbar))

    def pll(b: np.ndarray) -> float:
        eta = X @ b
        return float(np.sum(y * eta - np.logaddexp(0.0, eta)) - 0.5 * np.sum(pen * b * b))

    cur = pll(beta)
    for _ in range(maxit):
        mu = expit(X @ beta)
        w = np.maximum(mu * (1 - mu), 1e-10)
        grad = X.T @ (y - mu) - pen * beta
        hess = (X * w[:, None]).T @ X + np.diag(pen)
        step = np.linalg.solve(hess, grad)
        scale = 1.0
        for _h in range(30):
            cand = beta + scale * step
            new = pll(cand)
            if new >= cur - 1e-12:
                break
            scale *= 0.5
        beta = cand
        if abs(new - cur) < tol * (1 + abs(cur)) and np.max(np.abs(scale * step)) < 1e-7:
            cur = new
            break
        cur = new
    return beta


def make_folds(person_codes: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Person-grouped folds: permute unique persons, assign in turn."""
    n_persons = int(person_codes.max()) + 1
    order = np.random.default_rng(seed).permutation(n_persons)
    fold_of_person = np.empty(n_persons, dtype=int)
    fold_of_person[order] = np.arange(n_persons) % k
    return fold_of_person[person_codes]


def oof_predict(designs: dict[str, np.ndarray], y: np.ndarray, folds: np.ndarray, k: int) -> dict[str, np.ndarray]:
    out = {name: np.empty(len(y)) for name in designs}
    for fold in range(k):
        train = folds != fold
        test = ~train
        for name, X in designs.items():
            beta = fit_logit(X[train], y[train])
            out[name][test] = expit(X[test] @ beta)
    return out


# --------------------------------------------------------------- metrics
class Prep:
    """Pre-sorted prediction vector for fast weighted AUC."""

    def __init__(self, p: np.ndarray, y: np.ndarray):
        self.p = np.clip(p, 1e-9, 1 - 1e-9)
        self.order = np.argsort(self.p, kind="mergesort")
        ps = self.p[self.order]
        newgrp = np.empty(len(ps), dtype=bool)
        newgrp[0] = True
        newgrp[1:] = ps[1:] != ps[:-1]
        self.grp = np.cumsum(newgrp) - 1
        self.G = int(self.grp[-1]) + 1
        self.ys = y[self.order]

    def auc(self, w: np.ndarray | None) -> float:
        ws = np.ones(len(self.ys)) if w is None else w[self.order]
        pos = np.bincount(self.grp, weights=ws * self.ys, minlength=self.G)
        neg = np.bincount(self.grp, weights=ws * (1.0 - self.ys), minlength=self.G)
        wp, wn = pos.sum(), neg.sum()
        if wp <= 0 or wn <= 0:
            return float("nan")
        before = np.cumsum(neg) - neg
        return float(np.sum(pos * (before + 0.5 * neg)) / (wp * wn))


def brier(p: np.ndarray, y: np.ndarray, w: np.ndarray | None) -> float:
    if w is None:
        return float(np.mean((p - y) ** 2))
    return float(np.sum(w * (p - y) ** 2) / w.sum())


def prevalence(y: np.ndarray, w: np.ndarray | None) -> float:
    return float(y.mean()) if w is None else float(np.sum(w * y) / w.sum())


def scaled_brier(b: float, prev: float) -> float:
    return 1.0 - b / (prev * (1 - prev))


def net_benefit(p: np.ndarray, y: np.ndarray, w: np.ndarray | None, pt: float) -> float:
    sel = p >= pt
    if w is None:
        n = len(y)
        tp = float(np.sum(y[sel]))
        fp = float(np.sum(1 - y[sel]))
    else:
        n = float(w.sum())
        tp = float(np.sum((w * y)[sel]))
        fp = float(np.sum((w * (1 - y))[sel]))
    return (tp - fp * pt / (1 - pt)) / n


def nb_all(prev: float, pt: float) -> float:
    return prev - (1 - prev) * pt / (1 - pt)


def calibration(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(intercept with slope fixed at 1, slope) from out-of-fold predictions."""
    p = np.clip(p, 1e-9, 1 - 1e-9)
    lp = logit(p)
    # slope: y ~ a + b*lp
    X = np.column_stack([np.ones(len(lp)), lp])
    b = fit_logit(X, y, ridge=0.0)
    slope = float(b[1])
    # intercept: y ~ offset(lp), single-parameter Newton
    a = 0.0
    for _ in range(50):
        mu = expit(a + lp)
        g = float(np.sum(y - mu))
        h = float(np.sum(mu * (1 - mu)))
        step = g / h
        a += step
        if abs(step) < 1e-10:
            break
    return float(a), slope


# --------------------------------------------------------------- pooling
def reml_hk(yi: np.ndarray, vi: np.ndarray) -> dict:
    """Verbatim from the Bug-b patched pipeline (hk_patched)."""
    k = len(yi)
    if k < 3 or np.any(~np.isfinite(yi)) or np.any(~np.isfinite(vi)) or np.any(vi <= 0):
        raise ValueError("invalid meta-analysis inputs")

    def objective(tau2: float) -> float:
        weights = 1.0 / (vi + tau2)
        mu = float(np.sum(weights * yi) / np.sum(weights))
        return 0.5 * (
            float(np.sum(np.log(vi + tau2)))
            + math.log(float(np.sum(weights)))
            + float(np.sum(weights * (yi - mu) ** 2))
        )

    upper = max(1.0, float(np.var(yi, ddof=1) * 20), float(np.max(vi) * 20))
    optimum = minimize_scalar(objective, bounds=(0.0, upper), method="bounded", options={"xatol": 1e-12})
    tau2 = max(0.0, float(optimum.x))
    weights = 1.0 / (vi + tau2)
    mu = float(np.sum(weights * yi) / np.sum(weights))
    residual_q = float(np.sum(weights * (yi - mu) ** 2))
    hk_scale = residual_q / (k - 1)
    se = math.sqrt(max(0.0, hk_scale / float(np.sum(weights))))
    se_safeguarded = math.sqrt(max(1.0, hk_scale) / float(np.sum(weights)))
    critical = float(t.ppf(0.975, df=k - 1))
    ci_low = mu - critical * se
    ci_high = mu + critical * se
    pi_critical = float(t.ppf(0.975, df=max(1, k - 2)))
    pi_se = math.sqrt(max(0.0, tau2 + se**2))
    pi_low = mu - pi_critical * pi_se
    pi_high = mu + pi_critical * pi_se
    fixed_weights = 1.0 / vi
    fixed_mu = float(np.sum(fixed_weights * yi) / np.sum(fixed_weights))
    cochran_q = float(np.sum(fixed_weights * (yi - fixed_mu) ** 2))
    i2 = max(0.0, (cochran_q - (k - 1)) / cochran_q) if cochran_q > 0 else 0.0
    return {
        "pooled": mu,
        "se_hk": se,
        "se_hk_safeguarded": se_safeguarded,
        "hk_scale_safeguarded": max(1.0, hk_scale),
        "ci_low_safeguarded": mu - critical * se_safeguarded,
        "ci_high_safeguarded": mu + critical * se_safeguarded,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "prediction_low": pi_low,
        "prediction_high": pi_high,
        "prediction_low_safeguarded": mu - pi_critical * math.sqrt(tau2 + se_safeguarded**2),
        "prediction_high_safeguarded": mu + pi_critical * math.sqrt(tau2 + se_safeguarded**2),
        "tau2": tau2,
        "i2": i2,
        "cochran_q": cochran_q,
        "heterogeneity_p": float(chi2.sf(cochran_q, k - 1)),
        "hk_scale": hk_scale,
    }
