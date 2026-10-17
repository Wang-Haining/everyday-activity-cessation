#!/usr/bin/env python3
"""REML pooling with Hartung-Knapp and safeguarded uncertainty."""

from __future__ import annotations
import hashlib
import math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, t

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reml_hk(yi: np.ndarray, vi: np.ndarray) -> dict:
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

