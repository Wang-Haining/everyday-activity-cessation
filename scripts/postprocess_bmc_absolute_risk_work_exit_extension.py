#!/usr/bin/env python3
"""Synthesize aggregate outputs for the frozen BMC manuscript extension."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from probe_behavior_outcome_feasibility import sha256
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, t


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension-config", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _reml_hk(yi: np.ndarray, vi: np.ndarray) -> dict:
    """REML plus Hartung-Knapp; permits descriptive k=2 synthesis."""
    k = len(yi)
    if k < 2 or np.any(~np.isfinite(yi)) or np.any(~np.isfinite(vi)) or np.any(vi <= 0):
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
    optimum = minimize_scalar(
        objective,
        bounds=(0.0, upper),
        method="bounded",
        options={"xatol": 1e-12},
    )
    tau2 = max(0.0, float(optimum.x))
    weights = 1.0 / (vi + tau2)
    mu = float(np.sum(weights * yi) / np.sum(weights))
    residual_q = float(np.sum(weights * (yi - mu) ** 2))
    hk_scale = residual_q / (k - 1)
    se = math.sqrt(max(0.0, hk_scale / float(np.sum(weights))))
    critical = float(t.ppf(0.975, df=k - 1))

    fixed_weights = 1.0 / vi
    fixed_mu = float(np.sum(fixed_weights * yi) / np.sum(fixed_weights))
    q = float(np.sum(fixed_weights * (yi - fixed_mu) ** 2))
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    result = {
        "pooled": mu,
        "se_hk": se,
        "ci_low": mu - critical * se,
        "ci_high": mu + critical * se,
        "tau2": tau2,
        "i2": i2,
        "cochran_q": q,
        "heterogeneity_p": float(chi2.sf(q, k - 1)),
        "hk_scale": hk_scale,
    }
    if k >= 3:
        pi_critical = float(t.ppf(0.975, df=k - 2))
        pi_se = math.sqrt(max(0.0, tau2 + se**2))
        result.update({
            "prediction_low": mu - pi_critical * pi_se,
            "prediction_high": mu + pi_critical * pi_se,
        })
    else:
        result.update({"prediction_low": math.nan, "prediction_high": math.nan})
    return result


def _logit(value: float) -> float:
    return math.log(value / (1.0 - value))


def _inv_logit(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def pooled_risk_rows(risks: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    passed = risks.loc[
        risks["model_status"].eq("PASS") & risks["scenario"].ne("__model__")
    ].copy()
    for (outcome_id, scenario), group in passed.groupby(["outcome_id", "scenario"], sort=True):
        group = group.drop_duplicates("cohort")
        p = group["standardized_risk"].astype(float).to_numpy()
        se = group["standard_error"].astype(float).to_numpy()
        valid = (p > 0) & (p < 1) & (se > 0) & np.isfinite(p) & np.isfinite(se)
        group = group.loc[valid].copy()
        p = p[valid]
        se = se[valid]
        if len(group) < 2:
            continue
        yi = np.array([_logit(value) for value in p])
        sei = se / (p * (1.0 - p))
        fit = _reml_hk(yi, sei**2)
        rows.append({
            "outcome_id": outcome_id,
            "scenario": scenario,
            "k_cohorts": int(len(group)),
            "cohorts": ";".join(sorted(group["cohort"].tolist())),
            "pooled_standardized_risk": _inv_logit(fit["pooled"]),
            "ci_low": _inv_logit(fit["ci_low"]),
            "ci_high": _inv_logit(fit["ci_high"]),
            "prediction_low": _inv_logit(fit["prediction_low"]) if np.isfinite(fit["prediction_low"]) else math.nan,
            "prediction_high": _inv_logit(fit["prediction_high"]) if np.isfinite(fit["prediction_high"]) else math.nan,
            "i2": fit["i2"],
            "tau2_logit_scale": fit["tau2"],
        })
    return rows


def pooled_contrast_rows(contrasts: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for outcome_id, group in contrasts.loc[contrasts["model_status"].eq("PASS")].groupby("outcome_id"):
        group = group.drop_duplicates("cohort")
        if len(group) < 2:
            rows.append({
                "outcome_id": outcome_id,
                "model_status": "NOT_EVALUABLE_META_K1",
                "k_cohorts": int(len(group)),
                "cohorts": ";".join(sorted(group["cohort"].tolist())),
            })
            continue
        yi = np.log(group["estimate"].astype(float).to_numpy())
        sei = group["standard_error_log_scale"].astype(float).to_numpy()
        fit = _reml_hk(yi, sei**2)
        rows.append({
            "outcome_id": outcome_id,
            "model_status": "PASS_DESCRIPTIVE_K2" if len(group) == 2 else "PASS",
            "k_cohorts": int(len(group)),
            "cohorts": ";".join(sorted(group["cohort"].tolist())),
            "pooled_ratio_of_risk_ratios": math.exp(fit["pooled"]),
            "ci_low": math.exp(fit["ci_low"]),
            "ci_high": math.exp(fit["ci_high"]),
            "prediction_low": math.exp(fit["prediction_low"]) if np.isfinite(fit["prediction_low"]) else math.nan,
            "prediction_high": math.exp(fit["prediction_high"]) if np.isfinite(fit["prediction_high"]) else math.nan,
            "i2": fit["i2"],
            "tau2_log_scale": fit["tau2"],
        })
    return rows


def main() -> None:
    args = arguments()
    config = json.loads(args.extension_config.read_text())
    risk_paths = sorted(args.model_dir.glob("*-standardized-risks.csv"))
    contrast_paths = sorted(args.model_dir.glob("*-work-exit-contrasts.csv"))
    interval_paths = sorted(args.model_dir.glob("*-interval-provenance.csv"))
    manifests = sorted(args.model_dir.glob("*-manifest.json"))
    expected = len(config["candidate_cohorts"])
    if not all(len(paths) == expected for paths in [risk_paths, contrast_paths, interval_paths, manifests]):
        raise RuntimeError("incomplete cohort outputs")

    risks = pd.concat([pd.read_csv(path, low_memory=False) for path in risk_paths], ignore_index=True)
    contrasts = pd.concat([pd.read_csv(path, low_memory=False) for path in contrast_paths], ignore_index=True)
    intervals = pd.concat([pd.read_csv(path, low_memory=False) for path in interval_paths], ignore_index=True)
    forbidden = {"person_id", "respondent_id", "prediction", "residual"}
    for frame in [risks, contrasts, intervals]:
        if forbidden.intersection(frame.columns):
            raise RuntimeError("row-level field in aggregate output")

    pooled_risks = pd.DataFrame(pooled_risk_rows(risks))
    pooled_contrasts = pd.DataFrame(pooled_contrast_rows(contrasts))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(args.output_dir, 0o700)
    outputs = {
        "cohort_standardized_risks.csv": risks,
        "pooled_standardized_risks.csv": pooled_risks,
        "cohort_work_exit_contrasts.csv": contrasts,
        "pooled_work_exit_contrasts.csv": pooled_contrasts,
        "comparable_interval_provenance.csv": intervals,
    }
    hashes = {}
    for name, frame in outputs.items():
        path = args.output_dir / name
        frame.to_csv(path, index=False)
        os.chmod(path, 0o600)
        hashes[name] = sha256(path)

    manifest_data = [json.loads(path.read_text()) for path in manifests]
    checks = {
        "all_cohorts_present": sorted(item["cohort"] for item in manifest_data)
        == sorted(config["candidate_cohorts"]),
        "single_design_commit": len({item["design_commit"] for item in manifest_data}) == 1,
        "aggregate_only": all(item.get("aggregate_only") for item in manifest_data),
        "no_respondent_rows_exported": all(item.get("respondent_rows_exported") == 0 for item in manifest_data),
        "risk_range_valid": bool(
            risks.loc[risks["model_status"].eq("PASS"), "standardized_risk"].astype(float).between(0, 1).all()
        ),
        "scope_frozen": bool(risks["scope"].dropna().eq(config["scope"]).all() and contrasts["scope"].dropna().eq(config["scope"]).all()),
        "adjustment_frozen": bool(risks["adjustment"].dropna().eq(config["adjustment"]).all() and contrasts["adjustment"].dropna().eq(config["adjustment"]).all()),
    }
    validation = {
        "analysis_id": config["analysis_id"],
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "model_status_counts_standardized_risk": risks["model_status"].value_counts(dropna=False).to_dict(),
        "model_status_counts_work_exit": contrasts["model_status"].value_counts(dropna=False).to_dict(),
        "output_hashes": hashes,
        "input_hashes": {path.name: sha256(path) for path in risk_paths + contrast_paths + interval_paths},
    }
    validation_path = args.output_dir / "independent-validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, default=str) + "\n")
    os.chmod(validation_path, 0o600)
    if validation["status"] != "PASS":
        raise RuntimeError("validation failed")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
