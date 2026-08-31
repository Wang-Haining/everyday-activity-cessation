#!/usr/bin/env python3
"""Combine all model shards and produce prespecified cross-cohort summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, t


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--synthesis-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bh_fdr(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().astype(float)
    if valid.empty:
        return result
    ordered = valid.sort_values()
    m = len(ordered)
    adjusted = ordered.to_numpy() * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[ordered.index] = np.minimum(adjusted, 1.0)
    return result


def harmonization_stratum(row: pd.Series, config: dict) -> str:
    cohort = row["cohort"]
    if row["transition_id"] == "social_participation":
        for stratum, cohorts in config["harmonization_strata"]["social_participation"].items():
            if cohort in cohorts:
                return f"social_participation:{stratum}"
    if row["transition_id"] == "alcohol_frequency":
        for stratum, cohorts in config["harmonization_strata"]["alcohol_frequency"].items():
            if cohort in cohorts:
                return f"alcohol_frequency:{stratum}"
    if row["outcome_id"] == "hospitalization_next_interval":
        for stratum, cohorts in config["harmonization_strata"]["hospitalization_next_interval"].items():
            if cohort in cohorts:
                return f"hospitalization:{stratum}"
    if row["outcome_id"] == "bmi_change":
        for stratum, cohorts in config["harmonization_strata"]["bmi_change"].items():
            if cohort in cohorts:
                return f"bmi_change:{stratum}"
    return "common_definition"


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
        "ci_low": ci_low,
        "ci_high": ci_high,
        "prediction_low": pi_low,
        "prediction_high": pi_high,
        "tau2": tau2,
        "i2": i2,
        "cochran_q": cochran_q,
        "heterogeneity_p": float(chi2.sf(cochran_q, k - 1)),
        "hk_scale": hk_scale,
    }


def meta_row(group: pd.DataFrame, keys: dict) -> dict:
    binary = group["estimate_scale"].eq("risk_ratio").all()
    if binary:
        yi = np.log(group["estimate"].astype(float).to_numpy())
        sei = group["standard_error"].astype(float).to_numpy()
        null = 0.0
        scale = "risk_ratio"
    else:
        yi = group["standardized_estimate"].astype(float).to_numpy()
        sei = group["standardized_standard_error"].astype(float).to_numpy()
        null = 0.0
        scale = "standardized_mean_difference"
    fit = reml_hk(yi, sei**2)
    same_direction = float(np.mean(np.sign(yi) == np.sign(fit["pooled"]))) if fit["pooled"] != 0 else float(np.mean(yi == 0))
    pooled_magnitude = math.exp(fit["pooled"]) if binary else fit["pooled"]
    ci_low = math.exp(fit["ci_low"]) if binary else fit["ci_low"]
    ci_high = math.exp(fit["ci_high"]) if binary else fit["ci_high"]
    prediction_low = math.exp(fit["prediction_low"]) if binary else fit["prediction_low"]
    prediction_high = math.exp(fit["prediction_high"]) if binary else fit["prediction_high"]
    ci_excludes_null = fit["ci_low"] > null or fit["ci_high"] < null
    pi_excludes_null = fit["prediction_low"] > null or fit["prediction_high"] < null
    if binary:
        small_magnitude = 0.90 <= pooled_magnitude <= 1.11
    else:
        small_magnitude = abs(pooled_magnitude) < 0.10
    stable_null = (
        not ci_excludes_null
        and not pi_excludes_null
        and same_direction <= 0.75
        and small_magnitude
    )
    if ci_excludes_null and pi_excludes_null and same_direction >= 0.75:
        pattern_class = "REPLICATED_PREDICTION_INTERVAL_SUPPORTED"
    elif ci_excludes_null and same_direction >= 0.75:
        pattern_class = "DIRECTIONALLY_CONSISTENT_PI_UNCERTAIN"
    elif stable_null:
        pattern_class = "STABLE_NULL_PATTERN"
    elif same_direction >= 0.75:
        pattern_class = "DIRECTIONALLY_CONSISTENT_IMPRECISE"
    else:
        pattern_class = "HETEROGENEOUS_OR_INCONSISTENT"
    return {
        **keys,
        "effect_scale": scale,
        "k_cohorts": len(group),
        "cohorts": ";".join(sorted(group["cohort"].tolist())),
        "pooled_estimate": pooled_magnitude,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "prediction_low": prediction_low,
        "prediction_high": prediction_high,
        "tau2_analysis_scale": fit["tau2"],
        "i2": fit["i2"],
        "cochran_q": fit["cochran_q"],
        "heterogeneity_p": fit["heterogeneity_p"],
        "direction_consistency": same_direction,
        "cohort_estimates": ";".join(f"{row.cohort}:{float(row.estimate if binary else row.standardized_estimate):.6g}" for row in group.itertuples()),
        "nominal_p_lt_0_05_n": int(group["p_value"].astype(float).lt(.05).sum()),
        "descriptive_fdr_lt_0_05_n": int(group["bh_fdr"].astype(float).lt(.05).sum()),
        "pattern_class": pattern_class,
        "significance_used_for_retention": False,
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.synthesis_config.read_text(encoding="utf-8"))
    result_paths = sorted(args.model_dir.glob("*-systematic-results.csv"))
    manifest_paths = sorted(args.model_dir.glob("*-model-manifest.json"))
    if not result_paths or not manifest_paths:
        raise RuntimeError("model shards missing")
    frames = [pd.read_csv(path, low_memory=False) for path in result_paths]
    results = pd.concat(frames, ignore_index=True, sort=False)
    key = ["cohort", "scope", "transition_id", "contrast", "outcome_id", "adjustment"]
    if results.duplicated(key).any():
        raise RuntimeError("duplicate result cell/model rows across shards")
    results["bh_fdr"] = np.nan
    pass_mask = results["model_status"].eq("PASS")
    fdr_groups = ["cohort", "scope", "adjustment", "outcome_family"]
    for _, index in results.loc[pass_mask].groupby(fdr_groups, dropna=False).groups.items():
        results.loc[index, "bh_fdr"] = bh_fdr(results.loc[index, "p_value"])
    results["harmonization_stratum"] = results.apply(lambda row: harmonization_stratum(row, config), axis=1)

    primary = results.loc[
        results["model_status"].eq("PASS")
        & results["scope"].eq(config["primary_synthesis_scope"])
        & results["adjustment"].eq(config["primary_adjustment"])
    ].copy()
    meta_keys = ["transition_id", "transition_family", "contrast", "outcome_id", "outcome_family", "outcome_type", "harmonization_stratum"]
    meta_rows = []
    not_evaluable_rows = []
    for values, group in primary.groupby(meta_keys, dropna=False, sort=True):
        keys = dict(zip(meta_keys, values))
        if len(group) < config["minimum_meta_cohorts"]:
            not_evaluable_rows.append({**keys, "k_cohorts": len(group), "cohorts": ";".join(sorted(group["cohort"])), "status": "NOT_EVALUABLE_META_K_LT_3"})
            continue
        try:
            meta_rows.append(meta_row(group, keys))
        except Exception as exc:
            not_evaluable_rows.append({**keys, "k_cohorts": len(group), "cohorts": ";".join(sorted(group["cohort"])), "status": f"META_FAILURE_{type(exc).__name__}: {exc}"})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(args.output_dir, 0o700)
    results_path = args.output_dir / "systematic-results-matrix.csv"
    meta_path = args.output_dir / "cross-cohort-summary.csv"
    not_eval_path = args.output_dir / "cross-cohort-not-evaluable.csv"
    results.sort_values(key, kind="mergesort").to_csv(results_path, index=False)
    pd.DataFrame(meta_rows).sort_values(["pattern_class", "outcome_family", "outcome_id", "transition_id", "contrast"], kind="mergesort").to_csv(meta_path, index=False)
    pd.DataFrame(not_evaluable_rows).to_csv(not_eval_path, index=False)
    for path in (results_path, meta_path, not_eval_path):
        os.chmod(path, 0o600)

    meta = pd.DataFrame(meta_rows)
    class_counts = meta["pattern_class"].value_counts().to_dict() if not meta.empty else {}
    summary = {
        "analysis_id": config["analysis_id"],
        "model_shards": len(result_paths),
        "model_manifests": len(manifest_paths),
        "systematic_result_rows": int(len(results)),
        "pass_model_rows": int(pass_mask.sum()),
        "model_status_counts": results["model_status"].value_counts(dropna=False).to_dict(),
        "primary_full_pass_rows": int(len(primary)),
        "meta_evaluable_rows": len(meta_rows),
        "meta_not_evaluable_rows": len(not_evaluable_rows),
        "pattern_class_counts": class_counts,
        "significance_filtering": False,
        "input_result_hashes": {path.name: sha256(path) for path in result_paths},
        "input_manifest_hashes": {path.name: sha256(path) for path in manifest_paths},
        "systematic_results_sha256": sha256(results_path),
        "cross_cohort_summary_sha256": sha256(meta_path),
        "cross_cohort_not_evaluable_sha256": sha256(not_eval_path),
        "respondent_rows_exported": 0,
    }
    summary_path = args.output_dir / "postprocess-manifest.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.chmod(summary_path, 0o600)


if __name__ == "__main__":
    main()
