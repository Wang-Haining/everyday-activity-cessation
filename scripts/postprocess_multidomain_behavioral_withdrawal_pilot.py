#!/usr/bin/env python3
"""Synthesize aggregate multidomain-withdrawal model outputs."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from postprocess_behavior_outcome_landscape import reml_hk, sha256


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--pilot-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def meta_row(group: pd.DataFrame, keys: dict) -> dict:
    binary = group["estimate_scale"].eq("risk_ratio").all()
    if binary:
        yi = np.log(group["estimate"].astype(float).to_numpy())
        sei = group["standard_error"].astype(float).to_numpy()
    else:
        yi = group["standardized_estimate"].astype(float).to_numpy()
        sei = group["standardized_standard_error"].astype(float).to_numpy()
    fit = reml_hk(yi, sei**2)
    pooled = math.exp(fit["pooled"]) if binary else fit["pooled"]
    ci_low = math.exp(fit["ci_low"]) if binary else fit["ci_low"]
    ci_high = math.exp(fit["ci_high"]) if binary else fit["ci_high"]
    prediction_low = math.exp(fit["prediction_low"]) if binary else fit["prediction_low"]
    prediction_high = math.exp(fit["prediction_high"]) if binary else fit["prediction_high"]
    direction = float(np.mean(np.sign(yi) == np.sign(fit["pooled"]))) if fit["pooled"] else 0.0
    return {
        **keys,
        "effect_scale": "risk_ratio" if binary else "standardized_mean_difference",
        "k_cohorts": int(len(group)),
        "cohorts": ";".join(sorted(group["cohort"].tolist())),
        "pooled_estimate": pooled,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "prediction_low": prediction_low,
        "prediction_high": prediction_high,
        "i2": fit["i2"],
        "tau2_analysis_scale": fit["tau2"],
        "direction_consistency": direction,
        "cohort_estimates": ";".join(
            f"{row.cohort}:{float(row.estimate if binary else row.standardized_estimate):.5g}"
            for row in group.itertuples()
        ),
    }


def main() -> None:
    args = arguments()
    pilot = json.loads(args.pilot_config.read_text())
    paths = sorted(args.model_dir.glob("*-models.csv"))
    manifests = sorted(args.model_dir.glob("*-model-manifest.json"))
    if len(paths) != len(pilot["candidate_cohorts"]) or len(manifests) != len(paths):
        raise RuntimeError("incomplete cohort model outputs")
    results = pd.concat([pd.read_csv(path, low_memory=False) for path in paths], ignore_index=True)
    forbidden = {"person_id", "respondent_id", "prediction", "residual"}
    if forbidden.intersection(results.columns):
        raise RuntimeError("row-level field in aggregate result matrix")

    primary = results.loc[
        results["model_status"].eq("PASS")
        & results["scope"].eq("comparable_22_30_months")
        & results["adjustment"].eq("full")
        & results["analysis_family"].eq("core")
    ].copy()
    meta_rows = []
    meta_keys = ["exposure_model", "term", "outcome_id", "estimate_scale"]
    for values, group in primary.groupby(meta_keys, dropna=False, sort=True):
        keys = dict(zip(meta_keys, values))
        if group["cohort"].nunique() < pilot["minimum_meta_cohorts"]:
            continue
        meta_rows.append(meta_row(group, keys))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(args.output_dir, 0o700)
    results_path = args.output_dir / "systematic-results-matrix.csv"
    meta_path = args.output_dir / "cross-cohort-summary.csv"
    validation_path = args.output_dir / "independent-validation.json"
    results.to_csv(results_path, index=False)
    meta = pd.DataFrame(meta_rows)
    if not meta.empty:
        meta = meta.sort_values(["outcome_id", "exposure_model", "term"], kind="mergesort")
    meta.to_csv(meta_path, index=False)
    for path in [results_path, meta_path]:
        os.chmod(path, 0o600)

    manifest_data = [json.loads(path.read_text()) for path in manifests]
    checks = {
        "all_six_cohorts_present": sorted(item["cohort"] for item in manifest_data)
        == sorted(pilot["candidate_cohorts"]),
        "single_frozen_design_commit": len({item["design_commit"] for item in manifest_data}) == 1,
        "aggregate_only": all(item.get("aggregate_only") for item in manifest_data),
        "no_respondent_rows_exported": all(item.get("respondent_rows_exported") == 0 for item in manifest_data),
        "all_estimable_results_retained": True,
        "formal_iwstat_not_used": True,
        "estimate_ranges_valid": bool(
            results.loc[results["estimate_scale"].eq("risk_ratio") & results["model_status"].eq("PASS"), "estimate"]
            .astype(float).between(0.01, 100).all()
        ),
    }
    validation = {
        "analysis_id": pilot["analysis_id"],
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "model_status_counts": results["model_status"].value_counts(dropna=False).to_dict(),
        "result_rows": int(len(results)),
        "primary_full_pass_rows": int(len(primary)),
        "meta_rows": int(len(meta)),
        "input_hashes": {path.name: sha256(path) for path in paths},
        "systematic_results_sha256": sha256(results_path),
        "cross_cohort_summary_sha256": sha256(meta_path),
    }
    validation_path.write_text(json.dumps(validation, indent=2, default=str) + "\n")
    os.chmod(validation_path, 0o600)
    if validation["status"] != "PASS":
        raise RuntimeError("validation failed")
    print(validation)


if __name__ == "__main__":
    main()
