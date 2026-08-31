#!/usr/bin/env python3
"""Independent aggregate, provenance and reconciliation validator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from postprocess_behavior_outcome_landscape import reml_hk

from cohort_core import file_sha, write_json


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension-config", required=True, type=Path)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--final-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    extension = json.loads(args.extension_config.read_text())
    checks: list[dict[str, object]] = []
    cohorts = extension["specificity_cohorts"]
    model_commits = set()
    design_commits = set()
    for cohort in cohorts:
        probe_manifest_path = args.probe_dir / f"{cohort}-probe-manifest.json"
        model_manifest_path = args.model_dir / f"{cohort}-model-manifest.json"
        checks.append({"check": f"{cohort}_probe_manifest", "pass": probe_manifest_path.exists()})
        checks.append({"check": f"{cohort}_model_manifest", "pass": model_manifest_path.exists()})
        if probe_manifest_path.exists():
            manifest = json.loads(probe_manifest_path.read_text())
            checks.append({
                "check": f"{cohort}_probe_privacy",
                "pass": manifest.get("aggregate_only") is True
                and manifest.get("respondent_rows_exported") == 0
                and manifest.get("effect_models_fit") == 0,
            })
        if model_manifest_path.exists():
            manifest = json.loads(model_manifest_path.read_text())
            model_commits.add(manifest.get("code_commit"))
            design_commits.add(manifest.get("design_commit"))
            checks.append({
                "check": f"{cohort}_model_privacy",
                "pass": manifest.get("aggregate_only") is True
                and manifest.get("respondent_rows_exported") == 0,
            })
            for key, expected in manifest.get("outputs", {}).items():
                suffixes = {
                    "specificity": "specificity-models.csv", "delayed": "delayed-models.csv",
                    "sensitivity": "sensitivity-models.csv", "frailty": "frailty-models.csv",
                    "performance": "cv-performance.csv", "risks": "standardized-risks.csv",
                    "coverage": "event-coverage.csv", "context": "context-models.csv",
                }
                path = args.model_dir / f"{cohort}-{suffixes[key]}"
                checks.append({
                    "check": f"{cohort}_{key}_hash",
                    "pass": path.exists() and file_sha(path) == expected,
                })
    checks.append({"check": "single_model_code_commit", "pass": len(model_commits) == 1})
    checks.append({"check": "single_frozen_design_commit", "pass": len(design_commits) == 1})

    result_files = list(args.model_dir.glob("*-models.csv")) + list(args.model_dir.glob("*-performance.csv"))
    frames = [pd.read_csv(path, low_memory=False) for path in result_files]
    all_results = pd.concat(frames, ignore_index=True, sort=False)
    checks.append({"check": "no_model_failures", "pass": not all_results["model_status"].eq("MODEL_FAILURE").any()})
    checks.append({
        "check": "not_evaluable_retained",
        "pass": bool(all_results["model_status"].astype(str).str.startswith("NOT_EVALUABLE").any()),
    })
    forbidden = {"respondent_id", "prediction", "residual", "fold", "ipcw_weight"}
    checks.append({"check": "no_row_level_fields", "pass": not bool(forbidden.intersection(all_results.columns))})
    passed = all_results.loc[all_results["model_status"].eq("PASS")]
    estimates = pd.to_numeric(passed.get("estimate"), errors="coerce").dropna()
    checks.append({"check": "finite_passed_estimates", "pass": bool(np.isfinite(estimates).all())})
    rr = passed.loc[passed.get("estimate_scale").eq("risk_ratio"), "estimate"].astype(float)
    checks.append({"check": "risk_ratio_range", "pass": bool(rr.between(0.05, 20).all())})

    frailty = pd.read_csv(args.final_dir / "systematic-frailty-results.csv", low_memory=False)
    primary = frailty.loc[frailty["model_id"].isin(["M0_routine", "M1_withdrawal", "M2_fried", "M3_fried_withdrawal"])]
    same_sets = True
    for _, group in primary.groupby(["cohort", "outcome_id"]):
        observed = pd.to_numeric(group["n"], errors="coerce").dropna().unique()
        same_sets &= len(observed) <= 1
    checks.append({"check": "head_to_head_common_analysis_sets", "pass": same_sets})

    performance = pd.read_csv(args.final_dir / "systematic-performance-results.csv", low_memory=False)
    deltas = performance.loc[
        performance["model_status"].eq("PASS") & performance["metric"].astype(str).str.startswith("delta_")
    ]
    checks.append({
        "check": "cv_deltas_have_grouped_bootstrap_ci",
        "pass": bool(len(deltas)) and deltas[["ci_low", "ci_high"]].notna().all(axis=None),
    })

    meta = pd.read_csv(args.final_dir / "cross-cohort-meta-analysis.csv")
    target = meta.loc[
        meta["analysis_family"].eq("specificity")
        & meta["model_id"].eq("any_withdrawal")
        & meta["outcome_id"].eq("incident_diabetes")
        & meta["term"].eq("any_withdrawal")
    ]
    specificity = pd.read_csv(args.final_dir / "systematic-specificity-results.csv", low_memory=False)
    source = specificity.loc[
        specificity["model_status"].eq("PASS")
        & specificity["model_id"].eq("any_withdrawal")
        & specificity["outcome_id"].eq("incident_diabetes")
        & specificity["term"].eq("any_withdrawal")
    ]
    meta_reconciles = False
    if len(target) == 1 and source["cohort"].nunique() >= 3:
        fit = reml_hk(np.log(source["estimate"].astype(float)), source["standard_error"].astype(float) ** 2)
        meta_reconciles = math.isclose(math.exp(fit["pooled"]), float(target.iloc[0]["pooled_rr"]), rel_tol=1e-10)
    checks.append({"check": "diabetes_meta_recomputed", "pass": meta_reconciles})

    pooled_manifest_path = args.final_dir / "pooled-sensitivity-manifest.json"
    checks.append({"check": "pooled_sensitivity_manifest", "pass": pooled_manifest_path.exists()})
    if pooled_manifest_path.exists():
        pooled_manifest = json.loads(pooled_manifest_path.read_text())
        pooled_path = args.final_dir / "pooled-fixed-effects-sensitivity.csv"
        checks.append({
            "check": "pooled_sensitivity_privacy_and_hash",
            "pass": pooled_manifest.get("aggregate_only") is True
            and pooled_manifest.get("respondent_rows_exported") == 0
            and pooled_path.exists()
            and file_sha(pooled_path) == pooled_manifest["outputs"]["pooled"],
        })

    story_path = args.final_dir / "story-classification.json"
    checks.append({"check": "story_gate_written", "pass": story_path.exists()})
    figures = list(args.final_dir.glob("figure-*.png"))
    checks.append({
        "check": "three_clinical_figures_written",
        "pass": len(figures) >= 3 and all(path.stat().st_size > 10000 for path in figures),
    })
    failed = [check for check in checks if not check["pass"]]
    report = {
        "analysis_id": extension["analysis_id"], "status": "PASS" if not failed else "FAIL",
        "checks": checks, "failed": failed, "aggregate_only": True,
        "model_code_commits": sorted(str(value) for value in model_commits),
        "design_commits": sorted(str(value) for value in design_commits),
    }
    write_json(args.output, report)
    if failed:
        raise RuntimeError(f"validation failed: {[item['check'] for item in failed]}")
    print({"status": "PASS", "checks": len(checks)})


if __name__ == "__main__":
    main()
