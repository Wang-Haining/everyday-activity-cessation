#!/usr/bin/env python3
"""Aggregate synthesis and figures for the behavioral-contraction pilot."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from behavioral_contraction_biomarker_core import file_sha, write_frame, write_json
from postprocess_behavior_outcome_landscape import reml_hk
from probe_behavior_outcome_feasibility import sha256


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-config", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_many(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def pattern(meta: dict, binary: bool, yi: np.ndarray) -> str:
    same = float(np.mean(np.sign(yi) == np.sign(meta["pooled"]))) if meta["pooled"] != 0 else 0.0
    ci = meta["ci_low"] > 0 or meta["ci_high"] < 0
    pi = meta["prediction_low"] > 0 or meta["prediction_high"] < 0
    magnitude = math.exp(meta["pooled"]) if binary else meta["pooled"]
    if ci and pi and same >= 0.75:
        return "REPLICATED_PI_SUPPORTED"
    if ci and same >= 0.75:
        return "DIRECTIONALLY_CONSISTENT_PI_UNCERTAIN"
    if (0.90 <= magnitude <= 1.11 if binary else abs(magnitude) < 0.10) and not ci:
        return "STABLE_NULL_OR_SMALL"
    if same >= 0.75:
        return "DIRECTIONALLY_CONSISTENT_IMPRECISE"
    return "HETEROGENEOUS_OR_INCONSISTENT"


def meta_rows(clinical: pd.DataFrame, pilot: dict) -> list[dict]:
    passed = clinical.loc[clinical["model_status"].eq("PASS") & clinical["adjustment"].eq("full")].copy()
    keys = ["scope", "outcome_id", "exposure_model", "term", "estimate_scale"]
    rows = []
    for values, group in passed.groupby(keys, dropna=False):
        group = group.drop_duplicates(["cohort", *keys])
        if len(group) < pilot["minimum_meta_cohorts"]:
            continue
        binary = str(values[-1]) == "risk_ratio"
        if binary:
            yi = np.log(pd.to_numeric(group["estimate"]).to_numpy(float))
            se = pd.to_numeric(group["standard_error"]).to_numpy(float)
        else:
            valid = group["standardized_estimate"].notna() & group["standardized_se"].notna()
            group = group.loc[valid]
            if len(group) < pilot["minimum_meta_cohorts"]:
                continue
            yi = pd.to_numeric(group["standardized_estimate"]).to_numpy(float)
            se = pd.to_numeric(group["standardized_se"]).to_numpy(float)
        fit = reml_hk(yi, se**2)
        row = dict(zip(keys, values))
        row.update({
            "k": int(len(group)), "cohorts": "|".join(sorted(group["cohort"].unique())),
            "same_direction_fraction": float(np.mean(np.sign(yi) == np.sign(fit["pooled"]))),
            "pooled_estimate": math.exp(fit["pooled"]) if binary else fit["pooled"],
            "ci_low": math.exp(fit["ci_low"]) if binary else fit["ci_low"],
            "ci_high": math.exp(fit["ci_high"]) if binary else fit["ci_high"],
            "prediction_low": math.exp(fit["prediction_low"]) if binary else fit["prediction_low"],
            "prediction_high": math.exp(fit["prediction_high"]) if binary else fit["prediction_high"],
            "tau2": fit["tau2"], "i2_percent": 100 * fit["i2"],
            "pattern_class": pattern(fit, binary, yi),
        })
        rows.append(row)
    return rows














def main() -> None:
    args = arguments()
    pilot = json.loads(args.pilot_config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(args.output_dir, 0o700)
    clinical = read_many(sorted(args.model_dir.glob("*-clinical-models.csv")))
    biomarker = read_many(sorted(args.model_dir.glob("*-biomarker-models.csv")))
    support = read_many(sorted(args.probe_dir.glob("*-outcome-support.csv")))
    assay_support = read_many(sorted(args.probe_dir.glob("*-assay-support.csv")))
    flows = read_many(sorted(args.probe_dir.glob("*-flow.csv")))
    intervals = read_many(sorted(args.probe_dir.glob("*-intervals.csv")))
    if clinical.empty or support.empty or flows.empty:
        raise RuntimeError("incomplete cohort outputs")
    meta = pd.DataFrame(meta_rows(clinical, pilot))
    outputs = {
        "clinical": args.output_dir / "systematic-clinical-results.csv",
        "support": args.output_dir / "feasibility-matrix.csv",
        "flows": args.output_dir / "cohort-flow.csv",
        "intervals": args.output_dir / "followup-interval-audit.csv",
        "meta": args.output_dir / "cross-cohort-meta-analysis.csv",
    }
    write_frame(outputs["clinical"], clinical.to_dict("records"))
    write_frame(outputs["support"], support.to_dict("records"))
    write_frame(outputs["flows"], flows.to_dict("records"))
    write_frame(outputs["intervals"], intervals.to_dict("records"))
    write_frame(outputs["meta"], meta.to_dict("records") if not meta.empty else [{"status": "NOT_EVALUABLE_NO_META"}])
    if not biomarker.empty:
        outputs["biomarker"] = args.output_dir / "systematic-biomarker-results.csv"
        write_frame(outputs["biomarker"], biomarker.to_dict("records"))
    if not assay_support.empty:
        outputs["assay_support"] = args.output_dir / "assay-feasibility-matrix.csv"
        write_frame(outputs["assay_support"], assay_support.to_dict("records"))
    manifest = {
        "analysis_id": pilot["analysis_id"], "pilot_config_sha256": sha256(args.pilot_config),
        "model_manifests": {path.name: file_sha(path) for path in sorted(args.model_dir.glob("*-model-manifest.json"))},
        "probe_manifests": {path.name: file_sha(path) for path in sorted(args.probe_dir.glob("*-probe-manifest.json"))},
        "outputs": {key: file_sha(path) for key, path in outputs.items()},
        "aggregate_only": True, "negative_results_retained": True,
    }
    write_json(args.output_dir / "postprocess-manifest.json", manifest)
    print({"status": "PASS", "clinical_rows": len(clinical), "biomarker_rows": len(biomarker), "meta_rows": len(meta)})


if __name__ == "__main__":
    main()
