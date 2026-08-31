#!/usr/bin/env python3
"""Aggregate the baseline-opportunity clinical sensitivity without row-level output."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from postprocess_behavior_outcome_landscape import reml_hk

from cohort_core import write_frame, write_json

OUTCOMES = [
    "incident_diabetes",
    "incident_stroke",
    "incident_heart_disease",
    "incident_hypertension",
    "multimorbidity_progression",
    "incident_any_adl",
    "mortality",
]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    files = sorted(args.model_dir.glob("*-clinical-models.csv"))
    if not files:
        raise RuntimeError("no cohort model files")
    data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    cohort_rows = data.loc[
        data["scope"].eq("comparable_22_30_months")
        & data["outcome_id"].isin(OUTCOMES)
        & data["adjustment"].eq("full")
        & data["exposure_model"].eq("any_contraction")
        & (data["term"].eq("any_contraction") | data["term"].isna())
    ].copy()
    if cohort_rows.empty:
        raise RuntimeError("no sensitivity rows selected")

    meta_rows: list[dict[str, object]] = []
    passing = cohort_rows.loc[cohort_rows["model_status"].eq("PASS")].copy()
    for outcome, group in passing.groupby("outcome_id"):
        group = group.drop_duplicates("cohort")
        if len(group) < 3:
            continue
        fit = reml_hk(
            np.log(group["estimate"].astype(float).to_numpy()),
            group["standard_error"].astype(float).to_numpy() ** 2,
        )
        pooled = float(np.exp(fit["pooled"]))
        estimates = group["estimate"].astype(float).to_numpy()
        meta_rows.append(
            {
                "scope": "comparable_22_30_months__baseline_engagement_ge_1",
                "outcome_id": outcome,
                "k": int(len(group)),
                "cohorts": "|".join(sorted(group["cohort"].tolist())),
                "pooled_estimate": pooled,
                "ci_low": float(np.exp(fit["ci_low"])),
                "ci_high": float(np.exp(fit["ci_high"])),
                "prediction_low": float(np.exp(fit["prediction_low"])),
                "prediction_high": float(np.exp(fit["prediction_high"])),
                "tau2": float(fit["tau2"]),
                "i2_percent": 100 * float(fit["i2"]),
                "same_direction_n": int(np.sum(estimates > 1)) if pooled >= 1 else int(np.sum(estimates < 1)),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = args.output_dir / "cohort-specific-results.csv"
    meta_path = args.output_dir / "meta-analysis.csv"
    write_frame(cohort_path, cohort_rows.to_dict("records"))
    write_frame(meta_path, meta_rows)
    write_json(
        args.output_dir / "validation.json",
        {
            "status": "PASS",
            "restriction": "baseline_engagement_count >= 1",
            "cohort_files": len(files),
            "cohort_rows": int(len(cohort_rows)),
            "meta_rows": int(len(meta_rows)),
            "aggregate_only": True,
            "respondent_rows_exported": 0,
            "negative_results_retained": True,
        },
    )
    print({"status": "PASS", "cohort_rows": len(cohort_rows), "meta_rows": len(meta_rows)})


if __name__ == "__main__":
    main()
