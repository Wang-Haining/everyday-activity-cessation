#!/usr/bin/env python3
"""Pool aggregate frailty and disease-specific estimates."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from behavioral_contraction_biomarker_core import file_sha, write_json
from postprocess_behavior_outcome_landscape import reml_hk


OUTCOME_LABELS = {
    "incident_diabetes": "Diabetes",
    "incident_stroke": "Stroke",
    "incident_heart_disease": "Heart disease",
    "incident_hypertension": "Hypertension",
    "incident_cancer": "Cancer",
    "incident_arthritis": "Arthritis",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension-config", required=True, type=Path)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--final-dir", required=True, type=Path)
    return parser.parse_args()


def _read_family(model_dir: Path, suffix: str) -> pd.DataFrame:
    paths = sorted(model_dir.glob(f"*-{suffix}.csv"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(path, low_memory=False) for path in paths], ignore_index=True)


def _meta(group: pd.DataFrame, keys: dict[str, Any]) -> dict[str, Any]:
    yi = np.log(group["estimate"].astype(float).to_numpy())
    sei = group["standard_error"].astype(float).to_numpy()
    fit = reml_hk(yi, sei**2)
    return {
        **keys,
        "k_cohorts": int(group["cohort"].nunique()),
        "cohorts": ";".join(sorted(group["cohort"].unique())),
        "pooled_rr": math.exp(fit["pooled"]),
        "ci_low": math.exp(fit["ci_low"]),
        "ci_high": math.exp(fit["ci_high"]),
        "prediction_low": math.exp(fit["prediction_low"]),
        "prediction_high": math.exp(fit["prediction_high"]),
        "i2": fit["i2"],
        "tau2_log_scale": fit["tau2"],
        "direction_above_one": float((group["estimate"].astype(float) > 1).mean()),
        "cohort_estimates": ";".join(
            f"{row.cohort}:{float(row.estimate):.4f}" for row in group.itertuples()
        ),
    }


def meta_results(specificity: pd.DataFrame, frailty: pd.DataFrame) -> pd.DataFrame:
    candidates = pd.concat([
        specificity.loc[
            specificity["model_status"].eq("PASS")
            & specificity["term"].ne("__model__")
        ],
        frailty.loc[
            frailty["model_status"].eq("PASS")
            & frailty["term"].isin([
                "any_withdrawal", "loss_1", "loss_2plus", "prefrail", "frail", "fried4", "fi_per_0_1",
            ])
        ],
    ], ignore_index=True, sort=False)
    rows = []
    keys = ["analysis_family", "model_id", "outcome_id", "term"]
    for values, group in candidates.groupby(keys, dropna=False, sort=True):
        if group["cohort"].nunique() < 3:
            continue
        rows.append(_meta(group, dict(zip(keys, values))))
    return pd.DataFrame(rows)














def main() -> None:
    args = arguments()
    extension = json.loads(args.extension_config.read_text())
    args.final_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(args.final_dir, 0o700)
    families = {
        "specificity": _read_family(args.model_dir, "specificity-models"),
        "frailty": _read_family(args.model_dir, "frailty-models"),
        "context": _read_family(args.model_dir, "context-models"),
        "delayed": _read_family(args.model_dir, "delayed-models"),
        "sensitivity": _read_family(args.model_dir, "sensitivity-models"),
        "performance": _read_family(args.model_dir, "cv-performance"),
        "risks": _read_family(args.model_dir, "standardized-risks"),
        "coverage": _read_family(args.model_dir, "event-coverage"),
    }
    for name, frame in families.items():
        if frame.empty and name not in {"risks"}:
            raise RuntimeError(f"empty model family: {name}")
        path = args.final_dir / f"systematic-{name}-results.csv"
        frame.to_csv(path, index=False)
        os.chmod(path, 0o600)
    meta = meta_results(families["specificity"], families["frailty"])
    meta_path = args.final_dir / "cross-cohort-meta-analysis.csv"
    meta.to_csv(meta_path, index=False)
    os.chmod(meta_path, 0o600)
    manifest = {
        "analysis_id": extension["analysis_id"], "status": "PASS",
        "aggregate_only": True, "respondent_rows_exported": 0,
        "outputs": {
            path.name: file_sha(path) for path in args.final_dir.iterdir() if path.is_file()
        },
    }
    write_json(args.final_dir / "postprocess-manifest.json", manifest)
    print({"status": "PASS", "meta_rows": len(meta)})


if __name__ == "__main__":
    main()
