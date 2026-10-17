#!/usr/bin/env python3
"""Fit aggregate-only ELSA models for behavioral contraction and later HbA1c."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from behavioral_contraction_biomarker_core import (
    add_base_design,
    binary_support,
    coefficient_rows,
    continuous_support,
    fit_clustered,
    write_frame,
    write_json,
)
from elsa_behavior_hba1c_early_warning_core import build_frame, load_early_specs, outcome_mask
from probe_behavior_outcome_feasibility import sha256


BINARY_OUTCOMES = [
    "glycemic_category_progression",
    "incident_diabetes_range_hba1c",
    "incident_prediabetes_or_higher",
    "incident_fasting_glucose_diabetes_range",
]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--universe-config", required=True, type=Path)
    parser.add_argument("--pilot-config", required=True, type=Path)
    parser.add_argument("--assay-config", required=True, type=Path)
    parser.add_argument("--early-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def model_data(
    source: pd.DataFrame,
    outcome: str,
    exposure_model: str,
    pilot: dict[str, Any],
    adjustment: str = "full",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str], bool]:
    binary_outcome = outcome != "followup_hba1c"
    mask = (
        outcome_mask(source, outcome)
        if binary_outcome
        else source["primary_base"] & source["followup_hba1c"].notna()
    )
    data = source.loc[mask].copy()
    if exposure_model == "any_contraction":
        terms = ["any_contraction"]
    elif exposure_model == "one_loss_vs_zero":
        data = data.loc[data["loss_count_2"].isin([0, 1])].copy()
        data["one_loss"] = data["loss_count_2"].eq(1).astype(float)
        terms = ["one_loss"]
    elif exposure_model == "two_loss_vs_zero":
        data = data.loc[data["loss_count_2"].isin([0, 2])].copy()
        data["two_loss"] = data["loss_count_2"].eq(2).astype(float)
        terms = ["two_loss"]
    elif exposure_model == "mutually_adjusted_components":
        terms = ["alcohol_loss", "activity_loss"]
    else:
        raise ValueError(exposure_model)

    required = [
        "person_id",
        "t1_wave",
        "age",
        "sex",
        "baseline_engagement_count",
        "baseline_hba1c",
        *terms,
    ]
    if adjustment == "full":
        required += [
            "education_rank",
            "income_rank",
            "smoke_ever",
            "smoke_current",
            "baseline_multimorbidity",
        ]
    if binary_outcome:
        required.append(outcome)
    else:
        required.append("followup_hba1c")
    if outcome == "incident_fasting_glucose_diabetes_range":
        required.append("baseline_fasting_glucose")
    data = data.loc[data[required].notna().all(axis=1)].copy()
    y = data[outcome].astype(float) if binary_outcome else data["followup_hba1c"].astype(float)
    X = add_base_design(data, pilot, "full" if adjustment == "full" else "basic")
    X["baseline_hba1c"] = data["baseline_hba1c"].astype(float)
    if outcome == "incident_fasting_glucose_diabetes_range":
        X["baseline_fasting_glucose"] = data["baseline_fasting_glucose"].astype(float)
    for term in terms:
        X[term] = data[term].astype(float)
    return data, y, X, terms, binary_outcome


def standardized_risks(
    result: Any,
    X: pd.DataFrame,
    term: str,
) -> dict[str, float]:
    low = X.copy()
    high = X.copy()
    low[term] = 0.0
    high[term] = 1.0
    risk0 = float(np.mean(result.predict(low)))
    risk1 = float(np.mean(result.predict(high)))
    difference = risk1 - risk0
    return {
        "standardized_risk_unexposed": risk0,
        "standardized_risk_exposed": risk1,
        "standardized_risk_difference": difference,
        "assessments_per_additional_event": 1.0 / difference if difference > 0 else np.nan,
    }


def raw_metrics(data: pd.DataFrame, y: pd.Series, outcome: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    total_events = int(y.sum())
    for group_field, groups in [
        ("any_contraction", [0, 1]),
        ("loss_count_2", [0, 1, 2]),
    ]:
        for group in groups:
            selected = data[group_field].eq(group)
            n = int(selected.sum())
            events = int(y.loc[selected].sum())
            rows.append(
                {
                    "outcome": outcome,
                    "group_field": group_field,
                    "group": group,
                    "n": n,
                    "participants": int(data.loc[selected, "person_id"].nunique()),
                    "events": events,
                    "risk": events / n if n else np.nan,
                    "event_coverage": events / total_events if total_events else np.nan,
                    "flag_prevalence": n / len(data) if len(data) else np.nan,
                }
            )
    exposed = data["any_contraction"].eq(1)
    unexposed = data["any_contraction"].eq(0)
    exposed_risk = float(y.loc[exposed].mean()) if exposed.any() else np.nan
    unexposed_risk = float(y.loc[unexposed].mean()) if unexposed.any() else np.nan
    difference = exposed_risk - unexposed_risk
    rows.append(
        {
            "outcome": outcome,
            "group_field": "any_contraction_summary",
            "group": "contrast",
            "n": int(len(data)),
            "participants": int(data["person_id"].nunique()),
            "events": total_events,
            "risk": exposed_risk,
            "comparator_risk": unexposed_risk,
            "risk_difference": difference,
            "sensitivity": int(y.loc[exposed].sum()) / total_events if total_events else np.nan,
            "specificity": int((1 - y.loc[unexposed]).sum()) / int((1 - y).sum())
            if int((1 - y).sum())
            else np.nan,
            "positive_predictive_value": exposed_risk,
            "assessments_per_additional_event": 1.0 / difference if difference > 0 else np.nan,
        }
    )
    return rows


def main() -> None:
    args = arguments()
    assert args.data_root.is_dir(), f"missing data root: {args.data_root}"
    universe, pilot, assay, early = load_early_specs(
        args.universe_config, args.pilot_config, args.assay_config, args.early_config
    )
    source, source_audit = build_frame(args.data_root, universe, pilot, assay, early)
    model_rows: list[dict[str, object]] = []
    standardization_rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []

    specifications = [
        (outcome, exposure_model, "full")
        for outcome in [*BINARY_OUTCOMES, "followup_hba1c"]
        for exposure_model in [
            "any_contraction",
            "one_loss_vs_zero",
            "two_loss_vs_zero",
            "mutually_adjusted_components",
        ]
    ]
    specifications.append(
        ("incident_diabetes_range_hba1c", "any_contraction", "clinical_core")
    )

    for outcome, exposure_model, adjustment in specifications:
        base = {
            "outcome": outcome,
            "exposure_model": exposure_model,
            "adjustment": adjustment,
        }
        try:
            data, y, X, terms, binary_outcome = model_data(
                source, outcome, exposure_model, pilot, adjustment
            )
            if exposure_model == "any_contraction" and binary_outcome and adjustment == "full":
                raw_rows.extend(raw_metrics(data, y, outcome))
            if binary_outcome:
                status, counts = binary_support(data, y, X, terms, early)
            else:
                status, counts = continuous_support(data, "any_contraction", early)
                if exposure_model != "any_contraction":
                    status = "NOT_EVALUABLE_SECONDARY_CONTINUOUS_EXPOSURE"
            if status != "ESTIMABLE":
                model_rows.append({**base, **counts, "model_status": status})
                continue
            result, warning_text = fit_clustered(
                y, X, data["person_id"], binary_outcome
            )
            for coefficient in coefficient_rows(result, terms, binary_outcome):
                model_rows.append(
                    {
                        **base,
                        **counts,
                        "model_status": "PASS",
                        "warnings": warning_text,
                        **coefficient,
                    }
                )
            if binary_outcome and len(terms) == 1:
                standardization_rows.append(
                    {
                        **base,
                        "term": terms[0],
                        "n": len(data),
                        "events": int(y.sum()),
                        **standardized_risks(result, X, terms[0]),
                    }
                )
        except Exception as exc:
            model_rows.append(
                {
                    **base,
                    "model_status": "MODEL_FAILURE",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_frame(args.output_dir / "model-results.csv", model_rows)
    write_frame(args.output_dir / "standardized-risks.csv", standardization_rows)
    write_frame(args.output_dir / "raw-clinical-metrics.csv", raw_rows)
    write_json(
        args.output_dir / "validation.json",
        {
            "status": "PASS",
            "analysis_id": early["analysis_id"],
            "aggregate_only": True,
            "respondent_rows_exported": 0,
            "negative_results_retained": True,
            "config_sha256": sha256(args.early_config),
            "source_audit": source_audit,
            "output_rows": {
                "models": len(model_rows),
                "standardized_risks": len(standardization_rows),
                "raw_metrics": len(raw_rows),
            },
        },
    )
    print({"status": "PASS", "model_rows": len(model_rows)})


if __name__ == "__main__":
    main()
