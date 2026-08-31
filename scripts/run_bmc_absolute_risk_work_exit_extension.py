#!/usr/bin/env python3
"""Run the frozen BMC absolute-risk and work-exit extension for one cohort."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from probe_behavior_outcome_feasibility import load_lookup, sha256

from multidomain_behavioral_withdrawal_core import (
    file_sha,
    fit_clustered,
    load_episodes,
    load_specs,
    multidomain_frame,
    prepare_model,
    support_gate,
    write_frame,
    write_json,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe-config", required=True, type=Path)
    parser.add_argument("--pilot-config", required=True, type=Path)
    parser.add_argument("--extension-config", required=True, type=Path)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--design-commit", required=True)
    parser.add_argument("--design-commit-time", required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def _load_extension(args: argparse.Namespace, pilot: dict) -> dict:
    extension = json.loads(args.extension_config.read_text())
    if sha256(args.universe_config) != extension["parent_universe_sha256"]:
        raise RuntimeError("parent universe config drift")
    if sha256(args.pilot_config) != extension["parent_pilot_sha256"]:
        raise RuntimeError("parent pilot config drift")
    if extension["release_root"] != pilot["release_root"]:
        raise RuntimeError("release-root mismatch")
    return extension


def _scenario_design(X: pd.DataFrame, scenario: dict[str, int]) -> pd.DataFrame:
    result = X.copy()
    for field, value in scenario.items():
        if field not in result:
            raise RuntimeError(f"scenario field absent from design: {field}")
        result[field] = float(value)
    return result


def _gradient(X: pd.DataFrame, prediction: np.ndarray) -> np.ndarray:
    derivative = prediction * (1.0 - prediction)
    return np.mean(derivative[:, None] * X.to_numpy(float), axis=0)


def _fit_probability_model(
    y: pd.Series, X: pd.DataFrame, groups: pd.Series, model_id: str
):
    if model_id != "binomial_logistic_cluster_robust":
        raise RuntimeError(f"unsupported probability model: {model_id}")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = sm.GLM(y, X, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": groups}
        )
    return result, " | ".join(str(item.message) for item in captured)


def standardized_risk_rows(
    episodes: pd.DataFrame,
    behavior: pd.DataFrame,
    universe: dict,
    pilot: dict,
    extension: dict,
    cohort: str,
) -> list[dict]:
    rows: list[dict] = []
    for outcome_id in extension["risk_outcomes"]:
        base = {
            "cohort": cohort,
            "scope": extension["scope"],
            "adjustment": extension["adjustment"],
            "outcome_id": outcome_id,
            "exposure_model": extension["risk_exposure_model"],
        }
        try:
            data, y, X, terms, binary, coding = prepare_model(
                episodes,
                behavior,
                universe,
                pilot,
                cohort,
                outcome_id,
                extension["scope"],
                extension["adjustment"],
                extension["risk_exposure_model"],
            )
            if not binary:
                raise RuntimeError("standardized-risk outcome is not binary")
            status, counts = support_gate(data, y, X, terms, pilot, binary)
            common = {**base, **counts, "outcome_coding_status": coding}
            if status != "ESTIMABLE":
                rows.append({**common, "scenario": "__model__", "model_status": status})
                continue

            fit, warning_text = _fit_probability_model(
                y, X, data["person_id"], extension["risk_probability_model"]
            )
            covariance = fit.cov_params().loc[X.columns, X.columns].to_numpy(float)
            scenario_values: dict[str, dict] = {}
            for scenario_name, scenario in extension["risk_scenarios"].items():
                scenario_X = _scenario_design(X, scenario)
                prediction = np.asarray(fit.predict(scenario_X), dtype=float)
                if (
                    prediction.size == 0
                    or np.any(~np.isfinite(prediction))
                    or np.any(prediction < extension["prediction_range"][0])
                    or np.any(prediction > extension["prediction_range"][1])
                ):
                    rows.append({
                        **common,
                        "scenario": scenario_name,
                        "model_status": "NOT_EVALUABLE_PREDICTION_RANGE",
                    })
                    continue
                risk = float(np.mean(prediction))
                gradient = _gradient(scenario_X, prediction)
                variance = float(gradient @ covariance @ gradient)
                se = math.sqrt(max(0.0, variance))
                ci_low = max(0.0, risk - 1.959963984540054 * se)
                ci_high = min(1.0, risk + 1.959963984540054 * se)

                if scenario_name == "withdrawal_0":
                    observed = data["loss_1"].eq(0) & data["loss_2plus"].eq(0)
                elif scenario_name == "withdrawal_1":
                    observed = data["loss_1"].eq(1)
                elif scenario_name == "withdrawal_2plus":
                    observed = data["loss_2plus"].eq(1)
                else:
                    raise RuntimeError(f"unknown scenario {scenario_name}")

                scenario_values[scenario_name] = {
                    "risk": risk,
                    "gradient": gradient,
                    "variance": variance,
                }
                rows.append({
                    **common,
                    "scenario": scenario_name,
                    "model_status": "PASS",
                    "standardized_risk": risk,
                    "standard_error": se,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "observed_n": int(observed.sum()),
                    "observed_events": int(y.loc[observed].sum()),
                    "observed_risk": float(y.loc[observed].mean()),
                    "warnings": warning_text,
                })

            reference = scenario_values.get("withdrawal_0")
            if reference is not None:
                for row in rows:
                    if (
                        row.get("cohort") != cohort
                        or row.get("outcome_id") != outcome_id
                        or row.get("model_status") != "PASS"
                        or row.get("scenario") == "withdrawal_0"
                    ):
                        continue
                    current = scenario_values[row["scenario"]]
                    covariance_between = float(
                        current["gradient"] @ covariance @ reference["gradient"]
                    )
                    rd_variance = max(
                        0.0,
                        current["variance"] + reference["variance"] - 2 * covariance_between,
                    )
                    rd = float(current["risk"] - reference["risk"])
                    rd_se = math.sqrt(rd_variance)
                    row.update({
                        "risk_difference_vs_0": rd,
                        "risk_difference_standard_error": rd_se,
                        "risk_difference_ci_low": rd - 1.959963984540054 * rd_se,
                        "risk_difference_ci_high": rd + 1.959963984540054 * rd_se,
                    })
        except Exception as exc:
            rows.append({
                **base,
                "scenario": "__model__",
                "model_status": "MODEL_FAILURE",
                "failure_reason": f"{type(exc).__name__}: {exc}",
            })
    return rows


def work_exit_contrast_rows(
    episodes: pd.DataFrame,
    behavior: pd.DataFrame,
    universe: dict,
    pilot: dict,
    extension: dict,
    cohort: str,
) -> list[dict]:
    rows: list[dict] = []
    for outcome_id in extension["work_exit_outcomes"]:
        base = {
            "cohort": cohort,
            "scope": extension["scope"],
            "adjustment": extension["adjustment"],
            "outcome_id": outcome_id,
            "exposure_model": extension["work_exit_exposure_model"],
            "contrast": extension["direct_contrast"],
        }
        if "work_exit_retirement" not in behavior or "work_exit_no_retirement" not in behavior:
            rows.append({**base, "model_status": "NOT_EVALUABLE_RETIREMENT_NOT_MEASURED"})
            continue
        try:
            data, y, X, terms, binary, coding = prepare_model(
                episodes,
                behavior,
                universe,
                pilot,
                cohort,
                outcome_id,
                extension["scope"],
                extension["adjustment"],
                extension["work_exit_exposure_model"],
            )
            if not binary:
                raise RuntimeError("work-exit outcome is not binary")
            status, counts = support_gate(data, y, X, terms, pilot, binary)
            common = {**base, **counts, "outcome_coding_status": coding}
            if status != "ESTIMABLE":
                rows.append({**common, "model_status": status})
                continue

            fit, warnings = fit_clustered(y, X, data["person_id"], binary)
            covariance = fit.cov_params()
            beta_retirement = float(fit.params["work_exit_retirement"])
            beta_other = float(fit.params["work_exit_no_retirement"])
            beta = beta_other - beta_retirement
            variance = float(
                covariance.loc["work_exit_no_retirement", "work_exit_no_retirement"]
                + covariance.loc["work_exit_retirement", "work_exit_retirement"]
                - 2 * covariance.loc["work_exit_no_retirement", "work_exit_retirement"]
            )
            se = math.sqrt(max(0.0, variance))
            rows.append({
                **common,
                "model_status": "PASS",
                "estimate_scale": "ratio_of_risk_ratios",
                "estimate": math.exp(beta),
                "standard_error_log_scale": se,
                "ci_low": math.exp(beta - 1.959963984540054 * se),
                "ci_high": math.exp(beta + 1.959963984540054 * se),
                "retirement_exit_rr_vs_continued_work": math.exp(beta_retirement),
                "other_exit_rr_vs_continued_work": math.exp(beta_other),
                "warnings": warnings,
            })
        except Exception as exc:
            rows.append({
                **base,
                "model_status": "MODEL_FAILURE",
                "failure_reason": f"{type(exc).__name__}: {exc}",
            })
    return rows


def main() -> None:
    args = arguments()
    universe, pilot = load_specs(args.universe_config, args.pilot_config)
    extension = _load_extension(args, pilot)
    cohort = args.cohort.lower()
    if cohort not in extension["candidate_cohorts"]:
        raise RuntimeError("cohort outside frozen extension")

    root = Path(extension["release_root"])
    lookup, lookup_path = load_lookup(root, universe)
    episodes, intervals, source_audit = load_episodes(root, universe, cohort, lookup)
    behavior = multidomain_frame(episodes, universe, pilot, cohort)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    risk_path = args.output_dir / f"{cohort}-standardized-risks.csv"
    contrast_path = args.output_dir / f"{cohort}-work-exit-contrasts.csv"
    interval_path = args.output_dir / f"{cohort}-interval-provenance.csv"
    write_frame(
        risk_path,
        standardized_risk_rows(episodes, behavior, universe, pilot, extension, cohort),
    )
    write_frame(
        contrast_path,
        work_exit_contrast_rows(episodes, behavior, universe, pilot, extension, cohort),
    )
    interval_rows = [
        {**row, "cohort": cohort}
        for row in intervals
        if 22 <= float(row.get("scheduled_followup_months", math.nan)) <= 30
    ]
    if not interval_rows:
        interval_rows = [{
            "cohort": cohort,
            "t1": math.nan,
            "outcome_wave": math.nan,
            "scheduled_followup_months": math.nan,
            "provenance_status": "NOT_APPLICABLE_NO_COMPARABLE_INTERVAL",
        }]
    write_frame(interval_path, interval_rows)

    outputs = {
        "standardized_risks": risk_path,
        "work_exit_contrasts": contrast_path,
        "interval_provenance": interval_path,
    }
    manifest_path = args.output_dir / f"{cohort}-manifest.json"
    write_json(manifest_path, {
        "analysis_id": extension["analysis_id"],
        "cohort": cohort,
        "design_commit": args.design_commit,
        "design_commit_time": args.design_commit_time,
        "code_commit": args.code_commit,
        "design_sha256": sha256(args.design),
        "extension_config_sha256": sha256(args.extension_config),
        "pilot_config_sha256": sha256(args.pilot_config),
        "universe_config_sha256": sha256(args.universe_config),
        "lookup_sha256": sha256(lookup_path),
        "source_audit": source_audit,
        "outputs": {key: file_sha(path) for key, path in outputs.items()},
        "aggregate_only": True,
        "respondent_rows_exported": 0,
    })
    print({"status": "PASS", "cohort": cohort, "outputs": len(outputs)})


if __name__ == "__main__":
    main()
