#!/usr/bin/env python3
"""Fit every prespecified estimable behavior x outcome model for one cohort.

Respondent-level data never leave memory. Output is one aggregate row per
cell/model plus an auditable manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from probe_behavior_outcome_feasibility import (
    behavior_states,
    build_episodes,
    load_lookup,
    max_valid,
    numeric,
    outcome_fields,
    outcome_values,
    read_formal,
    row_count,
    sha256,
    source_status_and_elsa_dates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe-config", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--feasibility", required=True, type=Path)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def restricted_cubic_spline(values: pd.Series, knots: list[float], prefix: str) -> pd.DataFrame:
    x = pd.to_numeric(values, errors="coerce").to_numpy(float)
    knots_array = np.asarray(knots, dtype=float)
    if len(knots_array) < 3 or np.any(np.diff(knots_array) <= 0):
        raise ValueError("restricted cubic spline knots must be strictly increasing")
    result = pd.DataFrame(index=values.index)
    result[f"{prefix}_linear"] = (x - knots_array[0]) / 10.0
    last = knots_array[-1]
    penultimate = knots_array[-2]
    scale = (last - knots_array[0]) ** 2 * 10.0
    for index, knot in enumerate(knots_array[:-2], start=1):
        raw = (
            np.maximum(x - knot, 0.0) ** 3
            - ((last - knot) / (last - penultimate)) * np.maximum(x - penultimate, 0.0) ** 3
            + ((penultimate - knot) / (last - penultimate)) * np.maximum(x - last, 0.0) ** 3
        )
        result[f"{prefix}_nonlinear_{index}"] = raw / scale
    return result


def baseline_outcome(
    episodes: pd.DataFrame,
    outcome_id: str,
    spec: dict[str, Any],
    outcomes: dict[str, Any],
    cohort: str,
) -> pd.Series | None:
    outcome_type = spec["type"]
    if outcome_type in {"binary_source_status", "incident_binary", "derived_incident_binary"}:
        return None
    fields = outcome_fields(outcome_id, spec, outcomes, cohort)
    if outcome_id in {"incident_any_adl", "adl_count_increase", "incident_any_iadl", "iadl_count_increase", "multimorbidity_progression"}:
        return row_count(episodes, "t1", fields)
    if outcome_type == "binary_interval_event":
        return pd.to_numeric(episodes[f"t1__{fields[0]}"], errors="coerce").where(lambda x: x.isin([0.0, 1.0]))
    if outcome_type == "ordinal_worsening":
        return numeric(episodes[f"t1__{fields[0]}"]).where(lambda x: x.isin(spec["valid_values"]))
    if outcome_id == "depressive_symptom_change" and cohort == "klosa":
        result = pd.Series(np.nan, index=episodes.index, dtype=float)
        field_a = spec["klosa_module"]["fields_by_waves"]["3-4"]
        field_b = spec["klosa_module"]["fields_by_waves"]["5-10"]
        mask_a = episodes["t1"].isin([3, 4]) & episodes["outcome_wave"].isin([3, 4])
        mask_b = episodes["t1"].isin([5, 6, 7, 8, 9]) & episodes["outcome_wave"].isin([5, 6, 7, 8, 9])
        result.loc[mask_a] = numeric(episodes.loc[mask_a, f"t1__{field_a}"], [0, 30])
        result.loc[mask_b] = numeric(episodes.loc[mask_b, f"t1__{field_b}"], [0, 30])
        return result
    if outcome_type.startswith("continuous_change"):
        valid_range = spec.get("valid_range_by_cohort", {}).get(cohort, spec.get("valid_range"))
        if outcome_id == "grip_strength_change":
            return max_valid(episodes, "t1", fields, valid_range)
        return numeric(episodes[f"t1__{fields[0]}"], valid_range)
    return None


def baseline_behavior(
    episodes: pd.DataFrame, transition_id: str, spec: dict[str, Any], cohort: str
) -> pd.Series | None:
    if spec["kind"] == "binary":
        return None
    field = spec["fields"][cohort][0]
    values = pd.to_numeric(episodes[f"t0__{field}"], errors="coerce")
    if transition_id == "smoking_quantity":
        low, high = spec["valid_range"]
        return values.where(values.between(low, high))
    rule = spec.get("rules_by_cohort", {}).get(cohort, {})
    if "valid_range" in rule:
        low, high = rule["valid_range"]
        return values.where(values.between(low, high))
    return values.where(values.ge(0))


def multimorbidity(episodes: pd.DataFrame, fields: list[str]) -> pd.Series:
    values = episodes[[f"t1__{field}" for field in fields]].apply(pd.to_numeric, errors="coerce")
    valid = values.isin([0.0, 1.0]).all(axis=1)
    result = pd.Series(np.nan, index=episodes.index, dtype=float)
    result.loc[valid] = values.loc[valid].sum(axis=1)
    return result


def prepare_design(
    episodes: pd.DataFrame,
    universe: dict[str, Any],
    model_config: dict[str, Any],
    cohort: str,
    transition_id: str,
    transition_spec: dict[str, Any],
    outcome_id: str,
    outcome_spec: dict[str, Any],
    scope: str,
    transition_state: str,
    comparator_state: str,
    adjustment: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, Any]]:
    states = behavior_states(episodes, transition_id, transition_spec, cohort, universe["cohorts"][cohort]["primary_waves"])
    outcome, outcome_status = outcome_values(episodes, outcome_id, outcome_spec, universe["outcomes"], cohort)
    scope_mask = pd.Series(True, index=episodes.index) if scope == "all_primary_wave_intervals" else episodes["comparable_window"].fillna(False)
    mask = scope_mask & states.isin([transition_state, comparator_state]) & outcome.notna()
    cc = universe["cohorts"][cohort]
    data = pd.DataFrame(index=episodes.index)
    data["outcome"] = outcome
    data["transition"] = states.eq(transition_state).astype(float)
    data["age"] = numeric(episodes[f"t1__{cc['age']}"])
    data["sex"] = numeric(episodes[f"t1__{cc['sex']}"]).where(lambda x: x.isin([1.0, 2.0]))
    data["t1_wave"] = episodes["t1"].astype(float)
    data["person_id"] = episodes["person_id"]
    base = baseline_outcome(episodes, outcome_id, outcome_spec, universe["outcomes"], cohort)
    if base is not None:
        data["baseline_outcome"] = base
    omitted = []
    if adjustment == "full":
        education = numeric(episodes[f"t1__{cc['education']}"])
        income = numeric(episodes[f"t1__{cc['income']}"])
        data["education_rank"] = education.groupby(episodes["t1"]).rank(method="average", pct=True)
        data["income_rank"] = income.groupby(episodes["t1"]).rank(method="average", pct=True)
        if transition_spec["family"] != "smoking":
            data["smoke_ever"] = pd.to_numeric(episodes["t1__smokev"], errors="coerce").where(lambda x: x.isin([0.0, 1.0]))
            data["smoke_current"] = pd.to_numeric(episodes["t1__smoken"], errors="coerce").where(lambda x: x.isin([0.0, 1.0]))
        else:
            omitted.append("smoking_covariates_overlap_exposure")
        if outcome_id != "multimorbidity_progression":
            data["baseline_multimorbidity"] = multimorbidity(episodes, model_config["common_multimorbidity_fields"])
        else:
            omitted.append("baseline_multimorbidity_duplicates_baseline_outcome")
        behavior_base = baseline_behavior(episodes, transition_id, transition_spec, cohort)
        if behavior_base is not None:
            data["baseline_behavior"] = behavior_base
        else:
            omitted.append("baseline_behavior_constant_within_binary_transition_risk_set")
    data = data.loc[mask].copy()
    candidate_columns = [column for column in data.columns if column not in {"outcome", "person_id", "t1_wave"}]
    complete = data[["outcome", "person_id", "t1_wave", *candidate_columns]].notna().all(axis=1)
    data = data.loc[complete].copy()
    spline = restricted_cubic_spline(data["age"], model_config["age_rcs_knots"], "age")
    X = pd.DataFrame({"intercept": 1.0, "transition": data["transition"]}, index=data.index)
    X = pd.concat([X, spline], axis=1)
    X["sex_female_code2"] = data["sex"].eq(2.0).astype(float)
    wave_dummies = pd.get_dummies(data["t1_wave"].astype(int).astype(str), prefix="t1_wave", drop_first=True, dtype=float)
    X = pd.concat([X, wave_dummies], axis=1)
    if "baseline_outcome" in data:
        X["baseline_outcome"] = data["baseline_outcome"].astype(float)
    if adjustment == "full":
        for field in ["education_rank", "income_rank", "smoke_ever", "smoke_current", "baseline_multimorbidity", "baseline_behavior"]:
            if field in data:
                X[field] = data[field].astype(float)
    audit = {
        "outcome_status": outcome_status,
        "candidate_n": int(mask.sum()),
        "complete_case_n": int(len(data)),
        "omitted_covariates": omitted,
        "design_columns": list(X.columns),
    }
    return X.astype(float), data["outcome"].astype(float), data["person_id"], audit


def model_support(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    transition: pd.Series,
    outcome_spec: dict[str, Any],
    universe: dict[str, Any],
    model_config: dict[str, Any],
) -> tuple[bool, str, dict[str, int]]:
    transitioned = transition.eq(1.0)
    counts = {
        "n": int(len(y)),
        "people": int(groups.nunique()),
        "transitioned_n": int(transitioned.sum()),
        "parameters": int(X.shape[1]),
    }
    if counts["people"] < model_config["minimum_clusters"]:
        return False, "NOT_EVALUABLE_CLUSTERS_LT_MINIMUM", counts
    continuous = outcome_spec["type"] in {"continuous_change", "continuous_change_standardized_within_cohort"}
    if continuous:
        if len(y) < universe["minimum_continuous_episodes"] or transitioned.sum() < universe["minimum_continuous_transitioned"]:
            return False, "NOT_EVALUABLE_CONTINUOUS_COMPLETE_CASE_SUPPORT", counts
        return True, "PASS", counts
    events_transitioned = int(y.loc[transitioned].sum())
    nonevents_transitioned = int(transitioned.sum()) - events_transitioned
    total_events = int(y.sum())
    counts.update(
        {
            "transitioned_events": events_transitioned,
            "transitioned_nonevents": nonevents_transitioned,
            "total_events": total_events,
        }
    )
    if events_transitioned < universe["minimum_binary_transition_events"]:
        return False, "NOT_EVALUABLE_COMPLETE_CASE_TRANSITION_EVENTS", counts
    if nonevents_transitioned < universe["minimum_binary_transition_nonevents"]:
        return False, "NOT_EVALUABLE_COMPLETE_CASE_TRANSITION_NONEVENTS", counts
    if total_events < universe["events_per_parameter"] * X.shape[1]:
        return False, "NOT_EVALUABLE_COMPLETE_CASE_EPV", counts
    return True, "PASS", counts


def fit_one(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    outcome_spec: dict[str, Any],
) -> dict[str, Any]:
    continuous = outcome_spec["type"] in {"continuous_change", "continuous_change_standardized_within_cohort"}
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        if continuous:
            result = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
            estimate = float(result.params["transition"])
            se = float(result.bse["transition"])
            ci_low, ci_high = [float(value) for value in result.conf_int().loc["transition"]]
            outcome_sd = float(y.std(ddof=1))
            return {
                "estimate_scale": "mean_difference_raw_units",
                "estimate": estimate,
                "standard_error": se,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": float(result.pvalues["transition"]),
                "standardized_estimate": estimate / outcome_sd if outcome_sd > 0 else np.nan,
                "standardized_standard_error": se / outcome_sd if outcome_sd > 0 else np.nan,
                "standardized_ci_low": ci_low / outcome_sd if outcome_sd > 0 else np.nan,
                "standardized_ci_high": ci_high / outcome_sd if outcome_sd > 0 else np.nan,
                "marginal_risk_difference": np.nan,
                "marginal_risk_difference_se": np.nan,
                "marginal_risk_difference_ci_low": np.nan,
                "marginal_risk_difference_ci_high": np.nan,
                "converged": True,
                "prediction_range_ok": True,
                "warnings": " | ".join(str(item.message) for item in captured),
            }
        result = sm.GLM(y, X, family=sm.families.Poisson()).fit(
            cov_type="cluster", cov_kwds={"groups": groups}
        )
        coefficient = float(result.params["transition"])
        se_log = float(result.bse["transition"])
        ci_log = result.conf_int().loc["transition"].astype(float)
        if not np.all(np.isfinite([coefficient, se_log, *ci_log.to_list()])) or max(abs(coefficient), abs(float(ci_log.iloc[0])), abs(float(ci_log.iloc[1]))) > 50:
            raise RuntimeError("UNSTABLE_EXTREME_LOG_RR")
        X1 = X.copy()
        X0 = X.copy()
        X1["transition"] = 1.0
        X0["transition"] = 0.0
        beta = result.params.to_numpy(float)
        with np.errstate(over="ignore", invalid="ignore"):
            p1 = np.exp(X1.to_numpy(float) @ beta)
            p0 = np.exp(X0.to_numpy(float) @ beta)
        prediction_range_ok = bool(
            np.all(np.isfinite(p0))
            and np.all(np.isfinite(p1))
            and np.min([p0.min(), p1.min()]) >= 0
            and np.max([p0.max(), p1.max()]) <= 1
        )
        if prediction_range_ok:
            rd = float(np.mean(p1 - p0))
            gradient = np.mean(p1[:, None] * X1.to_numpy(float) - p0[:, None] * X0.to_numpy(float), axis=0)
            rd_var = float(gradient @ result.cov_params().to_numpy(float) @ gradient)
            rd_se = math.sqrt(max(rd_var, 0.0))
        else:
            rd = np.nan
            rd_se = np.nan
        return {
            "estimate_scale": "risk_ratio",
            "estimate": math.exp(coefficient),
            "standard_error": se_log,
            "ci_low": math.exp(float(ci_log.iloc[0])),
            "ci_high": math.exp(float(ci_log.iloc[1])),
            "p_value": float(result.pvalues["transition"]),
            "standardized_estimate": np.nan,
            "standardized_standard_error": np.nan,
            "standardized_ci_low": np.nan,
            "standardized_ci_high": np.nan,
            "marginal_risk_difference": rd,
            "marginal_risk_difference_se": rd_se,
            "marginal_risk_difference_ci_low": rd - 1.96 * rd_se if np.isfinite(rd_se) else np.nan,
            "marginal_risk_difference_ci_high": rd + 1.96 * rd_se if np.isfinite(rd_se) else np.nan,
            "converged": bool(result.converged),
            "prediction_range_ok": prediction_range_ok,
            "warnings": " | ".join(str(item.message) for item in captured),
        }


def main() -> None:
    args = parse_args()
    universe = json.loads(args.universe_config.read_text(encoding="utf-8"))
    model_config = json.loads(args.model_config.read_text(encoding="utf-8"))
    cohort = args.cohort.lower()
    if cohort not in universe["cohorts"]:
        raise RuntimeError(f"unknown cohort: {cohort}")
    if not model_config["frozen_before_first_effect_fit"]:
        raise RuntimeError("model specification not frozen")
    if args.shard_count < 1 or args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise RuntimeError("invalid shard index/count")
    universe["model_covariates_by_cohort"] = model_config["model_covariates_by_cohort"]
    feasibility = pd.read_csv(args.feasibility, dtype=str)
    feasibility = feasibility.loc[feasibility["cohort"].eq(cohort)].copy()
    if feasibility.empty:
        raise RuntimeError(f"{cohort}: no feasibility rows")

    root = Path(universe["release_root"])
    lookup, lookup_path = load_lookup(root, universe)
    formal, formal_audit = read_formal(root, universe, cohort, lookup)
    status, elsa_dates, source_audit = source_status_and_elsa_dates(root, universe, cohort)
    episodes, pair_rows = build_episodes(formal, status, universe, cohort, elsa_dates)

    key_columns = ["scope", "transition_id", "contrast", "transition_state", "comparator_state", "outcome_id"]
    candidate_cells = feasibility.loc[
        feasibility["basic_model_support"].eq("ESTIMABLE") | feasibility["full_model_support"].eq("ESTIMABLE"),
        key_columns,
    ].drop_duplicates().sort_values(key_columns, kind="mergesort").reset_index(drop=True)
    all_candidate_cells_n = len(candidate_cells)
    candidate_cells = candidate_cells.iloc[args.shard_index::args.shard_count].copy()
    output_rows: list[dict[str, Any]] = []
    for cell in candidate_cells.to_dict("records"):
        transition_id = cell["transition_id"]
        outcome_id = cell["outcome_id"]
        transition_spec = universe["behavioral_transitions"][transition_id]
        outcome_spec = universe["outcomes"][outcome_id]
        feasibility_match = feasibility.copy()
        for key in key_columns:
            feasibility_match = feasibility_match.loc[feasibility_match[key].eq(cell[key])]
        if len(feasibility_match) != 1:
            raise RuntimeError(f"{cohort}: nonunique feasibility cell {cell}")
        feas = feasibility_match.iloc[0]
        for adjustment in ("basic", "full"):
            preliminary_status = feas[f"{adjustment}_model_support"]
            base_row = {
                "cohort": cohort,
                **cell,
                "transition_family": transition_spec["family"],
                "outcome_family": outcome_spec["family"],
                "outcome_type": outcome_spec["type"],
                "adjustment": adjustment,
                "preliminary_support": preliminary_status,
            }
            if preliminary_status != "ESTIMABLE":
                output_rows.append({**base_row, "model_status": preliminary_status})
                continue
            try:
                X, y, groups, audit = prepare_design(
                    episodes,
                    universe,
                    model_config,
                    cohort,
                    transition_id,
                    transition_spec,
                    outcome_id,
                    outcome_spec,
                    cell["scope"],
                    cell["transition_state"],
                    cell["comparator_state"],
                    adjustment,
                )
                supported, support_status, counts = model_support(
                    X, y, groups, X["transition"], outcome_spec, universe, model_config
                )
                detail = {
                    **base_row,
                    "model_status": support_status,
                    **counts,
                    "design_columns": json.dumps(audit["design_columns"], ensure_ascii=False),
                    "omitted_covariates": json.dumps(audit["omitted_covariates"], ensure_ascii=False),
                    "candidate_n": audit["candidate_n"],
                }
                if not supported:
                    output_rows.append(detail)
                    continue
                fitted = fit_one(X, y, groups, outcome_spec)
                output_rows.append({**detail, "model_status": "PASS", **fitted})
            except Exception as exc:  # retain the cell and exact failure
                output_rows.append({**base_row, "model_status": "MODEL_FAILURE", "failure_reason": f"{type(exc).__name__}: {exc}"})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(args.output_dir, 0o700)
    shard_label = f"shard-{args.shard_index:02d}-of-{args.shard_count:02d}"
    result_path = args.output_dir / f"{cohort}-{shard_label}-systematic-results.csv"
    pd.DataFrame(output_rows).to_csv(result_path, index=False)
    os.chmod(result_path, 0o600)
    status_counts = pd.Series([row["model_status"] for row in output_rows]).value_counts().to_dict()
    manifest = {
        "analysis_id": model_config["analysis_id"],
        "cohort": cohort,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "all_candidate_cells": all_candidate_cells_n,
        "universe_config_sha256": sha256(args.universe_config),
        "model_config_sha256": sha256(args.model_config),
        "feasibility_sha256": sha256(args.feasibility),
        "lookup_sha256": sha256(lookup_path),
        "parent_universe_commit": model_config["parent_universe_commit"],
        "formal": formal_audit,
        "source": source_audit,
        "eligible_episodes": int(len(episodes)),
        "candidate_cells": int(len(candidate_cells)),
        "model_rows": len(output_rows),
        "model_status_counts": status_counts,
        "results_sha256": sha256(result_path),
        "respondent_rows_exported": 0,
        "negative_results_retained": True,
    }
    manifest_path = args.output_dir / f"{cohort}-{shard_label}-model-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.chmod(manifest_path, 0o600)


if __name__ == "__main__":
    main()
