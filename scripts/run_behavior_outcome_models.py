#!/usr/bin/env python3
"""Baseline-outcome and restricted-cubic-spline helpers."""

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from probe_behavior_outcome_feasibility import max_valid, numeric, outcome_fields, row_count

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

