#!/usr/bin/env python3
"""Shared helpers for the multidomain behavioral-withdrawal pilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from probe_behavior_outcome_feasibility import (
    add_binary_behavior_state,
    behavior_states,
    sha256,
)

from cohort_core import (
    add_base_design,
    coefficient_rows,
    covariate_frame,
    file_sha,
    fit_clustered,
    load_episodes,
    outcome_and_baseline,
    write_frame,
    write_json,
)


def load_specs(universe_path: Path, pilot_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    universe = json.loads(universe_path.read_text())
    pilot = json.loads(pilot_path.read_text())
    if sha256(universe_path) != pilot["parent_universe_sha256"]:
        raise RuntimeError("parent universe config drift")
    if universe["release_root"] != pilot["release_root"]:
        raise RuntimeError("release-root mismatch")
    universe = json.loads(json.dumps(universe))
    universe["minimum_age_at_t1"] = pilot["minimum_age_at_t1"]
    universe["comparable_outcome_window_months"] = pilot["comparable_window_months"]
    universe["model_covariates_by_cohort"] = {
        cohort: ["smokev", "smoken"] for cohort in pilot["candidate_cohorts"]
    }
    return universe, pilot


def multidomain_frame(
    episodes: pd.DataFrame, universe: dict[str, Any], pilot: dict[str, Any], cohort: str
) -> pd.DataFrame:
    cc = universe["cohorts"][cohort]
    result = pd.DataFrame(index=episodes.index)
    baseline_states: dict[str, pd.Series] = {}
    transitions: dict[str, pd.Series] = {}
    names = {
        "alcohol_current": "alcohol",
        "moderate_or_vigorous_activity": "activity",
        "paid_work": "work",
    }
    for transition_id, short in names.items():
        spec = universe["behavioral_transitions"][transition_id]
        transitions[short] = behavior_states(
            episodes, transition_id, spec, cohort, cc["primary_waves"]
        )
        baseline_states[short] = add_binary_behavior_state(episodes, spec, cohort, "t0")
        result[f"{short}_transition"] = transitions[short]
        result[f"{short}_baseline"] = baseline_states[short]
        result[f"{short}_loss"] = transitions[short].eq("1_to_0").astype(float).where(
            transitions[short].notna()
        )
        result[f"{short}_gain"] = transitions[short].eq("0_to_1").astype(float).where(
            transitions[short].notna()
        )

    core_valid = pd.concat(transitions, axis=1).notna().all(axis=1)
    core_valid &= pd.concat(baseline_states, axis=1).notna().all(axis=1)
    result["core_valid"] = core_valid
    result["baseline_engagement_count"] = sum(baseline_states.values()).where(core_valid)
    result["loss_count_core"] = (
        result["alcohol_loss"] + result["activity_loss"] + result["work_loss"]
    ).where(core_valid)
    result["any_withdrawal"] = result["loss_count_core"].ge(1).astype(float).where(core_valid)
    result["loss_1"] = result["loss_count_core"].eq(1).astype(float).where(core_valid)
    result["loss_2plus"] = result["loss_count_core"].ge(2).astype(float).where(core_valid)

    social_spec = universe["behavioral_transitions"]["social_participation"]
    if cohort in social_spec["fields"]:
        social_transition = behavior_states(
            episodes, "social_participation", social_spec, cohort, cc["primary_waves"]
        )
        social0 = add_binary_behavior_state(episodes, social_spec, cohort, "t0")
        social_valid = core_valid & social_transition.notna() & social0.notna()
        result["social_transition"] = social_transition
        result["social_baseline"] = social0
        result["social_loss"] = social_transition.eq("1_to_0").astype(float).where(social_valid)
        result["extended_valid"] = social_valid
        result["baseline_engagement_count_extended"] = (
            result["baseline_engagement_count"] + social0
        ).where(social_valid)
        result["loss_count_extended"] = (
            result["loss_count_core"] + result["social_loss"]
        ).where(social_valid)
        result["extended_loss_1"] = result["loss_count_extended"].eq(1).astype(float).where(social_valid)
        result["extended_loss_2plus"] = result["loss_count_extended"].ge(2).astype(float).where(social_valid)
    else:
        result["extended_valid"] = False

    retirement_spec = universe["behavioral_transitions"]["retirement_status"]
    if cohort in retirement_spec["fields"]:
        retirement1 = add_binary_behavior_state(episodes, retirement_spec, cohort, "t1")
        result["retirement_t1"] = retirement1
        work_risk = core_valid & result["work_baseline"].eq(1) & retirement1.notna()
        result["work_exit_risk"] = work_risk
        result["work_exit_retirement"] = (
            result["work_loss"].eq(1) & retirement1.eq(1)
        ).astype(float).where(work_risk)
        result["work_exit_no_retirement"] = (
            result["work_loss"].eq(1) & retirement1.eq(0)
        ).astype(float).where(work_risk)
    else:
        result["work_exit_risk"] = False
    return result


def scope_mask(episodes: pd.DataFrame, scope: str) -> pd.Series:
    if scope == "comparable_22_30_months":
        return episodes["comparable_window"].fillna(False)
    if scope == "all_primary_wave_intervals":
        return pd.Series(True, index=episodes.index)
    raise ValueError(scope)


def support_gate(
    data: pd.DataFrame,
    y: pd.Series,
    X: pd.DataFrame,
    terms: list[str],
    pilot: dict[str, Any],
    binary_outcome: bool,
) -> tuple[str, dict[str, int]]:
    counts = {
        "n": int(len(data)),
        "people": int(data["person_id"].nunique()),
        "parameters": int(X.shape[1]),
        "matrix_rank": int(np.linalg.matrix_rank(X.to_numpy(float))) if len(X) else 0,
    }
    if counts["people"] < pilot["minimum_clusters"]:
        return "NOT_EVALUABLE_CLUSTERS", counts
    if counts["matrix_rank"] < counts["parameters"]:
        return "NOT_EVALUABLE_RANK", counts
    if binary_outcome:
        counts["events"] = int(y.sum())
        if counts["events"] < pilot["events_per_parameter"] * counts["parameters"]:
            return "NOT_EVALUABLE_EPV", counts
        for term in terms:
            group = data[term].eq(1)
            events = int(y.loc[group].sum())
            nonevents = int(group.sum()) - events
            counts[f"{term}_n"] = int(group.sum())
            counts[f"{term}_events"] = events
            counts[f"{term}_nonevents"] = nonevents
            if events < pilot["minimum_exposed_events"]:
                return f"NOT_EVALUABLE_{term.upper()}_EVENTS", counts
            if nonevents < pilot["minimum_exposed_nonevents"]:
                return f"NOT_EVALUABLE_{term.upper()}_NONEVENTS", counts
    else:
        for term in terms:
            group_n = int(data[term].eq(1).sum())
            counts[f"{term}_n"] = group_n
            if group_n < pilot["minimum_continuous_group_n"]:
                return f"NOT_EVALUABLE_{term.upper()}_N", counts
    return "ESTIMABLE", counts


def prepare_model(
    episodes: pd.DataFrame,
    behavior: pd.DataFrame,
    universe: dict[str, Any],
    pilot: dict[str, Any],
    cohort: str,
    outcome_id: str,
    scope: str,
    adjustment: str,
    exposure_model: str,
    minimum_baseline_opportunities: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str], bool, str]:
    outcome, baseline, coding_status, binary_outcome = outcome_and_baseline(
        episodes, universe, cohort, outcome_id
    )
    behavior_for_covariates = behavior.copy()
    if exposure_model == "extended_score_categorical":
        behavior_for_covariates["baseline_engagement_count"] = behavior[
            "baseline_engagement_count_extended"
        ]
        valid = behavior["extended_valid"]
        terms = ["extended_loss_1", "extended_loss_2plus"]
    elif exposure_model == "work_exit_phenotype":
        valid = behavior["work_exit_risk"]
        terms = ["work_exit_retirement", "work_exit_no_retirement"]
    else:
        minimum_opportunities = (
            pilot["minimum_baseline_opportunities"]
            if minimum_baseline_opportunities is None
            else minimum_baseline_opportunities
        )
        valid = behavior["core_valid"] & behavior["baseline_engagement_count"].ge(
            minimum_opportunities
        )
        terms = {
            "any_withdrawal": ["any_withdrawal"],
            "score_categorical": ["loss_1", "loss_2plus"],
            "mutually_adjusted_components": ["alcohol_loss", "activity_loss", "work_loss"],
        }[exposure_model]

    data = covariate_frame(
        episodes, behavior_for_covariates, universe, cohort, outcome_id
    )
    data["outcome"] = outcome
    if baseline is not None:
        data["baseline_outcome"] = baseline
    for term in terms:
        data[term] = behavior[term]
    selected = scope_mask(episodes, scope) & valid & outcome.notna()
    required = [
        "outcome", "person_id", "t1_wave", "age", "sex", "baseline_engagement_count", *terms,
    ]
    if baseline is not None:
        required.append("baseline_outcome")
    if adjustment == "full":
        required += [
            "education_rank", "income_rank", "smoke_ever", "smoke_current",
            "baseline_multimorbidity",
        ]
    selected &= data[required].notna().all(axis=1)
    data = data.loc[selected].copy()
    X = add_base_design(data, pilot, adjustment)
    for term in terms:
        X[term] = data[term].astype(float)
    return data, data["outcome"].astype(float), X, terms, binary_outcome, coding_status


__all__ = [
    "coefficient_rows", "file_sha", "fit_clustered", "load_episodes", "load_specs",
    "multidomain_frame", "outcome_and_baseline", "prepare_model", "scope_mask",
    "support_gate", "write_frame", "write_json",
]
