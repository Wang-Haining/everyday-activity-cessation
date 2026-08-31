#!/usr/bin/env python3
"""Run gated cohort models for the behavioral-withdrawal frailty extension.

All respondent-level frames, predictions, folds and weights remain in memory on
Quartz. Only aggregate counts, estimates and performance summaries are written.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from probe_behavior_outcome_feasibility import binary, outcome_values
from scipy import stats
from statsmodels.imputation.mice import MICEData

from behavioral_withdrawal_frailty_core import (
    COMMON_DISEASE_FIELDS,
    add_direction_categories,
    build_context_frame,
    build_frailty_frame,
    c_statistic,
    competing_outcome,
    delayed_outcome,
    extend_four_wave,
    first_eligible_mask,
    load_extension_data,
    load_extension_specs,
    load_fi_long,
    load_lookup,
    load_source_components,
    multidomain_frame,
    routine_data_and_design,
    sha256,
    stable_person_fold,
)
from cohort_core import (
    add_base_design,
    coefficient_rows,
    file_sha,
    write_frame,
    write_json,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe-config", required=True, type=Path)
    parser.add_argument("--multidomain-config", required=True, type=Path)
    parser.add_argument("--extension-config", required=True, type=Path)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--design-commit", required=True)
    parser.add_argument("--design-commit-time", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--only-sensitivity", action="store_true")
    return parser.parse_args()


def _full_rank_base(X: pd.DataFrame) -> pd.DataFrame:
    """Deterministically remove constant or redundant routine-predictor columns."""
    if X.empty:
        result = X.copy()
        result.attrs["dropped_redundant_base_columns"] = []
        return result
    keep: list[str] = []
    rank = 0
    for column in X.columns:
        candidate = [*keep, column]
        candidate_rank = int(np.linalg.matrix_rank(X[candidate].to_numpy(float)))
        if candidate_rank > rank:
            keep.append(column)
            rank = candidate_rank
    result = X[keep].copy()
    result.attrs["dropped_redundant_base_columns"] = [column for column in X if column not in keep]
    return result


def _analysis_set(
    episodes: pd.DataFrame,
    behavior: pd.DataFrame,
    universe: dict[str, Any],
    extension: dict[str, Any],
    cohort: str,
    outcome_id: str,
    outcome: pd.Series,
    extra: pd.DataFrame,
    required: list[str],
    comparable: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    data, X = routine_data_and_design(
        episodes, behavior, universe, extension, cohort, outcome_id, outcome
    )
    additions = [column for column in extra.columns if column not in data.columns]
    data = data.join(extra[additions])
    eligible = behavior["core_valid"] & behavior["baseline_engagement_count"].ge(1) & outcome.notna()
    if comparable:
        eligible &= episodes["comparable_window"].fillna(False)
    candidate = X.index.intersection(data.index[eligible])
    complete_fields = list(dict.fromkeys(["outcome", *required]))
    complete = data.loc[candidate, complete_fields].notna().all(axis=1) & X.loc[candidate].notna().all(axis=1)
    index = candidate[complete]
    data = data.loc[index].copy()
    y = pd.to_numeric(data["outcome"], errors="coerce").astype(float)
    X = _full_rank_base(X.loc[index])
    return data, y, X


def _support(
    data: pd.DataFrame,
    y: pd.Series,
    X: pd.DataFrame,
    binary_terms: list[str],
    extension: dict[str, Any],
    minimum_term_events: int | None = None,
) -> tuple[str, dict[str, int]]:
    counts = {
        "n": int(len(data)),
        "people": int(data["person_id"].nunique()),
        "events": int(y.sum()),
        "nonevents": int(len(y) - y.sum()),
        "parameters": int(X.shape[1]),
        "matrix_rank": int(np.linalg.matrix_rank(X.to_numpy(float))) if len(X) else 0,
    }
    if counts["people"] < extension["minimum_clusters"]:
        return "NOT_EVALUABLE_CLUSTERS", counts
    if counts["matrix_rank"] < counts["parameters"]:
        return "NOT_EVALUABLE_RANK", counts
    if counts["events"] < extension["events_per_parameter"] * counts["parameters"]:
        return "NOT_EVALUABLE_EPV", counts
    threshold = extension["minimum_exposed_events"] if minimum_term_events is None else minimum_term_events
    for term in binary_terms:
        exposed = data[term].eq(1)
        events = int(y.loc[exposed].sum())
        nonevents = int(exposed.sum()) - events
        counts[f"{term}_n"] = int(exposed.sum())
        counts[f"{term}_events"] = events
        counts[f"{term}_nonevents"] = nonevents
        if events < threshold:
            return f"NOT_EVALUABLE_{term.upper()}_EVENTS", counts
        if nonevents < extension["minimum_exposed_nonevents"]:
            return f"NOT_EVALUABLE_{term.upper()}_NONEVENTS", counts
    return "ESTIMABLE", counts


def _fit(
    data: pd.DataFrame,
    y: pd.Series,
    X: pd.DataFrame,
) -> tuple[Any, str]:
    result = sm.GLM(y, X, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": data["person_id"]}
    )
    return result, ""


def _run_model(
    base: dict[str, Any],
    data: pd.DataFrame,
    y: pd.Series,
    Xbase: pd.DataFrame,
    model_terms: list[str],
    binary_terms: list[str],
    extension: dict[str, Any],
    minimum_term_events: int | None = None,
) -> tuple[list[dict[str, Any]], Any | None, pd.DataFrame | None]:
    X = Xbase.copy()
    for term in model_terms:
        X[term] = pd.to_numeric(data[term], errors="coerce").astype(float)
    status, counts = _support(data, y, X, binary_terms, extension, minimum_term_events)
    if status != "ESTIMABLE":
        return [{**base, **counts, "model_status": status}], None, None
    try:
        fit, warning_text = _fit(data, y, X)
        estimates = coefficient_rows(fit, model_terms, True)
        rows = [
            {**base, **counts, "model_status": "PASS", "warnings": warning_text, **row}
            for row in estimates
        ]
        if not rows:
            rows = [{**base, **counts, "model_status": "PASS", "warnings": warning_text, "term": "__model__"}]
        return rows, fit, X
    except Exception as exc:
        return [{**base, **counts, "model_status": "MODEL_FAILURE", "failure_reason": f"{type(exc).__name__}: {exc}"}], None, None


def _extra_frame(behavior: pd.DataFrame, frailty: pd.DataFrame | None) -> pd.DataFrame:
    frame = pd.DataFrame(index=behavior.index)
    for field in [
        "loss_1", "loss_2plus", "any_withdrawal", "transition_direction",
        "alcohol_loss", "activity_loss", "work_loss",
    ]:
        frame[field] = behavior[field]
    frame["two_domain_any"] = (
        behavior["alcohol_loss"].eq(1) | behavior["activity_loss"].eq(1)
    ).astype(float).where(behavior["core_valid"])
    for category in ["expansion", "mixed", "contraction_1", "contraction_2_plus"]:
        frame[f"direction_{category}"] = behavior["transition_direction"].eq(category).astype(float).where(
            behavior["transition_direction"].notna()
        )
    if frailty is not None:
        frame["prefrail"] = frailty["fried_category"].eq("prefrail").astype(float).where(
            frailty["fried_category"].notna()
        )
        frame["frail"] = frailty["fried_category"].eq("frail").astype(float).where(
            frailty["fried_category"].notna()
        )
        frame["fried4"] = frailty["fried4_no_activity_t1"]
        frame["fi_per_0_1"] = frailty["fi_t1"] / 0.1
        frame["withdrawal_only"] = (
            behavior["any_withdrawal"].eq(1) & frailty["fried_category"].ne("frail")
        ).astype(float).where(frailty["fried_category"].notna())
        frame["frailty_only"] = (
            behavior["any_withdrawal"].eq(0) & frailty["fried_category"].eq("frail")
        ).astype(float).where(frailty["fried_category"].notna())
        frame["both"] = (
            behavior["any_withdrawal"].eq(1) & frailty["fried_category"].eq("frail")
        ).astype(float).where(frailty["fried_category"].notna())
    return frame


def specificity_models(
    episodes: pd.DataFrame,
    behavior: pd.DataFrame,
    universe: dict[str, Any],
    extension: dict[str, Any],
    cohort: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    extra = _extra_frame(behavior, None)
    models = {
        "any_withdrawal": (["any_withdrawal"], ["any_withdrawal"]),
        "two_domain_any_current_adjusted": (["two_domain_any"], ["two_domain_any"]),
        "domain_specific_current_adjusted": (
            ["alcohol_loss", "activity_loss", "work_loss"],
            ["alcohol_loss", "activity_loss", "work_loss"],
        ),
        "withdrawal_gradient": (["loss_1", "loss_2plus"], ["loss_1", "loss_2plus"]),
        "direction_symmetry": (
            ["direction_expansion", "direction_mixed", "direction_contraction_1", "direction_contraction_2_plus"],
            ["direction_expansion", "direction_mixed", "direction_contraction_1", "direction_contraction_2_plus"],
        ),
    }
    for outcome_id in extension["specificity_outcomes"]:
        outcome, coding = outcome_values(
            episodes, outcome_id, universe["outcomes"][outcome_id], universe["outcomes"], cohort
        )
        for model_id, (terms, binary_terms) in models.items():
            data, y, X = _analysis_set(
                episodes, behavior, universe, extension, cohort, outcome_id, outcome, extra, terms
            )
            result, _, _ = _run_model(
                {"cohort": cohort, "outcome_id": outcome_id, "analysis_family": "specificity", "model_id": model_id, "outcome_coding_status": coding},
                data, y, X, terms, binary_terms, extension,
            )
            rows.extend(result)
        for model_id, terms in [
            ("three_domain_any_without_current_states", ["any_withdrawal"]),
            ("two_domain_any_without_current_states", ["two_domain_any"]),
            ("domain_specific_without_current_states", ["alcohol_loss", "activity_loss", "work_loss"]),
        ]:
            data, y, X = _analysis_set(
                episodes, behavior, universe, extension, cohort, outcome_id, outcome, extra, terms
            )
            X = X.drop(columns=[column for column in ["alcohol_t1", "activity_t1", "work_t1"] if column in X])
            result, _, _ = _run_model(
                {"cohort": cohort, "outcome_id": outcome_id, "analysis_family": "current_state_decomposition", "model_id": model_id, "outcome_coding_status": coding},
                data, y, X, terms, terms, extension,
            )
            rows.extend(result)
    return rows


def _model_definitions() -> dict[str, tuple[list[str], list[str]]]:
    return {
        "M0_routine": ([], []),
        "M1_withdrawal": (["loss_1", "loss_2plus"], ["loss_1", "loss_2plus"]),
        "M2_fried": (["prefrail", "frail"], ["prefrail", "frail"]),
        "M3_fried_withdrawal": (
            ["prefrail", "frail", "loss_1", "loss_2plus"],
            ["prefrail", "frail", "loss_1", "loss_2plus"],
        ),
    }


def _binary_model_definitions() -> dict[str, tuple[list[str], list[str]]]:
    return {
        "M0_routine": ([], []),
        "M1b_any_withdrawal": (["any_withdrawal"], ["any_withdrawal"]),
        "M2_fried": (["prefrail", "frail"], ["prefrail", "frail"]),
        "M3b_fried_any_withdrawal": (
            ["prefrail", "frail", "any_withdrawal"],
            ["prefrail", "frail", "any_withdrawal"],
        ),
    }


def _bootstrap_metrics(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    people: np.ndarray,
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    unique_people = np.unique(people)
    person_rows = {person: np.flatnonzero(people == person) for person in unique_people}
    rng = np.random.default_rng(seed)
    metrics: dict[tuple[str, str], list[float]] = {}
    for _ in range(replicates):
        sampled = rng.choice(unique_people, size=len(unique_people), replace=True)
        index = np.concatenate([person_rows[person] for person in sampled])
        yy = y[index]
        if yy.min() == yy.max():
            continue
        for model, prediction in predictions.items():
            metrics.setdefault((model, "c_statistic"), []).append(c_statistic(yy, prediction[index]))
            metrics.setdefault((model, "brier_score"), []).append(float(np.mean((yy - prediction[index]) ** 2)))
    rows = []
    for (model, metric), values in metrics.items():
        array = np.asarray(values, dtype=float)
        rows.append({
            "model_id": model,
            "metric": metric,
            "bootstrap_replicates": int(len(array)),
            "ci_low": float(np.quantile(array, 0.025)),
            "ci_high": float(np.quantile(array, 0.975)),
        })
    model_ids = list(predictions)
    routine = model_ids[0]
    fried = model_ids[2]
    pairs = [(routine, model_ids[1]), (routine, fried), (routine, model_ids[3]), (fried, model_ids[3])]
    for comparator, enhanced in pairs:
        for metric in ["c_statistic", "brier_score"]:
            difference = np.asarray(metrics[(enhanced, metric)]) - np.asarray(metrics[(comparator, metric)])
            rows.append({
                "model_id": f"{enhanced}_minus_{comparator}",
                "metric": f"delta_{metric}",
                "bootstrap_replicates": int(len(difference)),
                "ci_low": float(np.quantile(difference, 0.025)),
                "ci_high": float(np.quantile(difference, 0.975)),
            })
    return rows


def cross_validated_performance(
    data: pd.DataFrame,
    y: pd.Series,
    Xbase: pd.DataFrame,
    extension: dict[str, Any],
    cohort: str,
    outcome_id: str,
    definitions: dict[str, tuple[list[str], list[str]]] | None = None,
    analysis_family: str = "head_to_head_cv",
) -> list[dict[str, Any]]:
    definitions = _model_definitions() if definitions is None else definitions
    joint_model = list(definitions)[-1]
    largest_terms = definitions[joint_model][0]
    Xfull = Xbase.copy()
    for term in largest_terms:
        Xfull[term] = data[term].astype(float)
    status, counts = _support(
        data, y, Xfull, definitions[joint_model][1], extension
    )
    base = {"cohort": cohort, "outcome_id": outcome_id, "analysis_family": analysis_family, **counts}
    if status != "ESTIMABLE" or counts["events"] < extension["minimum_cv_events"]:
        return [{**base, "model_id": model, "model_status": "NOT_EVALUABLE_CV_SUPPORT"} for model in definitions]
    withdrawal_events = int(y.loc[data["any_withdrawal"].eq(1)].sum())
    if withdrawal_events < extension["minimum_cv_withdrawal_events"]:
        return [{**base, "model_id": model, "model_status": "NOT_EVALUABLE_CV_WITHDRAWAL_EVENTS"} for model in definitions]
    folds = data["person_id"].map(lambda person: stable_person_fold(cohort, str(person), extension["cv_folds"]))
    predictions = {model: np.full(len(data), np.nan, dtype=float) for model in definitions}
    for fold in range(extension["cv_folds"]):
        train = folds.ne(fold).to_numpy()
        test = folds.eq(fold).to_numpy()
        if not test.any():
            continue
        for model, (terms, _) in definitions.items():
            X = Xbase.copy()
            for term in terms:
                X[term] = data[term].astype(float)
            if np.linalg.matrix_rank(X.loc[train].to_numpy(float)) < X.shape[1]:
                raise RuntimeError(f"{cohort} {outcome_id} {model}: training design rank deficient")
            fit = sm.GLM(y.loc[train], X.loc[train], family=sm.families.Poisson()).fit()
            predictions[model][test] = np.clip(np.asarray(fit.predict(X.loc[test]), dtype=float), 0, 1)
    if any(np.isnan(value).any() for value in predictions.values()):
        raise RuntimeError(f"{cohort} {outcome_id}: incomplete grouped cross-validation")
    output = []
    bootstrap = _bootstrap_metrics(
        y.to_numpy(float), predictions, data["person_id"].to_numpy(), 500,
        int.from_bytes(f"{cohort}|{outcome_id}".encode(), "little") % (2**32),
    )
    bootstrap_lookup = {(row["model_id"], row["metric"]): row for row in bootstrap}
    point: dict[tuple[str, str], float] = {}
    for model, prediction in predictions.items():
        point[(model, "c_statistic")] = c_statistic(y.to_numpy(float), prediction)
        point[(model, "brier_score")] = float(np.mean((y.to_numpy(float) - prediction) ** 2))
        for metric in ["c_statistic", "brier_score"]:
            ci = bootstrap_lookup[(model, metric)]
            output.append({
                **base, "model_id": model, "model_status": "PASS", "metric": metric,
                "estimate": point[(model, metric)], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                "bootstrap_replicates": ci["bootstrap_replicates"],
            })
    model_ids = list(definitions)
    for comparator, enhanced in [
        (model_ids[0], model_ids[1]),
        (model_ids[0], model_ids[2]),
        (model_ids[0], model_ids[3]),
        (model_ids[2], model_ids[3]),
    ]:
        for metric in ["c_statistic", "brier_score"]:
            ci = bootstrap_lookup[(f"{enhanced}_minus_{comparator}", f"delta_{metric}")]
            output.append({
                **base,
                "model_id": f"{enhanced}_minus_{comparator}",
                "model_status": "PASS",
                "metric": f"delta_{metric}",
                "estimate": point[(enhanced, metric)] - point[(comparator, metric)],
                "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                "bootstrap_replicates": ci["bootstrap_replicates"],
            })
    return output


def frailty_models(
    episodes: pd.DataFrame,
    behavior: pd.DataFrame,
    frailty: pd.DataFrame,
    universe: dict[str, Any],
    extension: dict[str, Any],
    cohort: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    association_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    extra = _extra_frame(behavior, frailty)
    definitions = _model_definitions()
    extended_definitions = {
        **definitions,
        "M1b_any_withdrawal": (["any_withdrawal"], ["any_withdrawal"]),
        "M3b_fried_any_withdrawal": (
            ["prefrail", "frail", "any_withdrawal"],
            ["prefrail", "frail", "any_withdrawal"],
        ),
        "M4_fried4_withdrawal": (["fried4", "loss_1", "loss_2plus"], ["loss_1", "loss_2plus"]),
        "M5_fi_withdrawal": (["fi_per_0_1", "loss_1", "loss_2plus"], ["loss_1", "loss_2plus"]),
        "nonfrail_withdrawal": (["loss_1", "loss_2plus"], ["loss_1", "loss_2plus"]),
        "nonfrail_any_withdrawal": (["any_withdrawal"], ["any_withdrawal"]),
        "four_state": (["withdrawal_only", "frailty_only", "both"], ["withdrawal_only", "frailty_only", "both"]),
    }
    for outcome_id in extension["specificity_outcomes"]:
        outcome, coding = outcome_values(
            episodes, outcome_id, universe["outcomes"][outcome_id], universe["outcomes"], cohort
        )
        common_required = ["prefrail", "frail", "loss_1", "loss_2plus"]
        common_data, common_y, common_X = _analysis_set(
            episodes, behavior, universe, extension, cohort, outcome_id, outcome, extra, common_required
        )
        for model_id, (terms, binary_terms) in extended_definitions.items():
            if model_id in {"nonfrail_withdrawal", "nonfrail_any_withdrawal"}:
                selected = common_data["frail"].eq(0)
                data, y, X = common_data.loc[selected], common_y.loc[selected], common_X.loc[selected]
            elif model_id == "M4_fried4_withdrawal":
                data, y, X = _analysis_set(
                    episodes, behavior, universe, extension, cohort, outcome_id, outcome, extra,
                    ["fried4", "loss_1", "loss_2plus"],
                )
            elif model_id == "M5_fi_withdrawal":
                data, y, X = _analysis_set(
                    episodes, behavior, universe, extension, cohort, outcome_id, outcome, extra,
                    ["fi_per_0_1", "loss_1", "loss_2plus"],
                )
            else:
                data, y, X = common_data, common_y, common_X
            terms_for_fit = terms
            minimum = extension["minimum_cell_events"] if model_id == "four_state" else None
            result, fit, design = _run_model(
                {"cohort": cohort, "outcome_id": outcome_id, "analysis_family": "frailty_head_to_head", "model_id": model_id, "outcome_coding_status": coding},
                data, y, X, terms_for_fit, binary_terms, extension, minimum,
            )
            association_rows.extend(result)
            if model_id == "four_state" and fit is not None and design is not None:
                for state, values in {
                    "neither": {"withdrawal_only": 0, "frailty_only": 0, "both": 0},
                    "withdrawal_only": {"withdrawal_only": 1, "frailty_only": 0, "both": 0},
                    "frailty_only": {"withdrawal_only": 0, "frailty_only": 1, "both": 0},
                    "both": {"withdrawal_only": 0, "frailty_only": 0, "both": 1},
                }.items():
                    scenario = design.copy()
                    for field, value in values.items():
                        scenario[field] = value
                    prediction = np.clip(np.asarray(fit.predict(scenario), dtype=float), 0, 1)
                    risk_rows.append({
                        "cohort": cohort, "outcome_id": outcome_id, "risk_model": "four_state",
                        "state": state, "standardized_risk": float(prediction.mean()), "n": int(len(prediction)),
                    })
            elif model_id == "four_state":
                risk_rows.append({
                    "cohort": cohort, "outcome_id": outcome_id, "risk_model": "four_state",
                    "state": "NOT_EVALUABLE", "model_status": result[0]["model_status"],
                })

        if not common_data.empty:
            flags = {
                "withdrawal": common_data["any_withdrawal"].eq(1),
                "frail": common_data["frail"].eq(1),
                "either": common_data["any_withdrawal"].eq(1) | common_data["frail"].eq(1),
                "both": common_data["any_withdrawal"].eq(1) & common_data["frail"].eq(1),
            }
            for flag, selected in flags.items():
                coverage_rows.append({
                    "cohort": cohort, "outcome_id": outcome_id, "flag": flag,
                    "analysis_n": int(len(common_data)), "events": int(common_y.sum()),
                    "flag_n": int(selected.sum()), "flag_events": int(common_y.loc[selected].sum()),
                    "population_coverage": float(selected.mean()),
                    "event_coverage": float(common_y.loc[selected].sum() / common_y.sum()) if common_y.sum() else float("nan"),
                    "flag_risk": float(common_y.loc[selected].mean()) if selected.any() else float("nan"),
                })
        try:
            performance_rows.extend(
                cross_validated_performance(common_data, common_y, common_X, extension, cohort, outcome_id)
            )
            performance_rows.extend(
                cross_validated_performance(
                    common_data, common_y, common_X, extension, cohort, outcome_id,
                    definitions=_binary_model_definitions(), analysis_family="head_to_head_cv_binary_flag",
                )
            )
        except Exception as exc:
            performance_rows.append({
                "cohort": cohort, "outcome_id": outcome_id, "analysis_family": "head_to_head_cv",
                "model_id": "all", "model_status": "MODEL_FAILURE", "failure_reason": f"{type(exc).__name__}: {exc}",
            })
    return association_rows, performance_rows, risk_rows, coverage_rows


def _target_specific_context(
    episodes: pd.DataFrame,
    context: pd.DataFrame,
    outcome_id: str,
) -> pd.DataFrame:
    result = context.copy()
    target = COMMON_DISEASE_FIELDS[outcome_id]
    incident = []
    for field in COMMON_DISEASE_FIELDS.values():
        if field == target:
            continue
        at0 = binary(episodes[f"t0__{field}"])
        at1 = binary(episodes[f"t1__{field}"])
        incident.append((at0.eq(0) & at1.eq(1)).astype(float).where(at0.notna() & at1.notna()))
    frame = pd.concat(incident, axis=1)
    result["incident_non_target_disease"] = frame.eq(1).any(axis=1).astype(float).where(
        frame.notna().all(axis=1)
    )
    return result


def context_models(
    episodes: pd.DataFrame,
    behavior: pd.DataFrame,
    context: pd.DataFrame,
    universe: dict[str, Any],
    extension: dict[str, Any],
    cohort: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    extra_base = _extra_frame(behavior, None)
    context_fields = [
        "self_rated_health_worsening", "transition_hospitalization", "cesd_worsening",
        "bmi_or_weight_loss", "incident_non_target_disease", "fi_change",
    ]
    for outcome_id in extension["specificity_outcomes"]:
        outcome, coding = outcome_values(
            episodes, outcome_id, universe["outcomes"][outcome_id], universe["outcomes"], cohort
        )
        target_context = _target_specific_context(episodes, context, outcome_id)
        extra = pd.concat([extra_base, target_context], axis=1)
        available = [field for field in context_fields if extra[field].notna().any()]
        for context_id, fields in [(field, [field]) for field in available] + [("joint_context", available)]:
            required = ["any_withdrawal", *fields]
            data, y, X = _analysis_set(
                episodes, behavior, universe, extension, cohort, outcome_id, outcome, extra, required
            )
            base_rows, base_fit, _ = _run_model(
                {"cohort": cohort, "outcome_id": outcome_id, "analysis_family": "clinical_context", "model_id": f"base_same_set_{context_id}", "outcome_coding_status": coding},
                data, y, X, ["any_withdrawal"], ["any_withdrawal"], extension,
            )
            rows.extend(base_rows)
            X_context = X.copy()
            for field in fields:
                scale = 0.05 if field == "fi_change" else 1.0
                data[f"context_{field}"] = pd.to_numeric(data[field], errors="coerce") / scale
            context_terms = [f"context_{field}" for field in fields]
            context_rows_result, context_fit, _ = _run_model(
                {"cohort": cohort, "outcome_id": outcome_id, "analysis_family": "clinical_context", "model_id": f"plus_{context_id}", "outcome_coding_status": coding},
                data, y, X_context, ["any_withdrawal", *context_terms], ["any_withdrawal"], extension,
            )
            if base_fit is not None and context_fit is not None:
                beta0 = float(base_fit.params["any_withdrawal"])
                beta1 = float(context_fit.params["any_withdrawal"])
                attenuation = float((beta0 - beta1) / beta0 * 100) if abs(beta0) > 1e-12 else float("nan")
                for row in context_rows_result:
                    row["withdrawal_log_rr_attenuation_percent"] = attenuation
            rows.extend(context_rows_result)
    return rows


def delayed_models(
    episodes: pd.DataFrame,
    formal: pd.DataFrame,
    status: pd.DataFrame,
    behavior: pd.DataFrame,
    universe: dict[str, Any],
    extension: dict[str, Any],
    cohort: str,
) -> list[dict[str, Any]]:
    rows = []
    episodes4 = extend_four_wave(episodes, formal, status, universe, cohort)
    extra = _extra_frame(behavior, None)
    for outcome_id in extension["delayed_outcomes"]:
        outcome = delayed_outcome(episodes4, outcome_id)
        for model_id, terms in [("delayed_any", ["any_withdrawal"]), ("delayed_gradient", ["loss_1", "loss_2plus"])]:
            data, y, X = _analysis_set(
                episodes4, behavior, universe, extension, cohort, outcome_id, outcome, extra, terms,
                comparable=False,
            )
            result, _, _ = _run_model(
                {"cohort": cohort, "outcome_id": outcome_id, "analysis_family": "delayed_outcome", "model_id": model_id},
                data, y, X, terms, terms, extension,
            )
            rows.extend(result)
    return rows


def sensitivity_models(
    episodes: pd.DataFrame,
    behavior: pd.DataFrame,
    universe: dict[str, Any],
    extension: dict[str, Any],
    cohort: str,
) -> list[dict[str, Any]]:
    rows = []
    extra = _extra_frame(behavior, None)
    for outcome_id in extension["specificity_outcomes"]:
        primary, coding = outcome_values(
            episodes, outcome_id, universe["outcomes"][outcome_id], universe["outcomes"], cohort
        )
        composite, _ = competing_outcome(episodes, universe, cohort, outcome_id)
        for model_id, outcome in [("diagnosis_or_death", composite), ("first_eligible_interval", primary)]:
            data, y, X = _analysis_set(
                episodes, behavior, universe, extension, cohort, outcome_id, outcome, extra,
                ["any_withdrawal"],
            )
            if model_id == "first_eligible_interval":
                selected = first_eligible_mask(
                    episodes.loc[data.index], pd.Series(True, index=data.index)
                )
                data, y, X = data.loc[selected], y.loc[selected], X.loc[selected]
            result, _, _ = _run_model(
                {"cohort": cohort, "outcome_id": outcome_id, "analysis_family": "sensitivity", "model_id": model_id, "outcome_coding_status": coding},
                data, y, X, ["any_withdrawal"], ["any_withdrawal"], extension,
            )
            rows.extend(result)
        rows.extend(_ipcw_model(
            episodes, behavior, universe, extension, cohort, outcome_id, primary, coding
        ))
        if cohort in extension["multiple_imputation_cohorts"]:
            rows.extend(_multiple_imputation_model(
                episodes, behavior, universe, extension, cohort, outcome_id, primary, coding
            ))
    return rows


def _ipcw_model(
    episodes: pd.DataFrame,
    behavior: pd.DataFrame,
    universe: dict[str, Any],
    extension: dict[str, Any],
    cohort: str,
    outcome_id: str,
    outcome: pd.Series,
    coding: str,
) -> list[dict[str, Any]]:
    data, X = routine_data_and_design(
        episodes, behavior, universe, extension, cohort, outcome_id, outcome
    )
    field = COMMON_DISEASE_FIELDS[outcome_id]
    disease_free = binary(episodes[f"t1__{field}"]).eq(0)
    eligible = (
        episodes["comparable_window"].fillna(False)
        & behavior["core_valid"]
        & behavior["baseline_engagement_count"].ge(1)
        & behavior["any_withdrawal"].notna()
        & disease_free
    )
    candidate = X.index.intersection(data.index[eligible])
    complete = X.loc[candidate].notna().all(axis=1)
    index = candidate[complete]
    if not len(index):
        return [{
            "cohort": cohort, "outcome_id": outcome_id, "analysis_family": "sensitivity",
            "model_id": "stabilized_ipcw", "model_status": "NOT_EVALUABLE_EMPTY",
        }]
    data = data.loc[index].copy()
    X = _full_rank_base(X.loc[index])
    X["any_withdrawal"] = behavior.loc[index, "any_withdrawal"].astype(float)
    observed = outcome.loc[index].notna().astype(float)
    base = {
        "cohort": cohort, "outcome_id": outcome_id, "analysis_family": "sensitivity",
        "model_id": "stabilized_ipcw", "outcome_coding_status": coding,
        "risk_set_n": int(len(index)), "observed_n": int(observed.sum()),
        "unknown_n": int(len(observed) - observed.sum()),
    }
    try:
        if observed.nunique() < 2:
            return [{**base, "model_status": "NOT_EVALUABLE_NO_OBSERVATION_VARIATION"}]
        denominator = sm.GLM(observed, X, family=sm.families.Binomial()).fit()
        wave_columns = [column for column in X if column == "intercept" or column.startswith("t1_wave_")]
        numerator = sm.GLM(observed, X[wave_columns], family=sm.families.Binomial()).fit()
        p_denominator = np.clip(np.asarray(denominator.predict(X), dtype=float), 0.01, 0.99)
        p_numerator = np.clip(np.asarray(numerator.predict(X[wave_columns]), dtype=float), 0.01, 0.99)
        weights = pd.Series(p_numerator / p_denominator, index=index)
        observed_index = index[observed.eq(1)]
        weights = weights.loc[observed_index]
        low, high = np.quantile(weights, np.asarray(extension["ipcw_truncation_percentiles"]) / 100)
        weights = weights.clip(low, high)
        model_data = data.loc[observed_index]
        y = outcome.loc[observed_index].astype(float)
        Xoutcome = X.loc[observed_index]
        status, counts = _support(
            model_data.assign(any_withdrawal=behavior.loc[observed_index, "any_withdrawal"]),
            y, Xoutcome, ["any_withdrawal"], extension,
        )
        diagnostics = {
            **counts,
            "weight_mean": float(weights.mean()),
            "weight_min": float(weights.min()),
            "weight_max": float(weights.max()),
            "weight_effective_n": float(weights.sum() ** 2 / (weights.pow(2).sum())),
            "weight_truncation_low": float(low),
            "weight_truncation_high": float(high),
        }
        if status != "ESTIMABLE":
            return [{**base, **diagnostics, "model_status": status}]
        fit = sm.GEE(
            y.reset_index(drop=True), Xoutcome.reset_index(drop=True),
            groups=model_data["person_id"].reset_index(drop=True),
            family=sm.families.Poisson(), weights=weights.reset_index(drop=True),
        ).fit()
        return [
            {**base, **diagnostics, "model_status": "PASS", **row}
            for row in coefficient_rows(fit, ["any_withdrawal"], True)
        ]
    except Exception as exc:
        return [{**base, "model_status": "MODEL_FAILURE", "failure_reason": f"{type(exc).__name__}: {exc}"}]


def _multiple_imputation_model(
    episodes: pd.DataFrame,
    behavior: pd.DataFrame,
    universe: dict[str, Any],
    extension: dict[str, Any],
    cohort: str,
    outcome_id: str,
    outcome: pd.Series,
    coding: str,
) -> list[dict[str, Any]]:
    data, _ = routine_data_and_design(
        episodes, behavior, universe, extension, cohort, outcome_id, outcome
    )
    data["any_withdrawal"] = behavior["any_withdrawal"]
    eligible = (
        episodes["comparable_window"].fillna(False)
        & behavior["core_valid"]
        & behavior["baseline_engagement_count"].ge(1)
        & outcome.notna()
        & data[["age", "sex", "t1_wave", "baseline_engagement_count", "any_withdrawal"]].notna().all(axis=1)
    )
    fields = [
        "outcome", "any_withdrawal", "age", "sex", "t1_wave", "baseline_engagement_count",
        "education_rank", "income_rank", "smoke_ever", "smoke_current", "baseline_multimorbidity",
        "alcohol_t1", "activity_t1", "work_t1",
    ]
    frame = data.loc[eligible, fields].copy()
    base = {
        "cohort": cohort, "outcome_id": outcome_id, "analysis_family": "sensitivity",
        "model_id": "multiple_imputation_covariates", "outcome_coding_status": coding,
        "n": int(len(frame)), "people": int(data.loc[eligible, "person_id"].nunique()),
        "events": int(frame["outcome"].sum()), "imputations": int(extension["multiple_imputation_m"]),
    }
    if len(frame) == 0:
        return [{**base, "model_status": "NOT_EVALUABLE_EMPTY"}]
    missing = int(frame.isna().sum().sum())
    base["ordinary_covariate_missing_cells"] = missing
    try:
        mice = MICEData(frame, perturbation_method="gaussian", k_pmm=20)
        for _ in range(10):
            mice.update_all()
        betas: list[float] = []
        variances: list[float] = []
        for _ in range(extension["multiple_imputation_m"]):
            mice.update_all()
            imputed = mice.data.copy()
            X = add_base_design(imputed, extension, "full")
            for field in ["alcohol_t1", "activity_t1", "work_t1", "any_withdrawal"]:
                X[field] = imputed[field].astype(float)
            X = _full_rank_base(X)
            fit = sm.GLM(imputed["outcome"], X, family=sm.families.Poisson()).fit(
                cov_type="cluster", cov_kwds={"groups": data.loc[eligible, "person_id"]}
            )
            betas.append(float(fit.params["any_withdrawal"]))
            variances.append(float(fit.bse["any_withdrawal"]) ** 2)
        m = len(betas)
        beta = float(np.mean(betas))
        within = float(np.mean(variances))
        between = float(np.var(betas, ddof=1)) if m > 1 else 0.0
        total = within + (1 + 1 / m) * between
        relative = (1 + 1 / m) * between / within if within > 0 else 0.0
        degrees = (m - 1) * (1 + 1 / relative) ** 2 if relative > 0 else float("inf")
        critical = float(stats.t.ppf(0.975, degrees)) if math.isfinite(degrees) else 1.959963984540054
        se = math.sqrt(total)
        return [{
            **base, "model_status": "PASS", "term": "any_withdrawal", "estimate_scale": "risk_ratio",
            "estimate": math.exp(beta), "standard_error": se,
            "ci_low": math.exp(beta - critical * se), "ci_high": math.exp(beta + critical * se),
            "rubin_within_variance": within, "rubin_between_variance": between,
            "rubin_degrees_freedom": degrees,
        }]
    except Exception as exc:
        return [{**base, "model_status": "MODEL_FAILURE", "failure_reason": f"{type(exc).__name__}: {exc}"}]


def main() -> None:
    args = arguments()
    universe, multidomain, extension = load_extension_specs(
        args.universe_config, args.multidomain_config, args.extension_config
    )
    cohort = args.cohort.lower()
    probe_manifest_path = args.probe_dir / f"{cohort}-probe-manifest.json"
    if not probe_manifest_path.exists():
        raise RuntimeError(f"{cohort}: counts-only probe manifest missing")
    probe_manifest = json.loads(probe_manifest_path.read_text())
    if probe_manifest.get("effect_models_fit") != 0 or not probe_manifest.get("aggregate_only"):
        raise RuntimeError(f"{cohort}: invalid probe manifest")
    root = Path(extension["release_root"])
    lookup, lookup_path = load_lookup(root, universe)
    episodes, formal, status, _, input_audit = load_extension_data(root, universe, cohort, lookup)
    behavior = add_direction_categories(multidomain_frame(episodes, universe, multidomain, cohort))
    frailty = None
    context = None
    source_audit: dict[str, Any] = {}
    fi_audit: dict[str, Any] = {}
    if cohort in extension["frailty_cohorts"]:
        source, source_audit = load_source_components(root, extension, cohort)
        fi_long, fi_audit = load_fi_long(extension, cohort)
        frailty = build_frailty_frame(episodes, formal, source, fi_long, universe, extension, cohort)
        context = build_context_frame(episodes, frailty, universe, cohort)

    outputs: dict[str, Path] = {}
    outputs["sensitivity"] = args.output_dir / f"{cohort}-sensitivity-models.csv"
    write_frame(outputs["sensitivity"], sensitivity_models(episodes, behavior, universe, extension, cohort))
    if not args.only_sensitivity:
        outputs["specificity"] = args.output_dir / f"{cohort}-specificity-models.csv"
        outputs["delayed"] = args.output_dir / f"{cohort}-delayed-models.csv"
        write_frame(outputs["specificity"], specificity_models(episodes, behavior, universe, extension, cohort))
        write_frame(outputs["delayed"], delayed_models(episodes, formal, status, behavior, universe, extension, cohort))
    if not args.only_sensitivity and frailty is not None and context is not None:
        association, performance, risks, coverage = frailty_models(
            episodes, behavior, frailty, universe, extension, cohort
        )
        outputs["frailty"] = args.output_dir / f"{cohort}-frailty-models.csv"
        outputs["performance"] = args.output_dir / f"{cohort}-cv-performance.csv"
        outputs["risks"] = args.output_dir / f"{cohort}-standardized-risks.csv"
        outputs["coverage"] = args.output_dir / f"{cohort}-event-coverage.csv"
        outputs["context"] = args.output_dir / f"{cohort}-context-models.csv"
        write_frame(outputs["frailty"], association)
        write_frame(outputs["performance"], performance)
        write_frame(outputs["risks"], risks)
        write_frame(outputs["coverage"], coverage)
        write_frame(outputs["context"], context_models(
            episodes, behavior, context, universe, extension, cohort
        ))

    manifest_path = args.output_dir / f"{cohort}-model-manifest.json"
    previous_outputs: dict[str, str] = {}
    previous_output_commits: dict[str, str] = {}
    if args.only_sensitivity and manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text())
        previous_outputs = previous_manifest.get("outputs", {})
        previous_output_commits = previous_manifest.get(
            "output_code_commits",
            {key: previous_manifest.get("code_commit") for key in previous_outputs},
        )
    manifest = {
        "analysis_id": extension["analysis_id"],
        "mode": "models",
        "cohort": cohort,
        "design_commit": args.design_commit,
        "design_commit_time": args.design_commit_time,
        "code_commit": args.code_commit,
        "design_sha256": sha256(args.design),
        "extension_config_sha256": sha256(args.extension_config),
        "universe_config_sha256": sha256(args.universe_config),
        "multidomain_config_sha256": sha256(args.multidomain_config),
        "lookup_sha256": sha256(lookup_path),
        "input_audit": input_audit,
        "source_component_audit": source_audit,
        "fi_audit": fi_audit,
        "probe_manifest_sha256": sha256(probe_manifest_path),
        "outputs": {**previous_outputs, **{key: file_sha(path) for key, path in outputs.items()}},
        "output_code_commits": {
            **previous_output_commits,
            **{key: args.code_commit for key in outputs},
        },
        "aggregate_only": True,
        "respondent_rows_exported": 0,
    }
    write_json(manifest_path, manifest)
    print({"status": "PASS", "cohort": cohort, "aggregate_outputs": len(outputs)})


if __name__ == "__main__":
    main()
