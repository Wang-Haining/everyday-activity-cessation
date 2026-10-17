#!/usr/bin/env python3
"""Fit gated cohort-specific models for the behavioral-contraction pilot."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from behavioral_contraction_biomarker_core import (
    PRIMARY_MARKERS,
    add_base_design,
    add_burden,
    assay_values,
    binary_support,
    coefficient_rows,
    continuous_support,
    contraction_frame,
    covariate_frame,
    file_sha,
    fit_clustered,
    load_episodes,
    load_specs,
    outcome_and_baseline,
    write_frame,
    write_json,
)
from probe_behavior_outcome_feasibility import load_lookup, sha256
from probe_paper3_alcohol_diabetes_mechanisms import load_elsa_labs


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--universe-config", required=True, type=Path)
    parser.add_argument("--pilot-config", required=True, type=Path)
    parser.add_argument("--assay-config", required=True, type=Path)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--minimum-baseline-engagement", type=int, choices=[0, 1], default=0)
    parser.add_argument("--clinical-only", action="store_true")
    return parser.parse_args()


def model_data(
    episodes: pd.DataFrame,
    behavior: pd.DataFrame,
    universe: dict[str, Any],
    pilot: dict[str, Any],
    cohort: str,
    outcome_id: str,
    scope: str,
    adjustment: str,
    exposure_model: str,
    minimum_baseline_engagement: int = 0,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str], bool, str]:
    outcome, baseline, coding_status, binary_outcome = outcome_and_baseline(
        episodes, universe, cohort, outcome_id
    )
    selected = behavior["valid_behavior_pair"] & outcome.notna()
    if scope == "comparable_22_30_months":
        selected &= episodes["comparable_window"].fillna(False)
    data = covariate_frame(episodes, behavior, universe, cohort, outcome_id)
    data["outcome"] = outcome
    if baseline is not None:
        data["baseline_outcome"] = baseline
    data["any_contraction"] = behavior["any_contraction"]
    data["loss_count_1"] = behavior["loss_count_2"].eq(1).astype(float).where(behavior["loss_count_2"].notna())
    data["loss_count_2"] = behavior["loss_count_2"].eq(2).astype(float).where(behavior["loss_count_2"].notna())
    data["alcohol_loss"] = behavior["alcohol_loss"]
    data["activity_loss"] = behavior["activity_loss"]
    selected &= data["baseline_engagement_count"].ge(minimum_baseline_engagement)
    terms_by_model = {
        "any_contraction": ["any_contraction"],
        "loss_count_categorical": ["loss_count_1", "loss_count_2"],
        "mutually_adjusted_components": ["alcohol_loss", "activity_loss"],
    }
    terms = terms_by_model[exposure_model]
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


def fit_clinical(
    episodes: pd.DataFrame,
    behavior: pd.DataFrame,
    universe: dict[str, Any],
    pilot: dict[str, Any],
    cohort: str,
    minimum_baseline_engagement: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    outcomes = pilot["primary_binary_outcomes"] + pilot["primary_continuous_outcomes"]
    if cohort in ["elsa", "hrs"]:
        outcomes += pilot["two_cohort_outcomes"]
    for scope in ["comparable_22_30_months", "all_primary_wave_intervals"]:
        for outcome_id in outcomes:
            for adjustment in ["basic", "full"]:
                for exposure_model in [
                    "any_contraction", "loss_count_categorical", "mutually_adjusted_components"
                ]:
                    base = {
                        "cohort": cohort, "scope": scope, "outcome_id": outcome_id,
                        "adjustment": adjustment, "exposure_model": exposure_model,
                    }
                    try:
                        data, y, X, terms, binary_outcome, coding_status = model_data(
                            episodes, behavior, universe, pilot, cohort, outcome_id,
                            scope, adjustment, exposure_model, minimum_baseline_engagement,
                        )
                        base["outcome_coding_status"] = coding_status
                        if binary_outcome:
                            status, counts = binary_support(data, y, X, terms, pilot)
                        else:
                            status, counts = continuous_support(data, "any_contraction", pilot)
                            if exposure_model != "any_contraction":
                                for term in terms:
                                    if int(data[term].eq(1).sum()) < pilot["minimum_continuous_group_n"]:
                                        status = f"NOT_EVALUABLE_{term.upper()}_GROUP_N"
                                        break
                        if status != "ESTIMABLE":
                            rows.append({**base, **counts, "model_status": status})
                            continue
                        result, warning_text = fit_clustered(y, X, data["person_id"], binary_outcome)
                        estimates = coefficient_rows(result, terms, binary_outcome)
                        outcome_sd = float(y.std(ddof=1)) if not binary_outcome else np.nan
                        for estimate in estimates:
                            if not binary_outcome and outcome_sd > 0:
                                estimate.update({
                                    "standardized_estimate": estimate["estimate"] / outcome_sd,
                                    "standardized_se": estimate["standard_error"] / outcome_sd,
                                    "standardized_ci_low": estimate["ci_low"] / outcome_sd,
                                    "standardized_ci_high": estimate["ci_high"] / outcome_sd,
                                })
                            rows.append({
                                **base, **counts, "model_status": "PASS",
                                "converged": bool(getattr(result, "converged", True)),
                                "warnings": warning_text, **estimate,
                            })
                    except Exception as exc:
                        rows.append({
                            **base, "model_status": "MODEL_FAILURE",
                            "failure_reason": f"{type(exc).__name__}: {exc}",
                        })
    return rows


def joint_cell_gate(
    data: pd.DataFrame, y: pd.Series, pilot: dict[str, Any], binary_outcome: bool
) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    for state in [0, 1, 2, 3]:
        selected = data["joint_state"].eq(state)
        counts[f"joint_state_{state}_n"] = int(selected.sum())
        if binary_outcome:
            events = int(y.loc[selected].sum())
            counts[f"joint_state_{state}_events"] = events
            counts[f"joint_state_{state}_nonevents"] = int(selected.sum()) - events
            if events < pilot["minimum_joint_state_events"]:
                return f"NOT_EVALUABLE_STATE_{state}_EVENTS", counts
            if counts[f"joint_state_{state}_nonevents"] < pilot["minimum_joint_state_nonevents"]:
                return f"NOT_EVALUABLE_STATE_{state}_NONEVENTS", counts
        elif int(selected.sum()) < pilot["minimum_continuous_group_n"]:
            return f"NOT_EVALUABLE_STATE_{state}_N", counts
    return "ESTIMABLE", counts


def fit_assay_outcomes(
    episodes: pd.DataFrame,
    behavior: pd.DataFrame,
    assays: pd.DataFrame,
    universe: dict[str, Any],
    pilot: dict[str, Any],
    cohort: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    outcomes = pilot["primary_binary_outcomes"] + pilot["primary_continuous_outcomes"]
    if cohort in ["elsa", "hrs"]:
        outcomes += pilot["two_cohort_outcomes"]
    base_landmark = assays["landmark_valid"] & behavior["valid_behavior_pair"]
    marker_columns = {
        "burden_z": "burden_z", "crp": "z_crp", "hba1c": "z_hba1c",
        "triglycerides": "z_triglycerides", "hdl_adverse": "z_hdl_adverse",
    }
    scope_variants = []
    for assay_scope in sorted(assays.loc[base_landmark, "assay_scope"].dropna().unique()):
        scope_variants.append((str(assay_scope), base_landmark & assays["assay_scope"].eq(assay_scope)))
        scope_variants.append((
            f"{assay_scope}__crp_lt_10",
            base_landmark & assays["assay_scope"].eq(assay_scope)
            & assays["crp"].lt(pilot["crp_acute_inflammation_cutoff_mg_l"]),
        ))
    for assay_scope, assay_scope_mask in scope_variants:
        for outcome_id in outcomes:
            outcome, baseline, coding_status, binary_outcome = outcome_and_baseline(
                episodes, universe, cohort, outcome_id
            )
            for marker_id, marker_field in marker_columns.items():
                data = covariate_frame(episodes, behavior, universe, cohort, outcome_id)
                data["outcome"] = outcome
                data["any_contraction"] = behavior["any_contraction"]
                data["marker_z"] = assays[marker_field]
                if baseline is not None:
                    data["baseline_outcome"] = baseline
                selected = (
                    assay_scope_mask
                    & data[[
                        "outcome", "person_id", "t1_wave", "age", "sex",
                        "baseline_engagement_count", "education_rank", "income_rank",
                        "smoke_ever", "smoke_current", "baseline_multimorbidity",
                        "any_contraction", "marker_z",
                    ] + (["baseline_outcome"] if baseline is not None else [])].notna().all(axis=1)
                )
                data = data.loc[selected].copy()
                base = {
                    "cohort": cohort, "assay_scope": assay_scope,
                    "outcome_id": outcome_id, "outcome_coding_status": coding_status,
                    "marker": marker_id, "model_family": "contraction_plus_marker_main",
                }
                if binary_outcome:
                    X = add_base_design(data, pilot, "full")
                    X["any_contraction"] = data["any_contraction"]
                    X["marker_z"] = data["marker_z"]
                    status, counts = binary_support(data, data["outcome"], X, ["any_contraction"], pilot)
                else:
                    status, counts = continuous_support(data, "any_contraction", pilot)
                    X = add_base_design(data, pilot, "full") if status == "ESTIMABLE" else pd.DataFrame()
                    if status == "ESTIMABLE":
                        X["any_contraction"] = data["any_contraction"]
                        X["marker_z"] = data["marker_z"]
                if status != "ESTIMABLE":
                    rows.append({**base, **counts, "model_status": status})
                    continue
                try:
                    result, warning_text = fit_clustered(
                        data["outcome"].astype(float), X, data["person_id"], binary_outcome
                    )
                    for estimate in coefficient_rows(result, ["any_contraction", "marker_z"], binary_outcome):
                        rows.append({
                            **base, **counts, "model_status": "PASS", "warnings": warning_text, **estimate,
                        })
                except Exception as exc:
                    rows.append({**base, **counts, "model_status": "MODEL_FAILURE", "failure_reason": f"{type(exc).__name__}: {exc}"})

            # Burden interaction and four-state model are assessed once per outcome.
            data = covariate_frame(episodes, behavior, universe, cohort, outcome_id)
            data["outcome"] = outcome
            data["any_contraction"] = behavior["any_contraction"]
            data["burden_z"] = assays["burden_z"]
            data["high_burden"] = assays["high_burden"]
            data["joint_state"] = (
                data["any_contraction"].astype("Int64") + 2 * data["high_burden"].astype("Int64")
            )
            if baseline is not None:
                data["baseline_outcome"] = baseline
            required = [
                "outcome", "person_id", "t1_wave", "age", "sex", "baseline_engagement_count",
                "education_rank", "income_rank", "smoke_ever", "smoke_current",
                "baseline_multimorbidity", "any_contraction", "burden_z", "high_burden", "joint_state",
            ] + (["baseline_outcome"] if baseline is not None else [])
            selected = assay_scope_mask & data[required].notna().all(axis=1)
            data = data.loc[selected].copy()
            gate, cell_counts = joint_cell_gate(data, data["outcome"], pilot, binary_outcome)
            base = {
                "cohort": cohort, "assay_scope": assay_scope, "outcome_id": outcome_id,
                "outcome_coding_status": coding_status, "marker": "burden_z",
                "model_family": "burden_interaction",
            }
            if gate != "ESTIMABLE":
                rows.append({**base, **cell_counts, "model_status": gate})
            else:
                try:
                    X = add_base_design(data, pilot, "full")
                    X["any_contraction"] = data["any_contraction"]
                    X["burden_z"] = data["burden_z"]
                    X["contraction_x_burden"] = data["any_contraction"] * data["burden_z"]
                    if np.linalg.matrix_rank(X.to_numpy(float)) < X.shape[1]:
                        rows.append({**base, **cell_counts, "model_status": "NOT_EVALUABLE_INTERACTION_RANK"})
                    elif binary_outcome and int(data["outcome"].sum()) < pilot["events_per_parameter"] * X.shape[1]:
                        rows.append({**base, **cell_counts, "model_status": "NOT_EVALUABLE_INTERACTION_EPV"})
                    else:
                        result, warning_text = fit_clustered(data["outcome"], X, data["person_id"], binary_outcome)
                        for estimate in coefficient_rows(
                            result, ["any_contraction", "burden_z", "contraction_x_burden"], binary_outcome
                        ):
                            rows.append({**base, **cell_counts, "model_status": "PASS", "warnings": warning_text, **estimate})

                        joint_base = {
                            **base, "model_family": "joint_four_state",
                            "reference_state": "no_contraction_lower_burden",
                        }
                        X_joint = add_base_design(data, pilot, "full")
                        joint_terms = []
                        for state, label in [
                            (1, "contraction_lower_burden"),
                            (2, "no_contraction_high_burden"),
                            (3, "contraction_high_burden"),
                        ]:
                            term = f"joint__{label}"
                            X_joint[term] = data["joint_state"].eq(state).astype(float)
                            joint_terms.append(term)
                        if np.linalg.matrix_rank(X_joint.to_numpy(float)) < X_joint.shape[1]:
                            rows.append({**joint_base, **cell_counts, "model_status": "NOT_EVALUABLE_JOINT_RANK"})
                        else:
                            joint_result, joint_warning = fit_clustered(
                                data["outcome"], X_joint, data["person_id"], binary_outcome
                            )
                            for estimate in coefficient_rows(joint_result, joint_terms, binary_outcome):
                                rows.append({
                                    **joint_base, **cell_counts, "model_status": "PASS",
                                    "warnings": joint_warning, **estimate,
                                })
                except Exception as exc:
                    rows.append({**base, **cell_counts, "model_status": "MODEL_FAILURE", "failure_reason": f"{type(exc).__name__}: {exc}"})
    return rows


def fit_biomarkers_predict_contraction(
    episodes: pd.DataFrame,
    behavior: pd.DataFrame,
    assays: pd.DataFrame,
    universe: dict[str, Any],
    pilot: dict[str, Any],
    cohort: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    marker_columns = {
        "burden_z": "burden_z", "crp": "z_crp", "hba1c": "z_hba1c",
        "triglycerides": "z_triglycerides", "hdl_adverse": "z_hdl_adverse",
    }
    base_landmark = assays["landmark_valid"] & behavior["valid_behavior_pair"]
    scope_variants = []
    for assay_scope in sorted(assays.loc[base_landmark, "assay_scope"].dropna().unique()):
        scope_variants.append((str(assay_scope), base_landmark & assays["assay_scope"].eq(assay_scope)))
        scope_variants.append((
            f"{assay_scope}__crp_lt_10",
            base_landmark & assays["assay_scope"].eq(assay_scope)
            & assays["crp"].lt(pilot["crp_acute_inflammation_cutoff_mg_l"]),
        ))
    for assay_scope, assay_scope_mask in scope_variants:
        for marker_id, marker_field in marker_columns.items():
            data = covariate_frame(episodes, behavior, universe, cohort, "mortality", anchor="t0")
            data["outcome"] = behavior["any_contraction"]
            data["marker_z"] = assays[marker_field]
            required = [
                "outcome", "person_id", "t1_wave", "age", "sex", "baseline_engagement_count",
                "education_rank", "income_rank", "smoke_ever", "smoke_current",
                "baseline_multimorbidity", "marker_z",
            ]
            selected = assay_scope_mask & data[required].notna().all(axis=1)
            data = data.loc[selected].copy()
            base = {
                "cohort": cohort, "assay_scope": assay_scope, "outcome_id": "subsequent_any_contraction",
                "marker": marker_id, "model_family": "baseline_marker_predicts_contraction",
                "n": int(len(data)), "people": int(data["person_id"].nunique()),
                "events": int(data["outcome"].sum()),
            }
            try:
                X = add_base_design(data, pilot, "full")
                X["marker_z"] = data["marker_z"]
                if base["events"] < pilot["events_per_parameter"] * X.shape[1]:
                    rows.append({**base, "model_status": "NOT_EVALUABLE_EPV"})
                    continue
                result, warning_text = fit_clustered(data["outcome"], X, data["person_id"], True)
                estimate = coefficient_rows(result, ["marker_z"], True)[0]
                rows.append({**base, "model_status": "PASS", "warnings": warning_text, **estimate})
            except Exception as exc:
                rows.append({**base, "model_status": "MODEL_FAILURE", "failure_reason": f"{type(exc).__name__}: {exc}"})
    return rows


def z_by_wave_sex(values: pd.Series, wave: pd.Series, sex: pd.Series) -> pd.Series:
    mean = values.groupby([wave, sex]).transform("mean")
    sd = values.groupby([wave, sex]).transform("std")
    return (values - mean) / sd.where(sd > 0)


def fit_elsa_repeated(
    episodes: pd.DataFrame,
    behavior: pd.DataFrame,
    assays: pd.DataFrame,
    universe: dict[str, Any],
    pilot: dict[str, Any],
    assay_config: dict[str, Any],
    root: Path,
) -> list[dict[str, Any]]:
    labs, _ = load_elsa_labs(root, assay_config["biomarker_modules"]["elsa_landmark_repeated"])
    source = labs.set_index(["person_id", "lab_wave"])
    target_index = pd.MultiIndex.from_arrays([episodes["person_id"].astype("string"), episodes["outcome_wave"]])
    mapping = {
        "crp": "crp_mg_l", "hba1c": "hba1c_percent",
        "triglycerides": "triglycerides_mmol_l", "hdl": "hdl_mmol_l",
    }
    sex = pd.to_numeric(episodes["t1__ragender"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for marker, field in mapping.items():
        baseline = assays[marker].copy()
        followup = pd.Series(source[field].reindex(target_index).to_numpy(), index=episodes.index, dtype=float)
        if marker in ["crp", "triglycerides"]:
            baseline = np.log(baseline.where(baseline > 0) + (pilot["crp_log_offset"] if marker == "crp" else 0))
            followup = np.log(followup.where(followup > 0) + (pilot["crp_log_offset"] if marker == "crp" else 0))
        baseline_z = z_by_wave_sex(baseline, episodes["t0"], sex)
        followup_z = z_by_wave_sex(followup, episodes["outcome_wave"], sex)
        data = covariate_frame(episodes, behavior, universe, "elsa", "mortality")
        data["outcome"] = followup_z
        data["baseline_marker"] = baseline_z
        data["any_contraction"] = behavior["any_contraction"]
        required = [
            "outcome", "baseline_marker", "any_contraction", "person_id", "t1_wave", "age", "sex",
            "baseline_engagement_count", "education_rank", "income_rank", "smoke_ever", "smoke_current",
            "baseline_multimorbidity",
        ]
        selected = assays["landmark_valid"] & assays["assay_scope"].eq("comparable") & data[required].notna().all(axis=1)
        data = data.loc[selected].copy()
        status, counts = continuous_support(data, "any_contraction", pilot)
        base = {
            "cohort": "elsa", "assay_scope": "repeated_assay_t0_to_t2", "outcome_id": f"followup_{marker}",
            "marker": marker, "model_family": "baseline_adjusted_repeated_assay", **counts,
        }
        if status != "ESTIMABLE":
            rows.append({**base, "model_status": status})
            continue
        try:
            X = add_base_design(data, pilot, "full")
            X["baseline_marker"] = data["baseline_marker"]
            X["any_contraction"] = data["any_contraction"]
            result, warning_text = fit_clustered(data["outcome"], X, data["person_id"], False)
            estimate = coefficient_rows(result, ["any_contraction"], False)[0]
            rows.append({**base, "model_status": "PASS", "warnings": warning_text, **estimate})
        except Exception as exc:
            rows.append({**base, "model_status": "MODEL_FAILURE", "failure_reason": f"{type(exc).__name__}: {exc}"})
    return rows


def main() -> None:
    args = arguments()
    assert args.data_root.is_dir(), f"missing data root: {args.data_root}"
    universe, pilot, assay_config = load_specs(args.universe_config, args.pilot_config, args.assay_config)
    cohort = args.cohort.lower()
    root = args.data_root
    lookup, lookup_path = load_lookup(root, universe)
    episodes, _, source_audit = load_episodes(root, universe, cohort, lookup)
    behavior = contraction_frame(episodes, universe, cohort)
    clinical_rows = fit_clinical(
        episodes,
        behavior,
        universe,
        pilot,
        cohort,
        minimum_baseline_engagement=args.minimum_baseline_engagement,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clinical_path = args.output_dir / f"{cohort}-clinical-models.csv"
    write_frame(clinical_path, clinical_rows)
    outputs = {"clinical_models": clinical_path}
    assay_audit = None
    if cohort in pilot["assay_cohorts"] and not args.clinical_only:
        assays, assay_audit = assay_values(root, universe, assay_config, pilot, cohort, episodes)
        cc = universe["cohorts"][cohort]
        sex = pd.to_numeric(episodes[f"t1__{cc['sex']}"] , errors="coerce")
        assays = add_burden(assays, sex, pilot)
        biomarker_rows = fit_assay_outcomes(
            episodes, behavior, assays, universe, pilot, cohort
        )
        biomarker_rows += fit_biomarkers_predict_contraction(
            episodes, behavior, assays, universe, pilot, cohort
        )
        if cohort == "elsa":
            biomarker_rows += fit_elsa_repeated(
                episodes, behavior, assays, universe, pilot, assay_config, root
            )
        biomarker_path = args.output_dir / f"{cohort}-biomarker-models.csv"
        write_frame(biomarker_path, biomarker_rows)
        outputs["biomarker_models"] = biomarker_path
    manifest_path = args.output_dir / f"{cohort}-model-manifest.json"
    manifest = {
        "analysis_id": pilot["analysis_id"], "cohort": cohort,
        "execution_revision": "v0.1.1_duplicate_baseline_covariate_fix",
        "minimum_baseline_engagement": args.minimum_baseline_engagement,
        "clinical_only": args.clinical_only,
        "pilot_config_sha256": sha256(args.pilot_config),
        "universe_config_sha256": sha256(args.universe_config),
        "assay_config_sha256": sha256(args.assay_config), "lookup_sha256": sha256(lookup_path),
        "probe_manifest_sha256": file_sha(args.probe_dir / f"{cohort}-probe-manifest.json"),
        "source_audit": source_audit, "assay_audit": assay_audit,
        "outputs": {key: file_sha(path) for key, path in outputs.items()},
        "aggregate_only": True, "respondent_rows_exported": 0, "negative_results_retained": True,
    }
    write_json(manifest_path, manifest)
    print({"status": "PASS", "cohort": cohort, "clinical_rows": len(clinical_rows)})


if __name__ == "__main__":
    main()
