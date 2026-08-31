#!/usr/bin/env python3
"""Counts-only support probe for the frailty head-to-head extension."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from probe_behavior_outcome_feasibility import outcome_values

from behavioral_withdrawal_frailty_core import (
    add_direction_categories,
    build_context_frame,
    build_frailty_frame,
    competing_outcome,
    delayed_outcome,
    extend_four_wave,
    load_extension_data,
    load_extension_specs,
    load_fi_long,
    load_lookup,
    load_source_components,
    multidomain_frame,
    sha256,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe-config", required=True, type=Path)
    parser.add_argument("--multidomain-config", required=True, type=Path)
    parser.add_argument("--extension-config", required=True, type=Path)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--design-commit", required=True)
    parser.add_argument("--design-commit-time", required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"empty aggregate output: {path.name}")
    pd.DataFrame(rows).to_csv(path, index=False)
    os.chmod(path, 0o600)


def flow_rows(
    episodes: pd.DataFrame,
    behavior: pd.DataFrame,
    cohort: str,
) -> list[dict[str, Any]]:
    rows = []
    for scope, scope_mask in {
        "comparable_22_30_months": episodes["comparable_window"].fillna(False),
        "all_primary_wave_intervals": pd.Series(True, index=episodes.index),
    }.items():
        eligible = scope_mask & behavior["core_valid"] & behavior["baseline_engagement_count"].ge(1)
        row: dict[str, Any] = {
            "cohort": cohort,
            "scope": scope,
            "intervals": int(eligible.sum()),
            "people": int(episodes.loc[eligible, "person_id"].nunique()),
        }
        for count, label in [(0, "0"), (1, "1")]:
            cell = eligible & behavior["loss_count"].eq(count)
            row[f"loss_{label}_n"] = int(cell.sum())
        row["loss_2_plus_n"] = int((eligible & behavior["loss_count"].ge(2)).sum())
        for direction in ["stable", "expansion", "mixed", "contraction_1", "contraction_2_plus"]:
            row[f"direction_{direction}_n"] = int(
                (eligible & behavior["transition_direction"].eq(direction)).sum()
            )
        rows.append(row)
    return rows


def fried_rows(
    episodes: pd.DataFrame,
    behavior: pd.DataFrame,
    frailty: pd.DataFrame,
    cohort: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    component_rows = []
    components = ["shrinking", "exhaustion", "weakness", "slowness", "low_activity"]
    for wave in sorted(pd.to_numeric(episodes["t1"], errors="coerce").dropna().unique()):
        wave_mask = episodes["t1"].eq(wave) & behavior["core_valid"]
        row: dict[str, Any] = {
            "cohort": cohort,
            "t1_wave": int(wave),
            "behavior_complete_n": int(wave_mask.sum()),
        }
        for component in components:
            row[f"{component}_observed_n"] = int((wave_mask & frailty[component].notna()).sum())
            row[f"{component}_deficit_n"] = int((wave_mask & frailty[component].eq(1)).sum())
        complete = wave_mask & frailty["fried5_t1"].notna()
        row["fried5_complete_n"] = int(complete.sum())
        for category in ["robust", "prefrail", "frail"]:
            row[f"fried_{category}_n"] = int((complete & frailty["fried_category"].eq(category)).sum())
        row["fi_complete_n"] = int((wave_mask & frailty["fi_t1"].notna()).sum())
        component_rows.append(row)

    overlap_rows = []
    comparable = (
        episodes["comparable_window"].fillna(False)
        & behavior["core_valid"]
        & behavior["baseline_engagement_count"].ge(1)
        & frailty["fried5_t1"].notna()
    )
    loss_category = pd.Series(pd.NA, index=episodes.index, dtype="string")
    loss_category.loc[behavior["loss_count"].eq(0)] = "0"
    loss_category.loc[behavior["loss_count"].eq(1)] = "1"
    loss_category.loc[behavior["loss_count"].ge(2)] = "2_plus"
    for loss in ["0", "1", "2_plus"]:
        for category in ["robust", "prefrail", "frail"]:
            cell = comparable & loss_category.eq(loss) & frailty["fried_category"].eq(category)
            overlap_rows.append({
                "cohort": cohort,
                "loss_category": loss,
                "fried_category": category,
                "intervals": int(cell.sum()),
                "people": int(episodes.loc[cell, "person_id"].nunique()),
            })
    return component_rows, overlap_rows


def support_rows(
    episodes: pd.DataFrame,
    behavior: pd.DataFrame,
    universe: dict[str, Any],
    extension: dict[str, Any],
    cohort: str,
    frailty: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    rows = []
    comparable = (
        episodes["comparable_window"].fillna(False)
        & behavior["core_valid"]
        & behavior["baseline_engagement_count"].ge(1)
    )
    for outcome_id in extension["specificity_outcomes"]:
        outcome, coding = outcome_values(
            episodes,
            outcome_id,
            universe["outcomes"][outcome_id],
            universe["outcomes"],
            cohort,
        )
        eligible = comparable & outcome.notna()
        row: dict[str, Any] = {
            "cohort": cohort,
            "outcome_id": outcome_id,
            "outcome_coding_status": coding,
            "n": int(eligible.sum()),
            "people": int(episodes.loc[eligible, "person_id"].nunique()),
            "events": int(outcome.loc[eligible].sum()),
        }
        for condition, label in [
            (behavior["loss_count"].eq(0), "loss_0"),
            (behavior["loss_count"].eq(1), "loss_1"),
            (behavior["loss_count"].ge(2), "loss_2_plus"),
            (behavior["any_withdrawal"].eq(1), "any_withdrawal"),
        ]:
            cell = eligible & condition
            row[f"{label}_n"] = int(cell.sum())
            row[f"{label}_events"] = int(outcome.loc[cell].sum())
        if frailty is not None:
            fried_set = eligible & frailty["fried5_t1"].notna()
            row["fried_complete_n"] = int(fried_set.sum())
            row["fried_complete_events"] = int(outcome.loc[fried_set].sum())
            nonfrail_withdrawal = fried_set & frailty["fried_category"].ne("frail") & behavior["any_withdrawal"].eq(1)
            row["nonfrail_withdrawal_n"] = int(nonfrail_withdrawal.sum())
            row["nonfrail_withdrawal_events"] = int(outcome.loc[nonfrail_withdrawal].sum())
            for withdrawal, frail, label in [
                (False, False, "neither"),
                (True, False, "withdrawal_only"),
                (False, True, "frailty_only"),
                (True, True, "both"),
            ]:
                cell = fried_set & behavior["any_withdrawal"].eq(int(withdrawal)) & frailty["fried_category"].eq("frail").eq(frail)
                row[f"four_state_{label}_n"] = int(cell.sum())
                row[f"four_state_{label}_events"] = int(outcome.loc[cell].sum())
        composite, state = competing_outcome(episodes, universe, cohort, outcome_id)
        composite_set = comparable & composite.notna()
        row["diagnosis_or_death_n"] = int(composite_set.sum())
        row["diagnosis_or_death_events"] = int(composite.loc[composite_set].sum())
        for state_name in ["alive_no_diagnosis", "alive_new_diagnosis", "death", "unknown"]:
            row[f"state_{state_name}_n"] = int((comparable & state.eq(state_name)).sum())
        row["score_support"] = (
            "ESTIMABLE"
            if row["loss_1_events"] >= extension["minimum_exposed_events"]
            and row["loss_2_plus_events"] >= extension["minimum_exposed_events"]
            and row["loss_1_n"] - row["loss_1_events"] >= extension["minimum_exposed_nonevents"]
            and row["loss_2_plus_n"] - row["loss_2_plus_events"] >= extension["minimum_exposed_nonevents"]
            else "NOT_EVALUABLE_CATEGORY_SUPPORT"
        )
        rows.append(row)
    return rows


def delayed_rows(
    episodes: pd.DataFrame,
    formal: pd.DataFrame,
    status: pd.DataFrame,
    behavior: pd.DataFrame,
    universe: dict[str, Any],
    extension: dict[str, Any],
    cohort: str,
) -> list[dict[str, Any]]:
    episodes4 = extend_four_wave(episodes, formal, status, universe, cohort)
    rows = []
    for outcome_id in extension["delayed_outcomes"]:
        outcome = delayed_outcome(episodes4, outcome_id)
        eligible = behavior["core_valid"] & behavior["baseline_engagement_count"].ge(1) & outcome.notna()
        row: dict[str, Any] = {
            "cohort": cohort,
            "outcome_id": outcome_id,
            "n": int(eligible.sum()),
            "people": int(episodes.loc[eligible, "person_id"].nunique()),
            "events": int(outcome.loc[eligible].sum()),
        }
        for condition, label in [
            (behavior["loss_count"].eq(0), "loss_0"),
            (behavior["loss_count"].eq(1), "loss_1"),
            (behavior["loss_count"].ge(2), "loss_2_plus"),
        ]:
            cell = eligible & condition
            row[f"{label}_n"] = int(cell.sum())
            row[f"{label}_events"] = int(outcome.loc[cell].sum())
        row["support"] = (
            "ESTIMABLE"
            if row["loss_1_events"] >= extension["minimum_exposed_events"]
            and row["loss_2_plus_events"] >= extension["minimum_exposed_events"]
            else "NOT_EVALUABLE_CATEGORY_SUPPORT"
        )
        rows.append(row)
    return rows


def context_rows(
    episodes: pd.DataFrame,
    behavior: pd.DataFrame,
    context: pd.DataFrame,
    cohort: str,
) -> list[dict[str, Any]]:
    eligible = episodes["comparable_window"].fillna(False) & behavior["core_valid"]
    rows = []
    for field in context.columns:
        observed = eligible & context[field].notna()
        rows.append({
            "cohort": cohort,
            "context_field": field,
            "observed_n": int(observed.sum()),
            "positive_n": int((observed & context[field].gt(0)).sum()) if field != "fi_change" else int((observed & context[field].ge(0.05)).sum()),
            "withdrawal_observed_n": int((observed & behavior["any_withdrawal"].eq(1)).sum()),
        })
    return rows


def main() -> None:
    args = arguments()
    universe, multidomain, extension = load_extension_specs(
        args.universe_config, args.multidomain_config, args.extension_config
    )
    cohort = args.cohort.lower()
    if cohort not in extension["specificity_cohorts"]:
        raise RuntimeError("cohort outside frozen extension")
    root = Path(extension["release_root"])
    lookup, lookup_path = load_lookup(root, universe)
    episodes, formal, status, intervals, audit = load_extension_data(root, universe, cohort, lookup)
    behavior = add_direction_categories(multidomain_frame(episodes, universe, multidomain, cohort))
    source_audit: dict[str, Any] = {}
    fi_audit: dict[str, Any] = {}
    frailty = None
    context = None
    if cohort in extension["frailty_cohorts"]:
        source, source_audit = load_source_components(root, extension, cohort)
        fi_long, fi_audit = load_fi_long(extension, cohort)
        frailty = build_frailty_frame(episodes, formal, source, fi_long, universe, extension, cohort)
        context = build_context_frame(episodes, frailty, universe, cohort)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(args.output_dir, 0o700)
    outputs: dict[str, Path] = {}
    outputs["flow"] = args.output_dir / f"{cohort}-flow.csv"
    outputs["support"] = args.output_dir / f"{cohort}-support.csv"
    outputs["delayed_support"] = args.output_dir / f"{cohort}-delayed-support.csv"
    outputs["intervals"] = args.output_dir / f"{cohort}-intervals.csv"
    write_csv(outputs["flow"], flow_rows(episodes, behavior, cohort))
    write_csv(outputs["support"], support_rows(episodes, behavior, universe, extension, cohort, frailty))
    write_csv(outputs["delayed_support"], delayed_rows(episodes, formal, status, behavior, universe, extension, cohort))
    write_csv(outputs["intervals"], intervals)
    if frailty is not None and context is not None:
        component_rows, overlap_rows = fried_rows(episodes, behavior, frailty, cohort)
        outputs["fried_availability"] = args.output_dir / f"{cohort}-fried-availability.csv"
        outputs["fried_overlap"] = args.output_dir / f"{cohort}-fried-overlap.csv"
        outputs["context"] = args.output_dir / f"{cohort}-context-support.csv"
        write_csv(outputs["fried_availability"], component_rows)
        write_csv(outputs["fried_overlap"], overlap_rows)
        write_csv(outputs["context"], context_rows(episodes, behavior, context, cohort))

    manifest = {
        "analysis_id": extension["analysis_id"],
        "mode": "counts_only_probe",
        "cohort": cohort,
        "design_commit": args.design_commit,
        "design_commit_time": args.design_commit_time,
        "code_commit": args.code_commit,
        "design_sha256": sha256(args.design),
        "extension_config_sha256": sha256(args.extension_config),
        "universe_config_sha256": sha256(args.universe_config),
        "multidomain_config_sha256": sha256(args.multidomain_config),
        "lookup_sha256": sha256(lookup_path),
        "input_audit": audit,
        "source_component_audit": source_audit,
        "fi_audit": fi_audit,
        "outputs": {key: sha256(path) for key, path in outputs.items()},
        "aggregate_only": True,
        "respondent_rows_exported": 0,
        "effect_models_fit": 0,
    }
    manifest_path = args.output_dir / f"{cohort}-probe-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    os.chmod(manifest_path, 0o600)
    print({"status": "PASS", "cohort": cohort, "aggregate_outputs": len(outputs)})


if __name__ == "__main__":
    main()
