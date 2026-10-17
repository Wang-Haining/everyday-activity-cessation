#!/usr/bin/env python3
"""Counts-only support probe for one cohort in the pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from behavioral_contraction_biomarker_core import (
    add_burden,
    assay_values,
    contraction_frame,
    covariate_frame,
    file_sha,
    load_episodes,
    load_specs,
    outcome_and_baseline,
    write_frame,
    write_json,
)
from probe_behavior_outcome_feasibility import load_lookup, sha256


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--universe-config", required=True, type=Path)
    parser.add_argument("--pilot-config", required=True, type=Path)
    parser.add_argument("--assay-config", required=True, type=Path)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def support_label(row: dict, binary: bool, pilot: dict) -> str:
    if binary:
        if row["contraction_events"] < pilot["minimum_exposed_events"]:
            return "NOT_EVALUABLE_CONTRACTION_EVENTS"
        if row["contraction_nonevents"] < pilot["minimum_exposed_nonevents"]:
            return "NOT_EVALUABLE_CONTRACTION_NONEVENTS"
        if row["events"] < pilot["events_per_parameter"] * row["planned_parameters_full"]:
            return "NOT_EVALUABLE_EPV"
        return "ESTIMABLE"
    if row["n"] < pilot["minimum_continuous_n"]:
        return "NOT_EVALUABLE_CONTINUOUS_N"
    if min(row["contraction_n"], row["noncontraction_n"]) < pilot["minimum_continuous_group_n"]:
        return "NOT_EVALUABLE_CONTINUOUS_GROUP_N"
    return "ESTIMABLE"


def main() -> None:
    args = arguments()
    assert args.data_root.is_dir(), f"missing data root: {args.data_root}"
    universe, pilot, assay_config = load_specs(args.universe_config, args.pilot_config, args.assay_config)
    cohort = args.cohort.lower()
    if cohort not in pilot["candidate_cohorts"]:
        raise RuntimeError("cohort outside pilot")
    root = args.data_root
    lookup, lookup_path = load_lookup(root, universe)
    episodes, intervals, source_audit = load_episodes(root, universe, cohort, lookup)
    behavior = contraction_frame(episodes, universe, cohort)
    flow_rows = []
    for scope_name, scope_mask in {
        "comparable_22_30_months": episodes["comparable_window"].fillna(False),
        "all_primary_wave_intervals": pd.Series(True, index=episodes.index),
    }.items():
        selected = scope_mask & behavior["valid_behavior_pair"]
        row = {
            "cohort": cohort, "scope": scope_name, "episodes": int(selected.sum()),
            "people": int(episodes.loc[selected, "person_id"].nunique()),
        }
        for count in [0, 1, 2]:
            row[f"loss_count_{count}_n"] = int((selected & behavior["loss_count_2"].eq(count)).sum())
        for transition in ["0_to_0", "0_to_1", "1_to_0", "1_to_1"]:
            row[f"alcohol_{transition}_n"] = int((selected & behavior["alcohol_transition"].eq(transition)).sum())
            row[f"activity_{transition}_n"] = int((selected & behavior["activity_transition"].eq(transition)).sum())
        flow_rows.append(row)

    outcome_rows = []
    outcome_ids = pilot["primary_binary_outcomes"] + pilot["primary_continuous_outcomes"]
    if cohort in ["elsa", "hrs"]:
        outcome_ids += pilot["two_cohort_outcomes"]
    for scope_name, scope_mask in {
        "comparable_22_30_months": episodes["comparable_window"].fillna(False),
        "all_primary_wave_intervals": pd.Series(True, index=episodes.index),
    }.items():
        for outcome_id in outcome_ids:
            outcome, _, coding_status, binary_outcome = outcome_and_baseline(episodes, universe, cohort, outcome_id)
            selected = scope_mask & behavior["valid_behavior_pair"] & outcome.notna()
            contraction = selected & behavior["any_contraction"].eq(1)
            row = {
                "cohort": cohort, "scope": scope_name, "outcome_id": outcome_id,
                "outcome_coding_status": coding_status, "outcome_type": "binary" if binary_outcome else "continuous",
                "n": int(selected.sum()), "people": int(episodes.loc[selected, "person_id"].nunique()),
                "contraction_n": int(contraction.sum()),
                "noncontraction_n": int((selected & behavior["any_contraction"].eq(0)).sum()),
                "planned_parameters_full": 14,
            }
            if binary_outcome:
                row["events"] = int(outcome.loc[selected].sum())
                row["contraction_events"] = int(outcome.loc[contraction].sum())
                row["contraction_nonevents"] = int(contraction.sum() - outcome.loc[contraction].sum())
                for count in [0, 1, 2]:
                    cell = selected & behavior["loss_count_2"].eq(count)
                    row[f"loss_count_{count}_n"] = int(cell.sum())
                    row[f"loss_count_{count}_events"] = int(outcome.loc[cell].sum())
            row["support"] = support_label(row, binary_outcome, pilot)
            outcome_rows.append(row)

    assay_rows = []
    assay_audit = None
    if cohort in pilot["assay_cohorts"]:
        assays, assay_audit = assay_values(root, universe, assay_config, pilot, cohort, episodes)
        cc = universe["cohorts"][cohort]
        sex = pd.to_numeric(episodes[f"t1__{cc['sex']}"] , errors="coerce")
        assays = add_burden(assays, sex, pilot)
        landmark = assays["landmark_valid"] & behavior["valid_behavior_pair"]
        scope_variants = []
        for scope in sorted(assays.loc[landmark, "assay_scope"].dropna().unique()):
            scope_variants.append((str(scope), landmark & assays["assay_scope"].eq(scope)))
            scope_variants.append((
                f"{scope}__crp_lt_10",
                landmark & assays["assay_scope"].eq(scope)
                & assays["crp"].lt(pilot["crp_acute_inflammation_cutoff_mg_l"]),
            ))
        for scope, base in scope_variants:
            for outcome_id in outcome_ids:
                outcome, _, coding_status, binary_outcome = outcome_and_baseline(episodes, universe, cohort, outcome_id)
                selected = base & assays["burden_z"].notna() & outcome.notna()
                row = {
                    "cohort": cohort, "assay_scope": scope, "outcome_id": outcome_id,
                    "outcome_coding_status": coding_status, "binary_outcome": binary_outcome,
                    "n": int(selected.sum()), "people": int(episodes.loc[selected, "person_id"].nunique()),
                    "contraction_n": int((selected & behavior["any_contraction"].eq(1)).sum()),
                    "burden_nonmissing_n": int((base & assays["burden_z"].notna()).sum()),
                }
                for state, mask in {
                    "no_contraction_lower": behavior["any_contraction"].eq(0) & assays["high_burden"].eq(0),
                    "contraction_lower": behavior["any_contraction"].eq(1) & assays["high_burden"].eq(0),
                    "no_contraction_high": behavior["any_contraction"].eq(0) & assays["high_burden"].eq(1),
                    "contraction_high": behavior["any_contraction"].eq(1) & assays["high_burden"].eq(1),
                }.items():
                    cell = selected & mask
                    row[f"{state}_n"] = int(cell.sum())
                    if binary_outcome:
                        row[f"{state}_events"] = int(outcome.loc[cell].sum())
                assay_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "flow": args.output_dir / f"{cohort}-flow.csv",
        "outcome_support": args.output_dir / f"{cohort}-outcome-support.csv",
        "intervals": args.output_dir / f"{cohort}-intervals.csv",
    }
    write_frame(outputs["flow"], flow_rows)
    write_frame(outputs["outcome_support"], outcome_rows)
    write_frame(outputs["intervals"], intervals)
    if assay_rows:
        outputs["assay_support"] = args.output_dir / f"{cohort}-assay-support.csv"
        write_frame(outputs["assay_support"], assay_rows)
    manifest_path = args.output_dir / f"{cohort}-probe-manifest.json"
    manifest = {
        "analysis_id": pilot["analysis_id"], "cohort": cohort, "pilot_config_sha256": sha256(args.pilot_config),
        "universe_config_sha256": sha256(args.universe_config), "assay_config_sha256": sha256(args.assay_config),
        "lookup_sha256": sha256(lookup_path), "source_audit": source_audit,
        "assay_audit": assay_audit, "outputs": {key: file_sha(path) for key, path in outputs.items()},
        "aggregate_only": True, "respondent_rows_exported": 0,
    }
    write_json(manifest_path, manifest)
    print({"status": "PASS", "cohort": cohort, "outputs": len(outputs)})


if __name__ == "__main__":
    main()
