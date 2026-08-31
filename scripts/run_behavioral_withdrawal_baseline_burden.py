#!/usr/bin/env python3
"""Who the people who stopped were, before anything was stopped.

The manuscript argues that a cessation is how a decline becomes visible rather
than a cause of one. That argument is currently made from the external
literature and from the shape of the results. It can be made directly: if the
illness comes first, then respondents who stopped two or more activities should
already be carrying more disease at the interview that identifies the cessation,
and should already have been getting worse over the window in which they stopped.

Reports both, by cohort and by cessation group. Levels at the landmark interview
and change across the transition window. Aggregate only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from probe_behavior_outcome_feasibility import binary

from behavioral_withdrawal_frailty_core import (
    COMMON_DISEASE_FIELDS,
    add_direction_categories,
    build_context_frame,
    build_frailty_frame,
    load_extension_data,
    load_extension_specs,
    load_fi_long,
    load_lookup,
    load_source_components,
    multidomain_frame,
    sha256,
)
from cohort_core import file_sha, write_frame, write_json


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe-config", required=True, type=Path)
    parser.add_argument("--multidomain-config", required=True, type=Path)
    parser.add_argument("--extension-config", required=True, type=Path)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def _mean(series: pd.Series, mask: pd.Series) -> float:
    values = pd.to_numeric(series[mask], errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")


def _pct(series: pd.Series, mask: pd.Series) -> float:
    values = pd.to_numeric(series[mask], errors="coerce").dropna()
    return float(values.mean() * 100) if len(values) else float("nan")


def _n_observed(series: pd.Series, mask: pd.Series) -> int:
    return int(pd.to_numeric(series[mask], errors="coerce").notna().sum())


def burden_rows(
    episodes: pd.DataFrame,
    behavior: pd.DataFrame,
    frailty: pd.DataFrame | None,
    context: pd.DataFrame | None,
    universe: dict[str, Any],
    cohort: str,
) -> list[dict[str, Any]]:
    eligible = (
        behavior["core_valid"]
        & behavior["baseline_engagement_count"].ge(1)
        & episodes["comparable_window"].fillna(False)
    )

    # Level at the interview that identifies the cessation: how many of the six
    # common conditions the respondent already carried.
    conditions = pd.concat(
        [binary(episodes[f"t1__{field}"]) for field in COMMON_DISEASE_FIELDS.values()],
        axis=1,
    )
    condition_count = conditions.sum(axis=1).where(conditions.notna().all(axis=1))

    # The same count at the first interview, so the reader can see the burden was
    # already there before the window in which the activity stopped.
    conditions0 = pd.concat(
        [binary(episodes[f"t0__{field}"]) for field in COMMON_DISEASE_FIELDS.values()],
        axis=1,
    )
    condition_count0 = conditions0.sum(axis=1).where(conditions0.notna().all(axis=1))

    fields: dict[str, pd.Series] = {
        "conditions_at_first_interview": condition_count0,
        "conditions_at_landmark": condition_count,
        "conditions_gained_across_window": condition_count - condition_count0,
    }
    if frailty is not None:
        fields["frailty_index_at_landmark"] = frailty["fi_t1"]
        fields["frailty_index_change"] = frailty["fi_change"]
    if context is not None:
        for name in ["self_rated_health_worsening", "transition_hospitalization",
                     "cesd_worsening", "bmi_or_weight_loss"]:
            fields[name] = context[name]

    rows = []
    groups = [("none", behavior["loss_count_core"].eq(0)),
              ("one", behavior["loss_count_core"].eq(1)),
              ("two_or_more", behavior["loss_count_core"].ge(2))]
    for label, member in groups:
        mask = eligible & member.fillna(False)
        row = {
            "cohort": cohort, "cessation_group": label,
            "n_intervals": int(mask.sum()),
            "people": int(episodes.loc[mask, "person_id"].nunique()),
        }
        for name, series in fields.items():
            observed = _n_observed(series, mask)
            row[f"{name}_n"] = observed
            if observed == 0:
                row[name] = float("nan")
            elif name in {"self_rated_health_worsening", "transition_hospitalization",
                          "cesd_worsening", "bmi_or_weight_loss"}:
                row[name] = round(_pct(series, mask), 1)
            else:
                row[name] = round(_mean(series, mask), 3)
        rows.append(row)
    return rows


def main() -> None:
    args = arguments()
    universe, multidomain, extension = load_extension_specs(
        args.universe_config, args.multidomain_config, args.extension_config
    )
    cohort = args.cohort.lower()
    root = Path(extension["release_root"])
    lookup, lookup_path = load_lookup(root, universe)
    episodes, formal, status, _, input_audit = load_extension_data(root, universe, cohort, lookup)
    behavior = add_direction_categories(multidomain_frame(episodes, universe, multidomain, cohort))

    frailty = context = None
    if cohort in extension["frailty_cohorts"]:
        source, _ = load_source_components(root, extension, cohort)
        fi_long, _ = load_fi_long(extension, cohort)
        frailty = build_frailty_frame(episodes, formal, source, fi_long, universe, extension, cohort)
        context = build_context_frame(episodes, frailty, universe, cohort)

    out = args.output_dir / f"{cohort}-baseline-burden.csv"
    write_frame(out, burden_rows(episodes, behavior, frailty, context, universe, cohort))
    write_json(args.output_dir / f"{cohort}-baseline-burden-manifest.json", {
        "analysis_id": "behavioral_withdrawal_baseline_burden_v0.1",
        "cohort": cohort,
        "code_commit": args.code_commit,
        "universe_config_sha256": sha256(args.universe_config),
        "multidomain_config_sha256": sha256(args.multidomain_config),
        "lookup_sha256": sha256(lookup_path),
        "input_audit": input_audit,
        "descriptive_only": True,
        "aggregate_only": True,
        "respondent_rows_exported": 0,
        "outputs": {"burden": file_sha(out)},
    })
    print(json.dumps({"status": "PASS", "cohort": cohort}))


if __name__ == "__main__":
    main()
