#!/usr/bin/env python3
"""Does the gradient depend on where the age floor is put?

The protocol takes adults aged 60 or older at the landmark interview, which is
the WHO threshold and the one these cohorts are built around. A reviewer working
from a high-income-country convention will ask for 65. This refits the primary
graded models at 60, 65 and 70 on the same protocol, same covariates, same
support gates, and reports what each floor costs in intervals and events.

Aggregate only. Nothing but counts, estimates and standard errors is written.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from probe_behavior_outcome_feasibility import outcome_values
from run_behavioral_withdrawal_competing_context import _strip_state
from run_behavioral_withdrawal_frailty_extension import _analysis_set, _extra_frame, _run_model

from behavioral_withdrawal_frailty_core import (
    add_direction_categories,
    load_extension_data,
    load_extension_specs,
    load_lookup,
    multidomain_frame,
    sha256,
)
from cohort_core import file_sha, write_frame, write_json

OUTCOMES = ["mortality", "incident_any_adl", "incident_any_iadl",
            "multimorbidity_progression"]
GRADED = ["loss_1", "loss_2plus"]
FLOORS = [60, 65, 70]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe-config", required=True, type=Path)
    parser.add_argument("--multidomain-config", required=True, type=Path)
    parser.add_argument("--extension-config", required=True, type=Path)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def age_floor_models(
    episodes: pd.DataFrame,
    behavior: pd.DataFrame,
    universe: dict[str, Any],
    extension: dict[str, Any],
    cohort: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    extra = _extra_frame(behavior, None)
    for outcome_id in OUTCOMES:
        outcome, coding = outcome_values(
            episodes, outcome_id, universe["outcomes"][outcome_id],
            universe["outcomes"], cohort,
        )
        data, y, X = _analysis_set(
            episodes, behavior, universe, extension, cohort,
            outcome_id, outcome, extra, GRADED,
        )
        # The published design does not carry behavioural status at the landmark
        # interview; strip it so these estimates are comparable to the paper's.
        X = _strip_state(X, True)
        for floor in FLOORS:
            keep = data.index[data["age"].ge(floor)]
            base = {
                "cohort": cohort, "outcome_id": outcome_id,
                "analysis_family": "age_floor", "age_floor": floor,
                "outcome_coding_status": coding,
                "median_age": float(data.loc[keep, "age"].median()) if len(keep) else float("nan"),
            }
            result, _, _ = _run_model(
                base, data.loc[keep], y.loc[keep], X.loc[keep],
                GRADED, GRADED, extension,
            )
            rows.extend(result)
    return rows


def age_distribution(
    episodes: pd.DataFrame,
    behavior: pd.DataFrame,
    universe: dict[str, Any],
    cohort: str,
) -> list[dict[str, Any]]:
    eligible = (
        behavior["core_valid"]
        & behavior["baseline_engagement_count"].ge(1)
        & episodes["comparable_window"].fillna(False)
    )
    age = pd.to_numeric(episodes.loc[eligible, "t1__agey"], errors="coerce") \
        if "t1__agey" in episodes else None
    if age is None or age.notna().sum() == 0:
        for candidate in ["t1__age", "age_t1", "t1__ragey_e", "t1__agey_e"]:
            if candidate in episodes:
                age = pd.to_numeric(episodes.loc[eligible, candidate], errors="coerce")
                if age.notna().sum():
                    break
    if age is None:
        return []
    total = int(age.notna().sum())
    return [{
        "cohort": cohort, "eligible_intervals": total,
        "median_age": float(age.median()),
        "pct_65_plus": round(float(age.ge(65).sum()) / total * 100, 1) if total else float("nan"),
        "pct_70_plus": round(float(age.ge(70).sum()) / total * 100, 1) if total else float("nan"),
    }]


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

    outputs = {}
    outputs["age_floor"] = args.output_dir / f"{cohort}-age-floor-models.csv"
    write_frame(outputs["age_floor"], age_floor_models(
        episodes, behavior, universe, extension, cohort))
    rows = age_distribution(episodes, behavior, universe, cohort)
    if rows:
        outputs["age_distribution"] = args.output_dir / f"{cohort}-age-distribution.csv"
        write_frame(outputs["age_distribution"], rows)

    write_json(args.output_dir / f"{cohort}-age-floor-manifest.json", {
        "analysis_id": "behavioral_withdrawal_age_floor_v0.1",
        "cohort": cohort,
        "code_commit": args.code_commit,
        "age_floors": FLOORS,
        "universe_config_sha256": sha256(args.universe_config),
        "multidomain_config_sha256": sha256(args.multidomain_config),
        "lookup_sha256": sha256(lookup_path),
        "input_audit": input_audit,
        "support_gates_unchanged": True,
        "negative_results_retained": True,
        "aggregate_only": True,
        "respondent_rows_exported": 0,
        "outputs": {k: file_sha(v) for k, v in outputs.items()},
    })
    print(json.dumps({"status": "PASS", "cohort": cohort, "outputs": len(outputs)}))


if __name__ == "__main__":
    main()
