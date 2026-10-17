#!/usr/bin/env python3
"""Fit models under alternative smoking-adjustment specifications."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from behavioral_contraction_biomarker_core import file_sha, write_frame, write_json
from behavioral_withdrawal_frailty_core import (
    add_direction_categories,
    load_extension_data,
    load_extension_specs,
    load_lookup,
    multidomain_frame,
    routine_data_and_design,
    sha256,
)
from probe_behavior_outcome_feasibility import outcome_values
from run_behavioral_withdrawal_frailty_extension import (
    _extra_frame, _full_rank_base, _run_model,
)
from run_behavioral_withdrawal_competing_context import _strip_state

OUTCOMES = ["mortality", "incident_any_adl", "incident_any_iadl",
            "multimorbidity_progression"]
GRADED = ["loss_1", "loss_2plus"]

# Which design columns each variant keeps. A column is kept when any of its
# prefixes matches, because the design matrix carries spline and dummy expansions
# whose names are built from these stems.
MINIMAL = ("age", "sex", "female", "male", "wave", "opportunity", "available")
SMOKING = ("smok",)

VARIANTS = [
    ("all_minimal", MINIMAL, False),
    ("complete_minimal", MINIMAL, True),
    ("complete_full", None, True),
    ("all_nosmoking", "drop_smoking", False),
]


def _columns(X: pd.DataFrame, keep: Any) -> list[str]:
    if keep is None:
        return list(X.columns)
    if keep == "drop_smoking":
        return [c for c in X.columns if not any(s in c.lower() for s in SMOKING)]
    return [c for c in X.columns
            if any(s in c.lower() for s in keep) or c.lower().startswith("loss_")
            or c.lower() in {"const", "intercept"}]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--universe-config", required=True, type=Path)
    parser.add_argument("--multidomain-config", required=True, type=Path)
    parser.add_argument("--extension-config", required=True, type=Path)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def missingness_models(
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
        base_data, base_X = routine_data_and_design(
            episodes, behavior, universe, extension, cohort, outcome_id, outcome
        )
        additions = [c for c in extra.columns if c not in base_data.columns]
        base_data = base_data.join(extra[additions])
        eligible = (behavior["core_valid"]
                    & behavior["baseline_engagement_count"].ge(1)
                    & outcome.notna()
                    & episodes["comparable_window"].fillna(False))
        candidate = base_X.index.intersection(base_data.index[eligible])
        # the primary complete-case rule, so that "complete" means the same
        # thing here as it does in the paper
        primary_complete = (
            base_data.loc[candidate, ["outcome", *GRADED]].notna().all(axis=1)
            & base_X.loc[candidate].notna().all(axis=1)
        )

        for name, keep, complete_case_only in VARIANTS:
            columns = _columns(base_X, keep)
            X = base_X[columns]
            own_complete = (
                base_data.loc[candidate, ["outcome", *GRADED]].notna().all(axis=1)
                & X.loc[candidate].notna().all(axis=1)
            )
            index = candidate[primary_complete if complete_case_only else own_complete]
            if not len(index):
                continue
            data = base_data.loc[index].copy()
            y = pd.to_numeric(data["outcome"], errors="coerce").astype(float)
            design = _strip_state(_full_rank_base(X.loc[index]), True)
            base = {
                "cohort": cohort, "outcome_id": outcome_id,
                "analysis_family": "missingness", "variant": name,
                "covariate_columns": len(design.columns),
                "outcome_coding_status": coding,
                "eligible_intervals": int(len(candidate)),
                "complete_case_intervals": int(primary_complete.sum()),
            }
            result, _, _ = _run_model(base, data, y, design, GRADED, GRADED, extension)
            rows.extend(result)
    return rows


def main() -> None:
    args = arguments()
    assert args.data_root.is_dir(), f"missing data root: {args.data_root}"
    universe, multidomain, extension = load_extension_specs(
        args.universe_config, args.multidomain_config, args.extension_config
    )
    cohort = args.cohort.lower()
    root = args.data_root
    lookup, lookup_path = load_lookup(root, universe)
    episodes, formal, status, _, input_audit = load_extension_data(root, universe, cohort, lookup)
    behavior = add_direction_categories(multidomain_frame(episodes, universe, multidomain, cohort))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{cohort}-missingness-models.csv"
    rows = missingness_models(episodes, behavior, universe, extension, cohort)
    write_frame(out, rows)

    write_json(args.output_dir / f"{cohort}-missingness-manifest.json", {
        "analysis_id": "behavioral_withdrawal_missingness_v0.1",
        "cohort": cohort,
        "variants": [v[0] for v in VARIANTS],
        "universe_config_sha256": sha256(args.universe_config),
        "multidomain_config_sha256": sha256(args.multidomain_config),
        "lookup_sha256": sha256(lookup_path),
        "input_audit": input_audit,
        "support_gates_unchanged": True,
        "negative_results_retained": True,
        "aggregate_only": True,
        "respondent_rows_exported": 0,
        "outputs": {"missingness": file_sha(out)},
    })
    print(json.dumps({"status": "PASS", "cohort": cohort, "rows": len(rows)}))


if __name__ == "__main__":
    main()
