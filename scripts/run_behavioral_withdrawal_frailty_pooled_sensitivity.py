#!/usr/bin/env python3
"""Pooled individual-level sensitivity with cohort fixed effects on Quartz."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd
from probe_behavior_outcome_feasibility import outcome_values
from run_behavioral_withdrawal_frailty_extension import (
    _analysis_set,
    _extra_frame,
    _fit,
    _support,
)

from behavioral_withdrawal_frailty_core import (
    add_direction_categories,
    load_extension_data,
    load_extension_specs,
    load_lookup,
    multidomain_frame,
    sha256,
)
from cohort_core import (
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
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--design-commit", required=True)
    parser.add_argument("--design-commit-time", required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def _cohort_frame(
    episodes: pd.DataFrame,
    behavior: pd.DataFrame,
    universe: dict[str, Any],
    extension: dict[str, Any],
    cohort: str,
    outcome_id: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    outcome, _ = outcome_values(
        episodes, outcome_id, universe["outcomes"][outcome_id], universe["outcomes"], cohort
    )
    extra = _extra_frame(behavior, None)
    data, y, X = _analysis_set(
        episodes, behavior, universe, extension, cohort, outcome_id, outcome, extra,
        ["any_withdrawal"],
    )
    X = X.rename(columns={
        column: f"{cohort}_{column}" for column in X if column.startswith("t1_wave_")
    })
    data = data.copy()
    data["cohort"] = cohort
    data["person_id"] = cohort + "|" + data["person_id"].astype(str)
    data["any_withdrawal"] = behavior.loc[data.index, "any_withdrawal"].astype(float)
    return data.reset_index(drop=True), y.reset_index(drop=True), X.reset_index(drop=True)


def pooled_rows(
    cohort_data: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    extension: dict[str, Any],
    outcome_id: str,
) -> list[dict[str, Any]]:
    data = pd.concat([item[0] for item in cohort_data.values()], ignore_index=True)
    y = pd.concat([item[1] for item in cohort_data.values()], ignore_index=True)
    X = pd.concat([item[2] for item in cohort_data.values()], ignore_index=True, sort=False).fillna(0.0)
    cohort_dummies = pd.get_dummies(data["cohort"], prefix="cohort", drop_first=True, dtype=float)
    X = pd.concat([X, cohort_dummies], axis=1)
    rows = []
    for model_id, drop_current in [
        ("pooled_current_state_adjusted", False),
        ("pooled_without_current_states", True),
    ]:
        design = X.copy()
        if drop_current:
            design = design.drop(columns=[
                column for column in ["alcohol_t1", "activity_t1", "work_t1"] if column in design
            ])
        design["any_withdrawal"] = data["any_withdrawal"]
        status, counts = _support(data, y, design, ["any_withdrawal"], extension)
        base = {
            "outcome_id": outcome_id, "analysis_family": "pooled_fixed_effects_sensitivity",
            "model_id": model_id, "cohorts": ";".join(sorted(cohort_data)), **counts,
        }
        if status != "ESTIMABLE":
            rows.append({**base, "model_status": status})
            continue
        try:
            fit, warning_text = _fit(data, y, design)
            rows.extend([
                {**base, "model_status": "PASS", "warnings": warning_text, **row}
                for row in coefficient_rows(fit, ["any_withdrawal"], True)
            ])
        except Exception as exc:
            rows.append({
                **base, "model_status": "MODEL_FAILURE",
                "failure_reason": f"{type(exc).__name__}: {exc}",
            })
    return rows


def main() -> None:
    args = arguments()
    universe, multidomain, extension = load_extension_specs(
        args.universe_config, args.multidomain_config, args.extension_config
    )
    root = Path(extension["release_root"])
    lookup, lookup_path = load_lookup(root, universe)
    all_rows = []
    input_audits: dict[str, Any] = {}
    for outcome_id in extension["specificity_outcomes"]:
        cohort_data = {}
        for cohort in extension["specificity_cohorts"]:
            probe_manifest = args.probe_dir / f"{cohort}-probe-manifest.json"
            if not probe_manifest.exists():
                raise RuntimeError(f"{cohort}: probe manifest missing")
            episodes, _, _, _, audit = load_extension_data(root, universe, cohort, lookup)
            behavior = add_direction_categories(multidomain_frame(episodes, universe, multidomain, cohort))
            data, y, X = _cohort_frame(
                episodes, behavior, universe, extension, cohort, outcome_id
            )
            if len(data):
                cohort_data[cohort] = (data, y, X)
            input_audits[cohort] = audit
        all_rows.extend(pooled_rows(cohort_data, extension, outcome_id))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(args.output_dir, 0o700)
    output = args.output_dir / "pooled-fixed-effects-sensitivity.csv"
    write_frame(output, all_rows)
    manifest = {
        "analysis_id": extension["analysis_id"], "mode": "pooled_sensitivity",
        "design_commit": args.design_commit, "design_commit_time": args.design_commit_time,
        "code_commit": args.code_commit, "design_sha256": sha256(args.design),
        "extension_config_sha256": sha256(args.extension_config),
        "lookup_sha256": sha256(lookup_path), "input_audits": input_audits,
        "outputs": {"pooled": file_sha(output)},
        "aggregate_only": True, "respondent_rows_exported": 0,
    }
    write_json(args.output_dir / "pooled-sensitivity-manifest.json", manifest)
    print({"status": "PASS", "rows": len(all_rows)})


if __name__ == "__main__":
    main()
