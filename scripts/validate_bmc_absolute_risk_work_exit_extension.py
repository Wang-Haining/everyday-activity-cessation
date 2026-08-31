#!/usr/bin/env python3
"""Independently validate aggregate BMC extension outputs against the parent pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension-dir", required=True, type=Path)
    parser.add_argument("--parent-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _parent_row(parent: pd.DataFrame, cohort: str, outcome: str, exposure: str) -> pd.Series:
    rows = parent.loc[
        parent["cohort"].eq(cohort)
        & parent["outcome_id"].eq(outcome)
        & parent["scope"].eq("comparable_22_30_months")
        & parent["adjustment"].eq("full")
        & parent["exposure_model"].eq(exposure)
    ]
    if rows.empty:
        raise RuntimeError(f"missing parent row: {cohort} {outcome} {exposure}")
    return rows.iloc[0]


def _same_number(left, right) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    return bool(np.isclose(float(left), float(right), rtol=0, atol=1e-10))


def main() -> None:
    args = arguments()
    parent = pd.read_csv(args.parent_results, low_memory=False)
    risks = pd.read_csv(args.extension_dir / "cohort_standardized_risks.csv", low_memory=False)
    pooled_risks = pd.read_csv(args.extension_dir / "pooled_standardized_risks.csv", low_memory=False)
    contrasts = pd.read_csv(args.extension_dir / "cohort_work_exit_contrasts.csv", low_memory=False)
    pooled_contrasts = pd.read_csv(args.extension_dir / "pooled_work_exit_contrasts.csv", low_memory=False)
    intervals = pd.read_csv(args.extension_dir / "comparable_interval_provenance.csv", low_memory=False)

    details: list[dict] = []
    risk_models = risks.drop_duplicates(["cohort", "outcome_id"])
    for row in risk_models.itertuples(index=False):
        parent_row = _parent_row(parent, row.cohort, row.outcome_id, "score_categorical")
        expected_status = "PASS" if parent_row.model_status == "PASS" else parent_row.model_status
        observed_statuses = set(
            risks.loc[
                risks["cohort"].eq(row.cohort) & risks["outcome_id"].eq(row.outcome_id),
                "model_status",
            ]
        )
        observed_status = "PASS" if "PASS" in observed_statuses else next(iter(observed_statuses))
        details.append({
            "check": "risk_parent_reproduction",
            "cohort": row.cohort,
            "outcome_id": row.outcome_id,
            "status_match": observed_status == expected_status,
            "n_match": _same_number(row.n, parent_row.n),
            "events_match": _same_number(row.events, parent_row.events),
        })

    for row in contrasts.itertuples(index=False):
        if row.model_status == "NOT_EVALUABLE_RETIREMENT_NOT_MEASURED":
            continue
        parent_row = _parent_row(parent, row.cohort, row.outcome_id, "work_exit_phenotype")
        expected_status = "PASS" if parent_row.model_status == "PASS" else parent_row.model_status
        check = {
            "check": "work_exit_parent_reproduction",
            "cohort": row.cohort,
            "outcome_id": row.outcome_id,
            "status_match": row.model_status == expected_status,
            "n_match": _same_number(row.n, parent_row.n),
            "events_match": _same_number(row.events, parent_row.events),
        }
        if row.model_status == "PASS":
            expected_ratio = row.other_exit_rr_vs_continued_work / row.retirement_exit_rr_vs_continued_work
            check["direct_contrast_match"] = bool(
                np.isclose(row.estimate, expected_ratio, rtol=0, atol=1e-12)
            )
        details.append(check)

    passed_risks = risks.loc[risks["model_status"].eq("PASS")].copy()
    scenario_complete = True
    for _, group in passed_risks.groupby(["cohort", "outcome_id"]):
        scenario_complete &= set(group["scenario"]) == {
            "withdrawal_0", "withdrawal_1", "withdrawal_2plus"
        }

    expected_k = {
        outcome: int(group["cohort"].nunique())
        for outcome, group in passed_risks.groupby("outcome_id")
    }
    pooled_k_match = all(
        int(row.k_cohorts) == expected_k[row.outcome_id]
        for row in pooled_risks.itertuples(index=False)
    )
    manifest_paths = sorted((args.extension_dir.parent / "models").glob("*-manifest.json"))
    manifests = [json.loads(path.read_text()) for path in manifest_paths]
    design_times = {item["design_commit_time"] for item in manifests}
    code_commits = {item["code_commit"] for item in manifests}

    detail_checks = [
        all(value for key, value in row.items() if key.endswith("_match"))
        for row in details
    ]
    checks = {
        "parent_counts_status_reproduced": bool(all(detail_checks)),
        "all_three_scenarios_present_per_passing_model": bool(scenario_complete),
        "all_probabilities_in_range": bool(
            passed_risks["standardized_risk"].between(0, 1).all()
            and passed_risks["ci_low"].between(0, 1).all()
            and passed_risks["ci_high"].between(0, 1).all()
        ),
        "pooled_k_matches_contributing_cohorts": bool(pooled_k_match),
        "single_design_commit_time": len(design_times) == 1,
        "single_code_commit": len(code_commits) == 1,
        "intervals_within_frozen_window": bool(
            intervals.loc[intervals["scheduled_followup_months"].notna(), "scheduled_followup_months"]
            .between(22, 30).all()
        ),
        "mortality_meta_k1_not_pooled": bool(
            pooled_contrasts.loc[
                pooled_contrasts["outcome_id"].eq("mortality"), "model_status"
            ].eq("NOT_EVALUABLE_META_K1").all()
        ),
    }
    output = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "details": details,
        "standardized_risk_contributors": {
            outcome: sorted(group["cohort"].unique().tolist())
            for outcome, group in passed_risks.groupby("outcome_id")
        },
        "work_exit_statuses": [
            {
                "outcome_id": outcome,
                "model_status": status,
                "cohorts": sorted(group["cohort"].tolist()),
            }
            for (outcome, status), group in contrasts.groupby(["outcome_id", "model_status"])
        ],
    }
    args.output.write_text(json.dumps(output, indent=2, default=str) + "\n")
    if output["status"] != "PASS":
        raise RuntimeError("independent validation failed")
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
