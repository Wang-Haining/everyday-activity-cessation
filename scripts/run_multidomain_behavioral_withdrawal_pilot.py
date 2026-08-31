#!/usr/bin/env python3
"""Run counts-only or gated models for one cohort."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from probe_behavior_outcome_feasibility import load_lookup, sha256

from multidomain_behavioral_withdrawal_core import (
    coefficient_rows,
    file_sha,
    fit_clustered,
    load_episodes,
    load_specs,
    multidomain_frame,
    outcome_and_baseline,
    prepare_model,
    scope_mask,
    support_gate,
    write_frame,
    write_json,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["probe", "models"])
    parser.add_argument("--universe-config", required=True, type=Path)
    parser.add_argument("--pilot-config", required=True, type=Path)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--probe-dir", type=Path)
    parser.add_argument("--design-commit", required=True)
    parser.add_argument("--design-commit-time", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--minimum-baseline-opportunities", type=int, choices=[1, 2, 3], default=1)
    return parser.parse_args()


def flow_rows(episodes: pd.DataFrame, behavior: pd.DataFrame, cohort: str) -> list[dict]:
    rows = []
    for scope in ["comparable_22_30_months", "all_primary_wave_intervals"]:
        base = scope_mask(episodes, scope)
        core = base & behavior["core_valid"] & behavior["baseline_engagement_count"].ge(1)
        row = {
            "cohort": cohort,
            "scope": scope,
            "core_intervals": int(core.sum()),
            "core_people": int(episodes.loc[core, "person_id"].nunique()),
        }
        for count in [0, 1, 2, 3]:
            row[f"core_loss_{count}_n"] = int((core & behavior["loss_count_core"].eq(count)).sum())
        for field in ["alcohol_loss", "activity_loss", "work_loss"]:
            row[f"{field}_n"] = int((core & behavior[field].eq(1)).sum())
        extended = base & behavior["extended_valid"]
        if extended.any():
            extended &= behavior["baseline_engagement_count_extended"].ge(1)
            row["extended_intervals"] = int(extended.sum())
            for count in [0, 1, 2, 3, 4]:
                row[f"extended_loss_{count}_n"] = int(
                    (extended & behavior["loss_count_extended"].eq(count)).sum()
                )
        work = base & behavior["work_exit_risk"]
        if work.any():
            row["work_exit_risk_n"] = int(work.sum())
            row["work_exit_retirement_n"] = int((work & behavior["work_exit_retirement"].eq(1)).sum())
            row["work_exit_no_retirement_n"] = int(
                (work & behavior["work_exit_no_retirement"].eq(1)).sum()
            )
        rows.append(row)
    return rows


def support_rows(
    episodes: pd.DataFrame,
    behavior: pd.DataFrame,
    universe: dict,
    pilot: dict,
    cohort: str,
) -> list[dict]:
    rows = []
    outcomes = pilot["primary_binary_outcomes"] + pilot["secondary_binary_outcomes"]
    for scope in ["comparable_22_30_months", "all_primary_wave_intervals"]:
        for outcome_id in outcomes:
            outcome, _, coding_status, _ = outcome_and_baseline(episodes, universe, cohort, outcome_id)
            base = (
                scope_mask(episodes, scope)
                & behavior["core_valid"]
                & behavior["baseline_engagement_count"].ge(1)
                & outcome.notna()
            )
            row = {
                "cohort": cohort,
                "scope": scope,
                "outcome_id": outcome_id,
                "outcome_coding_status": coding_status,
                "n": int(base.sum()),
                "people": int(episodes.loc[base, "person_id"].nunique()),
                "events": int(outcome.loc[base].sum()),
            }
            for count, name in [(0, "0"), (1, "1")]:
                cell = base & behavior["loss_count_core"].eq(count)
                row[f"loss_{name}_n"] = int(cell.sum())
                row[f"loss_{name}_events"] = int(outcome.loc[cell].sum())
            cell = base & behavior["loss_count_core"].ge(2)
            row["loss_2plus_n"] = int(cell.sum())
            row["loss_2plus_events"] = int(outcome.loc[cell].sum())
            row["score_support"] = (
                "ESTIMABLE"
                if row["loss_1_events"] >= pilot["minimum_exposed_events"]
                and row["loss_2plus_events"] >= pilot["minimum_exposed_events"]
                and row["loss_1_n"] - row["loss_1_events"] >= pilot["minimum_exposed_nonevents"]
                and row["loss_2plus_n"] - row["loss_2plus_events"] >= pilot["minimum_exposed_nonevents"]
                else "NOT_EVALUABLE_CATEGORY_SUPPORT"
            )
            rows.append(row)
    return rows


def model_rows(
    episodes: pd.DataFrame,
    behavior: pd.DataFrame,
    universe: dict,
    pilot: dict,
    cohort: str,
    minimum_baseline_opportunities: int,
) -> list[dict]:
    rows = []
    binary_outcomes = pilot["primary_binary_outcomes"] + pilot["secondary_binary_outcomes"]
    continuous_outcomes = pilot["secondary_continuous_outcomes"]
    core_models = ["any_withdrawal", "score_categorical", "mutually_adjusted_components"]
    for scope in ["comparable_22_30_months", "all_primary_wave_intervals"]:
        for outcome_id in binary_outcomes + continuous_outcomes:
            for adjustment in ["basic", "full"]:
                for exposure_model in core_models:
                    base = {
                        "cohort": cohort,
                        "scope": scope,
                        "outcome_id": outcome_id,
                        "adjustment": adjustment,
                        "exposure_model": exposure_model,
                        "analysis_family": "core",
                        "minimum_baseline_opportunities": minimum_baseline_opportunities,
                    }
                    try:
                        data, y, X, terms, binary_outcome, coding_status = prepare_model(
                            episodes, behavior, universe, pilot, cohort, outcome_id,
                            scope, adjustment, exposure_model, minimum_baseline_opportunities,
                        )
                        base["outcome_coding_status"] = coding_status
                        status, counts = support_gate(data, y, X, terms, pilot, binary_outcome)
                        if status != "ESTIMABLE":
                            rows.append({**base, **counts, "model_status": status})
                            continue
                        fit, warning_text = fit_clustered(y, X, data["person_id"], binary_outcome)
                        estimates = coefficient_rows(fit, terms, binary_outcome)
                        outcome_sd = float(y.std(ddof=1)) if not binary_outcome else float("nan")
                        for estimate in estimates:
                            if not binary_outcome and outcome_sd > 0:
                                estimate.update({
                                    "standardized_estimate": estimate["estimate"] / outcome_sd,
                                    "standardized_standard_error": estimate["standard_error"] / outcome_sd,
                                    "standardized_ci_low": estimate["ci_low"] / outcome_sd,
                                    "standardized_ci_high": estimate["ci_high"] / outcome_sd,
                                })
                            rows.append({
                                **base, **counts, "model_status": "PASS",
                                "warnings": warning_text, **estimate,
                            })
                    except Exception as exc:
                        rows.append({
                            **base, "model_status": "MODEL_FAILURE",
                            "failure_reason": f"{type(exc).__name__}: {exc}",
                        })

            if outcome_id in pilot["primary_binary_outcomes"]:
                enrichment_models = []
                if cohort in sum(pilot["extended_social_cohorts"].values(), []):
                    enrichment_models.append("extended_score_categorical")
                if cohort in pilot["retirement_context_cohorts"]:
                    enrichment_models.append("work_exit_phenotype")
                for exposure_model in enrichment_models:
                    base = {
                        "cohort": cohort,
                        "scope": scope,
                        "outcome_id": outcome_id,
                        "adjustment": "full",
                        "exposure_model": exposure_model,
                        "analysis_family": "enrichment",
                        "minimum_baseline_opportunities": pilot["minimum_baseline_opportunities"],
                    }
                    try:
                        data, y, X, terms, binary_outcome, coding_status = prepare_model(
                            episodes, behavior, universe, pilot, cohort, outcome_id,
                            scope, "full", exposure_model,
                        )
                        base["outcome_coding_status"] = coding_status
                        status, counts = support_gate(data, y, X, terms, pilot, binary_outcome)
                        if status != "ESTIMABLE":
                            rows.append({**base, **counts, "model_status": status})
                            continue
                        fit, warning_text = fit_clustered(y, X, data["person_id"], binary_outcome)
                        for estimate in coefficient_rows(fit, terms, binary_outcome):
                            rows.append({
                                **base, **counts, "model_status": "PASS",
                                "warnings": warning_text, **estimate,
                            })
                    except Exception as exc:
                        rows.append({
                            **base, "model_status": "MODEL_FAILURE",
                            "failure_reason": f"{type(exc).__name__}: {exc}",
                        })
    return rows


def main() -> None:
    args = arguments()
    universe, pilot = load_specs(args.universe_config, args.pilot_config)
    cohort = args.cohort.lower()
    if cohort not in pilot["candidate_cohorts"]:
        raise RuntimeError("cohort outside frozen pilot")
    root = Path(pilot["release_root"])
    lookup, lookup_path = load_lookup(root, universe)
    episodes, intervals, source_audit = load_episodes(root, universe, cohort, lookup)
    behavior = multidomain_frame(episodes, universe, pilot, cohort)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "probe":
        outputs = {
            "flow": args.output_dir / f"{cohort}-flow.csv",
            "support": args.output_dir / f"{cohort}-support.csv",
            "intervals": args.output_dir / f"{cohort}-intervals.csv",
        }
        write_frame(outputs["flow"], flow_rows(episodes, behavior, cohort))
        write_frame(outputs["support"], support_rows(episodes, behavior, universe, pilot, cohort))
        write_frame(outputs["intervals"], intervals)
    else:
        if args.probe_dir is None or not (args.probe_dir / f"{cohort}-probe-manifest.json").exists():
            raise RuntimeError("counts-only probe manifest missing")
        outputs = {"models": args.output_dir / f"{cohort}-models.csv"}
        write_frame(
            outputs["models"],
            model_rows(
                episodes, behavior, universe, pilot, cohort,
                args.minimum_baseline_opportunities,
            ),
        )

    manifest_path = args.output_dir / f"{cohort}-{args.mode[:-1] if args.mode.endswith('s') else args.mode}-manifest.json"
    write_json(manifest_path, {
        "analysis_id": pilot["analysis_id"],
        "mode": args.mode,
        "cohort": cohort,
        "design_commit": args.design_commit,
        "design_commit_time": args.design_commit_time,
        "code_commit": args.code_commit,
        "design_sha256": sha256(args.design),
        "pilot_config_sha256": sha256(args.pilot_config),
        "universe_config_sha256": sha256(args.universe_config),
        "lookup_sha256": sha256(lookup_path),
        "source_audit": source_audit,
        "outputs": {key: file_sha(path) for key, path in outputs.items()},
        "aggregate_only": True,
        "respondent_rows_exported": 0,
    })
    print({"status": "PASS", "mode": args.mode, "cohort": cohort, "outputs": len(outputs)})


if __name__ == "__main__":
    main()
