#!/usr/bin/env python3
"""Counts-only feasibility probe for behavioral contraction and later HbA1c."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from behavioral_contraction_biomarker_core import write_frame, write_json
from elsa_behavior_hba1c_early_warning_core import build_frame, load_early_specs, outcome_mask
from probe_behavior_outcome_feasibility import sha256


OUTCOMES = [
    "glycemic_category_progression",
    "incident_diabetes_range_hba1c",
    "incident_prediabetes_or_higher",
    "incident_fasting_glucose_diabetes_range",
]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--universe-config", required=True, type=Path)
    parser.add_argument("--pilot-config", required=True, type=Path)
    parser.add_argument("--assay-config", required=True, type=Path)
    parser.add_argument("--early-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def count_row(data: pd.DataFrame, label: str, mask: pd.Series) -> dict[str, object]:
    return {
        "step": label,
        "person_intervals": int(mask.sum()),
        "participants": int(data.loc[mask, "person_id"].nunique()),
    }


def main() -> None:
    args = arguments()
    assert args.data_root.is_dir(), f"missing data root: {args.data_root}"
    universe, pilot, assay, early = load_early_specs(
        args.universe_config, args.pilot_config, args.assay_config, args.early_config
    )
    data, source_audit = build_frame(args.data_root, universe, pilot, assay, early)

    all_rows = pd.Series(True, index=data.index)
    landmark = data["eligible_landmark"]
    behavior = landmark & data["valid_behavior_pair"]
    opportunity = behavior & data["baseline_engagement_count"].ge(1)
    paired_hba1c = opportunity & data[["baseline_hba1c", "followup_hba1c"]].notna().all(axis=1)
    baseline_below = paired_hba1c & data["baseline_hba1c"].lt(6.5)
    undiagnosed = baseline_below & data["diagnosed_diabetes_t1"].eq(0)
    flow = [
        count_row(data, "age_60_plus_three_wave_episodes", all_rows),
        count_row(data, "eligible_assay_landmarks", landmark),
        count_row(data, "valid_paired_behaviors", behavior),
        count_row(data, "at_least_one_baseline_behavior", opportunity),
        count_row(data, "paired_baseline_followup_hba1c", paired_hba1c),
        count_row(data, "baseline_hba1c_below_6_5", baseline_below),
        count_row(data, "no_recorded_diabetes_at_transition_end", undiagnosed),
    ]

    feasibility: list[dict[str, object]] = []
    raw: list[dict[str, object]] = []
    for outcome in OUTCOMES:
        mask = outcome_mask(data, outcome)
        y = data[outcome]
        for wave, wave_mask in [("all", pd.Series(True, index=data.index))] + [
            (str(int(value)), data["t1_wave"].eq(value))
            for value in sorted(data.loc[mask, "t1_wave"].dropna().unique())
        ]:
            selected = mask & wave_mask
            for exposure_name, exposure in [
                ("any_contraction", data["any_contraction"]),
                ("alcohol_loss", data["alcohol_loss"]),
                ("activity_loss", data["activity_loss"]),
            ]:
                exposed = selected & exposure.eq(1)
                comparator = selected & exposure.eq(0)
                events = int(y.loc[selected].sum())
                exposed_events = int(y.loc[exposed].sum())
                row = {
                    "outcome": outcome,
                    "transition_end_wave": wave,
                    "exposure": exposure_name,
                    "n": int(selected.sum()),
                    "participants": int(data.loc[selected, "person_id"].nunique()),
                    "events": events,
                    "exposed_n": int(exposed.sum()),
                    "exposed_events": exposed_events,
                    "exposed_nonevents": int(exposed.sum()) - exposed_events,
                    "comparator_n": int(comparator.sum()),
                    "comparator_events": int(y.loc[comparator].sum()),
                }
                row["support"] = (
                    "ESTIMABLE_COUNTS"
                    if row["exposed_events"] >= early["minimum_exposed_events"]
                    and row["exposed_nonevents"] >= early["minimum_exposed_nonevents"]
                    else "NOT_EVALUABLE_EXPOSED_SUPPORT"
                )
                feasibility.append(row)
            for loss_count in [0, 1, 2]:
                group = selected & data["loss_count_2"].eq(loss_count)
                n = int(group.sum())
                events = int(y.loc[group].sum())
                raw.append(
                    {
                        "outcome": outcome,
                        "transition_end_wave": wave,
                        "loss_count": loss_count,
                        "n": n,
                        "participants": int(data.loc[group, "person_id"].nunique()),
                        "events": events,
                        "risk": events / n if n else None,
                        "median_followup_months": float(
                            data.loc[group, "observed_followup_months"].median()
                        )
                        if n
                        else None,
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_frame(args.output_dir / "flow.csv", flow)
    write_frame(args.output_dir / "feasibility.csv", feasibility)
    write_frame(args.output_dir / "raw-risk.csv", raw)
    write_json(
        args.output_dir / "validation.json",
        {
            "status": "PASS",
            "analysis_id": early["analysis_id"],
            "counts_only": True,
            "respondent_rows_exported": 0,
            "aggregate_only": True,
            "config_sha256": sha256(args.early_config),
            "source_audit": source_audit,
        },
    )
    print({"status": "PASS", "flow_rows": len(flow), "feasibility_rows": len(feasibility)})


if __name__ == "__main__":
    main()
