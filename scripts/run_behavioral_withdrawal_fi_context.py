#!/usr/bin/env python3
"""Fit concurrent-health contrasts that include the frailty index."""
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from behavioral_contraction_biomarker_core import write_frame
from behavioral_withdrawal_frailty_core import (
    load_extension_specs, load_extension_data, load_lookup, load_source_components,
    load_fi_long, build_frailty_frame, build_context_frame, add_direction_categories,
    multidomain_frame,
)
from probe_behavior_outcome_feasibility import outcome_values
from run_behavioral_withdrawal_frailty_extension import _analysis_set, _extra_frame, _run_model
from run_behavioral_withdrawal_competing_context import (
    arguments, _strip_state, GRADED, CONTEXT_OUTCOMES, CONTEXT_EXCLUSIONS,
)
CONTEXT_FIELDS = [
    "self_rated_health_worsening", "transition_hospitalization", "cesd_worsening",
    "bmi_or_weight_loss", "incident_non_target_disease", "fi_change",
]

def context_gradient_models(
    episodes: pd.DataFrame,
    behavior: pd.DataFrame,
    context: pd.DataFrame,
    universe: dict[str, Any],
    extension: dict[str, Any],
    cohort: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    extra_base = _extra_frame(behavior, None)
    for outcome_id in CONTEXT_OUTCOMES:
        outcome, coding = outcome_values(
            episodes, outcome_id, universe["outcomes"][outcome_id], universe["outcomes"], cohort
        )
        excluded = CONTEXT_EXCLUSIONS.get(outcome_id, set())
        extra = pd.concat([extra_base, context], axis=1)
        available = [
            field for field in CONTEXT_FIELDS
            if field not in excluded and extra[field].notna().any()
        ]
        if not available:
            continue
        without_fi = [field for field in available if field != "fi_change"]
        combinations = [(field, [field]) for field in available]
        combinations.append(("joint_context", available))
        if without_fi and without_fi != available:
            combinations.append(("joint_context_no_fi", without_fi))
        for context_id, fields in combinations:
            required = [*GRADED, *fields]
            data, y, X = _analysis_set(
                episodes, behavior, universe, extension, cohort,
                outcome_id, outcome, extra, required,
            )
            base = {
                "cohort": cohort, "outcome_id": outcome_id,
                "analysis_family": "clinical_context_graded",
                "context_id": context_id, "outcome_coding_status": coding,
                "context_fields": ";".join(fields),
            }
            X_arm = _strip_state(X, True)
            base_rows, base_fit, _ = _run_model(
                {**base, "model_id": f"base_same_set_{context_id}",
                 "state_adjustment": "unconditional"},
                data, y, X_arm, GRADED, GRADED, extension,
            )
            rows.extend(base_rows)
            # The same base with current behavioural status held, so the
            # attenuation from a context variable can be read separately from
            # the attenuation from current status itself.
            state_rows, state_fit, _ = _run_model(
                {**base, "model_id": f"base_same_set_{context_id}",
                 "state_adjustment": "current_state_held"},
                data, y, X, GRADED, GRADED, extension,
            )
            if base_fit is not None and state_fit is not None:
                for row in state_rows:
                    b0 = float(base_fit.params.get(row.get("term"), np.nan))
                    b1 = float(state_fit.params.get(row.get("term"), np.nan))
                    row["log_rr_attenuation_percent"] = (
                        float((b0 - b1) / b0 * 100) if abs(b0) > 1e-12 else np.nan
                    )
            rows.extend(state_rows)
            for field in fields:
                # The frailty index moves on a 0-1 scale, so a one-unit
                # coefficient is uninterpretable; report it per 0.05.
                scale = 0.05 if field == "fi_change" else 1.0
                data[f"context_{field}"] = pd.to_numeric(data[field], errors="coerce") / scale
            context_terms = [f"context_{field}" for field in fields]
            plus_rows, plus_fit, _ = _run_model(
                {**base, "model_id": f"plus_{context_id}",
                 "state_adjustment": "unconditional"},
                data, y, X_arm, [*GRADED, *context_terms], GRADED, extension,
            )
            if base_fit is not None and plus_fit is not None:
                attenuation = {}
                for term in GRADED:
                    b0 = float(base_fit.params[term])
                    b1 = float(plus_fit.params[term])
                    attenuation[term] = (
                        float((b0 - b1) / b0 * 100) if abs(b0) > 1e-12 else float("nan")
                    )
                for row in plus_rows:
                    row["log_rr_attenuation_percent"] = attenuation.get(row.get("term"), np.nan)
            rows.extend(plus_rows)
    return rows


def main():
    args = arguments()
    assert args.data_root.is_dir(), f"missing data root: {args.data_root}"
    universe, multidomain, extension = load_extension_specs(
        args.universe_config, args.multidomain_config, args.extension_config
    )
    cohort = args.cohort.lower()
    assert cohort in extension["frailty_cohorts"], f"no frailty data for cohort: {cohort}"
    lookup, _ = load_lookup(args.data_root, universe)
    episodes, formal, _, _, _ = load_extension_data(args.data_root, universe, cohort, lookup)
    behavior = add_direction_categories(multidomain_frame(episodes, universe, multidomain, cohort))
    source, _ = load_source_components(args.data_root, extension, cohort)
    fi_long, _ = load_fi_long(extension, cohort, args.data_root)
    frailty = build_frailty_frame(episodes, formal, source, fi_long, universe, extension, cohort)
    context = build_context_frame(episodes, frailty, universe, cohort)
    rows = context_gradient_models(episodes, behavior, context, universe, extension, cohort)
    rows = [row for row in rows if row["context_id"] in {"fi_change", "joint_context"}]
    assert rows, f"empty concurrent-health rows for {cohort}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_frame(args.output_dir / f"{cohort}-context-graded-models.csv", rows)
    print({"cohort": cohort, "aggregate_rows": len(rows)})


if __name__ == "__main__":
    main()
