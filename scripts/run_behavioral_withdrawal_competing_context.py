#!/usr/bin/env python3
"""Three analyses the clinical reviews asked for and the frozen release cannot answer.

Job 1  Death-or-outcome composite for the three non-fatal graded outcomes. The
       published estimates are among survivors who reached the outcome
       interview; cessation predicts death, so the exposed group loses more
       people before the outcome is measured. The composite counts death as an
       event and reports how many respondents entered through death rather than
       through a measured outcome.

Job 2  The gradient against measured decline, on the comparable 22 to 30 month
       window rather than the delayed four-wave design. Asks whether the count
       survives adjustment for health change that was already measurable at the
       landmark interview. Reported whatever the answer is.

Job 3  The reference group for the single-domain analysis, three ways: the four
       state transition counts, an explicit four-state model with never-had as
       the reference, and a within-domain restriction comparing stopped against
       continued among respondents who had the activity to begin with.

All respondent-level frames stay in memory on Quartz. Only counts, estimates
and standard errors are written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from probe_behavior_outcome_feasibility import outcome_values
from run_behavioral_withdrawal_frailty_extension import _analysis_set, _extra_frame, _run_model

from behavioral_withdrawal_frailty_core import (
    add_direction_categories,
    build_context_frame,
    build_frailty_frame,
    competing_outcome,
    load_extension_data,
    load_extension_specs,
    load_fi_long,
    load_lookup,
    load_source_components,
    multidomain_frame,
    sha256,
)
from cohort_core import (
    file_sha,
    write_frame,
    write_json,
)

# Death is the competing event, so a death-or-death composite is degenerate: the
# outcome column becomes a column of ones that still clears the support gate and
# produces a number with no meaning. Assert rather than trust the operator.
COMPOSITE_OUTCOMES = ["incident_any_adl", "incident_any_iadl", "multimorbidity_progression"]
CONTEXT_OUTCOMES = ["mortality", "incident_any_adl", "incident_any_iadl", "multimorbidity_progression"]
REFERENCE_OUTCOMES = ["mortality", "incident_any_adl", "incident_any_iadl"]

GRADED = ["loss_1", "loss_2plus"]
DOMAINS = ["alcohol", "activity", "work"]

CONTEXT_FIELDS = [
    "self_rated_health_worsening", "transition_hospitalization", "cesd_worsening",
    "bmi_or_weight_loss", "incident_non_target_disease", "fi_change",
]
# Multimorbidity progression is an increase in the common condition count, and
# incident_non_target_disease is an incident common condition. Adjusting one for
# the other adjusts an outcome for itself.
CONTEXT_EXCLUSIONS = {"multimorbidity_progression": {"incident_non_target_disease"}}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe-config", required=True, type=Path)
    parser.add_argument("--multidomain-config", required=True, type=Path)
    parser.add_argument("--extension-config", required=True, type=Path)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def _deaths(episodes: pd.DataFrame, universe: dict[str, Any], cohort: str) -> pd.Series:
    return episodes["next_status"].isin(universe["cohorts"][cohort]["death_codes"])


# The frailty extension's design carries each behaviour's status at the landmark
# interview, and the published primary models do not. That single difference
# changes the question: with those columns in, the count is compared between
# respondents who arrived at the same current status by different routes, which
# is the "does the change add anything to the state" question; with them out, it
# is the unconditional gradient the paper reports. Fit both and label them,
# because reading one as the other is how a 3.4 becomes a 1.8.
STATE_COLUMNS = ["alcohol_t1", "activity_t1", "work_t1"]
STATE_ARMS = [("unconditional", True), ("current_state_held", False)]


def _strip_state(X: pd.DataFrame, drop: bool) -> pd.DataFrame:
    return X.drop(columns=STATE_COLUMNS, errors="ignore") if drop else X


# --------------------------------------------------------------- Job 1
def composite_models(
    episodes: pd.DataFrame,
    behavior: pd.DataFrame,
    universe: dict[str, Any],
    extension: dict[str, Any],
    cohort: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    extra = _extra_frame(behavior, None)
    death = _deaths(episodes, universe, cohort)
    for outcome_id in COMPOSITE_OUTCOMES:
        assert outcome_id != "mortality", "the composite is degenerate when death is the outcome"
        primary, coding = outcome_values(
            episodes, outcome_id, universe["outcomes"][outcome_id], universe["outcomes"], cohort
        )
        composite, state = competing_outcome(episodes, universe, cohort, outcome_id)
        assert not composite.eq(1).all(), f"{outcome_id}: composite is all events"
        for terms, exposure_id in [(["any_withdrawal"], "any"), (GRADED, "graded")]:
            for model_id, outcome in [("survivors", primary), ("death_or_outcome", composite)]:
                data, y, X = _analysis_set(
                    episodes, behavior, universe, extension, cohort,
                    outcome_id, outcome, extra, terms,
                )
                # How much of this risk set is only here because someone died,
                # and how many events are deaths rather than measured outcomes.
                in_set = death.reindex(data.index).fillna(False)
                entered = in_set & primary.reindex(data.index).isna()
                for arm, drop in STATE_ARMS:
                    base = {
                        "cohort": cohort, "outcome_id": outcome_id,
                        "analysis_family": "competing_composite",
                        "exposure_model": exposure_id, "model_id": model_id,
                        "state_adjustment": arm,
                        "outcome_coding_status": coding,
                        "deaths_in_risk_set": int(in_set.sum()),
                        "entered_through_death": int(entered.sum()) if model_id == "death_or_outcome" else 0,
                    }
                    result, _, _ = _run_model(
                        base, data, y, _strip_state(X, drop), terms, terms, extension)
                    rows.extend(result)
    return rows


# --------------------------------------------------------------- Job 2
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
        # The frailty index exists on a subset of waves, so a joint model that
        # requires it can lose four fifths of the risk set and every event with
        # it. Report the joint model both ways and let the reader see which.
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


# --------------------------------------------------------------- Job 3
def transition_counts(
    episodes: pd.DataFrame,
    behavior: pd.DataFrame,
    universe: dict[str, Any],
    cohort: str,
) -> list[dict[str, Any]]:
    """What the reference group is actually made of, per domain and per state."""
    rows = []
    comparable = episodes["comparable_window"].fillna(False)
    eligible = behavior["core_valid"] & behavior["baseline_engagement_count"].ge(1) & comparable
    death = _deaths(episodes, universe, cohort)
    for domain in DOMAINS:
        transition = behavior[f"{domain}_transition"].where(eligible)
        for state in ["0_to_0", "0_to_1", "1_to_0", "1_to_1"]:
            member = transition.eq(state)
            rows.append({
                "cohort": cohort, "domain": domain, "transition_state": state,
                "n": int(member.sum()),
                "people": int(episodes.loc[member, "person_id"].nunique()),
                "deaths": int((member & death).sum()),
            })
    return rows


def reference_group_models(
    episodes: pd.DataFrame,
    behavior: pd.DataFrame,
    universe: dict[str, Any],
    extension: dict[str, Any],
    cohort: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    extra_base = _extra_frame(behavior, None)
    # The four transition states for each domain, as indicators. Never-had
    # (0 to 0) is the reference and is therefore not a term.
    states = pd.DataFrame(index=behavior.index)
    for domain in DOMAINS:
        transition = behavior[f"{domain}_transition"]
        for state, short in [("0_to_1", "started"), ("1_to_0", "stopped"), ("1_to_1", "continued")]:
            states[f"{domain}_{short}"] = transition.eq(state).astype(float).where(transition.notna())
    extra = pd.concat([extra_base, states], axis=1)

    for outcome_id in REFERENCE_OUTCOMES:
        outcome, coding = outcome_values(
            episodes, outcome_id, universe["outcomes"][outcome_id], universe["outcomes"], cohort
        )
        for domain in DOMAINS:
            # (a) four-state model. The three state indicators determine this
            # domain's status at the landmark, so its own t1 term would be
            # collinear and is dropped; the other two domains keep theirs.
            terms = [f"{domain}_{short}" for short in ["started", "stopped", "continued"]]
            data, y, X = _analysis_set(
                episodes, behavior, universe, extension, cohort,
                outcome_id, outcome, extra, terms,
            )
            X_four = X.drop(columns=[f"{domain}_t1"], errors="ignore")
            result, _, _ = _run_model(
                {"cohort": cohort, "outcome_id": outcome_id,
                 "analysis_family": "reference_group", "domain": domain,
                 "model_id": "four_state_never_had_reference",
                 "outcome_coding_status": coding},
                data, y, X_four, terms, terms, extension,
            )
            rows.extend(result)

            # (b) restricted to respondents who had the activity at the first
            # interview, so the comparison is stopped against continued and
            # nothing else.
            had_it = behavior[f"{domain}_baseline"].eq(1)
            keep = data.index[had_it.reindex(data.index).fillna(False)]
            term = f"{domain}_loss"
            data_r, y_r = data.loc[keep], y.loc[keep]
            X_r = X.loc[keep].drop(columns=[f"{domain}_t1"], errors="ignore")
            result, _, _ = _run_model(
                {"cohort": cohort, "outcome_id": outcome_id,
                 "analysis_family": "reference_group", "domain": domain,
                 "model_id": "restricted_stopped_vs_continued",
                 "outcome_coding_status": coding},
                data_r, y_r, X_r, [term], [term], extension,
            )
            rows.extend(result)
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

    context = None
    source_audit: dict[str, Any] = {}
    fi_audit: dict[str, Any] = {}
    if cohort in extension["frailty_cohorts"]:
        source, source_audit = load_source_components(root, extension, cohort)
        fi_long, fi_audit = load_fi_long(extension, cohort)
        frailty = build_frailty_frame(episodes, formal, source, fi_long, universe, extension, cohort)
        context = build_context_frame(episodes, frailty, universe, cohort)

    outputs: dict[str, Path] = {}

    outputs["composite"] = args.output_dir / f"{cohort}-composite-models.csv"
    write_frame(outputs["composite"], composite_models(
        episodes, behavior, universe, extension, cohort))

    outputs["transitions"] = args.output_dir / f"{cohort}-transition-counts.csv"
    write_frame(outputs["transitions"], transition_counts(
        episodes, behavior, universe, cohort))

    outputs["reference"] = args.output_dir / f"{cohort}-reference-group-models.csv"
    write_frame(outputs["reference"], reference_group_models(
        episodes, behavior, universe, extension, cohort))

    if context is not None:
        outputs["context"] = args.output_dir / f"{cohort}-context-graded-models.csv"
        write_frame(outputs["context"], context_gradient_models(
            episodes, behavior, context, universe, extension, cohort))

    manifest = {
        "analysis_id": "behavioral_withdrawal_competing_context_v0.3",
        "mode": "models",
        "cohort": cohort,
        "code_commit": args.code_commit,
        "parent_analysis_id": extension["analysis_id"],
        "design_sha256": sha256(args.design),
        "extension_config_sha256": sha256(args.extension_config),
        "universe_config_sha256": sha256(args.universe_config),
        "multidomain_config_sha256": sha256(args.multidomain_config),
        "lookup_sha256": sha256(lookup_path),
        "input_audit": input_audit,
        "source_component_audit": source_audit,
        "fi_audit": fi_audit,
        "outputs": {key: file_sha(path) for key, path in outputs.items()},
        "support_gates_unchanged": True,
        "negative_results_retained": True,
        "aggregate_only": True,
        "respondent_rows_exported": 0,
    }
    write_json(args.output_dir / f"{cohort}-model-manifest.json", manifest)
    print(json.dumps({"status": "PASS", "cohort": cohort, "outputs": len(outputs)}))


if __name__ == "__main__":
    main()
