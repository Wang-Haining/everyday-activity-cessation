#!/usr/bin/env python3
"""Test equality of the component cessation coefficients."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from behavioral_contraction_biomarker_core import file_sha, write_frame, write_json
from behavioral_withdrawal_frailty_core import (
    add_direction_categories,
    load_extension_data,
    load_extension_specs,
    load_lookup,
    multidomain_frame,
    sha256,
)
from probe_behavior_outcome_feasibility import outcome_values
from run_behavioral_withdrawal_frailty_extension import (
    _analysis_set, _extra_frame, _run_model,
)
from run_behavioral_withdrawal_competing_context import _strip_state

COMPONENTS = ["alcohol_loss", "activity_loss", "work_loss"]
CONTRASTS = [("alcohol_loss", "activity_loss"), ("alcohol_loss", "work_loss"),
             ("activity_loss", "work_loss")]
OUTCOMES = ["mortality", "incident_any_adl", "incident_any_iadl"]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--universe-config", required=True, type=Path)
    parser.add_argument("--multidomain-config", required=True, type=Path)
    parser.add_argument("--extension-config", required=True, type=Path)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def component_contrasts(
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
            outcome_id, outcome, extra, COMPONENTS,
        )
        X = _strip_state(X, True)
        base = {"cohort": cohort, "outcome_id": outcome_id,
                "analysis_family": "component_equality",
                "outcome_coding_status": coding}
        result, fit, design = _run_model(
            base, data, y, X, COMPONENTS, COMPONENTS, extension
        )
        if fit is None or design is None:
            rows.append({**base, "model_status": result[0].get("model_status", "UNKNOWN")})
            continue
        params = fit.params
        cov = pd.DataFrame(np.asarray(fit.cov_params()),
                           index=params.index, columns=params.index)
        for a, b in CONTRASTS:
            if a not in params.index or b not in params.index:
                continue
            diff = float(params[a] - params[b])
            var = float(cov.loc[a, a] + cov.loc[b, b] - 2 * cov.loc[a, b])
            se = float(np.sqrt(max(var, 0.0)))
            rows.append({
                **base, "model_status": "PASS", "contrast": f"{a}-{b}",
                "n": int(len(data)), "events": int(y.sum()),
                "log_difference": diff, "standard_error": se,
                "ratio_of_risk_ratios": float(np.exp(diff)),
                "ci_low": float(np.exp(diff - 1.96 * se)),
                "ci_high": float(np.exp(diff + 1.96 * se)),
                f"log_rr_{a}": float(params[a]), f"log_rr_{b}": float(params[b]),
            })
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
    out = args.output_dir / f"{cohort}-component-equality.csv"
    rows = component_contrasts(episodes, behavior, universe, extension, cohort)
    write_frame(out, rows)
    write_json(args.output_dir / f"{cohort}-component-equality-manifest.json", {
        "analysis_id": "behavioral_withdrawal_component_equality_v0.1",
        "cohort": cohort,
        "contrasts": [f"{a}-{b}" for a, b in CONTRASTS],
        "universe_config_sha256": sha256(args.universe_config),
        "multidomain_config_sha256": sha256(args.multidomain_config),
        "lookup_sha256": sha256(lookup_path),
        "input_audit": input_audit,
        "aggregate_only": True,
        "respondent_rows_exported": 0,
        "outputs": {"component_equality": file_sha(out)},
    })
    print(json.dumps({"status": "PASS", "cohort": cohort, "rows": len(rows)}))


if __name__ == "__main__":
    main()
