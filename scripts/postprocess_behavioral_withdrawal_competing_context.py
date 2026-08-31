#!/usr/bin/env python3
"""Pool the v0.3 cohort results with the project's estimator and write the tables.

Same REML with unmodified Hartung-Knapp intervals as every other synthesis here,
prediction interval and I-squared reported whenever at least three cohorts
contribute, and cohort-specific estimates carried alongside so a reader can see
what the pooled number is made of.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from postprocess_behavior_outcome_landscape import reml_hk

NAME = {"charls": "CHARLS", "elsa": "ELSA", "hrs": "HRS", "klosa": "KLoSA",
        "mhas": "MHAS", "share": "SHARE"}
COHORTS = list(NAME)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read(model_dir: Path, suffix: str) -> pd.DataFrame:
    frames = []
    for cohort in COHORTS:
        path = model_dir / f"{cohort}-{suffix}.csv"
        if path.exists():
            frame = pd.read_csv(path)
            if len(frame):
                frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def pool(group: pd.DataFrame) -> dict:
    passing = group[group.model_status.eq("PASS") & group.estimate.notna()
                    & group.standard_error.notna() & group.standard_error.gt(0)]
    passing = passing.drop_duplicates("cohort")
    out = {
        "k_cohorts": int(len(passing)),
        "cohorts": ";".join(sorted(passing.cohort)),
        "cohort_estimates": ";".join(
            f"{c}:{e:.4f}" for c, e in zip(passing.cohort, passing.estimate)
        ),
        "n_total": int(group[group.model_status.eq("PASS")].drop_duplicates("cohort").n.sum())
        if "n" in group else np.nan,
        "not_evaluable": ";".join(sorted(
            f"{r.cohort}:{r.model_status}"
            for _, r in group[group.model_status.ne("PASS")].drop_duplicates("cohort").iterrows()
        )),
    }
    if len(passing) < 3:
        out.update({"pooled_estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                    "prediction_low": np.nan, "prediction_high": np.nan,
                    "i2": np.nan, "tau2": np.nan})
        return out
    y = np.log(passing.estimate.to_numpy(float))
    v = passing.standard_error.to_numpy(float) ** 2
    fit = reml_hk(y, v)
    out.update({
        "pooled_estimate": float(np.exp(fit["pooled"])),
        "ci_low": float(np.exp(fit["ci_low"])),
        "ci_high": float(np.exp(fit["ci_high"])),
        "prediction_low": float(np.exp(fit["prediction_low"])),
        "prediction_high": float(np.exp(fit["prediction_high"])),
        "i2": float(fit["i2"] * 100),
        "tau2": float(fit["tau2"]),
    })
    return out


def synthesise(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    frame = frame[frame.term.notna()] if "term" in frame else frame
    rows = []
    for values, group in frame.groupby(keys, dropna=False):
        row = dict(zip(keys, values if isinstance(values, tuple) else (values,)))
        rows.append({**row, **pool(group)})
    return pd.DataFrame(rows)


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = {}

    composite = read(args.model_dir, "composite-models")
    pooled_composite = synthesise(
        composite, ["outcome_id", "exposure_model", "model_id", "state_adjustment", "term"])
    if not pooled_composite.empty:
        deaths = composite[composite.model_id.eq("death_or_outcome")].drop_duplicates(
            ["cohort", "outcome_id"])
        entered = deaths.groupby("outcome_id").entered_through_death.sum()
        pooled_composite["entered_through_death_total"] = pooled_composite.outcome_id.map(entered)
    written["composite"] = args.output_dir / "composite-synthesis.csv"
    pooled_composite.to_csv(written["composite"], index=False)

    context = read(args.model_dir, "context-graded-models")
    written["context"] = args.output_dir / "context-synthesis.csv"
    synthesise(context, ["outcome_id", "context_id", "model_id", "state_adjustment", "term"]).to_csv(
        written["context"], index=False)

    reference = read(args.model_dir, "reference-group-models")
    written["reference"] = args.output_dir / "reference-group-synthesis.csv"
    synthesise(reference, ["outcome_id", "domain", "model_id", "term"]).to_csv(
        written["reference"], index=False)

    transitions = read(args.model_dir, "transition-counts")
    written["transitions"] = args.output_dir / "transition-counts.csv"
    transitions.to_csv(written["transitions"], index=False)

    # The attenuation each cohort reported, kept per cohort because averaging a
    # ratio of coefficients across cohorts is not a quantity anyone can read.
    if not context.empty and "log_rr_attenuation_percent" in context:
        att = context[context.log_rr_attenuation_percent.notna()][
            ["cohort", "outcome_id", "context_id", "model_id", "state_adjustment", "term",
             "log_rr_attenuation_percent", "n", "events", "estimate", "ci_low", "ci_high"]]
        written["attenuation"] = args.output_dir / "context-attenuation-by-cohort.csv"
        att.to_csv(written["attenuation"], index=False)

    manifest = {
        "analysis_id": "behavioral_withdrawal_competing_context_v0.3",
        "mode": "postprocess",
        "synthesis": "cohort_specific_REML_Hartung_Knapp_prediction_interval",
        "minimum_cohorts_for_pooling": 3,
        "negative_results_retained": True,
        "aggregate_only": True,
        "respondent_rows_exported": 0,
        "outputs": {k: str(v.name) for k, v in written.items()},
    }
    (args.output_dir / "postprocess-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "outputs": len(written)}))


if __name__ == "__main__":
    main()
