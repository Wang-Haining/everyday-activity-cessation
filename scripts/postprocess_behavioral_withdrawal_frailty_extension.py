#!/usr/bin/env python3
"""Synthesize aggregate frailty-extension results and clinical story gates."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from postprocess_behavior_outcome_landscape import reml_hk

from cohort_core import file_sha, write_json

OUTCOME_LABELS = {
    "incident_diabetes": "Diabetes",
    "incident_stroke": "Stroke",
    "incident_heart_disease": "Heart disease",
    "incident_hypertension": "Hypertension",
    "incident_cancer": "Cancer",
    "incident_arthritis": "Arthritis",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension-config", required=True, type=Path)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--final-dir", required=True, type=Path)
    return parser.parse_args()


def _read_family(model_dir: Path, suffix: str) -> pd.DataFrame:
    paths = sorted(model_dir.glob(f"*-{suffix}.csv"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(path, low_memory=False) for path in paths], ignore_index=True)


def _meta(group: pd.DataFrame, keys: dict[str, Any]) -> dict[str, Any]:
    yi = np.log(group["estimate"].astype(float).to_numpy())
    sei = group["standard_error"].astype(float).to_numpy()
    fit = reml_hk(yi, sei**2)
    return {
        **keys,
        "k_cohorts": int(group["cohort"].nunique()),
        "cohorts": ";".join(sorted(group["cohort"].unique())),
        "pooled_rr": math.exp(fit["pooled"]),
        "ci_low": math.exp(fit["ci_low"]),
        "ci_high": math.exp(fit["ci_high"]),
        "prediction_low": math.exp(fit["prediction_low"]),
        "prediction_high": math.exp(fit["prediction_high"]),
        "i2": fit["i2"],
        "tau2_log_scale": fit["tau2"],
        "direction_above_one": float((group["estimate"].astype(float) > 1).mean()),
        "cohort_estimates": ";".join(
            f"{row.cohort}:{float(row.estimate):.4f}" for row in group.itertuples()
        ),
    }


def meta_results(specificity: pd.DataFrame, frailty: pd.DataFrame) -> pd.DataFrame:
    candidates = pd.concat([
        specificity.loc[
            specificity["model_status"].eq("PASS")
            & specificity["term"].ne("__model__")
        ],
        frailty.loc[
            frailty["model_status"].eq("PASS")
            & frailty["term"].isin([
                "any_withdrawal", "loss_1", "loss_2plus", "prefrail", "frail", "fried4", "fi_per_0_1",
            ])
        ],
    ], ignore_index=True, sort=False)
    rows = []
    keys = ["analysis_family", "model_id", "outcome_id", "term"]
    for values, group in candidates.groupby(keys, dropna=False, sort=True):
        if group["cohort"].nunique() < 3:
            continue
        rows.append(_meta(group, dict(zip(keys, values))))
    return pd.DataFrame(rows)


def story_gate(meta: pd.DataFrame, frailty: pd.DataFrame, performance: pd.DataFrame) -> dict[str, Any]:
    current = meta.loc[
        meta["analysis_family"].eq("specificity")
        & meta["model_id"].eq("any_withdrawal")
        & meta["term"].eq("any_withdrawal")
    ].set_index("outcome_id")
    diabetes_supported = bool("incident_diabetes" in current.index and current.loc["incident_diabetes", "ci_low"] > 1)
    stroke_supported = bool("incident_stroke" in current.index and current.loc["incident_stroke", "ci_low"] > 1)
    cd_replicated = bool(diabetes_supported or stroke_supported)

    nonfrail = frailty.loc[
        frailty["model_status"].eq("PASS")
        & frailty["model_id"].eq("nonfrail_any_withdrawal")
        & frailty["term"].eq("any_withdrawal")
    ]
    nonfrail_positive = set(nonfrail.loc[nonfrail["ci_low"].gt(1), "cohort"])
    delta = performance.loc[
        performance["model_status"].eq("PASS")
        & performance["analysis_family"].eq("head_to_head_cv_binary_flag")
        & performance["model_id"].eq("M3b_fried_any_withdrawal_minus_M2_fried")
    ]
    improved_cohorts = set()
    for cohort, group in delta.groupby("cohort"):
        c_improved = bool((group.loc[group["metric"].eq("delta_c_statistic"), "ci_low"] > 0).any())
        b_improved = bool((group.loc[group["metric"].eq("delta_brier_score"), "ci_high"] < 0).any())
        if c_improved or b_improved:
            improved_cohorts.add(cohort)
    adds_beyond_frailty = len(nonfrail_positive.intersection(improved_cohorts)) >= 2

    classification = (
        "RETAIN_CD" if cd_replicated and adds_beyond_frailty
        else "BROADER_CLINICAL_SIGNAL_NO_INCREMENTAL_VALUE" if not adds_beyond_frailty
        else "BROADER_CLINICAL_SIGNAL"
    )
    return {
        "classification": classification,
        "cardiometabolic_gradient_replicated_after_current_state_adjustment": cd_replicated,
        "diabetes_supported": diabetes_supported,
        "stroke_supported": stroke_supported,
        "nonfrail_positive_cohorts": sorted(nonfrail_positive),
        "cv_improved_cohorts": sorted(improved_cohorts),
        "adds_information_beyond_frailty": adds_beyond_frailty,
        "permitted_language": (
            "adds information beyond frailty" if adds_beyond_frailty
            else "does not add reproducible information beyond current behavior and frailty"
        ),
    }


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    os.chmod(path, 0o600)


def figure_decomposition(meta: pd.DataFrame, path: Path) -> None:
    models = [
        "two_domain_any_without_current_states",
        "three_domain_any_without_current_states",
        "any_withdrawal",
    ]
    labels = {
        models[0]: "Two-domain transition",
        models[1]: "Three-domain transition",
        models[2]: "After current behavior",
    }
    frame = meta.loc[meta["model_id"].isin(models) & meta["term"].isin(["two_domain_any", "any_withdrawal"])].copy()
    fig, ax = plt.subplots(figsize=(9.0, 5.3))
    outcomes = list(OUTCOME_LABELS)
    offsets = np.linspace(-0.22, 0.22, len(models))
    for offset, model in zip(offsets, models):
        sub = frame.loc[frame["model_id"].eq(model)].set_index("outcome_id")
        y = np.arange(len(outcomes)) + offset
        estimate = np.array([sub.loc[o, "pooled_rr"] if o in sub.index else np.nan for o in outcomes])
        low = np.array([sub.loc[o, "ci_low"] if o in sub.index else np.nan for o in outcomes])
        high = np.array([sub.loc[o, "ci_high"] if o in sub.index else np.nan for o in outcomes])
        ax.errorbar(estimate, y, xerr=[estimate - low, high - estimate], fmt="o", capsize=2, label=labels[model])
    ax.axvline(1, color="#6b7280", lw=1)
    ax.set_yticks(np.arange(len(outcomes)), [OUTCOME_LABELS[o] for o in outcomes])
    ax.set_xlabel("Adjusted risk ratio")
    ax.set_title("What remains after the patient's current behavior is known?")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    ax.grid(axis="x", color="#e5e7eb", lw=0.7)
    _save_figure(fig, path)


def figure_heatmap(specificity: pd.DataFrame, path: Path) -> None:
    frame = specificity.loc[
        specificity["model_status"].eq("PASS")
        & specificity["model_id"].eq("any_withdrawal")
        & specificity["term"].eq("any_withdrawal")
    ].copy()
    table = frame.pivot(index="cohort", columns="outcome_id", values="estimate").reindex(
        columns=list(OUTCOME_LABELS)
    )
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    sns.heatmap(
        np.log(table), center=0, cmap="vlag", annot=table, fmt=".2f", linewidths=0.5,
        xticklabels=[OUTCOME_LABELS[o] for o in table.columns], yticklabels=[x.upper() for x in table.index], ax=ax,
        cbar_kws={"label": "log risk ratio"},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Current-state-adjusted withdrawal signals across diseases")
    _save_figure(fig, path)


def figure_performance(performance: pd.DataFrame, path: Path) -> None:
    frame = performance.loc[
        performance["model_status"].eq("PASS")
        & performance["analysis_family"].eq("head_to_head_cv_binary_flag")
        & performance["model_id"].isin([
            "M1b_any_withdrawal_minus_M0_routine",
            "M2_fried_minus_M0_routine",
            "M3b_fried_any_withdrawal_minus_M2_fried",
        ])
        & performance["metric"].eq("delta_c_statistic")
    ].copy()
    labels = {
        "M1b_any_withdrawal_minus_M0_routine": "Withdrawal added to routine data",
        "M2_fried_minus_M0_routine": "Fried added to routine data",
        "M3b_fried_any_withdrawal_minus_M2_fried": "Withdrawal added after Fried",
    }
    frame["label"] = frame["model_id"].map(labels)
    frame["row"] = frame["cohort"].str.upper() + "  " + frame["outcome_id"].map(OUTCOME_LABELS)
    frame = frame.sort_values(["label", "row"])
    fig, axes = plt.subplots(1, 3, figsize=(12.5, max(4.0, 0.30 * len(frame) / 3)), sharey=False)
    for ax, (label, sub) in zip(axes, frame.groupby("label", sort=False)):
        y = np.arange(len(sub))
        ax.errorbar(sub["estimate"], y, xerr=[sub["estimate"] - sub["ci_low"], sub["ci_high"] - sub["estimate"]], fmt="o", capsize=2)
        ax.axvline(0, color="#6b7280", lw=1)
        ax.set_yticks(y, sub["row"])
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Change in C statistic")
        ax.grid(axis="x", color="#e5e7eb", lw=0.7)
    fig.suptitle("Grouped cross-validated information gain", y=1.02)
    _save_figure(fig, path)


def main() -> None:
    args = arguments()
    extension = json.loads(args.extension_config.read_text())
    args.final_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(args.final_dir, 0o700)
    families = {
        "specificity": _read_family(args.model_dir, "specificity-models"),
        "frailty": _read_family(args.model_dir, "frailty-models"),
        "context": _read_family(args.model_dir, "context-models"),
        "delayed": _read_family(args.model_dir, "delayed-models"),
        "sensitivity": _read_family(args.model_dir, "sensitivity-models"),
        "performance": _read_family(args.model_dir, "cv-performance"),
        "risks": _read_family(args.model_dir, "standardized-risks"),
        "coverage": _read_family(args.model_dir, "event-coverage"),
    }
    for name, frame in families.items():
        if frame.empty and name not in {"risks"}:
            raise RuntimeError(f"empty model family: {name}")
        path = args.final_dir / f"systematic-{name}-results.csv"
        frame.to_csv(path, index=False)
        os.chmod(path, 0o600)
    meta = meta_results(families["specificity"], families["frailty"])
    meta_path = args.final_dir / "cross-cohort-meta-analysis.csv"
    meta.to_csv(meta_path, index=False)
    os.chmod(meta_path, 0o600)
    gate = story_gate(meta, families["frailty"], families["performance"])
    write_json(args.final_dir / "story-classification.json", gate)
    sns.set_theme(style="whitegrid", context="talk")
    figure_decomposition(meta, args.final_dir / "figure-current-state-decomposition.png")
    figure_heatmap(families["specificity"], args.final_dir / "figure-six-disease-heatmap.png")
    figure_performance(families["performance"], args.final_dir / "figure-frailty-performance.png")
    manifest = {
        "analysis_id": extension["analysis_id"], "status": "PASS",
        "story_classification": gate["classification"],
        "aggregate_only": True, "respondent_rows_exported": 0,
        "outputs": {
            path.name: file_sha(path) for path in args.final_dir.iterdir() if path.is_file()
        },
    }
    write_json(args.final_dir / "postprocess-manifest.json", manifest)
    print({"status": "PASS", "classification": gate["classification"], "meta_rows": len(meta)})


if __name__ == "__main__":
    main()
