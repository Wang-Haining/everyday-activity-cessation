#!/usr/bin/env python3
"""Fit SHARE country-structure and HRS survey-weight sensitivity models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from behavioral_contraction_biomarker_core import fit_clustered, load_episodes
from multidomain_behavioral_withdrawal_core import (
    load_specs,
    multidomain_frame,
    prepare_model,
)
from probe_behavior_outcome_feasibility import load_lookup

TERMS = ["loss_1", "loss_2plus"]


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--universe-config", required=True, type=Path)
    p.add_argument("--pilot-config", required=True, type=Path)
    p.add_argument("--cohort", required=True)
    p.add_argument("--output-dir", required=True, type=Path)
    return p.parse_args()


def undecorated(frame: pd.DataFrame) -> dict[str, str]:
    """Map the bare variable name to its decorated column."""
    return {str(c).split(" ")[0].lstrip("﻿"): c for c in frame.columns}


def extra_column(root: Path, universe: dict, cohort: str, wanted: str) -> pd.Series | None:
    """Read one column the configured field set does not carry."""
    path = root / universe["formal_subdir"] / universe["cohorts"][cohort]["formal_file"]
    head = pd.read_csv(path, nrows=0)
    names = undecorated(head)
    cc = universe["cohorts"][cohort]
    keys = [names.get(cc["id"]), names.get(cc["wave"])]
    if wanted not in names or not all(keys):
        return None
    frame = pd.read_csv(path, usecols=[*keys, names[wanted]])
    frame.columns = ["person_id", "wave", wanted]
    # The formal CSV may type the identifier differently from the episode frame.
    frame["person_id"] = frame["person_id"].astype(str)
    frame["wave"] = pd.to_numeric(frame["wave"], errors="coerce")
    return frame


def rows(fit, label: str, n: int, events: float, extra: dict | None = None) -> list[dict]:
    out = []
    for term in TERMS:
        if term not in fit.params.index:
            continue
        beta = float(fit.params[term])
        se = float(fit.bse[term])
        row = {
            "model": label,
            "term": term,
            "n": int(n),
            "events": int(events),
            "estimate": float(np.exp(beta)),
            "ci_low": float(np.exp(beta - 1.96 * se)),
            "ci_high": float(np.exp(beta + 1.96 * se)),
            "standard_error": se,
        }
        row.update(extra or {})
        out.append(row)
    return out


def main() -> None:
    args = arguments()
    assert args.data_root.is_dir(), f"missing data root: {args.data_root}"
    universe, pilot = load_specs(args.universe_config, args.pilot_config)
    root = args.data_root
    lookup, _ = load_lookup(root, universe)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    episodes, _, _ = load_episodes(root, universe, args.cohort, lookup)
    behavior = multidomain_frame(episodes, universe, pilot, args.cohort)
    data, y, X, terms, binary, _ = prepare_model(
        episodes, behavior, universe, pilot, args.cohort,
        "mortality", "comparable_22_30_months", "full", "score_categorical",
    )
    if not len(data):
        print(f"{args.cohort}: no rows in the primary mortality risk set")
        return

    result: list[dict] = []
    fit, _ = fit_clustered(y, X, data["person_id"], binary)
    result += rows(fit, "as_published_respondent_clustered", len(data), y.sum())

    if args.cohort == "share":
        country = extra_column(root, universe, args.cohort, "country")
        if country is None:
            print("share: no country column; nothing to add")
        else:
            keys = data[["person_id", "t1_wave"]].reset_index()
            keys["person_id"] = keys["person_id"].astype(str)
            keys["t1_wave"] = pd.to_numeric(keys["t1_wave"], errors="coerce")
            merged = keys.merge(
                country.rename(columns={"wave": "t1_wave"}),
                on=["person_id", "t1_wave"], how="left").set_index("index")
            code = merged["country"].reindex(data.index)
            have = code.notna()
            print(f"share: country recorded for {int(have.sum()):,} of {len(data):,} intervals, "
                  f"{int(code.nunique())} countries")

            # Variance clustered by country rather than by respondent.
            sub = have
            fit2, _ = fit_clustered(y[sub], X[sub], code[sub].astype(int), binary)
            result += rows(fit2, "country_clustered_variance", int(sub.sum()), y[sub].sum())

            # Country fixed effects.
            dummies = pd.get_dummies(code[sub].astype(int).astype(str),
                                     prefix="country", drop_first=True, dtype=float)
            Xc = pd.concat([X[sub], dummies], axis=1)
            fit3, _ = fit_clustered(y[sub], Xc, data.loc[sub, "person_id"], binary)
            result += rows(fit3, "country_fixed_effects", int(sub.sum()), y[sub].sum())

            # Leave one country out.
            for value in sorted(code[sub].dropna().unique()):
                keep = sub & code.ne(value)
                if y[keep].sum() < 40:
                    continue
                f, _ = fit_clustered(y[keep], X[keep], data.loc[keep, "person_id"], binary)
                result += rows(f, "leave_one_country_out", int(keep.sum()), y[keep].sum(),
                               {"omitted_country_code": int(value)})

    if args.cohort == "hrs":
        weight = extra_column(root, universe, args.cohort, "wtresp")
        if weight is None:
            print("hrs: no respondent weight column")
        else:
            keys = data[["person_id", "t1_wave"]].reset_index()
            keys["person_id"] = keys["person_id"].astype(str)
            keys["t1_wave"] = pd.to_numeric(keys["t1_wave"], errors="coerce")
            merged = keys.merge(
                weight.rename(columns={"wave": "t1_wave"}),
                on=["person_id", "t1_wave"], how="left").set_index("index")
            w = pd.to_numeric(merged["wtresp"], errors="coerce").reindex(data.index)
            keep = w.notna() & w.gt(0)
            print(f"hrs: positive respondent weight for {int(keep.sum()):,} of {len(data):,}")
            model = sm.GLM(y[keep], X[keep], family=sm.families.Poisson(),
                           freq_weights=w[keep].to_numpy(float))
            fitw = model.fit(cov_type="cluster",
                             cov_kwds={"groups": data.loc[keep, "person_id"]})
            result += rows(fitw, "respondent_weighted", int(keep.sum()), y[keep].sum())
            funw, _ = fit_clustered(y[keep], X[keep], data.loc[keep, "person_id"], binary)
            result += rows(funw, "unweighted_same_rows", int(keep.sum()), y[keep].sum())

    frame = pd.DataFrame(result)
    frame.insert(0, "cohort", args.cohort)
    out = args.output_dir / f"{args.cohort}-design-sensitivity.csv"
    frame.to_csv(out, index=False)
    out.chmod(0o600)
    manifest = {
        "analysis_id": "behavioural_withdrawal_design_sensitivity_v0.1",
        "cohort": args.cohort,
        "models": sorted(frame.model.unique().tolist()) if len(frame) else [],
        "aggregate_only": True,
    }
    mpath = args.output_dir / f"{args.cohort}-design-sensitivity-manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    mpath.chmod(0o600)
    print(f"{args.cohort}: {len(frame)} estimate rows")


if __name__ == "__main__":
    main()
