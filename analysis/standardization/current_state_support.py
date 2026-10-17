#!/usr/bin/env python3
"""Aggregate support in the incident-context mortality risk set."""
from itertools import product
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pandas as pd
import run_behavioral_withdrawal_competing_context as r

args = r.arguments()
universe, multidomain, extension = r.load_extension_specs(
    args.universe_config, args.multidomain_config, args.extension_config)
cohort = args.cohort.lower()
assert cohort in extension["frailty_cohorts"], f"no current-state context fit for {cohort}"
lookup, _ = r.load_lookup(args.data_root, universe)
episodes, formal, _, _, _ = r.load_extension_data(args.data_root, universe, cohort, lookup)
behavior = r.add_direction_categories(r.multidomain_frame(episodes, universe, multidomain, cohort))
source, _ = r.load_source_components(args.data_root, extension, cohort)
frailty = r.build_frailty_frame(episodes, formal, source, universe, extension, cohort)
context = r.build_context_frame(episodes, frailty, universe, cohort)
outcome, _ = r.outcome_values(episodes, "mortality", universe["outcomes"]["mortality"], universe["outcomes"], cohort)
extra = pd.concat([r._extra_frame(behavior, None), context], axis=1)
data, y, X = r._analysis_set(episodes, behavior, universe, extension, cohort,
    "mortality", outcome, extra, [*r.GRADED, "incident_non_target_disease"])
assert len(data) > 0, f"empty incident-context mortality risk set for {cohort}"
fitted = pd.read_csv(ROOT / "results/context/models" / f"{cohort}-context-graded-models.csv")
match = fitted[(fitted.outcome_id == "mortality") & (fitted.model_id == "base_same_set_incident_non_target_disease") & (fitted.state_adjustment == "current_state_held")]
assert len(match) > 0, f"missing state fit for {cohort}"
assert set(match.n) == {len(data)}, (cohort, "n", set(match.n), len(data))
assert set(match.events) == {int(y.sum())}, (cohort, "events", set(match.events), int(y.sum()))
cells = pd.DataFrame({"cessation_count": behavior.loc[data.index, "loss_count_core"].astype(int),
    "initial_count": behavior.loc[data.index, "baseline_engagement_count"].astype(int),
    "landmark_pattern": behavior.loc[data.index, r.STATE_COLUMNS].astype(int).astype(str).agg("".join, axis=1),
    "events": y, "person_id": data.person_id})
grouped = cells.groupby(["cessation_count", "initial_count", "landmark_pattern"]).agg(
    n=("events", "size"), events=("events", "sum"), people=("person_id", "nunique"))
index = pd.MultiIndex.from_product([range(4), range(1, 4), ["".join(x) for x in product("01", repeat=3)]],
    names=["cessation_count", "initial_count", "landmark_pattern"])
table = grouped.reindex(index, fill_value=0).reset_index()
table.insert(0, "cohort", cohort)
table["events"] = table.events.astype(int)
table["structurally_attainable"] = [max(0, j - p.count("1")) <= l <= min(j, 3 - p.count("1"))
    for l, j, p in zip(table.cessation_count, table.initial_count, table.landmark_pattern)]
table["empty_cell"] = table.n.eq(0)
assert int(table.n.sum()) == len(data) and int(table.events.sum()) == int(y.sum())
assert table.loc[~table.structurally_attainable, "n"].eq(0).all()
overlap = cells.groupby(["initial_count", "landmark_pattern"]).cessation_count.apply(
    lambda x: bool(x.eq(0).any() and x.ge(2).any())).sum()
assert overlap == 0, f"unexpected zero-versus-two-plus overlap: {overlap}"
X = X.copy()
for term in r.GRADED:
    X[term] = data[term].astype(float)
summary = {"cohort": cohort, "n": len(data), "events": int(y.sum()),
    "matrix_columns": X.shape[1], "matrix_rank": int(np.linalg.matrix_rank(X.to_numpy(float))),
    "cells": len(table), "empty_cells": int(table.empty_cell.sum()),
    "structural_zero_cells": int((~table.structurally_attainable).sum()),
    "zero_vs_two_plus_overlap_strata": int(overlap), "model_status": ";".join(sorted(set(match.model_status)))}
args.output_dir.mkdir(parents=True, exist_ok=True)
table.to_csv(args.output_dir / f"{cohort}-support.csv", index=False)
pd.DataFrame([summary]).to_csv(args.output_dir / f"{cohort}-support-summary.csv", index=False)
print(json.dumps(summary))
