#!/usr/bin/env python3
"""Fit disease-specific and sensitivity models without a frailty-index input."""

import argparse
from pathlib import Path

from behavioral_contraction_biomarker_core import write_frame
from behavioral_withdrawal_frailty_core import (
    add_direction_categories, load_extension_data, load_extension_specs,
    load_lookup, multidomain_frame,
)
from run_behavioral_withdrawal_frailty_extension import specificity_models, sensitivity_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--universe-config", required=True, type=Path)
    parser.add_argument("--multidomain-config", required=True, type=Path)
    parser.add_argument("--extension-config", required=True, type=Path)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    assert args.data_root.is_dir(), f"missing data root: {args.data_root}"
    universe, multidomain, extension = load_extension_specs(
        args.universe_config, args.multidomain_config, args.extension_config
    )
    cohort = args.cohort.lower()
    assert cohort in extension["specificity_cohorts"], f"unknown cohort: {cohort}"
    lookup, _ = load_lookup(args.data_root, universe)
    episodes, _, _, _, _ = load_extension_data(args.data_root, universe, cohort, lookup)
    behavior = add_direction_categories(multidomain_frame(episodes, universe, multidomain, cohort))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for label, function in [("specificity", specificity_models), ("sensitivity", sensitivity_models)]:
        rows = function(episodes, behavior, universe, extension, cohort)
        assert rows, f"empty {label} rows for {cohort}"
        write_frame(args.output_dir / f"{cohort}-{label}-models.csv", rows)
        counts[label] = len(rows)
    print({"cohort": cohort, "aggregate_rows": counts})


if __name__ == "__main__":
    main()
