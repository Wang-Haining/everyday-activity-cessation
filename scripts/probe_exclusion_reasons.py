#!/usr/bin/env python3
"""Count exclusions from each cohort analysis population."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from behavioral_contraction_biomarker_core import load_episodes
from multidomain_behavioral_withdrawal_core import load_specs, multidomain_frame
from probe_behavior_outcome_feasibility import load_lookup

DOMAINS = ["alcohol", "activity", "work"]


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--universe-config", required=True, type=Path)
    p.add_argument("--pilot-config", required=True, type=Path)
    p.add_argument("--cohort", required=True)
    p.add_argument("--output-dir", required=True, type=Path)
    return p.parse_args()


def status_split(episodes: pd.DataFrame, cc: dict) -> dict[str, int]:
    """Outcome-interview status, if the episode frame carries one."""
    field = next((c for c in ("next_status", "t2_iwstat", "status_t2")
                  if c in episodes.columns), None)
    if field is None:
        return {"outcome_status_recorded": 0}
    s = pd.to_numeric(episodes[field], errors="coerce")
    death = set(cc.get("death_codes", []))
    alive = set(cc.get("alive_codes", []))
    return {
        "outcome_status_recorded": int(s.notna().sum()),
        "outcome_interview_died": int(s.isin(death).sum()),
        "outcome_interview_alive_response": int(s.isin(alive).sum()),
        "outcome_interview_other": int(s.notna().sum() - s.isin(death | alive).sum()),
    }


def proxy_split(episodes: pd.DataFrame) -> dict[str, int]:
    out: dict[str, int] = {}
    for tag in ("t0", "t1", "t2"):
        field = next((c for c in (f"proxy_{tag}", f"{tag}_proxy") if c in episodes.columns), None)
        if field is None:
            out[f"proxy_{tag}_recorded"] = 0
            continue
        s = pd.to_numeric(episodes[field], errors="coerce")
        out[f"proxy_{tag}_recorded"] = int(s.notna().sum())
        out[f"proxy_{tag}_yes"] = int(s.eq(1).sum())
    return out


def main() -> None:
    args = arguments()
    assert args.data_root.is_dir(), f"missing data root: {args.data_root}"
    universe, pilot = load_specs(args.universe_config, args.pilot_config)
    root = args.data_root
    lookup, _ = load_lookup(root, universe)
    cc = universe["cohorts"][args.cohort]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    episodes, _, _ = load_episodes(root, universe, args.cohort, lookup)
    behavior = multidomain_frame(episodes, universe, pilot, args.cohort)
    comparable = episodes["comparable_window"].fillna(False)

    valid = {d: behavior[f"{d}_transition"].notna() & behavior[f"{d}_baseline"].notna()
             for d in DOMAINS}
    core = behavior["core_valid"].fillna(False)
    opportunity = behavior["baseline_engagement_count"].ge(
        pilot["minimum_baseline_opportunities"]).fillna(False)

    row: dict[str, object] = {
        "cohort": args.cohort,
        "intervals_entering": int(len(episodes)),
        "people_entering": int(episodes["person_id"].nunique()),
        "outside_comparable_window": int((~comparable).sum()),
        "in_comparable_window": int(comparable.sum()),
    }

    inside = comparable
    row["behaviour_incomplete"] = int((inside & ~core).sum())
    for d in DOMAINS:
        row[f"unobserved_{d}"] = int((inside & ~valid[d]).sum())
    row["unobserved_two_or_more_domains"] = int(
        (inside & (sum((~valid[d]).astype(int) for d in DOMAINS) >= 2)).sum())
    row["behaviour_complete"] = int((inside & core).sum())
    row["no_behaviour_available_to_lose"] = int((inside & core & ~opportunity).sum())
    row["primary_risk_set"] = int((inside & core & opportunity).sum())
    row["primary_risk_set_people"] = int(
        episodes.loc[inside & core & opportunity, "person_id"].nunique())

    row.update(status_split(episodes, cc))
    row.update(proxy_split(episodes))

    frame = pd.DataFrame([row])
    out = args.output_dir / f"{args.cohort}-exclusion-reasons.csv"
    frame.to_csv(out, index=False)
    out.chmod(0o600)

    manifest = {
        "analysis_id": "behavioural_withdrawal_exclusion_reasons_v0.1",
        "cohort": args.cohort,
        "episode_columns_available": sorted(episodes.columns.tolist()),
        "minimum_baseline_opportunities": pilot["minimum_baseline_opportunities"],
        "aggregate_only": True,
    }
    mpath = args.output_dir / f"{args.cohort}-exclusion-manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    mpath.chmod(0o600)
    print(f"{args.cohort}: {row['intervals_entering']:,} entering, "
          f"{row['primary_risk_set']:,} in the primary risk set")


if __name__ == "__main__":
    main()
