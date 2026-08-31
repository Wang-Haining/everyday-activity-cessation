#!/bin/bash
#SBATCH --job-name=bw-frailty-pooled
#SBATCH --account=r01834
#SBATCH --partition=h100-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/pooled-%j.out
#SBATCH --error=logs/pooled-%j.err

set -euo pipefail
umask 077

ROOT=${ROOT:?set ROOT to the analysis directory}
PYTHONPATH="$ROOT/code:${PYTHONPATH:-}" python3 "$ROOT/code/run_behavioral_withdrawal_frailty_pooled_sensitivity.py" \
  --universe-config "$ROOT/metadata/behavioral_transition_outcome_scan_universe.json" \
  --multidomain-config "$ROOT/metadata/multidomain_behavioral_withdrawal_pilot.json" \
  --extension-config "$ROOT/metadata/behavioral_withdrawal_frailty_extension.json" \
  --design "$ROOT/metadata/behavioral_withdrawal_frailty_extension.md" \
  --probe-dir "$ROOT/outputs/probe" \
  --output-dir "$ROOT/outputs/final" \
  --design-commit c8b3e99a2d430ef0748a307f550cfa9109121dbc \
  --design-commit-time 2026-08-24T20:58:54-04:00 \
  --code-commit "${CODE_COMMIT:?CODE_COMMIT must be exported by sbatch}"
