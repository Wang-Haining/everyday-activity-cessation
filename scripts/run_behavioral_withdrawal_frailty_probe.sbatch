#!/bin/bash
#SBATCH --job-name=bw-frailty-probe
#SBATCH --account=r01834
#SBATCH --partition=h100-debug
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=56G
#SBATCH --time=01:00:00
#SBATCH --output=logs/probe-%A_%a.out
#SBATCH --error=logs/probe-%A_%a.err

set -euo pipefail
umask 077

ROOT=${ROOT:?set ROOT to the analysis directory}
COHORTS=(charls elsa hrs klosa mhas share)
COHORT=${COHORTS[$SLURM_ARRAY_TASK_ID]}

PYTHONPATH="$ROOT/code:${PYTHONPATH:-}" python3 "$ROOT/code/probe_behavioral_withdrawal_frailty_extension.py" \
  --universe-config "$ROOT/metadata/behavioral_transition_outcome_scan_universe.json" \
  --multidomain-config "$ROOT/metadata/multidomain_behavioral_withdrawal_pilot.json" \
  --extension-config "$ROOT/metadata/behavioral_withdrawal_frailty_extension.json" \
  --design "$ROOT/metadata/behavioral_withdrawal_frailty_extension.md" \
  --cohort "$COHORT" \
  --output-dir "$ROOT/outputs/probe" \
  --design-commit c8b3e99a2d430ef0748a307f550cfa9109121dbc \
  --design-commit-time 2026-08-24T20:58:54-04:00 \
  --code-commit "${CODE_COMMIT:?CODE_COMMIT must be exported by sbatch}"
