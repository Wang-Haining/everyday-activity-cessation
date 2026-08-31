#!/bin/bash
#SBATCH --job-name=md-withdraw-probe
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
python3 "$ROOT/code/run_multidomain_behavioral_withdrawal_pilot.py" \
  --mode probe \
  --universe-config "$ROOT/metadata/behavioral_transition_outcome_scan_universe.json" \
  --pilot-config "$ROOT/metadata/multidomain_behavioral_withdrawal_pilot.json" \
  --design "$ROOT/metadata/multidomain_behavioral_withdrawal_pilot.md" \
  --cohort "$COHORT" \
  --output-dir "$ROOT/outputs/probe" \
  --design-commit eb82133b938ca104ad8d6a2c9c3255f6d9066e6e \
  --design-commit-time 2026-08-24T17:49:48-04:00 \
  --code-commit "${CODE_COMMIT:?CODE_COMMIT must be exported by sbatch}"
