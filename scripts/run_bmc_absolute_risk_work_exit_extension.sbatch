#!/bin/bash
#SBATCH --job-name=bmc-risk-work-exit
#SBATCH --account=r01834
#SBATCH --partition=h100-debug
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=56G
#SBATCH --time=01:00:00
#SBATCH --output=logs/models-%A_%a.out
#SBATCH --error=logs/models-%A_%a.err

set -euo pipefail
umask 077

ROOT=${ROOT:?set ROOT to the analysis directory}
COHORTS=(charls elsa hrs klosa mhas share)
COHORT=${COHORTS[$SLURM_ARRAY_TASK_ID]}
python3 "$ROOT/code/run_bmc_absolute_risk_work_exit_extension.py" \
  --universe-config "$ROOT/metadata/behavioral_transition_outcome_scan_universe.json" \
  --pilot-config "$ROOT/metadata/multidomain_behavioral_withdrawal_pilot.json" \
  --extension-config "$ROOT/metadata/work_exit_reason_extension.json" \
  --design "$ROOT/metadata/work_exit_reason_extension.md" \
  --cohort "$COHORT" \
  --output-dir "$ROOT/outputs/models" \
  --design-commit ff797c4a0b1e8df009a494d8b43460e8443e40d4 \
  --design-commit-time 2026-08-25T14:58:48-04:00 \
  --code-commit "${CODE_COMMIT:?CODE_COMMIT must be exported by sbatch}"
