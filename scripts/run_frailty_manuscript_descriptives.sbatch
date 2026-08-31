#!/bin/bash
#SBATCH --job-name=bw-desc
#SBATCH --account=r01834
#SBATCH --partition=h100-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=56G
#SBATCH --time=01:00:00
#SBATCH --output=logs/descriptive-%j.out
#SBATCH --error=logs/descriptive-%j.err

set -euo pipefail
umask 077

PROJECT_ROOT=${ROOT:?set ROOT to the analysis directory}
CODE_COMMIT=${CODE_COMMIT:?CODE_COMMIT is required}
mkdir -p "$PROJECT_ROOT/outputs/manuscript_descriptive"
python3 "$PROJECT_ROOT/code/probe_frailty_manuscript_descriptives.py" \
  --universe-config "$PROJECT_ROOT/metadata/behavioral_transition_outcome_scan_universe.json" \
  --multidomain-config "$PROJECT_ROOT/metadata/multidomain_behavioral_withdrawal_pilot.json" \
  --extension-config "$PROJECT_ROOT/metadata/behavioral_withdrawal_frailty_extension.json" \
  --output-dir "$PROJECT_ROOT/outputs/manuscript_descriptive" \
  --code-commit "$CODE_COMMIT"
