#!/bin/bash
#SBATCH --job-name=bmc-risk-post
#SBATCH --account=r01834
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/postprocess-%j.out
#SBATCH --error=logs/postprocess-%j.err

set -euo pipefail
umask 077

ROOT=${ROOT:?set ROOT to the analysis directory}
python3 "$ROOT/code/postprocess_bmc_absolute_risk_work_exit_extension.py" \
  --extension-config "$ROOT/metadata/work_exit_reason_extension.json" \
  --model-dir "$ROOT/outputs/models" \
  --output-dir "$ROOT/outputs/final"
