#!/bin/bash
#SBATCH --job-name=bc-opportunity-post
#SBATCH --account=r01834
#SBATCH --partition=h100-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=56G
#SBATCH --time=01:00:00
#SBATCH --output=logs/opportunity-post-%j.out
#SBATCH --error=logs/opportunity-post-%j.err

set -euo pipefail
umask 077
ROOT=${ROOT:?set ROOT to the analysis directory}

PYTHONPATH="$ROOT/code:${PYTHONPATH:-}" python3 "$ROOT/code/postprocess_baseline_opportunity_sensitivity.py" \
  --model-dir "$ROOT/outputs/baseline-opportunity-v0.1/models" \
  --output-dir "$ROOT/outputs/baseline-opportunity-v0.1/final"
