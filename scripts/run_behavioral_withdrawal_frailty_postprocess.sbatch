#!/bin/bash
#SBATCH --job-name=bw-frailty-post
#SBATCH --account=r01834
#SBATCH --partition=h100-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=56G
#SBATCH --time=01:00:00
#SBATCH --output=logs/post-%j.out
#SBATCH --error=logs/post-%j.err

set -euo pipefail
umask 077

ROOT=${ROOT:?set ROOT to the analysis directory}
PYTHONPATH="$ROOT/code:${PYTHONPATH:-}" python3 "$ROOT/code/postprocess_behavioral_withdrawal_frailty_extension.py" \
  --extension-config "$ROOT/metadata/behavioral_withdrawal_frailty_extension.json" \
  --probe-dir "$ROOT/outputs/probe" \
  --model-dir "$ROOT/outputs/models" \
  --final-dir "$ROOT/outputs/final"

PYTHONPATH="$ROOT/code:${PYTHONPATH:-}" python3 "$ROOT/code/validate_behavioral_withdrawal_frailty_extension.py" \
  --extension-config "$ROOT/metadata/behavioral_withdrawal_frailty_extension.json" \
  --probe-dir "$ROOT/outputs/probe" \
  --model-dir "$ROOT/outputs/models" \
  --final-dir "$ROOT/outputs/final" \
  --output "$ROOT/outputs/final/independent-validation.json"
