#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HARMONIZED_ROOT:?Set HARMONIZED_ROOT to the harmonized data directory including derived_fi}"
U=config/behavioral_transition_outcome_scan_universe_v0.1.json
M=config/multidomain_behavioral_withdrawal_pilot_v0.1.json
F=config/behavioral_withdrawal_frailty_extension_v0.2.json
for c in charls elsa hrs share; do
  test -f "$HARMONIZED_ROOT/derived_fi/$c-person-wave.parquet" || { echo "Missing FI input: $c-person-wave.parquet" >&2; exit 1; }
 done
for c in charls elsa hrs share; do
  python scripts/probe_behavioral_withdrawal_frailty_extension.py --data-root "$HARMONIZED_ROOT" --universe-config "$U" --multidomain-config "$M" --extension-config "$F" --cohort "$c" --output-dir results/frailty/probe
  python scripts/run_behavioral_withdrawal_frailty_extension.py --data-root "$HARMONIZED_ROOT" --universe-config "$U" --multidomain-config "$M" --extension-config "$F" --cohort "$c" --probe-dir results/frailty/probe --output-dir results/frailty/models
  python scripts/run_behavioral_withdrawal_fi_context.py --data-root "$HARMONIZED_ROOT" --universe-config "$U" --multidomain-config "$M" --extension-config "$F" --cohort "$c" --output-dir results/context/frailty
 done
python - <<'PYCODE'
from pathlib import Path
import pandas as pd
root = Path('results/frailty')
paths = [root / 'models' / f'{cohort}-frailty-models.csv' for cohort in ['charls', 'elsa', 'hrs', 'share']]
frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
assert len(frame) > 0, f'expected frailty rows, got {len(frame)}'
frame.to_csv(root / 'systematic-frailty-results.csv', index=False)
print({'frailty_rows': len(frame), 'cohort_files': len(paths)})
PYCODE
