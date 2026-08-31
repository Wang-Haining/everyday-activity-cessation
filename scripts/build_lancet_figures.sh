#!/usr/bin/env bash
# Build the three manuscript figures on IU HPC Quartz.
#
# This machine's R has no third-party packages and no working cairo (XQuartz is
# not installed), so the figures are built where the rest of the analysis runs.
# Quartz carries ggplot2, ragg, systemfonts and textshaping in the module, and
# svglite, patchwork and cowplot in ~/R/lib_4.6.1. systemfonts resolves "Arial"
# to Liberation Sans, which is metrically identical to it.
#
#     bash scripts/build_lancet_figures.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_DIR="${REMOTE_DIR:-\$HOME/lancet_figures}"
HOST="${HOST:-quartz}"

echo "sending the figure data and the R to $HOST"
ssh "$HOST" "mkdir -p $REMOTE_DIR/figures/data $REMOTE_DIR/figures/R"
rsync -q "$ROOT"/figures/data/*.csv "$HOST:$REMOTE_DIR/figures/data/"
rsync -q "$ROOT"/figures/R/*.R      "$HOST:$REMOTE_DIR/figures/R/"

echo "drawing"
ssh "$HOST" "module load r/4.6.1 >/dev/null 2>&1; \
  cd $REMOTE_DIR && MS_ROOT=\$PWD R_LIBS_USER=\$HOME/R/lib_4.6.1 \
  Rscript figures/R/figures_gg.R" 2>&1 | grep -viE "^(Loading|during startup|[0-9]+: Setting|Warning message)" || true

echo "bringing the figures back"
mkdir -p "$ROOT/figures/lancet_r"
rsync -qr --delete "$HOST:$REMOTE_DIR/figures/lancet_r/" "$ROOT/figures/lancet_r/"
ls -1 "$ROOT/figures/lancet_r" | sed 's/^/  /'
