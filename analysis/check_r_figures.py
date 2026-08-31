#!/usr/bin/env python3
"""Measure every R-drawn figure and assert it is the size the layout declares.

A panel that regresses to a different size still looks like a figure; it only
shows up as "a little cramped" once it is placed beside its neighbours. This
reads the PDF media box rather than trusting the call that produced it.

The declared sizes are no longer written here. Forest heights are computed in
the R from the row count at a fixed row pitch, so a second copy of them in this
file would go stale the first time a forest gained a row, and would go stale
silently. The R writes what it drew into figures/lancet_r/sizes.csv instead and
this measures against that, plus two rules that do not depend on the content:
every figure is one of the two Lancet column widths, and nothing is taller than
a printed page.
"""
from __future__ import annotations

import csv
import pathlib
import sys

from pypdf import PdfReader

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/lancet_r"
TOL = 0.02          # inches
COLUMN_WIDTHS = (3.60, 7.20)
MAX_HEIGHT = 9.00   # a Lancet page, less its margins


def main() -> None:
    manifest = OUT / "sizes.csv"
    if not manifest.exists():
        sys.exit(f"no size manifest at {manifest}; run `make figures`")
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    if not rows:
        sys.exit("the size manifest is empty")

    problems = []
    for row in rows:
        stem = row["stem"]
        want_w, want_h = float(row["width"]), float(row["height"])
        pdf, png = OUT / f"{stem}.pdf", OUT / f"{stem}.png"
        if not pdf.exists():
            problems.append(f"{stem}: no PDF")
            continue
        if not png.exists() or png.stat().st_size == 0:
            problems.append(f"{stem}: no PNG")
        box = PdfReader(str(pdf)).pages[0].mediabox
        got_w, got_h = float(box.width) / 72, float(box.height) / 72
        if abs(got_w - want_w) > TOL or abs(got_h - want_h) > TOL:
            problems.append(f"{stem}: {got_w:.2f} x {got_h:.2f} in, "
                            f"declared {want_w:.2f} x {want_h:.2f}")
            continue
        if not any(abs(got_w - w) <= TOL for w in COLUMN_WIDTHS):
            problems.append(f"{stem}: {got_w:.2f} in wide is neither Lancet column")
        if got_h > MAX_HEIGHT:
            problems.append(f"{stem}: {got_h:.2f} in tall does not fit a page")
        print(f"  {stem:36s} {got_w:.2f} x {got_h:.2f} in")

    if problems:
        print("\nfigures that do not match their layout:", file=sys.stderr)
        for line in problems:
            print(f"  {line}", file=sys.stderr)
        sys.exit(1)
    print(f"all {len(rows)} figures are the size their layout declares")


if __name__ == "__main__":
    main()
