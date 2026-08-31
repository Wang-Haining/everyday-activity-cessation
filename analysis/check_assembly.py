"""Check a hand-assembled figure against the panels it was built from.

PowerPoint will place a panel at 105 per cent without saying so and let the
neighbour's white background cover the overflow. The result looks right at
thumbnail size and has lost a cohort label at full size. This measures the
x-axis spine of each panel in its source file and again in the assembled file,
reports the scale each panel was actually placed at, and reports any overlap.
It also checks resolution and looks for a frame drawn around the whole figure.

    python3 analysis/check_assembly.py figures/figure2_outcomes.png 7.20 \
        0:0.36  panel_fig2_a_two_year_death_risk panel_fig2_b_two_year_adl_risk

The third argument is the band of the assembled image, as fractions of its
height, holding that row of panels. Run once per row.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
PANELS = ROOT / "figures/panels"
PANEL_DPI = 600.0
SCALE_TOL = 0.02


def spine(img: np.ndarray, rows: tuple[int, int], cols: tuple[int, int]):
    """Longest near-solid horizontal dark run: the x-axis of a panel."""
    sub = img[rows[0]:rows[1], cols[0]:cols[1]] < 100
    best = None
    for r in range(sub.shape[0]):
        c = np.where(sub[r])[0]
        if len(c) < 50:
            continue
        run = c[-1] - c[0] + 1
        if run >= 0.98 * sub.shape[1]:
            continue  # a frame edge, not an axis
        if sub[r].sum() > 0.9 * run and (best is None or run > best[0]):
            best = (run, rows[0] + r, cols[0] + c[0], cols[0] + c[-1])
    return best


def right_clearance(path: pathlib.Path) -> float:
    im = np.array(Image.open(path).convert("L"))
    cols = np.where((im < 250).any(axis=0))[0]
    return (im.shape[1] - 1 - cols[-1]) / PANEL_DPI


def frame_check(big: np.ndarray) -> list[str]:
    edges = {"top": big[0], "bottom": big[-1], "left": big[:, 0],
             "right": big[:, -1]}
    return [k for k, v in edges.items() if (v < 100).mean() > 0.9]


def strip_frame(big: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Crop a solid border off, so it cannot be mistaken for an axis."""
    top, bottom, left, right = 0, big.shape[0], 0, big.shape[1]
    while top < bottom and (big[top] < 100).mean() > 0.9:
        top += 1
    while bottom > top and (big[bottom - 1] < 100).mean() > 0.9:
        bottom -= 1
    while left < right and (big[:, left] < 100).mean() > 0.9:
        left += 1
    while right > left and (big[:, right - 1] < 100).mean() > 0.9:
        right -= 1
    return big[top:bottom, left:right], left, top


def main() -> int:
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    assembled = pathlib.Path(sys.argv[1])
    width_in = float(sys.argv[2])
    lo_f, hi_f = (float(v) for v in sys.argv[3].split(":"))
    stems = sys.argv[4:]

    big = np.array(Image.open(assembled).convert("L"))
    height, width = big.shape
    dpi = width / width_in
    bad = False

    print(f"{assembled.name}: {width}x{height} px, {dpi:.1f} dpi at "
          f"{width_in:.2f} in wide")
    if dpi < 300:
        print("  FAIL  below the 300 dpi the journal asks for at final size")
        bad = True

    framed = frame_check(big)
    if framed:
        print(f"  FAIL  a solid line runs along the {', '.join(framed)} edge"
              f"{'s' if len(framed) > 1 else ''} of the image; journals do not "
              f"want a box drawn round a figure")
        bad = True

    big, dx, dy = strip_frame(big)
    height, width = big.shape
    rows = (int(height * lo_f), int(height * hi_f))
    n = len(stems)
    spans: list[tuple[str, float, float]] = []
    for i, stem in enumerate(stems):
        src = PANELS / f"{stem}.png"
        small = np.array(Image.open(src).convert("L"))
        ref = spine(small, (0, small.shape[0]), (0, small.shape[1]))
        band = (int(width * i / n) + 3, int(width * (i + 1) / n) - 3)
        got = spine(big, rows, band)
        if ref is None or got is None:
            print(f"  FAIL  {stem}: no axis spine found")
            bad = True
            continue
        scale = (got[0] / dpi) / (ref[0] / PANEL_DPI)
        placed = (small.shape[1] / PANEL_DPI) * scale
        left = (got[2] - ref[2] * scale * dpi / PANEL_DPI) / dpi
        spans.append((stem, left, left + placed))
        ok = abs(scale - 1) <= SCALE_TOL
        bad |= not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {stem}: placed {placed:.3f} in "
              f"({scale * 100:.1f}% of its native {small.shape[1] / PANEL_DPI:.2f} in), "
              f"left edge at {left:.3f} in")
        print(f"        its own right margin is {right_clearance(src):.3f} in, "
              f"so an overlap past that clips the labels")

    for (s1, _, r1), (s2, l2, _) in zip(spans, spans[1:]):
        gap = l2 - r1
        if gap < -0.005:
            print(f"  FAIL  {s2} sits over {s1} by {-gap:.3f} in, covering "
                  f"whatever is in that strip")
            bad = True
        elif gap > 0.02:
            print(f"  note  {gap:.3f} in of white between {s1} and {s2}")

    print("FAIL" if bad else "PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
