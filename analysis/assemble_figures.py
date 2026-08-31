"""Composite the panels into the finished figures, in vector, at exact size.

Hand assembly in PowerPoint put panels at 104.9 per cent and clipped two cohort
labels, and the result then went stale when a panel was regenerated. This does
the same job deterministically: it places each panel PDF at its stated position
on a page of the stated size, so nothing is scaled, nothing overlaps, and the
figure is rebuilt whenever the panels are. The output stays vector, which the
PowerPoint route never managed.

Panel letters are drawn here, once, over the transparent corner each panel
reserves for them.
"""
from __future__ import annotations

import pathlib

import pymupdf

ROOT = pathlib.Path(__file__).resolve().parents[1]
PANELS = ROOT / "figures/panels"
OUT = ROOT / "figures"
PT = 72.0          # points per inch
DPI = 600          # for the raster companion

# stem -> (letter, x_in, y_in, w_in, h_in)
FIGURES = {
    "figure2_outcomes": (7.20, 7.20, [
        ("panel_fig2_a_two_year_death_risk", "a", 0.00, 0.00, 3.60, 3.00),
        ("panel_fig2_b_two_year_adl_risk",   "b", 3.60, 0.00, 3.60, 3.00),
        ("panel_fig2_c_outcome_gradient",    "c", 0.00, 3.00, 7.20, 4.20),
    ]),
    "figure_reference_group": (7.20, 3.20, [
        ("panel_reference_a_transition_states",     "a", 0.00, 0.00, 3.60, 3.20),
        ("panel_reference_b_stopped_vs_continued",  "b", 3.60, 0.00, 3.60, 3.20),
    ]),
}

LETTER_SIZE = 10          # pt, bold, lower case
LETTER_INSET = (0.05, 0.19)   # inches from the panel's top-left to the baseline


def build() -> None:
    for stem, (w_in, h_in, panels) in FIGURES.items():
        doc = pymupdf.open()
        page = doc.new_page(width=w_in * PT, height=h_in * PT)
        for name, letter, x, y, w, h in panels:
            src = PANELS / f"{name}.pdf"
            if not src.exists():
                raise SystemExit(f"missing panel: {src.name}")
            with pymupdf.open(src) as s:
                box = s[0].rect
                if abs(box.width / PT - w) > 0.02 or abs(box.height / PT - h) > 0.02:
                    raise SystemExit(
                        f"{name}: panel is {box.width/PT:.2f} x {box.height/PT:.2f} in, "
                        f"the layout expects {w:.2f} x {h:.2f}")
                page.show_pdf_page(
                    pymupdf.Rect(x * PT, y * PT, (x + w) * PT, (y + h) * PT), s, 0)
            page.insert_text(
                pymupdf.Point((x + LETTER_INSET[0]) * PT, (y + LETTER_INSET[1]) * PT),
                letter, fontname="hebo", fontsize=LETTER_SIZE, color=(0, 0, 0))
        pdf = OUT / f"{stem}.pdf"
        doc.save(pdf, deflate=True)
        pix = doc[0].get_pixmap(dpi=DPI, colorspace=pymupdf.csRGB)
        pix.save(OUT / f"{stem}.png")
        doc.close()
        print(f"wrote {stem}.pdf and .png  {w_in:.2f} x {h_in:.2f} in, "
              f"{len(panels)} panels, {pix.width}x{pix.height} px at {DPI} dpi")


if __name__ == "__main__":
    build()
