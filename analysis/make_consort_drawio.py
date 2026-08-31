"""Draw the study flow diagram from the frozen flow counts.

Classic CONSORT: white boxes of one width, black rules, orthogonal connectors
and no diagonals. Every number is read from the same aggregate release the
manuscript is validated against, so the diagram cannot drift from the text.

Three outputs, all written here from one list of boxes: the PDF the appendix
includes, a 600 dpi PNG, and a .drawio copy.

The .drawio is a convenience copy for looking at the diagram in draw.io. It is
not the source and editing it changes nothing that ships, because the PDF is
drawn by this script rather than exported from it. Change the wording or the
geometry here and run this again; a hand edit in draw.io is lost on the next
run and, until then, sits beside a PDF that disagrees with it.

Counts are formatted by the house rule in lancet_numbers, which spaces
thousands from five digits up and leaves four unseparated. They used to be
formatted with commas here, so the appendix printed 205,794 against the
manuscript's 205 794 on the facing page.
"""
from __future__ import annotations

import html
import pathlib
import re
import sys

import lancet_numbers
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DESC = ROOT / "artifacts/behavioral_withdrawal_frailty_extension/manuscript_descriptive"
PILOT = ROOT / "artifacts/multidomain_behavioral_withdrawal_pilot/final"
OUT = ROOT / "figures/consort_diagram.drawio"
SCOPE = "comparable_22_30_months"

STAGES = [
    ("source_respondents",
     "Respondents in the six harmonised ageing cohorts"),
    ("age_60_three_wave_intervals",
     "Aged 60 years or older with three consecutive interviews"),
    ("comparable_22_30_month_intervals",
     "Outcome interview scheduled 22 to 30 months after the second interview"),
    ("three_behaviors_observed",
     "All three activities observed at the first and second interview"),
    ("primary_behavior_risk_set",
     "At least one activity present initially, with complete model covariates"),
]
OUTCOMES = [("mortality", "Death"),
            ("incident_any_adl", "New ADL limitation"),
            ("incident_any_iadl", "New IADL limitation")]

# Geometry, in points. One width for every box in the spine, one for every
# exclusion, so the diagram reads as a column rather than as a collage.
MAIN_X, MAIN_W = 40, 384
EXCL_X, EXCL_W = 470, 210
TOP, MAIN_H, GAP = 28, 46, 44
EXCL_H = 30
OUT_W, OUT_H, OUT_GAP = 122, 62, 9

BOX = ("rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#000000;"
       "strokeWidth=1;align=left;verticalAlign=middle;spacingLeft=8;"
       "spacingRight=8;fontFamily=Helvetica;fontSize=11;fontColor=#000000;")
OUTBOX = BOX.replace("align=left", "align=center").replace("spacingLeft=8;spacingRight=8;", "")
EDGE = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor=#000000;"
        "strokeWidth=1;endArrow=block;endFill=1;endSize=6;jumpStyle=none;")
NOTE = ("text;html=1;align=left;verticalAlign=middle;fontFamily=Helvetica;"
        "fontSize=10;fontColor=#000000;")


def cell(cid, value, style, x, y, w, h, align="left"):
    # A raw newline in an XML attribute collapses to a space. Escaping first
    # and then inserting the break means the parser hands draw.io a real <br>.
    label = html.escape(value).replace("\n", "&lt;br&gt;")
    return (f'<mxCell id="{cid}" value="{label}" style="{style}" '
            f'vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>'
            f'</mxCell>')


def edge(cid, src, dst, points, exit_xy=(0.5, 1.0), entry_xy=(0.5, 0.0)):
    """A connector that carries its own route.

    source and target stay, so draw.io keeps the boxes connected if someone
    moves one. The waypoints are written out as well, so the drawing does not
    depend on which router opens the file: what draw.io shows and what
    render_drawio.py draws are then the same polyline.
    """
    style = (EDGE + f"exitX={exit_xy[0]};exitY={exit_xy[1]};exitDx=0;exitDy=0;"
                    f"entryX={entry_xy[0]};entryY={entry_xy[1]};entryDx=0;entryDy=0;")
    route = "".join(f'<mxPoint x="{x:g}" y="{y:g}"/>' for x, y in points)
    return (f'<mxCell id="{cid}" style="{style}" edge="1" parent="1" '
            f'source="{src}" target="{dst}">'
            f'<mxGeometry relative="1" as="geometry">'
            f'<Array as="points">{route}</Array>'
            f'</mxGeometry></mxCell>')


def build(check: bool = False) -> None:
    flow = pd.read_csv(DESC / "manuscript-flow.csv")
    mat = pd.read_csv(PILOT / "systematic-results-matrix.csv")
    contrib = mat[mat.scope.eq(SCOPE) & mat.adjustment.eq("full")
                  & mat.exposure_model.eq("any_withdrawal")
                  & mat.term.eq("any_withdrawal")
                  & mat.model_status.eq("PASS")].drop_duplicates(
                      ["cohort", "outcome_id"])

    cx = MAIN_X + MAIN_W / 2      # the spine every connector leaves from
    cells, prev_intervals, y = [], None, TOP
    spine_y = []
    for i, (stage, title) in enumerate(STAGES):
        rows = flow[flow.stage.eq(stage)]
        people = int(rows.people.sum())
        iv = rows.intervals.sum()
        intervals = None if pd.isna(iv) or iv == 0 else int(iv)
        count = (f"{lancet_numbers.count(intervals)} person-intervals from "
                          f"{lancet_numbers.count(people)} participants"
                 if intervals else f"{lancet_numbers.count(people)} participants")
        cells.append(cell(f"s{i}", f"{title}\n{count}", BOX,
                          MAIN_X, y, MAIN_W, MAIN_H))
        spine_y.append(y)
        if i:
            cells.append(edge(f"e{i}", f"s{i-1}", f"s{i}",
                              [(cx, y - GAP), (cx, y)]))
            if intervals is not None and prev_intervals:
                dropped = prev_intervals - intervals
                if dropped > 0:
                    ey = y - GAP + (GAP - EXCL_H) / 2
                    cells.append(cell(f"x{i}",
                                      f"{lancet_numbers.count(dropped)} person-intervals excluded",
                                      BOX, EXCL_X, ey, EXCL_W, EXCL_H))
                    my = ey + EXCL_H / 2
                    cells.append(edge(f"ex{i}", f"s{i-1}", f"x{i}",
                                      [(cx, my), (EXCL_X, my)],
                                      exit_xy=(0.5, 1.0), entry_xy=(0.0, 0.5)))
        prev_intervals = intervals if intervals is not None else prev_intervals
        y += MAIN_H + GAP

    # The outcome row. Three orthogonal edges from one exit point give the
    # classic down, across, down. No diagonals anywhere in the file.
    row_w = len(OUTCOMES) * OUT_W + (len(OUTCOMES) - 1) * OUT_GAP
    x0 = MAIN_X + (MAIN_W - row_w) / 2
    for j, (oid, label) in enumerate(OUTCOMES):
        g = contrib[contrib.outcome_id.eq(oid)]
        n, ev, k = int(g.n.sum()), int(g.events.sum()), len(g)
        cells.append(cell(
            f"o{j}", f"{label}\n{lancet_numbers.count(n)} intervals\n"
            f"{lancet_numbers.count(ev)} events\n{k} cohorts",
            OUTBOX, x0 + j * (OUT_W + OUT_GAP), y, OUT_W, OUT_H,
            align="center"))
        ox = x0 + j * (OUT_W + OUT_GAP) + OUT_W / 2
        cells.append(edge(f"eo{j}", f"s{len(STAGES)-1}", f"o{j}",
                          [(cx, y - GAP), (cx, y - 20), (ox, y - 20), (ox, y)]))

    cells.append(cell("note",
                      "MHAS had no outcome interview scheduled within the "
                      "22 to 30 month window and contributed to sensitivity "
                      "analyses only.",
                      NOTE, MAIN_X, y + OUT_H + 14,
                      EXCL_X + EXCL_W - MAIN_X, 26))

    # The canvas goes in the file, so the renderer crops where the diagram
    # ends rather than guessing a margin.
    page_w = EXCL_X + EXCL_W + 40
    page_h = y + OUT_H + 52
    xml = (
        '<mxfile host="app.diagrams.net">\n'
        '  <diagram name="CONSORT" id="consort">\n'
        '    <mxGraphModel dx="1000" dy="800" grid="0" gridSize="10" guides="1" '
        'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
        f'pageWidth="{page_w:g}" pageHeight="{page_h:g}" '
        'math="0" shadow="0">\n'
        "      <root>\n"
        '        <mxCell id="0"/>\n'
        '        <mxCell id="1" parent="0"/>\n'
        + "".join("        " + c + "\n" for c in cells) +
        "      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n")
    counts = sorted(set(re.findall(r"\d[\d\u00a0]*", " ".join(
        html.unescape(v) for c in cells if "vertex" in c
        for v in re.findall(r'value="([^"]*)"', c)))))
    if check:
        # The .drawio is the drawing, so it can be edited by hand. Layout is
        # nobody's business but the editor's; a number is. This asserts that
        # every count still in the file is one the frozen release produces, and
        # that none has gone missing, without caring where the boxes sit.
        if not OUT.exists():
            sys.exit(f"refusing: {OUT.relative_to(ROOT)} has not been built")
        # labels only: dx, pageWidth and the geometry are the file's business
        labels = " ".join(html.unescape(v) for v in re.findall(
            r'value="([^"]*)"', OUT.read_text(encoding="utf-8")))
        have = sorted(set(re.findall(r"\d[\d\u00a0]*", labels)))
        stray = [c for c in have if c not in counts]
        missing = [c for c in counts if c not in have]
        if stray or missing:
            sys.exit("the flow diagram disagrees with the frozen release:\n"
                     + (f"  not in the release: {', '.join(stray)}\n" if stray else "")
                     + (f"  missing from the file: {', '.join(missing)}" if missing else ""))
        print(f"flow diagram: {len(counts)} counts all match the frozen release")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(xml)
    # imported here rather than at the top: --check reads the file and needs no
    # drawing library, and a checker that cannot run without one is a checker
    # that gets skipped
    import render_drawio
    render_drawio.draw(OUT, OUT.with_suffix(""))
    print(f"wrote {OUT.relative_to(ROOT)}  "
          f"{len(STAGES)} spine boxes, {len(OUTCOMES)} outcome boxes, "
          f"{sum(1 for c in cells if 'edge=' in c)} orthogonal connectors")


if __name__ == "__main__":
    build(check="--check" in sys.argv)
