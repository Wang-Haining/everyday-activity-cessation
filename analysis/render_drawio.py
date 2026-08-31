#!/usr/bin/env python3
"""Draw a .drawio file to PDF and PNG, so the .drawio is the figure.

The study flow diagram used to be drawn twice: once into a .drawio and once,
by separate code, into the PDF the appendix includes. Two drawings of one
diagram is one drawing too many, and the .drawio was the copy nobody rendered,
so nothing would have said if it drifted. Now the .drawio is the only drawing
and this renders it.

This understands exactly the vocabulary that file uses: rectangles and plain
text labels, and orthogonal connectors whose route is stored in the file. It
**refuses** on anything else, including a style key it does not know. That
refusal is the point: draw.io can express far more than this draws, so a shape
added by hand has to fail loudly here rather than be dropped from the PDF
without a word.

    python analysis/render_drawio.py figures/consort_diagram.drawio
"""
from __future__ import annotations

import html
import pathlib
import re
import sys
import xml.etree.ElementTree as ET

import pymupdf

FONT = "helv"                 # Helvetica, which is what the file asks for
KNOWN_VERTEX_KEYS = {
    "rounded", "whiteSpace", "html", "fillColor", "strokeColor", "strokeWidth",
    "align", "verticalAlign", "spacingLeft", "spacingRight", "fontFamily",
    "fontSize", "fontColor", "text", "points",
}
KNOWN_EDGE_KEYS = {
    "edgeStyle", "rounded", "html", "strokeColor", "strokeWidth", "endArrow",
    "endFill", "endSize", "jumpStyle", "exitX", "exitY", "exitDx", "exitDy",
    "entryX", "entryY", "entryDx", "entryDy",
}
# draw.io's point sizes are CSS pixels; the PDF is drawn in points. The label
# text is set a little smaller than the box's nominal size for the same reason
# the browser and the print engine disagree about what 11 means.
TEXT_SCALE = 0.78
LEADING = 1.32
STROKE = 0.9


def parse_style(style: str) -> dict:
    out = {}
    for part in style.split(";"):
        if not part:
            continue
        k, _, v = part.partition("=")
        out[k] = v
    return out


def check_style(cid: str, style: dict, known: set) -> None:
    unknown = sorted(set(style) - known)
    if unknown:
        sys.exit(f"refusing: cell {cid!r} uses style keys this renderer does "
                 f"not draw: {', '.join(unknown)}. Teach it, or draw the shape "
                 f"another way.")


def label_lines(value: str) -> list[str]:
    text = re.sub(r"<br\s*/?>", "\n", value or "", flags=re.I)
    if re.search(r"<[^>]+>", text):
        sys.exit(f"refusing: a label carries markup this renderer does not "
                 f"draw: {text[:60]!r}")
    return html.unescape(text).split("\n")


def read(path: pathlib.Path):
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    model = root.find(".//mxGraphModel")
    if model is None:
        sys.exit("refusing: no mxGraphModel; is this a compressed .drawio?")
    page = (float(model.get("pageWidth")), float(model.get("pageHeight")))
    vertices, edges = [], []
    for cell in model.findall("./root/mxCell"):
        style = parse_style(cell.get("style") or "")
        cid = cell.get("id")
        if cell.get("vertex") == "1":
            check_style(cid, style, KNOWN_VERTEX_KEYS)
            g = cell.find("mxGeometry")
            vertices.append(dict(
                id=cid, lines=label_lines(cell.get("value")),
                x=float(g.get("x")), y=float(g.get("y")),
                w=float(g.get("width")), h=float(g.get("height")),
                framed="text" not in style,
                align=style.get("align", "left"),
                size=float(style.get("fontSize", 11)),
                pad=float(style.get("spacingLeft", 0)),
            ))
        elif cell.get("edge") == "1":
            check_style(cid, style, KNOWN_EDGE_KEYS)
            pts = [(float(p.get("x")), float(p.get("y")))
                   for p in cell.findall("mxGeometry/Array[@as='points']/mxPoint")]
            if len(pts) < 2:
                sys.exit(f"refusing: edge {cid!r} has no stored route. Every "
                         f"connector must carry its own waypoints, so that what "
                         f"draw.io shows and what this draws cannot differ.")
            edges.append(dict(id=cid, points=pts,
                              arrow=style.get("endArrow", "none") != "none",
                              head=float(style.get("endSize", 6))))
    return page, vertices, edges


def draw(path: pathlib.Path, out_stem: pathlib.Path, dpi: int = 600) -> None:
    (width, height), vertices, edges = read(path)
    doc = pymupdf.open()
    page = doc.new_page(width=width, height=height)
    black = (0, 0, 0)

    for v in vertices:
        if v["framed"]:
            page.draw_rect(pymupdf.Rect(v["x"], v["y"], v["x"] + v["w"],
                                        v["y"] + v["h"]),
                           color=black, width=STROKE)
        size = v["size"] * TEXT_SCALE
        lead = size * LEADING
        ty = v["y"] + (v["h"] - lead * len(v["lines"])) / 2 + size
        for line in v["lines"]:
            tw = pymupdf.get_text_length(line, fontname=FONT, fontsize=size)
            tx = (v["x"] + v["pad"] if v["align"] == "left"
                  else v["x"] + (v["w"] - tw) / 2)
            page.insert_text((tx, ty), line, fontname=FONT, fontsize=size,
                             color=black)
            ty += lead

    for e in edges:
        pts = e["points"]
        for a, b in zip(pts, pts[1:]):
            page.draw_line(pymupdf.Point(*a), pymupdf.Point(*b),
                           color=black, width=STROKE)
        if e["arrow"]:
            (px, py), (qx, qy) = pts[-2], pts[-1]
            k = e["head"]
            if px == qx:                       # vertical approach
                s = 1 if qy > py else -1
                tri = [(qx, qy), (qx - k / 2, qy - s * k), (qx + k / 2, qy - s * k)]
            else:                              # horizontal approach
                s = 1 if qx > px else -1
                tri = [(qx, qy), (qx - s * k, qy - k / 2), (qx - s * k, qy + k / 2)]
            page.draw_polyline([pymupdf.Point(*p) for p in tri],
                               color=black, fill=black, width=0.2, closePath=True)

    doc.save(out_stem.with_suffix(".pdf"), deflate=True)
    doc[0].get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB).save(
        out_stem.with_suffix(".png"))
    doc.close()
    print(f"drew {out_stem.name}.pdf and .png from {path.name}  "
          f"{width/72:.2f} x {height/72:.2f} in, "
          f"{len(vertices)} shapes, {len(edges)} connectors")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = pathlib.Path(sys.argv[1])
    draw(src, src.with_suffix(""))


if __name__ == "__main__":
    main()
