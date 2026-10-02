# /// script
# requires-python = ">=3.11"
# dependencies = ["pymupdf", "shapely"]
# ///
"""Extract walls and plan linework from the builder's vector floor plan PDF.

Reads page 2 of the "Elevation R Rendering and Option Floor Plan" PDF (the
builder brochure, drawn as vectors) and writes plan.json in feet.

Coordinates in plan.json: X runs left to right as seen from the street,
Y runs from the front of the garage (Y=0) toward the back yard, both in feet.
The brochure is drawn left-handed; this house was built right-handed
(mirrored), so X is flipped here. Scale is calibrated to the lot survey:
34.0 ft wide by 54.5 ft deep at the first floor.

Usage: uv run extract_plan.py <floor-plan.pdf>
"""

import json
import sys
from pathlib import Path

import pymupdf
from shapely.geometry import LineString, Polygon, box, mapping
from shapely.ops import unary_union

HERE = Path(__file__).parent

# PDF regions (points) holding each floor of the base plan; the option
# insets elsewhere on the page are ignored.
F1_BOX = (50, 268, 292, 642)
F2_BOX = (318, 88, 560, 394)
# First floor exterior extents in PDF points, and the survey's dimensions.
F1_X0, F1_X1, F1_Y0, F1_Y1 = 56.2, 285.8, 273.9, 638.3
SURVEY_W, SURVEY_D = 34.0, 54.5
SX = SURVEY_W / (F1_X1 - F1_X0)
SY = SURVEY_D / (F1_Y1 - F1_Y0)
# The second floor is drawn beside the first; these offsets stack it.
F2_DX, F2_DY = 324.6 - F1_X0, 93.0 - F1_Y0


def to_feet(x, y, floor):
    if floor == 2:
        x, y = x - F2_DX, y - F2_DY
    return round((F1_X1 - x) * SX, 4), round((F1_Y1 - y) * SY, 4)


def inside(rect, region):
    return (rect.x0 >= region[0] and rect.x1 <= region[2]
            and rect.y0 >= region[1] and rect.y1 <= region[3])


def flatten(item, steps=12):
    kind = item[0]
    if kind == "l":
        return [[item[1], item[2]]]
    if kind == "c":
        p0, p1, p2, p3 = item[1:5]
        pts = []
        for i in range(steps + 1):
            t = i / steps
            a, b, c, d = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2, t ** 3
            pts.append(pymupdf.Point(a * p0.x + b * p1.x + c * p2.x + d * p3.x,
                                     a * p0.y + b * p1.y + c * p2.y + d * p3.y))
        return [pts]
    if kind == "re":
        r = item[1]
        return [[r.tl, r.tr, r.br, r.bl, r.tl]]
    if kind == "qu":
        q = item[1]
        return [[q.ul, q.ur, q.lr, q.ll, q.ul]]
    return []


def main(pdf_path):
    page = pymupdf.open(pdf_path)[1]
    drawings = page.get_drawings()
    out = {"scale_ft_per_pt": [SX, SY], "floors": {}}
    for floor, region in ((1, F1_BOX), (2, F2_BOX)):
        polys, lines = [], []
        for d in drawings:
            if not inside(d["rect"], region):
                continue
            is_wall = (d["type"] == "fs" and d["fill"] is not None
                       and 0.003 < d["fill"][0] < 0.005)
            if is_wall:
                # Each wall piece is drawn as triangles of three lines each.
                items = [it for it in d["items"] if it[0] == "l"]
                for k in range(0, len(items) - 2, 3):
                    tri = Polygon([to_feet(it[1].x, it[1].y, floor) for it in items[k:k + 3]])
                    if tri.is_valid and tri.area > 0:
                        polys.append(tri)
            elif (d["type"] == "s" and d["width"] is not None and d["width"] <= 0.3
                  and d["color"] is not None and d["color"][0] < 0.7):
                # Thin linework: fixtures, door swings, stairs, counters.
                # Thick black strokes are the sales markup arrows; text is fill.
                for it in d["items"]:
                    for seg in flatten(it):
                        lines.append([list(to_feet(p.x, p.y, floor)) for p in seg])
        walls = unary_union([p.buffer(0.01, join_style="mitre") for p in polys]).buffer(-0.01, join_style="mitre")
        geoms = list(walls.geoms) if hasattr(walls, "geoms") else [walls]
        out["floors"][str(floor)] = {
            "walls": [mapping(g) for g in geoms],
            "lines": lines,
        }
        print(f"floor {floor}: {len(polys)} wall pieces -> {len(geoms)} wall polygons, "
              f"{len(lines)} linework strokes, bounds {walls.bounds}")
    (HERE / "plan.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1])
