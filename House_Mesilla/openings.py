"""Find door and window openings in the extracted wall polygons.

The plan draws an opening as a gap in a wall run. Every square wall end
(an "end cap") is traced along its wall's centerline into open space; if it
reaches another wall within a door or window width, the empty strip between
is an opening.
"""

import math

from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

# Longest gap that pairs two wall ends (garage door); a wall end that runs
# into the side of another wall only counts up to a wide closet door.
MAX_PAIR_FT = 17.0
MAX_FACE_FT = 5.2


def end_caps(walls, min_t=0.12, max_t=0.75):
    """Short axis-aligned polygon edges: the square ends of wall runs.

    Returns (axis, pos, lo, hi, direction): for a wall running along X
    (axis "h") the cap sits at x=pos spanning y in [lo, hi], and direction
    is +1/-1 for the side facing open space.
    """
    union = unary_union(walls)
    caps = []
    for poly in walls:
        ring = list(poly.simplify(0.02).exterior.coords)
        for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
            length = math.hypot(x2 - x1, y2 - y1)
            if not min_t <= length <= max_t:
                continue
            if abs(x1 - x2) < 0.02:
                axis, pos, lo, hi = "h", x1, min(y1, y2), max(y1, y2)
                probe = lambda d: Point(pos + d * 0.06, (lo + hi) / 2)
            elif abs(y1 - y2) < 0.02:
                axis, pos, lo, hi = "v", y1, min(x1, x2), max(x1, x2)
                probe = lambda d: Point((lo + hi) / 2, pos + d * 0.06)
            else:
                continue
            for d in (1, -1):
                if not union.contains(probe(d)) and union.contains(probe(-d)):
                    caps.append((axis, pos, lo, hi, d))
    return caps


def strip(axis, a, b, lo, hi):
    a, b = sorted((a, b))
    return box(a, lo, b, hi) if axis == "h" else box(lo, a, hi, b)


def find_openings(walls):
    """Return openings as dicts with rect bounds, axis and width."""
    union = unary_union(walls)
    caps = end_caps(walls)
    candidates = []
    for cap in caps:
        axis, pos, lo, hi, d = cap
        # Facing cap on the same centerline -> paired opening.
        for other in caps:
            if other is cap or other[0] != axis or other[4] != -d:
                continue
            if abs(other[2] - lo) > 0.12 or abs(other[3] - hi) > 0.12:
                continue
            gap = (other[1] - pos) * d
            if 0.8 <= gap <= MAX_PAIR_FT:
                rect = strip(axis, pos, other[1], max(lo, other[2]), min(hi, other[3]))
                if not rect.buffer(-0.02).intersects(union):
                    candidates.append((gap, rect, axis))
        # Ray into the side face of another wall.
        ray = strip(axis, pos, pos + d * MAX_FACE_FT, lo + 0.02, hi - 0.02)
        hit = ray.intersection(union)
        if hit.is_empty:
            continue
        if axis == "h":
            ends = [g.bounds[0] if d > 0 else g.bounds[2] for g in getattr(hit, "geoms", [hit])]
        else:
            ends = [g.bounds[1] if d > 0 else g.bounds[3] for g in getattr(hit, "geoms", [hit])]
        far = min(ends, key=lambda e: (e - pos) * d)
        gap = (far - pos) * d
        if gap < 0.8:
            continue
        rect = strip(axis, pos, far, lo, hi)
        # The wall that was hit must span the whole strip width.
        cover = strip(axis, far, far + d * 0.1, lo, hi)
        if cover.difference(union).area > 0.35 * cover.area:
            continue
        candidates.append((gap, rect, axis))
    candidates.sort(key=lambda c: c[0])
    chosen = []
    for gap, rect, axis in candidates:
        if any(rect.buffer(-0.03).intersects(c[1]) for c in chosen):
            continue
        chosen.append((gap, rect, axis))
    return [{"rect": r.bounds, "axis": a, "width": round(g, 3)} for g, r, a in chosen]


def outline(walls, openings):
    """Building outline: walls plus filled openings, holes removed."""
    shape = unary_union(list(walls) + [box(*o["rect"]) for o in openings]).buffer(0.005)
    geoms = list(shape.geoms) if hasattr(shape, "geoms") else [shape]
    big = max(geoms, key=lambda g: g.area)
    return Polygon(big.exterior).buffer(-0.005, join_style="mitre")
