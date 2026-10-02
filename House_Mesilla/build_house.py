# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "shapely", "trimesh", "mapbox-earcut", "manifold3d"]
# ///
"""Build the house model from plan.json.

Outputs (next to this script):
  house.glb          every part as a named, colored mesh (meters, Y up)
  house_viewer.html  self-contained 3D viewer: layers, explode, section cut
  print/*.stl        stackable printing pieces at PRINT_SCALE, in mm

Layout of the world (feet, Z up): X runs left to right seen from the street,
Y runs from the garage door (Y=0) toward the back yard, Z=0 is the finished
first floor. Sources for every number are in README.md.

Usage: uv run build_house.py
"""

import base64
import json
import math
from pathlib import Path

import numpy as np
import trimesh
from shapely.affinity import scale as shp_scale
from shapely.geometry import LineString, Point, Polygon, box, shape
from shapely.ops import unary_union
from trimesh.visual.material import PBRMaterial

from openings import find_openings, outline

HERE = Path(__file__).parent
FT = 0.3048
PRINT_SCALE = 80  # 1:80 fits a 220 mm bed (34 x 54.5 ft -> 130 x 208 mm)

# ---------------------------------------------------------------- heights (ft)
GRADE = -0.67           # yard grade below finished floor
F1_PLATE = 9.0          # 9 ft first floor ceilings (Included Features sheet)
F2_FLOOR = 10.0         # 9 ft plate + floor framing
F2_PLATE = 18.0         # 8 ft second floor ceilings
HEEL = 0.5              # truss heel above the plate
DOOR_HEAD = 6.67        # 6'8" doors
F1_SILL, F1_HEAD = 2.5, 8.0
F2_SILL, F2_HEAD = 2.25, 6.83
WALL_GROW = 0.05        # plan walls are drawn ~3.4"; grow to ~4.5" framed walls
ROOF_PITCH = 6 / 12
TOWER_PITCH = 9 / 12
OVERHANG = 1.0

# ---------------------------------------------------------------- materials
COLORS = {
    "wall": "#e9e3d7",          # interior paint
    "trim": "#e5decf",          # SW 7526 Maison Blanche
    "siding": "#a99d8b",        # SW 7038 Tony Taupe, fiber cement lap
    "shake": "#8f8576",         # upper siding behind the garage
    "brick": "#c9b392",         # Irish Cream brick, white mortar
    "stone": "#d8c6a3",         # Buckskin Cream Chopped stone
    "shutter": "#5f6870",       # SW 7019 Gauntlet Grey
    "front_door": "#4e4236",    # SW 3542 Charwood
    "garage_door": "#e2dccd",
    "shingle": "#6b625a",       # 3-tab composition shingles
    "gutter": "#e5decf",
    "glass": "#9fc3d6",
    "slab": "#bdb8ae",
    "concrete": "#c8c3b8",
    "garage_floor": "#a9a59c",
    "tile": "#b9aa94",          # Emser Albero Ramo 8x24 wood-look tile
    "carpet": "#cfc6b6",        # Shaw Rosemary Park, Oyster
    "vinyl": "#a39b91",         # Lifeproof Sterling Oak vinyl plank (2021)
    "bath_tile": "#b9aa94",
    "cabinet": "#7a5236",       # Timberlake Fairfield Nutmeg
    "granite": "#cbbda6",       # Caledonia granite
    "marble": "#f2efe8",        # cultured marble vanity tops
    "fixture": "#fbfbf8",
    "steel": "#b8bcbf",
    "appliance": "#d9dbdc",
    "furnace": "#9aa0a3",
    "stair": "#a39b91",
    "rail": "#3b2f27",          # Jacobean stain handrail
    "grass": "#7d9a5a",
    "road": "#6e6e6a",
    "fence": "#8a6a4a",
    "floor_plate": "#d8cfbf",
    "furniture_gray": "#8c8d8f",
    "furniture_charcoal": "#4f5153",
    "furniture_oak": "#a77b4f",
    "furniture_black": "#2c2c2c",
    "light": "#f7e9b0",
    "ac": "#c9cbc9",
}
_MATERIALS = {}


def material(name):
    if name not in _MATERIALS:
        hexc = COLORS[name].lstrip("#")
        rgba = [int(hexc[i:i + 2], 16) for i in (0, 2, 4)] + [255]
        if name == "glass":
            rgba[3] = 90
        _MATERIALS[name] = PBRMaterial(
            name=name, baseColorFactor=rgba, metallicFactor=0.0,
            roughnessFactor=0.85 if name not in ("glass", "steel", "appliance") else 0.3,
            alphaMode="BLEND" if name == "glass" else "OPAQUE", doubleSided=False)
    return _MATERIALS[name]


# ---------------------------------------------------------------- mesh helpers
def prism(poly, z0, z1):
    """Extrude a shapely polygon (or multipolygon) between two heights."""
    geoms = getattr(poly, "geoms", [poly])
    meshes = []
    for g in geoms:
        if g.is_empty or g.area < 1e-4:
            continue
        g = g.simplify(0.002)
        m = trimesh.creation.extrude_polygon(g, z1 - z0)
        m.apply_translation([0, 0, z0])
        meshes.append(m)
    return trimesh.util.concatenate(meshes) if meshes else None


def cuboid(x0, y0, z0, x1, y1, z1):
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    z0, z1 = sorted((z0, z1))
    m = trimesh.creation.box(extents=[x1 - x0, y1 - y0, z1 - z0])
    m.apply_translation([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
    return m


def cylinder(x, y, z0, z1, r, sections=24):
    m = trimesh.creation.cylinder(radius=r, height=z1 - z0, sections=sections)
    m.apply_translation([x, y, (z0 + z1) / 2])
    return m


def face_extrude(poly_xz, y_face, depth):
    """Extrude a polygon drawn in the facade plane (x, z) toward -Y."""
    m = prism(poly_xz, 0, depth)
    if m is None:
        return None
    v = m.vertices.copy()
    m.vertices = np.column_stack([v[:, 0], y_face - v[:, 2], v[:, 1]])  # a rotation: normals stay outward
    return m


def side_extrude(poly_yz, x_face, depth):
    """Extrude a polygon drawn in a side plane (y, z) toward +X."""
    m = prism(poly_yz, 0, depth)
    if m is None:
        return None
    v = m.vertices.copy()
    m.vertices = np.column_stack([x_face + v[:, 2], v[:, 0], v[:, 1]])
    return m


def arch_ring(xc, z_spring, span, rise, band):
    """2D arch band in the (x, z) plane: a circular segment ring."""
    half = span / 2
    r = (half ** 2 + rise ** 2) / (2 * rise)
    zc = z_spring + rise - r
    outer = Point(xc, zc).buffer(r + band, 96)
    inner = Point(xc, zc).buffer(r, 96)
    cut = box(xc - half - band * 1.5, z_spring, xc + half + band * 1.5, z_spring + rise + band + 0.01)
    return outer.difference(inner).intersection(cut)


def arch_opening(xc, z0, span, z_spring, rise):
    """2D opening: rectangle up to the springline topped by a segment."""
    half = span / 2
    r = (half ** 2 + rise ** 2) / (2 * rise)
    zc = z_spring + rise - r
    top = Point(xc, zc).buffer(r, 96).intersection(box(xc - half, z_spring, xc + half, z_spring + rise))
    return unary_union([box(xc - half, z0, xc + half, z_spring + 0.001), top])


def hip_roof(x0, y0, x1, y1, plate, pitch, overhang=OVERHANG):
    """Solid hip roof over a rectangle; ridge along the longer side."""
    z_wall = plate + HEEL
    z_eave = z_wall - overhang * pitch
    ex0, ey0, ex1, ey1 = x0 - overhang, y0 - overhang, x1 + overhang, y1 + overhang
    w, d = ex1 - ex0, ey1 - ey0
    run = min(w, d) / 2
    z_ridge = z_eave + run * pitch
    if w >= d:
        ridge = [(ex0 + run, (ey0 + ey1) / 2), (ex1 - run, (ey0 + ey1) / 2)]
    else:
        ridge = [((ex0 + ex1) / 2, ey0 + run), ((ex0 + ex1) / 2, ey1 - run)]
    pts = [(ex0, ey0, z_eave), (ex1, ey0, z_eave), (ex1, ey1, z_eave), (ex0, ey1, z_eave)]
    pts += [(x, y, z_ridge) for x, y in ridge]
    pts += [(x, y, z_eave - 0.35) for x, y, _ in pts[:4]]  # roof deck thickness
    return trimesh.convex.convex_hull(np.array(pts)), (ex0, ey0, ex1, ey1, z_eave)


def gable_roof_y(x0, x1, y0, y1, plate, pitch, overhang=OVERHANG, back_overhang=None):
    """Solid gable roof with the ridge running along Y (gable faces front)."""
    back_overhang = overhang if back_overhang is None else back_overhang
    z_wall = plate + HEEL
    z_eave = z_wall - overhang * pitch
    ex0, ex1 = x0 - overhang, x1 + overhang
    ey0, ey1 = y0 - overhang, y1 + back_overhang
    xm = (x0 + x1) / 2
    z_ridge = z_wall + (x1 - x0) / 2 * pitch
    t = 0.35
    # Two sloped slabs, so the open gable end shows the wall behind it.
    slabs = []
    for xe in (ex0, ex1):
        pts = []
        for y in (ey0, ey1):
            pts += [(xe, y, z_eave), (xm, y, z_ridge), (xe, y, z_eave - t), (xm, y, z_ridge - t)]
        slabs.append(trimesh.convex.convex_hull(np.array(pts)))
    roof = slabs
    gable = Polygon([(x0, z_wall - HEEL), (x1, z_wall - HEEL), (x1, z_wall), (xm, z_ridge - t), (x0, z_wall)])
    return roof, gable, (ex0, ey0, ex1, ey1, z_eave, z_ridge)


# ---------------------------------------------------------------- scene
class Model:
    def __init__(self):
        self.parts = []   # (layer, name, mesh, material)
        self.solids = {}  # id(mesh) -> the closed solids it was built from

    def add(self, layer, name, mesh, mat):
        solids = [m for m in (mesh if isinstance(mesh, list) else [mesh]) if m is not None]
        if not solids:
            return
        mesh = trimesh.util.concatenate(solids) if len(solids) > 1 else solids[0]
        self.solids[id(mesh)] = solids
        self.parts.append((layer, name, mesh, mat))


# ---------------------------------------------------------------- plan data
def load_plan():
    plan = json.loads((HERE / "plan.json").read_text())
    floors = {}
    for key, data in plan["floors"].items():
        walls = [shape(g) for g in data["walls"]]
        ops = find_openings(walls)
        out = outline(walls, ops)
        floors[int(key)] = {"walls": walls, "openings": ops, "outline": out, "lines": data["lines"]}
    return floors


def classify(floor, op, out):
    """Name what each opening is from where it sits in the plan."""
    x0, y0, x1, y1 = op["rect"]
    r = box(x0, y0, x1, y1)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    exterior = r.buffer(0.05).intersects(out.exterior)
    if floor == 1:
        if exterior and op["width"] > 12:
            return "garage_door"
        if exterior and abs(cy - 14.27) < 0.4 and 10.5 < cx < 15:
            return "front_door"
        if exterior and cy > 54 and 22.5 < cx < 26.5:
            return "back_door"
        if not exterior and abs(cx - 15.15) < 0.4 and 33 < cy < 38.5:
            return "stair_halfwall"
    if exterior:
        return "window"
    return "door"


# Openings the detector cannot see: the study's double doors sit on a
# 45-degree wall corner. Given as wall-end points (ft).
MANUAL_HEADERS = {1: [((7.08, 23.62), (10.38, 20.42))]}

# Invisible lines that split the open plan into named rooms (room sizes from
# the brochure: great room 14'1" x 15'7", dining 14'1" x 14'5", nook 8'3" x 7'8",
# game room 14'7" x 13'8").
ROOM_SPLITS = {
    1: [[(0.0, 38.9), (19.1, 38.9)], [(14.1, 38.9), (14.1, 54.5)], [(14.1, 46.8), (22.3, 46.8), (22.3, 54.5)],
        [(14.0, 38.45), (19.1, 38.45)], [(7.0, 23.62), (15.4, 23.62)], [(15.3, 20.0), (15.3, 23.7)]],
    2: [[(0.0, 24.45), (19.0, 24.45)]],
}

ROOMS = {
    1: [("Great Room", (6.5, 46), "tile"), ("Kitchen", (29, 46), "tile"),
        ("Nook", (19.5, 50), "tile"), ("Dining", (6.5, 31), "tile"),
        ("Bedroom 4", (28, 31), "carpet"), ("Bath 2", (29, 22.5), "bath_tile"),
        ("2-Car Garage", (24, 10), "garage_floor"), ("Study", (5, 16), "tile"),
        ("Foyer", (12.8, 18), "tile"), ("Pantry", (20.6, 38.3), "tile"),
        ("Bedroom 4 Closet", (20.6, 31), "carpet"), ("Linen", (20.6, 24.5), "tile"),
        ("Hall", (21, 21.5), "tile"),
        ("Stairs", (17, 31), "tile")],
    2: [("Bedroom 2", (7.5, 49), "vinyl"), ("Bath 3", (5.5, 40.5), "bath_tile"),
        ("Bedroom 3", (5.5, 30), "vinyl"), ("Game Room", (7, 16), "vinyl"),
        ("Owner's Suite", (26.5, 28), "vinyl"), ("Owner's Bath", (29.5, 41), "bath_tile"),
        ("Owner's Walk-in Closet", (27.5, 50.5), "vinyl"), ("Laundry", (21.75, 41), "bath_tile"),
        ("Bedroom 2 Closet", (18, 51), "vinyl"), ("Hall", (13, 40), "vinyl"),
        ("Mechanical (Furnace)", (17, 43.6), "vinyl"), ("Linen", (9.6, 36.6), "vinyl"),
        ("Bedroom 3 Closet", (5, 36.5), "vinyl")],
}

# Lot survey (2020): lot lines and flatwork, same frame as the house.
LOT = [(-6.5, -31.6), (28.18, -31.6), (31.3, -31.2), (34.3, -30.3), (37.5, -29.4),
       (40.5, -28.33), (40.5, 115.2), (-6.5, 115.2)]


# ---------------------------------------------------------------- builders
def build_floor(model, floors, n):
    fl = floors[n]
    layer = f"Floor {n}"
    out = fl["outline"]
    z0 = 0.0 if n == 1 else F2_FLOOR
    z_top = F1_PLATE if n == 1 else F2_PLATE
    sill, head = (F1_SILL, F1_HEAD) if n == 1 else (F2_SILL, F2_HEAD)
    walls = unary_union(fl["walls"]).buffer(WALL_GROW, join_style="mitre").intersection(out)
    # Openings stay open: subtract them back out after growing the walls.
    op_rects = [box(*o["rect"]).buffer(0.0) for o in fl["openings"]]
    grown_ops = []
    for o in fl["openings"]:
        x0, y0, x1, y1 = o["rect"]
        if o["axis"] == "h":
            grown_ops.append(box(x0, y0 - WALL_GROW, x1, y1 + WALL_GROW))
        else:
            grown_ops.append(box(x0 - WALL_GROW, y0, x1 + WALL_GROW, y1))
    walls = walls.difference(unary_union(grown_ops))
    model.add(layer, f"F{n} Walls", prism(walls, z0, z_top), "wall")

    stair_hole = box(15.2, 25.75, 18.95, 38.45)
    for o, g in zip(fl["openings"], grown_ops):
        kind = classify(n, o, out)
        o["kind"] = kind
        x0, y0, x1, y1 = g.bounds
        if kind == "garage_door":
            model.add(layer, "Garage Door Header", prism(g, z0 + 7.0, z_top), "wall")
            build_garage_door(model, o)
        elif kind == "stair_halfwall":
            model.add(layer, "Stair Half Wall", prism(g, z0, z0 + 3.0), "wall")
            model.add(layer, "Stair Half Wall Cap", cuboid(x0 - 0.06, y0, z0 + 3.0, x1 + 0.06, y1, z0 + 3.12), "rail")
        elif kind in ("door", "front_door", "back_door"):
            model.add(layer, "Door Headers", prism(g, z0 + DOOR_HEAD, z_top), "wall")
            if kind == "front_door":
                build_front_door(model, o)
            elif kind == "back_door":
                build_back_door(model, o)
        elif kind == "window":
            s = sill
            if n == 1 and o["rect"][1] > 54 and 26 < (x0 + x1) / 2 < 33:
                s = 3.6  # kitchen sink window over the counter
            model.add(layer, f"F{n} Window Walls", [prism(g, z0, z0 + s), prism(g, z0 + head, z_top)], "wall")
            build_window(model, layer, o, z0 + s, z0 + head)
    for a, b in MANUAL_HEADERS.get(n, []):
        hdr = LineString([a, b]).buffer(0.17, cap_style="flat")
        model.add(layer, "Door Headers", prism(hdr, z0 + DOOR_HEAD, z_top), "wall")

    # Floor structure and finishes.
    if n == 1:
        model.add(layer, "F1 Slab", prism(out, -0.67, -0.02), "slab")
    else:
        plate = out.difference(stair_hole)
        model.add(layer, "F2 Floor Framing", prism(plate, F1_PLATE, F2_FLOOR - 0.02), "floor_plate")
    splits = [LineString(l).buffer(0.01) for l in ROOM_SPLITS.get(n, [])]
    splits += [LineString([a, b]).buffer(0.17, cap_style="flat") for a, b in MANUAL_HEADERS.get(n, [])]
    holes = unary_union([unary_union(fl["walls"])] + op_rects + splits)
    spaces = out.difference(holes)
    if n == 2:
        spaces = spaces.difference(stair_hole)
    # Opening off hairline joins where a split line meets a wall corner.
    spaces = spaces.buffer(-0.05, join_style="mitre").buffer(0.05, join_style="mitre")
    pieces = list(getattr(spaces, "geoms", [spaces]))
    rooms = []
    for name, (px, py), finish in ROOMS[n]:
        hit = [p for p in pieces if p.contains(Point(px, py))]
        if not hit:
            print(f"  ! room {name} not found")
            continue
        poly = hit[0]
        pieces.remove(poly)
        rooms.append({"name": name, "floor": n, "finish": finish,
                      "label": [px, py], "area_sqft": round(poly.area, 1)})
        model.add(layer, f"Room: {name}", prism(poly, z0 - 0.02, z0), finish)
    for p in pieces:
        if p.area > 1.0:
            model.add(layer, "Room: other", prism(p, z0 - 0.02, z0), "tile" if n == 1 else "vinyl")
    return rooms


def build_window(model, layer, o, z_sill, z_head):
    x0, y0, x1, y1 = o["rect"]
    t = 0.12
    if o["axis"] == "h":  # wall runs along X
        ym = (y0 + y1) / 2
        model.add(layer, "Windows", cuboid(x0, ym - 0.02, z_sill, x1, ym + 0.02, z_head), "glass")
        frame = [cuboid(x0, ym - 0.1, z_sill, x0 + t, ym + 0.1, z_head),
                 cuboid(x1 - t, ym - 0.1, z_sill, x1, ym + 0.1, z_head),
                 cuboid(x0, ym - 0.1, z_sill, x1, ym + 0.1, z_sill + t),
                 cuboid(x0, ym - 0.1, z_head - t, x1, ym + 0.1, z_head),
                 cuboid(x0, ym - 0.08, (z_sill + z_head) / 2 - 0.05, x1, ym + 0.08, (z_sill + z_head) / 2 + 0.05)]
        for k in (1, 2):  # colonial grid
            xg = x0 + (x1 - x0) * k / 3
            frame.append(cuboid(xg - 0.03, ym - 0.05, z_sill, xg + 0.03, ym + 0.05, z_head))
        frame.append(cuboid(x0 - 0.1, y0 - 0.2, z_sill - 0.08, x1 + 0.1, y1 + 0.2, z_sill))
    else:
        xm = (x0 + x1) / 2
        model.add(layer, "Windows", cuboid(xm - 0.02, y0, z_sill, xm + 0.02, y1, z_head), "glass")
        frame = [cuboid(xm - 0.1, y0, z_sill, xm + 0.1, y0 + t, z_head),
                 cuboid(xm - 0.1, y1 - t, z_sill, xm + 0.1, y1, z_head),
                 cuboid(xm - 0.1, y0, z_sill, xm + 0.1, y1, z_sill + t),
                 cuboid(xm - 0.1, y0, z_head - t, xm + 0.1, y1, z_head),
                 cuboid(xm - 0.08, y0, (z_sill + z_head) / 2 - 0.05, xm + 0.08, y1, (z_sill + z_head) / 2 + 0.05)]
        for k in (1, 2):
            yg = y0 + (y1 - y0) * k / 3
            frame.append(cuboid(xm - 0.05, yg - 0.03, z_sill, xm + 0.05, yg + 0.03, z_head))
        frame.append(cuboid(x0 - 0.2, y0 - 0.1, z_sill - 0.08, x1 + 0.2, y1 + 0.1, z_sill))
    model.add(layer, "Window Frames", frame, "trim")
    o["sill"], o["head"] = z_sill, z_head


def build_garage_door(model, o):
    x0, y0, x1, y1 = o["rect"]
    yf = y0 + 0.1
    model.add("Floor 1", "Garage Door", cuboid(x0, yf, 0, x1, yf + 0.12, 7.0), "garage_door")
    panels = []
    cols, rows = 8, 4
    pw, ph = (x1 - x0) / cols, 7.0 / rows
    for i in range(cols):
        for j in range(rows):
            panels.append(cuboid(x0 + i * pw + 0.18, yf - 0.04, j * ph + 0.22,
                                 x0 + (i + 1) * pw - 0.18, yf, (j + 1) * ph - 0.22))
    model.add("Floor 1", "Garage Door Panels", panels, "garage_door")
    model.add("Floor 1", "Garage Door Trim", [cuboid(x0 - 0.35, yf - 0.1, 0, x0, yf + 0.1, 7.35),
                                              cuboid(x1, yf - 0.1, 0, x1 + 0.35, yf + 0.1, 7.35)], "trim")
    # Coach lights on either side (option 64526).
    for x in (x0 - 0.9, x1 + 0.9):
        model.add("Floor 1", "Coach Lights", [cuboid(x - 0.25, -0.45, 5.6, x + 0.25, -0.05, 6.6),
                                              cuboid(x - 0.18, -0.4, 5.75, x + 0.18, -0.1, 6.45)], "furniture_black")
        model.add("Floor 1", "Coach Light Glow", cuboid(x - 0.14, -0.38, 5.85, x + 0.14, -0.12, 6.35), "light")


def build_front_door(model, o):
    x0, y0, x1, y1 = o["rect"]
    ym = (y0 + y1) / 2
    model.add("Floor 1", "Front Door", cuboid(x0 + 0.05, ym - 0.08, 0, x1 - 0.05, ym + 0.08, DOOR_HEAD), "front_door")
    model.add("Floor 1", "Front Door Hardware", cuboid(x1 - 0.55, y0 - 0.12, 3.0, x1 - 0.45, ym, 3.5), "steel")
    model.add("Floor 1", "Front Door Trim", [cuboid(x0 - 0.3, y0 - 0.1, 0, x0, y0 + 0.05, DOOR_HEAD + 0.3),
                                             cuboid(x1, y0 - 0.1, 0, x1 + 0.3, y0 + 0.05, DOOR_HEAD + 0.3),
                                             cuboid(x0 - 0.3, y0 - 0.1, DOOR_HEAD, x1 + 0.3, y0 + 0.05, DOOR_HEAD + 0.3)], "trim")


def build_back_door(model, o):
    x0, y0, x1, y1 = o["rect"]
    ym = (y0 + y1) / 2
    model.add("Floor 1", "Back Door", [cuboid(x0, ym - 0.08, 0, x1, ym + 0.08, 0.6),
                                       cuboid(x0, ym - 0.08, DOOR_HEAD - 0.4, x1, ym + 0.08, DOOR_HEAD),
                                       cuboid(x0, ym - 0.08, 0, x0 + 0.35, ym + 0.08, DOOR_HEAD),
                                       cuboid(x1 - 0.35, ym - 0.08, 0, x1, ym + 0.08, DOOR_HEAD)], "trim")
    model.add("Floor 1", "Back Door Glass", cuboid(x0 + 0.35, ym - 0.02, 0.6, x1 - 0.35, ym + 0.02, DOOR_HEAD - 0.4), "glass")


def build_cladding(model, floors):
    """Exterior skin: brick and stone on the front, siding elsewhere."""
    skin_out, skin_in = 0.1, 0.02
    f2_out = floors[2]["outline"]
    for n in (1, 2):
        out = floors[n]["outline"]
        ops = floors[n]["openings"]
        z0 = GRADE if n == 1 else F2_FLOOR
        coords = list(out.exterior.coords)
        if out.exterior.is_ccw is False:
            coords = coords[::-1]
        for (ax, ay), (bx, by) in zip(coords, coords[1:]):
            length = math.hypot(bx - ax, by - ay)
            if length < 0.2:
                continue
            ux, uy = (bx - ax) / length, (by - ay) / length
            nx, ny = uy, -ux  # outward normal for a CCW ring
            mx, my = (ax + bx) / 2, (ay + by) / 2
            if n == 1:
                covered = f2_out.buffer(0.3).contains(Point(mx, my))
                z1 = F2_FLOOR if covered else F1_PLATE + HEEL
            else:
                z1 = F2_PLATE + HEEL
            front = ny < -0.9
            if n == 1:
                mat = "brick" if (front or max(ay, by) <= 14.6) else "siding"
            else:
                # Game room front is brick (the tower covers its middle);
                # the owner's suite wall above the garage roof is shake.
                mat = "brick" if (front and max(ay, by) < 11) else ("shake" if front else "siding")
            # Openings on this edge, as intervals along it.
            cuts = []
            for o in ops:
                x0, y0, x1, y1 = o["rect"]
                r = box(x0, y0, x1, y1)
                if r.buffer(0.08).intersects(LineString([(ax, ay), (bx, by)])) and o.get("kind") in (
                        "window", "front_door", "back_door", "garage_door"):
                    ts = sorted(((x - ax) * ux + (y - ay) * uy) for x, y in ((x0, y0), (x1, y1), (x0, y1), (x1, y0)))
                    lo_z, hi_z = (o.get("sill", 0.0), o.get("head", DOOR_HEAD))
                    if o["kind"] == "garage_door":
                        lo_z, hi_z = 0.0, 7.0
                    elif o["kind"] != "window":
                        lo_z, hi_z = 0.0, DOOR_HEAD
                    cuts.append((ts[0], ts[-1], lo_z, hi_z))
            # Tower and porch: the stone tower replaces the skin in front.
            spans = [(0.0, length, z0, z1)]
            pieces = []
            for t0, t1, zlo, zhi in sorted(cuts):
                new = []
                for s0, s1, za, zb in spans:
                    if t1 <= s0 or t0 >= s1:
                        new.append((s0, s1, za, zb))
                        continue
                    if t0 > s0:
                        new.append((s0, t0, za, zb))
                    if zlo > za:
                        pieces.append((t0, t1, za, zlo))
                    if zhi < zb:
                        pieces.append((t0, t1, zhi, zb))
                    if t1 < s1:
                        new.append((t1, s1, za, zb))
                spans = new
            pieces += spans
            meshes = []
            for s0, s1, za, zb in pieces:
                if s1 - s0 < 0.01 or zb - za < 0.01:
                    continue
                p0 = (ax + ux * s0, ay + uy * s0)
                p1 = (ax + ux * s1, ay + uy * s1)
                quad = Polygon([(p0[0] - nx * skin_in, p0[1] - ny * skin_in),
                                (p1[0] - nx * skin_in, p1[1] - ny * skin_in),
                                (p1[0] + nx * skin_out, p1[1] + ny * skin_out),
                                (p0[0] + nx * skin_out, p0[1] + ny * skin_out)])
                if n == 1 and front and 7.0 <= mx <= 15.0:
                    continue  # behind the stone tower face
                meshes.append(prism(quad, za, zb))
            model.add("Exterior", f"Cladding {mat}", meshes, mat)
        # Corner trim boards where siding meets siding (not on masonry corners).
        for x, y in coords[:-1]:
            if y < (14.6 if n == 1 else 19.0):
                continue
            model.add("Exterior", "Corner Trim", cylinder(x, y, z0, (F2_FLOOR if n == 1 else F2_PLATE + HEEL), 0.12, 8), "trim")


def build_tower(model):
    """Elevation R's two-story stone entry tower with the arched porch."""
    tx0, tx1 = 7.0, 15.0
    y_face, depth = 9.58, 0.5
    z_top = F2_PLATE + HEEL
    face = box(tx0, GRADE, tx1, z_top)
    entry = arch_opening(12.8, GRADE, 3.9, 7.4, 0.9)
    window = arch_opening(10.79, F2_FLOOR + F2_SILL, 2.98, F2_FLOOR + F2_HEAD - 0.5, 0.45)
    stone = face.difference(entry).difference(window)
    # Split at the second floor line so the floors can be pulled apart.
    model.add("Exterior", "Stone Tower", face_extrude(stone.intersection(box(0, -5, 40, F2_FLOOR)), y_face + depth, depth), "stone")
    model.add("Exterior", "Stone Tower Upper", face_extrude(stone.intersection(box(0, F2_FLOOR, 40, 40)), y_face + depth, depth), "stone")
    model.add("Exterior", "Stone Arches", face_extrude(arch_ring(12.8, 7.4, 3.9, 0.9, 0.55), y_face, 0.12), "stone")
    model.add("Exterior", "Stone Arches Upper", face_extrude(arch_ring(10.79, F2_FLOOR + F2_HEAD - 0.5, 2.98, 0.45, 0.45), y_face, 0.12), "stone")
    # Porch side returns in brick between the tower face and the recess.
    model.add("Exterior", "Porch Ceiling", cuboid(10.6, y_face, F1_PLATE - 0.1, 15.0, 14.2, F1_PLATE), "trim")
    # Tower gable and its roof.
    roof, gable, ext = gable_roof_y(tx0, tx1, y_face, 16.6, F2_PLATE, TOWER_PITCH, overhang=0.6, back_overhang=0.0)
    model.add("Roof", "Tower Gable Roof", roof, "shingle")
    model.add("Roof", "Tower Gable", face_extrude(gable, y_face + depth, depth), "stone")
    model.add("Roof", "Tower Rake Trim", [face_extrude(Polygon([(tx0 - 0.6, F2_PLATE + HEEL - 0.45), (11.0, ext[5] - 0.05),
                                                                 (tx1 + 0.6, F2_PLATE + HEEL - 0.45), (tx1 + 0.6, F2_PLATE + HEEL - 0.9),
                                                                 (11.0, ext[5] - 0.5), (tx0 - 0.6, F2_PLATE + HEEL - 0.9)]),
                                                        y_face - 0.6 + 0.01, 0.15)], "trim")
    # Shutters beside the tower window and the stacked left-block windows.
    sh = []
    for xc, zlo, zhi, half in ((10.79, F2_FLOOR + F2_SILL, F2_FLOOR + F2_HEAD - 0.5, 1.49),):
        for side in (-1, 1):
            xa = xc + side * (half + 0.15)
            sh.append(cuboid(xa, y_face - 0.12, zlo, xa + side * 1.3, y_face, zhi))
    model.add("Exterior", "Shutters", sh, "shutter")


def build_front_details(model, floors):
    sh = []
    # Study window (F1) and game room window above it, both with shutters.
    for (x0, x1, zlo, zhi, yf) in ((3.75, 6.74, F1_SILL, F1_HEAD, 10.08), (3.01, 5.99, F2_FLOOR + F2_SILL, F2_FLOOR + F2_HEAD, 10.06)):
        for side, xa in ((-1, x0 - 0.15), (1, x1 + 0.15)):
            sh.append(cuboid(xa, yf - 0.22, zlo, xa + side * 1.3, yf - 0.1, zhi))
    model.add("Exterior", "Shutters", sh, "shutter")
    # Brick soldier arches: over the study window and across the garage.
    model.add("Exterior", "Brick Arches", [face_extrude(arch_ring(5.245, F1_HEAD, 3.6, 0.55, 0.45), -0.0 + 10.08 - 0.1, 0.14),
                                           face_extrude(arch_ring(24.5, 7.45, 17.2, 1.0, 0.55), -0.1, 0.14)], "brick")
    # Rowlock sills under the brick-faced windows.
    model.add("Exterior", "Window Sills", [cuboid(3.6, 9.8, F1_SILL - 0.3, 6.9, 10.1, F1_SILL)], "brick")


def build_roofs(model, floors):
    # Main hip over the second floor (behind the garage line), the front hip
    # over the game room, and the garage gable facing the street.
    main, ext_a = hip_roof(0.0, 18.42, 34.0, 54.5, F2_PLATE, ROOF_PITCH)
    model.add("Roof", "Main Hip Roof", main, "shingle")
    front, ext_b = hip_roof(0.0, 10.06, 15.0, 34.0, F2_PLATE, ROOF_PITCH)
    model.add("Roof", "Front Hip Roof", front, "shingle")
    garage, gable, ext_g = gable_roof_y(15.0, 34.0, 0.0, 18.42, F1_PLATE, ROOF_PITCH, back_overhang=0.0)
    model.add("Roof", "Garage Gable Roof", garage, "shingle")
    model.add("Roof", "Garage Gable (brick)", face_extrude(gable, 0.0, 0.1), "brick")
    # Fascia and gutters (whole-house gutters, option 63813) along the eaves.
    gut = []
    for ex0, ey0, ex1, ey1, ze in (ext_a[:5], ext_b[:5]):
        for (a, b) in (((ex0, ey0), (ex1, ey0)), ((ex1, ey0), (ex1, ey1)), ((ex1, ey1), (ex0, ey1)), ((ex0, ey1), (ex0, ey0))):
            line = LineString([a, b])
            if ex0 < 1 and ey0 > 18 and a[1] == ey0 and b[1] == ey0:
                line = LineString([(15.0 + OVERHANG, ey0), (ex1, ey0)])  # front of main hip: only east of the front hip
            gut.append(prism(line.buffer(0.22, cap_style="flat"), ze - 0.45, ze))
    ex0, ey0, ex1, ey1, ze, zr = ext_g
    for x in (ex0, ex1):
        gut.append(prism(LineString([(x, ey0), (x, ey1)]).buffer(0.22, cap_style="flat"), ze - 0.45, ze))
    model.add("Roof", "Gutters", gut, "gutter")
    spouts = [(-0.4, 54.9), (34.4, 54.9), (-0.4, 10.5), (34.4, 0.4), (14.6, 0.4)]
    for x, y in spouts:
        top = F2_PLATE if (y > 20 or x < 1) else F1_PLATE
        model.add("Exterior", "Downspouts", cuboid(x - 0.12, y - 0.12, GRADE, x + 0.12, y + 0.12, min(top, F2_FLOOR)), "gutter")
        if top > F2_FLOOR:
            model.add("Exterior", "Downspouts Upper", cuboid(x - 0.12, y - 0.12, F2_FLOOR, x + 0.12, y + 0.12, top), "gutter")


def build_stairs(model):
    # 15 risers from the kitchen end (Y=38.45) up toward the front (Y=25.75).
    risers = 15
    rise = F2_FLOOR / risers
    y_top, y_bot = 25.75, 38.45
    run = (y_bot - y_top) / (risers - 1)
    steps = []
    for i in range(risers - 1):
        y1 = y_bot - i * run
        steps.append(cuboid(15.3, y1 - run, 0.0, 18.9, y1, (i + 1) * rise))
    model.add("Floor 1", "Stairs", steps, "stair")
    # The stairs arrive at the front end (Y=25.75); the open side and the far
    # end of the stairwell get a painted guard with a stained cap.
    guard = [(15.12, 25.75, 15.3, 38.55), (15.12, 38.37, 18.95, 38.55)]
    model.add("Floor 2", "Stair Guard", [cuboid(x0, y0, F2_FLOOR, x1, y1, F2_FLOOR + 2.9) for x0, y0, x1, y1 in guard], "trim")
    model.add("Floor 2", "Stair Guard Cap", [cuboid(x0 - 0.04, y0, F2_FLOOR + 2.9, x1 + 0.04, y1, F2_FLOOR + 3.05) for x0, y0, x1, y1 in guard], "rail")
    # Handrail along the open side.
    model.add("Floor 1", "Handrail", trimesh.creation.cylinder(
        radius=0.08, segment=[[15.25, y_bot, 3.0], [15.25, y_top, F2_FLOOR + 3.0]]), "rail")


def build_kitchen(model):
    L = "Floor 1"
    base = [box(26.0, 52.25, 34.0, 54.21),     # back wall run with the sink
            box(31.95, 39.2, 33.71, 52.25),    # right wall run with the range
            box(25.8, 37.2, 33.71, 39.2)]      # run along the pantry/bedroom wall
    model.add(L, "Base Cabinets", [prism(b, 0, 2.95) for b in base], "cabinet")
    top = unary_union(base).buffer(0.08, join_style="mitre").intersection(box(0, 0, 33.71, 54.21))
    model.add(L, "Granite Countertops", prism(top, 2.95, 3.1), "granite")
    island = box(25.35, 43.1, 28.1, 48.4)
    model.add(L, "Island", prism(island.buffer(-0.15, join_style="mitre"), 0, 2.95), "cabinet")
    model.add(L, "Island Granite", prism(island.buffer(0.1, join_style="mitre"), 2.95, 3.1), "granite")
    model.add(L, "Sink", cuboid(28.75, 52.6, 2.4, 31.25, 53.9, 3.12), "steel")
    model.add(L, "Gas Range", [cuboid(32.0, 44.5, 0, 33.71, 46.75, 3.05), cuboid(33.4, 44.5, 3.05, 33.71, 46.75, 3.7)], "appliance")
    model.add(L, "Microwave Hood", cuboid(32.4, 44.5, 5.0, 33.71, 46.75, 6.4), "appliance")
    model.add(L, "Refrigerator", cuboid(22.8, 37.25, 0, 25.75, 39.6, 5.9), "appliance")
    model.add(L, "Dishwasher Front", cuboid(26.4, 52.2, 0.3, 28.4, 52.26, 2.8), "steel")
    uppers = [box(31.8 + 0.6, 39.2, 33.71, 44.3), box(31.8 + 0.6, 47.0, 33.71, 52.0),
              box(26.0, 52.9, 28.3, 54.21), box(31.6, 52.9, 33.71, 54.21)]
    model.add(L, "Upper Cabinets (36 in)", [prism(u, 4.5, 7.5) for u in uppers], "cabinet")


def build_baths_and_utility(model):
    F1, F2 = "Floor 1", "Floor 2"
    z2 = F2_FLOOR
    # Bath 2 (first floor).
    model.add(F1, "Bath 2 Tub", cuboid(31.3, 20.3, 0, 33.7, 25.2, 1.5), "fixture")
    model.add(F1, "Bath 2 Vanity", cuboid(26.1, 20.3, 0, 28.8, 22.1, 2.75), "cabinet")
    model.add(F1, "Bath 2 Vanity Top", cuboid(26.0, 20.25, 2.75, 28.9, 22.2, 2.85), "marble")
    toilet(model, F1, 29.75, 20.6, 0, "y")
    # Bath 3 (second floor): tub, toilet, double vanity.
    model.add(F2, "Bath 3 Tub", cuboid(0.35, 38.3, z2, 2.75, 43.0, z2 + 1.5), "fixture")
    toilet(model, F2, 4.35, 38.4, z2, "y")
    model.add(F2, "Bath 3 Vanity", cuboid(5.6, 38.3, z2, 10.7, 40.1, z2 + 2.75), "cabinet")
    model.add(F2, "Bath 3 Vanity Top", cuboid(5.5, 38.25, z2 + 2.75, 10.75, 40.2, z2 + 2.85), "marble")
    # Owner's bath: drawer bank vanity on the outside wall, Shower #2 option
    # (shower where the linen closet was, oval drop-in tub in the tub bay).
    model.add(F2, "Owner's Vanity + Drawer Bank", cuboid(31.8, 37.1, z2, 33.7, 42.6, z2 + 2.75), "cabinet")
    model.add(F2, "Owner's Vanity Top", cuboid(31.7, 37.0, z2 + 2.75, 33.75, 42.7, z2 + 2.85), "marble")
    toilet(model, F2, 32.6, 43.3, z2, "x")
    model.add(F2, "Owner's Oval Tub Deck", cuboid(24.8, 37.25, z2, 27.6, 42.0, z2 + 1.75), "fixture")
    model.add(F2, "Owner's Shower Floor", cuboid(24.8, 42.35, z2, 27.9, 45.2, z2 + 0.25), "bath_tile")
    model.add(F2, "Owner's Shower Glass", [cuboid(27.85, 42.35, z2 + 0.25, 27.95, 45.2, z2 + 6.5)], "glass")
    model.add(F2, "Linen Wall Cabinet", cuboid(28.2, 45.35, z2 + 4.0, 31.5, 46.1, z2 + 7.0), "cabinet")
    # Laundry, furnace closet, water heater.
    model.add(F2, "Washer", cuboid(19.3, 42.7, z2, 21.7, 45.1, z2 + 3.2), "appliance")
    model.add(F2, "Dryer", cuboid(21.9, 42.7, z2, 24.3, 45.1, z2 + 3.2), "appliance")
    model.add(F2, "Furnaces", [cuboid(15.6, 43.3, z2, 18.3, 45.1, z2 + 4.8), cuboid(15.6, 42.0, z2, 18.3, 43.2, z2 + 4.8)], "furnace")
    model.add(F1, "Water Heater", cylinder(32.75, 18.9, 1.5, 6.5, 0.95), "appliance")
    model.add(F1, "Water Heater Platform", cuboid(31.6, 17.8, 0, 33.7, 20.0, 1.5), "concrete")


def toilet(model, layer, x, y, z, facing):
    if facing == "y":  # tank against a wall at lower Y
        parts = [cuboid(x - 0.75, y, z + 1.5, x + 0.75, y + 0.65, z + 2.6),
                 cuboid(x - 0.6, y + 0.4, z, x + 0.6, y + 2.3, z + 1.35)]
    else:  # tank against the wall at higher X
        parts = [cuboid(x + 0.45, y - 0.75, z + 1.5, x + 1.1, y + 0.75, z + 2.6),
                 cuboid(x - 1.4, y - 0.6, z, x + 0.7, y + 0.6, z + 1.35)]
    model.add(layer, "Toilets", parts, "fixture")


def build_furniture(model):
    """Furniture the purchase records name; placement is a guess."""
    F1, F2, z2 = "Furniture", "Furniture", F2_FLOOR
    # Great room: Rooms To Go Sandia Heights gray sofa + loveseat, Slater black tables.
    model.add(F1, "Sofa (Sandia Heights)", [cuboid(0.5, 39.5, 0, 3.6, 47.0, 1.5), cuboid(0.5, 39.5, 1.5, 1.3, 47.0, 3.0),
                                            cuboid(0.5, 39.5, 1.5, 3.6, 40.2, 2.1), cuboid(0.5, 46.3, 1.5, 3.6, 47.0, 2.1)], "furniture_gray")
    model.add(F1, "Loveseat", [cuboid(2.5, 50.5, 0, 8.0, 53.6, 1.5), cuboid(2.5, 52.8, 1.5, 8.0, 53.6, 3.0)], "furniture_gray")
    model.add(F1, "Cocktail Table (Slater)", cuboid(5.0, 42.0, 0, 7.5, 46.0, 1.5), "furniture_black")
    # Dining: San Luis oak 5-piece rectangle table.
    model.add(F1, "Dining Table (San Luis)", [cuboid(4.9, 27.4, 2.3, 8.3, 32.6, 2.5),
                                              cuboid(6.4, 28.0, 0, 6.8, 32.0, 2.3)], "furniture_oak")
    chairs = []
    for x, y in ((4.0, 28.6), (4.0, 30.6), (9.2, 28.6), (9.2, 30.6)):
        chairs += [cuboid(x - 0.75, y, 0, x + 0.75, y + 1.5, 1.5)]
        back = x - 0.75 if x < 6 else x + 0.6
        chairs += [cuboid(back, y, 1.5, back + 0.15, y + 1.5, 3.3)]
    model.add(F1, "Dining Chairs", chairs, "furniture_oak")
    # Game room: Living Spaces Kerri charcoal 2-piece sectional with chaise, Harland ottoman.
    model.add(F2, "Sectional (Kerri)", [cuboid(0.5, 11.0, z2, 3.6, 20.5, z2 + 1.5), cuboid(0.5, 11.0, z2 + 1.5, 1.3, 20.5, z2 + 3.0),
                                         cuboid(0.5, 20.5, z2, 8.0, 23.6, z2 + 1.5), cuboid(0.5, 22.8, z2 + 1.5, 8.0, 23.6, z2 + 3.0),
                                         cuboid(3.6, 11.0, z2, 6.2, 16.0, z2 + 1.5)], "furniture_charcoal")
    model.add(F2, "Ottoman (Harland)", cuboid(6.2, 15.5, z2, 8.7, 18.0, z2 + 1.4), "furniture_gray")


def build_site(model):
    L = "Site"
    lot = Polygon(LOT)
    house = box(0, 0, 34, 54.5)
    model.add(L, "Lawn", prism(lot, GRADE - 0.4, GRADE), "grass")
    model.add(L, "Driveway", prism(Polygon([(15.0, 0), (34.0, 0), (34.0, -31.0), (38.5, -37.6), (13.5, -37.6), (15.0, -31.0)]), GRADE, GRADE + 0.06), "concrete")
    walk = unary_union([box(10.8, 0.0, 14.9, 14.2),            # entry porch slab under the tower arch
                        box(11.2, -4.5, 14.6, 0.0),
                        Point(15.0, -4.5).buffer(3.4).difference(Point(15.0, -4.5).buffer(0.0001)).intersection(box(11.2, -8.0, 15.0, -4.5))])
    model.add(L, "Front Walk and Porch", prism(walk, GRADE, -0.05), "concrete")
    model.add(L, "Rear Patio (survey: covered conc. porch)", prism(box(20.1, 54.5, 34.0, 61.1), GRADE, -0.08), "concrete")
    model.add(L, "Public Sidewalk", prism(box(-6.5, -41.6, 40.5, -37.6), GRADE - 0.1, GRADE - 0.04), "concrete")
    model.add(L, "Street", prism(box(-30, -75.0, 64.5, -41.6), GRADE - 0.6, GRADE - 0.25), "road")
    model.add(L, "AC Condenser", [cuboid(35.0, 16.3, GRADE, 37.6, 18.9, GRADE + 0.25), cuboid(35.2, 16.5, GRADE + 0.25, 37.4, 18.7, GRADE + 3.0)], "ac")
    # Stained wood fence: back lot line, both sides, returning to the house.
    fence_lines = [[(-6.5, 44.0), (-6.5, 115.2), (40.5, 115.2), (40.5, 44.0)], [(-6.5, 44.0), (0.0, 44.0)], [(34.0, 44.0), (40.5, 44.0)]]
    fence = [prism(LineString(f).buffer(0.08, cap_style="flat", join_style="mitre"), GRADE, GRADE + 6.0) for f in fence_lines]
    model.add(L, "Wood Fence (approx.)", fence, "fence")


# ---------------------------------------------------------------- outputs
def export_glb(model, path):
    scene = trimesh.Scene()
    flip = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)
    counts = {}
    for layer, name, mesh, mat in model.parts:
        m = mesh.copy()
        m.apply_scale(FT)
        m.apply_transform(flip)
        m.visual = trimesh.visual.TextureVisuals(material=material(mat))
        key = f"{layer}|{name}"
        counts[key] = counts.get(key, 0) + 1
        node = key if counts[key] == 1 else f"{key}|{counts[key]}"
        scene.add_geometry(m, node_name=node, geom_name=node)
    scene.export(path)


PRINT_PIECES = {
    "1_first_floor": ["Floor 1"],
    "2_second_floor": ["Floor 2"],
    "3_roof_main": ["Main Hip Roof", "Front Hip Roof", "Tower Gable Roof", "Tower Gable"],
    "4_roof_garage": ["Garage Gable Roof", "Garage Gable (brick)"],
}
PRINT_SKIP = ("Windows", "Glass", "Room:", "Light Glow", "Coach", "Handrail", "Hardware", "Window Frames", "Sills",
              "Shutters", "Trim", "Toilets", "Tub", "Vanity", "Washer", "Dryer", "Furnace", "Water Heater",
              "Refrigerator", "Range", "Microwave", "Sink", "Dishwasher", "Shower", "Linen Wall", "Garage Door Panels")


def export_print(model, floors, folder):
    import manifold3d as mf
    folder.mkdir(exist_ok=True)
    s = FT * 1000 / PRINT_SCALE
    report = {}
    for piece, keys in PRINT_PIECES.items():
        meshes = []
        for layer, name, mesh, mat in model.parts:
            if layer in keys or name in keys:
                if any(k in name for k in PRINT_SKIP) and name not in keys:
                    continue
                meshes.append(mesh)
        if piece == "1_first_floor":
            # Brick/siding skin belongs to the walls it covers.
            meshes += [m for l, nm, m, _ in model.parts if l == "Exterior" and nm.startswith("Cladding") and m.bounds[0][2] < F1_PLATE]
            meshes += [m for l, nm, m, _ in model.parts if nm in ("Stone Tower",)]
        if piece == "2_second_floor":
            meshes += [m for l, nm, m, _ in model.parts if l == "Exterior" and nm.startswith("Cladding") and m.bounds[0][2] >= F2_FLOOR - 0.01]
        result = None
        skipped = 0
        for m in meshes:
            # Union the closed solids each part was built from.
            for p in model.solids.get(id(m), [m]):
                mesh = mf.Mesh(vert_properties=np.asarray(p.vertices, dtype=np.float32),
                               tri_verts=np.asarray(p.faces, dtype=np.uint32))
                mesh.merge()
                man = mf.Manifold(mesh)
                if man.status() != mf.Error.NoError:
                    skipped += 1
                    continue
                if man.is_empty():
                    continue
                result = man if result is None else result + man
        if result is None:
            continue
        if piece == "1_first_floor":
            # Cut the tower down to the first-floor top so floor 2 sits flat.
            result = result ^ mf.Manifold.cube([200, 200, F2_FLOOR + 1.67]).translate([-50, -50, -1.67])
        if piece == "2_second_floor":
            result = result ^ mf.Manifold.cube([200, 200, 30]).translate([-50, -50, F1_PLATE])
        out = result.to_mesh()
        tm = trimesh.Trimesh(vertices=out.vert_properties[:, :3], faces=out.tri_verts)
        tm.apply_translation([0, 0, -tm.bounds[0][2]])
        tm.apply_scale(s)
        tm.export(folder / f"{piece}.stl")
        report[piece] = [round(v, 1) for v in tm.extents] + [skipped]
    return report


def plan_lines_for_viewer(floors):
    """Plan linework (fixtures, door swings, stair treads) as flat arrays."""
    out = {}
    for n, fl in floors.items():
        segs = []
        for line in fl["lines"]:
            for (ax, ay), (bx, by) in zip(line, line[1:]):
                segs.extend([round(ax, 3), round(ay, 3), round(bx, 3), round(by, 3)])
        out[n] = segs
    return out


def main():
    floors = load_plan()
    model = Model()
    rooms = []
    for n in (1, 2):
        rooms += build_floor(model, floors, n)
    build_cladding(model, floors)
    build_tower(model)
    build_front_details(model, floors)
    build_roofs(model, floors)
    build_stairs(model)
    build_kitchen(model)
    build_baths_and_utility(model)
    build_furniture(model)
    build_site(model)

    glb = HERE / "house.glb"
    export_glb(model, glb)
    meta = {
        "rooms": rooms,
        "levels": {"F1": 0.0, "F2": F2_FLOOR, "F1_plate": F1_PLATE, "F2_plate": F2_PLATE},
        "openings": {n: [{k: v for k, v in o.items()} for o in fl["openings"]] for n, fl in floors.items()},
        "lines": plan_lines_for_viewer(floors),
    }
    template = (HERE / "viewer_template.html").read_text()
    html = (template.replace("__GLB_BASE64__", base64.b64encode(glb.read_bytes()).decode())
                    .replace("__META_JSON__", json.dumps(meta)))
    (HERE / "house_viewer.html").write_text(html)
    report = export_print(model, floors, HERE / "print")
    print(f"house.glb {glb.stat().st_size / 1e6:.1f} MB, {len(model.parts)} parts")
    for r in rooms:
        print(f"  F{r['floor']} {r['name']:<24} {r['area_sqft']:>7} sq ft  {r['finish']}")
    for k, v in report.items():
        print(f"  print/{k}.stl  {v[0]} x {v[1]} x {v[2]} mm at 1:{PRINT_SCALE}" + (f"  ({v[3]} non-solid parts skipped)" if v[3] else ""))


if __name__ == "__main__":
    main()
