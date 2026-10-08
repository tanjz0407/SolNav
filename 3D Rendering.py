"""
3D Rendering.py
================
Reconstructed pipeline for solnav_3d_verified_map.html.

This replaces manual HTML editing going forward. Buildings/landmarks are
loaded from `buildings_data.json.gz` / `landmarks_data.json.gz` -- these
already contain every manual height correction you made by hand, extracted
directly from your edited HTML, so nothing is lost. Trees / roads / street
lamps / forest / water are rebuilt fresh each run from the OSM CSV exports,
so if you get updated CSVs you just re-run this script.

Directory layout expected:
    ./data/buildings_data.json.gz
    ./data/landmarks_data.json.gz
    ./data/view_state.json
    ./data/Tree_KL_OSM.csv
    ./data/Road_KL_OSM.csv
    ./data/RoadLamp_KL_OSM.csv
    ./data/Forest_KL_OSM.csv

Output:
    ./solnav_3d_verified_map.html
"""

import gzip
import json
import random
import re
from pathlib import Path

import pandas as pd
import pydeck as pdk

# Resolve paths relative to this script's own location, so it doesn't matter
# which folder you launch `python` from.
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"   # expects: SolNav/data/  (sibling of scripts/)
OUTPUT_HTML = SCRIPT_DIR.parent / "solnav_3d_verified_map.html"

# ---------------------------------------------------------------------------
# AREA OF INTEREST -- the single biggest lever for GPU load. Whole-KL scope
# (139k buildings, 28k+ roads, 26k trees) is simply more scene complexity
# than an integrated GPU can carry smoothly, regardless of vertex-level
# trimming. Set AOI_BBOX to (lon_min, lat_min, lon_max, lat_max) to render
# only that area, or None to render the whole city (heavy).
#
# Default below covers KLCC -> TRX -> Merdeka 118 (where this project
# started) -- swap in your own box as needed. Google Maps: right-click a
# point -> the lat/lon shown is what you want for two opposite corners.
# ---------------------------------------------------------------------------
AOI_BBOX = (101.698, 3.138, 101.723, 3.161)  # (lon_min, lat_min, lon_max, lat_max)
# AOI_BBOX = None  # uncomment for whole-city (only try this on a discrete GPU)


def _in_bbox(lon, lat, bbox):
    if bbox is None:
        return True
    lon_min, lat_min, lon_max, lat_max = bbox
    return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max


def _bbox_intersects(coords, bbox):
    """True if ANY point of a path/polygon falls inside bbox (keeps features
    that cross the boundary rather than chopping them, simpler + usually
    good enough at this scale)."""
    if bbox is None:
        return True
    return any(_in_bbox(c[0], c[1], bbox) for c in coords)


def clip_polygon_to_bbox(poly, bbox):
    """Sutherland-Hodgman polygon clipping against an axis-aligned bbox.

    Needed for forest/water specifically -- OSM natural=* polygons (parks,
    reservoirs) can span many kilometers. Just checking "does any point fall
    in the AOI" and then keeping the WHOLE polygon (as buildings/roads do)
    means one corner of a huge lake clipping into a small AOI renders that
    entire multi-km shape at full size, dwarfing everything else in frame.
    Actually clipping the geometry to the box fixes this properly.
    """
    if bbox is None or not poly:
        return poly
    lon_min, lat_min, lon_max, lat_max = bbox

    def clip_edge(points, inside_fn, intersect_fn):
        if not points:
            return points
        out = []
        prev = points[-1]
        prev_in = inside_fn(prev)
        for cur in points:
            cur_in = inside_fn(cur)
            if cur_in:
                if not prev_in:
                    out.append(intersect_fn(prev, cur))
                out.append(cur)
            elif prev_in:
                out.append(intersect_fn(prev, cur))
            prev, prev_in = cur, cur_in
        return out

    def lerp(a, b, t):
        return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]

    pts = poly
    # left
    pts = clip_edge(pts, lambda p: p[0] >= lon_min,
                     lambda a, b: lerp(a, b, (lon_min - a[0]) / (b[0] - a[0])) if b[0] != a[0] else a)
    # right
    pts = clip_edge(pts, lambda p: p[0] <= lon_max,
                     lambda a, b: lerp(a, b, (lon_max - a[0]) / (b[0] - a[0])) if b[0] != a[0] else a)
    # bottom
    pts = clip_edge(pts, lambda p: p[1] >= lat_min,
                     lambda a, b: lerp(a, b, (lat_min - a[1]) / (b[1] - a[1])) if b[1] != a[1] else a)
    # top
    pts = clip_edge(pts, lambda p: p[1] <= lat_max,
                     lambda a, b: lerp(a, b, (lat_max - a[1]) / (b[1] - a[1])) if b[1] != a[1] else a)
    return pts

# ---------------------------------------------------------------------------
# WKT parsing helpers (no shapely dependency needed)
# ---------------------------------------------------------------------------

_num = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
_pair = re.compile(rf'({_num})\s+({_num})')

# 6 decimal places on a lon/lat coordinate is ~11cm precision -- far finer
# than needed for a map at this scale, and cuts serialized size substantially
# versus the raw ~15-digit floats WKT exports contain.
_COORD_PRECISION = 6


def _parse_ring(ring_text):
    return [[round(float(x), _COORD_PRECISION), round(float(y), _COORD_PRECISION)]
            for x, y in _pair.findall(ring_text)]


def parse_wkt_point(wkt):
    m = _pair.search(wkt)
    return [round(float(m.group(1)), _COORD_PRECISION),
            round(float(m.group(2)), _COORD_PRECISION)] if m else None


def parse_wkt_linestring(wkt):
    return _parse_ring(wkt)


def parse_wkt_multilinestring(wkt):
    parts = re.findall(r'\(([^()]+)\)', wkt)
    return [_parse_ring(p) for p in parts if p.strip()]


def parse_wkt_polygon(wkt):
    rings = re.findall(r'\(([^()]+)\)', wkt)
    return _parse_ring(rings[0]) if rings else None


# ---------------------------------------------------------------------------
# Geometry simplification (Douglas-Peucker) -- OSM footprints/paths often
# carry far more vertices than needed to look correct at map scale. Millions
# of extra vertices across 139k buildings + 76k roads is enough to exhaust
# GPU buffer memory and crash the WebGL context (blank page, "context lost"
# in devtools) on anything but a high-end GPU. Simplifying trims the vertex
# count substantially with visually negligible shape change.
# ---------------------------------------------------------------------------

def _perp_distance(pt, start, end):
    x, y = pt
    x1, y1 = start
    x2, y2 = end
    if x1 == x2 and y1 == y2:
        return ((x - x1) ** 2 + (y - y1) ** 2) ** 0.5
    num = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    den = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
    return num / den


def simplify_rdp(points, tolerance):
    """Douglas-Peucker simplification. `points` is a list of [x, y] (extra
    trailing values like z are preserved on kept points, ignored in the
    distance calc). `tolerance` is in the same units as the coordinates
    (degrees here -- ~0.00003 deg is roughly 3m at KL's latitude)."""
    if len(points) < 3:
        return points

    def rdp(pts):
        if len(pts) < 3:
            return pts
        start, end = pts[0][:2], pts[-1][:2]
        max_dist = -1.0
        index = -1
        for i in range(1, len(pts) - 1):
            d = _perp_distance(pts[i][:2], start, end)
            if d > max_dist:
                max_dist = d
                index = i
        if max_dist > tolerance:
            left = rdp(pts[: index + 1])
            right = rdp(pts[index:])
            return left[:-1] + right
        return [pts[0], pts[-1]]

    return rdp(points)


BUILDING_SIMPLIFY_TOLERANCE_DEG = 0.00003   # ~3m at this latitude
ROAD_SIMPLIFY_TOLERANCE_DEG = 0.00004       # ~4m
FOREST_SIMPLIFY_TOLERANCE_DEG = 0.00006     # ~6m -- forest/water edges can take more simplification

# Minor road classes to skip entirely -- driveways, back-alleys, footpaths,
# steps, etc. These made up 62% of the raw road count (mostly `service`
# roads, i.e. driveways/parking access) while contributing very little
# visually at city scale. Excluding them is the single biggest lever for
# cutting GPU load, far more effective than vertex-level simplification.
# Comment out any class here you specifically want kept (e.g. re-add
# "footway" if you want pedestrian bridges/walkways to show).
ROAD_TYPES_TO_SKIP = {
    "service", "footway", "path", "steps", "pedestrian", "cycleway",
    "corridor", "track", "construction", "platform", "elevator",
    "proposed", "bus_stop", "rest_area", "bridleway", "services",
}


def parse_wkt_multipolygon(wkt):
    body = wkt[wkt.find('(') + 1: wkt.rfind(')')]
    polygons = re.split(r'\)\)\s*,\s*\(\(', body)
    outers = []
    for poly in polygons:
        poly = poly.strip('()')
        rings = poly.split('),(')
        outers.append(_parse_ring(rings[0]))
    return outers


# ---------------------------------------------------------------------------
# 1. BUILDINGS + LANDMARKS -- loaded verbatim (manual edits preserved)
# ---------------------------------------------------------------------------

def load_building_layers():
    with gzip.open(f"{DATA_DIR}/buildings_data.json.gz", "rt", encoding="utf-8") as f:
        buildings = json.load(f)
    with gzip.open(f"{DATA_DIR}/landmarks_data.json.gz", "rt", encoding="utf-8") as f:
        landmarks = json.load(f)

    before_n = len(buildings)
    buildings = [b for b in buildings if _bbox_intersects(b["polygon_coordinates"], AOI_BBOX)]
    print(f"Buildings: {before_n:,} -> {len(buildings):,} after AOI filter")

    before = sum(len(b["polygon_coordinates"]) for b in buildings)
    for b in buildings:
        pts = b["polygon_coordinates"]
        if len(pts) > 4:  # don't bother simplifying simple rectangles
            b["polygon_coordinates"] = simplify_rdp(pts, BUILDING_SIMPLIFY_TOLERANCE_DEG)
    after = sum(len(b["polygon_coordinates"]) for b in buildings)
    print(f"Building vertices simplified: {before:,} -> {after:,} ({100*(1-after/before):.0f}% reduction)")

    buildings_layer = pdk.Layer(
        "PolygonLayer",
        data=buildings,
        get_polygon="polygon_coordinates",
        get_elevation="height_m",
        get_fill_color="color",
        get_line_color=[255, 255, 255, 200],
        extruded=True,
        pickable=True,
    )
    landmarks_layer = pdk.Layer(
        "PolygonLayer",
        data=landmarks,
        get_polygon="polygon_coordinates",
        get_elevation="height_m",
        get_fill_color="color",
        get_line_color=[255, 255, 255, 200],
        extruded=True,
        pickable=True,
    )
    return buildings_layer, landmarks_layer


# ---------------------------------------------------------------------------
# 2. TREES -- trunk + canopy (two layers), instead of a single plain cylinder.
#    A ColumnLayer alone just looks like a pole. Pairing a thin brown trunk
#    column with a wider, billboarded green "canopy" disc on top gives a much
#    more tree-like silhouette from any camera angle, with no external mesh
#    or image assets needed.
# ---------------------------------------------------------------------------

TRUNK_FRACTION = 0.35   # trunk is roughly the bottom third of total tree height
TRUNK_RADIUS_M = 0.25
CANOPY_RADIUS_SCALE = 0.9  # canopy width relative to total tree height


def build_tree_layers(csv_path=f"{DATA_DIR}/Tree_KL_OSM.csv", seed=42):
    df = pd.read_csv(csv_path, encoding="latin1")
    df = df[df["natural"] == "tree"].copy()

    coords = df["WKT"].apply(parse_wkt_point)
    df["lon"] = coords.apply(lambda c: c[0] if c else None)
    df["lat"] = coords.apply(lambda c: c[1] if c else None)
    df = df.dropna(subset=["lon", "lat"])

    before_n = len(df)
    if AOI_BBOX is not None:
        df = df[df.apply(lambda r: _in_bbox(r["lon"], r["lat"], AOI_BBOX), axis=1)]
    print(f"Trees: {before_n:,} -> {len(df):,} after AOI filter")

    rng = random.Random(seed)

    def resolve_height(v):
        try:
            h = float(v)
            if h > 0:
                return h
        except (TypeError, ValueError):
            pass
        return round(rng.uniform(2, 6), 1)

    df["tree_height"] = df["height"].apply(resolve_height)
    df["trunk_height"] = (df["tree_height"] * TRUNK_FRACTION).round(2)
    # canopy sits centered a bit above the trunk top, scaled with overall size
    df["canopy_elevation"] = (df["trunk_height"] + (df["tree_height"] - df["trunk_height"]) * 0.5).round(2)
    df["canopy_radius"] = (df["tree_height"] * CANOPY_RADIUS_SCALE * 0.35).clip(lower=0.8).round(2)

    trim = df[["lon", "lat", "trunk_height", "canopy_elevation", "canopy_radius"]].copy()
    trim["lon"] = trim["lon"].round(6)
    trim["lat"] = trim["lat"].round(6)

    trunk_layer = pdk.Layer(
        "ColumnLayer",
        data=trim,
        get_position=["lon", "lat"],
        get_elevation="trunk_height",
        elevation_scale=1,
        radius=TRUNK_RADIUS_M,
        get_fill_color=[92, 64, 42, 255],   # bark brown
        pickable=False,  # skip picking pass -- cheap decoration, not something you need to click
        extruded=True,
    )
    # Canopy: a filled circle positioned at 3D height via get_position's z
    # component. ScatterplotLayer draws flat in the XY-plane at that height
    # (it does not billboard toward the camera -- that prop only exists on
    # IconLayer/TextLayer), which still reads as foliage mass from typical
    # pitched map angles without needing a real 3D sphere mesh.
    canopy_layer = pdk.Layer(
        "ScatterplotLayer",
        data=trim,
        get_position=["lon", "lat", "canopy_elevation"],
        get_radius="canopy_radius",
        radius_units="meters",
        get_fill_color=[46, 125, 50, 235],  # foliage green
        stroked=False,
        pickable=False,
    )
    return trunk_layer, canopy_layer


# ---------------------------------------------------------------------------
# 3. ROADS -- PathLayer, bridges lifted, tunnels lowered.
#
# Bridge height methodology (no per-bridge survey data exists publicly at
# this scale, so this is the most defensible estimate available):
#   - Where the OSM `maxheight` tag is present (real, surveyed vehicle
#     clearance restriction), use it directly + typical deck structure
#     thickness (~1.3m for a standard concrete girder deck).
#   - Otherwise, estimate clearance per stacked `layer` tier using published
#     vertical clearance standards for grade-separated crossings (~5.5m for
#     expressway-class crossings, ~5.0m for lower classes -- matches the
#     5.0-5.5m clustering actually observed in this dataset's real maxheight
#     values) + the same deck thickness, multiplied by the layer count for
#     multi-level interchanges.
# This is still an estimate, not measured data -- flag any bridge you
# specifically care about and it's worth looking up its real height instead.
# ---------------------------------------------------------------------------

DECK_THICKNESS_M = 1.3
CLEARANCE_MAJOR_M = 5.5   # motorway/trunk-class crossings
CLEARANCE_MINOR_M = 5.0   # everything else
TUNNEL_DEPTH_M = -6.0

MAJOR_HIGHWAY_CLASSES = {"motorway", "motorway_link", "trunk", "trunk_link"}


def _bridge_height(row):
    highway = str(row.get("highway", "")).lower()
    clearance = CLEARANCE_MAJOR_M if highway in MAJOR_HIGHWAY_CLASSES else CLEARANCE_MINOR_M
    level_height = clearance + DECK_THICKNESS_M

    maxheight = row.get("maxheight")
    try:
        mh = float(maxheight)
        if mh > 0:
            return round(mh + DECK_THICKNESS_M, 2)
    except (TypeError, ValueError):
        pass

    try:
        layer = float(row.get("layer"))
        layer = max(1, layer)
    except (TypeError, ValueError):
        layer = 1.0
    return round(level_height * layer, 2)


def _road_color(h):
    h = str(h).lower()
    if h in ("motorway", "motorway_link", "trunk", "trunk_link"):
        return [80, 80, 85, 255]     # dark asphalt for major highways
    if h in ("primary", "primary_link", "secondary", "secondary_link"):
        return [95, 95, 100, 255]
    if h in ("footway", "path", "steps", "pedestrian", "cycleway"):
        return [180, 170, 150, 220]  # pavement/paved-path tone
    return [110, 110, 112, 255]


LANE_WIDTH_M = 3.25          # standard lane width
MIN_ROAD_WIDTH_M = 3.0        # single-lane/footway minimum
FOOTWAY_WIDTH_M = 2.0


def _road_width(row):
    h = str(row.get("highway", "")).lower()
    if h in ("footway", "path", "steps", "pedestrian", "cycleway"):
        return FOOTWAY_WIDTH_M
    try:
        lanes = float(row.get("lanes"))
        if lanes > 0:
            return round(lanes * LANE_WIDTH_M, 1)
    except (TypeError, ValueError):
        pass
    # no lane count tagged -- fall back on a class-based guess
    if h.startswith("motorway"):
        return round(3 * LANE_WIDTH_M, 1)
    if h in ("trunk", "primary"):
        return round(2 * LANE_WIDTH_M, 1)
    return MIN_ROAD_WIDTH_M


def build_road_layers(csv_path=f"{DATA_DIR}/Road_KL_OSM.csv"):
    df = pd.read_csv(csv_path, encoding="latin1", low_memory=False)
    df = df.dropna(subset=["highway"])
    before_count = len(df)
    df = df[~df["highway"].isin(ROAD_TYPES_TO_SKIP)]
    print(f"Road features: {before_count:,} -> {len(df):,} after dropping minor road classes")

    def to_path(row):
        wkt = row["WKT"].strip().strip('"')
        if wkt.startswith("MULTILINESTRING"):
            rings = parse_wkt_multilinestring(wkt)
        elif wkt.startswith("LINESTRING"):
            rings = [parse_wkt_linestring(wkt)]
        else:
            return None
        if not rings:
            return None

        z = 0.0
        bridge = str(row.get("bridge", "")).lower()
        tunnel = str(row.get("tunnel", "")).lower()
        if bridge in ("yes", "viaduct", "boardwalk"):
            z = _bridge_height(row)
        elif tunnel in ("yes", "building_passage", "covered", "passage"):
            z = TUNNEL_DEPTH_M

        ring = max(rings, key=len)
        if len(ring) > 3:
            ring = simplify_rdp(ring, ROAD_SIMPLIFY_TOLERANCE_DEG)
        return [[round(lon, 6), round(lat, 6), z] for lon, lat in ring]

    df["path"] = df.apply(to_path, axis=1)
    df = df.dropna(subset=["path"])
    df = df[df["path"].apply(len) >= 2]

    before_n = len(df)
    df = df[df["path"].apply(lambda p: _bbox_intersects(p, AOI_BBOX))]
    print(f"Roads (after class filter): {before_n:,} -> {len(df):,} after AOI filter")

    df["road_color"] = df["highway"].apply(_road_color)
    df["road_width"] = df.apply(_road_width, axis=1)
    df["is_bridge"] = df["bridge"].astype(str).str.lower().isin(["yes", "viaduct", "boardwalk"])

    trim = df[["path", "road_color", "road_width", "is_bridge"]].copy()

    road_layer = pdk.Layer(
        "PathLayer",
        data=trim,
        get_path="path",
        get_width="road_width",
        get_color="road_color",
        width_units="meters",
        width_min_pixels=2,
        pickable=True,
    )

    pier_layer = build_bridge_piers(trim[trim["is_bridge"]])
    return road_layer, pier_layer


# ---------------------------------------------------------------------------
# 3b. BRIDGE PIER COLUMNS -- support columns sampled along each elevated
#     road path at regular intervals, from ground level up to the deck.
#     This is what actually makes an elevated road read as a *bridge*
#     (deck + supports) rather than just a line floating in the air.
# ---------------------------------------------------------------------------

PIER_SPACING_M = 45.0        # typical span between highway bridge piers
PIER_RADIUS_M = 0.9
PIER_DEG_TO_M = 111_320       # rough meters per degree latitude, for spacing math


def _resample_pier_points(path, spacing_m):
    """Walk a bridge path and drop a pier point roughly every `spacing_m`."""
    if len(path) < 2:
        return []
    points = []
    accumulated = 0.0
    prev = path[0]
    points.append(prev)
    for cur in path[1:]:
        dx = (cur[0] - prev[0]) * PIER_DEG_TO_M
        dy = (cur[1] - prev[1]) * PIER_DEG_TO_M
        seg_len = (dx ** 2 + dy ** 2) ** 0.5
        accumulated += seg_len
        if accumulated >= spacing_m:
            points.append(cur)
            accumulated = 0.0
        prev = cur
    return points


def build_bridge_piers(bridge_df):
    records = []
    for row in bridge_df.itertuples():
        deck_height = row.path[0][2]
        for lon, lat, _ in _resample_pier_points(row.path, PIER_SPACING_M):
            records.append({"lon": lon, "lat": lat, "pier_height": deck_height})

    pier_df = pd.DataFrame.from_records(records)
    return pdk.Layer(
        "ColumnLayer",
        data=pier_df,
        get_position=["lon", "lat"],
        get_elevation="pier_height",
        elevation_scale=1,
        radius=PIER_RADIUS_M,
        get_fill_color=[150, 150, 150, 255],  # concrete grey
        pickable=False,
        extruded=True,
    )


# ---------------------------------------------------------------------------
# 4. STREET LAMPS -- real highway=street_lamp rows only, small glowing poles
# ---------------------------------------------------------------------------

def build_lamp_layer(csv_path=f"{DATA_DIR}/RoadLamp_KL_OSM.csv", lamp_height=6.0):
    df = pd.read_csv(csv_path, encoding="latin1", low_memory=False)
    df = df[df["highway"] == "street_lamp"].copy()

    coords = df["WKT"].apply(parse_wkt_point)
    df["lon"] = coords.apply(lambda c: c[0] if c else None)
    df["lat"] = coords.apply(lambda c: c[1] if c else None)
    df = df.dropna(subset=["lon", "lat"])
    if AOI_BBOX is not None:
        df = df[df.apply(lambda r: _in_bbox(r["lon"], r["lat"], AOI_BBOX), axis=1)]
    df["lamp_height"] = lamp_height
    df["lon"] = df["lon"].round(6)
    df["lat"] = df["lat"].round(6)

    trim = df[["lon", "lat", "lamp_height"]].copy()

    return pdk.Layer(
        "ColumnLayer",
        data=trim,
        get_position=["lon", "lat"],
        get_elevation="lamp_height",
        elevation_scale=1,
        radius=0.4,
        get_fill_color=[255, 244, 180, 230],
        pickable=False,
        extruded=True,
    )


def build_other_road_furniture(csv_path=f"{DATA_DIR}/RoadLamp_KL_OSM.csv"):
    """Optional: bus stops, crossings, traffic signals, etc. as flat markers."""
    df = pd.read_csv(csv_path, encoding="latin1", low_memory=False)
    df = df[df["highway"] != "street_lamp"].copy()
    df = df.dropna(subset=["highway"])

    coords = df["WKT"].apply(parse_wkt_point)
    df["lon"] = coords.apply(lambda c: c[0] if c else None)
    df["lat"] = coords.apply(lambda c: c[1] if c else None)
    df = df.dropna(subset=["lon", "lat"])

    palette = {
        "bus_stop": [255, 0, 0, 200],
        "traffic_signals": [255, 255, 0, 200],
        "crossing": [0, 200, 255, 200],
    }
    df["marker_color"] = df["highway"].apply(lambda h: palette.get(h, [180, 180, 180, 160]))
    df["lon"] = df["lon"].round(6)
    df["lat"] = df["lat"].round(6)

    trim = df[["lon", "lat", "marker_color"]].copy()

    return pdk.Layer(
        "ScatterplotLayer",
        data=trim,
        get_position=["lon", "lat"],
        get_fill_color="marker_color",
        get_radius=2,
        radius_min_pixels=2,
        pickable=True,
    )


# ---------------------------------------------------------------------------
# 5. FOREST / WATER -- PolygonLayer, split by natural feature type
# ---------------------------------------------------------------------------

GREEN_TYPES = {"wood", "scrub", "heath", "grassland", "shrubbery", "bare_rock", "sand", "fell"}
WATER_TYPES = {"water", "wetland"}


def build_forest_layers(csv_path=f"{DATA_DIR}/Forest_KL_OSM.csv"):
    df = pd.read_csv(csv_path, encoding="latin1", low_memory=False)

    def to_polygons(wkt):
        wkt = wkt.strip().strip('"')
        if wkt.startswith("MULTIPOLYGON"):
            return parse_wkt_multipolygon(wkt)
        if wkt.startswith("POLYGON"):
            p = parse_wkt_polygon(wkt)
            return [p] if p else None
        return None

    df["polys"] = df["WKT"].apply(to_polygons)
    df = df.dropna(subset=["polys"])
    df = df.explode("polys").dropna(subset=["polys"])
    df = df[df["polys"].apply(lambda p: p is not None and len(p) >= 3)]

    # Clip (not just include/exclude) against the AOI -- forest/water
    # polygons can be huge (a whole park or reservoir), and just keeping the
    # whole shape because one corner overlaps the AOI would render a
    # multi-km polygon at full size, dwarfing everything else in frame.
    before_n = len(df)
    df["polys"] = df["polys"].apply(lambda p: clip_polygon_to_bbox(p, AOI_BBOX))
    df = df[df["polys"].apply(lambda p: p is not None and len(p) >= 3)]
    print(f"Forest/water polygons: {before_n:,} -> {len(df):,} after AOI clip")

    df["polys"] = df["polys"].apply(
        lambda p: simplify_rdp(p, FOREST_SIMPLIFY_TOLERANCE_DEG) if len(p) > 4 else p
    )

    green = df[df["natural"].isin(GREEN_TYPES)][["polys"]].copy()
    water = df[df["natural"].isin(WATER_TYPES)][["polys"]].copy()

    green_layer = pdk.Layer(
        "PolygonLayer",
        data=green,
        get_polygon="polys",
        get_fill_color=[40, 120, 40, 140],
        get_line_color=[20, 80, 20, 200],
        pickable=True,
        stroked=True,
        filled=True,
    )
    water_layer = pdk.Layer(
        "PolygonLayer",
        data=water,
        get_polygon="polys",
        get_fill_color=[40, 110, 200, 160],
        get_line_color=[20, 70, 150, 200],
        pickable=True,
        stroked=True,
        filled=True,
    )
    return green_layer, water_layer


# ---------------------------------------------------------------------------
# Assemble and export
# ---------------------------------------------------------------------------

def main():
    with open(f"{DATA_DIR}/view_state.json") as f:
        vs = json.load(f)

    buildings_layer, landmarks_layer = load_building_layers()
    green_layer, water_layer = build_forest_layers()
    road_layer, pier_layer = build_road_layers()
    tree_trunk_layer, tree_canopy_layer = build_tree_layers()
    lamp_layer = build_lamp_layer()
    # furniture_layer = build_other_road_furniture()  # optional -- bus stops/crossings/signals

    view_state = pdk.ViewState(**vs["initialViewState"])

    deck = pdk.Deck(
        layers=[
            water_layer,
            green_layer,
            road_layer,
            pier_layer,
            buildings_layer,
            landmarks_layer,
            tree_trunk_layer,
            tree_canopy_layer,
            lamp_layer,
            # furniture_layer,
        ],
        initial_view_state=view_state,
        map_provider=vs["mapProvider"],
        map_style=vs["mapStyle"],
    )

    deck.to_html(str(OUTPUT_HTML))

    # pydeck's Python API doesn't expose useDevicePixelRatio directly, but
    # deck.gl's underlying JSON spec accepts it as a top-level Deck prop --
    # patch it into the exported HTML. Capping it to 1 (instead of the
    # default, which follows the screen's native pixel density -- often 2+
    # on modern laptop displays) roughly quarters the GPU's fragment-shading
    # workload with no change to what geometry is actually drawn. This is
    # usually the single biggest lever on an integrated GPU, more impactful
    # than most geometry-level trims.
    with open(OUTPUT_HTML, "r", encoding="utf-8") as f:
        html = f.read()
    marker = "};\n    const tooltip"
    if marker in html and '"useDevicePixelRatio"' not in html:
        html = html.replace(marker, ',\n  "useDevicePixelRatio": 1\n' + marker, 1)
        with open(OUTPUT_HTML, "w", encoding="utf-8", newline="") as f:
            f.write(html)

    print(f"Wrote {OUTPUT_HTML}")


if __name__ == "__main__":
    main()