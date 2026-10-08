#!/usr/bin/env python3
"""
shadow_math.py
--------------
Shared, verified solar-position and shadow-geometry math used by both
live_shadow_server.py (the live WebSocket broadcaster) and shade_routing.py
(Stage 2 routing). Lives in its own module with zero external dependencies
so that anything which only needs the math -- like shade_routing.py -- never
has to import `websockets` just to get a shadow snapshot.

WHAT'S HERE
-----------
- get_solar_position(): sun altitude/azimuth for a given UTC time + lat/lon.
- load_source_geometry(): reads building/landmark/pillar/tree geometry back
  out of an already-rendered HTML map (read-only).
- build_shadows(): projects real shadow polygons from that geometry for a
  given sun position.

Nothing in this file talks to a socket, a browser, or the filesystem beyond
reading the one HTML file passed in.
"""

import json
import math
from datetime import datetime, timezone

KL_LAT = 3.15
KL_LON = 101.70

TALL_BUILDING_THRESHOLD_M = 40.0
SHADOW_ELEV = 0.15
MAX_SHADOW_LENGTH_M = 500.0

M_LAT = 111320.0


def m_lon(lat):
    return 111320.0 * math.cos(math.radians(lat))


# ---------------------------------------------------------------------------
# Solar position (verified against known noon/midnight/sunrise behaviour for
# Kuala Lumpur)
# ---------------------------------------------------------------------------
def get_solar_position(dt_utc, lat, lon):
    rad = math.pi / 180.0
    j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    d = (dt_utc - j2000).total_seconds() / 86400.0

    mean_lon = (280.460 + 0.9856474 * d) % 360
    mean_anom = math.radians((357.528 + 0.9856003 * d) % 360)
    ecl_lon = math.radians(
        mean_lon + 1.915 * math.sin(mean_anom) +
        0.020 * math.sin(2 * mean_anom)
    )
    obl_ecl = math.radians(23.439 - 0.0000004 * d)

    ra = math.atan2(math.cos(obl_ecl) * math.sin(ecl_lon), math.cos(ecl_lon))
    dec = math.asin(math.sin(obl_ecl) * math.sin(ecl_lon))

    gmst = (280.46061837 + 360.98564736629 * d) % 360
    lst = math.radians(gmst + lon)
    hour_angle = lst - ra
    hour_angle = math.atan2(math.sin(hour_angle), math.cos(hour_angle))

    lat_r = math.radians(lat)
    altitude = math.asin(
        math.sin(lat_r) * math.sin(dec) + math.cos(lat_r) *
        math.cos(dec) * math.cos(hour_angle)
    )
    azimuth = math.atan2(
        -math.sin(hour_angle),
        math.tan(dec) * math.cos(lat_r) -
        math.sin(lat_r) * math.cos(hour_angle),
    )

    return {
        "altitude_deg": altitude / rad,
        "azimuth_deg": (azimuth / rad + 360) % 360,
    }


def convex_hull(points):
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def shadow_polygon_for_footprint(footprint, offset_lon, offset_lat):
    base_pts = [(p[0], p[1]) for p in footprint]
    shifted_pts = [(p[0] + offset_lon, p[1] + offset_lat) for p in base_pts]
    hull = convex_hull(base_pts + shifted_pts)
    if len(hull) < 3:
        return None
    ring = [[round(x, 7), round(y, 7), SHADOW_ELEV] for x, y in hull]
    ring.append(ring[0])
    return ring


def shadow_offset_vector(lat, height_m, azimuth_deg, altitude_deg):
    if altitude_deg <= 0.5:
        return None
    shadow_len_m = min(
        height_m / math.tan(math.radians(altitude_deg)), MAX_SHADOW_LENGTH_M)
    shadow_dir_deg = (azimuth_deg + 180.0) % 360.0
    dir_r = math.radians(shadow_dir_deg)
    dx_m = shadow_len_m * math.sin(dir_r)
    dy_m = shadow_len_m * math.cos(dir_r)
    return dx_m / m_lon(lat), dy_m / M_LAT


# ---------------------------------------------------------------------------
# Reading source geometry from the HTML (read-only -- this file is never
# rewritten)
# ---------------------------------------------------------------------------
def find_balanced(content, start_idx):
    depth = 0
    in_str = False
    esc = False
    i = start_idx
    while i < len(content):
        c = content[i:i + 1]
        if in_str:
            if esc:
                esc = False
            elif c == b'\\':
                esc = True
            elif c == b'"':
                in_str = False
        else:
            if c == b'"':
                in_str = True
            elif c == b'[':
                depth += 1
            elif c == b']':
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    raise ValueError("no matching bracket found")


def extract_layer_data(content, layer_id):
    id_marker = ('"id":"%s"' % layer_id).encode('utf-8')
    id_idx = content.find(id_marker)
    if id_idx == -1:
        return None
    obj_start = content.rfind(b'{"@@type"', 0, id_idx)
    if obj_start == -1:
        return None
    data_marker = b'"data":['
    data_idx = content.find(data_marker, obj_start)
    if data_idx == -1:
        return None
    arr_start = data_idx + len(data_marker) - 1
    arr_end = find_balanced(content, arr_start)
    return json.loads(content[arr_start:arr_end + 1])


def load_source_geometry(html_path):
    with open(html_path, 'rb') as f:
        content = f.read()
    return {
        "buildings": extract_layer_data(content, 'e31da629-c637-4a8d-a8d7-ac962eed2170'),
        "landmarks": extract_layer_data(content, 'b05b9bf1-c11a-4487-81fc-d10d83bdb09e'),
        "pillars": extract_layer_data(content, 'kl-bridge-pillars'),
        "trees": extract_layer_data(content, 'kl-trees'),
    }


# ---------------------------------------------------------------------------
def build_shadows(geometry, sun):
    shadows = []

    def add_buildings(data, min_height=0.0):
        if not data:
            return 0
        n = 0
        for item in data:
            h = item.get('height_m')
            if h is None or h < min_height:
                continue
            offset = shadow_offset_vector(
                KL_LAT, h, sun['azimuth_deg'], sun['altitude_deg'])
            if offset is None:
                continue
            footprint = item.get('polygon_coordinates')
            if not footprint or len(footprint) < 3:
                continue
            ring = shadow_polygon_for_footprint(
                footprint, offset[0], offset[1])
            if ring:
                shadows.append(
                    {"polygon_coordinates": ring, "elev": SHADOW_ELEV})
                n += 1
        return n

    def add_pillars(data):
        if not data:
            return 0
        n = 0
        for item in data:
            h = item.get('elev')
            if h is None or h <= 0:
                continue
            offset = shadow_offset_vector(
                KL_LAT, h, sun['azimuth_deg'], sun['altitude_deg'])
            if offset is None:
                continue
            footprint = item.get('polygon_coordinates')
            if not footprint or len(footprint) < 3:
                continue
            ring = shadow_polygon_for_footprint(
                footprint, offset[0], offset[1])
            if ring:
                shadows.append(
                    {"polygon_coordinates": ring, "elev": SHADOW_ELEV})
                n += 1
        return n

    def add_trees(data, tiers_per_tree=3):
        if not data:
            return 0
        n = 0
        for i in range(0, len(data) - tiers_per_tree + 1, tiers_per_tree):
            group = data[i:i + tiers_per_tree]
            base = group[0]
            total_h = sum(t.get('elev', 0) for t in group)
            if total_h <= 0:
                continue
            offset = shadow_offset_vector(
                KL_LAT, total_h, sun['azimuth_deg'], sun['altitude_deg'])
            if offset is None:
                continue
            footprint = base.get('polygon_coordinates')
            if not footprint or len(footprint) < 3:
                continue
            ring = shadow_polygon_for_footprint(
                footprint, offset[0], offset[1])
            if ring:
                shadows.append(
                    {"polygon_coordinates": ring, "elev": SHADOW_ELEV})
                n += 1
        return n

    n1 = add_buildings(geometry["buildings"],
                        min_height=TALL_BUILDING_THRESHOLD_M)
    n2 = add_buildings(geometry["landmarks"], min_height=0.0)
    n3 = add_pillars(geometry["pillars"])
    n4 = add_trees(geometry["trees"])
    print(f"  buildings>{TALL_BUILDING_THRESHOLD_M}m: {n1} | landmarks: {n2} | "
          f"pillars: {n3} | trees: {n4}")

    return shadows
