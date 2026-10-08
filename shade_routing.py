#!/usr/bin/env python3
"""
shade_routing.py
------------------
Stage 2: wire the road graph (Stage 1) to real shadow geometry, so
"coolest path" is an actual shade-optimized route instead of a stub.

This reuses the exact solar-position and shadow-projection code from
shadow_math.py (same verified math, not a reimplementation -- and shared
with live_shadow_server.py), and the graph/A* engine from routing_graph.py
unchanged. Unlike importing from live_shadow_server.py directly, this file
has NO dependency on the `websockets` package, since it never opens a
socket -- it only needs the math and geometry-reading functions.

WHAT THIS ADDS OVER STAGE 1
-----------------------------
1. Computes a shadow snapshot for right now, a specific time, or a
   hypothetical sun position.
2. Builds a spatial grid index over the shadow polygons, since a naive
   edge-vs-polygon check would be ~56,000 edges x ~35,000 polygons =
   too slow without one.
3. Walks every graph edge and estimates what fraction of it currently
   sits inside a shadow (sampling every ~8 m along the edge).
4. Produces a real make_shade_weight() function Stage 1 can route with.
5. Reports walking ETA and minutes spent in direct sun for each route.

CHANGES IN THIS VERSION
-----------------------
- Edge shade is sampled by distance (every SAMPLE_SPACING_M metres, at
  segment midpoints) instead of 3 fixed points, so fractions are smooth
  and junction points are no longer counted twice.
- Source geometry (the big HTML file) is parsed once and cached, instead
  of once per scenario.
- get_current_shadow_snapshot() accepts dt_utc, so you can test real
  times of day (e.g. 09:00, 15:00 KL time) instead of hand-typed sun angles.
- Added walking ETA and "minutes in direct sun" to the route summaries.

USAGE
-----
    python shade_routing.py

Runs a demo: builds the graph, computes shadow snapshots, annotates
every edge, then compares the fastest route against the coolest route for
the same origin/destination -- printing extra distance/time versus how
much less sun exposure it buys.
"""

import math
from pathlib import Path
import time as time_module
from collections import defaultdict
from datetime import datetime, timezone, timedelta

# Reuse the exact, already-verified solar/shadow code -- not reimplemented.
from shadow_math import (
    get_solar_position, load_source_geometry, build_shadows,
    KL_LAT, KL_LON
)
from routing_graph import (
    build_graph, find_route, describe_route, make_shade_weight, haversine_m,
    WALK_SPEED_MPS
)

ROOT = Path(__file__).resolve().parent
HTML_SOURCE_PATH = str(ROOT / "solnav_3d_verified_map.html")

# ~110m cells -- coarse enough to be cheap, fine enough to be useful
GRID_CELL_DEG = 0.001

# Sample a point on each edge roughly this often (metres)
SAMPLE_SPACING_M = 8.0

# Walking speed (5 km/h) now lives in routing_graph.py and is imported above.

KL_TZ = timezone(timedelta(hours=8))

_GEOMETRY_CACHE = {}


# ---------------------------------------------------------------------------
def get_geometry(html_path=None):
    """Parses the source HTML once per path and reuses the result."""
    html_path = html_path or HTML_SOURCE_PATH
    if html_path not in _GEOMETRY_CACHE:
        print(f"Loading source geometry from {html_path} ...")
        _GEOMETRY_CACHE[html_path] = load_source_geometry(html_path)
    return _GEOMETRY_CACHE[html_path]


def kl_time_to_utc(hour, minute=0, day=None):
    """Convenience: build a UTC datetime for a given KL local time.
    day defaults to today (KL date)."""
    base = day or datetime.now(KL_TZ)
    local = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return local.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
def get_current_shadow_snapshot(html_path=None, sun_override=None, dt_utc=None):
    """Computes shadows for a chosen moment, using the live production code.

    html_path: path to the HTML file to read geometry from. Defaults to
    the module-level HTML_SOURCE_PATH -- resolved at CALL time, not at
    import time, so reassigning shade_routing.HTML_SOURCE_PATH before
    calling this actually takes effect.

    sun_override: optional {'altitude_deg':.., 'azimuth_deg':..} to compute
    shadows for a hypothetical sun position.

    dt_utc: optional timezone-aware UTC datetime to compute the real sun
    position for that moment (e.g. 15:00 KL time = 07:00 UTC). If neither
    sun_override nor dt_utc is given, the current real time is used."""
    if sun_override is not None:
        sun = sun_override
        print(f"[SIMULATED] Sun altitude={sun['altitude_deg']:.1f} deg, "
              f"azimuth={sun['azimuth_deg']:.1f} deg")
    else:
        moment = dt_utc or datetime.now(timezone.utc)
        sun = get_solar_position(moment, KL_LAT, KL_LON)
        tag = "TIME" if dt_utc else "LIVE"
        kl_str = moment.astimezone(KL_TZ).strftime("%Y-%m-%d %H:%M KL")
        print(f"[{tag} {kl_str}] Sun altitude={sun['altitude_deg']:.1f} deg, "
              f"azimuth={sun['azimuth_deg']:.1f} deg")

    if sun['altitude_deg'] <= 0.5:
        print("Sun below/near horizon -- no shadows exist at this time (nighttime).")
        return [], sun

    geometry = get_geometry(html_path)
    shadows = build_shadows(geometry, sun)
    print(f"Computed {len(shadows)} shadow polygons.")
    return shadows, sun


# ---------------------------------------------------------------------------
# Spatial index: bucket each shadow polygon into every grid cell its
# bounding box overlaps, so point-in-shadow lookups only check nearby
# polygons instead of all of them.
# ---------------------------------------------------------------------------
def build_shadow_index(shadows, cell_deg=GRID_CELL_DEG):
    grid = defaultdict(list)
    for idx, shadow in enumerate(shadows):
        ring = shadow["polygon_coordinates"]
        lons = [p[0] for p in ring]
        lats = [p[1] for p in ring]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        c0x, c1x = int(min_lon / cell_deg), int(max_lon / cell_deg)
        c0y, c1y = int(min_lat / cell_deg), int(max_lat / cell_deg)
        for cx in range(c0x, c1x + 1):
            for cy in range(c0y, c1y + 1):
                grid[(cx, cy)].append(idx)
    return grid


def point_in_polygon(lon, lat, ring):
    """Standard ray-casting point-in-polygon test."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > lat) != (yj > lat)) and \
           (lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside


def is_point_shaded(lon, lat, shadows, grid, cell_deg=GRID_CELL_DEG):
    cx, cy = int(lon / cell_deg), int(lat / cell_deg)
    candidates = grid.get((cx, cy), [])
    for idx in candidates:
        if point_in_polygon(lon, lat, shadows[idx]["polygon_coordinates"]):
            return True
    return False


# ---------------------------------------------------------------------------
def annotate_edge_shade(graph, shadows, grid, cell_deg=GRID_CELL_DEG,
                        spacing_m=SAMPLE_SPACING_M):
    """Returns {(n1, n2): shade_fraction} for every edge in both directions.

    Each edge is sampled at segment midpoints roughly every `spacing_m`
    metres (at least one sample per edge), so the fraction reflects how
    much of the edge's length is shaded rather than jumping in thirds."""
    edge_shade = {}
    seen_pairs = set()

    for n1, neighbors in graph.adjacency.items():
        for n2, dist_m, highway, name, bridge in neighbors:
            key = (min(n1, n2), max(n1, n2))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            lon1, lat1 = graph.node_coords[n1]
            lon2, lat2 = graph.node_coords[n2]

            n_samples = max(1, math.ceil(dist_m / spacing_m))
            shaded_count = 0
            for i in range(n_samples):
                t = (i + 0.5) / n_samples
                slon = lon1 + (lon2 - lon1) * t
                slat = lat1 + (lat2 - lat1) * t
                if is_point_shaded(slon, slat, shadows, grid, cell_deg):
                    shaded_count += 1

            fraction = shaded_count / n_samples
            edge_shade[(n1, n2)] = fraction
            edge_shade[(n2, n1)] = fraction

    return edge_shade


# ---------------------------------------------------------------------------
# Route statistics
# ---------------------------------------------------------------------------
def route_shade_stats(graph, result, edge_shade):
    """Given a computed route, report how much of it is actually shaded."""
    shaded_dist = 0.0
    total_dist = 0.0
    for seg in result["segments"]:
        frac = edge_shade.get((seg["from"], seg["to"]), 0.0)
        shaded_dist += seg["distance_m"] * frac
        total_dist += seg["distance_m"]
    return shaded_dist, total_dist


def route_eta_min(result, extra_s=0.0):
    """Walking ETA in minutes: distance / walking speed (+ optional extra
    seconds, e.g. for road crossings or stairs)."""
    return (result["total_distance_m"] / WALK_SPEED_MPS
            + result.get("delay_s", 0.0) + extra_s) / 60.0


def sun_exposed_min(graph, result, edge_shade):
    """Minutes of the walk spent in direct sun (unshaded distance / speed)."""
    shaded_m, total_m = route_shade_stats(graph, result, edge_shade)
    return (total_m - shaded_m) / WALK_SPEED_MPS / 60.0


def print_route_summary(graph, result, edge_shade):
    describe_route(result)
    shaded_m, total_m = route_shade_stats(graph, result, edge_shade)
    if total_m <= 0:
        print("  (empty route)")
        return
    print(f"  Shade exposure: {shaded_m:.0f}m / {total_m:.0f}m in shadow "
          f"({100 * shaded_m / total_m:.0f}%)")
    print(f"  Walking ETA: {route_eta_min(result):.1f} min "
          f"| in direct sun: {sun_exposed_min(graph, result, edge_shade):.1f} min")


# ---------------------------------------------------------------------------
def run_scenario(graph, main_component, origin, destination, sun_override=None,
                 label="", html_path=None, dt_utc=None):
    print(f"\n{'=' * 60}\nScenario: {label}\n{'=' * 60}")
    shadows, sun = get_current_shadow_snapshot(
        html_path=html_path, sun_override=sun_override, dt_utc=dt_utc)

    if not shadows:
        print("No shadows to route against for this scenario.")
        return

    grid = build_shadow_index(shadows)
    edge_shade = annotate_edge_shade(graph, shadows, grid)
    shaded_edges = sum(1 for v in edge_shade.values() if v > 0)
    print(f"{shaded_edges} of {len(edge_shade)} directed edge-entries have at least partial shade.")

    shade_weight_fn = make_shade_weight(edge_shade, shade_penalty=2.0)

    fastest = find_route(graph, origin, destination,
                         mode="fastest", main_component=main_component)
    coolest = find_route(graph, origin, destination, mode="coolest",
                         main_component=main_component, custom_weight_fn=shade_weight_fn)

    if fastest:
        print("\n-- Fastest route --")
        print_route_summary(graph, fastest, edge_shade)

    if coolest:
        print("\n-- Coolest route --")
        print_route_summary(graph, coolest, edge_shade)

    if fastest and coolest:
        extra_dist = coolest["total_distance_m"] - fastest["total_distance_m"]
        extra_min = route_eta_min(coolest) - route_eta_min(fastest)
        sun_saved_min = (sun_exposed_min(graph, fastest, edge_shade)
                         - sun_exposed_min(graph, coolest, edge_shade))
        fastest_shaded, fastest_total = route_shade_stats(
            graph, fastest, edge_shade)
        coolest_shaded, coolest_total = route_shade_stats(
            graph, coolest, edge_shade)
        pp_gain = (100 * coolest_shaded / coolest_total
                   - 100 * fastest_shaded / fastest_total)
        print(f"\nTrade-off: coolest route is {extra_dist:+.0f}m "
              f"({100 * extra_dist / fastest['total_distance_m']:+.0f}%) / "
              f"{extra_min:+.1f} min longer than fastest, "
              f"for {pp_gain:+.0f} percentage points more shade coverage "
              f"and {sun_saved_min:.1f} min less time in direct sun.")
    return fastest, coolest, edge_shade


if __name__ == "__main__":
    t0 = time_module.time()

    print("=== Stage 2: shade-aware routing ===")
    graph = build_graph()
    main_component = graph.largest_component()

    origin = (101.7118, 3.1583)       # near Petronas Towers
    destination = (101.7007, 3.1416)  # near Merdeka 118

    # Scenario A: right now, whatever the actual sun position is.
    run_scenario(graph, main_component, origin, destination,
                 label="LIVE (current real sun position)")

    # Scenarios B-E: real sun positions at four times of day (KL local time).
    for hour in (9, 11, 15, 17):
        run_scenario(graph, main_component, origin, destination,
                     dt_utc=kl_time_to_utc(hour),
                     label=f"Today at {hour:02d}:00 KL time")

    # Scenario F: simulated midday, high sun -- shadows are short and
    # localized instead of blanketing the whole network.
    run_scenario(graph, main_component, origin, destination,
                 sun_override={"altitude_deg": 75.0, "azimuth_deg": 180.0},
                 label="SIMULATED midday (sun nearly overhead)")

    # Scenario G: simulated mid-afternoon, moderate sun angle.
    run_scenario(graph, main_component, origin, destination,
                 sun_override={"altitude_deg": 35.0, "azimuth_deg": 260.0},
                 label="SIMULATED mid-afternoon")

    print(f"\nTotal run time: {time_module.time() - t0:.1f}s")
