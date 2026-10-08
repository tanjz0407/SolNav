"""
routing_graph.py
----------------
Stage 1: a pedestrian road graph built from Road_KL_OSM.csv, plus an A*
router. shade_routing.py (Stage 2) imports from this file:

    build_graph, find_route, describe_route, make_shade_weight, haversine_m

HOW THE GRAPH IS BUILT
----------------------
- Every LINESTRING in the CSV is read (the CSV is a QGIS-style export with
  a WKT geometry column, so no GIS library is needed).
- Only walkable ways are kept: motorways, trunk roads, construction/proposed
  ways, private/no-access ways and foot=no ways are dropped. Cycleways are
  kept only if foot access is explicitly allowed. Steps, footways, paths,
  pedestrian streets, corridors, crossings and ordinary roads are kept.
  One-way tags are ignored, since they don't apply to pedestrians.
- Each way is broken into one edge per pair of consecutive vertices. Ways
  connect wherever they share an identical coordinate (which is how OSM
  junctions appear in the export). A bridge crossing over a road does NOT
  connect to it unless they truly share a vertex.
- Edges are straight lines between two vertices, so shade sampling in
  shade_routing.py follows the real road/footpath shape.

ADJACENCY FORMAT (what shade_routing.py relies on)
--------------------------------------------------
    graph.adjacency[node] = [(neighbor, dist_m, highway, name, bridge), ...]
    graph.node_coords[node] = (lon, lat)

USAGE
-----
    python routing_graph.py

Runs a self-test: builds the graph and routes between two KL points.
"""

import csv
import heapq
import io
import math
import os
import sys
import time as time_module
from collections import defaultdict

CSV_FILENAME = "Road_KL_OSM.csv"

EXCLUDED_HIGHWAYS = {
    "", "motorway", "motorway_link", "trunk", "trunk_link", "construction",
    "proposed", "bus_stop", "rest_area", "services", "bridleway", "elevator",
    "raceway",
}
FOOT_ALLOWED = {"yes", "designated", "permissive"}
FOOT_BLOCKED = {"no", "private", "permit"}
ACCESS_BLOCKED = {"private", "no", "residents"}
SHELTER_COVERED = {"yes", "arcade", "shelter"}
SHELTER_TUNNEL = {"building_passage", "covered", "passage"}

NODE_GRID_DEG = 0.001  # ~110 m cells for nearest-node lookup

# ---- Pedestrian time model -------------------------------------------------
WALK_SPEED_KMH = 4.0
WALK_SPEED_MPS = WALK_SPEED_KMH / 3.6      # 1.389 m/s
# walking on stairs (footbridges etc.)
STEPS_SPEED_MPS = 0.8

# Expected waiting time (seconds) added once per road crossing. These are
# ASSUMPTIONS (tune them): signals ~ half of a typical KL red phase, zebra/
# uncontrolled = waiting for a gap in traffic.
CROSSING_WAIT_S = {
    "signal": 40.0,      # crossing=traffic_signals / crossing:signals=yes
    "zebra": 3.0,
    "marked": 8.0,
    "uncontrolled": 8.0,
    "unmarked": 12.0,
    "informal": 12.0,
}
DEFAULT_CROSSING_WAIT_S = 8.0
FOOTBRIDGE_HIGHWAYS = {"footway", "steps", "path", "pedestrian"}


# ---------------------------------------------------------------------------
def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres between two lon/lat points."""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * \
        math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
class Graph:
    def __init__(self):
        self.adjacency = defaultdict(list)
        self.node_coords = {}
        self.sheltered_edges = set()
        self.edge_delay_s = {}   # (min,max) -> extra seconds (waits, stairs)
        # (min,max) -> signal|zebra|marked|...|footbridge|steps
        self.edge_kind = {}
        self._node_grid = None

    # -- connectivity ------------------------------------------------------
    def largest_component(self):
        """Returns the set of node ids in the biggest connected component,
        so origins/destinations never snap to an isolated fragment."""
        seen = set()
        best = set()
        for start in self.adjacency:
            if start in seen:
                continue
            comp = {start}
            stack = [start]
            while stack:
                n = stack.pop()
                for nb in self.adjacency[n]:
                    m = nb[0]
                    if m not in comp:
                        comp.add(m)
                        stack.append(m)
            seen |= comp
            if len(comp) > len(best):
                best = comp
        return best

    # -- snapping ----------------------------------------------------------
    def _build_node_grid(self):
        grid = defaultdict(list)
        for nid, (lon, lat) in self.node_coords.items():
            grid[(int(lon / NODE_GRID_DEG), int(lat / NODE_GRID_DEG))].append(nid)
        self._node_grid = grid

    def nearest_node(self, lon, lat, allowed=None, max_rings=8):
        """Nearest graph node to (lon, lat). If `allowed` (a set of node ids)
        is given, only those nodes are considered."""
        if self._node_grid is None:
            self._build_node_grid()
        cx, cy = int(lon / NODE_GRID_DEG), int(lat / NODE_GRID_DEG)

        def scan(radius):
            best_id, best_d = None, float("inf")
            for gx in range(cx - radius, cx + radius + 1):
                for gy in range(cy - radius, cy + radius + 1):
                    for nid in self._node_grid.get((gx, gy), ()):
                        if allowed is not None and nid not in allowed:
                            continue
                        nlon, nlat = self.node_coords[nid]
                        d = haversine_m(lon, lat, nlon, nlat)
                        if d < best_d:
                            best_id, best_d = nid, d
            return best_id, best_d

        for r in range(max_rings + 1):
            best_id, best_d = scan(r)
            if best_id is not None:
                # widen once more so a closer node in a neighbouring cell wins
                best_id, best_d = scan(r + 1)
                return best_id
        # fallback: brute force
        pool = allowed if allowed is not None else self.node_coords.keys()
        best_id, best_d = None, float("inf")
        for nid in pool:
            nlon, nlat = self.node_coords[nid]
            d = haversine_m(lon, lat, nlon, nlat)
            if d < best_d:
                best_id, best_d = nid, d
        return best_id


# ---------------------------------------------------------------------------
def _read_text(csv_path):
    with open(csv_path, "rb") as f:
        raw = f.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        # the OSM export contains Windows-1252 characters (e.g. en dashes)
        return raw.decode("cp1252", errors="replace")


def _parse_linestring(wkt):
    try:
        inner = wkt[wkt.index("(") + 1: wkt.rindex(")")]
        pts = []
        for pair in inner.split(","):
            parts = pair.split()
            pts.append((float(parts[0]), float(parts[1])))
        return pts
    except (ValueError, IndexError):
        return None


def _is_walkable(row):
    hw = (row.get("highway") or "").strip()
    if hw in EXCLUDED_HIGHWAYS:
        return False
    foot = (row.get("foot") or "").strip()
    access = (row.get("access") or "").strip()
    if hw == "cycleway" and foot not in FOOT_ALLOWED:
        return False
    if foot in FOOT_BLOCKED:
        return False
    if access in ACCESS_BLOCKED and foot not in FOOT_ALLOWED:
        return False
    if (row.get("motorroad") or "").strip() == "yes":
        return False
    return True


def _is_sheltered(row):
    return ((row.get("covered") or "").strip() in SHELTER_COVERED
            or (row.get("tunnel") or "").strip() in SHELTER_TUNNEL
            or (row.get("indoor") or "").strip() == "yes"
            or (row.get("highway") or "").strip() == "corridor")


def _classify_way(row, highway):
    """Returns (kind, wait_seconds_for_whole_way)."""
    cross = (row.get("crossing") or "").strip()
    is_crossing = (row.get("footway") or "").strip(
    ) == "crossing" or highway == "crossing"
    if is_crossing:
        if cross == "traffic_signals" or (row.get("crossing:signals") or "").strip() == "yes":
            kind = "signal"
        elif cross == "zebra" or (row.get("crossing:markings") or "").strip() in ("zebra", "ladder"):
            kind = "zebra"
        elif cross in CROSSING_WAIT_S:
            kind = cross
        else:
            kind = "crossing"
        return kind, CROSSING_WAIT_S.get(kind, DEFAULT_CROSSING_WAIT_S)
    if ((row.get("bridge") or "").strip() != ""
            and highway in FOOTBRIDGE_HIGHWAYS
            and (row.get("footway") or "").strip() != "sidewalk"):
        return "footbridge", 0.0
    if highway == "steps":
        return "steps", 0.0
    return None, 0.0


def build_graph(csv_path=None):
    """Builds the pedestrian graph from the road CSV."""
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), CSV_FILENAME)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Road CSV not found: {csv_path}\n"
            f"Put {CSV_FILENAME} in the same folder as routing_graph.py, "
            f"or pass build_graph(csv_path=...).")

    t0 = time_module.time()
    print(f"Building pedestrian graph from {csv_path} ...")
    csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))
    text = _read_text(csv_path)
    reader = csv.DictReader(io.StringIO(text, newline=""))

    graph = Graph()
    node_ids = {}
    seen_pairs = set()
    ways_total = ways_kept = 0

    def node_id(pt):
        key = (round(pt[0], 7), round(pt[1], 7))
        nid = node_ids.get(key)
        if nid is None:
            nid = len(node_ids)
            node_ids[key] = nid
            graph.node_coords[nid] = key
        return nid

    for row in reader:
        ways_total += 1
        if not _is_walkable(row):
            continue
        pts = _parse_linestring(row.get("WKT") or "")
        if not pts or len(pts) < 2:
            continue
        ways_kept += 1

        highway = (row.get("highway") or "").strip()
        name = (row.get("name") or "").strip()
        bridge = (row.get("bridge") or "").strip() != ""
        sheltered = _is_sheltered(row)
        kind, way_wait_s = _classify_way(row, highway)
        way_len = sum(haversine_m(a[0], a[1], b[0], b[1])
                      for a, b in zip(pts, pts[1:])) or 1.0

        prev = node_id(pts[0])
        for pt in pts[1:]:
            cur = node_id(pt)
            if cur == prev:
                continue
            pair = (prev, cur) if prev < cur else (cur, prev)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                lon1, lat1 = graph.node_coords[prev]
                lon2, lat2 = graph.node_coords[cur]
                d = haversine_m(lon1, lat1, lon2, lat2)
                graph.adjacency[prev].append((cur, d, highway, name, bridge))
                graph.adjacency[cur].append((prev, d, highway, name, bridge))
                if sheltered:
                    graph.sheltered_edges.add(pair)
                if kind:
                    graph.edge_kind[pair] = kind
                delay = way_wait_s * d / way_len      # wait spread over the way
                if highway == "steps":
                    delay += d * (1.0 / STEPS_SPEED_MPS - 1.0 / WALK_SPEED_MPS)
                if delay > 0:
                    graph.edge_delay_s[pair] = delay
            prev = cur

    n_edges = len(seen_pairs)
    print(f"  ways read: {ways_total} | walkable ways kept: {ways_kept}")
    print(f"  nodes: {len(graph.node_coords)} | edges: {n_edges} | "
          f"sheltered edges: {len(graph.sheltered_edges)}")
    print(f"  graph built in {time_module.time() - t0:.1f}s")
    return graph


# ---------------------------------------------------------------------------
def make_shade_weight(edge_shade, shade_penalty=2.0):
    """Returns a cost function for 'coolest' routing.

    cost = distance * (1 + shade_penalty * sun_fraction)

    where sun_fraction = 1 - shade_fraction. A fully shaded edge costs just
    its length; a fully sunny edge costs (1 + shade_penalty) x its length.
    Cost is never below the distance, so the straight-line heuristic in
    find_route stays valid (A* remains optimal)."""
    def weight(n1, n2, dist_m):
        sun_fraction = 1.0 - edge_shade.get((n1, n2), 0.0)
        return dist_m * (1.0 + shade_penalty * sun_fraction)
    return weight


def find_route(graph, origin, destination, mode="fastest",
               main_component=None, custom_weight_fn=None):
    """A* route between two (lon, lat) points.

    mode: "fastest" (pure distance) or "coolest" (uses custom_weight_fn,
    e.g. from make_shade_weight). Returns a dict, or None if no route."""
    src = graph.nearest_node(origin[0], origin[1], allowed=main_component)
    dst = graph.nearest_node(
        destination[0], destination[1], allowed=main_component)
    if src is None or dst is None:
        return None

    if mode == "coolest" and custom_weight_fn is not None:
        base_weight = custom_weight_fn
    else:
        def base_weight(n1, n2, dist_m):
            return dist_m

    delays = graph.edge_delay_s

    def weight(n1, n2, dist_m):
        # waiting/stair time is converted to "metres of walking" so the
        # router really minimises time, and A*'s heuristic stays valid.
        key = (n1, n2) if n1 < n2 else (n2, n1)
        return base_weight(n1, n2, dist_m) + delays.get(key, 0.0) * WALK_SPEED_MPS

    dlon, dlat = graph.node_coords[dst]
    h_cache = {}

    def h(n):
        v = h_cache.get(n)
        if v is None:
            lon, lat = graph.node_coords[n]
            v = haversine_m(lon, lat, dlon, dlat)
            h_cache[n] = v
        return v

    g = {src: 0.0}
    came_from = {}
    heap = [(h(src), src)]
    closed = set()

    while heap:
        _, n = heapq.heappop(heap)
        if n in closed:
            continue
        if n == dst:
            break
        closed.add(n)
        gn = g[n]
        for nb, dist_m, highway, name, bridge in graph.adjacency[n]:
            if nb in closed:
                continue
            ng = gn + weight(n, nb, dist_m)
            if ng < g.get(nb, float("inf")):
                g[nb] = ng
                came_from[nb] = (n, dist_m, highway, name, bridge)
                heapq.heappush(heap, (ng + h(nb), nb))
    else:
        return None

    if dst != src and dst not in came_from:
        return None

    path = [dst]
    segments = []
    n = dst
    while n != src:
        prev, dist_m, highway, name, bridge = came_from[n]
        key = (prev, n) if prev < n else (n, prev)
        segments.append({"from": prev, "to": n, "distance_m": dist_m,
                         "highway": highway, "name": name, "bridge": bridge,
                         "delay_s": delays.get(key, 0.0),
                         "kind": graph.edge_kind.get(key)})
        path.append(prev)
        n = prev
    path.reverse()
    segments.reverse()

    crossings, last = {}, None
    for s in segments:
        if s["kind"] and s["kind"] != last:
            crossings[s["kind"]] = crossings.get(s["kind"], 0) + 1
        last = s["kind"]

    return {
        "delay_s": sum(s["delay_s"] for s in segments),
        "crossings": crossings,
        "mode": mode,
        "origin_node": src,
        "destination_node": dst,
        "path": path,
        "segments": segments,
        "coordinates": [list(graph.node_coords[p]) for p in path],
        "total_distance_m": sum(s["distance_m"] for s in segments),
        "total_cost": g[dst],
    }


def describe_route(result, walk_speed_mps=WALK_SPEED_MPS):
    """Prints a short human-readable summary of a route."""
    if not result:
        print("No route found.")
        return
    total = result["total_distance_m"]
    print(f"Route ({result['mode']}): {total:.0f} m, "
          f"~{(total / walk_speed_mps + result.get('delay_s', 0)) / 60:.1f} min walk "
          f"(incl. {result.get('delay_s', 0):.0f}s waiting/stairs), "
          f"{len(result['segments'])} segments")

    by_type = defaultdict(float)
    for s in result["segments"]:
        by_type[s["highway"] or "unknown"] += s["distance_m"]
    mix = ", ".join(f"{k} {v:.0f}m" for k, v in
                    sorted(by_type.items(), key=lambda kv: -kv[1])[:5])
    print(f"  Path types: {mix}")

    names = []
    for s in result["segments"]:
        nm = s["name"]
        if nm and (not names or names[-1] != nm):
            names.append(nm)
    if names:
        shown = " -> ".join(names[:8]) + (" -> ..." if len(names) > 8 else "")
        print(f"  Via: {shown}")

    if result.get("crossings"):
        print("  Crossings/obstacles:", result["crossings"])
    bridges = sum(s["distance_m"] for s in result["segments"] if s["bridge"])
    if bridges > 0:
        print(f"  On bridges/elevated: {bridges:.0f} m")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    graph = build_graph()
    main = graph.largest_component()
    print(
        f"Largest connected component: {len(main)} of {len(graph.node_coords)} nodes")

    origin = (101.7118, 3.1583)       # near Petronas Towers
    destination = (101.7007, 3.1416)  # near Merdeka 118

    t0 = time_module.time()
    route = find_route(graph, origin, destination,
                       mode="fastest", main_component=main)
    print(f"Routed in {time_module.time() - t0:.2f}s")
    describe_route(route)
