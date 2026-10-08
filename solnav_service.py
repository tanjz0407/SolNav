"""Reusable SolNav route calculation service.

This module contains the routing work shared by route_server.py, Streamlit,
and other Python interfaces. Importing it does not start a web server.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import tempfile
import threading
from urllib.request import Request, urlopen

import routing_graph as rg
import shade_routing as sr
from shadow_math import get_solar_position, KL_LAT, KL_LON


ROOT = Path(__file__).resolve().parent
DEFAULT_HTML_PATH = ROOT / "solnav_3d_verified_map.html"
DEFAULT_CSV_PATH = ROOT / "data" / "Roads.csv"
ASSETS = {
    "Roads.csv": {
        "url": "https://github.com/tanjz0407/SolNav/releases/download/v1.0/Roads.csv",
        "sha256": "ac3da9a2f4c73b6c6b19e0706807987017f4ee8eb4699a12332a00bd9ec655da",
    },
    "solnav_3d_verified_map.html": {
        "url": "https://github.com/tanjz0407/SolNav/releases/download/v1.0/solnav_3d_verified_map.html",
        "sha256": "b476fee1099b6f84220c25e3d3b91f61060524492fc463d6bca32a449576e1d8",
    },
}
MAX_SNAP_M = 600.0
PROFILES = {"fastest": None, "average": 0.6, "coolest": 4.0}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_asset(path):
    """Use the local asset when present, otherwise fetch its public release copy."""
    path = Path(path).expanduser().resolve()
    asset = ASSETS.get(path.name)
    if path.is_file():
        return path
    if asset is None:
        raise FileNotFoundError(f"Required SolNav file not found: {path}")

    cache_dir = Path(tempfile.gettempdir()) / "solnav_assets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / path.name
    expected = asset["sha256"]
    if cached.is_file() and _sha256(cached) == expected:
        return cached
    if cached.exists():
        cached.unlink()

    temporary = cached.with_suffix(cached.suffix + ".part")
    digest = hashlib.sha256()
    request = Request(asset["url"], headers={"User-Agent": "SolNav-Streamlit"})
    try:
        with urlopen(request, timeout=180) as response, temporary.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected:
            raise ValueError(
                f"Downloaded {path.name} failed SHA-256 verification "
                f"(expected {expected}, received {actual}).")
        os.replace(temporary, cached)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return cached


class SolNavRouteService:
    """Load SolNav's data once and calculate routes on demand."""

    def __init__(self, html_path=DEFAULT_HTML_PATH, csv_path=DEFAULT_CSV_PATH):
        self.html_path = _resolve_asset(html_path)
        self.csv_path = _resolve_asset(csv_path)

        self.html = self.html_path.read_bytes()
        self.graph = rg.build_graph(str(self.csv_path))
        self.main = self.graph.largest_component()
        sr.get_geometry(str(self.html_path))
        self.eta_predictor = self._load_eta_predictor()
        self._shade_cache = {}
        self._lock = threading.RLock()

    @staticmethod
    def _load_eta_predictor():
        try:
            import train_eta_st_gcn as eta_model
            predictor = eta_model.load_eta_predictor()
            if predictor:
                print("Loaded ST-GCN ETA model.")
            return predictor
        except Exception as exc:
            # Routing remains usable without TensorFlow or a saved checkpoint.
            print(f"ST-GCN ETA unavailable; using routing formula ({exc}).")
            return None

    @staticmethod
    def parse_time(hhmm):
        """Parse an HH:MM Kuala Lumpur time; default to the current UTC time."""
        if not hhmm:
            return datetime.now(timezone.utc)
        h, m = hhmm.split(":")
        return sr.kl_time_to_utc(int(h), int(m))

    def _shade_for(self, dt_utc):
        bucket = int(dt_utc.timestamp() // 600)
        if bucket in self._shade_cache:
            return self._shade_cache[bucket]
        sun = get_solar_position(dt_utc, KL_LAT, KL_LON)
        if sun["altitude_deg"] <= 0.5:
            edge_shade = {}
        else:
            shadows, _ = sr.get_current_shadow_snapshot(
                html_path=str(self.html_path), dt_utc=dt_utc)
            edge_shade = sr.annotate_edge_shade(
                self.graph, shadows, sr.build_shadow_index(shadows))
        self._shade_cache.clear()
        self._shade_cache[bucket] = (edge_shade, sun)
        return edge_shade, sun

    def compute_routes(self, origin, destination, dt_utc):
        """Return route results as a JSON-serializable dictionary."""
        with self._lock:
            return self._compute_routes(origin, destination, dt_utc)

    def _compute_routes(self, origin, destination, dt_utc):
        graph, main = self.graph, self.main
        for label, point in (("origin", origin), ("destination", destination)):
            node_id = graph.nearest_node(point[0], point[1], allowed=main)
            lon, lat = graph.node_coords[node_id]
            if rg.haversine_m(point[0], point[1], lon, lat) > MAX_SNAP_M:
                return {"error": f"Your {label} is outside the mapped Kuala Lumpur area."}

        edge_shade, sun = self._shade_for(dt_utc)
        routes = {}
        for name, penalty in PROFILES.items():
            if penalty is None:
                route = rg.find_route(
                    graph, origin, destination, mode="fastest", main_component=main)
            else:
                route = rg.find_route(
                    graph, origin, destination, mode="coolest", main_component=main,
                    custom_weight_fn=rg.make_shade_weight(edge_shade, penalty))
            if not route:
                return {"error": "No walking route found between these points."}

            shaded, total = sr.route_shade_stats(graph, route, edge_shade)
            formula_eta = sr.route_eta_min(route)
            try:
                predicted_eta = (self.eta_predictor(graph, route, dt_utc)
                                 if self.eta_predictor else None)
            except Exception as exc:
                print(f"ST-GCN ETA failed; using routing formula ({exc}).")
                predicted_eta = None
            routes[name] = {
                "coordinates": route["coordinates"],
                "distance_m": round(total),
                "eta_min": round(predicted_eta if predicted_eta is not None
                                 else formula_eta, 1),
                "eta_source": "st_gcn" if predicted_eta is not None
                else "routing_formula",
                "shade_pct": round(100 * shaded / total) if total else 0,
                "sun_min": round(sr.sun_exposed_min(graph, route, edge_shade), 1),
                "delay_s": round(route["delay_s"]),
                "crossings": route["crossings"],
            }
        return {"routes": routes, "night": sun["altitude_deg"] <= 0.5,
                "sun": {"altitude_deg": round(sun["altitude_deg"], 1),
                        "azimuth_deg": round(sun["azimuth_deg"], 1)}}
