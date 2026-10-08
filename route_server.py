#!/usr/bin/env python3
"""
route_server.py -- SolNav route API + map page server (stdlib only)

Serves your 3D map HTML with route_ui.html injected, and answers
GET /api/routes with fastest / average / coolest walking routes.
ETA = distance / ST-GCN model + traffic-light / crossing waits + stair time.

    python route_server.py "path\\to\\solnav_3d_verified_map.html"
    -> open http://localhost:8000
Same folder as: shadow_math.py, routing_graph.py, shade_routing.py,
route_ui.html and the road CSV (Road_KL_OSM.csv or Road_KL_OSM_1.csv).
"""
import json
import os
import sys
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from solnav_service import SolNavRouteService

HERE = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(
    HERE, "solnav_3d_verified_map.html")
CSV_PATH = os.path.join(HERE, "data", "Roads.csv")
PORT = 8000

_lock = threading.Lock()
SERVICE = None


def load():
    global SERVICE
    SERVICE = SolNavRouteService(html_path=HTML_PATH, csv_path=CSV_PATH)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            with open(os.path.join(HERE, "route_ui.html"), "rb") as f:
                ui = f.read()
            html = SERVICE.html
            i = html.rfind(b"</html>")
            i = len(html) if i == -1 else i
            return self._send(200, html[:i] + ui + html[i:], "text/html; charset=utf-8")
        if u.path == "/api/routes":
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                o = (float(q["olon"]), float(q["olat"]))
                d = (float(q["dlon"]), float(q["dlat"]))
                with _lock:
                    res = SERVICE.compute_routes(
                        o, d, SERVICE.parse_time(q.get("t", "")))
            except (KeyError, ValueError):
                res = {"error": "Bad request."}
            except Exception as e:
                res = {"error": f"Server error: {e}"}
            return self._send(200, json.dumps(res).encode(), "application/json")
        self._send(404, b"not found", "text/plain")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    if not os.path.isfile(HTML_PATH):
        sys.exit(f"HTML not found: {HTML_PATH}")
    load()
    print(f"SolNav ready -> http://localhost:{PORT}")
    try:
        ThreadingHTTPServer(("localhost", PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
