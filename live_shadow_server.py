#!/usr/bin/env python3
"""
live_shadow_server.py
----------------------
Runs a local WebSocket server that pushes real, physically-computed sun
position and shadow geometry to your already-open HTML map every sync
interval -- the browser tab updates live, no reload needed.

The actual solar-position and shadow-projection math lives in
shadow_math.py (shared with shade_routing.py's Stage 2 routing) -- this
file only handles the WebSocket broadcasting loop.

SETUP
-----
1. Install the one dependency:
       pip install websockets

2. Add the small WebSocket client block to your HTML (see
   live_shadow_client_snippet.js in this same folder -- or use the
   solnav_3d_verified_map.html I've already wired up with it).

3. Run this server:
       python live_shadow_server.py "C:\\Users\\User\\OneDrive\\Documents\\SolNav\\solnav_3d_verified_map.html"

   The HTML path argument is only used to read the current building/road/
   pillar/tree geometry (so shadows are cast from real structures) --
   it does NOT rewrite that file. Leave the server running, open the HTML
   in your browser, and shadows will update live every sync interval.

4. Stop with Ctrl+C.
"""

import sys
import os
import json
import asyncio
from datetime import datetime, timezone

try:
    import websockets
except ImportError:
    print("Missing dependency. Install it with:  pip install websockets")
    sys.exit(1)

from shadow_math import get_solar_position, load_source_geometry, build_shadows, KL_LAT, KL_LON

SYNC_INTERVAL_SECONDS = 1 * 60

HOST = "localhost"
PORT = 8765


# ---------------------------------------------------------------------------
CONNECTED_CLIENTS = set()


async def handle_client(websocket):
    CONNECTED_CLIENTS.add(websocket)
    print(f"Client connected ({len(CONNECTED_CLIENTS)} total)")
    try:
        # send the latest snapshot immediately on connect
        if LATEST_PAYLOAD is not None:
            await websocket.send(LATEST_PAYLOAD)
        async for _ in websocket:
            pass  # this server is push-only; ignore any client messages
    finally:
        CONNECTED_CLIENTS.discard(websocket)
        print(f"Client disconnected ({len(CONNECTED_CLIENTS)} total)")


LATEST_PAYLOAD = None


async def broadcast_loop(html_path):
    global LATEST_PAYLOAD
    while True:
        now_utc = datetime.now(timezone.utc)
        sun = get_solar_position(now_utc, KL_LAT, KL_LON)
        print(f"[{now_utc.isoformat()}] Sun altitude={sun['altitude_deg']:.1f} deg, "
              f"azimuth={sun['azimuth_deg']:.1f} deg")

        if sun['altitude_deg'] <= 0.5:
            print("  Sun below/near horizon -- clearing shadows (nighttime).")
            shadows = []
        else:
            geometry = load_source_geometry(html_path)
            shadows = build_shadows(geometry, sun)

        payload = json.dumps({
            "type": "shadow_update",
            "data": shadows,
            "sun": {
                "altitude_deg": sun["altitude_deg"],
                "azimuth_deg": sun["azimuth_deg"],
                "timestamp": now_utc.isoformat(),
            },
        })
        LATEST_PAYLOAD = payload

        if CONNECTED_CLIENTS:
            await asyncio.gather(*[c.send(payload) for c in list(CONNECTED_CLIENTS)],
                                 return_exceptions=True)
            print(f"  Broadcast {len(shadows)} shadow polygons to "
                  f"{len(CONNECTED_CLIENTS)} client(s).")
        else:
            print(
                f"  Computed {len(shadows)} shadow polygons (no clients connected yet).")

        await asyncio.sleep(SYNC_INTERVAL_SECONDS)


async def main_async(html_path):
    # default max_size is 1MB, which is too small for a full shadow payload
    # (tens of thousands of polygons can easily exceed a few MB)
    async with websockets.serve(handle_client, HOST, PORT, max_size=32 * 1024 * 1024):
        print(f"Live shadow server running at ws://{HOST}:{PORT}")
        print(f"Reading geometry from: {html_path}")
        print(
            f"Syncing every {SYNC_INTERVAL_SECONDS // 60} minute(s). Ctrl+C to stop.\n")
        await broadcast_loop(html_path)


def main():
    if len(sys.argv) < 2:
        print("Usage: python live_shadow_server.py path\\to\\solnav_3d_verified_map.html")
        sys.exit(1)
    html_path = sys.argv[1]

    if not os.path.isfile(html_path):
        print(f"ERROR: file not found: {html_path}")
        # Doubled backslashes ("\\\\") are a common copy-paste mistake --
        # e.g. pasting a Python-escaped path like "C:\\Users\\..." straight
        # into PowerShell, where backslash isn't a special character so it
        # never collapses to a single backslash. Detect it and suggest the
        # fix instead of just failing.
        if "\\\\" in html_path:
            suggested = html_path.replace("\\\\", "\\")
            print("This looks like a doubled-backslash path (a common copy-paste")
            print("issue from an escaped Python string). Did you mean:")
            print(f"    {suggested}")
        sys.exit(1)

    try:
        asyncio.run(main_async(html_path))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == '__main__':
    main()
