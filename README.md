SolNav is a pedestrian map and routing prototype for Kuala Lumpur. It runs a small local server that puts a route planner on top of a 3D map and gives you three walking options: fastest, balanced, and coolest (most shaded). It can also work out sun shadows from the map's buildings and stream them live, and an optional TensorFlow model can predict walking ETAs.

## Files included: 
- `solnav_3d_verified_map.html`: the interactive 3D map and its geometry (big, about 83 MB)
- `route_ui.html`: the route planner UI, injected into the map by the server
- `route_server.py`: local server for the map and the `/api/routes` endpoint
- `routing_graph.py`: builds the walking graph from the roads CSV and finds routes
- `shade_routing.py`: shade-aware routing, plus shade and ETA calculations
- `shadow_math.py`: sun position and shadow projection
- `live_shadow_server.py`: optional WebSocket server for live shadows
- `train_eta_st_gcn.py`: optional ETA model training and inference
- `data/`: roads, buildings, trees, trip ETAs, and `3D Rendering.py`
- `models/`: a trained ETA model with its config, scaler, and plots (if present)

## Setup:

You need Python 3.10+. Run everything from the project root. Routing and shadows only use the standard library, so you can skip installing anything unless you want the extras.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# Optional: live shadows
python -m pip install websockets

# Optional: ETA training and model inference
python -m pip install tensorflow numpy scikit-learn matplotlib
```

There's no `requirements.txt`. The groups above are just what the scripts import. Pick a TensorFlow build that supports your OS and Python version.

## Fix the filepaths first

A few scripts still have absolute paths from the original Windows checkout. If you're on another machine, update these:

- `route_server.py`: `csv_path` in `load()` should point to your roads CSV (the bundled `data/Roads.csv` is what it expects)
- `shade_routing.py`: `HTML_SOURCE_PATH` should point to your copy of the map HTML
- `train_eta_st_gcn.py`: `CSV`, `DEFAULT_TRIPS_CSV`, and `OUT` should point to your `data` and `models` folders

The roads file needs to be a QGIS-style CSV with WKT `LINESTRING` geometries and the usual OpenStreetMap road tags. The servers only read the map HTML; they never modify it.

## Run the route planner

```powershell
python route_server.py
# or pass the map explicitly:
python route_server.py .\solnav_3d_verified_map.html
```

Once the graph loads, open [http://localhost:8000](http://localhost:8000), enter a start and destination in KL (or use the location and pick-on-map controls), and compare the route cards. You can switch between overview and 3D views. Place search uses OpenStreetMap Nominatim, so you'll need internet for that. Stop the server with `Ctrl+C`.

## Live shadows (updated every minute)

In a second terminal:

```powershell
python live_shadow_server.py .\solnav_3d_verified_map.html
```

This serves `ws://localhost:8765` and sends a fresh shadow snapshot to the map every minute (an empty set at night). The map already includes the client, so just keep it running while you browse. `Ctrl+C` to stop.

## ETA model (optional)

With the ML packages installed and paths set, train on the bundled Google Maps trips:

```powershell
python train_eta_st_gcn.py --epochs 50
```

The default input is `data/Google_Maps_ETA_OCT25--26.csv`. A trip file needs `origin_lon`, `origin_lat`, `destination_lon`, `destination_lat`, and either `actual_eta_min` or `google_eta_min`. `departure_hour`, `weekend`, and `rain_flag` are optional.

```powershell
# Use your own trips
python train_eta_st_gcn.py --trips-csv .\data\my_trips.csv --epochs 50

# Train on SolNav's own ETA formula instead
python train_eta_st_gcn.py --simulated --simulated-trips 800
```

Run `python train_eta_st_gcn.py --help` for all options. Training saves weights, config, scaler, metrics, and plots to the `models` folder. The route server tries to load the model on startup; if TensorFlow or the model files are missing or incompatible, it logs a note and falls back to the built-in ETA formula.

## Other scripts

- `python routing_graph.py`: small fastest-route demo. It looks for `Road_KL_OSM.csv` next to the script, so edit `CSV_FILENAME` if needed.
- `python shade_routing.py`: fastest-vs-shaded demo. Set the paths above first.

## How routes are scored

The fastest route minimizes modeled walking time, including crossing and stair delays. Balanced and coolest routes weigh shade as well, using the sun's position and the geometry in the map. Each card shows distance, ETA, shaded share, and time spent in direct sun. ETAs are estimates, not live traffic data.