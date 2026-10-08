"""Streamlit interface for SolNav route planning."""
from datetime import time

import pydeck as pdk
import streamlit as st
import streamlit.components.v1 as components

from solnav_service import SolNavRouteService


st.set_page_config(page_title="SolNav", page_icon="🚶", layout="wide")


@st.cache_resource(show_spinner="Loading Kuala Lumpur roads and map geometry...")
def get_service():
    """Load the graph/model once per Streamlit process."""
    return SolNavRouteService()


st.title("SolNav — Kuala Lumpur walking routes")
st.caption("Compare the fastest, balanced, and coolest (shadiest) walking routes.")

with st.form("route_form"):
    st.subheader("Enter route coordinates")
    st.caption("Use longitude and latitude in decimal degrees. The map below is centered on Kuala Lumpur.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Origin**")
        origin_lon = st.number_input("Origin longitude", value=101.7000, format="%.6f")
        origin_lat = st.number_input("Origin latitude", value=3.1500, format="%.6f")
    with right:
        st.markdown("**Destination**")
        destination_lon = st.number_input("Destination longitude", value=101.7100, format="%.6f")
        destination_lat = st.number_input("Destination latitude", value=3.1600, format="%.6f")
    departure_time = st.time_input("Departure time (Kuala Lumpur)", value=time(12, 0))
    submitted = st.form_submit_button("Find walking routes", type="primary")

if submitted:
    try:
        service = get_service()
        dt_utc = service.parse_time(departure_time.strftime("%H:%M"))
        result = service.compute_routes(
            origin=(origin_lon, origin_lat),
            destination=(destination_lon, destination_lat),
            dt_utc=dt_utc,
        )
        st.session_state["route_result"] = result
        st.session_state["route_endpoints"] = {
            "origin": [origin_lon, origin_lat],
            "destination": [destination_lon, destination_lat],
        }
    except Exception as exc:
        st.error(f"Could not calculate routes: {exc}")

result = st.session_state.get("route_result")
if result:
    if "error" in result:
        st.error(result["error"])
    else:
        route_data = result["routes"]
        st.subheader("Route options")
        columns = st.columns(len(route_data))
        labels = {"fastest": "Fastest", "average": "Balanced", "coolest": "Coolest"}
        for column, (key, route) in zip(columns, route_data.items()):
            with column:
                st.markdown(f"**{labels.get(key, key.title())}**")
                st.metric("Estimated time", f"{route['eta_min']:.1f} min")
                st.write(f"Distance: {route['distance_m']:,} m")
                st.write(f"Shaded: {route['shade_pct']}%")
                st.write(f"Sun exposure: {route['sun_min']:.1f} min")
                st.caption(f"ETA source: {route['eta_source']}")

        st.subheader("Routes on the map")
        palette = {
            "fastest": [35, 140, 255, 220],
            "average": [255, 174, 0, 220],
            "coolest": [34, 190, 120, 230],
        }
        paths = [
            {"name": key, "path": route["coordinates"], "color": palette.get(key, [130, 90, 210, 220])}
            for key, route in route_data.items()
        ]
        endpoints = st.session_state["route_endpoints"]
        markers = [
            {"name": "Origin", "position": endpoints["origin"], "color": [30, 190, 90, 255]},
            {"name": "Destination", "position": endpoints["destination"], "color": [235, 65, 65, 255]},
        ]
        center_lon = (origin_lon + destination_lon) / 2
        center_lat = (origin_lat + destination_lat) / 2
        deck = pdk.Deck(
            layers=[
                pdk.Layer(
                    "PathLayer", data=paths, get_path="path", get_color="color",
                    get_width=7, width_min_pixels=3, pickable=True,
                ),
                pdk.Layer(
                    "ScatterplotLayer", data=markers, get_position="position",
                    get_fill_color="color", get_radius=70, radius_min_pixels=7,
                    pickable=True,
                ),
            ],
            initial_view_state=pdk.ViewState(
                latitude=center_lat, longitude=center_lon, zoom=13, pitch=35,
            ),
            tooltip={"text": "{name}"},
            map_style="light",
        )
        st.pydeck_chart(deck, use_container_width=True)
        st.caption("Blue: fastest · Gold: balanced · Green: coolest · Green/red dots: origin/destination")
        st.caption(
            f"Sun altitude: {result['sun']['altitude_deg']}° · "
            f"Azimuth: {result['sun']['azimuth_deg']}°"
            + (" · It is night, so there is no direct sunlight." if result["night"] else "")
        )

with st.expander("View the original 3D Kuala Lumpur map"):
    st.caption("This is the existing 3D map. Route inputs above calculate routes through the shared Python service.")
    try:
        service = get_service()
        components.html(service.html.decode("utf-8"), height=680, scrolling=False)
    except Exception as exc:
        st.error(f"Could not load the 3D map: {exc}")
