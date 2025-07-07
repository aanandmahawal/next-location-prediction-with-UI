# app.py
import streamlit as st
import folium
from streamlit_folium import st_folium
from model import load_models
from utils import prepare_input
from folium.plugins import PolyLineTextPath

# Page config
st.set_page_config(page_title="Geolife Next Location Predictor", layout="centered")

# Remove top margin via CSS
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown(
    "<h1 style='text-align: center; margin-bottom: 10px;'>📡 Predict Your Next Location</h1>",
    unsafe_allow_html=True,
)

# Session state to avoid rerun issues
if "predicted" not in st.session_state:
    st.session_state.predicted = False
    st.session_state.result = {}

# Input section
st.markdown("### 🧭 Enter Current Location and Time")
lat = st.number_input("Current Latitude", value=39.9847, format="%.6f")
lon = st.number_input("Current Longitude", value=116.3184, format="%.6f")
hour = st.slider("Current Hour (0–23)", 0, 23, 14)

# Prediction button
if st.button("🔮 Predict Next Location"):
    input_df = prepare_input(lat, lon, hour)
    lat_model, lon_model = load_models()

    delta_lat = lat_model.predict(input_df)[0]
    delta_lon = lon_model.predict(input_df)[0]
    pred_lat = lat + delta_lat
    pred_lon = lon + delta_lon

    st.session_state.predicted = True
    st.session_state.result = {
        "lat": lat,
        "lon": lon,
        "pred_lat": pred_lat,
        "pred_lon": pred_lon,
    }

# Output section
if st.session_state.predicted:
    res = st.session_state.result
    st.success("✅ Prediction Complete!")
    st.markdown(f"""
    **Current Location:**  
    Latitude: `{res["lat"]}`  
    Longitude: `{res["lon"]}`

    **Predicted Next Location:**  
    Latitude: `{res["pred_lat"]:.6f}`  
    Longitude: `{res["pred_lon"]:.6f}`
    """)

    # Create Map
    m = folium.Map(location=[res["lat"], res["lon"]], zoom_start=15)

    # Add markers
    folium.Marker(
        [res["lat"], res["lon"]],
        tooltip="Current Location",
        popup="📍 Current Location",
        icon=folium.Icon(color='blue')
    ).add_to(m)

    folium.Marker(
        [res["pred_lat"], res["pred_lon"]],
        tooltip="Predicted Next Location",
        popup="🎯 Predicted Location",
        icon=folium.Icon(color='green')
    ).add_to(m)

    # Draw arrow between current and predicted location
    arrow_line = folium.PolyLine(
        locations=[[res["lat"], res["lon"]], [res["pred_lat"], res["pred_lon"]]],
        color="red",
        weight=2.5
    ).add_to(m)

    # Add arrowhead using PolyLineTextPath
    PolyLineTextPath(
        arrow_line,
        '➤',
        repeat=True,
        offset=7,
        attributes={'fill': 'red', 'font-weight': 'bold', 'font-size': '24'}
    ).add_to(m)

    st.markdown("### 🗺️ Movement Map with Direction")
    st_folium(m, width=700, height=500)
