import joblib
import streamlit as st

@st.cache_resource
def load_models():
    lat_model = joblib.load("grid_lat_model.pkl")
    lon_model = joblib.load("grid_lon_model.pkl")
    return lat_model, lon_model
