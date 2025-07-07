import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def load_center():
    df = pd.read_csv("df_clean.csv")
    center_lat = df['lat'].mean()
    center_lon = df['lon'].mean()
    return center_lat, center_lon

def prepare_input(lat, lon, hour):
    center_lat, center_lon = load_center()
    return pd.DataFrame([{
        'dist_from_center': np.sqrt((lat - center_lat)**2 + (lon - center_lon)**2),
        'angle_from_center': np.arctan2(lat - center_lat, lon - center_lon),
        'hour_sin': np.sin(2 * np.pi * hour / 24),
        'hour_cos': np.cos(2 * np.pi * hour / 24),
        'hour': hour
    }])
