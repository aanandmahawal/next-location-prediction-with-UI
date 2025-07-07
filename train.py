import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Load CSV
df = pd.read_csv("df_clean.csv")

# Add some basic features (same as before)
df['dist_from_center'] = np.sqrt((df['lat'] - df['lat'].mean())**2 + (df['lon'] - df['lon'].mean())**2)
df['angle_from_center'] = np.arctan2(df['lat'] - df['lat'].mean(), df['lon'] - df['lon'].mean())
df['hour'] = 14  # dummy fixed hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Shift for next point prediction
df['target_lat'] = df['lat'].shift(-1)
df['target_lon'] = df['lon'].shift(-1)
df.dropna(inplace=True)

# Features and targets
X = df[['dist_from_center', 'angle_from_center', 'hour_sin', 'hour_cos', 'hour']]
y_lat = df['target_lat']
y_lon = df['target_lon']

X_train, _, y_lat_train, _ = train_test_split(X, y_lat, test_size=0.2, random_state=42)
_, _, y_lon_train, _ = train_test_split(X, y_lon, test_size=0.2, random_state=42)

# Train models
pipeline = make_pipeline(StandardScaler(), RandomForestRegressor(random_state=42))
pipeline.fit(X_train, y_lat_train)
joblib.dump(pipeline, "grid_lat_model.pkl")

pipeline.fit(X_train, y_lon_train)
joblib.dump(pipeline, "grid_lon_model.pkl")

print("✅ Models retrained and saved locally!")
