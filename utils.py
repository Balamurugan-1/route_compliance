import pandas as pd
from math import radians, cos, sin, asin, sqrt
import folium
from streamlit_folium import st_folium

def haversine(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6371 * c
    return km

def total_distance(coords):
    return sum(haversine(coords[i], coords[i+1]) for i in range(len(coords) - 1))

# In utils.py
def get_summary_metrics(planned_df, actual_df):
    required_cols = ["Route ID", "Stop ID", "Latitude", "Longitude"]
    time_cols = ["Expected Time", "Actual Time"]
    
    missing_planned_cols = [col for col in required_cols + ["Expected Time"] if col not in planned_df.columns]
    missing_actual_cols = [col for col in required_cols + ["Actual Time"] if col not in actual_df.columns]
    
    if missing_planned_cols:
        st.error(f"Error: Missing columns in planned_df: {missing_planned_cols}")
        return {"compliance": 0.0, "extra_distance_km": 0.0, "avg_delay_min": 0.0}
    if missing_actual_cols:
        st.error(f"Error: Missing columns in actual_df: {missing_actual_cols}")
        return {"compliance": 0.0, "extra_distance_km": 0.0, "avg_delay_min": 0.0}
    
    if planned_df.empty or actual_df.empty:
        st.error("Error: One or both input DataFrames are empty.")
        return {"compliance": 0.0, "extra_distance_km": 0.0, "avg_delay_min": 0.0}

    planned_coords = list(zip(planned_df["Latitude"], planned_df["Longitude"]))
    actual_coords = list(zip(actual_df["Latitude"], actual_df["Longitude"]))
    planned_dist = total_distance(planned_coords) if planned_coords else 0.0
    actual_dist = total_distance(actual_coords) if actual_coords else 0.0

    merged_df = pd.merge(
        planned_df[["Stop ID", "Expected Time"]],
        actual_df[["Stop ID", "Actual Time"]],
        on="Stop ID",
        how="inner"
    )

    if merged_df.empty:
        st.warning("Warning: No common stops between planned and actual routes. Compliance set to 0.")
        return {"compliance": 0.0, "extra_distance_km": actual_dist - planned_dist, "avg_delay_min": 0.0}

    try:
        delay_series = (
            (merged_df["Actual Time"] - merged_df["Expected Time"])
            .dt.total_seconds() // 60
        ).astype("Int64")
    except Exception as e:
        st.error(f"Error calculating delays: {e}")
        return {"compliance": 0.0, "extra_distance_km": actual_dist - planned_dist, "avg_delay_min": 0.0}

    delay_series_clean = delay_series.dropna()
    if len(delay_series_clean) > 0:
        compliance = 100 * (sum(abs(delay_series_clean) <= 5) / len(delay_series_clean))
        avg_delay = delay_series_clean.mean()
    else:
        st.warning("Warning: No valid delay data after cleaning. Compliance set to 0.")
        compliance = 0.0
        avg_delay = 0.0

    return {
        "compliance": round(compliance, 2),
        "extra_distance_km": round(actual_dist - planned_dist, 2),
        "avg_delay_min": round(avg_delay, 1)
    }

def generate_compliance_table(planned_df, actual_df):
    planned_df = planned_df.sort_values("Expected Time")
    planned_df["Planned Sequence"] = range(1, len(planned_df) + 1)
    
    df = pd.merge(planned_df, actual_df, on="Stop ID", how="outer", suffixes=("_planned", "_actual"))
    
    df["Delay (min)"] = (
        (df["Actual Time"] - df["Expected Time"]).dt.total_seconds() // 60
    ).astype("Int64")
    
    df["Deviation"] = df["Delay (min)"].apply(lambda x: "Yes" if pd.notnull(x) and abs(x) > 5 else "No")
    
    df["Skipped"] = df["Actual Time"].isnull()
    
    df["Extra"] = df["Expected Time"].isnull()
    
    common_stops = df[~df["Skipped"] & ~df["Extra"]]["Stop ID"].tolist()
    
    planned_common = df[df["Stop ID"].isin(common_stops)].sort_values("Expected Time")
    planned_positions = {row["Stop ID"]: idx+1 for idx, row in planned_common.iterrows()}
    
    actual_common = df[df["Stop ID"].isin(common_stops)].sort_values("Actual Time")
    actual_positions = {row["Stop ID"]: idx+1 for idx, row in actual_common.iterrows()}
    
    df["Planned Position"] = df["Stop ID"].map(planned_positions).fillna(0).astype(int)
    df["Actual Position"] = df["Stop ID"].map(actual_positions).fillna(0).astype(int)
    
    df["Out of Order"] = (df["Planned Position"] > 0) & (df["Actual Position"] > 0) & (df["Planned Position"] != df["Actual Position"])
    
    df = df.sort_values("Actual Time")
    distances = [0.0]
    for i in range(1, len(df)):
        if pd.notnull(df.iloc[i]["Latitude_actual"]) and pd.notnull(df.iloc[i-1]["Latitude_actual"]):
            prev = (df.iloc[i-1]["Latitude_actual"], df.iloc[i-1]["Longitude_actual"])
            curr = (df.iloc[i]["Latitude_actual"], df.iloc[i]["Longitude_actual"])
            distances.append(haversine(prev, curr))
        else:
            distances.append(0.0)
    df["Distance (km)"] = distances
    total_dist = sum(df["Distance (km)"])
    df["Fuel %"] = [ (d / total_dist) * 100 if total_dist > 0 else 0 for d in df["Distance (km)"] ]
    
    return df[[
        "Stop ID", "Expected Time", "Actual Time", "Delay (min)", "Deviation",
        "Skipped", "Extra", "Out of Order", "Planned Position", "Actual Position",
        "Distance (km)", "Fuel %"
    ]]

import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np

def visualize_route_map(planned_df, actual_df):
    required_cols = ["Route ID", "Stop ID", "Latitude", "Longitude"]
    missing_planned_cols = [col for col in required_cols if col not in planned_df.columns]
    missing_actual_cols = [col for col in required_cols if col not in actual_df.columns]
    
    if missing_planned_cols:
        st.error(f"Error: Missing columns in planned_df: {missing_planned_cols}")
        return None
    if missing_actual_cols:
        st.error(f"Error: Missing columns in actual_df: {missing_actual_cols}")
        return None
    if planned_df.empty or actual_df.empty:
        st.error("Error: One or both input DataFrames are empty.")
        return None

    planned_df = planned_df.dropna(subset=["Latitude", "Longitude"]).copy()
    actual_df = actual_df.dropna(subset=["Latitude", "Longitude"]).copy()
    
    if planned_df.empty and actual_df.empty:
        st.error("Error: No valid coordinates available to display.")
        return None

    all_coords = pd.concat([
        planned_df[["Latitude", "Longitude"]],
        actual_df[["Latitude", "Longitude"]]
    ])
    
    center_lat = all_coords["Latitude"].mean()
    center_lon = all_coords["Longitude"].mean()
    if pd.isna(center_lat) or pd.isna(center_lon):
        st.error("Error: Invalid coordinates for map centering.")
        return None
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    if not planned_df.empty:
        folium.PolyLine(
            locations=list(zip(planned_df["Latitude"], planned_df["Longitude"])),
            color="green",
            weight=2,
            tooltip="Planned Route"
        ).add_to(m)

    if not actual_df.empty:
        folium.PolyLine(
            locations=list(zip(actual_df["Latitude"], actual_df["Longitude"])),
            color="red",
            weight=2,
            tooltip="Actual Route"
        ).add_to(m)

    try:
        deviation_df = generate_compliance_table(planned_df, actual_df)
        planned_merged = planned_df.merge(deviation_df, on="Stop ID", how="left")
        for _, row in planned_merged.iterrows():
            if pd.notnull(row["Latitude"]) and pd.notnull(row["Longitude"]):
                status = "Skipped" if row.get("Skipped", False) else \
                         "Out of Order" if row.get("Out of Order", False) else "Normal"
                delay = f"{int(row['Delay (min)'])} min" if pd.notnull(row.get("Delay (min)")) else "N/A"
                popup = f"Stop: {row['Stop ID']}<br>Delay: {delay}<br>Status: {status}"
                folium.Marker(
                    location=[row["Latitude"], row["Longitude"]],
                    popup=popup,
                    icon=folium.Icon(color="green" if status == "Normal" else "orange", icon="circle")
                ).add_to(m)
        
        actual_merged = actual_df.merge(deviation_df, on="Stop ID", how="left")
        for _, row in actual_merged.iterrows():
            if pd.notnull(row["Latitude"]) and pd.notnull(row["Longitude"]) and row.get("Extra", False):
                popup = f"Stop: {row['Stop ID']}<br>Status: Extra"
                folium.Marker(
                    location=[row["Latitude"], row["Longitude"]],
                    popup=popup,
                    icon=folium.Icon(color="red", icon="circle")
                ).add_to(m)
    except Exception as e:
        st.warning(f"Warning: Could not add markers due to error: {e}. Displaying routes only.")

    if not all_coords.empty:
        sw = [all_coords["Latitude"].min(), all_coords["Longitude"].min()]
        ne = [all_coords["Latitude"].max(), all_coords["Longitude"].max()]
        m.fit_bounds([sw, ne])

    try:
        st_folium(m, width=800, height=500)
    except Exception as e:
        st.error(f"Error rendering map: {e}")
        return None



from sklearn.ensemble import RandomForestRegressor
import pickle
import pandas as pd

def train_delay_model(planned_df, actual_df):
    df = pd.merge(planned_df, actual_df, on=["Route ID", "Stop ID"], how="inner", suffixes=("_planned", "_actual"))
    df["Delay (min)"] = (df["Actual Time"] - df["Expected Time"]).dt.total_seconds() // 60
    
    df["Hour"] = df["Expected Time"].dt.hour
    df["DayOfWeek"] = df["Expected Time"].dt.dayofweek
    df["DistanceToPrev"] = [0] + [haversine(
        (df.iloc[i-1]["Latitude_actual"], df.iloc[i-1]["Longitude_actual"]),
        (df.iloc[i]["Latitude_actual"], df.iloc[i]["Longitude_actual"])
    ) for i in range(1, len(df))]

    X = df[["Hour", "DayOfWeek", "DistanceToPrev", "Latitude_planned", "Longitude_planned"]]
    y = df["Delay (min)"].astype(float)  # Ensure target is float for regression
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    with open("delay_model.pkl", "wb") as f:
        pickle.dump(model, f)
    return model


def predict_delays(planned_df, model):
    df = planned_df.copy()
    df["Hour"] = df["Expected Time"].dt.hour
    df["DayOfWeek"] = df["Expected Time"].dt.dayofweek
    df["DistanceToPrev"] = [0] + [haversine(
        (df.iloc[i-1]["Latitude"], df.iloc[i-1]["Longitude"]),
        (df.iloc[i]["Latitude"], df.iloc[i]["Longitude"])
    ) if pd.notnull(df.iloc[i-1]["Latitude"]) and pd.notnull(df.iloc[i]["Latitude"]) else 0
    for i in range(1, len(df))]
    
    df = df.rename(columns={"Latitude": "Latitude_planned", "Longitude": "Longitude_planned"})
    
    X = df[["Hour", "DayOfWeek", "DistanceToPrev", "Latitude_planned", "Longitude_planned"]]
    predictions = model.predict(X)
    df["Predicted Delay (min)"] = predictions
    return df[["Stop ID", "Predicted Delay (min)"]]