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

def get_summary_metrics(planned_df, actual_df):
    planned_coords = list(zip(planned_df["Latitude"], planned_df["Longitude"]))
    actual_coords = list(zip(actual_df["Latitude"], actual_df["Longitude"]))

    planned_dist = total_distance(planned_coords)
    actual_dist = total_distance(actual_coords)
    delay_series = (
        (actual_df["Actual Time"] - planned_df["Expected Time"])
        .dt.total_seconds() // 60
    ).astype("Int64")

    delay_series_clean = delay_series.dropna()
    if len(delay_series_clean) > 0:
        compliance = 100 - (sum(abs(delay_series_clean) > 5) / len(delay_series_clean)) * 100
        avg_delay = delay_series_clean.mean()
    else:
        compliance = 0.0
        avg_delay = 0.0

    return {
        "compliance": round(compliance, 2),
        "extra_distance_km": actual_dist - planned_dist,
        "avg_delay_min": avg_delay
    }

def generate_compliance_table(planned_df, actual_df):
    # Add Planned Sequence to planned_df
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

def visualize_route_map(planned_df, actual_df):
    start_coords = [planned_df.iloc[0]["Latitude"], planned_df.iloc[0]["Longitude"]]
    m = folium.Map(location=start_coords, zoom_start=13)

    folium.PolyLine(
        locations=list(zip(planned_df["Latitude"], planned_df["Longitude"])),
        color="green", weight=5, tooltip="Planned Route"
    ).add_to(m)

    folium.PolyLine(
        locations=list(zip(actual_df["Latitude"], actual_df["Longitude"])),
        color="red", weight=5, tooltip="Actual Route"
    ).add_to(m)

    st_folium(m, width=800, height=500)




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