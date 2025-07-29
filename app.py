
import streamlit as st
import pandas as pd
from utils import get_summary_metrics, visualize_route_map, generate_compliance_table
import plotly.express as px

st.set_page_config(page_title="FedEx Route Compliance Dashboard", layout="wide")
st.title("📦 FedEx Route Compliance Dashboard")

# Upload section
st.sidebar.header("Upload CSV Files")
planned_file = st.sidebar.file_uploader("📋 Planned Routes", type="csv")
actual_file = st.sidebar.file_uploader("📍 Actual Routes", type="csv")

if planned_file and actual_file:
    planned_df = pd.read_csv(planned_file, parse_dates=["Expected Time"])
    actual_df = pd.read_csv(actual_file, parse_dates=["Actual Time"])

    route_ids = planned_df["Route ID"].unique()
    selected_route = st.sidebar.selectbox("Select Route", route_ids)

    # Filter selected route
    planned_route = planned_df[planned_df["Route ID"] == selected_route].sort_values("Expected Time")
    actual_route = actual_df[actual_df["Route ID"] == selected_route].sort_values("Actual Time")

    # Compliance KPIs
    st.subheader("📊 Route Metrics")
    metrics = get_summary_metrics(planned_route, actual_route)

    col1, col2, col3 = st.columns(3)
    col1.metric("Compliance %", f"{metrics['compliance']}%")
    col2.metric("Extra Distance", f"{metrics['extra_distance_km']:.2f} km")
    col3.metric("Avg. Delay", f"{metrics['avg_delay_min']:.1f} min")

    # Map visualization
    st.subheader("🗺️ Planned vs Actual Route")
    visualize_route_map(planned_route, actual_route)

    # Deviation table
    st.subheader("📋 Stop-wise Deviation Report")
    deviation_df = generate_compliance_table(planned_route, actual_route)
    st.dataframe(deviation_df)

    csv = deviation_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Report", csv, f"{selected_route}_report.csv", "text/csv")

    # Plot delay histogram
    st.subheader("⏰ Delivery Time Deviation Histogram")
    fig = px.histogram(deviation_df[deviation_df["Delay (min)"].notnull()], x="Delay (min)", nbins=30, title="Delivery Delays")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Upload both the planned and actual route CSV files to continue.")
