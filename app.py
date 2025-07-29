import streamlit as st
import pandas as pd
from utils import get_summary_metrics, visualize_route_map, generate_compliance_table,predict_delays,train_delay_model
import plotly.express as px

st.set_page_config(page_title="FedEx Route Compliance Dashboard", layout="wide")
st.title("📦 FedEx Route Compliance Dashboard")

st.sidebar.header("Upload CSV Files")
planned_file = st.sidebar.file_uploader("📋 Planned Routes", type="csv")
actual_file = st.sidebar.file_uploader("📍 Actual Routes", type="csv")

if planned_file and actual_file:
    planned_df = pd.read_csv(planned_file, parse_dates=["Expected Time"])
    actual_df = pd.read_csv(actual_file, parse_dates=["Actual Time"])

    route_ids = planned_df["Route ID"].unique()
    selected_route = st.sidebar.selectbox("Select Route", route_ids)

    planned_route = planned_df[planned_df["Route ID"] == selected_route].sort_values("Expected Time")
    actual_route = actual_df[actual_df["Route ID"] == selected_route].sort_values("Actual Time")

    st.subheader("📊 Route Metrics")
    metrics = get_summary_metrics(planned_route, actual_route)

    col1, col2, col3 = st.columns(3)
    col1.metric("Compliance %", f"{metrics['compliance']}%")
    col2.metric("Extra Distance", f"{metrics['extra_distance_km']:.2f} km")
    col3.metric("Avg. Delay", f"{metrics['avg_delay_min']:.1f} min")


    st.subheader("🗺️ Planned vs Actual Route")
    try:
        st.write("Planned Route Columns:", planned_route.columns.tolist())
        st.write("Actual Route Columns:", actual_route.columns.tolist())
        visualize_route_map(planned_route, actual_route)
    except Exception as e:
        st.error(f"Error displaying map: {e}")

    st.subheader("📋 Stop-wise Deviation Report")
    deviation_df = generate_compliance_table(planned_route, actual_route)

    st.subheader("Filter Deviations")
    min_delay = st.number_input("Minimum Delay (min)", value=0)
    show_skipped = st.checkbox("Show Skipped Stops", value=True)
    show_extra = st.checkbox("Show Extra Stops", value=True)
    show_out_of_order = st.checkbox("Show Out of Order Stops", value=True)

    filtered_df = deviation_df.copy()
    if min_delay > 0:
        filtered_df = filtered_df[filtered_df["Delay (min)"].notnull() & (filtered_df["Delay (min)"] >= min_delay)]


    st.dataframe(filtered_df)

    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Report", csv, f"{selected_route}_report.csv", "text/csv")

    st.subheader("⏰ Delivery Time Deviation Histogram")
    fig = px.histogram(filtered_df[filtered_df["Delay (min)"].notnull()], x="Delay (min)", nbins=30, title="Delivery Delays")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔮 Predicted Delays")
    model = train_delay_model(planned_df,actual_df)
    predicted_delays = predict_delays(planned_route,model)
    st.dataframe(predicted_delays)

else:
    st.info("Upload both the planned and actual route CSV files to continue.")


