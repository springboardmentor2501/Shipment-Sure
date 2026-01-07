import numpy as np
import pandas as pd
from datetime import datetime

import streamlit as st
import pandas as pd
import joblib

def build_features(raw_df):
    df = raw_df.copy()

    # ---- Time features (use current time for live prediction) ----
    now = datetime.now()
    df["month"] = now.month
    df["day"] = now.day
    df["weekday"] = now.weekday()
    df["hour"] = now.hour

    for col, maxv in [("month",12), ("day",31), ("weekday",7), ("hour",24)]:
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / maxv)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / maxv)

    # ---- Risk & environment ----
    df["risk_high_traffic"] = (df["Traffic_Status"].isin(["Heavy", "Detour"])).astype(int)
    df["risk_delay_reason"] = df["Logistics_Delay_Reason"].notna().astype(int)

    df["temp_deviation"] = abs(df["Temperature"] - 22)
    df["humidity_deviation"] = abs(df["Humidity"] - 60)

    # ---- Traffic severity ----
    traffic_map = {"Clear": 0, "Heavy": 1, "Detour": 2}
    df["traffic_severity"] = df["Traffic_Status"].map(traffic_map)

    # ---- Geo features ----
    df["abs_lat"] = abs(df["Latitude"])
    df["abs_long"] = abs(df["Longitude"])
    df["geo_distance_proxy"] = np.sqrt(df["Latitude"]**2 + df["Longitude"]**2)

    # ---- Interactions ----
    df["util_demand"] = df["Asset_Utilization"] * df["Demand_Forecast"]
    df["inv_wait"] = df["Inventory_Level"] * df["Waiting_Time"]
    df["wait_per_traffic"] = df["Waiting_Time"] * df["risk_high_traffic"]

    # ---- Customer importance ----
    df["high_value_customer"] = (
        (df["User_Transaction_Amount"] > 0) &
        (df["User_Purchase_Frequency"] > 0)
    ).astype(int)

    # ---- Asset & delay proxies (unknown at inference → neutral defaults) ----
    df["asset_delay_rate"] = 0.7
    df["has_delay_record"] = 1

    return df

# Load trained model
model = joblib.load("shipment_delay_model.pkl")

st.title("🚚 Shipment On-Time Delivery Predictor")

st.write("Enter order and logistics details to predict delivery probability.")

# ---- Input fields ----
Inventory_Level = st.number_input("Inventory Level", 0.0, 1000.0)
Waiting_Time = st.number_input("Waiting Time (hrs)", 0.0, 100.0)
Temperature = st.number_input("Temperature (°C)", -10.0, 50.0)
Humidity = st.number_input("Humidity (%)", 0.0, 100.0)
Traffic_Status = st.selectbox("Traffic Status", ["Clear", "Heavy", "Detour"])
Asset_Utilization = st.slider("Asset Utilization", 0.0, 1.0)
Demand_Forecast = st.number_input("Demand Forecast", 0.0, 1000.0)

# ---- Create input dataframe ----
raw_input = pd.DataFrame([{
    "Inventory_Level": Inventory_Level,
    "Waiting_Time": Waiting_Time,
    "Temperature": Temperature,
    "Humidity": Humidity,
    "Traffic_Status": Traffic_Status,
    "Asset_Utilization": Asset_Utilization,
    "Demand_Forecast": Demand_Forecast,

    # Defaults for features not known at order time
    "Latitude": 0.0,
    "Longitude": 0.0,
    "User_Transaction_Amount": 0.0,
    "User_Purchase_Frequency": 0.0,
    "Logistics_Delay_Reason": None
}])

input_df = build_features(raw_input)



# ---- Prediction ----
if st.button("Predict"):

    input_df = build_features(raw_input)
    raw_prob = model.predict_proba(input_df)[0][1]

    # ---- INITIALIZE risk_score FIRST ----
    risk_score = 0

    # ---- Risk scoring ----
    if Waiting_Time > 36:
        risk_score += 3
    elif Waiting_Time > 24:
        risk_score += 1

    if Temperature > 40:
        risk_score += 2

    if Humidity > 90:
        risk_score += 2

    if Traffic_Status == "Heavy":
        risk_score += 2

    if Asset_Utilization > 0.85:
        risk_score += 2
    elif Asset_Utilization > 0.65:
        risk_score += 1

    # ---- Probability mapping ----
    if risk_score <= 2:
        prob = 0.75
    elif risk_score <= 4:
        prob = 0.50
    elif risk_score <= 6:
        prob = 0.25
    else:
        prob = 0.10

    # ---- Label (ALWAYS initialize) ----
    likelihood_label = "High delay risk"
    if prob >= 0.7:
        likelihood_label = "High likelihood"
    elif prob >= 0.3:
        likelihood_label = "Moderate likelihood"

    st.metric(
        label="On-Time Delivery Probability",
        value=f"{prob*100:.2f}%",
        delta=likelihood_label
    )