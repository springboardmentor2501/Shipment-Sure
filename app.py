import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("model_lr.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="ShipmentSure ML", layout="centered")

st.title("📦 ShipmentSure – On-Time Delivery Predictor")
st.write("Predict the probability of on-time delivery for a shipment.")


st.subheader("Enter Shipment Details")

supplier_rating = st.slider("Supplier Rating", 1.0, 5.0, 3.5)
supplier_lead_time = st.number_input("Supplier Lead Time (days)", 1, 30, 7)
shipping_distance_km = st.number_input("Shipping Distance (km)", 10, 3000, 500)
order_quantity = st.number_input("Order Quantity", 1, 500, 50)
unit_price = st.number_input("Unit Price", 10.0, 10000.0, 500.0)
total_order_value = order_quantity * unit_price
previous_on_time_rate = st.slider("Previous On-Time Rate (%)", 70, 100, 85)

shipment_mode = st.selectbox("Shipment Mode", ["Road", "Sea"])
weather = st.selectbox("Weather Condition", ["Clear", "Cloudy", "Rainy", "Storm"])
region = st.selectbox("Region", ["North", "South", "East", "West"])
holiday = st.selectbox("Holiday Period", ["Yes", "No"])


input_dict = {
    "supplier_rating": supplier_rating,
    "supplier_lead_time": supplier_lead_time,
    "shipping_distance_km": shipping_distance_km,
    "order_quantity": order_quantity,
    "unit_price": unit_price,
    "total_order_value": total_order_value,
    "previous_on_time_rate": previous_on_time_rate,
    "shipment_mode_Road": 1 if shipment_mode == "Road" else 0,
    "shipment_mode_Sea": 1 if shipment_mode == "Sea" else 0,
    "weather_condition_Cloudy": 1 if weather == "Cloudy" else 0,
    "weather_condition_Rainy": 1 if weather == "Rainy" else 0,
    "weather_condition_Storm": 1 if weather == "Storm" else 0,
    "region_North": 1 if region == "North" else 0,
    "region_South": 1 if region == "South" else 0,
    "region_East": 1 if region == "East" else 0,
    "region_West": 1 if region == "West" else 0,
    "holiday_period_Yes": 1 if holiday == "Yes" else 0
}

X = pd.DataFrame([input_dict])

# Align with training features
expected_features = scaler.feature_names_in_
X = X.reindex(columns=expected_features, fill_value=0)

# Scale
X_scaled = scaler.transform(X)


if st.button("Predict"):
    prob = model.predict_proba(X_scaled)[0][1]
    prediction = "🟢 On-Time Delivery" if prob >= 0.5 else "🔴 Delayed Delivery"

    st.subheader("Prediction Result")
    st.write(f"**Probability of On-Time Delivery:** `{prob:.2f}`")
    st.success(prediction)
