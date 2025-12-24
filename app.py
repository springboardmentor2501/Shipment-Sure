import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load("./models/shipment_delivery_model.pkl")
scaler = joblib.load("./models/scaler.pkl")
feature_cols = joblib.load("./models/feature_columns.pkl")

st.set_page_config(page_title="Shipment On-Time Delivery Prediction", layout="wide")

st.title("Shipment On-Time Delivery Prediction App")
st.write("Enter order-level features below to get the On-Time Delivery Probability")


# USER INPUT UI

# Numeric inputs
Customer_rating = st.number_input("Customer Rating (1–5)", 1, 5, 4)
Cost_of_the_Product = st.number_input("Cost of Product", 1, 10000, 200)
Weight_in_gms = st.number_input("Weight (gms)", 1, 10000, 1500)
supplier_rating = st.number_input("Supplier Rating", 1.0, 5.0, 4.0)
supplier_lead_time = st.number_input("Supplier Lead Time (days)", 1, 30, 7)
shipping_distance_km = st.number_input("Shipping Distance (km)", 1, 3000, 500)
unit_price = st.number_input("Unit Price", 1, 5000, 50)
total_order_value = st.number_input("Total Order Value", 1, 20000, 600)
previous_on_time_rate = st.number_input("Previous On-Time Rate (0–1)", 0.0, 1.0, 0.70)

# Derived feature
Cost_to_Weight_Ratio = Cost_of_the_Product / (Weight_in_gms + 1e-6)


# ONE-HOT CATEGORICAL INPUTS


st.subheader("Shipment-Related Categorical Features")

Mode = st.selectbox("Mode of Shipment", ["Road", "Ship"])
Product_imp = st.selectbox("Product Importance", ["low", "medium"])
Weather = st.selectbox("Weather Condition", ["Rainy", "Storm"])
Region = st.selectbox("Region", ["North", "South", "West"])
Carrier = st.selectbox("Carrier Name", ["FedEx", "LocalTruckers"])
Reason = st.selectbox("Delayed Reason Code", ["Traffic", "Weather"])
Holiday = st.selectbox("Holiday Period", ["Yes", "No"])


# CREATE INPUT ROW (One-Hot Mapping)

# Prepare template
input_data = {col: 0 for col in feature_cols}

# Fill numeric values
input_data.update(
    {
        "Customer_rating": Customer_rating,
        "Cost_of_the_Product": Cost_of_the_Product,
        "Weight_in_gms": Weight_in_gms,
        "supplier_rating": supplier_rating,
        "supplier_lead_time": supplier_lead_time,
        "shipping_distance_km": shipping_distance_km,
        "unit_price": unit_price,
        "total_order_value": total_order_value,
        "previous_on_time_rate": previous_on_time_rate,
        "Cost_to_Weight_Ratio": Cost_to_Weight_Ratio,
    }
)

# Apply One-Hot values
# Mode of shipment
if Mode == "Road":
    input_data["Mode_of_Shipment_Road"] = 1
if Mode == "Ship":
    input_data["Mode_of_Shipment_Ship"] = 1

# Product importance
if Product_imp == "low":
    input_data["Product_importance_low"] = 1
if Product_imp == "medium":
    input_data["Product_importance_medium"] = 1

# Weather
if Weather == "Rainy":
    input_data["weather_condition_Rainy"] = 1
if Weather == "Storm":
    input_data["weather_condition_Storm"] = 1

# Region
if Region == "North":
    input_data["region_North"] = 1
if Region == "South":
    input_data["region_South"] = 1
if Region == "West":
    input_data["region_West"] = 1

# Carrier
if Carrier == "FedEx":
    input_data["carrier_name_FedEx"] = 1
if Carrier == "LocalTruckers":
    input_data["carrier_name_LocalTruckers"] = 1

# Delayed reason
if Reason == "Traffic":
    input_data["delayed_reason_code_Traffic"] = 1
if Reason == "Weather":
    input_data["delayed_reason_code_Weather"] = 1

# Holiday
if Holiday == "Yes":
    input_data["holiday_period_Yes"] = 1

# -----------------------------
# PREDICTION
# -----------------------------

if st.button("Predict Delivery Status"):
    df_input = pd.DataFrame([input_data])

    # scale
    df_input_scaled = df_input.copy()
    numeric_cols = scaler.feature_names_in_
    df_input_scaled[numeric_cols] = scaler.transform(df_input[numeric_cols])

    # Predict
    prob = model.predict_proba(df_input_scaled)[0][1]

    st.subheader(" Prediction Result")
    st.write(f"### On-Time Delivery Probability: **{prob * 100:.2f}%**")

    if prob >= 0.70:
        st.success(" High chance of On-Time Delivery")
    elif prob >= 0.50:
        st.warning(" Moderate chance — Risky")
    else:
        st.error(" High chance of Delay")
