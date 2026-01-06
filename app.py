import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# -------------------------------
# TITLE
# -------------------------------
st.title("📦 Shipment Delivery Prediction App")
st.write("Predict whether a shipment will be delivered on time")

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_excel("shipment_dataset_10000 (1).xlsx")

TARGET = "on_time_delivery"

# -------------------------------
# FIX NUMERIC COLUMNS
# -------------------------------
num_cols = [
    "supplier_rating",
    "supplier_lead_time",
    "shipping_distance_km",
    "order_quantity",
    "unit_price",
    "total_order_value",
    "previous_on_time_rate"
]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.dropna(subset=num_cols, inplace=True)

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
df["cost_to_weight_ratio"] = df["total_order_value"] / (df["order_quantity"] + 1)
df["delay_reason_missing"] = df["delayed_reason_code"].isnull().astype(int)

df.drop(columns=[
    "order_id",
    "supplier_id",
    "order_date",
    "promised_delivery_date",
    "actual_delivery_date",
    "delayed_reason_code"
], inplace=True)

# -------------------------------
# FEATURES
# -------------------------------
cat_cols = [
    "shipment_mode",
    "weather_condition",
    "region",
    "holiday_period",
    "carrier_name"
]

num_features = [c for c in df.columns if c not in cat_cols + [TARGET]]

X = df.drop(columns=[TARGET])
y = df[TARGET]

# -------------------------------
# PREPROCESSING + MODEL
# -------------------------------
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
])

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    eval_metric="logloss",
    random_state=42
)

pipe = Pipeline([
    ("prep", preprocessor),
    ("model", model)
])

pipe.fit(X, y)

# -------------------------------
# USER INPUTS
# -------------------------------
supplier_rating = st.slider("Supplier Rating", 1.0, 5.0, 3.0)
supplier_lead_time = st.number_input("Supplier Lead Time (days)", 1, 30, 7)
shipping_distance_km = st.number_input("Shipping Distance (km)", 10, 2000, 500)
order_quantity = st.number_input("Order Quantity", 1, 500, 50)
unit_price = st.number_input("Unit Price", 1.0, 1000.0, 100.0)
previous_on_time_rate = st.slider("Previous On-Time Rate", 0.0, 1.0, 0.8)

shipment_mode = st.selectbox("Shipment Mode", df["shipment_mode"].unique())
weather_condition = st.selectbox("Weather Condition", df["weather_condition"].unique())
region = st.selectbox("Region", df["region"].unique())
holiday_period = st.selectbox("Holiday Period", df["holiday_period"].unique())
carrier_name = st.selectbox("Carrier Name", df["carrier_name"].unique())

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict Delivery Status"):
    input_df = pd.DataFrame([{
        "supplier_rating": supplier_rating,
        "supplier_lead_time": supplier_lead_time,
        "shipping_distance_km": shipping_distance_km,
        "order_quantity": order_quantity,
        "unit_price": unit_price,
        "total_order_value": unit_price * order_quantity,
        "previous_on_time_rate": previous_on_time_rate,
        "cost_to_weight_ratio": (unit_price * order_quantity) / (order_quantity + 1),
        "delay_reason_missing": 1,
        "shipment_mode": shipment_mode,
        "weather_condition": weather_condition,
        "region": region,
        "holiday_period": holiday_period,
        "carrier_name": carrier_name
    }])

    THRESHOLD = 0.8
    on_time_prob = 1 - delay_prob

    if delay_prob >= THRESHOLD:
        st.error(f"❌ Delivery Delay Expected (Risk: {delay_prob:.2f})")
    else:
        st.success(f"✅ On-Time Delivery Expected (Confidence: {on_time_prob:.2f})")

    col1, col2 = st.columns(2)
    col1.metric("Delay Probability", f"{delay_prob:.2f}")
    col2.metric("On-Time Probability", f"{on_time_prob:.2f}")
