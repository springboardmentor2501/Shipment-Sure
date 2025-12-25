'''import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Load pipeline
# -----------------------------
pipeline = joblib.load("shipment_pipeline.pkl")

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Shipment Delivery Prediction", page_icon="🚚", layout="wide")
st.title("🚚 Shipment Delivery Prediction Dashboard")

# -----------------------------
# Layout: inputs right, results left
# -----------------------------
col_results, col_inputs = st.columns([2, 1])  # wider left column for results

with col_inputs:
    st.header("📦 Enter Shipment Details")

    supplier_rating = st.slider("Supplier Rating (1-5)", 1, 5, 3, step=1)
    supplier_lead_time = st.slider("Supplier Lead Time (days)", 1, 30, 7, step=1)
    shipping_distance_km = st.slider("Shipping Distance (km)", 1, 5000, 100, step=50)
    order_quantity = st.number_input("Order Quantity", 1, 10000, 100, step=10)
    unit_price = st.number_input("Unit Price", 1.0, 1000.0, 50.0, step=1.0)
    total_order_value = order_quantity * unit_price
    previous_on_time_rate = st.slider("Previous On-Time Rate (%)", 0, 100, 90, step=5)
    delivery_duration = st.slider("Delivery Duration (days)", 1, 60, 10, step=1)
    order_weekday = st.selectbox("Order Weekday", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    order_month = st.selectbox("Order Month", list(range(1,13)))
    shipment_mode_code = st.selectbox("Shipment Mode", ["Road","Sea","Air"])
    region_code = st.selectbox("Region", ["East","West","North","South"])
    holiday_code = st.selectbox("Holiday Period", ["Yes","No"])
    weather_code = st.selectbox("Weather Condition", ["Clear","Cloudy","Rainy","Storm"])

    # Build input DataFrame
    user_input = pd.DataFrame({
        "supplier_rating": [supplier_rating],
        "supplier_lead_time": [supplier_lead_time],
        "shipping_distance_km": [shipping_distance_km],
        "order_quantity": [order_quantity],
        "unit_price": [unit_price],
        "total_order_value": [total_order_value],
        "previous_on_time_rate": [previous_on_time_rate],
        "delivery_duration": [delivery_duration],
        "order_weekday": [order_weekday],
        "order_month": [order_month],
        "shipment_mode_code": [shipment_mode_code],
        "region_code": [region_code],
        "holiday_code": [holiday_code],
        "weather_code": [weather_code]
    })

    predict_button = st.button("🔮 Predict")

with col_results:
    if predict_button:
        # Debug: show input
        st.write("Input Data:", user_input)

        # Get probability of class 1 (Delayed)
        prob = pipeline.predict_proba(user_input)[0][1]

        # Custom threshold (0.4 instead of 0.5)
        prediction = 1 if prob >= 0.4 else 0

        # -----------------------------
        # Probability chart (smaller size)
        # -----------------------------
        st.markdown("## 📊 Prediction Probabilities")
        fig, ax = plt.subplots(figsize=(4,3))
        ax.bar(["On-time", "Delayed"], pipeline.predict_proba(user_input)[0], color=["green", "red"])
        ax.set_ylabel("Probability")
        st.pyplot(fig)

        # -----------------------------
        # Feature importance chart (smaller size)
        # -----------------------------
        st.markdown("## 🔎 Top Features Driving Prediction")
        importances = pipeline.named_steps["model"].feature_importances_
        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        indices = np.argsort(importances)[-10:]  # top 10

        fig, ax = plt.subplots(figsize=(6,4))
        ax.barh(range(len(indices)), importances[indices], align="center", color="skyblue")
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels(feature_names[indices])
        ax.set_xlabel("Importance")
        st.pyplot(fig)

        # -----------------------------
        # Final prediction result (no balloons/snow)
        # -----------------------------
        st.markdown("## 🏁 Final Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ Delayed shipment predicted (Confidence: {prob:.2f})")
        else:
            st.success(f"✅ On-time delivery predicted (Confidence: {1-prob:.2f})")

#To run--> python -m streamlit run app.py'''

import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Load pipeline
# -----------------------------
pipeline = joblib.load("shipment_pipeline.pkl")

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Shipment Delivery Prediction", page_icon="🚚", layout="wide")
st.title("🚚 Shipment Delivery Prediction Dashboard")

# -----------------------------
# Layout: inputs right, results left
# -----------------------------
col_results, col_inputs = st.columns([2, 1])  # wider left column for results

with col_inputs:
    st.header("📦 Enter Shipment Details")

    supplier_rating = st.slider("Supplier Rating (1-5)", 1, 5, 3, step=1)
    supplier_lead_time = st.slider("Supplier Lead Time (days)", 1, 30, 7, step=1)
    shipping_distance_km = st.slider("Shipping Distance (km)", 1, 5000, 100, step=50)
    order_quantity = st.number_input("Order Quantity", 1, 10000, 100, step=10)
    unit_price = st.number_input("Unit Price", 1.0, 1000.0, 50.0, step=1.0)
    total_order_value = order_quantity * unit_price
    previous_on_time_rate = st.slider("Previous On-Time Rate (%)", 0, 100, 90, step=5)
    delivery_duration = st.slider("Delivery Duration (days)", 1, 60, 10, step=1)
    order_weekday = st.selectbox("Order Weekday", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    order_month = st.selectbox("Order Month", list(range(1,13)))
    shipment_mode_code = st.selectbox("Shipment Mode", ["Road","Sea","Air"])
    region_code = st.selectbox("Region", ["East","West","North","South"])
    holiday_code = st.selectbox("Holiday Period", ["Yes","No"])
    weather_code = st.selectbox("Weather Condition", ["Clear","Cloudy","Rainy","Storm"])

    # Build input DataFrame
    user_input = pd.DataFrame({
        "supplier_rating": [supplier_rating],
        "supplier_lead_time": [supplier_lead_time],
        "shipping_distance_km": [shipping_distance_km],
        "order_quantity": [order_quantity],
        "unit_price": [unit_price],
        "total_order_value": [total_order_value],
        "previous_on_time_rate": [previous_on_time_rate],
        "delivery_duration": [delivery_duration],
        "order_weekday": [order_weekday],
        "order_month": [order_month],
        "shipment_mode_code": [shipment_mode_code],
        "region_code": [region_code],
        "holiday_code": [holiday_code],
        "weather_code": [weather_code]
    })

    predict_button = st.button("🔮 Predict")

with col_results:
    if predict_button:
        # Debug: show input
        st.write("Input Data:", user_input)

        # Get probability of class 1 (Delayed)
        prob = pipeline.predict_proba(user_input)[0][1]

        # Custom threshold (0.4 instead of 0.5)
        prediction = 1 if prob >= 0.4 else 0

        # -----------------------------
        # Probability chart (smaller size)
        # -----------------------------
        st.markdown("## 📊 Prediction Probabilities")
        fig, ax = plt.subplots(figsize=(4,3))
        ax.bar(["On-time", "Delayed"], pipeline.predict_proba(user_input)[0], color=["green", "red"])
        ax.set_ylabel("Probability")
        st.pyplot(fig)

        # -----------------------------
        # Feature importance chart (smaller size)
        # -----------------------------
        st.markdown("## 🔎 Top Features Driving Prediction")
        importances = pipeline.named_steps["model"].feature_importances_
        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        indices = np.argsort(importances)[-10:]  # top 10

        fig, ax = plt.subplots(figsize=(6,4))
        ax.barh(range(len(indices)), importances[indices], align="center", color="skyblue")
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels(feature_names[indices])
        ax.set_xlabel("Importance")
        st.pyplot(fig)

        # -----------------------------
        # Final prediction result with colored background
        # -----------------------------
        st.markdown("## 🏁 Final Prediction Result")

        if prediction == 1:
            st.markdown(
                f"""
                <div style="background-color:#ffcccc;padding:20px;border-radius:10px">
                    <h3 style="color:red;">⚠️ Delayed shipment predicted</h3>
                    <p>Confidence: {prob:.2f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="background-color:#ccffcc;padding:20px;border-radius:10px">
                    <h3 style="color:green;">✅ On-time delivery predicted</h3>
                    <p>Confidence: {1-prob:.2f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

# To run --> python -m streamlit run app.py