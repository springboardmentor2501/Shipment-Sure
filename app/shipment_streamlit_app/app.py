import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap
import os

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Shipment Delivery Prediction",
    page_icon="🚚",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>
body { background-color: #0f172a; }
.block-container { padding-top: 1.5rem; }
.card {
    background-color: #111827;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 0 15px rgba(0,0,0,0.4);
    margin-bottom: 20px;
}
h1, h2, h3 { color: #f9fafb; }
p, label { color: #d1d5db; }
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg,#2563eb,#3b82f6);
    color: white;
    border-radius: 10px;
    height: 3em;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD PIPELINE (FIXED PATH)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH = os.path.join(BASE_DIR, "shipment_pipeline.pkl")

pipeline = joblib.load(PIPELINE_PATH)
preprocessor = pipeline.named_steps["preprocess"]
model = pipeline.named_steps["model"]

# =====================================================
# SESSION STATE – HISTORY
# =====================================================
if "history" not in st.session_state:
    st.session_state.history = []

# =====================================================
# TITLE
# =====================================================
st.markdown("""
<h1>🚚 Shipment Delivery Prediction Dashboard</h1>
<p style="color:#9ca3af">
AI-powered logistics risk analysis with explainable predictions
</p>
""", unsafe_allow_html=True)

# =====================================================
# LAYOUT
# =====================================================
col_results, col_inputs = st.columns([2, 1])

# =====================================================
# INPUT PANEL
# =====================================================
with col_inputs:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 📦 Enter Shipment Details")

    supplier_rating = st.slider("Supplier Rating (1–5)", 1, 5, 3)
    supplier_lead_time = st.slider("Supplier Lead Time (days)", 1, 30, 7)
    shipping_distance_km = st.slider("Shipping Distance (km)", 1, 5000, 100, step=50)
    order_quantity = st.number_input("Order Quantity", 1, 10000, 100, step=10)
    unit_price = st.number_input("Unit Price", 1.0, 1000.0, 50.0)
    total_order_value = order_quantity * unit_price
    previous_on_time_rate = st.slider("Previous On-Time Rate (%)", 0, 100, 90)
    delivery_duration = st.slider("Delivery Duration (days)", 1, 60, 10)

    order_weekday = st.selectbox(
        "Order Weekday",
        ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    )
    order_month = st.selectbox("Order Month", list(range(1, 13)))
    shipment_mode_code = st.selectbox("Shipment Mode", ["Road","Sea","Air"])
    region_code = st.selectbox("Region", ["East","West","North","South"])
    holiday_code = st.selectbox("Holiday Period", ["Yes","No"])
    weather_code = st.selectbox("Weather Condition", ["Clear","Cloudy","Rainy","Storm"])

    predict_button = st.button("🔮 Predict")
    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# INPUT DATAFRAME
# =====================================================
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

# =====================================================
# RESULTS PANEL
# =====================================================
with col_results:
    if predict_button:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        # -----------------------------
        # PREDICTION
        # -----------------------------
        probs = pipeline.predict_proba(user_input)[0]
        delayed_prob = probs[1]
        ontime_prob = probs[0]

        prediction = "Delayed" if delayed_prob >= 0.4 else "On-Time"
        confidence = delayed_prob if prediction == "Delayed" else ontime_prob

        st.session_state.history.append({
            "Prediction": prediction,
            "Confidence": round(confidence, 3),
            "Distance (km)": shipping_distance_km,
            "Order Qty": order_quantity
        })

        # -----------------------------
        # PROBABILITY CHART
        # -----------------------------
        st.markdown("## 📊 Prediction Probabilities")
        fig, ax = plt.subplots(figsize=(5,3))
        ax.bar(
            ["On-Time", "Delayed"],
            [ontime_prob, delayed_prob],
            color=["#22c55e", "#ef4444"]
        )
        ax.set_ylim(0, 1)
        st.pyplot(fig)

        # -----------------------------
        # FINAL RESULT
        # -----------------------------
        st.markdown("## 🏁 Final Prediction Result")
        if prediction == "Delayed":
            st.error(f"⚠️ Delayed shipment predicted | Confidence: {confidence:.2%}")
        else:
            st.success(f"✅ On-time delivery predicted | Confidence: {confidence:.2%}")

        # =================================================
        # SHAP EXPLAINABILITY
        # =================================================
        st.markdown("## 🧠 Why this prediction? (SHAP Explainability)")

        X_transformed = preprocessor.transform(user_input)
        feature_names = preprocessor.get_feature_names_out()

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)

        shap_vals = shap_values[-1][0] if isinstance(shap_values, list) else shap_values[0]
        shap_vals = np.array(shap_vals).reshape(-1)

        min_len = min(len(feature_names), len(shap_vals))
        feature_names = feature_names[:min_len]
        shap_vals = shap_vals[:min_len]

        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": shap_vals
        }).assign(abs=lambda x: x["SHAP Value"].abs()) \
          .sort_values("abs", ascending=False).head(10)

        fig2, ax2 = plt.subplots(figsize=(5,4))
        ax2.barh(
            shap_df["Feature"],
            shap_df["SHAP Value"],
            color=["#ef4444" if v > 0 else "#22c55e" for v in shap_df["SHAP Value"]]
        )
        ax2.set_xlabel("Impact on Delay Prediction")
        ax2.invert_yaxis()
        st.pyplot(fig2)

        st.markdown("""
        <p style="color:#9ca3af">
        🔴 Positive → increases delay risk<br>
        🟢 Negative → improves on-time delivery
        </p>
        """, unsafe_allow_html=True)

        # =================================================
        # PREDICTION HISTORY
        # =================================================
        st.markdown("## 🕘 Prediction History")
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# RUN:
# python -m streamlit run app/shipment_streamlit_app/app.py
# =====================================================
