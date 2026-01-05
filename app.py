from flask import Flask, render_template, request
import pandas as pd
import joblib
from datetime import datetime

app = Flask(__name__)

# ================= LOAD MODEL + METADATA =================
bundle = joblib.load("best_clean_model3.pkl")
model = bundle["model"]
THRESHOLD = bundle["threshold"]

# Top-20 features (exact order used in training)
FEATURE_COLUMNS = joblib.load("model_features3.pkl")

# Scaler trained ONLY on top-20 features
scaler = joblib.load("scaler_top20.pkl")


# ================= HOME =================
@app.route("/")
def index():
    return render_template("index.html")


# ================= PREDICT =================
@app.route("/predict", methods=["POST"])
def predict():

    # ---------- BASIC INPUT ----------
    supplier_rating = float(request.form["supplier_rating"])
    supplier_lead_time = int(request.form["supplier_lead_time"])
    previous_on_time_rate = float(request.form["previous_on_time_rate"])
    supplier_total_orders = int(request.form["supplier_total_orders"])

    shipping_distance_km = float(request.form["shipping_distance_km"])
    order_quantity = int(request.form["order_quantity"])
    unit_price = float(request.form["unit_price"])
    total_order_value = float(request.form["order_value"])

    carrier_avg_delay = float(request.form["carrier_avg_delay"])
    region_difficulty_score = float(request.form["region_difficulty_score"])

    order_date = datetime.strptime(request.form["order_date"], "%Y-%m-%d")
    promised_date = datetime.strptime(request.form["promised_delivery_date"], "%Y-%m-%d")

    # ---------- DATE FEATURES ----------
    order_day_of_week = order_date.weekday()
    order_month = order_date.month
    order_date_days = (order_date - datetime(2020, 1, 1)).days
    promised_delivery_date_days = (promised_date - datetime(2020, 1, 1)).days
    promised_lead_gap = promised_delivery_date_days - order_date_days

    # ---------- ENGINEERED FEATURES ----------
    carrier_delay_rate = carrier_avg_delay / 10
    supplier_reliability_trend = previous_on_time_rate

    # ---------- BUILD INPUT DICT ----------
    input_dict = {
        "supplier_rating": supplier_rating,
        "supplier_lead_time": supplier_lead_time,
        "previous_on_time_rate": previous_on_time_rate,
        "supplier_total_orders": supplier_total_orders,
        "shipping_distance_km": shipping_distance_km,
        "order_quantity": order_quantity,
        "unit_price": unit_price,
        "total_order_value": total_order_value,
        "carrier_avg_delay": carrier_avg_delay,
        "carrier_delay_rate": carrier_delay_rate,
        "region_difficulty_score": region_difficulty_score,
        "supplier_reliability_trend": supplier_reliability_trend,
        "order_day_of_week": order_day_of_week,
        "order_month": order_month,
        "order_date_days": order_date_days,
        "promised_delivery_date_days": promised_delivery_date_days,
        "promised_lead_gap": promised_lead_gap,
    }

    # ---------- ALIGN TO TOP-20 FEATURES ----------
    row = [input_dict.get(col, 0) for col in FEATURE_COLUMNS]
    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    # ---------- SCALE (CRITICAL FIX) ----------
    df_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=FEATURE_COLUMNS
    )

    # ---------- PREDICT ----------
    prob = model.predict_proba(df_scaled)[0][1]
    prediction = "On-Time Delivery" if prob >= THRESHOLD else "Delayed Delivery"

    prob_percent = prob * 100

    if prob_percent >= 60:
        chance_label = "High"
    elif prob_percent >= 40:
        chance_label = "Moderate"
    else:
        chance_label = "Low"

    return render_template(
        "result.html",
        prediction=prediction,
        chance=chance_label,
        probability=round(prob_percent, 1)
    )



if __name__ == "__main__":
    app.run(debug=True)
