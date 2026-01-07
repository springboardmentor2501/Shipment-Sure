import pickle
import pandas as pd
import numpy as np
import math
from datetime import datetime

# Load model and features
with open("rf_delivery_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_names.txt", "r") as f:
    feature_names = [line.strip() for line in f]

def expand_features(raw_data):
    """
    Copy of the inference expansion logic for self-contained testing.
    """
    payload = {f: 0.0 for f in feature_names}
    
    direct_fields = [
        "shipping_distance_km", "order_quantity", "unit_price", 
        "Promised_Lead_Time", "supplier_lead_time", 
        "previous_on_time_rate", "supplier_on_time_rate"
    ]
    for field in direct_fields:
        if field in raw_data and raw_data[field]:
            payload[field] = float(raw_data[field])

    payload["total_order_value"] = payload["order_quantity"] * payload["unit_price"]
    if payload["Promised_Lead_Time"] > 0:
        payload["Cost_per_Day"] = payload["total_order_value"] / payload["Promised_Lead_Time"]
        payload["Distance_per_Day"] = payload["shipping_distance_km"] / payload["Promised_Lead_Time"]
    
    payload["supplier_delay_rate"] = 1.0 - payload["supplier_on_time_rate"]
    payload["supplier_count"] = 20.0

    if payload["supplier_on_time_rate"] >= 0.9: payload["Reliable_Supplier"] = 1.0
    if payload["supplier_on_time_rate"] <= 0.7: payload["Risky_Supplier"] = 1.0
    if payload["supplier_count"] >= 50: payload["Experienced_Supplier"] = 1.0

    payload["Risk_Combo"] = payload["shipping_distance_km"] * payload["supplier_delay_rate"]
    payload["Safe_Combo"] = payload["Promised_Lead_Time"] * payload["supplier_on_time_rate"]
    
    if payload["Risk_Combo"] > 500: payload["Extreme_Risk"] = 1.0
    if payload["Risk_Combo"] > 200: payload["Very_Likely_Delayed"] = 1.0
    if payload["Safe_Combo"] > 10: payload["Almost_Certain_OnTime"] = 1.0

    def safe_log(v): return math.log(max(0.0001, v))
    def safe_sqrt(v): return math.sqrt(max(0, v))

    payload["log_Promised_Lead_Time"] = safe_log(payload["Promised_Lead_Time"])
    payload["sqrt_Promised_Lead_Time"] = safe_sqrt(payload["Promised_Lead_Time"])
    payload["log_shipping_distance_km"] = safe_log(payload["shipping_distance_km"])
    payload["sqrt_shipping_distance_km"] = safe_sqrt(payload["shipping_distance_km"])
    payload["log_unit_price"] = safe_log(payload["unit_price"])
    payload["sqrt_unit_price"] = safe_sqrt(payload["unit_price"])
    payload["log_Risk_Combo"] = safe_log(payload["Risk_Combo"])
    payload["sqrt_Risk_Combo"] = safe_sqrt(payload["Risk_Combo"])
    payload["log_supplier_delay_rate"] = safe_log(payload["supplier_delay_rate"])
    payload["sqrt_supplier_delay_rate"] = safe_sqrt(payload["supplier_delay_rate"])

    if payload["Promised_Lead_Time"] <= 2: payload["Lead_Bin_Fast"] = 1.0
    elif payload["Promised_Lead_Time"] <= 5: payload["Lead_Bin_Normal"] = 1.0
    elif payload["Promised_Lead_Time"] <= 10: payload["Lead_Bin_Slow"] = 1.0
    else: payload["Lead_Bin_VerySlow"] = 1.0

    if payload["shipping_distance_km"] <= 300: payload["Dist_Bin_Regional"] = 1.0
    elif payload["shipping_distance_km"] <= 1000: payload["Dist_Bin_National"] = 1.0
    elif payload["shipping_distance_km"] <= 3000: payload["Dist_Bin_Continental"] = 1.0
    else: payload["Dist_Bin_International"] = 1.0

    if payload["unit_price"] < 25: payload["Price_Quantile_MediumLow"] = 1.0
    elif payload["unit_price"] < 50: payload["Price_Quantile_Medium"] = 1.0
    elif payload["unit_price"] < 100: payload["Price_Quantile_MediumHigh"] = 1.0
    else: payload["Price_Quantile_High"] = 1.0

    bool_fields = ["Is_Peak_Season", "Is_Weekend_Order", "Is_Weekend_Delivery", "holiday_period_Yes"]
    for field in bool_fields:
        payload[field] = 1.0 if raw_data.get(field) in [True, "on", 1, "1"] else 0.0

    cat_mappings = {
        "supplier_rating": raw_data.get("supplier_rating"),
        "shipment_mode": raw_data.get("shipment_mode"),
        "carrier_name": raw_data.get("carrier_name"),
        "region": raw_data.get("region"),
        "weather_condition": raw_data.get("weather_condition")
    }
    for prefix, val in cat_mappings.items():
        if val:
            key = f"{prefix}_{val}"
            if key in payload: payload[key] = 1.0

    now = datetime.now()
    month, day, day_of_year = now.month, now.weekday(), now.timetuple().tm_yday
    
    payload["Month_sin"] = math.sin(2 * math.pi * month / 12)
    payload["Month_cos"] = math.cos(2 * math.pi * month / 12)
    payload["DayOfWeek_sin"] = math.sin(2 * math.pi * day / 7)
    payload["DayOfWeek_cos"] = math.cos(2 * math.pi * day / 7)
    payload["DayOfYear_sin"] = math.sin(2 * math.pi * day_of_year / 365)
    payload["DayOfYear_cos"] = math.cos(2 * math.pi * day_of_year / 365)

    df = pd.DataFrame([payload])
    df = df.reindex(columns=feature_names)
    return df.astype(np.float32)

def run_tests():
    test_cases = [
        {
            "name": "Case 1: The 'Perfect' Ship (Regional, Safe)",
            "data": {
                "shipping_distance_km": 50, "order_quantity": 10, "unit_price": 100,
                "Promised_Lead_Time": 7, "supplier_lead_time": 1, 
                "previous_on_time_rate": 0.99, "supplier_on_time_rate": 0.99,
                "supplier_rating": "5.0", "weather_condition": "Clear", "shipment_mode": "Road"
            },
            "expected": "On-Time"
        },
        {
            "name": "Case 2: The 'Impossible' Order (Global, Risky)",
            "data": {
                "shipping_distance_km": 2500, "order_quantity": 50, "unit_price": 500,
                "Promised_Lead_Time": 1, "supplier_lead_time": 10, 
                "previous_on_time_rate": 0.40, "supplier_on_time_rate": 0.50,
                "supplier_rating": "3.0", "weather_condition": "Storm", "shipment_mode": "Sea"
            },
            "expected": "Delayed"
        },
        {
            "name": "Case 3: Regional Rush (Tight but Reliable)",
            "data": {
                "shipping_distance_km": 300, "order_quantity": 25, "unit_price": 45,
                "Promised_Lead_Time": 2, "supplier_lead_time": 1, 
                "previous_on_time_rate": 0.90, "supplier_on_time_rate": 0.95,
                "supplier_rating": "4.0", "weather_condition": "Clear", "shipment_mode": "Road"
            },
            "expected": "On-Time"
        },
        {
            "name": "Case 4: Long Haul Storm (National, Slow)",
            "data": {
                "shipping_distance_km": 1500, "order_quantity": 200, "unit_price": 30,
                "Promised_Lead_Time": 3, "supplier_lead_time": 5, 
                "previous_on_time_rate": 0.70, "supplier_on_time_rate": 0.75,
                "supplier_rating": "3.0", "weather_condition": "Rainy", "shipment_mode": "Sea",
                "Is_Peak_Season": 1
            },
            "expected": "Delayed"
        }
    ]

    print("\n" + "="*80)
    print(f"{'TEST CASE NAME':<45} | {'PREDICTED':<12} | {'CONFIDENCE':<10} | {'STATUS'}")
    print("-" * 80)
    
    passed_count = 0
    for case in test_cases:
        input_df = expand_features(case["data"])
        probs = model.predict_proba(input_df)[0]
        prob_delayed = float(probs[1]) 
        prediction = "Delayed" if prob_delayed >= 0.5 else "On-Time"
        confidence = prob_delayed if prediction == "Delayed" else (1 - prob_delayed)
        
        status = "✅ PASS" if prediction == case["expected"] else "❌ FAIL"
        if status == "✅ PASS": passed_count += 1
        
        print(f"{case['name']:<45} | {prediction:<12} | {confidence*100:>8.1f}% | {status}")

    print("="*80)
    print(f"SUMMARY: {passed_count}/{len(test_cases)} Tests Passed")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_tests()
