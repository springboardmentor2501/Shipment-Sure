import pickle
import pandas as pd
import numpy as np
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load model and features
with open("rf_delivery_model_v2.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_names_v2.txt", "r") as f:
    feature_names = [line.strip() for line in f]

def expand_features_from_csv(row):
    payload = {f: 0.0 for f in feature_names}
    
    # Direct mapping
    payload["supplier_lead_time"] = float(row["supplier_lead_time"])
    payload["shipping_distance_km"] = float(row["shipping_distance_km"])
    payload["order_quantity"] = float(row["order_quantity"])
    payload["unit_price"] = float(row["unit_price"])
    payload["total_order_value"] = float(row["total_order_value"])
    payload["previous_on_time_rate"] = float(row["previous_on_time_rate"])
    payload["Promised_Lead_Time"] = float(row["Promised_Lead_Time_Days"])
    
    # Flags
    cols_to_map = [
        "holiday_period_Yes", "shipment_mode_Road", "shipment_mode_Sea",
        "weather_condition_Cloudy", "weather_condition_Rainy", "weather_condition_Storm",
        "region_East", "region_North", "region_South", "region_West",
        "carrier_name_DHL", "carrier_name_Delhivery", "carrier_name_EcomExpress", "carrier_name_FedEx",
        # "delayed_reason_code_Operational", "delayed_reason_code_Traffic", "delayed_reason_code_Weather" # LEAKAGE: Not available at inference
    ]
    for col in cols_to_map:
        if col in row and row[col] in [True, 1, "True", "on", "1"]:
            payload[col] = 1.0

    # Derived
    payload["supplier_on_time_rate"] = payload["previous_on_time_rate"] / 100.0
    payload["supplier_delay_rate"] = 1.0 - payload["supplier_on_time_rate"]
    payload["supplier_count"] = 20.0

    if payload["supplier_on_time_rate"] >= 0.9: payload["Reliable_Supplier"] = 1.0
    if payload["supplier_on_time_rate"] <= 0.7: payload["Risky_Supplier"] = 1.0
    payload["Experienced_Supplier"] = 1.0

    # --- Heuristic Logistics (Physics-Based) ---
    # Estimate travel time based on mode
    # Default speed (km/day) assumptions
    speed = 500.0 # Road/Default
    if row.get("shipment_mode") == "Sea" or row.get("shipment_mode_Sea") in [1, True]:
        speed = 40.0 # Sea is slow
    elif row.get("shipment_mode") == "Air" or row.get("shipment_mode_Air") in [1, True]:
        speed = 2000.0
    
    # Check boolean columns in row for mode if raw string not present
    if "shipment_mode_Sea" in row and row["shipment_mode_Sea"] in [1, True]: speed = 50.0
    if "shipment_mode_Road" in row and row["shipment_mode_Road"] in [1, True]: speed = 500.0
    
    transit_days = payload["shipping_distance_km"] / max(1.0, speed)
    payload["Est_Transit_Days"] = transit_days
    payload["Est_Total_Days"] = payload["supplier_lead_time"] + transit_days
    
    # The 'Gap' - positive means we need MORE time than promised (Delay Likely)
    payload["Time_Gap"] = payload["Est_Total_Days"] - payload["Promised_Lead_Time"]
    
    if payload["Time_Gap"] > 0: payload["Big_Gap_Risk"] = 1.0
    if payload["Time_Gap"] < -2: payload["Safe_Buffer_Excellent"] = 1.0

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

    # Bins
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

    rating = round(float(row["supplier_rating"]), 1)
    key = f"supplier_rating_{rating}"
    if key in payload: payload[key] = 1.0

    month = int(row["Order_Month"])
    day = int(row["Order_Day_of_Week"])
    day_of_year = month * 30
    
    payload["Month_sin"] = math.sin(2 * math.pi * month / 12)
    payload["Month_cos"] = math.cos(2 * math.pi * month / 12)
    payload["DayOfWeek_sin"] = math.sin(2 * math.pi * day / 7)
    payload["DayOfWeek_cos"] = math.cos(2 * math.pi * day / 7)
    payload["DayOfYear_sin"] = math.sin(2 * math.pi * day_of_year / 365)
    payload["DayOfYear_cos"] = math.cos(2 * math.pi * day_of_year / 365)

    return [payload[f] for f in feature_names]

def run(scale_previous=False):
    df_x = pd.read_csv("../Splited_Data/feature_test.csv")
    df_y = pd.read_csv("../Splited_Data/target_test.csv")
    
    y_true = df_y["on_time_delivery"].tolist() # Official Ground Truth
    
    feature_matrix = []
    for idx, row in df_x.iterrows():
        vec = expand_features_from_csv(row)
        feature_matrix.append(vec)
        
    X = np.array(feature_matrix).astype(np.float32)
    y_pred_raw = model.predict(X)
    
    # Model: 1=Delayed, 0=OnTime
    # Ground Truth: 1=OnTime, 0=Delayed
    # So we must flip the prediction to compare with y_true
    y_pred = 1 - y_pred_raw
    
    correct = accuracy_score(y_true, y_pred, normalize=False)
    total = len(y_true)
    incorrect = total - correct
    acc = (correct / total) * 100
    
    print("\n" + "="*50)
    print("       FINAL VALIDATION SUMMARY")
    print("="*50)
    print(f"Total Test Samples:      {total:,}")
    print(f"Correct Predictions:     {correct:,}")
    print(f"Incorrect Predictions:   {incorrect:,}")
    print("-" * 50)
    print(f"Final Accuracy:          {acc:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    run()
