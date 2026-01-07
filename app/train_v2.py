import pandas as pd
import numpy as np
import math
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# 1. Load Original Feature Names
with open("feature_names.txt", "r") as f:
    old_feature_names = [line.strip() for line in f]

# 2. Define New Feature Names (Removing Leakage)
leakage_features = [
    "delayed_reason_code_Operational",
    "delayed_reason_code_Traffic",
    "delayed_reason_code_Weather"
]
feature_names_v2 = [f for f in old_feature_names if f not in leakage_features]

# Add New Physics Features
new_physics_features = [
    "Est_Transit_Days", 
    "Est_Total_Days", 
    "Time_Gap", 
    "Big_Gap_Risk", 
    "Safe_Buffer_Excellent"
]
feature_names_v2.extend(new_physics_features)

print(f"Original Features: {len(old_feature_names)}")
print(f"New Features: {len(feature_names_v2)}")

def expand_features_for_training(row):
    payload = {f: 0.0 for f in feature_names_v2} # Default 0.0 for all
    
    # --- Direct Mapping ---
    # Ensure keys match CSV columns exactly or correct mapping
    payload["supplier_lead_time"] = float(row["supplier_lead_time"])
    payload["shipping_distance_km"] = float(row["shipping_distance_km"])
    payload["order_quantity"] = float(row["order_quantity"])
    payload["unit_price"] = float(row["unit_price"])
    payload["total_order_value"] = float(row["total_order_value"])
    payload["previous_on_time_rate"] = float(row["previous_on_time_rate"])
    payload["Promised_Lead_Time"] = float(row["Promised_Lead_Time_Days"])
    
    # --- Boolean Flags ---
    # Map CSV boolean/int columns to feature flags
    # Note: feature_train.csv has columns like 'shipment_mode_Road'
    # We can just check if they exist in the row and are true
    cols_to_check = [
        "holiday_period_Yes", 
        "shipment_mode_Road", "shipment_mode_Sea",
        "weather_condition_Cloudy", "weather_condition_Rainy", "weather_condition_Storm",
        "region_East", "region_North", "region_South", "region_West",
        "carrier_name_DHL", "carrier_name_Delhivery", "carrier_name_EcomExpress", "carrier_name_FedEx"
    ]
    
    for col in cols_to_check:
        if col in row:
             # Handle various CSV formats (True, 1, "True")
             val = row[col]
             if val in [True, 1, "True", "on", "1"]:
                 if col in payload: # Only if it's in our target features
                     payload[col] = 1.0

    # --- Derived Features (Parity with app.py) ---
    payload["supplier_on_time_rate"] = payload["previous_on_time_rate"] / 100.0
    payload["supplier_delay_rate"] = 1.0 - payload["supplier_on_time_rate"]
    payload["supplier_count"] = 20.0 # Heuristic constant from app.py
    
    # Status Flags
    if payload["supplier_on_time_rate"] >= 0.9: payload["Reliable_Supplier"] = 1.0
    if payload["supplier_on_time_rate"] <= 0.7: payload["Risky_Supplier"] = 1.0
    payload["Experienced_Supplier"] = 1.0 

    # --- Heuristic Logistics (Physics-Based) ---
    # Estimate travel time based on mode
    # Default speed (km/day) assumptions
    speed = 500.0 # Road/Default
    if row.get("shipment_mode") == "Sea" or row.get("shipment_mode_Sea") in [1, True]:
        speed = 40.0 # Sea is slow (avg cargo ship 15-20 knots ~ 30-40km/h -> 700-900km/day? Actually port time is high. Conservatively low.)
        # Actually Sea freight is VERY slow. 
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

    # Risk Combos
    payload["Risk_Combo"] = payload["shipping_distance_km"] * payload["supplier_delay_rate"]
    payload["Safe_Combo"] = payload["Promised_Lead_Time"] * payload["supplier_on_time_rate"]
    
    if payload["Risk_Combo"] > 500: payload["Extreme_Risk"] = 1.0
    if payload["Risk_Combo"] > 200: payload["Very_Likely_Delayed"] = 1.0
    if payload["Safe_Combo"] > 10: payload["Almost_Certain_OnTime"] = 1.0
    
    payload["Cost_per_Day"] = payload["total_order_value"] / max(1.0, payload["Promised_Lead_Time"])
    payload["Distance_per_Day"] = payload["shipping_distance_km"] / max(1.0, payload["Promised_Lead_Time"])

    # Transforms
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

    # Supplier Rating One-Hot
    rating = round(float(row["supplier_rating"]), 1)
    key = f"supplier_rating_{rating}"
    if key in payload: payload[key] = 1.0

    # Cyclical Time
    month = int(row["Order_Month"])
    day = int(row["Order_Day_of_Week"])
    day_of_year = month * 30 
    
    payload["Month_sin"] = math.sin(2 * math.pi * month / 12)
    payload["Month_cos"] = math.cos(2 * math.pi * month / 12)
    payload["DayOfWeek_sin"] = math.sin(2 * math.pi * day / 7)
    payload["DayOfWeek_cos"] = math.cos(2 * math.pi * day / 7)
    payload["DayOfYear_sin"] = math.sin(2 * math.pi * day_of_year / 365)
    payload["DayOfYear_cos"] = math.cos(2 * math.pi * day_of_year / 365)

    return [payload[f] for f in feature_names_v2]

def run_training():
    print("Loading raw training data...")
    df_x = pd.read_csv("../Splited_Data/feature_train.csv")
    df_y = pd.read_csv("../Splited_Data/target_train.csv") # 'on_time_delivery'
    
    print(f"Processing {len(df_x)} training rows...")
    X_train = []
    for idx, row in df_x.iterrows():
        X_train.append(expand_features_for_training(row))
    X_train = np.array(X_train)
    
    # Setup Target
    # Original target: 1 = On-Time, 0 = Delayed
    # Model Expectation: Class 1 = Delayed, Class 0 = On-Time
    # So we invert the target
    y_raw = df_y["on_time_delivery"].values
    y_train = 1 - y_raw
    
    print(f"Class Distribution: {np.bincount(y_train)} (0=OnTime, 1=Delayed)")
    
    # Train Model with GradientBoosting
    from sklearn.ensemble import GradientBoostingClassifier
    print("Training Gradient Boosting...")
    
    # Simple GBC first to check potential
    gbc = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42)
    gbc.fit(X_train, y_train)
    
    train_pred = gbc.predict(X_train)
    acc = accuracy_score(y_train, train_pred)
    print(f"Training Accuracy (GBC): {acc:.4f}")
    
    # Feature Importance
    print("\nTop 10 Features:")
    importances = gbc.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i in range(10):
        print(f"{i+1}. {feature_names_v2[indices[i]]}: {importances[indices[i]]:.4f}")

    # Save Model
    print("Saving model to rf_delivery_model_v2.pkl...")
    with open("rf_delivery_model_v2.pkl", "wb") as f:
        pickle.dump(gbc, f)
        
    # Save Feature Names
    print("Saving feature names to feature_names_v2.txt...")
    with open("feature_names_v2.txt", "w") as f:
        for name in feature_names_v2:
            f.write(f"{name}\n")
            
    print("Done.")

if __name__ == "__main__":
    run_training()
