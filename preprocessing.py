import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_excel("shipment_dataset_10000.xlsx")
print("Original shape:", df.shape)

df["delay_days"] = (df["actual_delivery_date"] - df["promised_delivery_date"]).dt.days
df["on_time_delivery"] = (df["delay_days"] <= 0).astype(int)

print(df["on_time_delivery"].value_counts())

# =============================
# 3. SUPPLIER FEATURES
# =============================
if "supplier_id" in df.columns:
    supplier_stats = df.groupby("supplier_id").agg(
        supplier_total_orders=("on_time_delivery", "count"),
        supplier_on_time_rate=("on_time_delivery", "mean"),
        supplier_avg_delay=("delay_days", "mean")
    )

    supplier_stats["supplier_delay_rate"] = 1 - supplier_stats["supplier_on_time_rate"]

    df = df.merge(supplier_stats, on="supplier_id", how="left")

    # Supplier trend (rolling 3)
    df["supplier_reliability_trend"] = (
        df.groupby("supplier_id")["on_time_delivery"]
        .rolling(3).mean().reset_index(level=0, drop=True)
    )
    df["supplier_reliability_trend"] = df["supplier_reliability_trend"].fillna(
        df["supplier_on_time_rate"]
    )

# =============================
# 4. CARRIER FEATURES
# =============================
carrier_stats = df.groupby("carrier_name").agg(
    carrier_total_shipments=("on_time_delivery", "count"),
    carrier_on_time_rate=("on_time_delivery", "mean"),
    carrier_avg_delay=("delay_days", "mean")
)
carrier_stats["carrier_delay_rate"] = 1 - carrier_stats["carrier_on_time_rate"]

df = df.merge(carrier_stats, on="carrier_name", how="left")

# Carrier trend (rolling 5)
df["carrier_delay_trend"] = (
    df.groupby("carrier_name")["delay_days"]
    .rolling(5).mean().reset_index(level=0, drop=True)
)
df["carrier_delay_trend"] = df["carrier_delay_trend"].fillna(df["carrier_avg_delay"])

# =============================
# 5. REGION FEATURES
# =============================
region_stats = df.groupby("region").agg(
    region_on_time_rate=("on_time_delivery", "mean")
)
region_stats["region_difficulty_score"] = (1 - region_stats["region_on_time_rate"]) * 10

df = df.merge(region_stats, on="region", how="left")

# 6. LANE FEATURES

df["distance_bucket"] = pd.cut(
    df["shipping_distance_km"],
    bins=[0,100,300,600,1000,2000],
    labels=[1,2,3,4,5]
).astype(int)

df["lane_risk_score"] = df["distance_bucket"] + df["region_difficulty_score"]

df["lane_difficulty_v2"] = (
    df["lane_risk_score"] +
    df["carrier_delay_rate"] +
    df["distance_bucket"]
)


df["order_day_of_week"] = df["order_date"].dt.dayofweek
df["order_month"] = df["order_date"].dt.month
df["order_week"] = df["order_date"].dt.isocalendar().week.astype(int)
df["is_weekend"] = df["order_day_of_week"].isin([5,6]).astype(int)

df["order_date_days"] = (df["order_date"] - df["order_date"].min()).dt.days
df["promised_delivery_date_days"] = (
    df["promised_delivery_date"] - df["promised_delivery_date"].min()
).dt.days

df["promised_lead_gap"] = df["promised_delivery_date_days"] - df["order_date_days"]

# =============================
# 8. ADVANCED FEATURES (NOW SAFE)
# =============================
df["speed_km_per_day"] = df["shipping_distance_km"] / (df["supplier_lead_time"] + 1)
df["value_density"] = df["total_order_value"] / (df["shipping_distance_km"] + 1)

df["order_value_bucket"] = pd.qcut(
    df["total_order_value"], q=4, labels=[1,2,3,4]
).astype(int)

df["weather_risk"] = df["weather_condition"].isin(["rain","storm","snow"]).astype(int)
df["weather_distance_interaction"] = df["weather_risk"] * df["shipping_distance_km"]

df["is_start_of_week"] = (df["order_day_of_week"] == 0).astype(int)
df["is_end_of_week"] = (df["order_day_of_week"] >= 4).astype(int)

df["is_peak_season"] = df["order_month"].isin([10,11,12]).astype(int)
df["is_fiscal_year_end"] = (df["order_month"] == 3).astype(int)

df["log_distance"] = np.log1p(df["shipping_distance_km"])
df["log_order_value"] = np.log1p(df["total_order_value"])
df["log_unit_price"] = np.log1p(df["unit_price"])

# =============================
# 9. DROP LEAKAGE
# =============================
df = df.drop(columns=[
    "actual_delivery_date","delay_days","delivery_delay_days",
    "delay_flag","late_by_days","delivery_status"
], errors="ignore")

df = df.drop(columns=["order_id","supplier_id"], errors="ignore")

df = df.drop(columns=["order_date","promised_delivery_date"], errors="ignore")

# =============================
# 10. HANDLE MISSING
# =============================
categorical_cols = [
    'shipment_mode','weather_condition','region',
    'holiday_period','carrier_name','delayed_reason_code'
]

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

numeric_cols = [c for c in df.columns if c not in categorical_cols + ["on_time_delivery"]]
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# =============================
# 10B. SAVE FULL FEATURE SET FOR EDA (before encoding & scaling)
# =============================
df_eda = df.copy()
df_eda.to_excel("EDA_FEATURE_DATA.xlsx", index=False)
print("📊 Saved EDA_FEATURE_DATA.xlsx — ready for EDA & visualization!")

# =============================
# 11. ONE-HOT ENCODING
# =============================
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# =============================
# 12. SCALING
# =============================
scale_cols = [c for c in df.columns if c != "on_time_delivery"]
scaler = StandardScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])
joblib.dump(scaler, "scaler.pkl")
# =============================
# 13. TRAIN/TEST SPLIT
# =============================
X = df.drop(columns=["on_time_delivery"])
y = df["on_time_delivery"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
#
X_train.to_csv("X_train_clean.csv", index=False)
X_test.to_csv("X_test_clean.csv", index=False)
y_train.to_csv("y_train_clean.csv", index=False)
y_test.to_csv("y_test_clean.csv", index=False)

print("\n✨ Preprocessing with ALL HIGH-SIGNAL FEATURES Complete — Ready for Maximum ML Performance! ✨")
