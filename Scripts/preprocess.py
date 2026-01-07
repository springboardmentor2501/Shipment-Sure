import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# ---------------------------
# 1. LOAD THE DATASET
# ---------------------------
df = pd.read_excel("shipment_dataset_10000.xlsx")

# ---------------------------
# 2. ENCODE CATEGORICAL COLUMNS
# ---------------------------
categorical_columns = ["carrier_name", "holiday_period", "delayed_reason_code"]

encoder = OneHotEncoder(drop="first", sparse_output=False)

encoded_data = encoder.fit_transform(df[categorical_columns])

encoded_df = pd.DataFrame(
    encoded_data,
    columns=encoder.get_feature_names_out(categorical_columns)
)

df = pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1)

# ---------------------------
# 3. NORMALIZE NUMERICAL COLUMNS
# ---------------------------
numeric_columns = [
    "supplier_rating",
    "supplier_lead_time",
    "shipping_distance_km",
    "order_quantity",
    "unit_price",
    "total_order_value",
    "previous_on_time_rate"
]

scaler = MinMaxScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# ---------------------------
# 4. FEATURE ENGINEERING
# ---------------------------
df["cost_to_distance_ratio"] = df["total_order_value"] / (df["shipping_distance_km"] + 1)

# ---------------------------
# 5. HEATMAP (ONLY NUMERIC COLUMNS)
# ---------------------------
plt.figure(figsize=(18, 10))

# Only include numeric columns for correlation
numeric_df = df.select_dtypes(include=['float64', 'int64'])

corr = numeric_df.corr()

sns.heatmap(corr, annot=False, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Processed Shipment Dataset", fontsize=16)

plt.tight_layout()
plt.savefig("correlation_heatmap.png")  # Saves heatmap in the folder
plt.show()

# ---------------------------
# 6. TRAIN–TEST SPLIT
# ---------------------------

target = "on_time_delivery"

X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 7. SAVE OUTPUT FILES
# ---------------------------
df.to_csv("processed_full_dataset.csv", index=False)
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

