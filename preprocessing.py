import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel("shipment_dataset_10000.xlsx")

categorical_cols = [
    'shipment_mode', 'weather_condition', 'region',
    'holiday_period', 'carrier_name', 'delayed_reason_code'
]
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

numeric_cols = [
    'supplier_rating','supplier_lead_time','shipping_distance_km',
    'order_quantity','unit_price','total_order_value',
    'previous_on_time_rate'
]
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Convert dates to days since earliest date
date_cols = ['order_date','promised_delivery_date','actual_delivery_date']
for col in date_cols:
    df[col] = (df[col] - df[col].min()).dt.days

# 3. ENCODE CATEGORICAL COLUMNS
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# FEATURE ENGINEERING
df['delivery_delay_days'] = df['actual_delivery_date'] - df['promised_delivery_date']
df['value_per_unit'] = df['total_order_value'] / df['order_quantity']
df['distance_per_lead'] = df['shipping_distance_km'] / df['supplier_lead_time']


scale_cols = [
    'supplier_rating','supplier_lead_time','shipping_distance_km',
    'order_quantity','unit_price','total_order_value',
    'previous_on_time_rate','delivery_delay_days',
    'value_per_unit','distance_per_lead'
]

scaler = StandardScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# heatmap
corr = df.corr()
plt.figure(figsize=(14,12))
plt.imshow(corr, aspect='auto')
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.colorbar()
plt.title("Correlation Heatmap - Shipment Dataset")
plt.tight_layout()
plt.show()


X = df.drop('on_time_delivery', axis=1)
y = df['on_time_delivery']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# FEATURE IMPORTANCE

drop_cols = ['order_id', 'supplier_id', 'delivery_delay_days']
X_train_imp = X_train.drop(columns=drop_cols, errors='ignore')

model = RandomForestClassifier(random_state=42)
model.fit(X_train_imp, y_train)

importances = model.feature_importances_
features = X_train_imp.columns

# Sort by importance
indices = np.argsort(importances)[::-1]
sorted_features = features[indices]
sorted_importances = importances[indices]

# Plot
plt.figure(figsize=(12, 10))
plt.barh(sorted_features[:20], sorted_importances[:20])
plt.gca().invert_yaxis()
plt.title("Top 20 Feature Importances (No IDs, No Leakage)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# Save dataset
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Preprocessing finished ")
