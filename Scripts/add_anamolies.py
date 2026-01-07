import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_excel("shipment_dataset_10000.xlsx")

# Make a copy
df_anomaly = df.copy()

# 1️⃣ Introduce Missing Values
for col in df_anomaly.select_dtypes(include=['float', 'int']).columns:
    df_anomaly.loc[np.random.choice(df_anomaly.index, 10), col] = np.nan

# 2️⃣ Introduce Outliers
for col in df_anomaly.select_dtypes(include=['int', 'float']).columns:
    df_anomaly.loc[np.random.choice(df_anomaly.index, 5), col] = df_anomaly[col].mean() * 10

# 3️⃣ Wrong Data Types
num_cols = df_anomaly.select_dtypes(include=['int', 'float']).columns
if len(num_cols) > 0:
    df_anomaly.loc[5, num_cols[0]] = "WrongValue"

# 4️⃣ Duplicate Rows
df_anomaly = pd.concat([df_anomaly, df_anomaly.iloc[:5]], ignore_index=True)

# 5️⃣ Inconsistent category values
cat_cols = df_anomaly.select_dtypes(include=['object']).columns
if len(cat_cols) > 0:
    df_anomaly.loc[10, cat_cols[0]] = "??INVALID??"

# Save new dataset
df_anomaly.to_excel("dataset_with_anomalies.xlsx", index=False)

print("dataset_with_anomalies.xlsx created successfully!")
