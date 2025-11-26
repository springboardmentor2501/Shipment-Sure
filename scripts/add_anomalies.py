import pandas as pd
import numpy as np

# Load original dataset
df = pd.read_excel("shipment_dataset_10000.xlsx")
df_anom = df.copy()

print("Adding simple anomalies...")

# 1. Add some missing values
df_anom.loc[df_anom.sample(20).index, 'supplier_rating'] = np.nan
df_anom.loc[df_anom.sample(20).index, 'order_quantity'] = np.nan

# 2. Add duplicate rows
df_anom = pd.concat([df_anom, df_anom.sample(10)], ignore_index=True)

# 3. Add datatype mismatch
df_anom.loc[df_anom.sample(10).index, 'order_quantity'] = "error_value"

# 4. Add outliers
df_anom.loc[df_anom.sample(10).index, 'shipping_distance_km'] = 99999

# 5. Add wrong dates (delivery before order)
df_anom.loc[df_anom.sample(10).index, 'actual_delivery_date'] = (
    df_anom['order_date'] - pd.Timedelta(days=3)
)

# Save file
df_anom.to_excel("simple_anomalies_dataset.xlsx", index=False)

print("Anomaly file created: simple_anomalies_dataset.xlsx")
