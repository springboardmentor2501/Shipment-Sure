import pandas as pd
import numpy as np

#load dataset
df = pd.read_excel('shipment_dataset_10000.xlsx')

#craeted a copy of dataset
df_anomaly = df.copy()

#missing value
for column in df_anomaly.columns:
    df_anomaly.loc[df_anomaly.sample(frac=0.02).index, column] = np.nan

#duplicate rows
duplicate = df_anomaly.sample(50)
df_anomaly = pd.concat([df_anomaly, duplicate], ignore_index=True)

#Mismatch datatype
df_anomaly.loc[df_anomaly.sample(20).index, 'order_quantity'] = "thirty"
df_anomaly.loc[df_anomaly.sample(20).index, 'shipping_distance_km'] = "1km"
df_anomaly.loc[df_anomaly.sample(20).index, 'unit_price'] = "two thousand"

#outliers
df_anomaly.loc[df_anomaly.sample(10).index, 'order_quantity'] = 173538
df_anomaly.loc[df_anomaly.sample(10).index, 'unit_price'] = 73256
df_anomaly.loc[df_anomaly.sample(10).index, 'shipping_distance_km'] = 900230

#logical errors

#Shipment date before order date
df_anomaly.loc[df_anomaly.sample(30).index, 'actual_delivery_date'] = \
    df_anomaly['order_date'] - pd.to_timedelta(np.random.randint(1, 10), unit='D')

# SAVE CORRUPTED DATASET

df_anomaly.to_excel("shipment_dataset_with_anomalies.xlsx", index=False)

print("Corrupted dataset created")