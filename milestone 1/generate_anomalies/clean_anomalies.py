import pandas as pd
import numpy as np

df = pd.read_excel("dataset_with_anomalies.xlsx")

# Remove rows with NULLs
df = df.dropna()

# Remove duplicates
df = df.drop_duplicates()

# Fix datatype mismatch (convert invalid strings to NaN then drop)
df["order_quantity"] = pd.to_numeric(df["order_quantity"], errors="coerce")
df = df.dropna(subset=["order_quantity"])

# Remove outliers
df = df[df["shipping_distance_km"] < 50000]

# Fix negative values
df = df[df["order_quantity"] >= 0]
df = df[df["unit_price"] >= 0]

# Remove impossible ratings
df = df[df["supplier_rating"] <= 5]

# Remove incorrect date rows
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
df["actual_delivery_date"] = pd.to_datetime(df["actual_delivery_date"], errors="coerce")
df = df[df["actual_delivery_date"] >= df["order_date"]]

# Remove non-standard region/carrier anomalies
df = df[~df["region"].str.contains("Norrth|Ameriica", na=False)]
df = df[~df["carrier_name"].str.contains("!!!", na=False)]

# Save clean dataset
df.to_excel("cleaned_dataset.xlsx", index=False)

print("Clean dataset created successfully.")
