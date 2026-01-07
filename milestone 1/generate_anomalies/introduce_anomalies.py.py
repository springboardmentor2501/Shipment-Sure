import pandas as pd

# Load original dataset
df = pd.read_excel("shipment_dataset_10000.xlsx")

# --- Insert anomalies in first 10 rows ---

# 1. Null value
df.loc[0, "unit_price"] = None

# 2. Duplicate row (copy of row 3)
df.loc[1] = df.loc[2]

# 3. Datatype mismatch
df.loc[2, "order_quantity"] = "fifty"

# 4. Outlier value
df.loc[3, "shipping_distance_km"] = 99999

# 5. Wrong date (actual < order)
df.loc[4, "order_date"] = "2025-12-10"
df.loc[4, "actual_delivery_date"] = "2025-11-10"

# 6. Wrong region spelling
df.loc[5, "region"] = "Norrth Ameriica"

# 7. Wrong carrier name
df.loc[6, "carrier_name"] = "DHL!!!"

# 8. Negative quantity
df.loc[7, "order_quantity"] = -20

# 9. Negative price
df.loc[8, "unit_price"] = -500

# 10. Impossible rating value
df.loc[9, "supplier_rating"] = 500

# Save the new file with anomalies
df.to_excel("dataset_with_anomalies.xlsx", index=False)

print("Anomalies added successfully in rows 1–10.")

