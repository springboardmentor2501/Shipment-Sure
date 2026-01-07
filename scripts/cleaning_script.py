import pandas as pd
import numpy as np
import re

# ---- File paths ----
input_file = r"C:\Users\mldar\OneDrive\Desktop\Infosys\subset_with_anomalies_100.xlsx"
output_file = r"C:\Users\mldar\OneDrive\Desktop\Infosys\cleaned_subset_100.xlsx"

# ---- Load ----
df = pd.read_excel(input_file)

# ---- Convert dirty number strings ----
def clean_num(x):
    if isinstance(x, (int, float)): 
        return x
    s = str(x)
    match = re.search(r"-?\d+(\.\d+)?", s)
    return float(match.group()) if match else np.nan

num_cols = ["Weight_in_gms", "Cost_of_the_Product", 
            "Customer_rating", "supplier_rating",
            "shipping_distance_km"]

for col in num_cols:
    if col in df.columns:
        df[col] = df[col].apply(clean_num)

# ---- Outlier capping ----
for col in ["Weight_in_gms", "Cost_of_the_Product", "shipping_distance_km"]:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)

# ---- Missing value treatment ----
if "Weight_in_gms" in df:
    df["Weight_in_gms"].fillna(df["Weight_in_gms"].median(), inplace=True)

if "supplier_rating" in df:
    df["supplier_rating"].fillna(df["supplier_rating"].mean(), inplace=True)

if "Customer_rating" in df:
    df["Customer_rating"].fillna(df["Customer_rating"].mode()[0], inplace=True)

if "shipping_distance_km" in df:
    df["shipping_distance_km"].fillna(df["shipping_distance_km"].median(), inplace=True)

# ---- Remove duplicates ----
df.drop_duplicates(inplace=True)

# ---- Save cleaned dataset ----
df.to_excel(output_file, index=False)
print("Cleaning completed!")
print("Cleaned file saved to:", output_file)
