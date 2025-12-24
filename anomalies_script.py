import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. READ ORIGINAL DATASET
# --------------------------------------------------

file_name = "enhanced_on_time_delivery_dataset.xlsx"  # keep this file in the same folder as this script

df = pd.read_excel(file_name)
print("Original shape:", df.shape)

# Detect numeric columns automatically
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
print("Numeric columns:", numeric_cols)

# Make a copy to add anomalies (keep original safe)
df_anom = df.copy()


# --------------------------------------------------
# 2. INTRODUCE ANOMALIES
# --------------------------------------------------

# 2.1 Add Missing Values (NaN) to first 3 numeric columns
for col in numeric_cols[:3]:
    # pick up to 10 random rows (or less if dataset is smaller)
    idx = np.random.choice(df_anom.index, size=min(10, len(df_anom)), replace=False)
    df_anom.loc[idx, col] = np.nan

# 2.2 Add Outliers (very large values) to first numeric column
if numeric_cols:
    col0 = numeric_cols[0]
    idx = np.random.choice(df_anom.index, size=min(5, len(df_anom)), replace=False)
    # multiply existing values at those rows by 10 to create outliers
    df_anom.loc[idx, col0] = df_anom.loc[idx, col0] * 10

# 2.3 Add wrong datatype values (strings) to second numeric column
if len(numeric_cols) > 1:
    col1 = numeric_cols[1]

    # convert the whole column to object so strings are allowed (avoids dtype warning)
    df_anom[col1] = df_anom[col1].astype(object)

    # insert bad string values in first two rows of that column
    df_anom.loc[df_anom.index[0], col1] = "wrong_value"
    df_anom.loc[df_anom.index[1], col1] = "abc"

# Save dataset with anomalies
df_anom.to_excel("dataset_with_anomalies.xlsx", index=False)
print("Saved: dataset_with_anomalies.xlsx")


# --------------------------------------------------
# 3. CLEAN THE ANOMALIES
# --------------------------------------------------

clean_df = df_anom.copy()

# 3.1 Fix incorrect datatypes (convert strings to NaN in numeric columns)
for col in numeric_cols:
    clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

# 3.2 Fill missing values with median value (no inplace to avoid warnings)
for col in numeric_cols:
    median_value = clean_df[col].median()
    clean_df[col] = clean_df[col].fillna(median_value)

# 3.3 Remove / Cap Outliers using IQR method
for col in numeric_cols:
    Q1 = clean_df[col].quantile(0.25)
    Q3 = clean_df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # cap values outside [lower, upper]
    clean_df[col] = np.where(
        clean_df[col] > upper,
        upper,
        np.where(clean_df[col] < lower, lower, clean_df[col]),
    )

# Save cleaned dataset
clean_df.to_excel("cleaned_dataset.xlsx", index=False)
print("Saved: cleaned_dataset.xlsx")
print("Done!")
