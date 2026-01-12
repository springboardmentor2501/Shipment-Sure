import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("shipment_dataset_10000 (1).xlsx")

print("Loaded dataset. First 5 rows:")
print(df.head())

print("\nDataset shape (rows, columns):", df.shape)
print("\nColumn types and non-null counts:")
print(df.info())

print("\nData types:")
print(df.dtypes)

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nStatistical summary (numeric columns):")
print(df.describe().round(3))

# -----------------------------
# Target distribution
# -----------------------------
if 'on_time_delivery' in df.columns:
    print("\nTarget variable counts (on_time_delivery):")
    print(df['on_time_delivery'].value_counts())

    sns.countplot(x='on_time_delivery', data=df)
    plt.title("Target distribution (on_time_delivery)")
    plt.savefig("eda_target_distribution.png")
    plt.close()

    print("Saved 'eda_target_distribution.png'")
else:
    print("\nTarget column not found.")

# -----------------------------
# Numeric distributions
# -----------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(f"eda_hist_{col}.png")
    plt.close()

# -----------------------------
# Categorical distributions
# -----------------------------
cat_cols = df.select_dtypes(include=["object"]).columns

for col in cat_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, data=df)
    plt.title(f"Counts of {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"eda_count_{col}.png")
    plt.close()

# -----------------------------
# Correlation heatmap (numeric only)
# -----------------------------
plt.figure(figsize=(10, 8))
num_df = df.select_dtypes(include=[np.number])

corr = num_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Numeric Feature Correlation")
plt.tight_layout()
plt.savefig("eda_correlation_heatmap.png")
plt.close()

print("Saved correlation heatmap")

# -----------------------------
# EDA Summary
# -----------------------------
summary = pd.DataFrame({
    "column": df.columns,
    "dtype": df.dtypes.astype(str),
    "missing_values": df.isnull().sum().values
})
summary.to_csv("eda_summary.csv", index=False)
print("Saved eda_summary.csv")

# -----------------------------
# 🔥 CORRELATION WITH TARGET (FIXED)
# -----------------------------
print("\nTop correlations with target (on_time_delivery):")

corr_target = (
    df.select_dtypes(include=[np.number])
      .corr()["on_time_delivery"]
      .sort_values(ascending=False)
)

print(corr_target.head(10))

print("\nEDA complete.")
