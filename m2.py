# %%
# %%
"""
=============================================================================
SHIPMENTSURE PROJECT
MILESTONE 2: DATA PREPROCESSING AND FEATURE ENGINEERING
=============================================================================
Author: Ashwika K
=============================================================================
"""

# %%
# =========================
# IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)

print("MILESTONE 2 STARTED")

# %%
# =========================
# STEP 1: LOAD DATASET
# =========================
file_path = "shipment_dataset_10000 -1.xlsx"
df = pd.read_excel(file_path)

print("Dataset Loaded")
print("Shape:", df.shape)
print(df.head())

df_original = df.copy()
df_processed = df.copy()

# %%
# =========================
# STEP 2: HANDLE MISSING VALUES
# =========================
print("Missing values before:")
print(df_processed.isnull().sum())

num_cols = df_processed.select_dtypes(include=["int64", "float64"]).columns
cat_cols = df_processed.select_dtypes(include=["object"]).columns

for col in num_cols:
    df_processed[col].fillna(df_processed[col].median(), inplace=True)

for col in cat_cols:
    df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)

print("Missing values after:", df_processed.isnull().sum().sum())

# %%
# =========================
# STEP 3: REMOVE DUPLICATES
# =========================
duplicates = df_processed.duplicated().sum()
df_processed.drop_duplicates(inplace=True)
print("Duplicates removed:", duplicates)

# %%
# =========================
# STEP 4: CATEGORICAL ENCODING
# =========================

# Target column
target_col = "Reached.on.Time_Y.N"

# Label Encoding - Gender
if "Gender" in df_processed.columns:
    le = LabelEncoder()
    df_processed["Gender_Encoded"] = le.fit_transform(df_processed["Gender"])
    df_processed.drop("Gender", axis=1, inplace=True)

# One-hot encoding
df_processed = pd.get_dummies(
    df_processed,
    columns=["Mode_of_Shipment", "Product_importance", "Warehouse_block"],
    drop_first=True
)

print("Encoding completed")
print("Shape:", df_processed.shape)

# %%
# =========================
# STEP 5: FEATURE ENGINEERING
# =========================

df_processed["Cost_Weight_Ratio"] = (
    df_processed["Cost_of_the_Product"] / (df_processed["Weight_in_gms"] + 1)
)

df_processed["High_Discount"] = (
    df_processed["Discount_offered"] > df_processed["Discount_offered"].median()
).astype(int)

df_processed["Frequent_Customer"] = (
    df_processed["Prior_purchases"] > df_processed["Prior_purchases"].median()
).astype(int)

df_processed["High_Value_Product"] = (
    df_processed["Cost_of_the_Product"] >
    df_processed["Cost_of_the_Product"].quantile(0.75)
).astype(int)

df_processed["Customer_Engagement"] = (
    df_processed["Customer_care_calls"] + df_processed["Customer_rating"]
)

print("Feature engineering completed")

# %%
# =========================
# STEP 6: REMOVE IRRELEVANT COLUMNS
# =========================
if "ID" in df_processed.columns:
    df_processed.drop("ID", axis=1, inplace=True)

print("Final feature count:", df_processed.shape[1])

# %%
# =========================
# STEP 7: CORRELATION ANALYSIS
# =========================
corr = df_processed.corr()

plt.figure(figsize=(16, 12))
sns.heatmap(corr, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# %%
# =========================
# STEP 8: TRAIN TEST SPLIT
# =========================
X = df_processed.drop(target_col, axis=1)
y = df_processed[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# %%
# =========================
# STEP 9: FEATURE SCALING
# =========================
scaler = StandardScaler()
num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

print("Scaling completed")

# %%
# =========================
# STEP 10: SAVE FILES
# =========================
df_processed.to_csv("preprocessed_data_full.csv", index=False)
X_train.assign(**{target_col: y_train}).to_csv("train_data_scaled.csv", index=False)
X_test.assign(**{target_col: y_test}).to_csv("test_data_scaled.csv", index=False)

print("Files saved successfully")

# %%
# =========================
# MILESTONE 2 COMPLETE
# =========================
print("MILESTONE 2 COMPLETED SUCCESSFULLY")



