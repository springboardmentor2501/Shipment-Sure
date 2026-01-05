# ===========================================
# COMPLETE EDA SCRIPT WITH ONE IMAGE OUTPUT
# ===========================================

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("EDA_FEATURE_DATA.xlsx")
# 1️⃣ Show all column names
print("\n===== ALL FEATURES (COLUMNS) =====")
for i, col in enumerate(df.columns):
    print(f"{i+1}. {col}")

# 2️⃣ Total number of features
print("\nTotal number of features:", df.shape[1])

# 3️⃣ Dataset shape
print("\nDataset shape (rows, columns):", df.shape)

print("\n🎉 All plots saved in ONE IMAGE: EDA_Combined_Plots.png")
