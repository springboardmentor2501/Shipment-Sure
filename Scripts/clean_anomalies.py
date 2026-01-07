import pandas as pd
import numpy as np

# Load dataset with anomalies
df = pd.read_excel("dataset_with_anomalies.xlsx")

# 1️⃣ Fix Missing Values
for col in df.select_dtypes(include=['float', 'int']).columns:
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna("Unknown", inplace=True)

# 2️⃣ Remove Outliers using IQR
for col in df.select_dtypes(include=['int', 'float']).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] > upper, upper, df[col])
    df[col] = np.where(df[col] < lower, lower, df[col])

# 3️⃣ Fix Wrong Data Types
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col])
    except:
        pass

# 4️⃣ Remove Duplicates
df.drop_duplicates(inplace=True)

# 5️⃣ Fix inconsistent category values
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.replace(r"[^A-Za-z0-9 ]", "", regex=True)

# Save the cleaned dataset
df.to_excel("cleaned_dataset.xlsx", index=False)

print("cleaned_dataset.xlsx created successfully!")
