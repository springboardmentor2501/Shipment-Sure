import pandas as pd
import numpy as np

# Load dataset with anomalies
df = pd.read_excel("shipment_dataset_with_anomalies.xlsx")

# fix mismatch
numeric_columns = ['order_quantity', 'shipping_distance_km', 'unit_price']

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # convert invalid to NaN

# handle missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# duplicates
df = df.drop_duplicates()

# outliers
def outliers(col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 - 1.5 * IQR
    df[col] = np.where(df[col] > upper, upper,
                       np.where(df[col] < lower, lower, df[col]))

for col in numeric_columns:
    outliers(col)

#logical errors
mask = df['actual_delivery_date'] < df['order_date']
df.loc[mask, 'actual_delivery_date'] = df.loc[mask, 'order_date'] + pd.Timedelta(days=3)

#save dataset
df.to_excel("shipment_dataset_cleaning_done.xlsx", index=False)

print("Data cleaned and saved successfully")