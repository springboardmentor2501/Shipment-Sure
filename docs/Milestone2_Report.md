# 📘 Milestone 2 – Week 3–4
Exploratory Data Analysis (EDA), Preprocessing & Feature Engineering

## 🟦 1. Overview

Milestone 2 focuses on analyzing the cleaned dataset through Exploratory Data Analysis (EDA), performing data preprocessing, and building feature-engineered variables that improve machine learning performance.

This milestone ensures the dataset is fully prepared for Model Building (Milestone 3).

## 🟦 2. Exploratory Data Analysis (EDA)

### ✔ 2.1 Loading the Cleaned Dataset

The cleaned dataset generated from Milestone 1 (simple_cleaned_dataset.xlsx) was loaded and inspected.

```python
df = pd.read_excel("simple_cleaned_dataset.xlsx")
```

### ✔ 2.2 Dataset Overview

Performed the following:

- df.info() — checked data types
- df.describe() — statistical summary
- df.isnull().sum() — verified no missing values
- Confirmed cleaned data consistency

### ✔ 2.3 Target Variable Distribution

Checked distribution of the target variable on_time_delivery to understand class balance.

```python
df['on_time_delivery'].value_counts()
```

### ✔ 2.4 Visual EDA

Generated essential visualizations:
- Histograms of numerical columns
- Boxplot of delivery_days vs on_time_delivery
- Shipping distance distribution
- Supplier rating distribution
- Shipment mode frequency

### ✔ 2.5 Correlation Heatmap

Visualized relationships among numerical features.

```python
sns.heatmap(df.corr(), annot=False, cmap='Blues')
```

### Key EDA Insights

- Deliveries with higher delivery days tend to be delayed.
- Lower supplier ratings correlate with delays.
- Long-distance shipments show higher risk of late delivery.
- Some shipment modes (e.g., Air) had better on-time performance.

## 🟦 3. Preprocessing

To prepare for ML model building, key preprocessing steps were applied.

### ✔ 3.1 Encoding Categorical Variables

Converted categorical columns to numerical using One-Hot Encoding:

```python
df = pd.get_dummies(df, drop_first=True)
```

### ✔ 3.2 Scaling Numerical Features

Normalized numerical columns using StandardScaler:

```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()

   num_cols = ['delivery_days','order_quantity','shipping_distance_km']
   df[num_cols] = scaler.fit_transform(df[num_cols])
```
### ✔ 3.3 Train–Test Split

```python
   from sklearn.model_selection import train_test_split
   X = df.drop('on_time_delivery', axis=1)
   y = df['on_time_delivery']

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
)
```

This prepares the dataset for ML model training in Milestone 3.

## 🟦 4. Feature Engineering

Created additional meaningful features to enhance model accuracy.

### ✔ 4.1 Delivery Speed Category

Categorized delivery_days into speed groups:

```python
df['delivery_speed'] = pd.cut(
    df['delivery_days'],
    bins=[-1,2,5,10,100],
    labels=['Fast','Normal','Slow','Very_Slow']
)
df = pd.get_dummies(df, columns=['delivery_speed'], drop_first=True)
```

### ✔ 4.2 Long Distance Indicator

Flag shipments with unusually long distance:

```python
df['long_distance'] = (df['shipping_distance_km'] > 700).astype(int)
```

### ✔ 4.3 High Supplier Rating Indicator

```python
df['high_rating'] = (df['supplier_rating'] >= 4).astype(int)
```

These features help ML models capture delivery behavior more accurately.

## 🟦 5. Saving the Final Processed Dataset

The final transformed dataset was saved for use in Milestone 3:

```python
df.to_excel("processed_milestone2_dataset.xlsx", index=False)
```

This file becomes the input for model training and evaluation.

## 🟦 6. Deliverables for Milestone 2
| Deliverable | Status |
|-------------|--------|
| EDA Notebook (milestone2_eda.ipynb) | ✔ Completed |
| Visualizations & Insights | ✔ Completed |
| Encoded & Scaled Dataset | ✔ Completed |
| Feature Engineered Dataset | ✔ Completed |
| Final Processed File (processed_milestone2_dataset.xlsx) | ✔ Created |

## 🟦 7. Summary

Milestone 2 successfully completed the EDA, preprocessing, and feature engineering steps.
The dataset is now fully prepared for machine learning model building in Milestone 3.