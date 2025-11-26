# 📘 Milestone 1 – Week 1–2
Data Understanding, Anomaly Generation & Data Cleaning

## 🟦 1. Overview

Milestone 1 focuses on understanding the dataset, analyzing its structure, generating real-world anomalies artificially, and finally cleaning the dataset to prepare it for further processing (EDA, Feature Engineering, and ML modeling).

This milestone ensures the data is consistent, accurate, and reliable for analysis and modeling.

## 🟦 2. Dataset Understanding
### ✔ Dataset Loaded

- File: shipment_dataset_10000.xlsx
- Contains ~10,000 rows and multiple supplier, shipment, and logistics-related fields.

### ✔ Tasks Performed

- Reviewed dataset schema
- Checked data types for all columns
- Inspected numeric and categorical distributions
- Validated date columns (order → promised → actual delivery)
- Confirmed there were no initial anomalies in the original dataset

### ✔ Key Columns

- order_id, supplier_id, supplier_rating
- order_date, actual_delivery_date, promised_delivery_date
- shipment_mode, shipping_distance_km
- order_quantity, unit_price, total_order_value
- on_time_delivery (target variable)

<img width="2710" height="1037" alt="Dataset-Understanding" src="https://github.com/user-attachments/assets/6079c9a9-eefa-4336-95c7-5ac3f78e9b6d" />

## 🟦 3. Anomaly Generation (add_anomalies.py)

To simulate real-world supply chain issues, multiple anomalies were added intentionally.

### ✔ Added Missing Values

Random NaN values in:

- supplier_rating
- order_quantity
- other key columns  

### ✔ Added Duplicate Rows

- Random duplicate entries inserted
- Matches real-world data merging issues

### ✔ Added Datatype Errors

- Inserted "error_value" in numeric columns like order_quantity
- Simulates wrong data entry

### ✔ Added Outliers

- Set unrealistic values such as
  - shipping_distance_km = 99999
  - extreme order quantities

### ✔ Added Wrong Dates

- Modified actual_delivery_date to be earlier than order_date
- Represents invalid business data

### 📄 Output File:

simple_anomalies_dataset.xlsx

<img width="2735" height="1111" alt="Anomaly-Generation" src="https://github.com/user-attachments/assets/bc060696-2c52-4abb-8356-be1808a70acc" />

## 🟦 4. Anomaly Cleaning (clean_anomalies.py)

After generating anomalies, the dataset was cleaned using systematic preprocessing.

### ✔ Removed Duplicate Records

- drop_duplicates() applied

### ✔ Fixed Datatype Mismatches

- Converted "error_value" → NaN → numeric

### ✔ Filled Missing Values

- Median for numeric columns
- Mode for categorical columns

### ✔ Corrected Wrong Dates

Rows where
```nginx
    actual_delivery_date < order_date
```
were removed

### ✔ Treated Outliers

- Applied IQR-based capping on fields like shipping_distance_km

### ✔ Engineered New Feature

- delivery_days = actual_delivery_date - order_date

### 📄 Output File:

simple_cleaned_dataset.xlsx

<img width="2816" height="760" alt="Anomaly-Cleaning" src="https://github.com/user-attachments/assets/29b37094-9dac-4fc6-bfde-cfcad6f8d926" />

## 🟦 5. Validation of Cleaning

Proper before/after comparisons were performed:

### ✔ Duplicate count comparison

- Original: 0
- After anomalies: >0
- After cleaning: 0

### ✔ Missing values comparison

- Missing values increased after anomalies
- Missing values reduced after cleaning

### ✔ Datatype issues fixed

- "error_value" removed
- Numeric columns restored

### ✔ Date issues fixed

- No rows have invalid delivery dates

### ✔ Outliers capped

- Extreme values replaced with upper IQR limit
- Dataset is now fully ready for EDA (Milestone 2).

<img width="2816" height="1202" alt="Validation-of-Cleaning" src="https://github.com/user-attachments/assets/53296eec-6c66-4b8d-a024-c116f4159f69" />

## 🟦 6. Deliverables for Milestone 1
| Deliverable | Status |
|-------------|--------|
| Original dataset | ✔ Completed |
| Dataset with anomalies | ✔ Completed |
| Cleaned dataset | ✔ Completed |
| Anomaly scripts (add_anomalies.py, clean_anomalies.py) | ✔ Completed |
| Validation of before/after dataset changes | ✔ Completed |

## 🟦 7. Summary

Milestone 1 successfully covered data understanding, anomaly generation, and anomaly cleaning.
The final cleaned dataset is now ready for Exploratory Data Analysis (EDA) in Milestone 2.
