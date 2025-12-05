# 🚚 ShipmentSure – AI-Powered Delivery Time Prediction

An end-to-end Machine Learning project that predicts whether a shipment will be delivered On-Time or Delayed.

**🔥 Built by:** Pranav Ghorpade  
**⭐ Internship Project – Infosys Springboard AI/ML (2025)**

## 📌 Overview

ShipmentSure is an AI-powered predictive analytics system that helps logistics companies estimate whether a shipment will arrive on time.

This project follows a complete ML pipeline:

1️⃣ Data Cleaning (handling missing values, duplicates, anomalies)  
2️⃣ Feature Engineering  
3️⃣ Model Training (XGBoost, Random Forest, Logistic Regression)  
4️⃣ Model Evaluation  
5️⃣ Deployment using Streamlit Web App

## 📂 Project Structure

```graphql
ShipmentSure/
│
├── app/
│ └── streamlit_app.py # Streamlit UI for prediction
│
├── data/
│ ├── simple_cleaned_dataset.xlsx
│ ├── simple_anomalies_dataset.xlsx
│ ├── processed_milestone2_dataset.xlsx
│ └── shipment_dataset_10000.xlsx
│
├── scripts/
│ ├── add_anomalies.py
│ ├── clean_anomalies.py
│ └── model_training.ipynb
│
├── docs/
│ ├── Milestone1_Report.md
│ ├── Milestone2_Report.md
│ ├── Milestone3_Report.md
│ └── Milestone4_Report.md
│
├── best_model.pkl # Saved XGBoost model + feature list
├── README.md # Project documentation
├── requirements.txt
└── LICENSE
```

## 🎯 Objective

To build a reliable ML model that predicts:

➡ On-Time Delivery (1)  
➡ Delayed Delivery (0)

using real-world shipment data containing:

- supplier information
- shipping mode
- carrier details
- delivery speed
- weather conditions
- engineered fields like delivery_days & order value

## 🧹 Milestone 1 – Data Cleaning

✔ Removed missing values  
✔ Handled duplicates  
✔ Fixed anomalies  
✔ Exported cleaned dataset

## 🔍 Milestone 2 – EDA

Performed Exploratory Data Analysis:

📌 Distribution plots  
📌 Histograms & correlations  
📌 Key insights about delays  
📌 Feature relationships

Generated engineered columns:

- delivery_days
- total_order_value
- long_distance
- high_rating

## 🤖 Milestone 3 – Model Building & Evaluation

Trained 3 models:

| Model | Performance |
|-------|-------------|
| Logistic Regression | Baseline |
| Random Forest | Better |
| XGBoost Classifier | ⭐ Best Model |

Saved model:  

```r
`best_model.pkl` → (model + 32 feature names + metrics)
```

## 💻 Milestone 4 – Model Deployment (Streamlit App)

A modern UI was built using Streamlit:

🌟 Features:

- Real-time delivery prediction
- 32-feature preprocessing & one-hot encoding
- Probability score (0–100%)
- Clean dark theme UI
- Sidebar navigation
- Model Info Page – Shows model type + features
- EDA Preview Page – Shows dataset sample, summary stats, distribution plots

## 🎥 Workflow

```pgsql
User Inputs
↓
One-Hot Encoding + Preprocessing
↓
Load Saved Model
↓
Predict Probability
↓
Display Result on UI
```

## 🚀 Run the App

```arduino
streamlit run app/streamlit_app.py
```

## 📦 Tech Stack
### Languages
- Python

### Libraries
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Matplotlib / Seaborn
- Joblib

### Tools
- VS Code
- Git & GitHub
- Excel

## 📘 Key Files
| File | Description |
|------|-------------|
| streamlit_app.py | Main application UI |
| best_model.pkl | XGBoost model used for deployment |
| model_training.ipynb | Model building notebook |
| clean_anomalies.py | Data cleaning script |
| add_anomalies.py | Synthetic anomaly-generation |
| /docs/*.md | Milestone reports |

## 📊 Results

- The final XGBoost model provides high prediction accuracy.

- The system correctly identifies risk of shipment delays.

- The Streamlit UI provides intuitive real-time predictions.

## 🏁 Conclusion

ShipmentSure demonstrates a complete machine-learning lifecycle:

✔ Data →  
✔ EDA →  
✔ Feature Engineering →  
✔ Model Training →  
✔ Deployment →  
✔ Real-time Prediction

A perfect industry-level project for logistics, supply chain analytics, and AI-based forecasting systems.

## 📬 Contact

**Pranav Ghorpade**  
📧 pranavghorpade82@gmail.com  
🔗 GitHub: Pranav-0440