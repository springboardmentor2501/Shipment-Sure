🚚 ShipmentSure – On-Time Delivery Prediction System
📌 Project Overview

ShipmentSure is an end-to-end Machine Learning project designed to predict whether a shipment will be delivered on time based on order-level and logistics-related features. The system helps logistics and supply chain stakeholders proactively identify potential delivery delays and make data-driven decisions.

The project covers the complete ML lifecycle, including data analysis, preprocessing, model building, evaluation, and deployment through an interactive web application.

🎯 Problem Statement

Late deliveries negatively impact customer satisfaction and operational efficiency in logistics systems. This project aims to build a predictive model that estimates the probability of on-time delivery using historical shipment data, enabling early risk identification.

🧠 Machine Learning Approach

The solution follows a structured ML pipeline:

Exploratory Data Analysis to understand shipment patterns

Data preprocessing and feature engineering

Handling class imbalance using SMOTE

Model training with multiple algorithms

Hyperparameter tuning using GridSearchCV

Model evaluation using confusion matrix and ROC-AUC

Deployment via a Streamlit web interface

🛠️ Tech Stack

Programming Language: Python

Libraries: Pandas, NumPy, Scikit-learn

Imbalanced Learning: SMOTE

Model Tuning: GridSearchCV

Visualization: Matplotlib, Seaborn

Deployment: Streamlit

📊 Model Evaluation

The final model was evaluated using multiple metrics to ensure balanced performance on both delayed and on-time deliveries.

🔹 Confusion Matrix

The confusion matrix below shows the classification performance of the tuned Random Forest model after handling class imbalance.

🔹 ROC Curve

The ROC curve illustrates the trade-off between the true positive rate and false positive rate across different classification thresholds.

🌐 Web Application (Milestone 4)

A Streamlit-based web application was developed to make the model accessible to non-technical users.
Users can input shipment-related details such as supplier rating, lead time, shipment mode, weather condition, and holiday period to receive a predicted probability of on-time delivery.

🖥️ Streamlit Interface

Home Screen

<img width="1040" height="851" alt="Screenshot 2026-01-07 145338" src="https://github.com/user-attachments/assets/a2e24bc8-0d02-481c-bfe0-d63703141c5b" />

Prediction Output
<img width="938" height="841" alt="Screenshot 2026-01-07 145500" src="https://github.com/user-attachments/assets/44611457-e6f4-4df4-90b3-efa64a6f8529" />


✅ Key Outcomes

Built a reliable ML model for delivery delay prediction

Addressed real-world challenges like class imbalance

Deployed a user-friendly web interface for predictions

Created a modular and reusable ML pipeline

👩‍💻 Author

Vishakha Kolhe
Final Year B.Tech (CSE – AI & Analytics)
Infosys Springboard Internship Project – ShipmentSure

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
