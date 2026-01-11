🚚 ShipmentSure
Predicting On-Time Delivery Using Supplier & Logistics Data

ShipmentSure is an end-to-end machine learning project that predicts whether a shipment will be delivered on time or delayed, based on supplier performance, shipping conditions, and order characteristics.
The model is trained on real-world–style logistics data and deployed as an interactive Streamlit web application.

🎯 Objective

To build a supervised binary classification system that helps logistics and manufacturing companies:

Predict delivery delays in advance

Identify factors affecting on-time delivery

Improve operational planning and supplier evaluation

📊 Dataset

Source: Supply Chain / Shipment Dataset

Records: ~10,000 shipment orders

Target Variable: on_time_delivery

1 → On-time delivery

0 → Delayed delivery

Key Features

Supplier rating and lead time

Shipping distance and order quantity

Shipment mode and carrier

Weather conditions and regional factors

Historical on-time delivery performance

🧠 Problem Type

Data Type: Tabular / Structured Data

Learning Type: Supervised Learning

Task: Binary Classification

⚙️ Tech Stack
Component	Tools Used
Programming	Python
Data Handling	Pandas, NumPy
Modeling	scikit-learn (Random Forest)
Preprocessing	One-Hot Encoding, Pipelines
Deployment	Streamlit
Hosting	Hugging Face Spaces
🔄 Project Workflow

Data loading and cleaning

Feature selection and preprocessing

Automated encoding of categorical variables

Model training using Random Forest

Model evaluation with classification metrics

Model serialization (.pkl)

Deployment as an interactive web app

📈 Model Details

Algorithm: Random Forest Classifier

Why Random Forest?

Handles non-linear relationships effectively

Works well with mixed numerical and categorical data

Robust to noise and reduces overfitting

Evaluation Metrics

Accuracy

Precision

Recall

F1-Score

🖥️ Web Application

The deployed Streamlit application allows users to:

Upload a shipment dataset (CSV or Excel)

Generate real-time predictions for each order

Instantly view on-time vs delayed shipment status


📁 Project Structure
ShipmentSure/
│
├── app.py                  # Streamlit application
├── shipmentsure_model.pkl  # Trained ML model
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation

🚀 Deployment

The application is deployed using Hugging Face Spaces with the Streamlit SDK.
All dependencies are managed via requirements.txt.

🧪 Run Locally
pip install -r requirements.txt
streamlit run app.py

📌 Key Learnings

Building an end-to-end machine learning pipeline

Handling real-world logistics and supply chain data

Automated preprocessing using scikit-learn pipelines

Model deployment and basic MLOps practices

👩‍💻 Author

Ashwika
Machine Learning & Data Science Enthusiast
