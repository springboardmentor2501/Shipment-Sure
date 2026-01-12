# ShipmentSure – On-Time Delivery Prediction

## Project Overview
Machine Learning system to predict shipment on-time delivery using order-level features.

## Tech Stack
- Python
- Scikit-learn
- Streamlit
- SMOTE
- GridSearchCV

---

## Web Application Interface

The following screenshot shows the Streamlit-based web application developed as part of Milestone 4.  
The interface allows users to enter shipment-level details and predicts the probability of on-time delivery.

![Streamlit App](streamlit_app.png)

---

## Model Evaluation Results

The confusion matrix below represents the performance of the Random Forest model after handling class imbalance and hyperparameter tuning.

![Random Forest Confusion Matrix](rf_confusion.png)

The ROC curve illustrates the trade-off between true positive rate and false positive rate for the trained model.

![ROC Curve](roc_curve.png)



## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
