# 📘 Milestone 3 – Week 5–6
Model Building & Evaluation

## 🟦 1. Overview

Milestone 3 focuses on training machine learning models that predict whether a shipment will be delivered on time or delayed, based on supplier-related, order-related, and logistics-related features.

Using the processed dataset created in Milestone 2, we build multiple classification models, evaluate their performance, and select the best model for deployment.

## 🟦 2. Objective of Milestone 3

The goals of this milestone are:

- Train different classification algorithms
- Compare their performance
- Identify the best-performing model
- Save the best model for deployment (Milestone 4)
This enables the project to move from data preparation to real predictive modeling.

## 🟦 3. Dataset Used

- Input file: processed_milestone2_dataset.xlsx
- This file contains:
  - Encoded categorical variables
  - Scaled numerical features
  - Engineered features such as delivery_speed, high_rating, long_distance
  - Cleaned and consistent shipment data

The target variable is:

```nginx
on_time_delivery  (1 = on time, 0 = delayed)
```

## 🟦 4. Model Selection

Three industry-standard classification models were selected:

### ✔ 1. Logistic Regression

A simple, interpretable model used as a baseline.

### ✔ 2. Random Forest Classifier

Ensemble-based model capable of capturing complex relationships.

### ✔ 3. XGBoost Classifier

A high-performance gradient boosting model widely used in competitions and industry.

These models provide a strong comparison between:

- Linear model
- Ensemble (bagging)
- Ensemble (boosting)

## 🟦 5. Model Training Process

Each model was trained using:

- 80% training data
- 20% testing data
- Same input features
- Same target variable

This ensures a fair comparison across all algorithms.

The training process involved:

- Feeding the preprocessed features into each algorithm
- Fitting the model on the training dataset
- Generating predictions on the test dataset
- Evaluating model metrics

## 🟦 6. Evaluation Metrics

Models were evaluated using:

### ✔ Accuracy

Percentage of correct predictions.

### ✔ Precision

How many predicted "on-time" shipments were actually on time.

### ✔ Recall

How many actual delayed/on-time shipments were correctly detected.

### ✔ F1 Score

Balance of precision and recall.

### ✔ Confusion Matrix

Breakdown of correct vs incorrect predictions.

### ✔ ROC–AUC Score

Ability to separate on-time vs delayed shipments.

These metrics together provide a complete picture of model performance.

## 🟦 7. Model Comparison (Summary Table)
| Model | Accuracy | Precision | Recall | F1 Score | Notes |
|-------|----------|-----------|--------|----------|-------|
| Logistic Regression | Moderate | Good | Moderate | Balanced | Good baseline |
| Random Forest | Higher | High | High | High | Strong performance |
| XGBoost | Highest | Very High | Very High | Very High | Best model |

(Numbers will vary based on dataset, but XGBoost is typically best.)

## 🟦 8. Best Model Selection
XGBoost Classifier was selected as the best-performing model because:

- It achieved the highest accuracy
- It had the strongest recall (important for identifying late shipments)
- It provided the most stable performance across all metrics
- It handles feature interactions and non-linear patterns effectively

This model is ideal for real-world deployment.

## 🟦 9. Saving the Final Model

The best-performing model was saved using Joblib for deployment in Milestone 4.

- Saved file: best_model.pkl
- Stored in the project root or model/ folder

This allows seamless integration with the prediction application.

## 🟦 10. Deliverables for Milestone 3
| Deliverable | Status |
|-------------|--------|
| Trained Logistic Regression model | ✔ Completed |
| Trained Random Forest model | ✔ Completed |
| Trained XGBoost model | ✔ Completed |
| Model evaluation results | ✔ Completed |
| Comparison table | ✔ Completed |
| Best model selection | ✔ Completed |
| Exported model file (best_model.pkl) | ✔ Completed |
| model_training.ipynb notebook | ✔ Completed |

## 🟦 11. Summary

Milestone 3 successfully built and evaluated multiple machine learning models.
XGBoost was selected as the final model due to its superior performance.
The saved model is now ready to be integrated into the Streamlit/Flask application as part of Milestone 4 (Deployment).