import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve
)

import xgboost as xgb

# ===============================
# Load Feature Engineered Dataset
# ===============================
data = pd.read_excel("feature_eng_dataset.xlsx")

# ===============================
# Define Features and Target
# ===============================
TARGET_COL = "delivery_delay_days"

features = data.drop(columns=[TARGET_COL])
target = data[TARGET_COL]

# Retain only numeric and boolean columns
features = features.select_dtypes(include=["number", "bool"])

# Convert delay days into binary outcome
# 0 -> On-time | 1 -> Delayed
target = (target > 0).astype(int)

print("Class distribution:")
print(target.value_counts())

# ===============================
# Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=42,
    stratify=target
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ===============================
# Model Definitions
# ===============================
model_registry = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )
}

# ===============================
# Hyperparameter Search Space
# ===============================
hyperparams = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1]
    },
    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10]
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 6],
        "learning_rate": [0.05, 0.1]
    }
}

# ===============================
# Training & Evaluation
# ===============================
evaluation_results = []
trained_models = {}

for model_name in model_registry:
    print(f"\nFitting model: {model_name}")

    grid_search = GridSearchCV(
        estimator=model_registry[model_name],
        param_grid=hyperparams[model_name],
        scoring="roc_auc",
        cv=3,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_estimator = grid_search.best_estimator_
    trained_models[model_name] = best_estimator

    predictions = best_estimator.predict(X_test)
    probabilities = best_estimator.predict_proba(X_test)[:, 1]

    evaluation_results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1-Score": f1_score(y_test, predictions),
        "ROC-AUC": roc_auc_score(y_test, probabilities)
    })

results_table = pd.DataFrame(evaluation_results)
print(results_table)

# ===============================
# Confusion Matrix Visualization
# ===============================
for model_name, model in trained_models.items():
    cmatrix = confusion_matrix(y_test, model.predict(X_test))

    plt.figure(figsize=(4, 4))
    plt.imshow(cmatrix)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cmatrix[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.show()

# ===============================
# ROC Curves
# ===============================
plt.figure(figsize=(6, 5))

for model_name, model in trained_models.items():
    prob_scores = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob_scores)
    auc_score = roc_auc_score(y_test, prob_scores)

    plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_score:.2f})")

plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.show()

# ===============================
# Best Model Selection
# ===============================
results_table["Final_Score"] = (
    results_table["Accuracy"] + results_table["F1-Score"]
)

best_model_summary = results_table.sort_values(
    by="Final_Score", ascending=False
).iloc[0]

best_model_summary

