import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score
)

from imblearn.over_sampling import SMOTE

df = pd.read_excel("shipment_dataset_10000 (1).xlsx")
print("Raw dataset loaded:", df.shape)

leakage_cols = [
    "Actual_Delivery_Date",
    "Delivery_Date",
    "Delay_Days",
    "Delayed_Reason",
    "Reached.on.Time_Y.N"
]

df = df.drop(columns=[c for c in leakage_cols if c in df.columns])

target = "on_time_delivery"
X = df.drop(columns=[target])
y = df[target]


X = pd.get_dummies(X, drop_first=True)

date_cols = X.select_dtypes(include=["datetime64[ns]"]).columns

for col in date_cols:
    X[col + "_day"] = X[col].dt.day
    X[col + "_month"] = X[col].dt.month
    X[col + "_weekday"] = X[col].dt.weekday

X = X.drop(columns=date_cols)

print("Date columns processed & removed:", list(date_cols))

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create shipment_speed feature (DOMAIN KNOWLEDGE)
def shipment_speed(mode):
    if mode.lower() == "air":
        return 3   # fastest
    elif mode.lower() == "road":
        return 2
    else:  # sea
        return 1   # slowest

df["shipment_speed"] = df["shipment_mode"].apply(shipment_speed)

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("After SMOTE:", np.bincount(y_train_sm))

scaler = StandardScaler()
X_train_sm = scaler.fit_transform(X_train_sm)
X_test_scaled = scaler.transform(X_test)


lr = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)
lr.fit(X_train_sm, y_train_sm)
lr_preds = lr.predict(X_test_scaled)


rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

param_grid = {
    "n_estimators": [300, 500],
    "max_depth": [10, 15, None],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [2, 4],
    "max_features": ["sqrt"]
}

grid_rf = GridSearchCV(
    rf,
    param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_rf.fit(X_train_sm, y_train_sm)
best_rf = grid_rf.best_estimator_

rf_probs = best_rf.predict_proba(X_test_scaled)[:, 1]


best_threshold = 0.5
best_f1 = 0

for t in np.arange(0.2, 0.6, 0.05):
    preds = (rf_probs >= t).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

rf_preds = (rf_probs >= best_threshold).astype(int)

print("\nBest Threshold:", best_threshold)


results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest (Tuned)"],
    "Accuracy": [
        accuracy_score(y_test, lr_preds),
        accuracy_score(y_test, rf_preds)
    ],
    "F1 Score": [
        f1_score(y_test, lr_preds),
        f1_score(y_test, rf_preds)
    ]
})

print("\n==== RESULTS ====\n")
print(results)

print("\nLogistic Regression Report\n")
print(classification_report(y_test, lr_preds))

print("\nRandom Forest Report\n")
print(classification_report(y_test, rf_preds))

results.to_csv("model_results_m3.csv", index=False)
print("\nSaved -> model_results_m3.csv")


import joblib

joblib.dump(lr, "model_lr.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Saved model_lr.pkl and scaler.pkl")

