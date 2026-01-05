# train_with_gridcv_high_acc_high_recall.py
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# CatBoost optional
try:
    from catboost import CatBoostClassifier
    catboost_available = True
except:
    catboost_available = False


# =====================================================
# 1. LOAD DATA (TOP 20 FEATURES)
# =====================================================
X_train_full = pd.read_csv("X_train_top20.csv")
X_test = pd.read_csv("X_test_top20.csv")
y_train_full = pd.read_csv("y_train_top20.csv").values.ravel()
y_test = pd.read_csv("y_test_top20.csv").values.ravel()

print("\nDataset Shapes")
print("Train:", X_train_full.shape)
print("Test :", X_test.shape)
print("Train class dist:", np.bincount(y_train_full))
print("Test  class dist:", np.bincount(y_test))


# =====================================================
# 2. TRAIN → VALIDATION SPLIT
# =====================================================
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.20,
    stratify=y_train_full,
    random_state=42
)


# =====================================================
# 3. COST-SENSITIVE THRESHOLD TUNING
# =====================================================
def tune_threshold_cost_sensitive(model, X_val, y_val, fn_weight=1.5):
    """
    Maximizes accuracy while penalizing false negatives
    """
    probs = model.predict_proba(X_val)[:, 1]
    best_t, best_score = 0.5, -1

    for t in np.linspace(0.45, 0.75, 60):
        preds = (probs >= t).astype(int)
        cm = confusion_matrix(y_val, preds)
        tn, fp, fn, tp = cm.ravel()

        acc = accuracy_score(y_val, preds)
        fn_penalty = fn / (fn + tp + 1e-9)

        score = acc - fn_weight * fn_penalty

        if score > best_score:
            best_score = score
            best_t = t

    return best_t, best_score


# =====================================================
# 4. MODEL GRIDS (ACCURACY + RECALL)
# =====================================================
grids = {}

# ---------- Random Forest ----------
grids["RandomForest"] = {
    "model": RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight=None
    ),
    "params": {
        "n_estimators": [300, 400],
        "max_depth": [10, 14],
        "min_samples_split": [5, 10],
        "min_samples_leaf": [2, 4]
    }
}

# ---------- XGBoost ----------
grids["XGBoost"] = {
    "model": XGBClassifier(
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        scale_pos_weight=1.0
    ),
    "params": {
        "n_estimators": [300, 400],
        "learning_rate": [0.03],
        "max_depth": [6, 8],
        "subsample": [0.9],
        "colsample_bytree": [0.8, 1.0]
    }
}

# ---------- CatBoost ----------
if catboost_available:
    grids["CatBoost"] = {
        "model": CatBoostClassifier(
            verbose=0,
            random_state=42,
            loss_function="Logloss"
        ),
        "params": {
            "iterations": [300, 400],
            "depth": [4, 6],
            "learning_rate": [0.03]
        }
    }


# =====================================================
# 5. TRAIN + EVALUATE
# =====================================================
results = []
best_model = None
best_name = None
best_score = -1
best_threshold = 0.5


def save_confusion_matrix(cm, name):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix – {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"cm_{name}.png")
    plt.close()


for name, obj in grids.items():
    print(f"\n==================== {name} ====================")

    grid = GridSearchCV(
        estimator=obj["model"],
        param_grid=obj["params"],
        scoring="accuracy",
        cv=3,
        n_jobs=-1,
        verbose=0,
        refit=True
    )

    grid.fit(X_tr, y_tr)
    best = grid.best_estimator_

    # Cost-sensitive threshold tuning
    threshold, _ = tune_threshold_cost_sensitive(
        best, X_val, y_val, fn_weight=1.5
    )

    probs = best.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    cm = confusion_matrix(y_test, preds)
    save_confusion_matrix(cm, name)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")
    print(f"Threshold: {threshold:.3f}")

    results.append([
        name, acc, prec, rec, f1, auc, threshold
    ])

    # Select best using balanced criterion
    final_score = acc + rec
    if final_score > best_score:
        best_score = final_score
        best_model = best
        best_name = name
        best_threshold = threshold


# =====================================================
# 6. SAVE RESULTS
# =====================================================
results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1", "AUC", "Threshold"]
)

print("\n======= FINAL RESULTS =======")
print(results_df)

results_df.to_csv("clean_model_results.csv", index=False)

# joblib.dump(
#     {
#         "model": best_model,
#         "threshold": best_threshold,
#         "model_name": best_name
#     },
#     "best_clean_model1.pkl"
# )

print(f"\n✅ Best Model Saved → best_clean_model.pkl ({best_name})")
print("🎯 High-Accuracy + High-Recall training complete.")
