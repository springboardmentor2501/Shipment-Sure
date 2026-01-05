import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score

# =============================
# 1. LOAD CLEAN, SCALED DATA
# =============================
X_train = pd.read_csv("X_train_clean.csv")
X_test  = pd.read_csv("X_test_clean.csv")
y_train = pd.read_csv("y_train_clean.csv").values.ravel()
y_test  = pd.read_csv("y_test_clean.csv").values.ravel()

print("Original feature count:", X_train.shape[1])

# =============================
# 2. TRAIN CATBOOST (FOR FEATURE IMPORTANCE)
# =============================
base_model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    verbose=0,
    random_state=42
)

base_model.fit(X_train, y_train)

# =============================
# 3. FEATURE IMPORTANCE
# =============================
importances = base_model.get_feature_importance()
features = X_train.columns

fi = pd.DataFrame({
    "feature": features,
    "importance": importances
}).sort_values("importance", ascending=False)

fi.to_csv("feature_importance.csv", index=False)

print("\n🔝 Top 20 Important Features:")
print(fi.head(20))

# =============================
# 4. SELECT TOP 20 FEATURES
# =============================
TOP_K = 20
top_features = fi.head(TOP_K)["feature"].tolist()

print(f"\n✅ Selected Top {TOP_K} Features:")
for f in top_features:
    print(" -", f)

# =============================
# 5. CREATE REDUCED DATASETS
# =============================
X_train_top20 = X_train[top_features]
X_test_top20  = X_test[top_features]

print("\nReduced feature count:", X_train_top20.shape[1])

# =============================
# 6. SAVE REDUCED DATASETS (CRITICAL)
# =============================
X_train_top20.to_csv("X_train_top20.csv", index=False)
X_test_top20.to_csv("X_test_top20.csv", index=False)

pd.Series(y_train).to_csv("y_train_top20.csv", index=False)
pd.Series(y_test).to_csv("y_test_top20.csv", index=False)

joblib.dump(top_features, "model_features.pkl")

print("\n💾 Saved files:")
print(" - X_train_top20.csv")
print(" - X_test_top20.csv")
print(" - y_train_top20.csv")
print(" - y_test_top20.csv")
print(" - model_features.pkl")

# =============================
# 7. SANITY CHECK (OPTIONAL BUT RECOMMENDED)
# =============================
check_model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    verbose=0,
    random_state=42
)

check_model.fit(X_train_top20, y_train)
preds = check_model.predict(X_test_top20)

f1 = f1_score(y_test, preds)

print("\n📊 TOP-20 FEATURE MODEL PERFORMANCE")
print(f"F1 Score: {f1:.4f}")
