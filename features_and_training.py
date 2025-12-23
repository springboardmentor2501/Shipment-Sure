import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, auc, precision_recall_curve)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

INPUT_PATH = "shipment_dataset_with_product_id.csv"
OUTPUT_DIR = Path("./final_deployment_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"
PLOTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(INPUT_PATH)
TARGET = "on_time_delivery"

for col in ['order_date', 'promised_delivery_date', 'actual_delivery_date']:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Normalize carrier names to lowercase to avoid case sensitivity issues
if 'carrier_name' in df.columns:
    original_carriers = df['carrier_name'].nunique()
    df['carrier_name'] = df['carrier_name'].str.lower().str.strip()
    normalized_carriers = df['carrier_name'].nunique()
    print(f"  Carrier names: {original_carriers} -> {normalized_carriers} (after lowercase normalization)")

# Also normalize other text columns that might have case issues
for col in ['region', 'shipment_mode', 'weather_condition', 'holiday_period']:
    if col in df.columns:
        df[col] = df[col].str.lower().str.strip()

print(f"\nDataset: {df.shape}")
print(f"Target: {df[TARGET].value_counts().to_dict()}")

# -------------------------
# HISTORICAL AGGREGATIONS 
# -------------------------
print("\n" + "="*80)
print("CREATING HISTORICAL FEATURES")
print("="*80)

def add_historical_stats(df, group_col, target_col, prefix):
    global_mean = df[target_col].mean()
    grouped = df.groupby(group_col)[target_col].agg(['mean', 'std', 'count', 'min', 'max']).reset_index()
    grouped.columns = [group_col, f'{prefix}_mean', f'{prefix}_std', f'{prefix}_count', 
                       f'{prefix}_min', f'{prefix}_max']
    df = df.merge(grouped, on=group_col, how='left')
    
    # Fill missing
    df[f'{prefix}_mean'] = df[f'{prefix}_mean'].fillna(global_mean)
    df[f'{prefix}_std'] = df[f'{prefix}_std'].fillna(0)
    df[f'{prefix}_count'] = df[f'{prefix}_count'].fillna(0)
    df[f'{prefix}_min'] = df[f'{prefix}_min'].fillna(global_mean)
    df[f'{prefix}_max'] = df[f'{prefix}_max'].fillna(global_mean)
    
    # Bayesian smoothing
    confidence = df[f'{prefix}_count'] / (df[f'{prefix}_count'] + 10)
    df[f'{prefix}_smooth'] = confidence * df[f'{prefix}_mean'] + (1 - confidence) * global_mean
    
    # Range
    df[f'{prefix}_range'] = df[f'{prefix}_max'] - df[f'{prefix}_min']
    
    return df

print("Supplier stats...")
df = add_historical_stats(df, 'supplier_id', TARGET, 'sup')

print("Product stats...")
df = add_historical_stats(df, 'product_id', TARGET, 'prod')

print("Carrier stats...")
df = add_historical_stats(df, 'carrier_name', TARGET, 'car')

print("Region stats...")
df = add_historical_stats(df, 'region', TARGET, 'reg')

# Combined group stats
print("Supplier-Carrier combination...")
df['sup_car'] = df['supplier_id'].astype(str) + '_' + df['carrier_name']
df = add_historical_stats(df, 'sup_car', TARGET, 'supcar')

print("Product-Region combination...")
df['prod_reg'] = df['product_id'].astype(str) + '_' + df['region']
df = add_historical_stats(df, 'prod_reg', TARGET, 'prodreg')

print("Historical features created")


#-------------------------
# FEATURE ENGINEERING
#-------------------------
print("FEATURE ENGINEERING")
print("="*80)

# Date features
df['order_month'] = df['order_date'].dt.month
df['order_dow'] = df['order_date'].dt.dayofweek
df['order_day'] = df['order_date'].dt.day
df['order_quarter'] = df['order_date'].dt.quarter
df['is_weekend'] = df['order_dow'].isin([5, 6]).astype(int)
df['is_month_end'] = (df['order_day'] >= 25).astype(int)
df['is_month_start'] = (df['order_day'] <= 5).astype(int)

# Lead time
df['lead_time'] = (df['promised_delivery_date'] - df['order_date']).dt.days
df['promised_dow'] = df['promised_delivery_date'].dt.dayofweek
df['promised_weekend'] = df['promised_dow'].isin([5, 6]).astype(int)
df['lead_time_sq'] = df['lead_time'] ** 2
df['lead_time_log'] = np.log1p(df['lead_time'])

# Supplier
df['sup_score'] = df['supplier_rating'] * df['previous_on_time_rate'] / 100
df['sup_score_sq'] = df['sup_score'] ** 2
df['excellent_sup'] = ((df['supplier_rating'] >= 4.5) & (df['previous_on_time_rate'] >= 90)).astype(int)
df['poor_sup'] = ((df['supplier_rating'] < 3.0) | (df['previous_on_time_rate'] < 70)).astype(int)

# Order
df['value_per_unit'] = df['total_order_value'] / (df['order_quantity'] + 1)
df['log_value'] = np.log1p(df['total_order_value'])
df['log_qty'] = np.log1p(df['order_quantity'])
df['value_qty_ratio'] = df['total_order_value'] / (df['order_quantity'] + 1)

# Shipping
df['log_dist'] = np.log1p(df['shipping_distance_km'])
df['dist_sq'] = df['shipping_distance_km'] ** 2
df['dist_long'] = (df['shipping_distance_km'] > 700).astype(int)
df['dist_short'] = (df['shipping_distance_km'] < 300).astype(int)

# Mode
# df['is_air'] = (df['shipment_mode'] == 'Air').astype(int)
# df['is_road'] = (df['shipment_mode'] == 'Road').astype(int)
# df['is_sea'] = (df['shipment_mode'] == 'Sea').astype(int)
df['is_air'] = (df['shipment_mode'] == 'air').astype(int)
df['is_road'] = (df['shipment_mode'] == 'road').astype(int)
df['is_sea'] = (df['shipment_mode'] == 'sea').astype(int)

# Weather & Holiday
# df['bad_weather'] = df['weather_condition'].isin(['Storm', 'Rainy']).astype(int)
# df['is_holiday'] = (df['holiday_period'] == 'Yes').astype(int)
df['bad_weather'] = df['weather_condition'].isin(['storm', 'rainy']).astype(int)
df['is_holiday'] = (df['holiday_period'] == 'yes').astype(int)


# -------------------------
# ADVANCED INTERACTIONS
# -------------------------
print("Creating interaction features...")

# Historical × Current
df['sup_hist_x_rating'] = df['sup_smooth'] * df['supplier_rating']
df['sup_hist_x_ontime'] = df['sup_smooth'] * df['previous_on_time_rate']
df['prod_hist_x_value'] = df['prod_smooth'] * df['log_value']
df['car_hist_x_dist'] = df['car_smooth'] * df['log_dist']
df['reg_hist_x_holiday'] = df['reg_smooth'] * df['is_holiday']

# Combined reliability
df['combined_rel'] = (df['sup_smooth'] * 0.35 + 
                      df['prod_smooth'] * 0.20 +
                      df['car_smooth'] * 0.30 +
                      df['reg_smooth'] * 0.15)

df['combined_rel_sq'] = df['combined_rel'] ** 2
df['combined_rel_cb'] = df['combined_rel'] ** 3

# Risk scores
df['high_risk'] = ((df['sup_smooth'] < 0.5) | (df['car_smooth'] < 0.5) | 
                   (df['bad_weather'] == 1) | (df['is_holiday'] == 1)).astype(int)

df['low_risk'] = ((df['sup_smooth'] > 0.8) & (df['car_smooth'] > 0.8) & 
                  (df['bad_weather'] == 0) & (df['is_holiday'] == 0)).astype(int)

# Speed & timing
df['speed_needed'] = df['shipping_distance_km'] / (df['lead_time'] + 1)
df['tight_deadline'] = ((df['lead_time'] < 5) & (df['shipping_distance_km'] > 500)).astype(int)

# Weather risk
df['weather_risk'] = df['bad_weather'] * (1 - df['combined_rel'])

# Consistency scores (low std = more consistent)
df['sup_consistent'] = (df['sup_std'] < 0.1).astype(int)
df['car_consistent'] = (df['car_std'] < 0.1).astype(int)

# Count-based confidence
df['sup_high_volume'] = (df['sup_count'] > df['sup_count'].quantile(0.75)).astype(int)
df['prod_high_volume'] = (df['prod_count'] > df['prod_count'].quantile(0.75)).astype(int)

# Multi-way interactions
df['sup_car_prod'] = df['supcar_smooth'] * df['prod_smooth']
df['all_reliable'] = ((df['sup_smooth'] > 0.7) & (df['car_smooth'] > 0.7) & 
                      (df['prod_smooth'] > 0.7)).astype(int)

print(f"Total features: {df.shape[1]}")



#-------------------------
print("PREPARING DATA")
print("="*80)

drop_cols = ['order_id', 'actual_delivery_date', 'delayed_reason_code',
             'order_date', 'promised_delivery_date',
             'supplier_id', 'product_id', 'carrier_name', 'region',
             'shipment_mode', 'weather_condition', 'holiday_period',
             'sup_car', 'prod_reg']

drop_cols = [c for c in drop_cols if c in df.columns]

X = df.drop(columns=[TARGET] + drop_cols, errors='ignore')
y = df[TARGET].astype(int)

# Keep only numeric
X = X.select_dtypes(include=[np.number])

print(f"Features: {X.shape}")
print(f"Target: {y.shape}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# Scale
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# SMOTE
print("Applying SMOTE...")
sm = SMOTE(random_state=42, k_neighbors=5)
X_train_bal, y_train_bal = sm.fit_resample(X_train_sc, y_train)
print(f"After SMOTE: {X_train_bal.shape}")




#-------------------------
print("TRAINING MODELS")
print("="*80)

results = {}

#logistic regression
print("\n[1/4] logistic regression (Optimized)...")
lr = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)

lr.fit(X_train_bal, y_train_bal)
y_pred = lr.predict(X_test_sc)
y_proba = lr.predict_proba(X_test_sc)[:, 1]


results['logistic regression'] = {
    'model': lr,
    'accuracy': accuracy_score(y_test, y_pred),
    'roc_auc': roc_auc_score(y_test, y_proba),
    'report': classification_report(y_test, y_pred, output_dict=True),
    'cm': confusion_matrix(y_test, y_pred),
    'y_pred': y_pred,
    'y_proba': y_proba
}

print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}, ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
cr = results['logistic regression']['report']
print(f"  F1 Class 0: {cr['0']['f1-score']:.4f}, F1 Class 1: {cr['1']['f1-score']:.4f}")


# XGBoost - Highly optimized
print("\n[2/4] XGBoost (Optimized)...")
xgb = XGBClassifier(
    n_estimators=2000,
    max_depth=12,
    learning_rate=0.02,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=1,
    gamma=0.05,
    reg_alpha=0.05,
    reg_lambda=0.8,
    scale_pos_weight=2.5,
    eval_metric='logloss',
    random_state=42,
    tree_method='hist',
    n_jobs=-1
)
xgb.fit(X_train_bal, y_train_bal)
y_pred = xgb.predict(X_test_sc)
y_proba = xgb.predict_proba(X_test_sc)[:, 1]
acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

results['XGBoost'] = {
    'model': xgb,
    'accuracy': acc,
    'roc_auc': roc_auc,
    'report': classification_report(y_test, y_pred, output_dict=True),
    'cm': confusion_matrix(y_test, y_pred),
    'y_pred': y_pred,
    'y_proba': y_proba
}

print(f"  Accuracy: {acc:.4f}, ROC-AUC: {roc_auc:.4f}")
cr = results['XGBoost']['report']
print(f"  F1 Class 0: {cr['0']['f1-score']:.4f}, F1 Class 1: {cr['1']['f1-score']:.4f}")
if acc >= 0.90:
    print("TARGET REACHED: >= 90%")


# CatBoost - Highly optimized
print("\n[3/4] CatBoost (Optimized)...")
cat = CatBoostClassifier(
    iterations=2000,
    depth=12,
    learning_rate=0.02,
    l2_leaf_reg=2,
    border_count=128,
    random_strength=0.2,
    bagging_temperature=0.1,
    auto_class_weights='Balanced',
    random_state=42,
    verbose=0
)
cat.fit(X_train_bal, y_train_bal)
y_pred = cat.predict(X_test_sc)
y_proba = cat.predict_proba(X_test_sc)[:, 1]
acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

# Save the CatBoost model and preprocessor
cat.save_model(OUTPUT_DIR / "catboost_model.cbm")
joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")

# Save feature information for the Streamlit app
feature_info = {
    'feature_names': X.columns.tolist(),
    'drop_columns': drop_cols,
    'target_column': TARGET
}
joblib.dump(feature_info, OUTPUT_DIR / "feature_info.pkl")

print(f" Model saved: {OUTPUT_DIR / 'catboost_model.cbm'}")
print(f" Scaler saved: {OUTPUT_DIR / 'scaler.pkl'}")
print(f" Feature info saved: {OUTPUT_DIR / 'feature_info.pkl'}")

results['CatBoost'] = {
    'model': cat,
    'accuracy': acc,
    'roc_auc': roc_auc,
    'report': classification_report(y_test, y_pred, output_dict=True),
    'cm': confusion_matrix(y_test, y_pred),
    'y_pred': y_pred,
    'y_proba': y_proba
}

print(f"  Accuracy: {acc:.4f}, ROC-AUC: {roc_auc:.4f}")
cr = results['CatBoost']['report']
print(f"  F1 Class 0: {cr['0']['f1-score']:.4f}, F1 Class 1: {cr['1']['f1-score']:.4f}")
if acc >= 0.90:
    print("TARGET REACHED: >= 90% ")


# LightGBM - Highly optimized
print("\n[4/4] LightGBM (Optimized)...")
lgbm = LGBMClassifier(
    n_estimators=2000,
    max_depth=12,
    learning_rate=0.02,
    num_leaves=80,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_samples=8,
    reg_alpha=0.05,
    reg_lambda=0.8,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgbm.fit(X_train_bal, y_train_bal)
y_pred = lgbm.predict(X_test_sc)
y_proba = lgbm.predict_proba(X_test_sc)[:, 1]
acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

results['LightGBM'] = {
    'model': lgbm,
    'accuracy': acc,
    'roc_auc': roc_auc,
    'report': classification_report(y_test, y_pred, output_dict=True),
    'cm': confusion_matrix(y_test, y_pred),
    'y_pred': y_pred,
    'y_proba': y_proba
}

print(f"  Accuracy: {acc:.4f}, ROC-AUC: {roc_auc:.4f}")
cr = results['LightGBM']['report']
print(f"  F1 Class 0: {cr['0']['f1-score']:.4f}, F1 Class 1: {cr['1']['f1-score']:.4f}")
if acc >= 0.90:
    print(" TARGET REACHED: >= 90%")


#-------------------------

print("FINAL RESULTS")
print("="*80)

for name, res in results.items():
    cr = res['report']
    print(f"\n{name}:")
    print(f"  Accuracy: {res['accuracy']:.4f}")
    print(f"  Class 0: P={cr['0']['precision']:.4f}, R={cr['0']['recall']:.4f}, F1={cr['0']['f1-score']:.4f}")
    print(f"  Class 1: P={cr['1']['precision']:.4f}, R={cr['1']['recall']:.4f}, F1={cr['1']['f1-score']:.4f}")
    print(f"  Confusion Matrix: {res['cm'].tolist()}")

best_acc = max(res['accuracy'] for res in results.values())
best_model = [n for n, r in results.items() if r['accuracy'] == best_acc][0]

print(f"\n{'='*80}")
if best_acc >= 0.90:
    print(f"SUCCESS: {best_model} achieved {best_acc:.4f} (>= 90%)")
else:
    print(f"Best: {best_model} with {best_acc:.4f}")
print(f"{'='*80}")


print("MODEL PERFORMANCE COMPARISON TABLE")
print("="*80)

comparison_data = []
for name, res in results.items():
    cr = res['report']
    comparison_data.append({
        'Model': name,
        'Accuracy': res['accuracy'],
        'ROC-AUC': res['roc_auc'],
        'Precision_Class_0': cr['0']['precision'],
        'Recall_Class_0': cr['0']['recall'],
        'F1_Score_Class_0': cr['0']['f1-score'],
        'Precision_Class_1': cr['1']['precision'],
        'Recall_Class_1': cr['1']['recall'],
        'F1_Score_Class_1': cr['1']['f1-score'],
        'Macro_Avg_F1': cr['macro avg']['f1-score'],
        'Weighted_Avg_F1': cr['weighted avg']['f1-score']
    })

comparison_df = pd.DataFrame(comparison_data).sort_values('Accuracy', ascending=False)
print("\n" + comparison_df.to_string(index=False))

# Save comparison table
comparison_df.to_csv(REPORTS_DIR / "model_performance_comparison.csv", index=False)
print(f"\nSaved: {REPORTS_DIR / 'model_performance_comparison.csv'}")

print("CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Accuracy comparison
ax = axes[0, 0]
sns.barplot(data=comparison_df, x='Model', y='Accuracy', ax=ax, palette='viridis')
ax.axhline(y=0.90, color='red', linestyle='--', linewidth=2, label='90% Target')
ax.axhline(y=0.95, color='green', linestyle='--', linewidth=2, label='95% Target')
ax.set_ylim(0.85, 1.0)
ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy')
ax.legend()
for i, v in enumerate(comparison_df['Accuracy']):
    ax.text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')


# ROC-AUC comparison
ax = axes[0, 1]
sns.barplot(data=comparison_df, x='Model', y='ROC-AUC', ax=ax, palette='plasma')
ax.axhline(y=0.95, color='green', linestyle='--', linewidth=2, label='95% Target')
ax.set_ylim(0.85, 1.0)
ax.set_title('Model ROC-AUC Comparison', fontsize=14, fontweight='bold')
ax.set_ylabel('ROC-AUC Score')
ax.legend()
for i, v in enumerate(comparison_df['ROC-AUC']):
    ax.text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')


#F1 score
ax = axes[1, 0]
f1_data = comparison_df[['Model', 'F1_Score_Class_0', 'F1_Score_Class_1']].melt(
    id_vars='Model', var_name='Class', value_name='F1_Score'
)
f1_data['Class'] = f1_data['Class'].map({
    'F1_Score_Class_0': 'Not On Time (0)',
    'F1_Score_Class_1': 'On Time (1)'
})
sns.barplot(data=f1_data, x='Model', y='F1_Score', hue='Class', ax=ax, palette='Set2')
ax.set_ylim(0.85, 1.0)
ax.set_title('F1-Score by Class', fontsize=14, fontweight='bold')
ax.set_ylabel('F1-Score')
ax.legend(title='Class')

# Precision-Recall comparison
ax = axes[1, 1]
metrics_data = []
for _, row in comparison_df.iterrows():
    metrics_data.append({'Model': row['Model'], 'Metric': 'Precision (0)', 'Value': row['Precision_Class_0']})
    metrics_data.append({'Model': row['Model'], 'Metric': 'Recall (0)', 'Value': row['Recall_Class_0']})
    metrics_data.append({'Model': row['Model'], 'Metric': 'Precision (1)', 'Value': row['Precision_Class_1']})
    metrics_data.append({'Model': row['Model'], 'Metric': 'Recall (1)', 'Value': row['Recall_Class_1']})
metrics_df = pd.DataFrame(metrics_data)
sns.barplot(data=metrics_df, x='Model', y='Value', hue='Metric', ax=ax, palette='tab10')
ax.set_ylim(0.85, 1.0)
ax.set_title('Precision & Recall by Class', fontsize=14, fontweight='bold')
ax.set_ylabel('Score')
ax.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "model_performance_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {PLOTS_DIR / 'model_performance_comparison.png'}")


# Roc-curve
plt.figure(figsize=(10, 8))

for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.5000)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "roc_curves_all_models.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {PLOTS_DIR / 'roc_curves_all_models.png'}")

#  CONFUSION MATRICES FOR EACH MODEL
fig, axes = plt.subplots(1, 4, figsize=(24, 5))

for idx, (name, res) in enumerate(results.items()):
    cm = res['cm']
    ax = axes[idx]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Not On Time (0)', 'On Time (1)'],
                yticklabels=['Not On Time (0)', 'On Time (1)'],
                cbar_kws={'label': 'Count'})
    
    acc = res['accuracy']
    roc_auc = res['roc_auc']
    cr = res['report']
    
    ax.set_title(f'{name}\nAccuracy: {acc:.4f} | ROC-AUC: {roc_auc:.4f}\n'
                 f'F1(0): {cr["0"]["f1-score"]:.4f} | F1(1): {cr["1"]["f1-score"]:.4f}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "confusion_matrices_all_models.png", dpi=300, bbox_inches='tight')
plt.close()
print(f" Saved: {PLOTS_DIR / 'confusion_matrices_all_models.png'}")


#FEATURE IMPORTANCE FOR EACH MODEL
for name, res in results.items():
    if hasattr(res['model'], 'feature_importances_'):
        importances = res['model'].feature_importances_
        feature_imp_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(20)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=feature_imp_df, x='Importance', y='Feature', palette='viridis')
        plt.title(f'Top 20 Feature Importances - {name}', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
        plt.ylabel('Feature', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"feature_importance_{name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {PLOTS_DIR / f'feature_importance_{name}.png'}")
        
        # Save feature importance to CSV
        feature_imp_df.to_csv(REPORTS_DIR / f"feature_importance_{name}.csv", index=False)


# COMBINED FEATURE IMPORTANCE COMPARISON
fig, axes = plt.subplots(1, 4, figsize=(24, 8))

for idx, (name, res) in enumerate(results.items()):
    if hasattr(res['model'], 'feature_importances_'):
        importances = res['model'].feature_importances_
        feature_imp_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(15)
        
        ax = axes[idx]
        sns.barplot(data=feature_imp_df, x='Importance', y='Feature', ax=ax, palette='rocket')
        ax.set_title(f'{name} - Top 15 Features', fontsize=12, fontweight='bold')
        ax.set_xlabel('Importance', fontsize=10, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "feature_importance_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {PLOTS_DIR / 'feature_importance_comparison.png'}")


# SAVE DETAILED REPORTS
for name, res in results.items():
    # Classification report
    cr_df = pd.DataFrame(res['report']).transpose()
    cr_df.to_csv(REPORTS_DIR / f"{name}_classification_report.csv")
    
    # Confusion matrix
    cm_df = pd.DataFrame(res['cm'],
                         columns=['Predicted_Not_On_Time', 'Predicted_On_Time'],
                         index=['Actual_Not_On_Time', 'Actual_On_Time'])
    cm_df.to_csv(REPORTS_DIR / f"{name}_confusion_matrix.csv")

print(f"\n{'='*80}")
print(f"ALL OUTPUTS SAVED TO: {OUTPUT_DIR.resolve()}")
print(f"  - Reports: {REPORTS_DIR.resolve()}")
print(f"  - Plots: {PLOTS_DIR.resolve()}")
print(f"{'='*80}")
