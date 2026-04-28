import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("aqi_global_model_dataset.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# =========================
# 2. REMOVE LEAKAGE FEATURES
# =========================
leakage_cols = ['PM10', 'PM2_5', 'NO2', 'SO2']
df = df.drop(columns=[c for c in leakage_cols if c in df.columns])

# =========================
# 3. OUTPUT DIR
# =========================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"results_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# =========================
# 4. TRAIN / TEST SPLIT
# =========================
train_df = df[df['Date'] < "2026-01-01"].copy()
test_df  = df[df['Date'] >= "2026-01-01"].copy()

target = "AQI"
drop_cols = ['AQI', 'Date']

X_train = train_df.drop(columns=drop_cols)
y_train = train_df[target].values

X_test = test_df.drop(columns=drop_cols)
y_test = test_df[target].values

# =========================
# 5. CATEGORICAL HANDLING
# =========================
cat_cols = ['Region','District','Station']
cat_features = [c for c in cat_cols if c in X_train.columns]

# CatBoost uses raw categories (as strings)
for c in cat_features:
    X_train[c] = X_train[c].astype(str)
    X_test[c] = X_test[c].astype(str)

# For LightGBM (native categorical)
X_train_lgb = X_train.copy()
X_test_lgb = X_test.copy()
for c in cat_features:
    X_train_lgb[c] = X_train_lgb[c].astype('category')
    X_test_lgb[c] = X_test_lgb[c].astype('category')

# For XGBoost (label encoding)
X_train_xgb = X_train.copy()
X_test_xgb = X_test.copy()
for c in cat_features:
    cats = pd.concat([X_train_xgb[c], X_test_xgb[c]]).astype('category').cat.categories
    mapping = {k: i for i, k in enumerate(cats)}
    X_train_xgb[c] = X_train_xgb[c].map(mapping)
    X_test_xgb[c] = X_test_xgb[c].map(mapping)

# =========================
# 6. TIME SERIES CV (OOF)
# =========================
n_splits = 5
split_size = len(X_train) // (n_splits + 1)

oof_cat = np.zeros(len(X_train))
oof_lgb = np.zeros(len(X_train))
oof_xgb = np.zeros(len(X_train))

for i in range(n_splits):
    split_point = (i + 1) * split_size

    X_tr = X_train.iloc[:split_point]
    y_tr = y_train[:split_point]

    X_val = X_train.iloc[split_point:split_point + split_size]
    y_val = y_train[split_point:split_point + split_size]

    # CatBoost
    cat_model = CatBoostRegressor(
        iterations=300, learning_rate=0.05, depth=6,
        loss_function='RMSE', verbose=False
    )
    cat_model.fit(X_tr, y_tr, cat_features=cat_features)
    oof_cat[split_point:split_point + split_size] = cat_model.predict(X_val)

    # LightGBM
    lgb_model = LGBMRegressor(
        n_estimators=300, learning_rate=0.05,
        num_leaves=64, random_state=42
    )
    lgb_model.fit(
        X_train_lgb.iloc[:split_point], y_tr,
        categorical_feature=cat_features
    )
    oof_lgb[split_point:split_point + split_size] = lgb_model.predict(
        X_train_lgb.iloc[split_point:split_point + split_size]
    )

    # XGBoost
    xgb_model = XGBRegressor(
        n_estimators=300, learning_rate=0.05,
        max_depth=6, subsample=0.8, colsample_bytree=0.8,
        random_state=42, tree_method="hist"
    )
    xgb_model.fit(X_train_xgb.iloc[:split_point], y_tr)
    oof_xgb[split_point:split_point + split_size] = xgb_model.predict(
        X_train_xgb.iloc[split_point:split_point + split_size]
    )

# =========================
# 7. META MODEL
# =========================
stack_train = np.column_stack([oof_cat, oof_lgb, oof_xgb])

meta_model = Ridge(alpha=1.0)
meta_model.fit(stack_train, y_train)

# =========================
# 8. TRAIN BASE MODELS ON FULL DATA
# =========================
cat_final = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6, verbose=False)
cat_final.fit(X_train, y_train, cat_features=cat_features)

lgb_final = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=64)
lgb_final.fit(X_train_lgb, y_train, categorical_feature=cat_features)

xgb_final = XGBRegressor(
    n_estimators=300, learning_rate=0.05,
    max_depth=6, subsample=0.8, colsample_bytree=0.8,
    random_state=42, tree_method="hist"
)
xgb_final.fit(X_train_xgb, y_train)

# =========================
# 9. TEST PREDICTIONS
# =========================
pred_cat = cat_final.predict(X_test)
pred_lgb = lgb_final.predict(X_test_lgb)
pred_xgb = xgb_final.predict(X_test_xgb)

stack_test = np.column_stack([pred_cat, pred_lgb, pred_xgb])
y_pred = meta_model.predict(stack_test)

# =========================
# 10. METRICS
# =========================
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

metrics = {"RMSE": float(rmse), "MAE": float(mae), "R2": float(r2)}

with open(os.path.join(output_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

# =========================
# 11. SAVE MODELS
# =========================
joblib.dump(cat_final, os.path.join(output_dir, "catboost.pkl"))
joblib.dump(lgb_final, os.path.join(output_dir, "lightgbm.pkl"))
joblib.dump(xgb_final, os.path.join(output_dir, "xgboost.pkl"))
joblib.dump(meta_model, os.path.join(output_dir, "meta_model.pkl"))

# =========================
# 12. PLOTS
# =========================
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7)
m, M = min(y_test), max(y_test)
plt.plot([m, M], [m, M])
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Stacked Model: Actual vs Predicted")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "actual_vs_predicted.png"), dpi=300)
plt.close()

residuals = y_test - y_pred
plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0)
plt.xlabel("Predicted AQI")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "residuals.png"), dpi=300)
plt.close()

print("Saved to:", output_dir)
print("Metrics:", metrics)