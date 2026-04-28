import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# =========================
# 1. LOAD DATA
# =========================
data_path = "aqi_global_model_dataset.csv"
df = pd.read_csv(data_path)

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# =========================
# 2. REMOVE LEAKAGE FEATURES
# =========================
leakage_cols = ['PM10', 'PM2_5', 'NO2', 'SO2']
df = df.drop(columns=[c for c in leakage_cols if c in df.columns])

# =========================
# 3. CREATE OUTPUT DIRECTORY
# =========================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"results_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# =========================
# 4. TRAIN-TEST SPLIT
# =========================
train_df = df[df['Date'] < "2026-01-01"]
test_df = df[df['Date'] >= "2026-01-01"]

# =========================
# 5. FEATURES & TARGET
# =========================
target = "AQI"
drop_cols = ['AQI', 'Date']

X_train = train_df.drop(columns=drop_cols)
y_train = train_df[target]

X_test = test_df.drop(columns=drop_cols)
y_test = test_df[target]

# =========================
# 6. ENCODE CATEGORICALS (XGBoost requires numeric)
# =========================
categorical_cols = ['Region', 'District', 'Station']

for col in categorical_cols:
    if col in X_train.columns:
        # Fit encoding on full dataset to avoid mismatch
        df[col] = df[col].astype('category')
        mapping = dict(enumerate(df[col].cat.categories))
        reverse_mapping = {v: k for k, v in mapping.items()}
        
        X_train[col] = X_train[col].map(reverse_mapping)
        X_test[col] = X_test[col].map(reverse_mapping)

# =========================
# 7. MODEL
# =========================
model = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    tree_method="hist"
)

model.fit(X_train, y_train)

# =========================
# 8. PREDICTION
# =========================
y_pred = model.predict(X_test)

# =========================
# 9. METRICS
# =========================
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

metrics = {
    "RMSE": float(rmse),
    "MAE": float(mae),
    "R2": float(r2)
}

with open(os.path.join(output_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

# =========================
# 10. SAVE MODEL
# =========================
joblib.dump(model, os.path.join(output_dir, "xgboost_model.pkl"))

# =========================
# 11. PLOTS (CLEANED)
# =========================

# Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val])

plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "actual_vs_predicted.png"), dpi=300)
plt.close()

# Residuals
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0)

plt.xlabel("Predicted AQI")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "residuals.png"), dpi=300)
plt.close()

# Feature Importance
importance = model.feature_importances_
feature_names = np.array(X_train.columns)

idx = np.argsort(importance)[-20:]

plt.figure(figsize=(10, 8))
plt.barh(range(len(idx)), importance[idx])
plt.yticks(range(len(idx)), feature_names[idx])

plt.xlabel("Importance")
plt.title("Feature Importance (Top 20)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=300, bbox_inches='tight')
plt.close()

# =========================
# 12. OUTPUT
# =========================
print("Saved to:", output_dir)
print("Metrics:", metrics)