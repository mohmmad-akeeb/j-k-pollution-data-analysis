"""
MODEL DESCRIPTION (Version 2: Forecasting-Oriented Model)

This model is a global time-aware regression model built using LightGBM,
designed for true forecasting of AQI using only historical information.

Key Characteristics:
- Input includes:
    • Lag features (AQI_lag_k, pollutant_lag_k)
    • Rolling statistics (past averages, variability)
    • Time features (month, year, cyclical encoding)
    • Hierarchical categorical features (Region, District, Station)

- Explicitly excludes:
    • Current-time pollutant values (PM10, PM2_5, NO2, SO2)

- Target:
    • AQI at time step (t)

Model Behavior:
- The model learns temporal dynamics:
      AQI(t) = f(AQI(t-1), AQI(t-12), PM2_5(t-1), ..., seasonality, location)
- Relies entirely on past observations and seasonal patterns
- Captures both short-term persistence and annual cycles

Implications:
- This is a valid forecasting model
- No leakage from future or same-time information
- Performance is lower than estimation model (expected trade-off)

Use Case:
- Predicting future AQI values (e.g., next month)
- Scenario where future pollutant values are unknown

Limitations:
- Reduced accuracy due to absence of strong contemporaneous predictors
- Sensitive to quality of lag and rolling feature engineering

Conclusion:
This model correctly frames AQI prediction as a time series forecasting problem,
ensuring causal consistency and real-world applicability.
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor

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
# Remove current-time pollutant values
leakage_cols = ['PM10', 'PM2_5', 'NO2', 'SO2']
df = df.drop(columns=[col for col in leakage_cols if col in df.columns])

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

# Convert categorical
categorical_cols = ['Region', 'District', 'Station']
for col in categorical_cols:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')

# =========================
# 6. MODEL
# =========================
model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=64,
    min_data_in_leaf=20,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# 7. PREDICTION
# =========================
y_pred = model.predict(X_test)

# =========================
# 8. METRICS
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
# 9. SAVE MODEL
# =========================
joblib.dump(model, os.path.join(output_dir, "lightgbm_model.pkl"))

# =========================
# 9. PLOTS (IMPROVED)
# =========================

# ---------- Plot 1: Actual vs Predicted ----------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)

# Perfect prediction line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val])

plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "actual_vs_predicted.png"), dpi=300)
plt.close()


# ---------- Plot 2: Residuals ----------
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


# ---------- Plot 3: Feature Importance ----------
importance = model.feature_importances_
feature_names = np.array(X_train.columns)

# Sort top 20 features
sorted_idx = np.argsort(importance)[-20:]
top_features = feature_names[sorted_idx]
top_importance = importance[sorted_idx]

plt.figure(figsize=(10, 8))
plt.barh(range(len(top_features)), top_importance)

plt.yticks(range(len(top_features)), top_features)
plt.xlabel("Importance")
plt.title("Feature Importance (Top 20)")

# Prevent label cutoff
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=300, bbox_inches='tight')
plt.close()

# =========================
# 11. OUTPUT
# =========================
print("Saved to:", output_dir)
print("Metrics:", metrics)