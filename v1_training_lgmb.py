"""
MODEL DESCRIPTION (Version 1: Estimation-Oriented Model)

This model is a global regression model built using LightGBM. It is trained on
data from all stations simultaneously (global modeling approach) to increase
effective sample size and improve generalization.

Key Characteristics:
- Input includes:
    • Current pollutant values (PM10, PM2_5, NO2, SO2)
    • Lag features (past AQI and pollutants)
    • Rolling statistics
    • Time features (month, year, cyclical encoding)
    • Hierarchical categorical features (Region, District, Station)

- Target:
    • AQI at current time step (t)

Model Behavior:
- The model primarily learns a mapping:
      AQI(t) = f(PM10(t), PM2_5(t), NO2(t), SO2(t), ...)
- Current pollutant values dominate prediction (as seen in feature importance)
- Lag features contribute but are secondary

Implications:
- This is NOT a pure forecasting model
- It estimates AQI given pollutant measurements at the same time
- High accuracy (high R²) is expected because of strong direct relationships

Use Case:
- AQI estimation when pollutant data is already available
- Real-time AQI calculation from sensor inputs

Limitations:
- Cannot be used for future forecasting unless future pollutant values are known
- Contains temporal leakage for forecasting tasks

Conclusion:
This model solves a supervised regression problem, not a forward-looking
time series forecasting problem.
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

# Convert Date if present
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])

# =========================
# 2. CREATE OUTPUT DIRECTORY
# =========================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"results_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# =========================
# 3. TRAIN-TEST SPLIT (TIME-BASED)
# =========================
# Assuming Date exists
df = df.sort_values('Date')

train_df = df[df['Date'] < "2026-01-01"]
test_df = df[df['Date'] >= "2026-01-01"]

# =========================
# 4. FEATURES & TARGET
# =========================
target = "AQI"

drop_cols = ['AQI', 'Date'] if 'Date' in df.columns else ['AQI']
X_train = train_df.drop(columns=drop_cols)
y_train = train_df[target]

X_test = test_df.drop(columns=drop_cols)
y_test = test_df[target]

# Convert categorical columns
categorical_cols = ['Region', 'District', 'Station']
for col in categorical_cols:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')

# =========================
# 5. TRAIN MODEL
# =========================
model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# 6. PREDICTIONS
# =========================
y_pred = model.predict(X_test)

# =========================
# 7. METRICS
# =========================
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

metrics = {
    "RMSE": rmse,
    "MAE": mae,
    "R2": r2
}

# Save metrics
with open(os.path.join(output_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

# =========================
# 8. SAVE MODEL
# =========================
model_path = os.path.join(output_dir, "lightgbm_model.pkl")
joblib.dump(model, model_path)

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
# 10. OUTPUT SUMMARY
# =========================
print("Results saved in:", output_dir)
print("Metrics:", metrics)