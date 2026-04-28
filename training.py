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
# 9. PLOTS
# =========================

# Plot 1: Actual vs Predicted
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted")
plt.savefig(os.path.join(output_dir, "actual_vs_predicted.png"))
plt.close()

# Plot 2: Residuals
residuals = y_test - y_pred
plt.figure()
plt.scatter(y_pred, residuals)
plt.xlabel("Predicted AQI")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.axhline(y=0)
plt.savefig(os.path.join(output_dir, "residuals.png"))
plt.close()

# Plot 3: Feature Importance
importance = model.feature_importances_
feature_names = X_train.columns

plt.figure()
sorted_idx = np.argsort(importance)[-20:]  # top 20
plt.barh(range(len(sorted_idx)), importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), feature_names[sorted_idx])
plt.title("Feature Importance (Top 20)")
plt.savefig(os.path.join(output_dir, "feature_importance.png"))
plt.close()

# =========================
# 10. OUTPUT SUMMARY
# =========================
print("Results saved in:", output_dir)
print("Metrics:", metrics)