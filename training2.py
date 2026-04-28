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
# 10. PLOTS
# =========================

# Actual vs Predicted
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted")
plt.savefig(os.path.join(output_dir, "actual_vs_predicted.png"))
plt.close()

# Residual plot
residuals = y_test - y_pred
plt.figure()
plt.scatter(y_pred, residuals)
plt.axhline(y=0)
plt.xlabel("Predicted AQI")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.savefig(os.path.join(output_dir, "residuals.png"))
plt.close()

# Feature importance
importance = model.feature_importances_
feature_names = X_train.columns

plt.figure()
idx = np.argsort(importance)[-20:]
plt.barh(range(len(idx)), importance[idx])
plt.yticks(range(len(idx)), feature_names[idx])
plt.title("Feature Importance (Top 20)")
plt.savefig(os.path.join(output_dir, "feature_importance.png"))
plt.close()

# =========================
# 11. OUTPUT
# =========================
print("Saved to:", output_dir)
print("Metrics:", metrics)