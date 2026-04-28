import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("aqi_global_model_dataset.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Station', 'Date'])

# =========================
# 2. REMOVE LEAKAGE
# =========================
df = df.drop(columns=[c for c in ['PM10','PM2_5','NO2','SO2'] if c in df.columns])

# =========================
# 3. OUTPUT DIR
# =========================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"results_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# =========================
# 4. TRAIN-TEST SPLIT
# =========================
train_df = df[df['Date'] < "2026-01-01"].copy()
test_df  = df[df['Date'] >= "2026-01-01"].copy()

# =========================
# 5. ETS BASELINE (PER STATION)
# =========================
train_df['ets_pred'] = 0.0
test_df['ets_pred'] = 0.0

stations = df['Station'].unique()

for station in stations:
    train_station = train_df[train_df['Station'] == station]
    test_station  = test_df[test_df['Station'] == station]

    series = train_station['AQI'].values

    if len(series) < 24:
        # fallback: naive last value
        train_df.loc[train_station.index, 'ets_pred'] = series.mean()
        test_df.loc[test_station.index, 'ets_pred'] = series[-1]
        continue

    model = ExponentialSmoothing(
        series,
        trend='add',
        seasonal='add',
        seasonal_periods=12
    ).fit()

    # fitted values (train)
    train_df.loc[train_station.index, 'ets_pred'] = model.fittedvalues

    # forecast (test)
    forecast = model.forecast(len(test_station))
    test_df.loc[test_station.index, 'ets_pred'] = forecast

# =========================
# 6. RESIDUAL TARGET
# =========================
train_df['residual'] = train_df['AQI'] - train_df['ets_pred']

# =========================
# 7. FEATURES
# =========================
drop_cols = ['AQI', 'Date', 'residual']
X_train = train_df.drop(columns=drop_cols)
y_train = train_df['residual']

X_test = test_df.drop(columns=['AQI','Date'])
y_test = test_df['AQI']

# =========================
# 8. CATEGORICAL HANDLING
# =========================
cat_cols = ['Region','District','Station']
cat_features = [c for c in cat_cols if c in X_train.columns]

for c in cat_features:
    X_train[c] = X_train[c].astype(str)
    X_test[c] = X_test[c].astype(str)

# =========================
# 9. CATBOOST ON RESIDUALS
# =========================
model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    loss_function='RMSE',
    random_seed=42,
    verbose=False
)

model.fit(X_train, y_train, cat_features=cat_features)

# =========================
# 10. FINAL PREDICTION
# =========================
residual_pred = model.predict(X_test)

# final forecast
y_pred = test_df['ets_pred'].values + residual_pred

# =========================
# 11. METRICS
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
# 12. SAVE MODEL
# =========================
joblib.dump(model, os.path.join(output_dir, "catboost_residual_model.pkl"))

# =========================
# 13. PLOTS
# =========================
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7)
min_val, max_val = min(y_test), max(y_test)
plt.plot([min_val, max_val], [min_val, max_val])
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Hybrid: Actual vs Predicted")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "actual_vs_predicted.png"), dpi=300)
plt.close()

residuals = y_test - y_pred
plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0)
plt.xlabel("Predicted AQI")
plt.ylabel("Residuals")
plt.title("Hybrid Residual Plot")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "residuals.png"), dpi=300)
plt.close()

# =========================
# 14. OUTPUT
# =========================
print("Saved to:", output_dir)
print("Metrics:", metrics)