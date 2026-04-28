import os
import json
import pandas as pd
import numpy as np

from datetime import datetime
from itertools import product
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
df = df.drop(columns=[c for c in ['PM10','PM2_5','NO2','SO2'] if c in df.columns])

# =========================
# 3. TRAIN / TEST SPLIT
# =========================
train_df = df[df['Date'] < "2026-01-01"].copy()
test_df  = df[df['Date'] >= "2026-01-01"].copy()

target = "AQI"
drop_cols = ['AQI', 'Date']

X_train_full = train_df.drop(columns=drop_cols)
y_train_full = train_df[target].values

X_test = test_df.drop(columns=drop_cols)
y_test = test_df[target].values

# =========================
# 4. CATEGORICAL ENCODING
# =========================
cat_cols = ['Region', 'District', 'Station']

for col in cat_cols:
    if col in X_train_full.columns:
        combined = pd.concat([X_train_full[col], X_test[col]]).astype('category')
        mapping = {k: i for i, k in enumerate(combined.cat.categories)}
        X_train_full[col] = X_train_full[col].map(mapping)
        X_test[col] = X_test[col].map(mapping)

# =========================
# 5. VALIDATION SPLIT (TIME-AWARE)
# =========================
split_idx = int(len(X_train_full) * 0.8)

X_train = X_train_full.iloc[:split_idx].values
y_train = y_train_full[:split_idx]

X_val = X_train_full.iloc[split_idx:].values
y_val = y_train_full[split_idx:]

X_test = X_test.values

# =========================
# 6. OUTPUT SETUP
# =========================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"xgb_cpu_tuning_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

results_file = os.path.join(output_dir, "results.jsonl")
best_file = os.path.join(output_dir, "best_result.json")

# =========================
# 7. PARAM GRID
# =========================
param_grid = {
    "learning_rate": [0.01, 0.02],
    "n_estimators": [800, 1000],
    "max_depth": [4, 6],
    "subsample": [0.7, 0.8],
    "colsample_bytree": [0.7, 0.8],
    "reg_alpha": [0, 0.1],
    "reg_lambda": [1, 3]
}

keys = list(param_grid.keys())
values = list(param_grid.values())

best_rmse = float("inf")
best_result = None

# =========================
# 8. GRID SEARCH LOOP (CPU)
# =========================
with open(results_file, "a") as f:
    for i, combo in enumerate(product(*values)):
        params = dict(zip(keys, combo))

        model = XGBRegressor(
            **params,
            tree_method="hist",   # CPU optimized
            random_state=42,
            n_jobs=-1             # full CPU usage
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        result = {
            "params": params,
            "RMSE": float(rmse),
            "MAE": float(mae),
            "R2": float(r2)
        }

        # Append result
        f.write(json.dumps(result) + "\n")
        f.flush()

        # Track best
        if rmse < best_rmse:
            best_rmse = rmse
            best_result = result

        print(f"[{i}] RMSE={rmse:.4f}, Best={best_rmse:.4f}")

# =========================
# 9. SAVE BEST RESULT
# =========================
with open(best_file, "w") as f:
    json.dump(best_result, f, indent=4)

print("\nBest configuration:")
print(best_result)