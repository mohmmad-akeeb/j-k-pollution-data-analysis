import pandas as pd
import numpy as np

# =========================
# 1. LOAD DATA
# =========================
input_path = "imputed_complete.csv"      # input CSV
output_path = "aqi_global_model_dataset.csv"  # output CSV

df = pd.read_csv(input_path)

# =========================
# 2. BASIC PREPROCESSING
# =========================
# Convert Date column
df['Date'] = pd.to_datetime(df['Date'])

# Sort properly (critical for lag creation)
df = df.sort_values(['Station', 'Date']).reset_index(drop=True)

# =========================
# 3. TIME FEATURES
# =========================
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year

# Cyclical encoding for seasonality
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# =========================
# 4. LAG FEATURES
# =========================
lags = [1, 2, 3, 6, 12]

columns_to_lag = ['AQI', 'PM10', 'PM2_5', 'NO2', 'SO2']

for col in columns_to_lag:
    for lag in lags:
        df[f'{col}_lag_{lag}'] = df.groupby('Station')[col].shift(lag)

# =========================
# 5. ROLLING FEATURES
# =========================
# Use shift(1) to avoid leakage (only past data)
df['AQI_roll_mean_3'] = df.groupby('Station')['AQI'].shift(1).rolling(3).mean()
df['AQI_roll_mean_6'] = df.groupby('Station')['AQI'].shift(1).rolling(6).mean()
df['AQI_roll_std_3'] = df.groupby('Station')['AQI'].shift(1).rolling(3).std()

df['PM2_5_roll_mean_3'] = df.groupby('Station')['PM2_5'].shift(1).rolling(3).mean()

# =========================
# 6. DROP ORIGINAL UNUSED COLUMNS (optional)
# =========================
# Keep Date if needed, otherwise drop
# df = df.drop(columns=['Date'])

# =========================
# 7. REMOVE NaNs (created by lagging/rolling)
# =========================
df = df.dropna().reset_index(drop=True)

# =========================
# 8. FINAL DATASET CHECK
# =========================
# Target remains AQI (current row)
# Features are all lag/rolling/time/categorical columns

print("Final dataset shape:", df.shape)

# =========================
# 9. SAVE OUTPUT
# =========================
df.to_csv(output_path, index=False)

print(f"Processed dataset saved to: {output_path}")