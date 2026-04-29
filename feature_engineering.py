"""
=============================================================================
  FEATURE ENGINEERING — Temporal, Lag, and Multivariate Features
=============================================================================
  Three feature pipelines:
    1. MICE imputation features (lags, leads, sin/cos month)
    2. Univariate tree features (lags, rolling means)
    3. Multivariate tree features (pollutant cross-lags)
=============================================================================
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. MICE Imputation Features
# ---------------------------------------------------------------------------
def engineer_imputation_features(df_tier):
    """
    Engineer temporal and sequential features for MICE imputation.

    Adds cyclical month encoding, linear time index, and per-station
    lag/lead features. Groups by Station to prevent cross-station leakage.

    Parameters
    ----------
    df_tier : pd.DataFrame
        A tier-specific DataFrame with Date, Station, and pollutant columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with added features: Month_Sin, Month_Cos, Time_Index,
        and per-pollutant Lag1/Lead1 columns.
    """
    print("Engineering Temporal and Sequential Features...")
    df = df_tier.copy()

    # 1. Enforce Datetime and Sorting (Critical to prevent cross-station leakage)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Station", "Date"]).reset_index(drop=True)

    # 2. Cyclical Seasonality (Sine/Cosine Transformation)
    # Teaches the model that December (12) and January (1) are adjacent
    df["Month"] = df["Date"].dt.month
    df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df = df.drop(columns=["Month"])

    # 3. Linear Time Index (Trend)
    # Counts months from 0..N to capture long-term pollution drift
    min_date = df["Date"].min()
    df["Time_Index"] = (df["Date"].dt.year - min_date.year) * 12 + (
        df["Date"].dt.month - min_date.month
    )

    # 4. Temporal Lags and Leads (grouped by Station)
    pollutants = ["PM10", "PM2_5", "NO2", "SO2", "AQI"]

    for col in pollutants:
        if col in df.columns:
            df[f"{col}_Lag1"] = df.groupby("Station")[col].shift(1)
            df[f"{col}_Lead1"] = df.groupby("Station")[col].shift(-1)

    print(f"Added Features: Month_Sin, Month_Cos, Time_Index, plus Lags/Leads.")
    return df


# ---------------------------------------------------------------------------
# 2. Univariate Tree Features
# ---------------------------------------------------------------------------
def engineer_tree_features(df, target_col="Calculated_AQI"):
    """
    Engineer lag and rolling features for univariate tree-based models
    (e.g., CatBoost single-station forecasting).

    Parameters
    ----------
    df : pd.DataFrame
        Single-station DataFrame with Date and target column.
    target_col : str
        Name of the target column (default: 'Calculated_AQI').

    Returns
    -------
    pd.DataFrame
        DataFrame with Month, Year, Lag_1, Lag_2, Lag_12, Rolling_Mean_3.
        First 12 rows dropped (warmup period).
    """
    print("Engineering features for Tree-Based Models...")
    df_tree = df.copy()

    df_tree["Date"] = pd.to_datetime(df_tree["Date"])
    df_tree = df_tree.sort_values("Date").reset_index(drop=True)

    # Temporal features
    df_tree["Month"] = df_tree["Date"].dt.month
    df_tree["Year"] = df_tree["Date"].dt.year

    # Lag features
    df_tree["Lag_1"] = df_tree[target_col].shift(1)
    df_tree["Lag_2"] = df_tree[target_col].shift(2)
    df_tree["Lag_12"] = df_tree[target_col].shift(12)

    # Rolling features (shifted by 1 to prevent leakage)
    df_tree["Rolling_Mean_3"] = (
        df_tree[target_col].shift(1).rolling(window=3).mean()
    )

    # Drop warmup period
    return df_tree.dropna().reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Multivariate Tree Features
# ---------------------------------------------------------------------------
def engineer_multivariate_features(df, target_col="Calculated_AQI"):
    """
    Engineer multivariate features including pollutant cross-lags
    for enhanced tree-based models.

    Adds all univariate features PLUS 12-month lags for each
    sub-pollutant (PM10, PM2_5, NO2, SO2) — the "chemical context"
    from the same season last year.

    Parameters
    ----------
    df : pd.DataFrame
        Single-station DataFrame with Date, target, and pollutant columns.
    target_col : str
        Name of the target column (default: 'Calculated_AQI').

    Returns
    -------
    pd.DataFrame
        DataFrame with extended feature set. First 12 rows dropped.
    """
    print("Engineering Multivariate Features (Deep Seasonal Echoes)...")
    df_tree = df.copy()
    df_tree["Date"] = pd.to_datetime(df_tree["Date"])
    df_tree = df_tree.sort_values("Date").reset_index(drop=True)

    # 1. Temporal Features
    df_tree["Month"] = df_tree["Date"].dt.month
    df_tree["Year"] = df_tree["Date"].dt.year

    # 2. Core Target Lags (AQI History)
    df_tree["Lag_1"] = df_tree[target_col].shift(1)
    df_tree["Lag_2"] = df_tree[target_col].shift(2)
    df_tree["Lag_12"] = df_tree[target_col].shift(12)
    df_tree["Rolling_Mean_3"] = (
        df_tree[target_col].shift(1).rolling(window=3).mean()
    )

    # 3. Multivariate Features (12-month pollutant echoes)
    # Only 12-month lags so we don't need to forecast them into the future
    pollutants = ["PM10", "PM2_5", "NO2", "SO2"]
    for p in pollutants:
        if p in df_tree.columns:
            df_tree[f"{p}_Lag_12"] = df_tree[p].shift(12)

    # Drop warmup period
    return df_tree.dropna().reset_index(drop=True)
