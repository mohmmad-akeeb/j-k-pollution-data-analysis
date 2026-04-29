"""
=============================================================================
  MICE IMPUTATION — Multivariate Imputation by Chained Equations
=============================================================================
  Per-station iterative imputation using ExtraTreesRegressor.
  Handles stations with 100% missing sensors (dead-sensor bypass).
=============================================================================
"""

import warnings
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor

POLLUTANTS = ["PM10", "PM2_5", "NO2", "SO2", "AQI"]
EXCLUDE_COLS = ["Date", "Region", "District", "Station"]


def execute_mice_imputation(df_engineered):
    """
    Execute per-station MICE imputation with dead-sensor bypass.

    For each station:
    - Identifies columns that are 100% NaN (structurally absent sensors)
    - Runs IterativeImputer only on columns with partial data
    - Leaves dead-sensor columns as NaN
    - Clips negative values at 0

    Parameters
    ----------
    df_engineered : pd.DataFrame
        DataFrame with engineered features (from engineer_imputation_features).

    Returns
    -------
    pd.DataFrame
        Imputed DataFrame with helper columns removed, negatives floored at 0.
    """
    print("Initializing Robust MICE Pipeline...")
    df = df_engineered.copy()
    warnings.filterwarnings("ignore", category=UserWarning)

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

    estimator = ExtraTreesRegressor(
        n_estimators=50, max_depth=5, min_samples_leaf=2, random_state=42
    )

    imputer = IterativeImputer(
        estimator=estimator, max_iter=10, tol=1e-3,
        imputation_order="roman", random_state=42,
    )

    result_frames = []
    grouped = df.groupby("Station")
    total_stations = len(grouped)

    for i, (station, g) in enumerate(grouped, 1):
        print(f"Imputing Station {i}/{total_stations}: {station}...", end="\r")

        missing_ratios = g[feature_cols].isna().mean()
        dead_cols = missing_ratios[missing_ratios == 1.0].index.tolist()
        valid_cols = [c for c in feature_cols if c not in dead_cols]

        g_imputed = g.copy()

        if len(valid_cols) > 1:
            X_imputed = imputer.fit_transform(g[valid_cols].values)
            g_imputed[valid_cols] = X_imputed
        elif len(valid_cols) == 1:
            g_imputed[valid_cols[0]] = g_imputed[valid_cols[0]].fillna(
                g_imputed[valid_cols[0]].median()
            )

        result_frames.append(g_imputed)

    print("\nImputation Complete! Dead sensors successfully bypassed.")

    df_final = pd.concat(result_frames, ignore_index=True)

    # Cleanup engineered helper columns
    cols_to_drop = [
        c for c in df_final.columns
        if "Lag" in c or "Lead" in c or c in ["Month_Sin", "Month_Cos", "Time_Index"]
    ]
    df_final = df_final.drop(columns=cols_to_drop, errors="ignore")

    # Floor negative values at 0
    for p in POLLUTANTS:
        if p in df_final.columns:
            df_final[p] = df_final[p].clip(lower=0)

    return df_final
