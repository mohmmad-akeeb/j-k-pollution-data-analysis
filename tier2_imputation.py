"""
=============================================================================
  TIER 2 IMPUTATION — Temporal and Spatial Salvage
=============================================================================
  1. temporal_imputation: Station-wise temporal imputation with strict hierarchy
  2. spatial_salvage_tier2: Hierarchical spatial salvage using Tier 1 anchors
=============================================================================
"""

import pandas as pd
import numpy as np


def temporal_imputation(df, pollutant_cols=None):
    """
    Station-wise temporal imputation with strict hierarchy:

    1) Linear interpolation for short gaps (<=2)
    2) Local rolling regression for medium gaps (3-5)
    3) Conditional ffill/bfill (<=2, low-variance only)

    No cross-station leakage.
    """

    if pollutant_cols is None:
        pollutant_cols = ['PM10', 'PM2_5', 'NO2', 'SO2']

    df = df.copy()

    df = df.sort_values(['Station', 'Date']).reset_index(drop=True)

    def _linear_short_gaps(g, col):
        return g[col].interpolate(method='linear', limit=2, limit_direction='both')

    def _identify_nan_runs(mask):
        # returns list of (start_idx, end_idx, length)
        runs = []
        start = None
        for i, val in enumerate(mask):
            if val and start is None:
                start = i
            elif not val and start is not None:
                runs.append((start, i - 1, i - start))
                start = None
        if start is not None:
            runs.append((start, len(mask) - 1, len(mask) - start))
        return runs

    def _rolling_regression_fill(g, col):
        values = g[col].values.copy()
        time_index = np.arange(len(g))

        nan_mask = np.isnan(values)
        runs = _identify_nan_runs(nan_mask)

        for start, end, length in runs:
            if 3 <= length <= 5:
                # window +-3
                left = max(0, start - 3)
                right = min(len(g), end + 4)

                idx = time_index[left:right]
                y = values[left:right]

                valid = ~np.isnan(y)
                if valid.sum() < 3:
                    continue

                X = idx[valid]
                Y = y[valid]

                # simple linear regression
                beta1, beta0 = np.polyfit(X, Y, 1)

                # fill only missing segment
                for t in range(start, end + 1):
                    values[t] = beta0 + beta1 * time_index[t]

        return pd.Series(values, index=g.index)

        
    def _conditional_propagation(g, col):
        x = g[col].copy()
        values = x.values.astype(float)
    
        station_std = np.nanstd(values)
        nan_mask = np.isnan(values)
        runs = _identify_nan_runs(nan_mask)
    
        for start, end, length in runs:
            if length > 2:
                continue
    
            left_idx = start - 1
            right_idx = end + 1
    
            left_val = values[left_idx] if left_idx >= 0 else np.nan
            right_val = values[right_idx] if right_idx < len(values) else np.nan
    
            if np.isnan(left_val) or np.isnan(right_val):
                continue
    
            local_std = np.std([left_val, right_val])
    
            if local_std >= station_std:
                continue
    
            # safe interpolation (not blind propagation)
            step = (right_val - left_val) / (length + 1)
            for i in range(length):
                values[start + i] = left_val + step * (i + 1)
    
        return pd.Series(values, index=g.index)

    def process_station(g):
        g = g.sort_values('Date').copy()

        for col in pollutant_cols:
            if col not in g.columns:
                continue

            # Step 1: linear interpolation (short gaps)
            g[col] = _linear_short_gaps(g, col)

            # Step 2: rolling regression (medium gaps)
            g[col] = _rolling_regression_fill(g, col)

            # Step 3: constrained propagation
            g[col] = _conditional_propagation(g, col)

        return g

    result = []
    for station, g in df.groupby('Station'):
        result.append(process_station(g))

    print(f"Temporal Imputation Complete for {len(df['Station'].unique())} stations.")
    return pd.concat(result, ignore_index=True)


def spatial_salvage_tier2(df_tier_2, df_tier_1, pollutant_cols=['PM10', 'PM2_5', 'NO2', 'SO2']):
    print("Initiating Hierarchical Spatial Salvage for Tier 2...\n")
    df_t2 = df_tier_2.copy()
    
    # Create a combined dataframe for our broad District fallback
    df_all = pd.concat([df_tier_1, df_tier_2], ignore_index=True)
    
    # --- BUILD THE ANCHOR LOOKUP TABLES ---
    # 1. Primary Anchor: Medians of Gold Standard (Tier 1) stations in the same District
    t1_dist = df_tier_1.groupby(['District', 'Date'])[pollutant_cols].median().reset_index()
    
    # 2. Secondary Anchor: Medians of ALL available stations in the same District
    all_dist = df_all.groupby(['District', 'Date'])[pollutant_cols].median().reset_index()
    
    # 3. Final Anchor: Medians of Gold Standard stations in the entire Region (Jammu/Kashmir)
    t1_reg = df_tier_1.groupby(['Region', 'Date'])[pollutant_cols].median().reset_index()
    
    # --- APPLY HIERARCHICAL FILLING ---
    for col in pollutant_cols:
        if col not in df_t2.columns:
            continue
            
        missing_before = df_t2[col].isna().sum()
        
        # 1. Try to fill from Tier 1 District Medians
        df_t2 = df_t2.merge(t1_dist[['District', 'Date', col]], on=['District', 'Date'], how='left', suffixes=('', '_anchor1'))
        df_t2[col] = df_t2[col].fillna(df_t2[f'{col}_anchor1'])
        
        # 2. Try to fill from All Stations District Medians
        df_t2 = df_t2.merge(all_dist[['District', 'Date', col]], on=['District', 'Date'], how='left', suffixes=('', '_anchor2'))
        df_t2[col] = df_t2[col].fillna(df_t2[f'{col}_anchor2'])
        
        # 3. Try to fill from Tier 1 Regional Medians
        df_t2 = df_t2.merge(t1_reg[['Region', 'Date', col]], on=['Region', 'Date'], how='left', suffixes=('', '_anchor3'))
        df_t2[col] = df_t2[col].fillna(df_t2[f'{col}_anchor3'])
        
        # Cleanup temporary columns
        cols_to_drop = [c for c in df_t2.columns if '_anchor' in c]
        df_t2 = df_t2.drop(columns=cols_to_drop)
        
        missing_after = df_t2[col].isna().sum()
        print(f"Pollutant {col}: Successfully rescued {missing_before - missing_after} missing values.")
        
    print("\nSpatial Salvage Complete. Preserved original observed data.")
    return df_t2
