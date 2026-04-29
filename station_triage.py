"""
=============================================================================
  STATION TRIAGE — Tier Classification for Air Quality Stations
=============================================================================
  Classifies stations into:
    Tier 1 (Gold Standard)  — Low missingness + high AQI correlation
    Tier 2 (Salvageable)    — Moderate missingness / block gaps
    Tier 3 (Lost Cause)     — >80% core data missing
=============================================================================
"""

import pandas as pd
import numpy as np

from aqi_calculator import compute_subindex, _sub_index


# ---------------------------------------------------------------------------
# AQI Correlation Computation
# ---------------------------------------------------------------------------

# Breakpoints for the initial (pre-rounding) AQI reconstruction
_BREAKPOINTS_SIMPLE = {
    "PM2_5": [
        (0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200),
        (91, 120, 201, 300), (121, 250, 301, 400), (251, 500, 401, 500),
    ],
    "PM10": [
        (0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200),
        (251, 350, 201, 300), (351, 430, 301, 400), (431, 600, 401, 500),
    ],
    "NO2": [
        (0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200),
        (181, 280, 201, 300), (281, 400, 301, 400), (401, 1000, 401, 500),
    ],
    "SO2": [
        (0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200),
        (381, 800, 201, 300), (801, 1600, 301, 400), (1601, 2000, 401, 500),
    ],
}


def _compute_subindex_simple(value, pollutant):
    """Simple sub-index (no rounding) used for correlation check."""
    for (BLo, BHi, ILo, IHi) in _BREAKPOINTS_SIMPLE[pollutant]:
        if BLo <= value <= BHi:
            return _sub_index(value, BLo, BHi, ILo, IHi)
    return None


def _compute_aqi_simple(row):
    """Simple AQI (no rounding) for correlation check."""
    sub_indices = []
    for p in ["PM2_5", "PM10", "NO2", "SO2"]:
        if pd.notna(row[p]):
            si = _compute_subindex_simple(row[p], p)
            if si is not None:
                sub_indices.append(si)
    return max(sub_indices) if sub_indices else None


def compute_station_correlation(df):
    """
    Compute per-station correlation between reported AQI and
    reconstructed AQI (from sub-indices).

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with columns: Station, AQI, PM2_5, PM10, NO2, SO2.

    Returns
    -------
    pd.DataFrame
        Columns: ['Station', 'AQI_Correlation']
        Sorted descending by correlation.
    """
    cols = ["AQI", "PM10", "PM2_5", "NO2", "SO2"]
    aqi_df = df.dropna(subset=cols).copy()

    aqi_df["AQI_reconstructed"] = aqi_df.apply(_compute_aqi_simple, axis=1)

    station_eval = (
        aqi_df.groupby("Station")[["AQI", "AQI_reconstructed"]]
        .corr()
        .iloc[0::2, -1]
        .reset_index()
        .rename(columns={"AQI_reconstructed": "AQI_Correlation"})
    )

    station_eval = station_eval.sort_values(by="AQI_Correlation", ascending=False)
    station_eval = station_eval.reset_index(drop=True)
    station_eval = station_eval.drop("level_1", axis=1)

    print(f"Computed AQI correlations for {len(station_eval)} stations")
    return station_eval


# ---------------------------------------------------------------------------
# Triage Function
# ---------------------------------------------------------------------------
def triage_stations(df, correlation_df, missing_threshold=0.05):
    """
    Separates environmental monitoring stations into Tiers based on
    data integrity.

    Parameters
    ----------
    df : pd.DataFrame
        The main dataframe containing all station data.
    correlation_df : pd.DataFrame
        DataFrame with columns ['Station', 'AQI_Correlation'].
    missing_threshold : float
        Maximum acceptable core missingness (0.0 to 1.0) for Tier 1.

    Returns
    -------
    tuple of (triage_summary, df_tier1, df_tier2, df_tier3)
        triage_summary : pd.DataFrame with tier assignments and metrics.
        df_tier1 : DataFrame of Tier 1 (Gold Standard) stations.
        df_tier2 : DataFrame of Tier 2 (Salvageable) stations.
        df_tier3 : DataFrame of Tier 3 (Lost Cause) stations.
    """
    print("Initiating Station Triage Protocol...\n")
    df_copy = df.copy()

    # 1. Define Pollutant Groups
    core_pollutants = ["AQI", "PM10", "PM2_5"]
    secondary_pollutants = ["NO2", "SO2"]
    all_pollutants = core_pollutants + secondary_pollutants

    # 2. Calculate Missingness Metrics per Station
    missing_stats = (
        df_copy.groupby("Station")[all_pollutants]
        .apply(lambda x: x.isna().mean())
        .reset_index()
    )

    # Aggregated missingness based ONLY on core pollutants
    missing_stats["Core_Missing_Pct"] = missing_stats[core_pollutants].mean(axis=1)

    # Structural absence flags
    missing_stats["NO2_Absent"] = missing_stats["NO2"] == 1.0
    missing_stats["SO2_Absent"] = missing_stats["SO2"] == 1.0

    # 3. Merge with AQI Correlation Data
    triage_df = pd.merge(missing_stats, correlation_df, on="Station", how="left")
    triage_df["AQI_Correlation"] = triage_df["AQI_Correlation"].fillna(0)

    # 4. Apply Tier Logic
    triage_df["Tier"] = "Tier 2 (Salvageable)"

    tier_1_mask = (triage_df["Core_Missing_Pct"] <= missing_threshold) & (
        triage_df["AQI_Correlation"] >= 0.90
    )
    triage_df.loc[tier_1_mask, "Tier"] = "Tier 1 (Gold Standard)"

    tier_3_mask = triage_df["Core_Missing_Pct"] >= 0.80
    triage_df.loc[tier_3_mask, "Tier"] = "Tier 3 (Lost Cause)"

    # 5. Extract station lists
    tier_1_stations = triage_df[triage_df["Tier"] == "Tier 1 (Gold Standard)"][
        "Station"
    ].tolist()
    tier_2_stations = triage_df[triage_df["Tier"] == "Tier 2 (Salvageable)"][
        "Station"
    ].tolist()
    tier_3_stations = triage_df[triage_df["Tier"] == "Tier 3 (Lost Cause)"][
        "Station"
    ].tolist()

    # 6. Split the dataset
    df_tier_1 = df_copy[df_copy["Station"].isin(tier_1_stations)].copy()
    df_tier_2 = df_copy[df_copy["Station"].isin(tier_2_stations)].copy()
    df_tier_3 = df_copy[df_copy["Station"].isin(tier_3_stations)].copy()

    # 7. Diagnostic Summary
    print(f"--- Triage Complete ---")
    print(f"Total Stations Processed: {len(triage_df)}")
    print(f"Tier 1 (Gold Standard)  : {len(tier_1_stations)} stations")
    print(f"Tier 2 (Salvageable)    : {len(tier_2_stations)} stations")
    print(f"Tier 3 (Lost Cause)     : {len(tier_3_stations)} stations\n")

    print(f"Sensors structurally missing (100% NaN):")
    print(f" - Stations missing SO2 entirely: {triage_df['SO2_Absent'].sum()}")
    print(f" - Stations missing NO2 entirely: {triage_df['NO2_Absent'].sum()}")

    return triage_df, df_tier_1, df_tier_2, df_tier_3
