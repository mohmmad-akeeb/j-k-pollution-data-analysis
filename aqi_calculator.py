"""
=============================================================================
  AQI CALCULATOR — CPCB Sub-Index & AQI Reconstruction
=============================================================================
  Implements the Central Pollution Control Board (CPCB) AQI formula:
    AQI = max(sub-index for each pollutant)

  Includes rounding fixes for float gaps and outlier capping.
=============================================================================
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CPCB Breakpoint Table
# ---------------------------------------------------------------------------
# Each tuple: (BLo, BHi, ILo, IHi)
#   BLo/BHi = concentration breakpoints
#   ILo/IHi = corresponding AQI index breakpoints

CPCB_BREAKPOINTS = {
    "PM2_5": [
        (0, 30, 0, 50),
        (31, 60, 51, 100),
        (61, 90, 101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400),
        (251, 500, 401, 500),
    ],
    "PM10": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 250, 101, 200),
        (251, 350, 201, 300),
        (351, 430, 301, 400),
        (431, 600, 401, 500),
    ],
    "NO2": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 180, 101, 200),
        (181, 280, 201, 300),
        (281, 400, 301, 400),
        (401, 1000, 401, 500),
    ],
    "SO2": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 380, 101, 200),
        (381, 800, 201, 300),
        (801, 1600, 301, 400),
        (1601, 2000, 401, 500),
    ],
}


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------
def _sub_index(Cp, BLo, BHi, ILo, IHi):
    """CPCB linear interpolation formula for a single pollutant."""
    return ((IHi - ILo) / (BHi - BLo)) * (Cp - BLo) + ILo


def compute_subindex(value, pollutant):
    """
    Compute the AQI sub-index for a single pollutant concentration.

    Parameters
    ----------
    value : float
        Measured concentration of the pollutant.
    pollutant : str
        One of 'PM2_5', 'PM10', 'NO2', 'SO2'.

    Returns
    -------
    float or None
        The sub-index value, or None if the value is NaN or out of range.
    """
    if pd.isna(value):
        return None

    # Round to nearest integer to bridge float gaps between breakpoints
    val_rounded = np.round(value)

    # Handle extreme outliers above the maximum CPCB bracket
    max_Cp = CPCB_BREAKPOINTS[pollutant][-1][1]
    if val_rounded > max_Cp:
        return 500.0  # Cap at maximum possible AQI

    for (BLo, BHi, ILo, IHi) in CPCB_BREAKPOINTS[pollutant]:
        if BLo <= val_rounded <= BHi:
            return _sub_index(val_rounded, BLo, BHi, ILo, IHi)

    return None


def compute_aqi(row):
    """
    Compute the AQI for a single row using the CPCB max sub-index rule.

    The AQI is the maximum sub-index across all available pollutants.
    CPCB recommends at least 3 parameters with one being PM2.5 or PM10.

    Parameters
    ----------
    row : pd.Series
        A row containing pollutant columns (PM2_5, PM10, NO2, SO2).

    Returns
    -------
    float or NaN
        The calculated AQI, rounded to nearest integer.
    """
    sub_indices = []

    for p in ["PM2_5", "PM10", "NO2", "SO2"]:
        if p in row and pd.notna(row[p]):
            si = compute_subindex(row[p], p)
            if si is not None:
                sub_indices.append(si)

    if not sub_indices:
        return np.nan

    return np.round(max(sub_indices))


def reconstruct_aqi_column(df):
    """
    Apply compute_aqi to an entire DataFrame, adding a 'Calculated_AQI' column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with pollutant columns (PM2_5, PM10, NO2, SO2).

    Returns
    -------
    pd.DataFrame
        Same DataFrame with 'Calculated_AQI' column added.
    """
    df = df.copy()
    df["Calculated_AQI"] = df.apply(compute_aqi, axis=1)
    valid = df["Calculated_AQI"].notna().sum()
    print(f"AQI reconstructed for {valid}/{len(df)} rows")
    return df
