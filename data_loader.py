"""
=============================================================================
  DATA LOADER — J&K Air Quality Dataset
=============================================================================
  Handles dtype-safe CSV loading with datetime parsing.
=============================================================================
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POLLUTANTS = ["AQI", "PM10", "PM2_5", "NO2", "SO2"]

DTYPE_MAP = {
    "Region": "string",
    "District": "string",
    "Station": "string",
    "AQI": "float64",
    "PM10": "float64",
    "PM2_5": "float64",
    "NO2": "float64",
    "SO2": "float64",
}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_pollution_data(path: str) -> pd.DataFrame:
    """
    Load the J&K pollution CSV with enforced dtypes and datetime parsing.

    Parameters
    ----------
    path : str
        Path to the CSV file. Expected columns:
        Region, District, Station, Date, AQI, PM10, PM2_5, NO2, SO2

    Returns
    -------
    pd.DataFrame
        Clean DataFrame with proper types and Date as datetime.
    """
    df = pd.read_csv(
        path,
        dtype=DTYPE_MAP,
        converters={"Date": lambda x: pd.to_datetime(x)},
    )

    print(f"Loaded {len(df)} rows, {df['Station'].nunique()} stations from {path}")
    return df
