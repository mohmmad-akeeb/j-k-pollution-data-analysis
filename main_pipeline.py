#!/usr/bin/env python3
"""
=============================================================================
  MAIN PIPELINE — J&K Air Quality Analysis Orchestrator
=============================================================================
  End-to-end pipeline:
    1. Load data
    2. Reconstruct AQI (CPCB formula)
    3. Triage stations into Tiers
    4. Engineer features for MICE
    5. Run MICE imputation on Tier 1
    6. Recalculate AQI post-imputation
    7. Evaluate AQI reconstruction quality
    8. Tier 2 Temporal Imputation
    9. Tier 2 Spatial Salvage
    10. Recalculate AQI for Tier 2
    11. Forecast Showdown

  Usage:
    python main_pipeline.py                                        # Full pipeline
    python main_pipeline.py --skip-forecast                        # Data prep only
    python main_pipeline.py --station "Jammu JKPCC Narwal  184"    # Single station
    python main_pipeline.py --models prophet catboost              # Select models
    python main_pipeline.py --missing-threshold 0.20               # Custom threshold
=============================================================================
"""

import argparse
import sys
import os
import warnings

import pandas as pd

# Local modules
from data_loader import load_pollution_data
from aqi_calculator import reconstruct_aqi_column
from station_triage import compute_station_correlation, triage_stations
from feature_engineering import engineer_imputation_features
from mice_imputation import execute_mice_imputation
from plotting import evaluate_aqi_reconstruction, plot_combined_forecasts
from tier2_imputation import temporal_imputation, spatial_salvage_tier2


# ---------------------------------------------------------------------------
# Available forecasting models (lazy-imported to avoid heavy deps at startup)
# ---------------------------------------------------------------------------
AVAILABLE_MODELS = ["prophet", "sarima", "catboost", "multivariate_catboost", "hybrid"]


def run_forecasts(df_imputed, station_name, models, split_date="2025-10-01", output_dir=None):
    """Run selected forecasting models on a single station."""
    import json
    import os
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    from forecasting import (
        run_prophet_hindcast,
        run_sarima_hindcast,
        run_catboost_hindcast,
        run_multivariate_catboost,
        run_hybrid_ensemble,
    )

    results = {}
    predictions_dict = {}
    train_predictions_dict = {}
    test_dates = None
    test_actuals = None
    train_dates = None
    train_y = None

    # We need train data for the combined plot
    station_df = df_imputed[df_imputed["Station"] == station_name].copy()
    station_df["Date"] = pd.to_datetime(station_df["Date"])
    station_df = station_df.sort_values("Date").reset_index(drop=True)
    split_dt = pd.to_datetime(split_date)
    train = station_df[station_df["Date"] < split_dt]
    train_dates = train["Date"].values
    train_y = train["Calculated_AQI"].values

    json_data = {
        "station_name": station_name,
        "split_date": split_date,
        "train_dates_full": [pd.Timestamp(d).isoformat() for d in train_dates],
        "train_y_full": train_y.tolist(),
        "test_dates": None,
        "test_actuals": None,
        "models": {}
    }

    def _store_json_data(model_name_key, t_dates, t_acts, preds, tr_dates, tr_preds, lower_ci, upper_ci):
        if json_data["test_dates"] is None:
            json_data["test_dates"] = [pd.Timestamp(d).isoformat() for d in t_dates]
            json_data["test_actuals"] = t_acts.tolist() if hasattr(t_acts, 'tolist') else list(t_acts)
            
        mae = mean_absolute_error(t_acts, preds)
        rmse = np.sqrt(mean_squared_error(t_acts, preds))
        mape = mean_absolute_percentage_error(t_acts, preds) * 100
        
        def to_list(obj):
            if obj is None: return None
            if hasattr(obj, "tolist"): return obj.tolist()
            return list(obj)
        
        json_data["models"][model_name_key] = {
            "metrics": {"MAE": mae, "RMSE": rmse, "MAPE": mape},
            "test_predictions": to_list(preds),
            "train_dates_model": [pd.Timestamp(d).isoformat() for d in tr_dates] if tr_dates is not None else None,
            "train_predictions": to_list(tr_preds),
            "lower_ci": to_list(lower_ci),
            "upper_ci": to_list(upper_ci),
        }

    if "prophet" in models:
        print("\n" + "=" * 65)
        print("  PROPHET FORECAST")
        print("=" * 65)
        model, t_dates, t_acts, preds, tr_dates, tr_preds, lower_ci, upper_ci = run_prophet_hindcast(df_imputed, station_name, split_date, output_dir)
        results["prophet"] = model
        predictions_dict["Prophet"] = preds
        train_predictions_dict["Prophet"] = (tr_dates, tr_preds)
        test_dates, test_actuals = t_dates, t_acts
        _store_json_data("Prophet", t_dates, t_acts, preds, tr_dates, tr_preds, lower_ci, upper_ci)

    if "sarima" in models:
        print("\n" + "=" * 65)
        print("  SARIMA FORECAST")
        print("=" * 65)
        model, t_dates, t_acts, preds, tr_dates, tr_preds, lower_ci, upper_ci = run_sarima_hindcast(df_imputed, station_name, split_date, output_dir)
        results["sarima"] = model
        predictions_dict["SARIMA"] = preds
        train_predictions_dict["SARIMA"] = (tr_dates, tr_preds)
        test_dates, test_actuals = t_dates, t_acts
        _store_json_data("SARIMA", t_dates, t_acts, preds, tr_dates, tr_preds, lower_ci, upper_ci)

    if "catboost" in models:
        print("\n" + "=" * 65)
        print("  CATBOOST FORECAST")
        print("=" * 65)
        model, t_dates, t_acts, preds, tr_dates, tr_preds, lower_ci, upper_ci = run_catboost_hindcast(df_imputed, station_name, split_date, output_dir)
        results["catboost"] = model
        predictions_dict["CatBoost"] = preds
        train_predictions_dict["CatBoost"] = (tr_dates, tr_preds)
        test_dates, test_actuals = t_dates, t_acts
        _store_json_data("CatBoost", t_dates, t_acts, preds, tr_dates, tr_preds, lower_ci, upper_ci)

    if "multivariate_catboost" in models:
        print("\n" + "=" * 65)
        print("  MULTIVARIATE CATBOOST FORECAST")
        print("=" * 65)
        model, t_dates, t_acts, preds, tr_dates, tr_preds, lower_ci, upper_ci = run_multivariate_catboost(df_imputed, station_name, split_date, output_dir)
        results["multivariate_catboost"] = model
        predictions_dict["Multivariate CatBoost"] = preds
        train_predictions_dict["Multivariate CatBoost"] = (tr_dates, tr_preds)
        test_dates, test_actuals = t_dates, t_acts
        _store_json_data("Multivariate CatBoost", t_dates, t_acts, preds, tr_dates, tr_preds, lower_ci, upper_ci)

    if "hybrid" in models:
        print("\n" + "=" * 65)
        print("  HYBRID ENSEMBLE FORECAST (PROPHET + CATBOOST)")
        print("=" * 65)
        model, t_dates, t_acts, preds, tr_dates, tr_preds, lower_ci, upper_ci = run_hybrid_ensemble(df_imputed, station_name, split_date, output_dir)
        results["hybrid"] = model
        predictions_dict["Hybrid (Prophet+CatBoost)"] = preds
        train_predictions_dict["Hybrid (Prophet+CatBoost)"] = (tr_dates, tr_preds)
        test_dates, test_actuals = t_dates, t_acts
        _store_json_data("Hybrid (Prophet+CatBoost)", t_dates, t_acts, preds, tr_dates, tr_preds, lower_ci, upper_ci)

    if output_dir:
        json_path = os.path.join(output_dir, "model_results_and_plot_data.json")
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        print(f"\nSaved JSON metrics and plot data to {json_path}")

    if predictions_dict and test_dates is not None:
        plot_combined_forecasts(
            train_dates, train_y, test_dates, test_actuals, 
            predictions_dict, train_predictions_dict, station_name, split_date, output_dir=output_dir
        )

    return results


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="J&K Air Quality Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data", type=str, default="j_and_k_pollution_data.csv",
        help="Path to the input CSV (default: j_and_k_pollution_data.csv)",
    )
    parser.add_argument(
        "--missing-threshold", type=float, default=0.20,
        help="Maximum core missingness for Tier 1 (default: 0.20)",
    )
    parser.add_argument(
        "--skip-forecast", action="store_true",
        help="Run data prep only, skip forecasting",
    )
    parser.add_argument(
        "--station", type=str, default=None,
        help="Station name to forecast (default: first Tier 1 station)",
    )
    parser.add_argument(
        "--models", nargs="+", choices=AVAILABLE_MODELS, default=AVAILABLE_MODELS,
        help="Which forecasting models to run (default: all)",
    )
    parser.add_argument(
        "--split-date", type=str, default="2025-10-01",
        help="Train/test split date (default: 2025-10-01)",
    )
    parser.add_argument(
        "--output", type=str, default="final_imputed_dataset.csv",
        help="Output path for final consolidated data (default: final_imputed_dataset.csv)",
    )

    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Load Data ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  STEP 1: LOADING DATA")
    print("=" * 65)
    df = load_pollution_data(args.data)

    # ── Step 2: Compute AQI Correlations ──────────────────────────────
    print("\n" + "=" * 65)
    print("  STEP 2: COMPUTING AQI CORRELATIONS")
    print("=" * 65)
    station_corr = compute_station_correlation(df)

    # ── Step 3: Triage Stations ───────────────────────────────────────
    print("\n" + "=" * 65)
    print("  STEP 3: STATION TRIAGE")
    print("=" * 65)
    triage_summary, df_gold, df_salvage, df_junk = triage_stations(
        df, station_corr, missing_threshold=args.missing_threshold
    )

    print("\nTier 1 (Gold Standard) stations:")
    for s in df_gold["Station"].unique():
        print(f"  ★ {s}")

    # ── Step 4: Engineer Features for MICE ────────────────────────────
    print("\n" + "=" * 65)
    print("  STEP 4: ENGINEERING IMPUTATION FEATURES")
    print("=" * 65)
    df_tier1_engineered = engineer_imputation_features(df_gold)

    # ── Step 5: MICE Imputation ───────────────────────────────────────
    print("\n" + "=" * 65)
    print("  STEP 5: MICE IMPUTATION (TIER 1)")
    print("=" * 65)
    df_tier1_imputed = execute_mice_imputation(df_tier1_engineered)

    # ── Step 6: Recalculate AQI Post-Imputation ──────────────────────
    print("\n" + "=" * 65)
    print("  STEP 6: RECALCULATING AQI (CPCB FORMULA)")
    print("=" * 65)
    df_tier1_imputed = reconstruct_aqi_column(df_tier1_imputed)

    # ── Step 7: Evaluate Reconstruction ──────────────────────────────
    print("\n" + "=" * 65)
    print("  STEP 7: AQI RECONSTRUCTION EVALUATION")
    print("=" * 65)
    eval_results = evaluate_aqi_reconstruction(df_gold, df_tier1_imputed, output_dir=output_dir)

    # ── Step 8: Tier 2 Temporal Imputation ───────────────────────────
    print("\n" + "=" * 65)
    print("  STEP 8: TIER 2 TEMPORAL IMPUTATION")
    print("=" * 65)
    df_tier2_temporal = temporal_imputation(df_salvage)

    # ── Step 9: Tier 2 Spatial Salvage ───────────────────────────────
    print("\n" + "=" * 65)
    print("  STEP 9: TIER 2 SPATIAL SALVAGE")
    print("=" * 65)
    df_tier2_rescued = spatial_salvage_tier2(df_tier2_temporal, df_gold)

    # ── Step 10: Recalculate AQI for Tier 2 ──────────────────────────
    print("\n" + "=" * 65)
    print("  STEP 10: RECALCULATING AQI FOR TIER 2")
    print("=" * 65)
    df_tier2_rescued = reconstruct_aqi_column(df_tier2_rescued)

    # ── Save Consolidated Final Output ───────────────────────────────
    print("\n" + "=" * 65)
    print("  CONSOLIDATING FINAL DATASET")
    print("=" * 65)
    df_tier1_imputed["Data_Quality_Tier"] = "Tier_1_Gold"
    df_tier2_rescued["Data_Quality_Tier"] = "Tier_2_Rescued"
    df_junk["Data_Quality_Tier"] = "Tier_3_Raw"
    
    final_df = pd.concat([df_tier1_imputed, df_tier2_rescued, df_junk], ignore_index=True)
    final_df.to_csv(args.output, index=False)
    print(f"Final consolidated data saved to: {args.output}")

    # ── Step 11: Forecasting ──────────────────────────────────────────
    if not args.skip_forecast:
        print("\n" + "=" * 65)
        print("  STEP 11: FORECASTING")
        print("=" * 65)

        if args.station:
            forecast_station = args.station
        else:
            forecast_station = df_gold["Station"].unique()[0]

        print(f"Target station: {forecast_station}")
        print(f"Models: {', '.join(args.models)}")
        print(f"Split date: {args.split_date}")

        results = run_forecasts(
            df_tier1_imputed, forecast_station, args.models, args.split_date, output_dir=output_dir
        )

        print("\n" + "=" * 65)
        print("  PIPELINE COMPLETE")
        print("=" * 65)
    else:
        print("\n" + "=" * 65)
        print("  PIPELINE COMPLETE (forecasting skipped)")
        print("=" * 65)


if __name__ == "__main__":
    main()
