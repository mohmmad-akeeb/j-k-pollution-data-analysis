"""
=============================================================================
  PLOTTING — Evaluation & Forecasting Visualizations
=============================================================================
  1. evaluate_aqi_reconstruction  — 3-panel scatter/residual/error plot
  2. plot_forecasting_showdown    — universal forecast vs actuals plot
  3. plot_combined_forecasts      — combined model comparison
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_aqi_reconstruction(df_original, df_reconstructed, output_dir=None):
    """
    Evaluate AQI reconstruction quality with metrics and 3-panel diagnostic plot.

    Parameters
    ----------
    df_original : pd.DataFrame
        Original dataset with 'Station', 'Date', 'AQI' columns.
    df_reconstructed : pd.DataFrame
        Imputed dataset with 'Station', 'Date', 'Calculated_AQI' columns.
    output_dir : str, optional
        Directory to save the plot.

    Returns
    -------
    pd.DataFrame
        Merged evaluation DataFrame with actual and calculated AQI.
    """
    print("Initiating AQI Validation Diagnostics...\n")

    eval_df = pd.merge(
        df_original[["Station", "Date", "AQI"]],
        df_reconstructed[["Station", "Date", "Calculated_AQI"]],
        on=["Station", "Date"],
        how="inner",
    )
    eval_df = eval_df.dropna(subset=["AQI", "Calculated_AQI"])

    y_true = eval_df["AQI"]
    y_pred = eval_df["Calculated_AQI"]

    if len(y_true) == 0:
        raise ValueError(
            "Zero overlapping rows found. Did you run compute_aqi to create 'Calculated_AQI'?"
        )

    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"--- Performance Metrics ---")
    print(f"Observations Evaluated : {len(y_true)}")
    print(f"Mean Absolute Error (MAE): {mae:.2f} AQI points")
    print(f"Root Mean Squared Error  : {rmse:.2f} AQI points")
    print(f"R-squared (R2) Score     : {r2:.4f}\n")

    # 3-panel diagnostic plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.suptitle("AQI Reconstruction Diagnostics", fontsize=16)

    # Plot A: Scatter
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5, ax=axes[0])
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([0, max_val], [0, max_val], color="red", linestyle="--", label="Perfect Fit")
    axes[0].set_title("Actual vs. Calculated AQI")
    axes[0].set_xlabel("Actual Sensor AQI")
    axes[0].set_ylabel("Calculated AQI")
    axes[0].legend()

    # Plot B: Residual Distribution
    residuals = y_true - y_pred
    sns.histplot(residuals, bins=50, kde=True, ax=axes[1], color="purple")
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_title("Distribution of Errors (Residuals)")
    axes[1].set_xlabel("Error (Actual - Calculated)")
    axes[1].set_ylabel("Frequency")

    # Plot C: Error by Magnitude
    sns.scatterplot(x=y_true, y=residuals, alpha=0.4, ax=axes[2], color="teal")
    axes[2].axhline(0, color="red", linestyle="--")
    axes[2].set_title("Residuals vs. Actual Values")
    axes[2].set_xlabel("Actual Sensor AQI")
    axes[2].set_ylabel("Error (Residual)")

    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "aqi_reconstruction_diagnostics.png"))
        plt.close()
    else:
        plt.show()

    return eval_df


def plot_forecasting_showdown(
    train_dates, train_y, test_dates, test_actuals, test_preds,
    station_name, model_name, split_date,
    train_preds=None, lower_ci=None, upper_ci=None, output_dir=None
):
    """
    Universal plotting function for all forecasting models.
    """
    plt.figure(figsize=(14, 6), dpi=300)

    # Training data
    plt.plot(train_dates, train_y, label="Training Data (Actual)", color="black", linewidth=2)

    # Test data (actual)
    plt.plot(test_dates, test_actuals, label="Test Data (Actual True Future)",
             color="black", linestyle=":", linewidth=2)

    # Model forecast
    model_color = "green" if "Boost" in model_name else ("blue" if "Prophet" in model_name else "red")
    plt.plot(test_dates, test_preds, label=f"{model_name} Forecast",
             color=model_color, linestyle="--", linewidth=2)

    # Confidence intervals
    if lower_ci is not None and upper_ci is not None:
        plt.fill_between(test_dates, lower_ci, upper_ci,
                         color=model_color, alpha=0.2, label="Confidence Interval")

    # In-sample predictions
    if train_preds is not None:
        start_idx = len(train_dates) - len(train_preds)
        plt.plot(train_dates[start_idx:], train_preds,
                 label=f"{model_name} Fitted (Past)",
                 color=model_color, linestyle="--", linewidth=1.5, alpha=0.6)

    plt.axvline(x=pd.to_datetime(split_date), color="purple", linestyle="-",
                alpha=0.8, label="Train/Test Split")

    plt.title(f"{model_name} Forecast vs Reality: {station_name}", fontsize=16, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("AQI", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        safe_name = model_name.replace(" ", "_").lower()
        plt.savefig(os.path.join(output_dir, f"{safe_name}_forecast.png"), bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_combined_forecasts(train_dates_full, train_y_full, test_dates, test_actuals, 
                            predictions_dict, train_predictions_dict, station_name, split_date, output_dir=None):
    """
    Generates three journal-quality plots for model comparison:
    1. A full timeline plot (Training + Testing).
    2. A zoomed-in plot focusing ONLY on the testing window.
    3. A zoomed-in plot focusing ONLY on the training window.
    """
    # Academic-friendly color palette (colorblind safe & prints well)
    colors = ['#d7191c', '#2b83ba', '#4dac26', '#7b3294', '#fdae61']
    # Distinct markers help reviewers differentiate lines if printed in black and white
    markers = ['o', 's', '^', 'D', 'v']

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # =========================================================
    # PLOT 1: THE FULL TIMELINE (MACRO VIEW)
    # =========================================================
    plt.figure(figsize=(14, 6), dpi=300) # 300 DPI for publication quality
    
    plt.plot(train_dates_full, train_y_full, label="Training Data (Observed)", color="black", linewidth=1.5, alpha=0.8)
    plt.plot(test_dates, test_actuals, label="Testing Data (Observed)", color="black", linestyle=":", linewidth=2.5)

    for i, (model_name, preds) in enumerate(predictions_dict.items()):
        plt.plot(test_dates, preds, label=f"{model_name} Test Forecast", 
                 color=colors[i % len(colors)], linestyle="--", linewidth=2)
                 
    for i, (model_name, (tr_dates, tr_preds)) in enumerate(train_predictions_dict.items()):
        plt.plot(tr_dates, tr_preds, color=colors[i % len(colors)], linestyle="-.", linewidth=1.2, alpha=0.6)

    plt.axvline(x=pd.to_datetime(split_date), color="grey", linestyle="-", alpha=0.5, label="Train/Test Split")

    plt.title(f"Macro-Level Forecasting Showdown: {station_name}", fontsize=16, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("AQI", fontsize=12)
    plt.legend(loc='upper left', framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "forecast_showdown_full_timeline.png"), bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # =========================================================
    # PLOT 2: THE ZOOMED TEST WINDOW (MICRO VIEW)
    # =========================================================
    plt.figure(figsize=(10, 6), dpi=300)
    
    plt.plot(test_dates, test_actuals, label="Actual True Future", 
             color="black", linestyle="-", linewidth=3, marker='*', markersize=10, alpha=0.8)

    for i, (model_name, preds) in enumerate(predictions_dict.items()):
        plt.plot(test_dates, preds, label=f"{model_name} Prediction", 
                 color=colors[i % len(colors)], linestyle="--", linewidth=2, 
                 marker=markers[i % len(markers)], markersize=7)

    plt.title(f"Micro-Level Forecast Accuracy (Holdout Period): {station_name}", fontsize=16, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("AQI", fontsize=12)
    plt.legend(loc='best', framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "forecast_showdown_zoomed_test.png"), bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
    # =========================================================
    # PLOT 3: THE ZOOMED TRAINING WINDOW (MICRO VIEW)
    # =========================================================
    plt.figure(figsize=(12, 6), dpi=300)
    
    plt.plot(train_dates_full, train_y_full, label="Training Data (Observed)", 
             color="black", linestyle="-", linewidth=2.5, marker='*', markersize=8, alpha=0.8)

    for i, (model_name, (tr_dates, tr_preds)) in enumerate(train_predictions_dict.items()):
        plt.plot(tr_dates, tr_preds, label=f"{model_name} Fitted", 
                 color=colors[i % len(colors)], linestyle="--", linewidth=1.5, 
                 marker=markers[i % len(markers)], markersize=6, alpha=0.8)

    plt.title(f"Micro-Level Model Fit (Training Period): {station_name}", fontsize=16, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("AQI", fontsize=12)
    plt.legend(loc='best', framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "forecast_showdown_zoomed_train.png"), bbox_inches='tight')
        plt.close()
        print(f"Saved journal-quality forecasting plots to {output_dir}/")
    else:
        plt.show()
