"""
=============================================================================
  FORECASTING — Prophet, SARIMA, CatBoost, Multivariate CatBoost
=============================================================================
  Four hindcast forecasting models with strict chronological splits.
  All models use the universal plot_forecasting_showdown for visualization.
=============================================================================
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

from plotting import plot_forecasting_showdown
from feature_engineering import engineer_tree_features, engineer_multivariate_features


# ---------------------------------------------------------------------------
# 1. Prophet Hindcast
# ---------------------------------------------------------------------------
def run_prophet_hindcast(df, station_name, split_date="2025-10-01", output_dir=None):
    """
    Run Facebook Prophet hindcast forecasting for a single station.

    Returns
    -------
    tuple of (model, test_dates, test_actuals, predictions, train_dates_model, train_preds)
    """
    from prophet import Prophet

    print(f"Initiating Prophet Forecasting for Station: {station_name}")

    station_df = df[df["Station"] == station_name].copy()
    station_df["Date"] = pd.to_datetime(station_df["Date"])

    prophet_df = station_df[["Date", "Calculated_AQI"]].rename(
        columns={"Date": "ds", "Calculated_AQI": "y"}
    )
    prophet_df = prophet_df.sort_values("ds").reset_index(drop=True)

    split_dt = pd.to_datetime(split_date)
    train = prophet_df[prophet_df["ds"] < split_dt].copy()
    test = prophet_df[prophet_df["ds"] >= split_dt].copy()

    print(f"Training months (Past): {len(train)}")
    print(f"Testing months (Future): {len(test)}\n")

    model = Prophet(
        yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False
    )
    model.fit(train)

    future = model.make_future_dataframe(periods=len(test), freq="MS")
    forecast = model.predict(future)

    predictions = forecast.tail(len(test))["yhat"].values
    actuals = test["y"].values
    test_dates = test["ds"].values
    
    train_preds = forecast.head(len(train))["yhat"].values
    train_dates_model = train["ds"].values

    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = mean_absolute_percentage_error(actuals, predictions) * 100

    print("--- Prophet Holdout Performance ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} AQI points")
    print(f"Root Mean Squared Error  : {rmse:.2f} AQI points")
    print(f"Mean Abs Percentage Error: {mape:.2f}%\n")

    plot_forecasting_showdown(
        train_dates=train_dates_model, train_y=train["y"].values,
        test_dates=test_dates, test_actuals=actuals,
        test_preds=predictions, station_name=station_name,
        model_name="Prophet", split_date=split_date,
        train_preds=train_preds,
        lower_ci=forecast.tail(len(test))["yhat_lower"].values,
        upper_ci=forecast.tail(len(test))["yhat_upper"].values,
        output_dir=output_dir
    )

    return model, test_dates, actuals, predictions, train_dates_model, train_preds, forecast.tail(len(test))["yhat_lower"].values, forecast.tail(len(test))["yhat_upper"].values


# ---------------------------------------------------------------------------
# 2. SARIMA Hindcast
# ---------------------------------------------------------------------------
def run_sarima_hindcast(df, station_name, split_date="2025-10-01", output_dir=None):
    """
    Run Auto-ARIMA (SARIMA) hindcast forecasting for a single station.

    Returns
    -------
    tuple of (model, test_dates, test_actuals, predictions, train_dates_model, train_preds)
    """
    from pmdarima import auto_arima

    print(f"Initiating SARIMA (Auto-ARIMA) Forecasting for Station: {station_name}")
    warnings.filterwarnings("ignore")

    station_df = df[df["Station"] == station_name].copy()
    station_df["Date"] = pd.to_datetime(station_df["Date"])
    station_df = station_df.sort_values("Date").set_index("Date")

    split_dt = pd.to_datetime(split_date)
    train = station_df[station_df.index < split_dt]["Calculated_AQI"]
    test = station_df[station_df.index >= split_dt]["Calculated_AQI"]

    print("Running mathematical grid search (this may take a moment)...")

    model = auto_arima(
        train, seasonal=True, m=12, stepwise=True, trace=False,
        error_action="ignore", suppress_warnings=True,
    )

    in_sample_preds, in_sample_confint = model.predict_in_sample(return_conf_int=True)
    forecast, conf_int = model.predict(n_periods=len(test), return_conf_int=True)

    in_sample_confint = np.asarray(in_sample_confint)
    conf_int = np.asarray(conf_int)

    actuals = test.values
    test_dates = test.index.values
    predictions = forecast.values if hasattr(forecast, 'values') else np.asarray(forecast)

    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = mean_absolute_percentage_error(actuals, predictions) * 100

    print("\n--- SARIMA Holdout Performance ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} AQI points")
    print(f"Root Mean Squared Error  : {rmse:.2f} AQI points")
    print(f"Mean Abs Percentage Error: {mape:.2f}%\n")

    in_sample_arr = np.asarray(in_sample_preds)
    start_idx = len(train) - len(in_sample_arr)
    train_dates_model = train.index.values[start_idx:]
    
    plot_forecasting_showdown(
        train_dates=train.index.values, train_y=train.values,
        test_dates=test_dates, test_actuals=actuals,
        test_preds=predictions, station_name=station_name,
        model_name="SARIMA", split_date=split_date,
        train_preds=in_sample_arr,
        lower_ci=conf_int[:, 0], upper_ci=conf_int[:, 1],
        output_dir=output_dir
    )

    return model, test_dates, actuals, predictions, train_dates_model, in_sample_arr, conf_int[:, 0], conf_int[:, 1]


# ---------------------------------------------------------------------------
# 3. CatBoost Hindcast (Univariate)
# ---------------------------------------------------------------------------
def run_catboost_hindcast(df, station_name, split_date="2025-10-01", output_dir=None):
    """
    Run univariate CatBoost hindcast with recursive multi-step forecasting.

    Returns
    -------
    tuple of (model, test_dates, test_actuals, predictions, train_dates_model, train_preds)
    """
    from catboost import CatBoostRegressor

    print(f"Initiating CatBoost Forecasting for Station: {station_name}")

    station_df = df[df["Station"] == station_name].copy()
    feature_df = engineer_tree_features(station_df)

    split_dt = pd.to_datetime(split_date)
    train = feature_df[feature_df["Date"] < split_dt].copy()
    test = feature_df[feature_df["Date"] >= split_dt].copy()

    features = ["Month", "Year", "Lag_1", "Lag_2", "Lag_12", "Rolling_Mean_3"]
    target = "Calculated_AQI"

    X_train = train[features]
    y_train = train[target]

    model = CatBoostRegressor(
        iterations=200, depth=3, learning_rate=0.05,
        loss_function="RMSE", verbose=0, random_seed=42,
    )
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    train_dates_model = train["Date"].values

    print("Executing strict recursive multi-step forecasting...")
    history = list(y_train.values)
    future_predictions = []

    for i in range(len(test)):
        current_date = test.iloc[i]["Date"]

        curr_month = current_date.month
        curr_year = current_date.year
        curr_lag_1 = history[-1]
        curr_lag_2 = history[-2]
        curr_lag_12 = history[-12]
        curr_roll_3 = np.mean(history[-3:])

        X_step = pd.DataFrame(
            [[curr_month, curr_year, curr_lag_1, curr_lag_2, curr_lag_12, curr_roll_3]],
            columns=features,
        )

        step_pred = model.predict(X_step)[0]
        future_predictions.append(step_pred)
        history.append(step_pred)

    actuals = test[target].values
    test_dates = test["Date"].values

    mae = mean_absolute_error(actuals, future_predictions)
    rmse = np.sqrt(mean_squared_error(actuals, future_predictions))
    mape = mean_absolute_percentage_error(actuals, future_predictions) * 100

    print("\n--- CatBoost Holdout Performance ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} AQI points")
    print(f"Root Mean Squared Error  : {rmse:.2f} AQI points")
    print(f"Mean Abs Percentage Error: {mape:.2f}%\n")

    plot_forecasting_showdown(
        train_dates=train["Date"].values, train_y=train[target].values,
        test_dates=test_dates, test_actuals=actuals,
        test_preds=future_predictions, station_name=station_name,
        model_name="CatBoost", split_date=split_date, train_preds=train_preds,
        output_dir=output_dir
    )

    return model, test_dates, actuals, future_predictions, train_dates_model, train_preds, None, None


# ---------------------------------------------------------------------------
# 4. Multivariate CatBoost Hindcast
# ---------------------------------------------------------------------------
def run_multivariate_catboost(df, station_name, split_date="2025-10-01", output_dir=None):
    """
    Run multivariate CatBoost hindcast using pollutant cross-lags.

    Returns
    -------
    tuple of (model, test_dates, test_actuals, predictions, train_dates_model, train_preds)
    """
    from catboost import CatBoostRegressor

    print(f"Initiating Multivariate CatBoost for Station: {station_name}")

    station_df = df[df["Station"] == station_name].copy()
    feature_df = engineer_multivariate_features(station_df)

    split_dt = pd.to_datetime(split_date)
    train = feature_df[feature_df["Date"] < split_dt].copy()
    test = feature_df[feature_df["Date"] >= split_dt].copy()

    target = "Calculated_AQI"
    features = [
        c for c in train.columns
        if c not in ["Date", "Station", "Region", "District", target]
        and ("Lag" in c or c in ["Month", "Year", "Rolling_Mean_3"])
    ]

    X_train = train[features]
    y_train = train[target]

    model = CatBoostRegressor(
        iterations=250, depth=4, learning_rate=0.05,
        loss_function="RMSE", verbose=0, random_seed=42,
    )
    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    train_dates_model = train["Date"].values

    print("Executing multivariate recursive forecasting...")
    history = list(y_train.values)
    future_predictions = []

    for i in range(len(test)):
        current_row = test.iloc[i]

        curr_lag_1 = history[-1]
        curr_lag_2 = history[-2]
        curr_lag_12 = history[-12]
        curr_roll_3 = np.mean(history[-3:])

        step_features = {
            "Month": current_row["Month"],
            "Year": current_row["Year"],
            "Lag_1": curr_lag_1,
            "Lag_2": curr_lag_2,
            "Lag_12": curr_lag_12,
            "Rolling_Mean_3": curr_roll_3,
        }

        for p in ["PM10", "PM2_5", "NO2", "SO2"]:
            col_name = f"{p}_Lag_12"
            if col_name in features:
                step_features[col_name] = current_row[col_name]

        X_step = pd.DataFrame([step_features], columns=features)
        step_pred = model.predict(X_step)[0]

        future_predictions.append(step_pred)
        history.append(step_pred)

    actuals = test[target].values
    test_dates = test["Date"].values

    mae = mean_absolute_error(actuals, future_predictions)
    rmse = np.sqrt(mean_squared_error(actuals, future_predictions))
    mape = mean_absolute_percentage_error(actuals, future_predictions) * 100

    print("\n--- Multivariate CatBoost Performance ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} AQI points")
    print(f"Root Mean Squared Error  : {rmse:.2f} AQI points")
    print(f"Mean Abs Percentage Error: {mape:.2f}%\n")

    plot_forecasting_showdown(
        train_dates=train["Date"].values, train_y=train[target].values,
        test_dates=test_dates, test_actuals=actuals,
        test_preds=future_predictions, station_name=station_name,
        model_name="Multivariate CatBoost", split_date=split_date,
        train_preds=train_preds,
        output_dir=output_dir
    )

    return model, test_dates, actuals, future_predictions, train_dates_model, train_preds, None, None


# ---------------------------------------------------------------------------
# 5. Hybrid Ensemble (Prophet + CatBoost)
# ---------------------------------------------------------------------------
def run_hybrid_ensemble(df, station_name, split_date="2025-10-01", output_dir=None):
    """
    Run Hybrid Ensemble (Prophet + CatBoost) hindcast.

    Returns
    -------
    tuple of (model, test_dates, test_actuals, predictions, train_dates_model, train_preds)
    """
    from prophet import Prophet
    from catboost import CatBoostRegressor

    print(f"Initiating Hybrid Architecture (Prophet + CatBoost) for: {station_name}")
    
    # 1. Prepare Data and Features (Using our deep seasonal echoes)
    station_df = df[df['Station'] == station_name].copy()
    feature_df = engineer_multivariate_features(station_df)
    
    split_dt = pd.to_datetime(split_date)
    train = feature_df[feature_df['Date'] < split_dt].copy()
    test = feature_df[feature_df['Date'] >= split_dt].copy()
    
    target = 'Calculated_AQI'
    features = [c for c in train.columns if c not in ['Date', 'Station', 'Region', 'District', target] and ('Lag' in c or c in ['Month', 'Year', 'Rolling_Mean_3'])]

    # ==========================================
    # PHASE 1: THE PROPHET MACRO-TREND
    # ==========================================
    print("Training Phase 1: Prophet Baseline...")
    prophet_train = train[['Date', target]].rename(columns={'Date': 'ds', target: 'y'})
    
    prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    prophet_model.fit(prophet_train)
    
    # Get Prophet's guesses for the past to calculate its mistakes
    prophet_train_preds = prophet_model.predict(prophet_train)['yhat'].values
    
    # Calculate the Residuals (Actual - Prophet Guess)
    train['Prophet_Residual'] = train[target] - prophet_train_preds

    # ==========================================
    # PHASE 2: THE CATBOOST MICRO-CORRECTION
    # ==========================================
    print("Training Phase 2: CatBoost Residual Correction...")
    X_train = train[features]
    y_train_res = train['Prophet_Residual'] # <--- Target is now the MISTAKES, not the AQI!
    
    cat_model = CatBoostRegressor(
        iterations=250,
        depth=4,
        learning_rate=0.05,
        loss_function='RMSE',
        verbose=0,
        random_seed=42
    )
    cat_model.fit(X_train, y_train_res)

    # ==========================================
    # PHASE 3: THE FUTURE RECURSIVE FORECAST
    # ==========================================
    print("Executing Hybrid Forecast...")
    
    # A. Get Prophet's baseline guesses for the test set
    prophet_test = test[['Date']].rename(columns={'Date': 'ds'})
    prophet_test_preds = prophet_model.predict(prophet_test)['yhat'].values
    
    # B. Recursively predict CatBoost's corrections
    history_aqi = list(train[target].values)
    future_predictions = []
    
    for i in range(len(test)):
        current_row = test.iloc[i]
        
        # Calculate dynamic AQI history
        curr_lag_1 = history_aqi[-1]
        curr_lag_2 = history_aqi[-2]
        curr_lag_12 = history_aqi[-12]
        curr_roll_3 = np.mean(history_aqi[-3:])
        
        step_features = {
            'Month': current_row['Month'],
            'Year': current_row['Year'],
            'Lag_1': curr_lag_1,
            'Lag_2': curr_lag_2,
            'Lag_12': curr_lag_12,
            'Rolling_Mean_3': curr_roll_3
        }
        
        for p in ['PM10', 'PM2_5', 'NO2', 'SO2']:
            col_name = f'{p}_Lag_12'
            if col_name in features:
                step_features[col_name] = current_row[col_name]
                
        X_step = pd.DataFrame([step_features], columns=features)
        
        # CatBoost predicts the RESIDUAL (the correction)
        step_res_pred = cat_model.predict(X_step)[0]
        
        # FINAL MATH: Prophet Base + CatBoost Correction
        final_step_pred = prophet_test_preds[i] + step_res_pred
        
        future_predictions.append(final_step_pred)
        history_aqi.append(final_step_pred)

    # ==========================================
    # EVALUATION & PACKAGING
    # ==========================================
    actuals = test[target].values
    test_dates = test['Date'].values
    
    mae = mean_absolute_error(actuals, future_predictions)
    rmse = np.sqrt(mean_squared_error(actuals, future_predictions))
    mape = mean_absolute_percentage_error(actuals, future_predictions) * 100
    
    print("\n--- Hybrid Ensemble Performance ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} AQI points")
    print(f"Root Mean Squared Error  : {rmse:.2f} AQI points")
    print(f"Mean Abs Percentage Error: {mape:.2f}%\n")
    
    # Package train predictions for plotting
    train_cat_res_preds = cat_model.predict(X_train)
    final_train_preds = prophet_train_preds + train_cat_res_preds
    train_dates_model = train['Date'].values
    
    plot_forecasting_showdown(
        train_dates=train["Date"].values, train_y=train[target].values,
        test_dates=test_dates, test_actuals=actuals,
        test_preds=future_predictions, station_name=station_name,
        model_name="Hybrid", split_date=split_date,
        train_preds=final_train_preds,
        output_dir=output_dir
    )
    
    return (prophet_model, cat_model), test_dates, actuals, future_predictions, train_dates_model, final_train_preds, None, None
