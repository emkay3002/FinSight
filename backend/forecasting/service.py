from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Reuse existing curator modules
from fintech_data_curator.src.db.mongo_utils import fetch_symbol_data, save_forecast
from fintech_data_curator.src.models.arima_model import train_arima, forecast_arima
from fintech_data_curator.src.models.lstm_model import build_lstm_model, train_lstm, forecast_lstm
from fintech_data_curator.src.models.metrics import rmse, mae, mape
from fintech_data_curator.src.models.utils import create_sequences, fit_minmax_scaler, transform_with_scaler, inverse_transform_with_scaler
from fintech_data_curator.src.utils import DATA_DIR, safe_write_csv, safe_write_json, LOGGER
from fintech_data_curator.src.visualizations.plotly_charts import plot_forecast


FORECASTS_DIR = (Path(__file__).resolve().parents[2] / "fintech_data_curator" / "data" / "forecasts")
FORECASTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data(symbol: str, use_mongo: bool, limit_days: int = 180) -> pd.DataFrame:
    if use_mongo:
        return fetch_symbol_data(symbol, limit_days=limit_days)
    csv_path = Path(__file__).resolve().parents[2] / "fintech_data_curator" / "data" / f"{symbol}_dataset.csv"
    df = pd.read_csv(csv_path, parse_dates=["date"])  # type: ignore[arg-type]
    df["date"] = pd.to_datetime(df["date"], utc=True)
    cols = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
    return df[cols].dropna().sort_values("date").tail(limit_days).reset_index(drop=True)


def run_arima_service(df: pd.DataFrame, horizon: int) -> Tuple[np.ndarray, pd.Series]:
    series = df.set_index("date")["close"].astype(float)
    split = int(len(series) * 0.8)
    train_series = series.iloc[:split]
    test_series = series.iloc[split:]
    fitted = train_arima(train_series)
    future_fc = forecast_arima(fitted, steps=horizon)
    return future_fc, test_series


def run_lstm_service(df: pd.DataFrame, horizon: int, window: int = 20, epochs: int = 20) -> Tuple[np.ndarray, pd.Series, object, int]:
    series = df.set_index("date")["close"].astype(float)
    split = int(len(series) * 0.8)
    train_vals = series.iloc[:split].values
    test_vals = series.iloc[split:].values

    max_window = max(5, min(window, max(1, len(train_vals) // 3)))
    scaler = fit_minmax_scaler(train_vals)
    train_scaled = transform_with_scaler(scaler, train_vals)
    test_scaled = transform_with_scaler(scaler, test_vals) if len(test_vals) > 0 else np.array([])

    X_train, y_train = create_sequences(train_scaled, max_window)
    if X_train.shape[0] == 0 or y_train.shape[0] == 0:
        raise ValueError(f"Insufficient data to train LSTM (series length={len(series)}, window={max_window})")

    LOGGER.info("Starting LSTM training... samples=%d, window=%d, epochs=%d", X_train.shape[0], max_window, epochs)
    model = build_lstm_model(max_window)
    model = train_lstm(model, X_train, y_train, epochs=epochs, batch_size=32, verbose=1)

    full_scaled = train_scaled if test_scaled.size == 0 else np.concatenate([train_scaled, test_scaled])
    last_window = full_scaled[-max_window:]
    if len(last_window) < max_window:
        raise ValueError("Insufficient data to create last window for forecasting")

    future_scaled = forecast_lstm(model, last_window, steps=horizon)
    future_pred = inverse_transform_with_scaler(scaler, future_scaled)

    return future_pred, pd.Series(test_vals, index=series.index[split:]), model, max_window


def evaluate_and_persist_service(symbol: str, model_name: str, y_true: np.ndarray, y_pred_eval: np.ndarray, y_pred_full: np.ndarray, start_date: pd.Timestamp, use_mongo: bool) -> Dict[str, float]:
    metrics = {
        "rmse": rmse(y_true, y_pred_eval),
        "mae": mae(y_true, y_pred_eval),
        "mape": mape(y_true, y_pred_eval),
    }
    pred_dates = pd.date_range(start=start_date, periods=len(y_pred_full), freq="D", tz="UTC")
    df_pred = pd.DataFrame({"date": pred_dates, "prediction": y_pred_full})

    out_csv = FORECASTS_DIR / f"{symbol}_{model_name}_h{len(y_pred_full)}.csv"
    safe_write_csv(out_csv, df_pred)
    out_json = FORECASTS_DIR / f"{symbol}_{model_name}_h{len(y_pred_full)}.json"
    safe_write_json(out_json, {"symbol": symbol, "model": model_name, "horizon": len(y_pred_full), "predictions": df_pred.to_dict(orient="records"), "metrics": metrics})

    if use_mongo:
        save_forecast(symbol, model_name, df_pred.to_dict(orient="records"), metrics)

    return metrics


def plot_service(df_hist: pd.DataFrame, df_pred: pd.DataFrame, symbol: str, model_name: str) -> Path:
    return plot_forecast(df_hist=df_hist, df_pred=df_pred, symbol=symbol, model_name=model_name)
