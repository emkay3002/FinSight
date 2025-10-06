from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .db.mongo_utils import fetch_symbol_data, save_forecast
from .models.arima_model import train_arima, forecast_arima
from .models.lstm_model import build_lstm_model, train_lstm, forecast_lstm
from .models.metrics import rmse, mae, mape
from .models.utils import create_sequences, fit_minmax_scaler, transform_with_scaler, inverse_transform_with_scaler
from .utils import LOGGER, DATA_DIR, safe_write_csv, safe_write_json
from .config import load_config
from .visualizations.plotly_charts import plot_forecast

try:
    from huggingface_hub import HfApi
except Exception:  # noqa: BLE001
    HfApi = None  # type: ignore


FORECASTS_DIR = DATA_DIR / "forecasts"
FORECASTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ARIMA/LSTM forecasts on curated data")
    parser.add_argument("--symbol", required=True, type=str)
    parser.add_argument("--model", required=True, choices=["arima", "lstm"], type=str)
    parser.add_argument("--horizon", required=True, type=int, help="Forecast horizon in days")
    parser.add_argument("--mongo", action="store_true", help="Load from and save to MongoDB")
    parser.add_argument("--upload", action="store_true", help="Upload trained model to Hugging Face")
    parser.add_argument("--repo_id", type=str, default=None, help="Hugging Face repo id (e.g., user/repo)")
    return parser.parse_args()


def load_data(symbol: str, use_mongo: bool, limit_days: int = 180) -> pd.DataFrame:
    if use_mongo:
        return fetch_symbol_data(symbol, limit_days=limit_days)
    # fallback: load from CSV if available
    csv_path = DATA_DIR / f"{symbol}_dataset.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found; run curator or enable --mongo")
    df = pd.read_csv(csv_path, parse_dates=["date"])  # date becomes naive dt
    df["date"] = pd.to_datetime(df["date"], utc=True)
    cols = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[cols].dropna().sort_values("date").tail(limit_days)
    return df.reset_index(drop=True)


def evaluate_and_persist(symbol: str, model_name: str, y_true: np.ndarray, y_pred_for_eval: np.ndarray, y_pred_full: np.ndarray, start_date: pd.Timestamp, use_mongo: bool) -> Dict[str, float]:
    metrics = {
        "rmse": rmse(y_true, y_pred_for_eval),
        "mae": mae(y_true, y_pred_for_eval),
        "mape": mape(y_true, y_pred_for_eval),
    }

    pred_len = int(len(y_pred_full))
    pred_dates = pd.date_range(start=start_date, periods=pred_len, freq="D", tz="UTC")
    df_pred = pd.DataFrame({"date": pred_dates, "prediction": y_pred_full})

    # Save locally
    out_csv = FORECASTS_DIR / f"{symbol}_{model_name}_h{pred_len}.csv"
    safe_write_csv(out_csv, df_pred)
    out_json = FORECASTS_DIR / f"{symbol}_{model_name}_h{pred_len}.json"
    safe_write_json(out_json, {"symbol": symbol, "model": model_name, "horizon": pred_len, "predictions": df_pred.to_dict(orient="records"), "metrics": metrics})

    # Save to Mongo
    if use_mongo:
        save_forecast(symbol, model_name, df_pred.to_dict(orient="records"), metrics)

    return metrics


def run_arima(df: pd.DataFrame, horizon: int) -> tuple[np.ndarray, pd.Series]:
    series = df.set_index("date")["close"].astype(float)
    split = int(len(series) * 0.8)
    train_series = series.iloc[:split]
    test_series = series.iloc[split:]

    fitted = train_arima(train_series)
    # Align forecast start at end of train
    in_sample_fc = forecast_arima(fitted, steps=len(test_series))
    # Out-of-sample future forecast for requested horizon
    future_fc = forecast_arima(fitted, steps=horizon)

    return future_fc, test_series


def run_lstm(df: pd.DataFrame, horizon: int, window: int = 20, epochs: int = 20) -> tuple[np.ndarray, pd.Series, object]:
    series = df.set_index("date")["close"].astype(float)
    split = int(len(series) * 0.8)
    train_vals = series.iloc[:split].values
    test_vals = series.iloc[split:].values

    # Dynamically adjust window for short histories
    max_window = max(5, min(window, max(1, len(train_vals) // 3)))

    scaler = fit_minmax_scaler(train_vals)
    train_scaled = transform_with_scaler(scaler, train_vals)
    test_scaled = transform_with_scaler(scaler, test_vals) if len(test_vals) > 0 else np.array([])

    X_train, y_train = create_sequences(train_scaled, max_window)
    if X_train.shape[0] == 0:
        raise ValueError(f"Insufficient data to train LSTM (series length={len(series)}, window={max_window})")

    model = build_lstm_model(max_window)
    model = train_lstm(model, X_train, y_train, epochs=epochs)

    # Prepare last window from the tail of the entire scaled series
    full_scaled = train_scaled if test_scaled.size == 0 else np.concatenate([train_scaled, test_scaled])
    last_window = full_scaled[-max_window:]
    if len(last_window) < max_window:
        raise ValueError("Insufficient data to create last window for forecasting")

    future_scaled = forecast_lstm(model, last_window, steps=horizon)
    future_pred = inverse_transform_with_scaler(scaler, future_scaled)

    return future_pred, pd.Series(test_vals, index=series.index[split:]), model


def maybe_upload_model(model_path: Path, repo_id: str, filename: str) -> None:
    if HfApi is None:
        LOGGER.info("huggingface_hub not available; skipping upload")
        return
    api = HfApi()
    api.upload_file(path_or_fileobj=str(model_path), path_in_repo=filename, repo_id=repo_id, repo_type="model")


def main() -> None:
    args = parse_args()
    symbol = args.symbol
    model_name = args.model
    horizon = int(args.horizon)
    use_mongo = args.mongo

    df = load_data(symbol, use_mongo=use_mongo, limit_days=180)

    # Visualization uses last window of history; predictions start one day after last date
    last_date = df["date"].max()
    start_date = (pd.to_datetime(last_date).tz_convert("UTC") if pd.to_datetime(last_date).tzinfo else pd.to_datetime(last_date).tz_localize("UTC")) + pd.Timedelta(days=1)

    if model_name == "arima":
        y_pred_full, test_series = run_arima(df, horizon=horizon)
        y_true_for_eval = test_series.tail(min(len(test_series), len(y_pred_full)))
        eval_pred = y_pred_full[: len(y_true_for_eval)]
        metrics = evaluate_and_persist(symbol, model_name, y_true_for_eval.values, eval_pred, y_pred_full, start_date, use_mongo)
        df_pred = pd.DataFrame({"date": pd.date_range(start=start_date, periods=len(y_pred_full), freq="D", tz="UTC"), "prediction": y_pred_full})
        plot_forecast(df_hist=df.tail(100), df_pred=df_pred, symbol=symbol, model_name=model_name)

    elif model_name == "lstm":
        y_pred_full, test_series, model = run_lstm(df, horizon=horizon)
        y_true_for_eval = test_series.tail(min(len(test_series), len(y_pred_full)))
        eval_pred = y_pred_full[: len(y_true_for_eval)]
        metrics = evaluate_and_persist(symbol, model_name, y_true_for_eval.values, eval_pred, y_pred_full, start_date, use_mongo)
        df_pred = pd.DataFrame({"date": pd.date_range(start=start_date, periods=len(y_pred_full), freq="D", tz="UTC"), "prediction": y_pred_full})
        plot_path = plot_forecast(df_hist=df.tail(100), df_pred=df_pred, symbol=symbol, model_name=model_name)

        # Save model weights
        models_dir = Path(__file__).resolve().parent.parent / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        weights_path = models_dir / f"{symbol}_lstm_weights.h5"
        model.save_weights(weights_path)
        if args.upload and args.repo_id:
            maybe_upload_model(weights_path, repo_id=args.repo_id, filename=weights_path.name)

    else:
        raise ValueError("Unsupported model")


if __name__ == "__main__":
    main()
