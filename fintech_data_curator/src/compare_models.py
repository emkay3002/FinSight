from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    from backend.forecasting.service import load_data, run_arima_service, run_lstm_service
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from backend.forecasting.service import load_data, run_arima_service, run_lstm_service  # type: ignore

from fintech_data_curator.src.models.metrics import rmse, mae, mape


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare ARIMA vs LSTM forecasts")
    p.add_argument("--symbol", required=True, type=str)
    p.add_argument("--horizon", required=True, type=int)
    p.add_argument("--mongo", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    symbol = args.symbol
    horizon = int(args.horizon)

    df = load_data(symbol, use_mongo=args.mongo, limit_days=180)

    # ARIMA
    arima_pred, arima_test = run_arima_service(df, horizon=horizon)
    arima_true = arima_test.tail(min(len(arima_test), len(arima_pred))).values
    arima_eval_pred = arima_pred[: len(arima_true)]
    arima_metrics = {"rmse": rmse(arima_true, arima_eval_pred), "mae": mae(arima_true, arima_eval_pred), "mape": mape(arima_true, arima_eval_pred)}

    # LSTM
    lstm_pred, lstm_test, _model, _win = run_lstm_service(df, horizon=horizon)
    lstm_true = lstm_test.tail(min(len(lstm_test), len(lstm_pred))).values
    lstm_eval_pred = lstm_pred[: len(lstm_true)]
    lstm_metrics = {"rmse": rmse(lstm_true, lstm_eval_pred), "mae": mae(lstm_true, lstm_eval_pred), "mape": mape(lstm_true, lstm_eval_pred)}

    print(f"Symbol: {symbol} | Horizon: {horizon} days\n")
    print("Model     RMSE        MAE         MAPE (%)")
    print("-------------------------------------------")
    print(f"ARIMA  {arima_metrics['rmse']:.4f}   {arima_metrics['mae']:.4f}   {arima_metrics['mape']:.2f}")
    print(f"LSTM   {lstm_metrics['rmse']:.4f}   {lstm_metrics['mae']:.4f}   {lstm_metrics['mape']:.2f}")


if __name__ == "__main__":
    main()
