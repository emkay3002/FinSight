from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    from backend.forecasting.service import (
        load_data,
        run_arima_service,
        run_lstm_service,
        evaluate_and_persist_service,
        plot_service,
    )
except ModuleNotFoundError:
    # Allow running from inside fintech_data_curator by adding project root
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from backend.forecasting.service import (  # type: ignore
        load_data,
        run_arima_service,
        run_lstm_service,
        evaluate_and_persist_service,
        plot_service,
    )

from .utils import LOGGER, DATA_DIR

try:
    from huggingface_hub import HfApi
except Exception:  # noqa: BLE001
    HfApi = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ARIMA/LSTM forecasts on curated data")
    parser.add_argument("--symbol", required=True, type=str)
    parser.add_argument("--model", required=True, choices=["arima", "lstm"], type=str)
    parser.add_argument("--horizon", required=True, type=int, help="Forecast horizon in days")
    parser.add_argument("--mongo", action="store_true", help="Load from and save to MongoDB")
    parser.add_argument("--upload", action="store_true", help="Upload trained model to Hugging Face")
    parser.add_argument("--repo_id", type=str, default=None, help="Hugging Face repo id (e.g., user/repo)")
    return parser.parse_args()


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

    last_date = df["date"].max()
    start_date = (pd.to_datetime(last_date).tz_convert("UTC") if pd.to_datetime(last_date).tzinfo else pd.to_datetime(last_date).tz_localize("UTC")) + pd.Timedelta(days=1)

    if model_name == "arima":
        y_pred_full, test_series = run_arima_service(df, horizon=horizon)
        y_true_for_eval = test_series.tail(min(len(test_series), len(y_pred_full))).values
        eval_pred = y_pred_full[: len(y_true_for_eval)]
        metrics = evaluate_and_persist_service(symbol, model_name, y_true_for_eval, eval_pred, y_pred_full, start_date, use_mongo)
        df_pred = pd.DataFrame({"date": pd.date_range(start=start_date, periods=len(y_pred_full), freq="D", tz="UTC"), "prediction": y_pred_full})
        plot_service(df_hist=df.tail(100), df_pred=df_pred, symbol=symbol, model_name=model_name)

    elif model_name == "lstm":
        y_pred_full, test_series, model, used_window = run_lstm_service(df, horizon=horizon)
        y_true_for_eval = test_series.tail(min(len(test_series), len(y_pred_full))).values
        eval_pred = y_pred_full[: len(y_true_for_eval)]
        metrics = evaluate_and_persist_service(symbol, model_name, y_true_for_eval, eval_pred, y_pred_full, start_date, use_mongo)
        df_pred = pd.DataFrame({"date": pd.date_range(start=start_date, periods=len(y_pred_full), freq="D", tz="UTC"), "prediction": y_pred_full})
        plot_service(df_hist=df.tail(100), df_pred=df_pred, symbol=symbol, model_name=model_name)

        models_dir = Path(__file__).resolve().parent.parent / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        weights_compliant = models_dir / f"{symbol}_lstm.weights.h5"
        weights_friendly = models_dir / f"{symbol}_lstm_weights.h5"
        model.save_weights(weights_compliant)
        try:
            model.save_weights(weights_friendly)
        except Exception:
            pass
        if args.upload and args.repo_id:
            maybe_upload_model(weights_compliant, repo_id=args.repo_id, filename=weights_compliant.name)

    else:
        raise ValueError("Unsupported model")


if __name__ == "__main__":
    main()
