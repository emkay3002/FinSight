from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go


OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_forecast(df_hist: pd.DataFrame, df_pred: pd.DataFrame, symbol: str, model_name: str) -> Path:
    """Render candlestick with forecast overlay and save as HTML.

    df_hist: DataFrame with columns [date(datetime), open, high, low, close]
    df_pred: DataFrame with columns [date(datetime), prediction(float)]
    """
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df_hist["date"],
            open=df_hist["open"],
            high=df_hist["high"],
            low=df_hist["low"],
            close=df_hist["close"],
            name=f"{symbol} OHLC",
        )
    )

    if not df_pred.empty:
        fig.add_trace(
            go.Scatter(
                x=df_pred["date"],
                y=df_pred["prediction"],
                mode="lines+markers",
                name=f"{model_name.upper()} forecast",
            )
        )

    fig.update_layout(
        title=f"{symbol} Historical and {model_name.upper()} Forecast",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
    )

    out_path = OUTPUT_DIR / f"forecast_{symbol}_{model_name}.html"
    fig.write_html(out_path)
    return out_path
