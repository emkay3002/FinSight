from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from ..mongo_storage import get_db
from ..utils import LOGGER


def fetch_symbol_data(symbol: str, limit_days: int = 60) -> pd.DataFrame:
    db = get_db()
    if db is None:
        raise RuntimeError("MongoDB not configured")
    # Prefer merged datasets for richer features, but we only need OHLC
    cur = db.datasets.find({"symbol": symbol}).sort("date", 1)
    rows: List[Dict[str, Any]] = list(cur)
    if not rows:
        # fallback to prices
        cur = db.prices.find({"symbol": symbol}).sort("date", 1)
        rows = list(cur)
    if not rows:
        raise ValueError(f"No data found for symbol {symbol}")
    df = pd.DataFrame(rows)
    # Ensure datetime and required columns
    df["date"] = pd.to_datetime(df["date"], utc=True)
    cols = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[cols].dropna().sort_values("date")
    if limit_days and limit_days > 0:
        df = df.tail(limit_days)
    return df.reset_index(drop=True)


def save_forecast(symbol: str, model_name: str, predictions: List[Dict[str, Any]], metrics: Dict[str, Any]) -> None:
    db = get_db()
    if db is None:
        LOGGER.info("MongoDB not configured; skipping forecast save")
        return
    coll = db.datasets.database["forecasts"]
    # Normalize datetimes to UTC
    norm_preds: List[Dict[str, Any]] = []
    for p in predictions:
        d = dict(p)
        if "date" in d:
            d["date"] = pd.to_datetime(d["date"]).to_pydatetime().replace(tzinfo=timezone.utc)
        norm_preds.append(d)

    doc = {
        "symbol": symbol,
        "model": model_name,
        "created_at": datetime.now(timezone.utc),
        "metrics": metrics,
        "predictions": norm_preds,
    }
    coll.insert_one(doc)
