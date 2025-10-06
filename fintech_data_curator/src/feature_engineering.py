from __future__ import annotations

import pandas as pd


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("date").reset_index(drop=True)
    # Daily returns
    out["return"] = out["close"].pct_change()
    # Rolling volatility (std of returns over 5 days)
    out["volatility_5d"] = out["return"].rolling(window=5, min_periods=3).std()
    # Moving averages
    out["ma_5"] = out["close"].rolling(window=5, min_periods=3).mean()
    out["ma_10"] = out["close"].rolling(window=10, min_periods=5).mean()
    return out
