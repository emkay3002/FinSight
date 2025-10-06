from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, Optional

import pandas as pd
import requests
import yfinance as yf

from .config import load_config
from .utils import LOGGER


AssetType = Literal["stock", "crypto"]


@dataclass(frozen=True)
class PriceRequest:
    exchange: str
    symbol: str
    days: int
    asset_type: AssetType


def _end_start_dates(days: int) -> tuple[datetime, datetime]:
    end = datetime.utcnow().date()
    start = end - timedelta(days=days + 5)  # add buffer for weekends/holidays
    return datetime.combine(start, datetime.min.time()), datetime.combine(end, datetime.min.time())


def fetch_stock_prices(symbol: str, days: int) -> pd.DataFrame:
    start_dt, end_dt = _end_start_dates(days)
    LOGGER.info(f"Fetching stock prices for %s from %s to %s", symbol, start_dt.date(), end_dt.date())
    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=start_dt.date(), end=end_dt.date(), interval="1d")
    if hist.empty:
        raise ValueError(f"No historical data returned for stock {symbol}")

    df = hist.reset_index().rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").tail(days)
    df.insert(0, "symbol", symbol)
    return df[["date", "symbol", "open", "high", "low", "close", "volume"]]


def _fetch_crypto_prices_yfinance(symbol: str, days: int) -> Optional[pd.DataFrame]:
    # Many cryptos are available via yfinance using e.g., BTC-USD
    try:
        return fetch_stock_prices(symbol, days)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("yfinance crypto fetch failed for %s: %s", symbol, exc)
        return None


def _fetch_crypto_prices_cmc(symbol: str, days: int) -> Optional[pd.DataFrame]:
    cfg = load_config()
    if not cfg.coinmarketcap_api_key:
        return None

    # CoinMarketCap historical endpoint often requires symbol mapping and paid tiers.
    # Here we attempt a quotes/historical-like flow with daily aggregation via market quotes
    # As a safe fallback, we return None if unsuccessful.
    headers = {"X-CMC_PRO_API_KEY": cfg.coinmarketcap_api_key}
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=days + 5)

    params = {
        "symbol": symbol.replace("-USD", ""),  # crude normalization
        "time_start": int(start_dt.timestamp()),
        "time_end": int(end_dt.timestamp()),
        "interval": "daily",
    }
    try:
        resp = requests.get(cfg.coinmarketcap_base_url, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        quotes = data.get("data", {}).get("quotes", [])
        if not quotes:
            return None
        rows = []
        for q in quotes:
            ts = pd.to_datetime(q.get("time_open")).date()
            o = q.get("quote", {}).get("USD", {}).get("open")
            h = q.get("quote", {}).get("USD", {}).get("high")
            l = q.get("quote", {}).get("USD", {}).get("low")
            c = q.get("quote", {}).get("USD", {}).get("close")
            v = q.get("quote", {}).get("USD", {}).get("volume")
            if None in (ts, o, h, l, c, v):
                continue
            rows.append({
                "date": ts,
                "symbol": symbol,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
            })
        df = pd.DataFrame(rows).sort_values("date").tail(days)
        return df[["date", "symbol", "open", "high", "low", "close", "volume"]]
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("CoinMarketCap fetch failed for %s: %s", symbol, exc)
        return None


def fetch_crypto_prices(symbol: str, days: int) -> pd.DataFrame:
    LOGGER.info("Fetching crypto prices for %s for last %d days", symbol, days)
    df = _fetch_crypto_prices_yfinance(symbol, days)
    if df is not None and not df.empty:
        return df

    df = _fetch_crypto_prices_cmc(symbol, days)
    if df is not None and not df.empty:
        return df

    raise ValueError(f"No historical data returned for crypto {symbol}")


def fetch_prices(exchange: str, symbol: str, days: int, asset_type: AssetType) -> pd.DataFrame:
    if days < 5 or days > 30:
        raise ValueError("days must be between 5 and 30")

    if asset_type == "stock":
        return fetch_stock_prices(symbol, days)
    if asset_type == "crypto":
        return fetch_crypto_prices(symbol, days)

    raise ValueError(f"Unsupported asset_type: {asset_type}")
