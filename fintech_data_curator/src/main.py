from __future__ import annotations

import argparse
from typing import Literal

import pandas as pd

from .data_fetcher import fetch_prices
from .feature_engineering import add_technical_indicators
from .news_scraper import fetch_news
from .sentiment_analysis import add_sentiment
from .data_merger import merge_price_and_news
from .utils import LOGGER, get_save_paths, safe_write_csv, safe_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fintech Data Curator")
    parser.add_argument("--exchange", type=str, required=True, help="Exchange name (e.g., NASDAQ, PSX, Binance)")
    parser.add_argument("--symbol", type=str, required=True, help="Ticker or symbol (e.g., AAPL, BTC-USD)")
    parser.add_argument("--asset_type", type=str, choices=["stock", "crypto"], default="stock")
    parser.add_argument("--days", type=int, default=10, help="Number of days (5-30)")
    return parser.parse_args()


def run(exchange: str, symbol: str, days: int, asset_type: Literal["stock", "crypto"]) -> pd.DataFrame:
    prices = fetch_prices(exchange=exchange, symbol=symbol, days=days, asset_type=asset_type)
    prices_features = add_technical_indicators(prices)

    news = fetch_news(symbol=symbol, days=days, is_crypto=(asset_type == "crypto"))
    news_sent = add_sentiment(news)

    merged = merge_price_and_news(prices_features, news_sent)

    paths = get_save_paths(symbol)
    LOGGER.info("Saving CSV -> %s", paths.csv_path)
    safe_write_csv(paths.csv_path, merged)
    LOGGER.info("Saving JSON -> %s", paths.json_path)
    safe_write_json(paths.json_path, {"symbol": symbol, "exchange": exchange, "rows": merged.to_dict(orient="records")})

    return merged


def main() -> None:
    args = parse_args()
    run(exchange=args.exchange, symbol=args.symbol, days=args.days, asset_type=args.asset_type)  # noqa: F841


if __name__ == "__main__":
    main()
