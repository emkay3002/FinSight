from __future__ import annotations

from src.main import run


if __name__ == "__main__":
    # Stocks
    run(exchange="NASDAQ", symbol="AAPL", days=10, asset_type="stock")
    run(exchange="NASDAQ", symbol="MSFT", days=10, asset_type="stock")
    # Crypto
    run(exchange="Binance", symbol="BTC-USD", days=10, asset_type="crypto")
    print("Sample runs completed. Check the data/ directory for outputs.")
