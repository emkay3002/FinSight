# fintech_data_curator

A modular Python project to fetch and curate structured (prices) and unstructured (news) financial data for stocks and cryptocurrencies, compute simple indicators, analyze sentiment, and export merged datasets.

## Features
- Fetch daily OHLCV for past 5–30 days
- Compute returns, volatility, 5/10-day moving averages
- Scrape related headlines via RSS/newspaper
- Sentiment analysis using TextBlob
- Merge prices and news by date; save CSV and JSON

## Tech
- Python 3.10+
- pandas, yfinance, requests, BeautifulSoup4, feedparser, newspaper3k, TextBlob

## Install
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m textblob.download_corpora  # one-time for TextBlob
```

## Usage
Run the CLI entry point:
```bash
python -m src.main --exchange NASDAQ --symbol AAPL --days 10
```
Outputs are saved under `data/` as `<SYMBOL>_dataset.csv` and `<SYMBOL>_dataset.json`.

## Sample runs
```bash
python sample_runs.py
```
This will generate outputs for AAPL, MSFT, and BTC-USD.

## Environment variables (optional)
- `COINMARKETCAP_API_KEY` to use CoinMarketCap for crypto historical prices.

## Project structure
```
fintech_data_curator/
  data/
  src/
    __init__.py
    main.py
    config.py
    data_fetcher.py
    news_scraper.py
    feature_engineering.py
    sentiment_analysis.py
    data_merger.py
    utils.py
  sample_runs.py
  requirements.txt
  README.md
  .gitignore
```
