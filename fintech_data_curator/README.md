# fintech_data_curator

A modular Python project to fetch and curate structured (prices) and unstructured (news) financial data for stocks and cryptocurrencies, compute simple indicators, analyze sentiment, and export merged datasets.

## Features
- Fetch daily OHLCV for past 5–30 days
- Compute returns, volatility, 5/10-day moving averages
- Scrape related headlines via RSS/newspaper
- Sentiment analysis using TextBlob
- Merge prices and news by date; save CSV and JSON
- Optional MongoDB saving with normalized collections and indexes

## Tech
- Python 3.10+
- pandas, yfinance, requests, BeautifulSoup4, feedparser, newspaper3k, TextBlob, pymongo, python-dotenv

## Install
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m textblob.download_corpora  # one-time for TextBlob
```

## Environment
Create a `.env` file (ignored by git) in the project root, based on the following keys:
```
MONGODB_URI=
MONGODB_DB=fintech_data_curator
COINMARKETCAP_API_KEY=
```

## Usage
Run the CLI entry point:
```bash
python -m src.main --exchange NASDAQ --symbol AAPL --days 10 --asset_type stock --mongo
```
Outputs are saved under `data/` as `<SYMBOL>_dataset.csv` and `<SYMBOL>_dataset.json` and, if configured, in MongoDB.

## Sample runs
```bash
python sample_runs.py
```
This will generate outputs for AAPL, MSFT, and BTC-USD.

## MongoDB schema
- `prices`: unique by `symbol`+`date` (index)
- `news`: unique by `url` (sparse unique), fallback key `symbol`+`date`+`title`
- `datasets`: merged view unique by `symbol`+`date` (index)

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
    mongo_storage.py
    utils.py
  sample_runs.py
  requirements.txt
  README.md
  .gitignore
```
