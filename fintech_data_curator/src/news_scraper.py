from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

import feedparser
import pandas as pd
from bs4 import BeautifulSoup

from .utils import LOGGER


YAHOO_FINANCE_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&lang=en-US"
REUTERS_RSS = "https://www.reuters.com/markets/companies/{symbol}"  # not pure RSS; scraped if needed
COINDESK_RSS = "https://www.coindesk.com/arc/outboundfeeds/rss/"


@dataclass(frozen=True)
class NewsItem:
    date: datetime
    title: str
    summary: str
    source: str
    url: str


def _parse_feed_entries(entries, source: str, symbol: str) -> List[NewsItem]:
    items: List[NewsItem] = []
    for e in entries:
        title = e.get("title", "").strip()
        summary = BeautifulSoup(e.get("summary", ""), "html.parser").get_text(" ").strip()
        link = e.get("link", "")
        published = e.get("published_parsed") or e.get("updated_parsed")
        if published:
            dt = datetime(*published[:6])
        else:
            dt = datetime.utcnow()
        items.append(NewsItem(date=dt, title=title, summary=summary, source=source, url=link))
    return items


def fetch_news(symbol: str, days: int, is_crypto: bool) -> pd.DataFrame:
    cutoff = datetime.utcnow() - timedelta(days=days)
    items: List[NewsItem] = []

    # Yahoo Finance for both stocks and crypto tickers (e.g., AAPL, BTC-USD)
    try:
        yahoo_url = YAHOO_FINANCE_RSS.format(symbol=symbol)
        parsed = feedparser.parse(yahoo_url)
        if parsed.entries:
            items.extend(_parse_feed_entries(parsed.entries, "Yahoo Finance", symbol))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Yahoo RSS fetch failed for %s: %s", symbol, exc)

    # CoinDesk RSS for crypto
    if is_crypto:
        try:
            parsed = feedparser.parse(COINDESK_RSS)
            if parsed.entries:
                # Filter entries containing the symbol or coin name heuristically
                sym_upper = symbol.split("-")[0].upper()
                rel = [e for e in parsed.entries if sym_upper in (e.get("title", "") + e.get("summary", "")).upper()]
                items.extend(_parse_feed_entries(rel, "CoinDesk", symbol))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("CoinDesk RSS fetch failed: %s", exc)

    # Filter by cutoff and to last N days
    items = [it for it in items if it.date >= cutoff]
    if not items:
        return pd.DataFrame(columns=["date", "title", "summary", "source", "url"]).astype({"date": "datetime64[ns]"})

    df = pd.DataFrame([{
        "date": it.date.date(),
        "title": it.title,
        "summary": it.summary,
        "source": it.source,
        "url": it.url,
    } for it in items])
    df = df.sort_values("date")
    return df.reset_index(drop=True)
