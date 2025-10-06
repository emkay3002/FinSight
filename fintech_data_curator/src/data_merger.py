from __future__ import annotations

from typing import List

import pandas as pd


def merge_price_and_news(price_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
    p = price_df.copy()
    n = news_df.copy()

    # Ensure date types
    p["date"] = pd.to_datetime(p["date"]).dt.date
    if not n.empty:
        n["date"] = pd.to_datetime(n["date"]).dt.date

    # Aggregate news by date
    if not n.empty:
        agg = n.groupby("date").agg(
            headlines=("title", list),
            summaries=("summary", list),
            sources=("source", lambda s: list(pd.unique(s))),
            urls=("url", list),
            avg_sentiment=("sentiment", "mean"),
            sentiment_labels=("sentiment_label", list),
            num_headlines=("title", "count"),
        ).reset_index()
    else:
        agg = pd.DataFrame(columns=[
            "date", "headlines", "summaries", "sources", "urls", "avg_sentiment", "sentiment_labels", "num_headlines"
        ])

    merged = p.merge(agg, on="date", how="left")

    # Fill defaults for days without news
    merged["headlines"] = merged["headlines"].apply(lambda x: x if isinstance(x, list) else [])
    merged["summaries"] = merged["summaries"].apply(lambda x: x if isinstance(x, list) else [])
    merged["sources"] = merged["sources"].apply(lambda x: x if isinstance(x, list) else [])
    merged["urls"] = merged["urls"].apply(lambda x: x if isinstance(x, list) else [])
    merged["avg_sentiment"] = merged["avg_sentiment"].fillna(0.0)
    merged["sentiment_labels"] = merged["sentiment_labels"].apply(lambda x: x if isinstance(x, list) else [])
    merged["num_headlines"] = merged["num_headlines"].fillna(0).astype(int)

    return merged.sort_values("date").reset_index(drop=True)
