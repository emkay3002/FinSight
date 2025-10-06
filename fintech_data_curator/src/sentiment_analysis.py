from __future__ import annotations

from typing import Iterable

import pandas as pd
from textblob import TextBlob


def _polarity(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    return float(TextBlob(text).sentiment.polarity)


def add_sentiment(df_news: pd.DataFrame) -> pd.DataFrame:
    df = df_news.copy()
    if df.empty:
        df["sentiment"] = []
        df["sentiment_label"] = []
        return df
    df["sentiment"] = (
        df[["title", "summary"]]
        .fillna("")
        .apply(lambda r: _polarity((r["title"] + ". " + r["summary"]).strip()), axis=1)
    )
    df["sentiment_label"] = pd.cut(
        df["sentiment"], bins=[-1.0, -0.05, 0.05, 1.0], labels=["negative", "neutral", "positive"], include_lowest=True
    )
    return df
