from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from pymongo import ASCENDING, MongoClient, UpdateOne
from pymongo.collection import Collection

from .config import load_config
from .utils import LOGGER


@dataclass
class MongoCollections:
    prices: Collection
    news: Collection
    datasets: Collection


def get_db() -> Optional[MongoCollections]:
    cfg = load_config()
    if not cfg.mongo_uri:
        LOGGER.info("MONGODB_URI not set; skipping MongoDB storage")
        return None
    client = MongoClient(cfg.mongo_uri, appname="fintech_data_curator")
    db = client[cfg.mongo_db]

    prices = db["prices"]
    news = db["news"]
    datasets = db["datasets"]

    # Indexes (idempotent)
    prices.create_index([("symbol", ASCENDING), ("date", ASCENDING)], name="symbol_date", unique=True)
    news.create_index([("symbol", ASCENDING), ("date", ASCENDING)], name="symbol_date")
    news.create_index([("url", ASCENDING)], name="url_unique", unique=True, sparse=True)
    datasets.create_index([("symbol", ASCENDING), ("date", ASCENDING)], name="symbol_date")

    return MongoCollections(prices=prices, news=news, datasets=datasets)


def upsert_prices(coll: Collection, rows: Iterable[Dict[str, Any]]) -> None:
    ops = []
    for r in rows:
        key = {"symbol": r["symbol"], "date": r["date"]}
        ops.append(UpdateOne(key, {"$set": r}, upsert=True))
    if ops:
        res = coll.bulk_write(ops, ordered=False)
        LOGGER.info("Mongo prices upserted: matched=%s upserted=%s", res.matched_count, getattr(res, "upserted_count", 0))


def upsert_news(coll: Collection, rows: Iterable[Dict[str, Any]]) -> None:
    ops = []
    for r in rows:
        key = {"url": r.get("url")} if r.get("url") else {"symbol": r["symbol"], "date": r["date"], "title": r["title"]}
        ops.append(UpdateOne(key, {"$setOnInsert": r}, upsert=True))
    if ops:
        res = coll.bulk_write(ops, ordered=False)
        LOGGER.info("Mongo news upserted: matched=%s upserted=%s", res.matched_count, getattr(res, "upserted_count", 0))


def upsert_dataset(coll: Collection, rows: Iterable[Dict[str, Any]]) -> None:
    ops = []
    for r in rows:
        key = {"symbol": r["symbol"], "date": r["date"]}
        ops.append(UpdateOne(key, {"$set": r}, upsert=True))
    if ops:
        res = coll.bulk_write(ops, ordered=False)
        LOGGER.info("Mongo dataset upserted: matched=%s upserted=%s", res.matched_count, getattr(res, "upserted_count", 0))
