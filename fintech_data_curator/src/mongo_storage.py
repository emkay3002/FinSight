from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
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


def _ensure_datetime(value: Any) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    return value


def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    if "date" in out:
        out["date"] = _ensure_datetime(out["date"])
    return out


def upsert_prices(coll: Collection, rows: Iterable[Dict[str, Any]]) -> None:
    ops = []
    for r in rows:
        nr = _normalize_row(r)
        key = {"symbol": nr["symbol"], "date": nr["date"]}
        ops.append(UpdateOne(key, {"$set": nr}, upsert=True))
    if ops:
        res = coll.bulk_write(ops, ordered=False)
        LOGGER.info("Mongo prices upserted: matched=%s upserted=%s", res.matched_count, getattr(res, "upserted_count", 0))


def upsert_news(coll: Collection, rows: Iterable[Dict[str, Any]]) -> None:
    ops = []
    for r in rows:
        nr = _normalize_row(r)
        key = {"url": nr.get("url")} if nr.get("url") else {"symbol": nr["symbol"], "date": nr["date"], "title": nr["title"]}
        ops.append(UpdateOne(key, {"$setOnInsert": nr}, upsert=True))
    if ops:
        res = coll.bulk_write(ops, ordered=False)
        LOGGER.info("Mongo news upserted: matched=%s upserted=%s", res.matched_count, getattr(res, "upserted_count", 0))


def upsert_dataset(coll: Collection, rows: Iterable[Dict[str, Any]]) -> None:
    ops = []
    for r in rows:
        nr = _normalize_row(r)
        key = {"symbol": nr["symbol"], "date": nr["date"]}
        ops.append(UpdateOne(key, {"$set": nr}, upsert=True))
    if ops:
        res = coll.bulk_write(ops, ordered=False)
        LOGGER.info("Mongo dataset upserted: matched=%s upserted=%s", res.matched_count, getattr(res, "upserted_count", 0))
