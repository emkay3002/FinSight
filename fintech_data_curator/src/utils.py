from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SavePaths:
    csv_path: Path
    json_path: Path


def get_save_paths(symbol: str) -> SavePaths:
    safe_symbol = symbol.replace("/", "-")
    csv_path = DATA_DIR / f"{safe_symbol}_dataset.csv"
    json_path = DATA_DIR / f"{safe_symbol}_dataset.json"
    return SavePaths(csv_path=csv_path, json_path=json_path)


def setup_logger(name: str = "fintech_data_curator") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def safe_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def safe_write_csv(path: Path, df) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


LOGGER = setup_logger()
