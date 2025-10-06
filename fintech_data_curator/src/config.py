from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Config:
    coinmarketcap_api_key: Optional[str]
    coinmarketcap_base_url: str = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical"


def load_config() -> Config:
    return Config(
        coinmarketcap_api_key=os.getenv("COINMARKETCAP_API_KEY"),
    )
