from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def train_arima(series: pd.Series, order: Tuple[int, int, int] = (1, 1, 1)) -> ARIMA:
    model = ARIMA(series.astype(float), order=order)
    fitted = model.fit()
    return fitted


def forecast_arima(fitted: ARIMA, steps: int) -> np.ndarray:
    fc = fitted.forecast(steps=steps)
    return np.asarray(fc)
