from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def fit_minmax_scaler(values: np.ndarray) -> MinMaxScaler:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(values.reshape(-1, 1))
    return scaler


def transform_with_scaler(scaler: MinMaxScaler, values: np.ndarray) -> np.ndarray:
    return scaler.transform(values.reshape(-1, 1)).flatten()


def inverse_transform_with_scaler(scaler: MinMaxScaler, values: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(values.reshape(-1, 1)).flatten()


def create_sequences(series: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    if window < 1:
        raise ValueError("window must be >= 1")
    if len(series) <= window:
        # Not enough data to form a single (window -> target) pair
        return np.empty((0, window, 1)), np.empty((0,))

    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i : i + window])
        y.append(series[i + window])

    if not X:
        return np.empty((0, window, 1)), np.empty((0,))

    X_arr = np.array(X)
    y_arr = np.array(y)
    return X_arr.reshape((X_arr.shape[0], X_arr.shape[1], 1)), y_arr
