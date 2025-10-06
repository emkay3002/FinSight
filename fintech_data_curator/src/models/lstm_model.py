from __future__ import annotations

from typing import Tuple

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def build_lstm_model(input_window: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(input_window, 1)),
        layers.LSTM(32),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def train_lstm(model: keras.Model, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 20, batch_size: int = 32) -> keras.Model:
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model


def forecast_lstm(model: keras.Model, last_window: np.ndarray, steps: int) -> np.ndarray:
    # Iterative forecasting: roll the window forward using predictions
    preds = []
    window = last_window.copy().reshape(1, -1, 1)
    for _ in range(steps):
        yhat = model.predict(window, verbose=0).flatten()[0]
        preds.append(yhat)
        # shift window and append new value
        new_window = np.append(window.flatten()[1:], yhat)
        window = new_window.reshape(1, -1, 1)
    return np.array(preds)
