"""Modelo de machine learning para predecir la dirección del día siguiente.

Importante: el modelo solo se usa en investigación/backtesting. La división
entre entrenamiento y prueba es cronológica para evitar usar datos futuros.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


FEATURES = [
    "return_1d",
    "return_5d",
    "sma_10_ratio",
    "sma_20_ratio",
    "sma_50_ratio",
    "volatility_20",
    "volume_change",
    "rsi_14",
]


@dataclass
class MLBacktestResult:
    data: pd.DataFrame
    test_start: pd.Timestamp


def add_ml_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    close = result["Close"]

    result["return_1d"] = close.pct_change()
    result["return_5d"] = close.pct_change(5)
    result["sma_10_ratio"] = close / close.rolling(10).mean() - 1
    result["sma_20_ratio"] = close / close.rolling(20).mean() - 1
    result["sma_50_ratio"] = close / close.rolling(50).mean() - 1
    result["volatility_20"] = result["return_1d"].rolling(20).std()
    result["volume_change"] = result["Volume"].pct_change().replace([np.inf, -np.inf], np.nan)

    delta = close.diff()
    gains = delta.clip(lower=0).rolling(14).mean()
    losses = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gains / losses.replace(0, np.nan)
    result["rsi_14"] = 100 - (100 / (1 + rs))

    return result.replace([np.inf, -np.inf], np.nan)


def prepare_ml_data(data: pd.DataFrame) -> pd.DataFrame:
    result = add_ml_features(data)
    result["target"] = (result["Close"].shift(-1) > result["Close"]).astype(float)
    result.loc[result.index[-1], "target"] = np.nan
    return result.dropna(subset=FEATURES + ["target"]).copy()


def run_ml_backtest(data: pd.DataFrame, train_fraction: float = 0.7) -> MLBacktestResult:
    """Entrena con el tramo inicial y genera predicciones para el tramo final."""
    prepared = prepare_ml_data(data)
    split = int(len(prepared) * train_fraction)

    if split < 100 or len(prepared) - split < 20:
        raise ValueError("No hay suficientes datos para entrenar y probar el modelo.")

    train = prepared.iloc[:split]
    test = prepared.iloc[split:].copy()

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=8,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(train[FEATURES], train["target"].astype(int))

    test["prediction"] = model.predict(test[FEATURES])
    test["probability_up"] = model.predict_proba(test[FEATURES])[:, 1]
    test["signal"] = test["prediction"].astype(int)

    return MLBacktestResult(test, test.index[0])
