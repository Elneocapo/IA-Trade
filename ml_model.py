"""Modelo de machine learning para detectar movimientos potencialmente accionables.

Importante: el modelo solo se usa en investigación/backtesting. La validación
es cronológica para evitar usar datos futuros.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# Además de indicadores resumidos, damos al modelo una ventana de velas
# recientes. Así puede aprender patrones de corto plazo sin mirar el futuro.
# Los precios se normalizan respecto al cierre actual para que el modelo aprenda
# la forma/movimiento de las velas y no el precio absoluto de AAPL.
RAW_LAG_BARS = 30

FEATURES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "sma_10_ratio",
    "sma_20_ratio",
    "sma_50_ratio",
    "volatility_20",
    "range_14",
    "volume_ratio_20",
    "rsi_14",
    "macd_diff",
]

# Umbral mínimo para considerar que el movimiento futuro tiene suficiente
# magnitud para ser interesante después de costes simulados.
ACTIONABLE_MOVE_MULTIPLIER = 0.5
MIN_ACTIONABLE_MOVE = 0.002  # 0.2%


@dataclass
class MLBacktestResult:
    data: pd.DataFrame
    test_start: pd.Timestamp


def add_ml_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    close = result["Close"]
    high = result["High"]
    low = result["Low"]
    volume = result["Volume"]

    result["return_1d"] = close.pct_change()
    result["return_5d"] = close.pct_change(5)
    result["return_20d"] = close.pct_change(20)

    result["sma_10_ratio"] = close / close.rolling(10).mean() - 1
    result["sma_20_ratio"] = close / close.rolling(20).mean() - 1
    result["sma_50_ratio"] = close / close.rolling(50).mean() - 1

    result["volatility_20"] = result["return_1d"].rolling(20).std()
    result["range_14"] = ((high - low) / close).rolling(14).mean()
    result["volume_ratio_20"] = volume / volume.rolling(20).mean()

    delta = close.diff()
    gains = delta.clip(lower=0).rolling(14).mean()
    losses = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gains / losses.replace(0, np.nan)
    result["rsi_14"] = 100 - (100 / (1 + rs))

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    result["macd_diff"] = macd - macd_signal

    # Ventana de las últimas 30 velas. Incluimos OHLC relativo al cierre de
    # cada momento y volumen relativo, de forma que el modelo vea más contexto
    # que un puñado de indicadores agregados.
    open_price = result["Open"]
    for lag in range(1, RAW_LAG_BARS + 1):
        lag_close = close.shift(lag)
        result[f"lag_{lag}_open"] = open_price.shift(lag) / lag_close - 1
        result[f"lag_{lag}_high"] = high.shift(lag) / lag_close - 1
        result[f"lag_{lag}_low"] = low.shift(lag) / lag_close - 1
        result[f"lag_{lag}_close"] = lag_close / close - 1
        result[f"lag_{lag}_volume"] = volume.shift(lag) / volume.rolling(20).mean()

    return result.replace([np.inf, -np.inf], np.nan)


def prepare_ml_data(data: pd.DataFrame, horizon_bars: int = 1) -> pd.DataFrame:
    """Prepara features y una etiqueta para un horizonte futuro fijo.

    El horizonte forma parte de la definición del problema: si es 4 en 15m,
    la etiqueta pregunta si dentro de las próximas 4 velas se obtiene un
    movimiento alcista suficientemente grande. Las features siguen usando
    exclusivamente información disponible en la barra actual.
    """
    if horizon_bars < 1:
        raise ValueError("horizon_bars debe ser >= 1.")

    result = add_ml_features(data)

    future_return = (
        result["Close"].shift(-horizon_bars) / result["Close"] - 1
    )
    required_move = np.maximum(
        result["volatility_20"] * ACTIONABLE_MOVE_MULTIPLIER,
        MIN_ACTIONABLE_MOVE,
    )
    result["target"] = (future_return > required_move).astype(float)
    result.loc[result.index[-horizon_bars:], "target"] = np.nan

    return result.dropna(subset=FEATURES + ["target"]).copy()


def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
        class_weight="balanced",
    )


def run_ml_backtest(
    data: pd.DataFrame,
    train_fraction: float = 0.7,
    horizon_bars: int = 1,
) -> MLBacktestResult:
    """Entrena con el tramo inicial y genera probabilidades en el tramo final."""
    prepared = prepare_ml_data(data, horizon_bars=horizon_bars)
    split = int(len(prepared) * train_fraction)

    if split < 100 or len(prepared) - split < 20:
        raise ValueError("No hay suficientes datos para entrenar y probar el modelo.")

    train = prepared.iloc[:split]
    test = prepared.iloc[split:].copy()

    model = build_model()
    model.fit(train[FEATURES], train["target"].astype(int))

    test["prediction"] = model.predict(test[FEATURES])
    test["probability_up"] = model.predict_proba(test[FEATURES])[:, 1]
    test["signal"] = test["prediction"].astype(int)

    return MLBacktestResult(test, test.index[0])
