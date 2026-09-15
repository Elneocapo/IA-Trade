"""Modelo de machine learning para investigación y backtesting.

Importante: el modelo solo se usa en simulación. La validación es cronológica
para evitar usar datos futuros.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# El modelo recibe movimientos y geometría de las velas en varias escalas,
# además de indicadores clásicos. Evitamos pasar 30 velas OHLCV completas una
# por una: eso añadía mucho ruido y provocaba fragmentación del DataFrame.
RETURN_WINDOWS = (1, 2, 4, 8, 16, 30)

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
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "close_position_in_range",
    "range_pct",
    "volatility_5",
    "volatility_10",
    "volume_ratio_5",
    "volume_ratio_10",
    "hour_sin",
    "hour_cos",
    "day_of_week",
]

# Exigimos que SUBE/BAJA represente un movimiento suficientemente grande.
# La prueba anterior con 0.5x volatilidad producía demasiados objetivos que
# apenas se diferenciaban de ruido. Esta versión es deliberadamente más
# estricta para comprobar si existe señal cuando el movimiento importa.
ACTIONABLE_MOVE_MULTIPLIER = 1.0
MIN_ACTIONABLE_MOVE = 0.003  # 0.3%


@dataclass
class MLBacktestResult:
    data: pd.DataFrame
    test_start: pd.Timestamp


def add_ml_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    close = result["Close"]
    open_price = result["Open"]
    high = result["High"]
    low = result["Low"]
    volume = result["Volume"]

    features = {}
    features["return_1d"] = close.pct_change()
    features["return_5d"] = close.pct_change(5)
    features["return_20d"] = close.pct_change(20)

    features["sma_10_ratio"] = close / close.rolling(10).mean() - 1
    features["sma_20_ratio"] = close / close.rolling(20).mean() - 1
    features["sma_50_ratio"] = close / close.rolling(50).mean() - 1

    features["volatility_20"] = features["return_1d"].rolling(20).std()
    features["range_14"] = ((high - low) / close).rolling(14).mean()
    features["volume_ratio_20"] = volume / volume.rolling(20).mean()

    delta = close.diff()
    gains = delta.clip(lower=0).rolling(14).mean()
    losses = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gains / losses.replace(0, np.nan)
    features["rsi_14"] = 100 - (100 / (1 + rs))

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    features["macd_diff"] = macd - macd_signal

    # Movimiento acumulado en distintas escalas.
    for window in RETURN_WINDOWS:
        features[f"return_{window}bars"] = close.pct_change(window)

    candle_range = (high - low).replace(0, np.nan)
    body = close - open_price
    features["body_pct"] = body / close
    features["upper_wick_pct"] = (high - pd.concat([open_price, close], axis=1).max(axis=1)) / close
    features["lower_wick_pct"] = (pd.concat([open_price, close], axis=1).min(axis=1) - low) / close
    features["close_position_in_range"] = (close - low) / candle_range
    features["range_pct"] = candle_range / close

    features["volatility_5"] = features["return_1d"].rolling(5).std()
    features["volatility_10"] = features["return_1d"].rolling(10).std()
    features["volume_ratio_5"] = volume / volume.rolling(5).mean()
    features["volume_ratio_10"] = volume / volume.rolling(10).mean()

    # Contexto temporal: útil sobre todo en 15m. En diario aporta poca señal,
    # pero no introduce información futura.
    index = result.index
    if isinstance(index, pd.DatetimeIndex):
        minutes = index.hour * 60 + index.minute
        phase = 2 * np.pi * minutes / (24 * 60)
        features["hour_sin"] = np.sin(phase)
        features["hour_cos"] = np.cos(phase)
        features["day_of_week"] = index.dayofweek / 4.0
    else:
        features["hour_sin"] = 0.0
        features["hour_cos"] = 1.0
        features["day_of_week"] = 0.0

    # Construimos todas las columnas de una vez para evitar el warning de
    # DataFrame altamente fragmentado que aparecía con 150 asignaciones.
    feature_frame = pd.DataFrame(features, index=result.index)
    result = pd.concat([result, feature_frame], axis=1)
    return result.replace([np.inf, -np.inf], np.nan)


def prepare_ml_data(data: pd.DataFrame, horizon_bars: int = 1) -> pd.DataFrame:
    """Prepara features y objetivo futuro de tres clases.

    1 = subida accionable, 0 = movimiento sin ventaja clara, -1 = bajada
    accionable. El umbral exige un movimiento proporcional a la volatilidad
    reciente para que el modelo no aprenda simplemente el ruido de mercado.
    """
    if horizon_bars < 1:
        raise ValueError("horizon_bars debe ser >= 1.")

    result = add_ml_features(data)
    future_return = result["Close"].shift(-horizon_bars) / result["Close"] - 1
    required_move = np.maximum(
        result["volatility_20"] * ACTIONABLE_MOVE_MULTIPLIER,
        MIN_ACTIONABLE_MOVE,
    )

    result["target"] = 0
    result.loc[future_return > required_move, "target"] = 1
    result.loc[future_return < -required_move, "target"] = -1
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
    prepared = prepare_ml_data(data, horizon_bars=horizon_bars)
    split = int(len(prepared) * train_fraction)

    if split < 100 or len(prepared) - split < 20:
        raise ValueError("No hay suficientes datos para entrenar y probar el modelo.")

    train = prepared.iloc[:split]
    test = prepared.iloc[split:].copy()

    model = build_model()
    model.fit(train[FEATURES], train["target"].astype(int))

    test["prediction"] = model.predict(test[FEATURES])
    probabilities = model.predict_proba(test[FEATURES])
    class_to_column = {int(cls): i for i, cls in enumerate(model.classes_)}
    test["probability_down"] = probabilities[:, class_to_column.get(-1, 0)] if -1 in class_to_column else 0.0
    test["probability_neutral"] = probabilities[:, class_to_column.get(0, 0)] if 0 in class_to_column else 0.0
    test["probability_up"] = probabilities[:, class_to_column.get(1, 0)] if 1 in class_to_column else 0.0
    test["signal"] = test["prediction"].astype(int)

    return MLBacktestResult(test, test.index[0])
