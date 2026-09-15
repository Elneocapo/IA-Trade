"""Estrategia base sencilla para investigar el sistema.

No es una recomendación de inversión. Es una estrategia de cruce de medias
móviles para validar que el pipeline funciona antes de introducir modelos ML.
"""

import pandas as pd


SHORT_WINDOW = 20
LONG_WINDOW = 50


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["sma_short"] = result["Close"].rolling(SHORT_WINDOW).mean()
    result["sma_long"] = result["Close"].rolling(LONG_WINDOW).mean()
    return result


def generate_signal(data: pd.DataFrame) -> pd.DataFrame:
    result = add_indicators(data)
    result["signal"] = 0
    result.loc[result["sma_short"] > result["sma_long"], "signal"] = 1
    return result
