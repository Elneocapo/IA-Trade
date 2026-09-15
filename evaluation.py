"""Herramientas para evaluar estrategias sin usar datos futuros."""

from __future__ import annotations

import pandas as pd

from config import COMMISSION, INITIAL_CASH


def simulate_predictions(
    data: pd.DataFrame,
    predictions: pd.Series,
    initial_cash: float = INITIAL_CASH,
    commission: float = COMMISSION,
) -> pd.DataFrame:
    """Simula una estrategia long-only usando predicciones ya generadas."""
    result = data.loc[predictions.index].copy()
    result["prediction"] = predictions.astype(int)
    result["position"] = result["prediction"].shift(1).fillna(0).astype(int)

    cash = float(initial_cash)
    shares = 0.0
    rows = []

    for date, row in result.iterrows():
        price = float(row["Close"])
        target = int(row["position"])

        if target == 1 and shares == 0 and cash > 0:
            budget = cash / (1 + commission)
            shares = budget / price
            cash -= budget * (1 + commission)
        elif target == 0 and shares > 0:
            cash += shares * price * (1 - commission)
            shares = 0.0

        rows.append({
            "date": date,
            "price": price,
            "prediction": int(row["prediction"]),
            "position": target,
            "cash": cash,
            "shares": shares,
            "portfolio_value": cash + shares * price,
        })

    return pd.DataFrame(rows).set_index("date")


def simulate_probability_strategy(
    data: pd.DataFrame,
    probability_up: pd.Series,
    entry_threshold: float = 0.58,
    exit_threshold: float = 0.48,
    initial_cash: float = INITIAL_CASH,
    commission: float = COMMISSION,
) -> pd.DataFrame:
    """Simula una estrategia basada en confianza, con entrada y salida separadas.

    La probabilidad calculada al cierre de un día solo puede cambiar la posición
    al día siguiente. Los umbrales son fijos y no se optimizan sobre el test.
    """
    if not 0.5 < entry_threshold < 1.0:
        raise ValueError("entry_threshold debe estar entre 0.5 y 1.0.")
    if 0.0 < exit_threshold >= entry_threshold:
        raise ValueError("exit_threshold debe ser menor que entry_threshold.")

    result = data.loc[probability_up.index].copy()
    result["probability_up"] = probability_up.astype(float)

    desired = []
    in_position = False
    for probability in result["probability_up"]:
        if not in_position and probability >= entry_threshold:
            in_position = True
        elif in_position and probability < exit_threshold:
            in_position = False
        desired.append(int(in_position))

    result["position"] = pd.Series(desired, index=result.index).shift(1).fillna(0).astype(int)

    cash = float(initial_cash)
    shares = 0.0
    rows = []

    for date, row in result.iterrows():
        price = float(row["Close"])
        target = int(row["position"])

        if target == 1 and shares == 0 and cash > 0:
            budget = cash / (1 + commission)
            shares = budget / price
            cash -= budget * (1 + commission)
        elif target == 0 and shares > 0:
            cash += shares * price * (1 - commission)
            shares = 0.0

        rows.append({
            "date": date,
            "price": price,
            "probability_up": float(row["probability_up"]),
            "position": target,
            "cash": cash,
            "shares": shares,
            "portfolio_value": cash + shares * price,
        })

    return pd.DataFrame(rows).set_index("date")


def buy_and_hold_curve(data: pd.DataFrame, initial_cash: float = INITIAL_CASH) -> pd.Series:
    """Curva de patrimonio de comprar al principio y mantener."""
    first = float(data["Close"].iloc[0])
    return initial_cash * data["Close"] / first


def summarize_values(values: pd.Series) -> dict:
    """Calcula rentabilidad y máximo drawdown de una curva de patrimonio."""
    start = float(values.iloc[0])
    end = float(values.iloc[-1])
    peak = values.cummax()
    drawdown = values / peak - 1
    return {
        "initial_value": start,
        "final_value": end,
        "return_pct": (end / start - 1) * 100,
        "max_drawdown_pct": float(drawdown.min() * 100),
    }


def count_trades(positions: pd.Series) -> int:
    """Cuenta cambios de posición (entradas y salidas)."""
    return int((positions.diff().fillna(positions).abs() > 0).sum())
