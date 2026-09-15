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
    """Simula una estrategia long-only usando predicciones ya generadas.

    La predicción de un día solo puede convertirse en posición al día siguiente,
    evitando comprar con información que todavía no estaba disponible.
    """
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

        rows.append(
            {
                "date": date,
                "price": price,
                "prediction": int(row["prediction"]),
                "position": target,
                "cash": cash,
                "shares": shares,
                "portfolio_value": cash + shares * price,
            }
        )

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
