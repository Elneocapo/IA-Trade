"""Evaluación robusta del modelo y comparación con buy & hold."""

from __future__ import annotations

import pandas as pd

from config import COMMISSION, INITIAL_CASH


def simulate_predictions(
    data: pd.DataFrame,
    predictions: pd.Series,
    initial_cash: float = INITIAL_CASH,
    commission: float = COMMISSION,
) -> pd.DataFrame:
    """Simula una estrategia long-only usando solo predicciones ya generadas."""
    result = data.loc[predictions.index].copy()
    result["prediction"] = predictions.astype(int)
    result["position"] = result["prediction"].shift(1).fillna(0).astype(int)

    cash = initial_cash
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


def buy_and_hold(data: pd.DataFrame, initial_cash: float = INITIAL_CASH) -> pd.Series:
    """Rentabilidad de comprar al inicio y mantener hasta el final."""
    first = float(data["Close"].iloc[0])
    last = float(data["Close"].iloc[-1])
    return pd.Series(
        initial_cash * (last / first),
        index=[data.index[-1]],
        name="buy_and_hold_value",
    )


def summarize_values(values: pd.Series) -> dict:
    start = float(values.iloc[0])
    end = float(values.iloc[-1])
    peak = values.cummax()
    drawdown = (values / peak) - 1
    return {
        "initial_value": start,
        "final_value": end,
        "return_pct": (end / start - 1) * 100,
        "max_drawdown_pct": float(drawdown.min() * 100),
    }


def count_trades(positions: pd.Series) -> int:
    return int((positions.diff().abs() > 0).sum())
