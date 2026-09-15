"""Backtest básico de la estrategia sobre datos históricos."""

import pandas as pd

from config import COMMISSION, INITIAL_CASH
from portfolio import Portfolio
from strategy import generate_signal


def run_backtest(data: pd.DataFrame) -> pd.DataFrame:
    data = generate_signal(data)
    portfolio = Portfolio(INITIAL_CASH)
    rows = []
    previous_signal = 0

    for date, row in data.iterrows():
        price = float(row["Close"])
        signal = int(row["signal"])

        if signal == 1 and previous_signal == 0:
            portfolio.buy_with_all_cash(price, COMMISSION)
        elif signal == 0 and previous_signal == 1:
            portfolio.sell_all(price, COMMISSION)

        rows.append(
            {
                "date": date,
                "price": price,
                "signal": signal,
                "cash": portfolio.cash,
                "shares": portfolio.shares,
                "portfolio_value": portfolio.value(price),
            }
        )
        previous_signal = signal

    return pd.DataFrame(rows).set_index("date")


def summarize(results: pd.DataFrame) -> dict:
    start = float(results["portfolio_value"].iloc[0])
    end = float(results["portfolio_value"].iloc[-1])
    peak = results["portfolio_value"].cummax()
    drawdown = (results["portfolio_value"] / peak) - 1

    return {
        "initial_value": start,
        "final_value": end,
        "return_pct": (end / start - 1) * 100,
        "max_drawdown_pct": float(drawdown.min() * 100),
    }
