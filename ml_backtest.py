"""Backtest de la estrategia basada en machine learning."""

import pandas as pd
from sklearn.metrics import accuracy_score

from config import COMMISSION, INITIAL_CASH
from ml_model import run_ml_backtest
from portfolio import Portfolio


def run_ml_portfolio_backtest(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ml_result = run_ml_backtest(data)
    test = ml_result.data
    portfolio = Portfolio(INITIAL_CASH)
    rows = []
    previous_signal = 0

    for date, row in test.iterrows():
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
                "probability_up": float(row["probability_up"]),
                "cash": portfolio.cash,
                "shares": portfolio.shares,
                "portfolio_value": portfolio.value(price),
            }
        )
        previous_signal = signal

    results = pd.DataFrame(rows).set_index("date")
    predictions = test["prediction"].astype(int)
    targets = test["target"].astype(int)
    accuracy = float(accuracy_score(targets, predictions) * 100)

    start = float(results["portfolio_value"].iloc[0])
    end = float(results["portfolio_value"].iloc[-1])
    peak = results["portfolio_value"].cummax()
    drawdown = results["portfolio_value"] / peak - 1

    stats = {
        "initial_value": start,
        "final_value": end,
        "return_pct": (end / start - 1) * 100,
        "max_drawdown_pct": float(drawdown.min() * 100),
        "accuracy_pct": accuracy,
        "test_start": ml_result.test_start,
        "test_days": len(test),
    }
    return results, stats
