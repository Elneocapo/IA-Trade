import pandas as pd

from evaluation import count_trades, summarize_values


def test_summarize_values():
    values = pd.Series([50.0, 55.0, 52.0, 60.0])
    stats = summarize_values(values)
    assert stats["initial_value"] == 50.0
    assert stats["final_value"] == 60.0
    assert round(stats["return_pct"], 2) == 20.0
    assert round(stats["max_drawdown_pct"], 2) == round(-3 / 55 * 100, 2)


def test_count_trades():
    positions = pd.Series([0, 0, 1, 1, 0, 1, 1])
    assert count_trades(positions) == 3
