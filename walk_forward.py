"""Validación walk-forward del modelo de machine learning."""

from __future__ import annotations

import pandas as pd

from evaluation import count_trades, simulate_probability_strategy
from ml_model import FEATURES, build_model, prepare_ml_data


ENTRY_THRESHOLD = 0.58
EXIT_THRESHOLD = 0.48


def run_walk_forward(
    data: pd.DataFrame,
    train_size: int = 252,
    test_size: int = 63,
    entry_threshold: float = ENTRY_THRESHOLD,
    exit_threshold: float = EXIT_THRESHOLD,
) -> tuple[pd.DataFrame, dict]:
    """Entrena en bloques históricos y prueba siempre en datos posteriores."""
    clean = prepare_ml_data(data)
    predictions = []

    start = 0
    while start + train_size + test_size <= len(clean):
        train = clean.iloc[start : start + train_size]
        test = clean.iloc[start + train_size : start + train_size + test_size]

        model = build_model()
        model.fit(train[FEATURES], train["target"].astype(int))

        probability = pd.Series(
            model.predict_proba(test[FEATURES])[:, 1],
            index=test.index,
            name="probability_up",
        )
        predictions.append(probability)
        start += test_size

    if not predictions:
        raise ValueError(
            "No hay suficientes datos para walk-forward. "
            "Necesitamos más historial o bloques de prueba más pequeños."
        )

    probability_series = pd.concat(predictions).sort_index()
    results = simulate_probability_strategy(
        clean,
        probability_series,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
    )

    targets = clean.loc[probability_series.index, "target"].astype(int)
    predicted_direction = (probability_series >= 0.5).astype(int)
    accuracy = float((predicted_direction == targets).mean() * 100)
    trades = count_trades(results["position"])
    confidence = float(probability_series.sub(0.5).abs().mean() * 100)
    days_in_market = float(results["position"].mean() * 100)

    return results, {
        "accuracy_pct": accuracy,
        "trades": trades,
        "test_start": results.index[0],
        "test_end": results.index[-1],
        "test_days": len(results),
        "windows": len(predictions),
        "entry_threshold": entry_threshold,
        "exit_threshold": exit_threshold,
        "average_confidence_pct": confidence,
        "days_in_market_pct": days_in_market,
    }
