"""Validación walk-forward del modelo de machine learning."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from evaluation import (
    count_trades,
    simulate_fixed_horizon_strategy,
    simulate_probability_strategy,
)
from ml_model import FEATURES, build_model, prepare_ml_data


ENTRY_THRESHOLD = 0.58
EXIT_THRESHOLD = 0.48


def run_walk_forward(
    data: pd.DataFrame,
    train_size: int = 252,
    test_size: int = 63,
    entry_threshold: float = ENTRY_THRESHOLD,
    exit_threshold: float = EXIT_THRESHOLD,
    horizon_bars: int = 1,
) -> tuple[pd.DataFrame, dict]:
    """Entrena en bloques históricos y prueba siempre en datos posteriores.

    ``horizon_bars`` define tanto la etiqueta futura como, cuando es mayor que
    1, la duración de la posición. Así evitamos que el modelo prediga un
    horizonte corto mientras la estrategia mantiene una operación mucho más
    tiempo.
    """
    clean = prepare_ml_data(data, horizon_bars=horizon_bars)
    predictions = []
    feature_importances = []
    window_stats = []

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
        feature_importances.append(model.feature_importances_)

        predicted_direction = (probability >= 0.5).astype(int)
        targets = test["target"].astype(int)
        window_accuracy = float((predicted_direction == targets).mean() * 100)
        window_balanced_accuracy = float(
            balanced_accuracy_score(targets, predicted_direction) * 100
        )
        window_brier = float(((probability - targets) ** 2).mean())
        window_stats.append({
            "start": test.index[0],
            "end": test.index[-1],
            "accuracy_pct": window_accuracy,
            "balanced_accuracy_pct": window_balanced_accuracy,
            "brier_score": window_brier,
        })
        start += test_size

    if not predictions:
        raise ValueError(
            "No hay suficientes datos para walk-forward. "
            "Necesitamos más historial o bloques de prueba más pequeños."
        )

    probability_series = pd.concat(predictions).sort_index()
    if horizon_bars == 1:
        results = simulate_probability_strategy(
            clean,
            probability_series,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
        )
    else:
        results = simulate_fixed_horizon_strategy(
            clean,
            probability_series,
            horizon_bars=horizon_bars,
            entry_threshold=entry_threshold,
        )

    targets = clean.loc[probability_series.index, "target"].astype(int)
    predicted_direction = (probability_series >= 0.5).astype(int)
    accuracy = float((predicted_direction == targets).mean() * 100)
    balanced_accuracy = float(
        balanced_accuracy_score(targets, predicted_direction) * 100
    )
    brier_score = float(((probability_series - targets) ** 2).mean())
    target_positive_rate = float(targets.mean() * 100)
    trades = count_trades(results["position"])
    confidence = float(probability_series.sub(0.5).abs().mean() * 100)
    days_in_market = float(results["position"].mean() * 100)

    importances = pd.DataFrame(
        feature_importances, columns=FEATURES
    ).mean().sort_values(ascending=False)
    top_features = importances.head(5).to_dict()

    return results, {
        "accuracy_pct": accuracy,
        "balanced_accuracy_pct": balanced_accuracy,
        "target_positive_rate_pct": target_positive_rate,
        "brier_score": brier_score,
        "trades": trades,
        "test_start": results.index[0],
        "test_end": results.index[-1],
        "test_days": len(results),
        "windows": len(predictions),
        "entry_threshold": entry_threshold,
        "exit_threshold": exit_threshold,
        "horizon_bars": horizon_bars,
        "average_confidence_pct": confidence,
        "days_in_market_pct": days_in_market,
        "window_stats": window_stats,
        "top_features": top_features,
    }
