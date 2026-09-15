"""Evaluación walk-forward sencilla para reducir el riesgo de sobreajuste."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from evaluation import simulate_predictions
from strategy import FEATURE_COLUMNS


def run_walk_forward(
    data: pd.DataFrame,
    train_size: int = 252,
    test_size: int = 63,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entrena por bloques y prueba siempre en datos posteriores al entrenamiento."""
    clean = data.dropna(subset=FEATURE_COLUMNS + ["target"]).copy()
    all_predictions = []

    start = 0
    while start + train_size + test_size <= len(clean):
        train = clean.iloc[start : start + train_size]
        test = clean.iloc[start + train_size : start + train_size + test_size]

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=5,
            random_state=42,
            class_weight="balanced",
        )
        model.fit(train[FEATURE_COLUMNS], train["target"])
        prediction = pd.Series(
            model.predict(test[FEATURE_COLUMNS]),
            index=test.index,
            name="prediction",
        )
        all_predictions.append(prediction)
        start += test_size

    if not all_predictions:
        raise RuntimeError("No hay suficientes datos para ejecutar walk-forward.")

    predictions = pd.concat(all_predictions).sort_index()
    results = simulate_predictions(clean, predictions)
    accuracy = float((predictions == clean.loc[predictions.index, "target"]).mean() * 100)
    return results, pd.DataFrame({"target": clean.loc[predictions.index, "target"], "prediction": predictions, "correct": predictions == clean.loc[predictions.index, "target"]})
