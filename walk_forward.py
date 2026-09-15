"""Validación walk-forward del modelo de machine learning."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from evaluation import count_trades, simulate_predictions
from ml_model import FEATURES, prepare_ml_data


def run_walk_forward(
    data: pd.DataFrame,
    train_size: int = 252,
    test_size: int = 63,
) -> tuple[pd.DataFrame, dict]:
    """Entrena en bloques históricos y prueba siempre en datos posteriores.

    Cada bloque de prueba es completamente posterior a su entrenamiento.
    No se reutilizan observaciones futuras para entrenar predicciones pasadas.
    """
    clean = prepare_ml_data(data)
    predictions = []

    start = 0
    while start + train_size + test_size <= len(clean):
        train = clean.iloc[start : start + train_size]
        test = clean.iloc[start + train_size : start + train_size + test_size]

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=8,
            random_state=42,
            class_weight="balanced",
        )
        model.fit(train[FEATURES], train["target"].astype(int))

        block_prediction = pd.Series(
            model.predict(test[FEATURES]),
            index=test.index,
            name="prediction",
        )
        predictions.append(block_prediction)
        start += test_size

    if not predictions:
        raise ValueError(
            "No hay suficientes datos para walk-forward. "
            "Necesitamos más historial o bloques de prueba más pequeños."
        )

    prediction_series = pd.concat(predictions).sort_index()
    results = simulate_predictions(clean, prediction_series)

    targets = clean.loc[prediction_series.index, "target"].astype(int)
    accuracy = float((prediction_series == targets).mean() * 100)
    trades = count_trades(results["position"])

    return results, {
        "accuracy_pct": accuracy,
        "trades": trades,
        "test_start": results.index[0],
        "test_end": results.index[-1],
        "test_days": len(results),
        "windows": len(predictions),
    }
