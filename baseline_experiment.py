"""Baselines científicos para comprobar si el modelo aporta señal real."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from ml_model import FEATURES, build_model, prepare_binary_ml_data


def _stats(target: pd.Series, prediction: np.ndarray) -> dict:
    target = target.astype(int)
    prediction = np.asarray(prediction, dtype=int)
    return {
        "accuracy_pct": float((prediction == target.to_numpy()).mean() * 100),
        "balanced_accuracy_pct": float(
            balanced_accuracy_score(target, prediction) * 100
        ),
    }


def _random_stats(target: pd.Series, rng: np.random.Generator, repeats: int = 500) -> dict:
    target_values = target.astype(int).to_numpy()
    accuracies = []
    balanced = []
    for _ in range(repeats):
        prediction = rng.integers(0, 2, size=len(target_values))
        accuracies.append((prediction == target_values).mean() * 100)
        balanced.append(balanced_accuracy_score(target_values, prediction) * 100)
    return {
        "accuracy_pct": float(np.mean(accuracies)),
        "balanced_accuracy_pct": float(np.mean(balanced)),
    }


def _evaluate_fold(train: pd.DataFrame, test: pd.DataFrame, rng: np.random.Generator) -> dict:
    target = test["target"].astype(int)

    majority_class = int(train["target"].astype(int).mode().iloc[0])
    majority_prediction = np.full(len(test), majority_class)

    momentum_prediction = (test["return_20d"] > 0).astype(int).to_numpy()

    model = build_model()
    model.fit(train[FEATURES], train["target"].astype(int))
    model_prediction = model.predict(test[FEATURES])

    return {
        "random": _random_stats(target, rng),
        "majority": _stats(target, majority_prediction),
        "momentum": _stats(target, momentum_prediction),
        "random_expected_accuracy_pct": 50.0,
        "random_expected_balanced_pct": 50.0,
        "model": _stats(target, model_prediction),
        "majority_class": majority_class,
        "samples": len(test),
    }


def run_baseline_experiment(
    data: pd.DataFrame,
    horizon_bars: int = 1,
    initial_train_pct: float = 0.50,
    validation_block_pct: float = 0.10,
    final_test_pct: float = 0.10,
) -> dict:
    """Compara azar, mayoría, momentum y Random Forest en los mismos bloques.

    La comparación es binaria (SUBE vs BAJA), eliminando NEUTRO para que todos
    los métodos compitan sobre exactamente las mismas muestras. El tiempo avanza
    siempre hacia delante: 50% entrenamiento inicial, cuatro bloques de
    validación del 10% y un 10% final intocable como test.
    """
    prepared = prepare_binary_ml_data(data, horizon_bars=horizon_bars)
    n = len(prepared)
    initial_train_end = int(n * initial_train_pct)
    block_size = max(30, int(n * validation_block_pct))
    final_test_size = max(30, int(n * final_test_pct))
    final_test_start = n - final_test_size

    if initial_train_end < 150 or final_test_start <= initial_train_end + block_size:
        raise ValueError("No hay suficientes datos para comparar los baselines.")

    rng = np.random.default_rng(42)
    validation_folds = []
    train_end = initial_train_end

    while train_end + block_size <= final_test_start:
        validation_start = train_end
        validation_end = min(train_end + block_size, final_test_start)
        train_end_purged = train_end - horizon_bars
        train = prepared.iloc[:train_end_purged]
        validation = prepared.iloc[validation_start:validation_end]

        if len(train) >= 100 and len(validation) >= 30:
            stats = _evaluate_fold(train, validation, rng)
            stats.update(
                {
                    "fold": len(validation_folds) + 1,
                    "validation_start": validation.index[0],
                    "validation_end": validation.index[-1],
                }
            )
            validation_folds.append(stats)

        train_end = validation_end

    final_train = prepared.iloc[: final_test_start - horizon_bars]
    final_test = prepared.iloc[final_test_start:]
    if len(final_train) < 100 or len(final_test) < 30 or not validation_folds:
        raise ValueError("No hay suficientes datos para el test final de baselines.")

    final_stats = _evaluate_fold(final_train, final_test, rng)

    methods = ("random", "majority", "momentum", "model")
    validation_means = {}
    for method in methods:
        validation_means[method] = {
            "accuracy_pct": float(
                np.mean([fold[method]["accuracy_pct"] for fold in validation_folds])
            ),
            "balanced_accuracy_pct": float(
                np.mean(
                    [fold[method]["balanced_accuracy_pct"] for fold in validation_folds]
                )
            ),
        }

    return {
        "horizon": horizon_bars,
        "samples": len(prepared),
        "validation_folds": validation_folds,
        "validation_means": validation_means,
        "final_test": final_stats,
        "final_test_start": final_test.index[0],
        "final_test_end": final_test.index[-1],
    }
