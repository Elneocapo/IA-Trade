"""Experimento de señal condicionada por régimen de mercado.

Solo investigación/simulación: no ejecuta operaciones reales.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from ml_model import FEATURES, build_model, prepare_binary_ml_data


def _regime(frame: pd.DataFrame) -> pd.Series:
    """Clasifica el contexto usando únicamente información disponible en t."""
    trend_up = (frame["sma_20_ratio"] > 0) & (frame["sma_50_ratio"] > 0) & (frame["return_20d"] > 0)
    trend_down = (frame["sma_20_ratio"] < 0) & (frame["sma_50_ratio"] < 0) & (frame["return_20d"] < 0)
    result = pd.Series("LATERAL", index=frame.index, dtype="object")
    result.loc[trend_up] = "ALCISTA"
    result.loc[trend_down] = "BAJISTA"
    return result


def _metrics(y_true, y_pred) -> dict:
    return {
        "samples": int(len(y_true)),
        "accuracy_pct": float(accuracy_score(y_true, y_pred) * 100),
        "balanced_accuracy_pct": float(balanced_accuracy_score(y_true, y_pred) * 100),
    }


def run_regime_experiment(data: pd.DataFrame, horizon_bars: int = 1) -> dict:
    prepared = prepare_binary_ml_data(data, horizon_bars=horizon_bars)
    prepared = prepared.copy()
    prepared["regime"] = _regime(prepared)

    n = len(prepared)
    initial_train_end = int(n * 0.50)
    block_size = int(n * 0.10)
    final_test_start = n - block_size

    if initial_train_end < 100 or block_size < 15:
        raise ValueError("No hay suficientes datos para el experimento de regímenes.")

    folds = []
    for fold_number in range(4):
        val_start = initial_train_end + fold_number * block_size
        val_end = val_start + block_size
        train = prepared.iloc[:val_start]
        validation = prepared.iloc[val_start:val_end]
        if len(validation) == 0:
            continue

        model = build_model()
        model.fit(train[FEATURES], train["target"].astype(int))
        validation = validation.copy()
        validation["prediction"] = model.predict(validation[FEATURES])

        regime_rows = {}
        for regime in ("ALCISTA", "LATERAL", "BAJISTA"):
            subset = validation[validation["regime"] == regime]
            if subset.empty:
                continue
            y = subset["target"].astype(int)
            # Mayoritaria y momentum se calculan sobre exactamente las mismas muestras.
            majority_prediction = np.full(len(subset), int(train["target"].mean() >= 0.5))
            momentum_prediction = (subset["return_20d"] > 0).astype(int)
            regime_rows[regime] = {
                "model": _metrics(y, subset["prediction"]),
                "majority": _metrics(y, majority_prediction),
                "momentum": _metrics(y, momentum_prediction),
            }
        folds.append({"fold": fold_number + 1, "regimes": regime_rows})

    final_train = prepared.iloc[:final_test_start]
    final_test = prepared.iloc[final_test_start:].copy()
    final_model = build_model()
    final_model.fit(final_train[FEATURES], final_train["target"].astype(int))
    final_test["prediction"] = final_model.predict(final_test[FEATURES])

    final_rows = {}
    for regime in ("ALCISTA", "LATERAL", "BAJISTA"):
        subset = final_test[final_test["regime"] == regime]
        if subset.empty:
            continue
        y = subset["target"].astype(int)
        majority_prediction = np.full(len(subset), int(final_train["target"].mean() >= 0.5))
        momentum_prediction = (subset["return_20d"] > 0).astype(int)
        final_rows[regime] = {
            "model": _metrics(y, subset["prediction"]),
            "majority": _metrics(y, majority_prediction),
            "momentum": _metrics(y, momentum_prediction),
        }

    return {
        "horizon": horizon_bars,
        "folds": folds,
        "final_test": final_rows,
        "final_test_start": final_test.index[0],
        "final_test_end": final_test.index[-1],
        "final_test_samples": len(final_test),
    }
