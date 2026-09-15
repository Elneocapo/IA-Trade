"""Validación walk-forward del modelo de machine learning."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, log_loss

from evaluation import (
    count_trades,
    simulate_fixed_horizon_strategy,
    simulate_probability_strategy,
)
from ml_model import FEATURES, build_model, prepare_binary_ml_data, prepare_ml_data


ENTRY_THRESHOLD = 0.58
EXIT_THRESHOLD = 0.48


def _probability_columns(model, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve P(down), P(neutral), P(up) aunque una clase falte en una ventana."""
    columns = {int(cls): i for i, cls in enumerate(model.classes_)}
    zero = np.zeros(len(probabilities))
    p_down = probabilities[:, columns[-1]] if -1 in columns else zero
    p_neutral = probabilities[:, columns[0]] if 0 in columns else zero
    p_up = probabilities[:, columns[1]] if 1 in columns else zero
    return p_down, p_neutral, p_up


def _multiclass_brier(
    targets: pd.Series,
    p_down: pd.Series,
    p_neutral: pd.Series,
    p_up: pd.Series,
) -> float:
    actual_down = (targets == -1).astype(float)
    actual_neutral = (targets == 0).astype(float)
    actual_up = (targets == 1).astype(float)
    return float(
        (
            (p_down - actual_down) ** 2
            + (p_neutral - actual_neutral) ** 2
            + (p_up - actual_up) ** 2
        ).mean()
    )


def _confidence_bins(predictions: pd.DataFrame, data: pd.DataFrame, horizon_bars: int) -> list[dict]:
    """Mide si una P(subida) alta realmente implica mejor resultado futuro."""
    future_return = data["Close"].shift(-horizon_bars) / data["Close"] - 1
    frame = predictions.copy()
    frame["future_return"] = future_return.reindex(frame.index)
    frame = frame.dropna(subset=["future_return"])

    bins = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01]
    labels = ["<40%", "40-50%", "50-60%", "60-70%", "70-80%", ">=80%"]
    frame["confidence_bin"] = pd.cut(
        frame["probability_up"], bins=bins, labels=labels, right=False
    )

    result = []
    for label in labels:
        group = frame[frame["confidence_bin"] == label]
        if group.empty:
            continue
        result.append(
            {
                "bin": label,
                "samples": int(len(group)),
                "mean_probability_pct": float(group["probability_up"].mean() * 100),
                "up_rate_pct": float((group["future_return"] > 0).mean() * 100),
                "mean_future_return_pct": float(group["future_return"].mean() * 100),
                "median_future_return_pct": float(group["future_return"].median() * 100),
            }
        )
    return result


def run_directional_walk_forward(
    data: pd.DataFrame,
    train_size: int = 252,
    test_size: int = 63,
    horizon_bars: int = 1,
) -> dict:
    """Diagnóstico binario: solo SUBE vs BAJA en movimientos accionables.

    Los NEUTRO se eliminan antes de dividir en bloques. La evaluación sigue
    siendo estrictamente cronológica y no modifica la estrategia principal.
    """
    clean = prepare_binary_ml_data(data, horizon_bars=horizon_bars)
    predictions = []
    window_stats = []

    start = 0
    while start + train_size + test_size <= len(clean):
        train = clean.iloc[start : start + train_size]
        test = clean.iloc[start + train_size : start + train_size + test_size]

        model = build_model()
        model.fit(train[FEATURES], train["target"].astype(int))
        predicted = model.predict(test[FEATURES])
        probabilities = model.predict_proba(test[FEATURES])
        classes = {int(cls): i for i, cls in enumerate(model.classes_)}
        p_up = probabilities[:, classes[1]] if 1 in classes else np.zeros(len(test))

        predictions.append(
            pd.DataFrame(
                {"target": test["target"].astype(int), "prediction": predicted, "probability_up": p_up},
                index=test.index,
            )
        )
        window_stats.append(
            float(balanced_accuracy_score(test["target"].astype(int), predicted) * 100)
        )
        start += test_size

    if not predictions:
        raise ValueError(
            "No hay suficientes movimientos accionables para el diagnóstico binario."
        )

    frame = pd.concat(predictions).sort_index()
    accuracy = float((frame["prediction"] == frame["target"]).mean() * 100)
    balanced_accuracy = float(
        balanced_accuracy_score(frame["target"], frame["prediction"]) * 100
    )
    up_rate = float((frame["target"] == 1).mean() * 100)
    baseline_accuracy = max(up_rate, 100 - up_rate)
    brier = float(np.mean((frame["probability_up"] - frame["target"]) ** 2))
    binary_log_loss = float(
        log_loss(frame["target"], frame["probability_up"], labels=[0, 1])
    )

    return {
        "samples": len(frame),
        "windows": len(predictions),
        "accuracy_pct": accuracy,
        "balanced_accuracy_pct": balanced_accuracy,
        "baseline_accuracy_pct": baseline_accuracy,
        "target_up_rate_pct": up_rate,
        "brier_score": brier,
        "log_loss": binary_log_loss,
        "mean_probability_up_pct": float(frame["probability_up"].mean() * 100),
        "window_balanced_accuracy_pct": window_stats,
    }


def run_walk_forward(
    data: pd.DataFrame,
    train_size: int = 252,
    test_size: int = 63,
    entry_threshold: float = ENTRY_THRESHOLD,
    exit_threshold: float = EXIT_THRESHOLD,
    horizon_bars: int = 1,
) -> tuple[pd.DataFrame, dict]:
    """Entrena en bloques históricos y prueba siempre en datos posteriores."""
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

        probabilities = model.predict_proba(test[FEATURES])
        p_down, p_neutral, p_up = _probability_columns(model, probabilities)

        window_predictions = pd.DataFrame(
            {
                "probability_down": p_down,
                "probability_neutral": p_neutral,
                "probability_up": p_up,
            },
            index=test.index,
        )
        predictions.append(window_predictions)
        feature_importances.append(model.feature_importances_)

        predicted_direction = model.predict(test[FEATURES])
        targets = test["target"].astype(int)
        window_accuracy = float((predicted_direction == targets).mean() * 100)
        window_balanced_accuracy = float(
            balanced_accuracy_score(targets, predicted_direction) * 100
        )
        window_brier = _multiclass_brier(
            targets,
            pd.Series(p_down, index=test.index),
            pd.Series(p_neutral, index=test.index),
            pd.Series(p_up, index=test.index),
        )
        window_stats.append(
            {
                "start": test.index[0],
                "end": test.index[-1],
                "accuracy_pct": window_accuracy,
                "balanced_accuracy_pct": window_balanced_accuracy,
                "brier_score": window_brier,
            }
        )
        start += test_size

    if not predictions:
        raise ValueError(
            "No hay suficientes datos para walk-forward. "
            "Necesitamos más historial o bloques de prueba más pequeños."
        )

    prediction_frame = pd.concat(predictions).sort_index()
    probability_up = prediction_frame["probability_up"]

    if horizon_bars == 1:
        results = simulate_probability_strategy(
            clean,
            probability_up,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
        )
    else:
        results = simulate_fixed_horizon_strategy(
            clean,
            probability_up,
            horizon_bars=horizon_bars,
            entry_threshold=entry_threshold,
        )

    targets = clean.loc[prediction_frame.index, "target"].astype(int)
    predicted_direction = prediction_frame[["probability_down", "probability_neutral", "probability_up"]].idxmax(axis=1)
    predicted_direction = predicted_direction.map(
        {
            "probability_down": -1,
            "probability_neutral": 0,
            "probability_up": 1,
        }
    )
    accuracy = float((predicted_direction == targets).mean() * 100)
    balanced_accuracy = float(
        balanced_accuracy_score(targets, predicted_direction) * 100
    )
    brier_score = _multiclass_brier(
        targets,
        prediction_frame["probability_down"],
        prediction_frame["probability_neutral"],
        prediction_frame["probability_up"],
    )
    target_up_rate = float((targets == 1).mean() * 100)
    target_down_rate = float((targets == -1).mean() * 100)
    target_neutral_rate = float((targets == 0).mean() * 100)
    trades = count_trades(results["position"])
    confidence = float(prediction_frame["probability_up"].mean() * 100)
    days_in_market = float(results["position"].mean() * 100)

    class_probabilities = prediction_frame[[
        "probability_down",
        "probability_neutral",
        "probability_up",
    ]].to_numpy()
    encoded_targets = targets.map({-1: 0, 0: 1, 1: 2}).to_numpy()
    multiclass_log_loss = float(
        log_loss(encoded_targets, class_probabilities, labels=[0, 1, 2])
    )

    importances = pd.DataFrame(feature_importances, columns=FEATURES).mean().sort_values(ascending=False)
    top_features = importances.head(8).to_dict()
    confidence_analysis = _confidence_bins(prediction_frame, clean, horizon_bars)

    return results, {
        "accuracy_pct": accuracy,
        "balanced_accuracy_pct": balanced_accuracy,
        "target_positive_rate_pct": target_up_rate,
        "target_up_rate_pct": target_up_rate,
        "target_down_rate_pct": target_down_rate,
        "target_neutral_rate_pct": target_neutral_rate,
        "brier_score": brier_score,
        "multiclass_log_loss": multiclass_log_loss,
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
        "confidence_analysis": confidence_analysis,
    }
