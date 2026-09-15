"""Validación walk-forward y experimentos de entrenamiento del modelo."""

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


def _evaluate_classifier(model, frame: pd.DataFrame) -> dict:
    """Evalúa un modelo ya entrenado sobre un bloque temporal separado."""
    targets = frame["target"].astype(int)
    predictions = model.predict(frame[FEATURES])
    probabilities = model.predict_proba(frame[FEATURES])
    p_down, p_neutral, p_up = _probability_columns(model, probabilities)
    probability_frame = np.column_stack([p_down, p_neutral, p_up])
    encoded_targets = targets.map({-1: 0, 0: 1, 1: 2}).to_numpy()
    brier = _multiclass_brier(
        targets,
        pd.Series(p_down, index=frame.index),
        pd.Series(p_neutral, index=frame.index),
        pd.Series(p_up, index=frame.index),
    )
    return {
        "accuracy_pct": float((predictions == targets).mean() * 100),
        "balanced_accuracy_pct": float(
            balanced_accuracy_score(targets, predictions) * 100
        ),
        "brier_score": brier,
        "log_loss": float(
            log_loss(encoded_targets, probability_frame, labels=[0, 1, 2])
        ),
        "samples": len(frame),
    }


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


def run_training_horizon_experiment(
    data: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 5, 10, 20),
) -> list[dict]:
    """Compara horizontes con entrenamiento, validación y test separados.

    La división es cronológica: 60% entrenamiento, 20% validación y 20% test.
    Se elimina un margen de `horizon_bars` al final del entrenamiento para que
    el objetivo futuro de esas muestras no pueda mirar dentro de validación.

    El test permanece completamente fuera del entrenamiento y validación. Este
    experimento no cambia la estrategia ni busca el horizonte que dé más dinero:
    primero comprueba dónde existe señal predictiva y cuánto generaliza el modelo.
    """
    results = []

    for horizon in horizons:
        prepared = prepare_ml_data(data, horizon_bars=horizon)
        if len(prepared) < 300:
            continue

        train_end = int(len(prepared) * 0.60)
        validation_end = int(len(prepared) * 0.80)
        train_end_purged = train_end - horizon

        train = prepared.iloc[:train_end_purged]
        validation = prepared.iloc[train_end:validation_end]
        test = prepared.iloc[validation_end:]

        if len(train) < 100 or len(validation) < 30 or len(test) < 30:
            continue

        model = build_model()
        model.fit(train[FEATURES], train["target"].astype(int))

        train_stats = _evaluate_classifier(model, train)
        validation_stats = _evaluate_classifier(model, validation)
        test_stats = _evaluate_classifier(model, test)

        results.append(
            {
                "horizon": horizon,
                "train": train_stats,
                "validation": validation_stats,
                "test": test_stats,
                "test_start": test.index[0],
                "test_end": test.index[-1],
            }
        )

    if not results:
        raise ValueError("No hay suficientes datos para el experimento de horizontes.")

    return results


def run_expanding_temporal_experiment(
    data: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    initial_train_pct: float = 0.50,
    validation_block_pct: float = 0.10,
    final_test_pct: float = 0.10,
) -> list[dict]:
    """Comprueba si el aprendizaje generaliza a través de varios periodos.

    El tiempo avanza siempre hacia delante. Para cada horizonte:
      1. Se empieza entrenando con el primer 50% del historial.
      2. Se valida sobre bloques consecutivos del 10%, ampliando el entrenamiento
         después de cada bloque.
      3. El último 10% queda como test final y no se usa para decidir nada.

    Cada entrenamiento purga `horizon` filas al final para evitar que el objetivo
    futuro de las últimas muestras invada el bloque siguiente. No se optimizan
    hiperparámetros con el test final.
    """
    if not 0 < initial_train_pct < 1:
        raise ValueError("initial_train_pct debe estar entre 0 y 1.")
    if not 0 < validation_block_pct < 1:
        raise ValueError("validation_block_pct debe estar entre 0 y 1.")
    if not 0 < final_test_pct < 1:
        raise ValueError("final_test_pct debe estar entre 0 y 1.")

    results = []

    for horizon in horizons:
        prepared = prepare_ml_data(data, horizon_bars=horizon)
        n = len(prepared)
        initial_train_end = int(n * initial_train_pct)
        block_size = max(30, int(n * validation_block_pct))
        final_test_size = max(30, int(n * final_test_pct))
        final_test_start = n - final_test_size

        if initial_train_end < 150 or final_test_start <= initial_train_end + block_size:
            continue

        validation_folds = []
        train_end = initial_train_end

        while train_end + block_size <= final_test_start:
            validation_start = train_end
            validation_end = min(train_end + block_size, final_test_start)
            train_end_purged = train_end - horizon
            train = prepared.iloc[:train_end_purged]
            validation = prepared.iloc[validation_start:validation_end]

            if len(train) >= 100 and len(validation) >= 30:
                model = build_model()
                model.fit(train[FEATURES], train["target"].astype(int))
                train_stats = _evaluate_classifier(model, train)
                validation_stats = _evaluate_classifier(model, validation)
                validation_folds.append(
                    {
                        "fold": len(validation_folds) + 1,
                        "train_start": train.index[0],
                        "train_end": train.index[-1],
                        "validation_start": validation.index[0],
                        "validation_end": validation.index[-1],
                        "train": train_stats,
                        "validation": validation_stats,
                    }
                )

            train_end = validation_end

        final_train = prepared.iloc[: final_test_start - horizon]
        final_test = prepared.iloc[final_test_start:]
        if len(final_train) < 100 or len(final_test) < 30 or not validation_folds:
            continue

        final_model = build_model()
        final_model.fit(final_train[FEATURES], final_train["target"].astype(int))
        final_train_stats = _evaluate_classifier(final_model, final_train)
        final_test_stats = _evaluate_classifier(final_model, final_test)

        validation_balanced = [
            fold["validation"]["balanced_accuracy_pct"] for fold in validation_folds
        ]
        validation_accuracy = [
            fold["validation"]["accuracy_pct"] for fold in validation_folds
        ]

        results.append(
            {
                "horizon": horizon,
                "folds": validation_folds,
                "validation_mean_accuracy_pct": float(np.mean(validation_accuracy)),
                "validation_mean_balanced_accuracy_pct": float(np.mean(validation_balanced)),
                "final_train": final_train_stats,
                "final_test": final_test_stats,
                "final_test_start": final_test.index[0],
                "final_test_end": final_test.index[-1],
            }
        )

    if not results:
        raise ValueError("No hay suficientes datos para el experimento temporal expansivo.")

    return results


def run_directional_walk_forward(
    data: pd.DataFrame,
    train_size: int = 252,
    test_size: int = 63,
    horizon_bars: int = 1,
) -> dict:
    """Diagnóstico binario: solo SUBE vs BAJA en movimientos accionables."""
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
