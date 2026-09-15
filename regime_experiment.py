"""Experimento de señal condicionada por régimen de mercado.

Solo investigación/simulación: no ejecuta operaciones reales.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from config import INTERVAL, PERIOD, TICKER
from market import download_market_data
from ml_model import FEATURES, build_model, prepare_binary_ml_data


REGIMES = ("ALCISTA", "LATERAL", "BAJISTA")


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
    prepared = prepare_binary_ml_data(data, horizon_bars=horizon_bars).copy()
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
        for regime in REGIMES:
            subset = validation[validation["regime"] == regime]
            if subset.empty:
                continue
            y = subset["target"].astype(int)
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
    for regime in REGIMES:
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


def _print(stats: dict) -> None:
    print("=== EXPERIMENTO DE REGÍMENES | ¿DÓNDE FUNCIONA LA IA? ===")
    print(f"Activo:                 {TICKER}")
    print(f"Horizonte:              {stats['horizon']} barra")
    print("Regímenes:              ALCISTA / LATERAL / BAJISTA")
    print("Regla de régimen:       solo información disponible antes del objetivo")
    print("Validación:             4 bloques temporales consecutivos")
    print("Test final:             último 10%, reservado")
    print()
    print("FOLD | RÉGIMEN   | N | RF balanced | Momentum | Mayoritaria")
    for fold in stats["folds"]:
        for regime in REGIMES:
            row = fold["regimes"].get(regime)
            if row is None:
                continue
            print(
                f"{fold['fold']:>4} | {regime:<9} | {row['model']['samples']:>2} | "
                f"{row['model']['balanced_accuracy_pct']:>11.2f}% | "
                f"{row['momentum']['balanced_accuracy_pct']:>8.2f}% | "
                f"{row['majority']['balanced_accuracy_pct']:>11.2f}%"
            )

    print()
    print(
        f"TEST FINAL: {stats['final_test_start'].date()} -> {stats['final_test_end'].date()} | "
        f"{stats['final_test_samples']} muestras"
    )
    print("RÉGIMEN   | N | RF balanced | Momentum | Mayoritaria")
    for regime in REGIMES:
        row = stats["final_test"].get(regime)
        if row is None:
            continue
        print(
            f"{regime:<9} | {row['model']['samples']:>2} | "
            f"{row['model']['balanced_accuracy_pct']:>11.2f}% | "
            f"{row['momentum']['balanced_accuracy_pct']:>8.2f}% | "
            f"{row['majority']['balanced_accuracy_pct']:>11.2f}%"
        )
    print()
    print("Lectura: buscamos un régimen donde RF supere a los baselines de forma repetida, no solo en un fold.")


if __name__ == "__main__":
    print("=== IA-TRADE | EXPERIMENTO DE REGÍMENES ===")
    print(f"Descargando {TICKER} ({PERIOD}, {INTERVAL})...")
    market_data = download_market_data(TICKER, PERIOD, INTERVAL)
    stats = run_regime_experiment(market_data, horizon_bars=1)
    _print(stats)
    print("Modo: SIMULACIÓN. No se ha enviado ninguna orden real.")
