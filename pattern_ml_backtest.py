"""IA de patrones secuenciales con validación walk-forward intradía.

La red recibe una secuencia de velas recientes y se evalúa siempre sobre
periodos posteriores que no ha visto durante su entrenamiento.
Todo el proceso es investigación histórica y simulación.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from config import (
    COMMISSION,
    INITIAL_CASH,
    INTRADAY_HORIZON_BARS,
    INTRADAY_INTERVAL,
    INTRADAY_PERIOD,
    TICKER,
)
from market import download_market_data

LOOKBACK_BARS = 32
ENTRY_PROBABILITY = 0.54
EXIT_PROBABILITY = 0.50
MIN_TRAIN_FRACTION = 0.45
WALK_FORWARD_SPLITS = 4


def build_sequences(data):
    """Crea ejemplos donde cada muestra contiene las últimas 32 velas."""
    close = data["Close"].astype(float)
    open_price = data["Open"].astype(float)
    high = data["High"].astype(float)
    low = data["Low"].astype(float)
    volume = data["Volume"].astype(float).replace(0, np.nan)

    log_return = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
    range_pct = (high - low) / close
    body_pct = (close - open_price) / close
    volume_change = np.log(volume / volume.shift(1)).replace([np.inf, -np.inf], np.nan)

    frame = np.column_stack([
        log_return.to_numpy(),
        range_pct.to_numpy(),
        body_pct.to_numpy(),
        ((high - close) / close).to_numpy(),
        ((close - low) / close).to_numpy(),
        volume_change.to_numpy(),
    ])

    future_return = close.shift(-INTRADAY_HORIZON_BARS) / close - 1
    rolling_vol = log_return.rolling(32, min_periods=16).std()
    threshold = np.maximum(rolling_vol.fillna(0.0).to_numpy(), 0.0025)

    future_values = future_return.to_numpy()
    target = np.full(len(data), np.nan)
    target[future_values > threshold] = 1
    target[future_values < -threshold] = 0

    sequences = []
    targets = []
    indices = []
    for end in range(LOOKBACK_BARS - 1, len(data) - INTRADAY_HORIZON_BARS):
        window = frame[end - LOOKBACK_BARS + 1:end + 1]
        if not np.isfinite(window).all() or not np.isfinite(target[end]):
            continue
        sequences.append(window)
        targets.append(int(target[end]))
        indices.append(data.index[end])

    if not sequences:
        raise ValueError("No hay suficientes secuencias limpias para entrenar la IA.")

    return np.asarray(sequences, dtype=float), np.asarray(targets, dtype=int), indices


def train_network(X_train, y_train):
    """Entrena la red y devuelve también su curva de pérdida."""
    scaler = StandardScaler()
    X_flat = X_train.reshape(len(X_train), -1)
    X_scaled = scaler.fit_transform(X_flat)

    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=0.002,
        batch_size=64,
        learning_rate_init=0.0005,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        tol=0.00005,
        random_state=42,
        verbose=True,
    )
    model.fit(X_scaled, y_train)
    return model, scaler


def predict_block(model, scaler, X_test):
    """Genera probabilidades para un bloque totalmente fuera de muestra."""
    X_test_scaled = scaler.transform(X_test.reshape(len(X_test), -1))
    probabilities = model.predict_proba(X_test_scaled)
    classes = {int(c): i for i, c in enumerate(model.classes_)}
    return probabilities[:, classes[1]] if 1 in classes else np.zeros(len(X_test))


def simulate_strategy(close, p_up):
    """Simula entradas/salidas frecuentes sin conexión a ningún broker."""
    cash = INITIAL_CASH
    position = 0.0
    equity = []
    trade_count = 0

    for price, probability in zip(close.to_numpy(), p_up):
        if position == 0 and probability >= ENTRY_PROBABILITY:
            position = (cash * (1 - COMMISSION)) / price
            cash = 0.0
            trade_count += 1
        elif position > 0 and probability <= EXIT_PROBABILITY:
            cash = position * price * (1 - COMMISSION)
            position = 0.0
            trade_count += 1
        equity.append(cash if position == 0 else position * price)

    if position > 0:
        cash = position * close.iloc[-1] * (1 - COMMISSION)
        trade_count += 1

    return cash, trade_count, equity


def make_walk_forward_splits(total_samples):
    """Crea bloques temporales: cada bloque se prueba después de entrenar."""
    first_test = int(total_samples * MIN_TRAIN_FRACTION)
    remaining = total_samples - first_test
    test_size = max(1, remaining // WALK_FORWARD_SPLITS)

    splits = []
    train_end = first_test
    for split_id in range(WALK_FORWARD_SPLITS):
        test_start = train_end
        test_end = total_samples if split_id == WALK_FORWARD_SPLITS - 1 else min(
            total_samples, test_start + test_size
        )
        if test_end <= test_start:
            break
        splits.append((split_id + 1, train_end, test_start, test_end))
        train_end = test_end

    return splits


def main() -> None:
    print("=== IA-Trade | IA DE PATRONES + WALK-FORWARD ===")
    print(f"Activo: {TICKER}")
    print(f"Datos: {INTRADAY_PERIOD} | {INTRADAY_INTERVAL}")
    print(f"Patrón observado: últimas {LOOKBACK_BARS} velas")
    print(f"Horizonte: {INTRADAY_HORIZON_BARS} velas")
    print("Arquitectura: 256 → 128 → 64 neuronas")
    print("Validación: 4 bloques temporales fuera de muestra")
    print("Modo: SIMULACIÓN / sin broker")
    print()

    data = download_market_data(
        TICKER,
        period=INTRADAY_PERIOD,
        interval=INTRADAY_INTERVAL,
    )
    X, y, indices = build_sequences(data)
    splits = make_walk_forward_splits(len(X))

    print(f"Secuencias totales:      {len(X)}")
    print(f"Entrenamiento inicial:   {splits[0][1]}")
    print(f"Bloques OOS:             {len(splits)}")
    print(f"Entrada IA:              >= {ENTRY_PROBABILITY:.2f}")
    print(f"Salida IA:               <= {EXIT_PROBABILITY:.2f}")
    print()
    print("Entrenando y evaluando por bloques temporales...")

    all_probabilities = []
    all_test_indices = []
    all_test_targets = []
    all_losses = []
    block_scores = []

    for split_id, train_end, test_start, test_end in splits:
        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]

        print()
        print(f"--- BLOQUE {split_id}/{len(splits)} ---")
        print(f"Entrena con:             {len(X_train)} secuencias")
        print(f"Prueba con:              {len(X_test)} secuencias nuevas")

        model, scaler = train_network(X_train, y_train)
        p_up = predict_block(model, scaler, X_test)
        predictions = (p_up >= 0.5).astype(int)
        score = balanced_accuracy_score(y_test, predictions) * 100
        block_scores.append(score)
        all_probabilities.extend(p_up.tolist())
        all_test_indices.extend(indices[test_start:test_end])
        all_test_targets.extend(y_test.tolist())
        all_losses.append(model.loss_curve_)

        print(f"Épocas realizadas:       {model.n_iter_}")
        print(f"Pérdida final:            {model.loss_:.5f}")
        print(f"Balanced accuracy OOS:   {score:.2f}%")

    probabilities = np.asarray(all_probabilities, dtype=float)
    y_test_all = np.asarray(all_test_targets, dtype=int)
    predictions_all = (probabilities >= 0.5).astype(int)
    balanced = balanced_accuracy_score(y_test_all, predictions_all) * 100

    close = data.loc[all_test_indices, "Close"].astype(float)
    final_value, trade_count, equity = simulate_strategy(close, probabilities)
    buy_hold = INITIAL_CASH * (close.iloc[-1] / close.iloc[0])

    print()
    print("=== RESULTADO FINAL WALK-FORWARD ===")
    print(f"Valor final IA:          {final_value:.2f} €")
    print(f"Rentabilidad IA:         {(final_value / INITIAL_CASH - 1) * 100:.2f} %")
    print(f"Buy & Hold final:        {buy_hold:.2f} €")
    print(f"Operaciones:             {trade_count}")
    print(f"Operaciones/día aprox.:  {trade_count / max(len(close) / 26, 1):.2f}")
    print(f"Balanced accuracy OOS:   {balanced:.2f}%")
    print(f"OOS por bloque:          {', '.join(f'{s:.1f}%' for s in block_scores)}")

    if final_value > buy_hold:
        print("🟢 La IA supera a Buy & Hold en este test.")
    else:
        print("🔴 La IA no supera a Buy & Hold en este test.")

    print()
    print("--- ENTRENAMIENTO ---")
    print("La gráfica muestra la pérdida de cada entrenamiento temporal.")
    plt.figure(figsize=(9, 5))
    for split_id, loss_curve in enumerate(all_losses, start=1):
        plt.plot(loss_curve, label=f"Bloque {split_id}")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.title("Entrenamiento walk-forward de la IA de patrones")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
