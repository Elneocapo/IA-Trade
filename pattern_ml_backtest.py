"""IA de patrones secuenciales para backtesting intradía.

La red recibe una secuencia de velas recientes en lugar de una sola fila de
indicadores. Todo el proceso es investigación histórica y simulación.
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
TRAIN_FRACTION = 0.70
ENTRY_PROBABILITY = 0.60
EXIT_PROBABILITY = 0.45


def build_sequences(data):
    """Crea ejemplos donde cada muestra contiene las últimas 32 velas."""
    close = data["Close"].astype(float)
    open_price = data["Open"].astype(float)
    high = data["High"].astype(float)
    low = data["Low"].astype(float)
    volume = data["Volume"].astype(float)

    log_return = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
    range_pct = (high - low) / close
    body_pct = (close - open_price) / close
    volume_change = np.log(volume / volume.shift(1)).replace([np.inf, -np.inf], np.nan)

    # Variables relativas para que la red aprenda forma/movimiento y no precio absoluto.
    frame = np.column_stack([
        log_return.to_numpy(),
        range_pct.to_numpy(),
        body_pct.to_numpy(),
        ((high - close) / close).to_numpy(),
        ((close - low) / close).to_numpy(),
        volume_change.to_numpy(),
    ])

    future_return = close.shift(-INTRADAY_HORIZON_BARS) / close - 1
    # Objetivo binario: subida suficientemente grande vs bajada suficientemente grande.
    rolling_vol = log_return.rolling(32).std()
    threshold = np.maximum(rolling_vol * 1.0, 0.0025)
    target = np.full(len(data), np.nan)
    target[future_return.to_numpy() > threshold.to_numpy()] = 1
    target[future_return.to_numpy() < -threshold.to_numpy()] = 0

    sequences = []
    targets = []
    indices = []
    for end in range(LOOKBACK_BARS - 1, len(data) - INTRADAY_HORIZON_BARS):
        window = frame[end - LOOKBACK_BARS + 1:end + 1]
        if not np.isfinite(window).all() or not np.isfinite(target[end]):
            continue
        # Normalización por secuencia: elimina escala de precio y deja la forma.
        sequences.append(window)
        targets.append(int(target[end]))
        indices.append(data.index[end])

    if not sequences:
        raise ValueError("No hay suficientes secuencias limpias para entrenar la IA.")

    X = np.asarray(sequences, dtype=float)
    y = np.asarray(targets, dtype=int)
    return X, y, indices


def train_network(X_train, y_train):
    """Entrena una red neuronal sobre secuencias aplanadas."""
    scaler = StandardScaler()
    X_flat = X_train.reshape(len(X_train), -1)
    X_scaled = scaler.fit_transform(X_flat)

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=64,
        learning_rate_init=0.001,
        max_iter=120,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=42,
        verbose=False,
    )
    model.fit(X_scaled, y_train)
    return model, scaler


def main() -> None:
    print("=== IA-Trade | IA DE PATRONES SECUENCIALES ===")
    print(f"Activo: {TICKER}")
    print(f"Datos: {INTRADAY_PERIOD} | {INTRADAY_INTERVAL}")
    print(f"Patrón observado: últimas {LOOKBACK_BARS} velas")
    print(f"Horizonte: {INTRADAY_HORIZON_BARS} velas")
    print("Modo: SIMULACIÓN / sin broker")
    print()

    data = download_market_data(TICKER, period=INTRADAY_PERIOD, interval=INTRADAY_INTERVAL)
    X, y, indices = build_sequences(data)

    split = int(len(X) * TRAIN_FRACTION)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    test_indices = indices[split:]

    print(f"Secuencias totales:      {len(X)}")
    print(f"Entrenamiento:           {len(X_train)}")
    print(f"Test fuera de muestra:   {len(X_test)}")
    print()
    print("Entrenando red neuronal por épocas...")

    model, scaler = train_network(X_train, y_train)
    print(f"Épocas realizadas:       {model.n_iter_}")
    print(f"Pérdida final:            {model.loss_:.5f}")

    X_test_scaled = scaler.transform(X_test.reshape(len(X_test), -1))
    probabilities = model.predict_proba(X_test_scaled)
    classes = {int(c): i for i, c in enumerate(model.classes_)}
    p_up = probabilities[:, classes[1]] if 1 in classes else np.zeros(len(X_test))
    predictions = (p_up >= 0.5).astype(int)
    balanced = balanced_accuracy_score(y_test, predictions) * 100

    # Simulación sencilla: entrada cuando la confianza es alta y salida cuando cae.
    close = data.loc[test_indices, "Close"].astype(float)
    entries = p_up >= ENTRY_PROBABILITY
    exits = p_up < EXIT_PROBABILITY

    cash = INITIAL_CASH
    position = 0.0
    equity = []
    trade_count = 0

    for price, enter, exit_ in zip(close.to_numpy(), entries, exits):
        if position == 0 and enter:
            position = (cash * (1 - COMMISSION)) / price
            cash = 0.0
            trade_count += 1
        elif position > 0 and exit_:
            cash = position * price * (1 - COMMISSION)
            position = 0.0
            trade_count += 1
        equity.append(cash if position == 0 else position * price)

    final_value = equity[-1] if equity else INITIAL_CASH
    buy_hold = INITIAL_CASH * (close.iloc[-1] / close.iloc[0])

    print()
    print("--- RESULTADO ---")
    print(f"Valor final IA:          {final_value:.2f} €")
    print(f"Rentabilidad IA:         {(final_value / INITIAL_CASH - 1) * 100:.2f} %")
    print(f"Buy & Hold final:        {buy_hold:.2f} €")
    print(f"Operaciones:             {trade_count}")
    print(f"Balanced accuracy OOS:   {balanced:.2f}%")

    if final_value > buy_hold:
        print("🟢 La IA supera a Buy & Hold en este test.")
    else:
        print("🔴 La IA no supera a Buy & Hold en este test.")

    print()
    print("--- ENTRENAMIENTO ---")
    print("Se muestra la evolución de la pérdida de la red durante el entrenamiento.")
    plt.figure(figsize=(9, 5))
    plt.plot(model.loss_curve_)
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.title("Entrenamiento de la IA de patrones")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
