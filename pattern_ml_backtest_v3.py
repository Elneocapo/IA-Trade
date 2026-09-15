"""IA de patrones intradía v3: tres estados (bajada, neutro, subida).

A diferencia de v2, no elimina las velas de movimiento pequeño. La red aprende
las tres situaciones y genera una probabilidad en cada vela OOS. Esto permite
que la estrategia decida también cuándo NO hacer nada.

Todo es investigación histórica y simulación, sin broker.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from config import COMMISSION, INITIAL_CASH, INTRADAY_HORIZON_BARS, INTRADAY_INTERVAL, INTRADAY_PERIOD, TICKER
from market import download_market_data

LOOKBACK_BARS = 32
MIN_MOVE_RETURN = 0.003
MIN_TRAIN_FRACTION = 0.45
WALK_FORWARD_SPLITS = 4
TRAIN_VALIDATION_FRACTION = 0.15
TRAIN_EPOCHS = 120
TRAIN_PATIENCE = 20

# La red debe exigir una ventaja clara sobre el estado contrario antes de entrar.
ENTRY_UP_PROBABILITY = 0.55
EXIT_DOWN_PROBABILITY = 0.50
MIN_PROBABILITY_EDGE = 0.08

# 17 características, todas calculadas usando solo presente/pasado.

def build_sequences(data):
    close = data["Close"].astype(float)
    open_price = data["Open"].astype(float)
    high = data["High"].astype(float)
    low = data["Low"].astype(float)
    volume = data["Volume"].astype(float).replace(0, np.nan)

    log_return = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
    range_pct = (high - low) / close
    body_pct = (close - open_price) / close
    upper_wick_pct = (high - np.maximum(open_price, close)) / close
    lower_wick_pct = (np.minimum(open_price, close) - low) / close
    volume_change = np.log(volume / volume.shift(1)).replace([np.inf, -np.inf], np.nan)

    ema_8 = close.ewm(span=8, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()
    distance_ema_8 = close / ema_8 - 1
    distance_ema_21 = close / ema_21 - 1
    ema_spread = ema_8 / ema_21 - 1

    momentum_4 = close / close.shift(4) - 1
    momentum_8 = close / close.shift(8) - 1
    volatility_8 = log_return.rolling(8, min_periods=8).std()
    volatility_16 = log_return.rolling(16, min_periods=16).std()
    volatility_32 = log_return.rolling(32, min_periods=16).std()

    relative_volume = volume / volume.rolling(16, min_periods=8).mean() - 1
    rolling_high_16 = high.rolling(16, min_periods=16).max()
    rolling_low_16 = low.rolling(16, min_periods=16).min()
    rolling_high_32 = high.rolling(32, min_periods=16).max()
    rolling_low_32 = low.rolling(32, min_periods=16).min()
    position_16 = (close - rolling_low_16) / (rolling_high_16 - rolling_low_16)
    position_32 = (close - rolling_low_32) / (rolling_high_32 - rolling_low_32)

    feature_series = [
        log_return, range_pct, body_pct, upper_wick_pct, lower_wick_pct,
        volume_change, distance_ema_8, distance_ema_21, ema_spread,
        momentum_4, momentum_8, volatility_8, volatility_16, volatility_32,
        relative_volume, position_16, position_32,
    ]
    frame = np.column_stack([series.to_numpy() for series in feature_series])

    future_return = close.shift(-INTRADAY_HORIZON_BARS) / close - 1
    future_values = future_return.to_numpy()

    sequences, targets, indices = [], [], []
    raw_returns, thresholds = [], []

    for end in range(LOOKBACK_BARS - 1, len(data) - INTRADAY_HORIZON_BARS):
        window = frame[end - LOOKBACK_BARS + 1:end + 1]
        target_return = future_values[end]
        volatility = volatility_32.iloc[end]
        if not np.isfinite(window).all() or not np.isfinite(target_return) or not np.isfinite(volatility):
            continue

        threshold = max(MIN_MOVE_RETURN, float(volatility))
        if target_return >= threshold:
            label = 2  # subida fuerte
        elif target_return <= -threshold:
            label = 0  # bajada fuerte
        else:
            label = 1  # movimiento neutro

        sequences.append(window)
        targets.append(label)
        indices.append(data.index[end])
        raw_returns.append(float(target_return))
        thresholds.append(float(threshold))

    if not sequences:
        raise ValueError("No hay suficientes datos para construir secuencias.")

    return (
        np.asarray(sequences, dtype=float),
        np.asarray(targets, dtype=int),
        indices,
        np.asarray(raw_returns, dtype=float),
        np.asarray(thresholds, dtype=float),
    )


def train_network(X_train, y_train):
    scaler = StandardScaler()
    X_flat = X_train.reshape(len(X_train), -1)

    split = int(len(X_flat) * (1 - TRAIN_VALIDATION_FRACTION))
    split = min(max(split, 1), len(X_flat) - 1)
    X_fit, X_val = X_flat[:split], X_flat[split:]
    y_fit, y_val = y_train[:split], y_train[split:]

    X_fit_scaled = scaler.fit_transform(X_fit)
    X_val_scaled = scaler.transform(X_val)

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=0.02,
        batch_size=64,
        learning_rate_init=0.0005,
        max_iter=1,
        warm_start=True,
        shuffle=True,
        random_state=42,
    )

    losses = []
    validation_scores = []
    best_score = -np.inf
    best_state = None
    patience = 0

    print("Entrenando red neuronal por épocas con validación temporal...")
    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.fit(X_fit_scaled, y_fit)
        losses.append(model.loss_)

        val_pred = model.predict(X_val_scaled)
        score = balanced_accuracy_score(y_val, val_pred)
        validation_scores.append(score)
        print(f"Época {epoch:03d} | pérdida = {model.loss_:.5f} | validación temporal = {score:.2%}")

        if score > best_score + 0.0001:
            best_score = score
            best_state = {
                "coefs_": [coef.copy() for coef in model.coefs_],
                "intercepts_": [bias.copy() for bias in model.intercepts_],
            }
            patience = 0
        else:
            patience += 1
            if patience >= TRAIN_PATIENCE:
                print("Parada temprana: la validación temporal dejó de mejorar.")
                break

    if best_state is not None:
        model.coefs_ = best_state["coefs_"]
        model.intercepts_ = best_state["intercepts_"]

    model.loss_curve_ = losses
    model.n_iter_ = len(losses)
    model.loss_ = losses[-1]
    return model, scaler


def make_walk_forward_splits(total_samples):
    first_test = int(total_samples * MIN_TRAIN_FRACTION)
    remaining = total_samples - first_test
    test_size = max(1, remaining // WALK_FORWARD_SPLITS)
    splits = []
    train_end = first_test
    for split_id in range(WALK_FORWARD_SPLITS):
        test_start = train_end
        test_end = total_samples if split_id == WALK_FORWARD_SPLITS - 1 else min(total_samples, test_start + test_size)
        if test_end <= test_start:
            break
        splits.append((split_id + 1, train_end, test_start, test_end))
        train_end = test_end
    return splits


def simulate_strategy(close, probabilities):
    cash = INITIAL_CASH
    position = 0.0
    trade_count = 0
    equity = []

    for price, probs in zip(close.to_numpy(), probabilities):
        p_down, p_neutral, p_up = probs

        if position == 0:
            edge = p_up - p_down
            if p_up >= ENTRY_UP_PROBABILITY and edge >= MIN_PROBABILITY_EDGE:
                position = (cash * (1 - COMMISSION)) / price
                cash = 0.0
                trade_count += 1
        else:
            edge = p_up - p_down
            if p_down >= EXIT_DOWN_PROBABILITY or edge <= 0:
                cash = position * price * (1 - COMMISSION)
                position = 0.0
                trade_count += 1

        equity.append(cash if position == 0 else position * price)

    if position > 0:
        cash = position * close.iloc[-1] * (1 - COMMISSION)
        trade_count += 1

    return cash, trade_count, equity


def main():
    print("=== IA-Trade | 3 ESTADOS + PATRONES + WALK-FORWARD ===")
    print(f"Activo: {TICKER}")
    print(f"Datos: {INTRADAY_PERIOD} | {INTRADAY_INTERVAL}")
    print(f"Patrón observado: últimas {LOOKBACK_BARS} velas")
    print(f"Horizonte: {INTRADAY_HORIZON_BARS} velas")
    print("Estados: BAJADA FUERTE / NEUTRO / SUBIDA FUERTE")
    print(f"Movimiento fuerte: >= {MIN_MOVE_RETURN:.2%} y >= volatilidad")
    print(f"Entrada: subida >= {ENTRY_UP_PROBABILITY:.0%} y ventaja >= {MIN_PROBABILITY_EDGE:.0%}")
    print(f"Salida: bajada >= {EXIT_DOWN_PROBABILITY:.0%} o ventaja de subida <= 0")
    print("Validación: walk-forward + validación temporal interna")
    print("Modo: SIMULACIÓN / sin broker")
    print()

    data = download_market_data(TICKER, period=INTRADAY_PERIOD, interval=INTRADAY_INTERVAL)
    X, y, indices, raw_returns, thresholds = build_sequences(data)
    splits = make_walk_forward_splits(len(X))

    print(f"Secuencias totales:                 {len(X)}")
    print(f"Bajadas fuertes:                    {int((y == 0).sum())}")
    print(f"Neutras:                             {int((y == 1).sum())}")
    print(f"Subidas fuertes:                    {int((y == 2).sum())}")
    print(f"Bloques OOS:                        {len(splits)}")

    all_probabilities, all_indices, all_targets = [], [], []
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
        X_test_scaled = scaler.transform(X_test.reshape(len(X_test), -1))
        probabilities = model.predict_proba(X_test_scaled)
        predictions = np.argmax(probabilities, axis=1)
        score = balanced_accuracy_score(y_test, predictions) * 100
        block_scores.append(score)

        all_probabilities.extend(probabilities.tolist())
        all_indices.extend(indices[test_start:test_end])
        all_targets.extend(y_test.tolist())
        all_losses.append(model.loss_curve_)

        print(f"Épocas realizadas:       {model.n_iter_}")
        print(f"Pérdida final:            {model.loss_:.5f}")
        print(f"Balanced accuracy OOS:   {score:.2f}%")

    probabilities_all = np.asarray(all_probabilities)
    targets_all = np.asarray(all_targets)
    predictions_all = np.argmax(probabilities_all, axis=1)
    overall_score = balanced_accuracy_score(targets_all, predictions_all) * 100

    close = data.loc[all_indices, "Close"].astype(float)
    final_value, trade_count, equity = simulate_strategy(close, probabilities_all)
    buy_hold = INITIAL_CASH * (close.iloc[-1] / close.iloc[0])

    print()
    print("=== RESULTADO FINAL WALK-FORWARD ===")
    print(f"Valor final IA:          {final_value:.2f} €")
    print(f"Rentabilidad IA:         {(final_value / INITIAL_CASH - 1) * 100:.2f} %")
    print(f"Buy & Hold final:        {buy_hold:.2f} €")
    print(f"Operaciones:             {trade_count}")
    print(f"Operaciones/día aprox.:  {trade_count / max(len(close) / 26, 1):.2f}")
    print(f"Balanced accuracy OOS:   {overall_score:.2f}%")
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
    plt.title("Entrenamiento walk-forward con 3 estados")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
