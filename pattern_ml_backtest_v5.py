"""IA-Trade v5: predicción multivariable del recorrido futuro.

La red no intenta responder solo «sube/baja». Predice simultáneamente:
- retorno al final del horizonte
- máximo recorrido favorable
- máximo recorrido adverso
- momento relativo del máximo favorable
- momento relativo del máximo adverso

Todo se entrena sobre el 70% inicial y el 30% final queda completamente ciego.
Modo exclusivamente SIMULACIÓN.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from config import COMMISSION, INITIAL_CASH, INTRADAY_HORIZON_BARS, INTRADAY_INTERVAL, INTRADAY_PERIOD, TICKER
from market import download_market_data

LOOKBACK_BARS = 32
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TRAIN_EPOCHS = 300

# La decisión usa predicciones, no una etiqueta binaria.
MIN_EXPECTED_RETURN = 0.0015       # 0.15%
MIN_EXPECTED_MFE = 0.0030          # 0.30%
MAX_EXPECTED_MAE = 0.0025          # 0.25% de riesgo adverso esperado
MIN_REWARD_RISK = 1.40


def build_features(data):
    close = data["Close"].astype(float)
    open_price = data["Open"].astype(float)
    high = data["High"].astype(float)
    low = data["Low"].astype(float)
    volume = data["Volume"].astype(float).replace(0, np.nan)

    log_return = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
    features = [
        log_return,
        (high - low) / close,
        (close - open_price) / close,
        (high - np.maximum(open_price, close)) / close,
        (np.minimum(open_price, close) - low) / close,
        np.log(volume / volume.shift(1)).replace([np.inf, -np.inf], np.nan),
    ]

    ema8 = close.ewm(span=8, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    features += [close / ema8 - 1, close / ema21 - 1, ema8 / ema21 - 1]
    features += [close / close.shift(4) - 1, close / close.shift(8) - 1]
    features += [log_return.rolling(n, min_periods=n).std() for n in (8, 16, 32)]
    features.append(volume / volume.rolling(16, min_periods=8).mean() - 1)

    for n in (16, 32):
        rh = high.rolling(n, min_periods=n).max()
        rl = low.rolling(n, min_periods=n).min()
        features.append((close - rl) / (rh - rl))

    return np.column_stack([f.to_numpy() for f in features])


def build_dataset(data):
    frame = build_features(data)
    close = data["Close"].astype(float).to_numpy()
    high = data["High"].astype(float).to_numpy()
    low = data["Low"].astype(float).to_numpy()

    sequences = []
    targets = []
    indices = []

    horizon = INTRADAY_HORIZON_BARS
    for end in range(LOOKBACK_BARS - 1, len(data) - horizon):
        window = frame[end - LOOKBACK_BARS + 1 : end + 1]
        if not np.isfinite(window).all():
            continue

        entry = close[end]
        future_close = close[end + horizon]
        future_highs = high[end + 1 : end + horizon + 1]
        future_lows = low[end + 1 : end + horizon + 1]

        if not np.isfinite(entry) or entry <= 0:
            continue

        close_return = future_close / entry - 1.0
        mfe = np.max(future_highs) / entry - 1.0
        mae = 1.0 - np.min(future_lows) / entry  # positivo = movimiento adverso
        mfe_idx = int(np.argmax(future_highs)) + 1
        mae_idx = int(np.argmin(future_lows)) + 1

        # Los tiempos se normalizan a 0..1 para que la red pueda aprenderlos mejor.
        time_to_mfe = mfe_idx / horizon
        time_to_mae = mae_idx / horizon

        sequences.append(window)
        targets.append([close_return, mfe, mae, time_to_mfe, time_to_mae])
        indices.append(data.index[end])

    return np.asarray(sequences), np.asarray(targets), indices


def train_network(X_train, y_train):
    X_flat = X_train.reshape(len(X_train), -1)
    split = int(len(X_flat) * (1 - VALIDATION_FRACTION))
    X_fit, X_val = X_flat[:split], X_flat[split:]
    y_fit, y_val = y_train[:split], y_train[split:]

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_fit_scaled = x_scaler.fit_transform(X_fit)
    X_val_scaled = x_scaler.transform(X_val)
    y_fit_scaled = y_scaler.fit_transform(y_fit)
    y_val_scaled = y_scaler.transform(y_val)

    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=0.01,
        batch_size=64,
        learning_rate_init=0.0005,
        max_iter=1,
        warm_start=True,
        shuffle=True,
        random_state=42,
    )

    train_losses = []
    val_losses = []
    best_val = np.inf
    best_coefs = None
    best_intercepts = None
    best_epoch = 0

    print(f"Entrenando {TRAIN_EPOCHS} épocas. El TEST FINAL permanece completamente oculto.")
    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.fit(X_fit_scaled, y_fit_scaled)
        train_loss = float(model.loss_)
        val_pred = model.predict(X_val_scaled)
        val_loss = float(np.mean((val_pred - y_val_scaled) ** 2))

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_coefs = [c.copy() for c in model.coefs_]
            best_intercepts = [b.copy() for b in model.intercepts_]
            best_epoch = epoch

        if epoch == 1 or epoch % 25 == 0:
            print(f"Época {epoch:03d} | pérdida={train_loss:.5f} | validación={val_loss:.5f}")

    model.coefs_ = best_coefs
    model.intercepts_ = best_intercepts
    model.n_iter_ = TRAIN_EPOCHS
    model.loss_curve_ = train_losses
    model.loss_ = train_losses[-1]

    print(f"Entrenamiento completo. Mejor validación en época {best_epoch}: {best_val:.5f}.")
    return model, x_scaler, y_scaler, train_losses, val_losses, best_val


def simulate_strategy(close, predictions):
    cash = INITIAL_CASH
    position = 0.0
    entry_price = 0.0
    trades = 0
    equity = []

    for price, pred in zip(close.to_numpy(), predictions):
        expected_return, expected_mfe, expected_mae, time_mfe, time_mae = pred
        reward_risk = expected_mfe / max(expected_mae, 1e-6)

        should_enter = (
            position == 0
            and expected_return >= MIN_EXPECTED_RETURN
            and expected_mfe >= MIN_EXPECTED_MFE
            and expected_mae <= MAX_EXPECTED_MAE
            and reward_risk >= MIN_REWARD_RISK
        )

        # Salimos cuando el recorrido esperado se agota, el riesgo esperado aumenta,
        # o la predicción final pasa a ser claramente negativa.
        should_exit = (
            position > 0
            and (
                expected_return <= 0
                or reward_risk < 1.0
                or expected_mae > MAX_EXPECTED_MAE * 1.35
            )
        )

        if should_enter:
            position = cash * (1 - COMMISSION) / price
            cash = 0.0
            entry_price = price
            trades += 1
        elif should_exit:
            cash = position * price * (1 - COMMISSION)
            position = 0.0
            entry_price = 0.0
            trades += 1

        equity.append(cash if position == 0 else position * price)

    if position > 0:
        cash = position * close.iloc[-1] * (1 - COMMISSION)
        trades += 1

    return cash, trades, equity


def main():
    print("=== IA-Trade | PREDICCIÓN MULTIVARIABLE -> EXAMEN FINAL ===")
    print(f"Activo: {TICKER}")
    print(f"Datos: {INTRADAY_PERIOD} | {INTRADAY_INTERVAL}")
    print(f"Patrón: últimas {LOOKBACK_BARS} velas | horizonte: {INTRADAY_HORIZON_BARS} velas")
    print("La IA predice: retorno final + máximo favorable + máximo adverso + timing de ambos")
    print("Plan: 70% histórico para entrenar + 30% final totalmente oculto")
    print("Modo: SIMULACIÓN / sin broker\n")

    data = download_market_data(TICKER, period=INTRADAY_PERIOD, interval=INTRADAY_INTERVAL)
    X, y, indices = build_dataset(data)
    split = int(len(X) * TRAIN_FRACTION)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    test_indices = indices[split:]

    print(f"Secuencias totales:       {len(X)}")
    print(f"Entrenamiento:            {len(X_train)}")
    print(f"TEST FINAL CIEGO:         {len(X_test)}\n")

    model, x_scaler, y_scaler, train_losses, val_losses, best_val = train_network(X_train, y_train)

    X_test_scaled = x_scaler.transform(X_test.reshape(len(X_test), -1))
    pred_scaled = model.predict(X_test_scaled)
    predictions = y_scaler.inverse_transform(pred_scaled)

    mae_by_target = np.mean(np.abs(predictions - y_test), axis=0)
    direction_accuracy = np.mean(np.sign(predictions[:, 0]) == np.sign(y_test[:, 0])) * 100
    mfe_mae_error = np.mean(np.abs(predictions[:, 1] - y_test[:, 1])) * 100
    adverse_error = np.mean(np.abs(predictions[:, 2] - y_test[:, 2])) * 100

    close = data.loc[test_indices, "Close"].astype(float)
    final_value, trades, equity = simulate_strategy(close, predictions)
    buy_hold = INITIAL_CASH * (close.iloc[-1] / close.iloc[0])

    print("\n=== EXAMEN FINAL (DATOS NUNCA VISTOS) ===")
    print(f"Error validación global:           {best_val:.5f}")
    print(f"Acierto de dirección TEST:         {direction_accuracy:.2f}%")
    print(f"Error medio retorno final:         {mae_by_target[0] * 100:.3f}%")
    print(f"Error medio máximo favorable:      {mfe_mae_error:.3f}%")
    print(f"Error medio máximo adverso:        {adverse_error:.3f}%")
    print(f"Error medio timing favorable:      {mae_by_target[3] * INTRADAY_HORIZON_BARS:.2f} velas")
    print(f"Error medio timing adverso:        {mae_by_target[4] * INTRADAY_HORIZON_BARS:.2f} velas")
    print(f"Valor final IA:                    {final_value:.2f} €")
    print(f"Rentabilidad IA:                   {(final_value / INITIAL_CASH - 1) * 100:.2f} %")
    print(f"Buy & Hold final:                  {buy_hold:.2f} €")
    print(f"Operaciones:                       {trades}")
    print(f"Operaciones/día aprox.:            {trades / max(len(close) / 26, 1):.2f}")

    if final_value > buy_hold and direction_accuracy > 50:
        print("🟢 La predicción multivariable encuentra una señal interesante en el examen ciego.")
    elif direction_accuracy > 50:
        print("🟡 Predice parte del recorrido, pero la estrategia todavía no supera Buy & Hold.")
    else:
        print("🔴 Todavía no generaliza bien en datos completamente nuevos.")

    print("\n--- ENTRENAMIENTO ---")
    print(f"Épocas realizadas: {len(train_losses)}")
    plt.figure(figsize=(9, 5))
    plt.plot(train_losses, label="Pérdida entrenamiento")
    plt.plot(val_losses, label="Pérdida validación temporal")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.title("Aprendizaje multivariable: entrenamiento vs validación")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
