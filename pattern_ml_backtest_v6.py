"""IA-Trade v6: aprendizaje multiactivo para buscar patrones que generalicen.

Entrena con varios activos usando solo su 70% histórico inicial y deja el 30%
final de AAPL completamente ciego. Predice recorrido, riesgo y timing futuro.
Modo exclusivamente SIMULACIÓN.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from config import COMMISSION, INITIAL_CASH, INTRADAY_HORIZON_BARS, INTRADAY_INTERVAL, INTRADAY_PERIOD, TICKER
from market import download_market_data

LOOKBACK_BARS = 32
TRAIN_FRACTION_PER_ASSET = 0.70
VALIDATION_FRACTION = 0.15
TRAIN_EPOCHS = 300

TRAIN_ASSETS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "SPY", "QQQ"]
TEST_ASSET = TICKER

# Decisión deliberadamente conservadora: solo entra si retorno esperado,
# recorrido favorable y relación recompensa/riesgo son razonables.
MIN_EXPECTED_RETURN = 0.0015
MIN_EXPECTED_MFE = 0.0030
MAX_EXPECTED_MAE = 0.0025
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


def make_dataset(data):
    frame = build_features(data)
    close = data["Close"].astype(float).to_numpy()
    high = data["High"].astype(float).to_numpy()
    low = data["Low"].astype(float).to_numpy()
    X, y, indices = [], [], []
    horizon = INTRADAY_HORIZON_BARS

    for end in range(LOOKBACK_BARS - 1, len(data) - horizon):
        window = frame[end - LOOKBACK_BARS + 1 : end + 1]
        if not np.isfinite(window).all():
            continue
        entry = close[end]
        future_highs = high[end + 1 : end + horizon + 1]
        future_lows = low[end + 1 : end + horizon + 1]
        if not np.isfinite(entry) or entry <= 0:
            continue

        close_return = close[end + horizon] / entry - 1
        mfe = np.max(future_highs) / entry - 1
        mae = 1 - np.min(future_lows) / entry
        time_mfe = (np.argmax(future_highs) + 1) / horizon
        time_mae = (np.argmin(future_lows) + 1) / horizon
        X.append(window)
        y.append([close_return, mfe, mae, time_mfe, time_mae])
        indices.append(data.index[end])
    return np.asarray(X), np.asarray(y), indices


def train_network(X, y):
    X_flat = X.reshape(len(X), -1)
    split = int(len(X_flat) * (1 - VALIDATION_FRACTION))
    X_fit, X_val = X_flat[:split], X_flat[split:]
    y_fit, y_val = y[:split], y[split:]

    xs = StandardScaler()
    ys = StandardScaler()
    X_fit = xs.fit_transform(X_fit)
    X_val = xs.transform(X_val)
    y_fit = ys.fit_transform(y_fit)
    y_val = ys.transform(y_val)

    model = MLPRegressor(
        hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
        alpha=0.01, batch_size=128, learning_rate_init=0.0005,
        max_iter=1, warm_start=True, shuffle=True, random_state=42,
    )
    losses, val_losses = [], []
    best = np.inf
    best_coefs = best_intercepts = None
    best_epoch = 0

    print(f"Entrenando {TRAIN_EPOCHS} épocas sobre varios activos...")
    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.fit(X_fit, y_fit)
        train_loss = float(model.loss_)
        pred = model.predict(X_val)
        val_loss = float(np.mean((pred - y_val) ** 2))
        losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best:
            best = val_loss
            best_coefs = [c.copy() for c in model.coefs_]
            best_intercepts = [b.copy() for b in model.intercepts_]
            best_epoch = epoch
        if epoch == 1 or epoch % 25 == 0:
            print(f"Época {epoch:03d} | pérdida={train_loss:.5f} | validación={val_loss:.5f}")

    model.coefs_ = best_coefs
    model.intercepts_ = best_intercepts
    model.loss_curve_ = losses
    model.n_iter_ = TRAIN_EPOCHS
    print(f"Entrenamiento completo. Mejor validación en época {best_epoch}: {best:.5f}.")
    return model, xs, ys, losses, val_losses


def simulate(close, predictions):
    cash = INITIAL_CASH
    position = 0.0
    trades = 0
    equity = []

    for price, pred in zip(close.to_numpy(), predictions):
        expected_return, expected_mfe, expected_mae, _, _ = pred
        reward_risk = expected_mfe / max(expected_mae, 1e-6)
        enter = (
            position == 0
            and expected_return >= MIN_EXPECTED_RETURN
            and expected_mfe >= MIN_EXPECTED_MFE
            and expected_mae <= MAX_EXPECTED_MAE
            and reward_risk >= MIN_REWARD_RISK
        )
        exit_ = (
            position > 0
            and (expected_return <= 0 or reward_risk < 1.0 or expected_mae > MAX_EXPECTED_MAE * 1.35)
        )
        if enter:
            position = cash * (1 - COMMISSION) / price
            cash = 0
            trades += 1
        elif exit_:
            cash = position * price * (1 - COMMISSION)
            position = 0
            trades += 1
        equity.append(cash if position == 0 else position * price)

    if position > 0:
        cash = position * close.iloc[-1] * (1 - COMMISSION)
        trades += 1
    return cash, trades, equity


def main():
    print("=== IA-Trade v6 | APRENDER PATRONES DE VARIOS ACTIVOS ===")
    print(f"Activos de entrenamiento: {', '.join(TRAIN_ASSETS)}")
    print(f"Examen final ciego: {TEST_ASSET}")
    print(f"Datos: {INTRADAY_PERIOD} | {INTRADAY_INTERVAL}")
    print(f"Patrón: últimas {LOOKBACK_BARS} velas | horizonte: {INTRADAY_HORIZON_BARS} velas")
    print("Cada activo aporta solo su 70% inicial al entrenamiento.")
    print("El 30% final de AAPL permanece completamente oculto.")
    print("Modo: SIMULACIÓN / sin broker\n")

    datasets = {}
    for asset in TRAIN_ASSETS:
        print(f"Descargando {asset}...")
        data = download_market_data(asset, period=INTRADAY_PERIOD, interval=INTRADAY_INTERVAL)
        X, y, idx = make_dataset(data)
        datasets[asset] = (data, X, y, idx)
        print(f"  {asset}: {len(X)} secuencias")

    # El examen de AAPL se separa antes de construir el conjunto de entrenamiento.
    test_data, test_X, test_y, test_idx = datasets[TEST_ASSET]
    test_split = int(len(test_X) * TRAIN_FRACTION_PER_ASSET)
    X_test = test_X[test_split:]
    y_test = test_y[test_split:]
    test_indices = test_idx[test_split:]

    train_X, train_y = [], []
    for asset, (_, X, y, _) in datasets.items():
        split = int(len(X) * TRAIN_FRACTION_PER_ASSET)
        train_X.append(X[:split])
        train_y.append(y[:split])
    X_train = np.concatenate(train_X, axis=0)
    y_train = np.concatenate(train_y, axis=0)

    # Orden temporal aproximado dentro de cada activo no se mezcla con el futuro de AAPL:
    # AAPL final nunca entra en este conjunto.
    print(f"\nMuestras entrenamiento: {len(X_train)}")
    print(f"Muestras TEST FINAL AAPL: {len(X_test)}")

    model, xs, ys, losses, val_losses = train_network(X_train, y_train)
    X_test_scaled = xs.transform(X_test.reshape(len(X_test), -1))
    predictions = ys.inverse_transform(model.predict(X_test_scaled))

    errors = np.mean(np.abs(predictions - y_test), axis=0) * 100
    direction = np.mean(np.sign(predictions[:, 0]) == np.sign(y_test[:, 0])) * 100
    close = test_data.loc[test_indices, "Close"].astype(float)
    final_value, trades, _ = simulate(close, predictions)
    buy_hold = INITIAL_CASH * close.iloc[-1] / close.iloc[0]

    print("\n=== EXAMEN FINAL (AAPL NUNCA VISTO) ===")
    print(f"Acierto dirección:                 {direction:.2f}%")
    print(f"Error retorno final:               {errors[0]:.3f}%")
    print(f"Error máximo favorable:            {errors[1]:.3f}%")
    print(f"Error máximo adverso:              {errors[2]:.3f}%")
    print(f"Error timing favorable:            {errors[3] * INTRADAY_HORIZON_BARS / 100:.2f} velas")
    print(f"Error timing adverso:              {errors[4] * INTRADAY_HORIZON_BARS / 100:.2f} velas")
    print(f"Valor final IA:                    {final_value:.2f} €")
    print(f"Rentabilidad IA:                   {(final_value / INITIAL_CASH - 1) * 100:.2f} %")
    print(f"Buy & Hold final:                  {buy_hold:.2f} €")
    print(f"Operaciones:                       {trades}")
    print(f"Operaciones/día aprox.:            {trades / max(len(close) / 26, 1):.2f}")

    if final_value > buy_hold and direction > 50:
        print("🟢 Hay una señal interesante de generalización multiactivo.")
    elif direction > 50:
        print("🟡 Aprende algo de dirección, pero la estrategia aún no gana al mercado.")
    else:
        print("🔴 No hay evidencia suficiente de que los patrones generalicen.")

    print("\n--- ENTRENAMIENTO ---")
    print(f"Épocas realizadas: {len(losses)}")
    plt.figure(figsize=(9, 5))
    plt.plot(losses, label="Pérdida entrenamiento")
    plt.plot(val_losses, label="Pérdida validación")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.title("Aprendizaje multiactivo")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
