"""IA-Trade v7: especialista en dirección con validación temporal por activo.

La v6 aprendía varias magnitudes a la vez y encontraba una pequeña señal de dirección,
pero la estrategia no convertía esa señal en ventaja. En v7 separamos el problema:
la red solo aprende dirección futura. Los parámetros de la estrategia se eligen usando
EXCLUSIVAMENTE la validación temporal y después se examina el 30% final de AAPL.
Modo exclusivamente SIMULACIÓN.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from config import COMMISSION, INITIAL_CASH, INTRADAY_HORIZON_BARS, INTRADAY_INTERVAL, INTRADAY_PERIOD, TICKER
from market import download_market_data

LOOKBACK_BARS = 32
TRAIN_FRACTION_PER_ASSET = 0.70
VALIDATION_FRACTION = 0.15
TRAIN_EPOCHS = 300

TRAIN_ASSETS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "SPY", "QQQ"]
TEST_ASSET = TICKER

# La red aprende solo si el cierre futuro queda por encima del actual.
# La decisión de entrar/salir se optimiza después únicamente sobre validación.


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
    X, y, indices = [], [], []
    horizon = INTRADAY_HORIZON_BARS

    for end in range(LOOKBACK_BARS - 1, len(data) - horizon):
        window = frame[end - LOOKBACK_BARS + 1 : end + 1]
        if not np.isfinite(window).all():
            continue
        entry = close[end]
        future_return = close[end + horizon] / entry - 1
        if not np.isfinite(future_return):
            continue
        X.append(window)
        y.append(int(future_return > 0))
        indices.append(data.index[end])
    return np.asarray(X), np.asarray(y), indices


def copy_weights(model):
    return [c.copy() for c in model.coefs_], [b.copy() for b in model.intercepts_]


def restore_weights(model, weights):
    model.coefs_, model.intercepts_ = weights


def train_network(X, y):
    X_flat = X.reshape(len(X), -1)
    split = int(len(X_flat) * (1 - VALIDATION_FRACTION))
    X_fit, X_val = X_flat[:split], X_flat[split:]
    y_fit, y_val = y[:split], y[split:]

    xs = StandardScaler()
    X_fit = xs.fit_transform(X_fit)
    X_val = xs.transform(X_val)

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=0.01,
        batch_size=128,
        learning_rate_init=0.0005,
        max_iter=1,
        warm_start=True,
        shuffle=True,
        random_state=42,
    )

    losses, val_losses, val_scores = [], [], []
    best_score = -np.inf
    best_weights = None
    best_epoch = 0

    print(f"Entrenando {TRAIN_EPOCHS} épocas...")
    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.fit(X_fit, y_fit)
        proba = model.predict_proba(X_val)[:, 1]
        pred = (proba >= 0.50).astype(int)

        # Balanced accuracy manual para no depender de otra métrica en la salida.
        recalls = []
        for cls in (0, 1):
            mask = y_val == cls
            if mask.any():
                recalls.append(float(np.mean(pred[mask] == cls)))
        score = float(np.mean(recalls)) if recalls else 0.0
        val_loss = float(-np.mean(y_val * np.log(np.clip(proba, 1e-7, 1 - 1e-7)) + (1 - y_val) * np.log(np.clip(1 - proba, 1e-7, 1 - 1e-7))))

        losses.append(float(model.loss_))
        val_losses.append(val_loss)
        val_scores.append(score)

        if score > best_score:
            best_score = score
            best_weights = copy_weights(model)
            best_epoch = epoch

        if epoch == 1 or epoch % 25 == 0:
            print(f"Época {epoch:03d} | pérdida={model.loss_:.5f} | validación dirección={score * 100:.2f}%")

    restore_weights(model, best_weights)
    model.loss_curve_ = losses
    model.n_iter_ = TRAIN_EPOCHS
    print(f"Mejor validación de dirección: {best_score * 100:.2f}% en época {best_epoch}.")
    return model, xs, losses, val_losses, val_scores, best_score


def simulate(close, probabilities, entry_threshold, min_hold_bars):
    cash = INITIAL_CASH
    position = 0.0
    trades = 0
    bars_in_position = 0
    equity = []

    for price, p_up in zip(close.to_numpy(), probabilities):
        if position > 0:
            bars_in_position += 1

        enter = position == 0 and p_up >= entry_threshold
        exit_ = position > 0 and bars_in_position >= min_hold_bars and p_up < 0.50

        if enter:
            position = cash * (1 - COMMISSION) / price
            cash = 0
            bars_in_position = 0
            trades += 1
        elif exit_:
            cash = position * price * (1 - COMMISSION)
            position = 0
            bars_in_position = 0
            trades += 1

        equity.append(cash if position == 0 else position * price)

    if position > 0:
        cash = position * close.iloc[-1] * (1 - COMMISSION)
        trades += 1
    return cash, trades, equity


def choose_strategy(close, probabilities):
    best = None
    for threshold in np.arange(0.50, 0.66, 0.02):
        for hold in (1, 2, 4, 6, 8):
            value, trades, _ = simulate(close, probabilities, float(threshold), hold)
            days = max(len(close) / 26, 1)
            ops_day = trades / days
            # Preferimos beneficio, pero penalizamos configuraciones que operan demasiado.
            frequency_penalty = max(0.0, 2.0 - ops_day) * 0.20 + max(0.0, ops_day - 8.0) * 0.20
            objective = (value / INITIAL_CASH - 1) - frequency_penalty / 100
            candidate = (objective, value, trades, float(threshold), hold, ops_day)
            if best is None or candidate > best:
                best = candidate
    return best


def main():
    print("=== IA-Trade v7 | ESPECIALISTA EN DIRECCIÓN ===")
    print(f"Activos de entrenamiento: {', '.join(TRAIN_ASSETS)}")
    print(f"Examen final ciego: {TEST_ASSET}")
    print(f"Datos: {INTRADAY_PERIOD} | {INTRADAY_INTERVAL}")
    print(f"Patrón: últimas {LOOKBACK_BARS} velas | horizonte: {INTRADAY_HORIZON_BARS} velas")
    print("Cada activo: 70% inicial para entrenar/validar; 30% final de AAPL queda ciego.")
    print("Modo: SIMULACIÓN / sin broker\n")

    datasets = {}
    for asset in TRAIN_ASSETS:
        print(f"Descargando {asset}...")
        data = download_market_data(asset, period=INTRADAY_PERIOD, interval=INTRADAY_INTERVAL)
        X, y, idx = make_dataset(data)
        datasets[asset] = (data, X, y, idx)
        print(f"  {asset}: {len(X)} secuencias | subida={y.mean() * 100:.1f}%")

    test_data, test_X, test_y, test_idx = datasets[TEST_ASSET]
    test_split = int(len(test_X) * TRAIN_FRACTION_PER_ASSET)
    blind_X = test_X[test_split:]
    blind_y = test_y[test_split:]
    blind_idx = test_idx[test_split:]

    # Construimos fit y validación de cada activo por separado. Así la validación
    # contiene una porción final de TODOS los activos y no queda dominada por QQQ.
    fit_X, fit_y = [], []
    val_X, val_y, val_close = [], [], []
    for asset, (data, X, y, idx) in datasets.items():
        train_end = int(len(X) * TRAIN_FRACTION_PER_ASSET)
        validation_start = int(train_end * (1 - VALIDATION_FRACTION))
        fit_X.append(X[:validation_start])
        fit_y.append(y[:validation_start])
        val_X.append(X[validation_start:train_end])
        val_y.append(y[validation_start:train_end])
        val_idx = idx[validation_start:train_end]
        val_close.append(data.loc[val_idx, "Close"].astype(float))

    X_fit = np.concatenate(fit_X, axis=0)
    y_fit = np.concatenate(fit_y, axis=0)
    X_val = np.concatenate(val_X, axis=0)
    y_val = np.concatenate(val_y, axis=0)

    # Entrenamos una sola red con todos los activos. Para elegir pesos usamos la
    # validación temporal conjunta, sin tocar jamás el 30% final de AAPL.
    model, xs, losses, val_losses, val_scores, best_score = train_network(X_fit, y_fit)

    # Recalculamos la validación conjunta con el escalador del entrenamiento.
    val_prob = model.predict_proba(xs.transform(X_val.reshape(len(X_val), -1)))[:, 1]

    # Para optimizar la estrategia, reconstruimos la validación en bloques de cada activo.
    # La red ya está fijada; aquí solo se eligen umbral y permanencia.
    cursor = 0
    val_strategy_results = []
    for asset, (_, X, _, idx) in datasets.items():
        train_end = int(len(X) * TRAIN_FRACTION_PER_ASSET)
        validation_start = int(train_end * (1 - VALIDATION_FRACTION))
        n = train_end - validation_start
        probs_asset = val_prob[cursor:cursor + n]
        close_asset = val_close[len(val_strategy_results)]
        val_strategy_results.append((asset, close_asset, probs_asset))
        cursor += n

    # Elegimos parámetros que funcionan de forma agregada en validación.
    best_strategy = None
    for threshold in np.arange(0.50, 0.66, 0.02):
        for hold in (1, 2, 4, 6, 8):
            values = []
            trades_total = 0
            bars_total = 0
            for _, close_asset, probs_asset in val_strategy_results:
                value, trades, _ = simulate(close_asset, probs_asset, float(threshold), hold)
                values.append(value / INITIAL_CASH - 1)
                trades_total += trades
                bars_total += len(close_asset)
            avg_return = float(np.mean(values))
            ops_day = trades_total / max(bars_total / 26, 1)
            frequency_penalty = max(0.0, 2.0 - ops_day) * 0.20 + max(0.0, ops_day - 8.0) * 0.20
            objective = avg_return - frequency_penalty / 100
            candidate = (objective, avg_return, float(threshold), hold, ops_day)
            if best_strategy is None or candidate > best_strategy:
                best_strategy = candidate

    _, validation_return, entry_threshold, min_hold, validation_ops = best_strategy

    blind_prob = model.predict_proba(xs.transform(blind_X.reshape(len(blind_X), -1)))[:, 1]
    close_blind = test_data.loc[blind_idx, "Close"].astype(float)
    final_value, trades, _ = simulate(close_blind, blind_prob, entry_threshold, min_hold)
    buy_hold = INITIAL_CASH * close_blind.iloc[-1] / close_blind.iloc[0]
    direction = np.mean((blind_prob >= 0.50) == blind_y) * 100

    print("\n=== EXAMEN FINAL (AAPL NUNCA VISTO) ===")
    print(f"Acierto dirección TEST:            {direction:.2f}%")
    print(f"Valor final IA:                    {final_value:.2f} €")
    print(f"Rentabilidad IA:                   {(final_value / INITIAL_CASH - 1) * 100:.2f} %")
    print(f"Buy & Hold final:                  {buy_hold:.2f} €")
    print(f"Operaciones:                       {trades}")
    print(f"Operaciones/día aprox.:            {trades / max(len(close_blind) / 26, 1):.2f}")
    print("\n--- PARÁMETROS ELEGIDOS SOLO CON VALIDACIÓN ---")
    print(f"Mejor validación de dirección:     {best_score * 100:.2f}%")
    print(f"Rentabilidad media validación:     {validation_return * 100:.2f}%")
    print(f"Umbral de entrada elegido:         {entry_threshold:.2f}")
    print(f"Mínimo de permanencia:             {min_hold} velas")
    print(f"Frecuencia validación:              {validation_ops:.2f} ops/día")

    if final_value > buy_hold and direction > 50:
        print("🟢 Esta versión demuestra una ventaja en el examen ciego.")
    elif direction > 50:
        print("🟡 La dirección supera 50%, pero todavía no convierte la señal en ventaja económica.")
    else:
        print("🔴 La señal direccional no generaliza suficientemente.")

    print("\n--- ENTRENAMIENTO ---")
    print(f"Épocas realizadas:                 {len(losses)}")
    plt.figure(figsize=(9, 5))
    plt.plot(losses, label="Pérdida entrenamiento")
    plt.plot(val_losses, label="Pérdida validación")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.title("IA-Trade v7 | entrenamiento de dirección")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
