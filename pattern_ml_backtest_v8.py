"""IA-Trade v8: red especialista en dirección con validación temporal equilibrada.

La v6 encontró una pequeña señal direccional (52.81%) pero su estrategia perdió.
La v8 simplifica el objetivo: predecir si el cierre estará arriba o abajo dentro del
horizonte. La validación se hace por separado en cada activo, se promedian los resultados
y solo después se eligen los parámetros de estrategia. El 30% final de AAPL queda ciego.
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
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TRAIN_EPOCHS = 300
TRAIN_ASSETS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "SPY", "QQQ"]
TEST_ASSET = TICKER


def build_features(data):
    close = data["Close"].astype(float)
    op = data["Open"].astype(float)
    high = data["High"].astype(float)
    low = data["Low"].astype(float)
    volume = data["Volume"].astype(float).replace(0, np.nan)
    ret = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
    f = [
        ret,
        (high - low) / close,
        (close - op) / close,
        (high - np.maximum(op, close)) / close,
        (np.minimum(op, close) - low) / close,
        np.log(volume / volume.shift(1)).replace([np.inf, -np.inf], np.nan),
    ]
    ema8 = close.ewm(span=8, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    f += [close / ema8 - 1, close / ema21 - 1, ema8 / ema21 - 1]
    f += [close / close.shift(4) - 1, close / close.shift(8) - 1]
    f += [ret.rolling(n, min_periods=n).std() for n in (8, 16, 32)]
    f.append(volume / volume.rolling(16, min_periods=8).mean() - 1)
    for n in (16, 32):
        rh = high.rolling(n, min_periods=n).max()
        rl = low.rolling(n, min_periods=n).min()
        f.append((close - rl) / (rh - rl))
    return np.column_stack([x.to_numpy() for x in f])


def make_dataset(data):
    frame = build_features(data)
    close = data["Close"].astype(float).to_numpy()
    X, y, idx = [], [], []
    h = INTRADAY_HORIZON_BARS
    for end in range(LOOKBACK_BARS - 1, len(data) - h):
        window = frame[end - LOOKBACK_BARS + 1:end + 1]
        if not np.isfinite(window).all():
            continue
        future_return = close[end + h] / close[end] - 1
        if not np.isfinite(future_return):
            continue
        X.append(window)
        y.append(int(future_return > 0))
        idx.append(data.index[end])
    return np.asarray(X), np.asarray(y), idx


def balanced_accuracy(y_true, prob):
    pred = (prob >= 0.50).astype(int)
    recalls = []
    for cls in (0, 1):
        mask = y_true == cls
        if mask.any():
            recalls.append(float(np.mean(pred[mask] == cls)))
    return float(np.mean(recalls)) if recalls else 0.0


def copy_weights(model):
    return [c.copy() for c in model.coefs_], [b.copy() for b in model.intercepts_]


def train_network(fit_X, fit_y, val_sets):
    X_fit = fit_X.reshape(len(fit_X), -1)
    scaler = StandardScaler()
    X_fit = scaler.fit_transform(X_fit)
    val_scaled = [(scaler.transform(X.reshape(len(X), -1)), y) for X, y, _ in val_sets]

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
        model.fit(X_fit, fit_y)
        per_asset_loss = []
        per_asset_score = []
        for Xv, yv in val_scaled:
            p = model.predict_proba(Xv)[:, 1]
            p_clip = np.clip(p, 1e-7, 1 - 1e-7)
            per_asset_loss.append(float(-np.mean(yv * np.log(p_clip) + (1 - yv) * np.log(1 - p_clip))))
            per_asset_score.append(balanced_accuracy(yv, p))
        score = float(np.mean(per_asset_score))
        val_loss = float(np.mean(per_asset_loss))
        losses.append(float(model.loss_))
        val_losses.append(val_loss)
        val_scores.append(score)
        if score > best_score:
            best_score = score
            best_weights = copy_weights(model)
            best_epoch = epoch
        if epoch == 1 or epoch % 25 == 0:
            print(f"Época {epoch:03d} | pérdida={model.loss_:.5f} | validación media={score * 100:.2f}%")

    model.coefs_, model.intercepts_ = best_weights
    model.loss_curve_ = losses
    model.n_iter_ = TRAIN_EPOCHS
    print(f"Mejor validación equilibrada: {best_score * 100:.2f}% en época {best_epoch}.")
    return model, scaler, losses, val_losses, val_scores, best_score


def simulate(close, probabilities, threshold, min_hold):
    cash = INITIAL_CASH
    position = 0.0
    trades = 0
    held = 0
    equity = []
    for price, p_up in zip(close.to_numpy(), probabilities):
        if position > 0:
            held += 1
        enter = position == 0 and p_up >= threshold
        exit_ = position > 0 and held >= min_hold and p_up < 0.50
        if enter:
            position = cash * (1 - COMMISSION) / price
            cash = 0
            held = 0
            trades += 1
        elif exit_:
            cash = position * price * (1 - COMMISSION)
            position = 0
            held = 0
            trades += 1
        equity.append(cash if position == 0 else position * price)
    if position > 0:
        cash = position * close.iloc[-1] * (1 - COMMISSION)
        trades += 1
    return cash, trades, equity


def choose_strategy(val_results):
    best = None
    for threshold in np.arange(0.50, 0.66, 0.02):
        for hold in (1, 2, 4, 6, 8):
            returns = []
            total_trades = 0
            total_bars = 0
            for _, close, prob in val_results:
                value, trades, _ = simulate(close, prob, float(threshold), hold)
                returns.append(value / INITIAL_CASH - 1)
                total_trades += trades
                total_bars += len(close)
            avg_return = float(np.mean(returns))
            ops_day = total_trades / max(total_bars / 26, 1)
            # Solo una pequeña preferencia por una frecuencia útil, sin forzar operaciones.
            penalty = max(0.0, 2.0 - ops_day) * 0.002 + max(0.0, ops_day - 8.0) * 0.002
            objective = avg_return - penalty
            candidate = (objective, avg_return, float(threshold), hold, ops_day)
            if best is None or candidate > best:
                best = candidate
    return best


def main():
    print("=== IA-Trade v8 | ESPECIALISTA EN DIRECCIÓN MULTIACTIVO ===")
    print(f"Activos: {', '.join(TRAIN_ASSETS)}")
    print(f"Examen final ciego: {TEST_ASSET}")
    print(f"Datos: {INTRADAY_PERIOD} | {INTRADAY_INTERVAL}")
    print(f"Patrón: {LOOKBACK_BARS} velas | horizonte: {INTRADAY_HORIZON_BARS} velas")
    print("El 30% final de AAPL permanece completamente ciego.")
    print("Modo: SIMULACIÓN / sin broker\n")

    datasets = {}
    for asset in TRAIN_ASSETS:
        print(f"Descargando {asset}...")
        data = download_market_data(asset, period=INTRADAY_PERIOD, interval=INTRADAY_INTERVAL)
        X, y, idx = make_dataset(data)
        datasets[asset] = (data, X, y, idx)
        print(f"  {asset}: {len(X)} secuencias | subida={y.mean() * 100:.1f}%")

    # 30% final de AAPL: reservado desde el principio.
    test_data, test_X, test_y, test_idx = datasets[TEST_ASSET]
    blind_start = int(len(test_X) * TRAIN_FRACTION)
    blind_X = test_X[blind_start:]
    blind_y = test_y[blind_start:]
    blind_idx = test_idx[blind_start:]

    fit_X, fit_y = [], []
    val_sets = []
    for asset, (data, X, y, idx) in datasets.items():
        train_end = int(len(X) * TRAIN_FRACTION)
        val_start = int(train_end * (1 - VALIDATION_FRACTION))
        fit_X.append(X[:val_start])
        fit_y.append(y[:val_start])
        val_idx = idx[val_start:train_end]
        val_close = data.loc[val_idx, "Close"].astype(float)
        val_sets.append((X[val_start:train_end], y[val_start:train_end], val_close))

    X_fit = np.concatenate(fit_X, axis=0)
    y_fit = np.concatenate(fit_y, axis=0)
    print(f"\nMuestras entrenamiento: {len(X_fit)}")
    print(f"Muestras validación por activo: {sum(len(x[0]) for x in val_sets)}")
    print(f"Muestras TEST FINAL AAPL: {len(blind_X)}")

    model, scaler, losses, val_losses, val_scores, best_score = train_network(X_fit, y_fit, val_sets)

    # Elegimos la estrategia sobre validación temporal, no sobre AAPL ciego.
    val_results = []
    for asset, (_, X, _, idx) in zip(datasets.keys(), val_sets):
        val_X_asset, _, val_close = idx
        prob = model.predict_proba(scaler.transform(val_X_asset.reshape(len(val_X_asset), -1)))[:, 1]
        val_results.append((asset, val_close, prob))

    _, validation_return, threshold, min_hold, validation_ops = choose_strategy(val_results)

    blind_prob = model.predict_proba(scaler.transform(blind_X.reshape(len(blind_X), -1)))[:, 1]
    close_blind = test_data.loc[blind_idx, "Close"].astype(float)
    final_value, trades, _ = simulate(close_blind, blind_prob, threshold, min_hold)
    buy_hold = INITIAL_CASH * close_blind.iloc[-1] / close_blind.iloc[0]
    direction = float(np.mean((blind_prob >= 0.50) == blind_y) * 100)

    print("\n=== EXAMEN FINAL (AAPL NUNCA VISTO) ===")
    print(f"Acierto dirección TEST:            {direction:.2f}%")
    print(f"Valor final IA:                    {final_value:.2f} €")
    print(f"Rentabilidad IA:                   {(final_value / INITIAL_CASH - 1) * 100:.2f} %")
    print(f"Buy & Hold final:                  {buy_hold:.2f} €")
    print(f"Operaciones:                       {trades}")
    print(f"Operaciones/día aprox.:            {trades / max(len(close_blind) / 26, 1):.2f}")
    print("\n--- ELECCIÓN SOLO CON VALIDACIÓN ---")
    print(f"Mejor validación direccional:      {best_score * 100:.2f}%")
    print(f"Rentabilidad media validación:     {validation_return * 100:.2f}%")
    print(f"Umbral entrada:                    {threshold:.2f}")
    print(f"Mínimo permanencia:                {min_hold} velas")
    print(f"Frecuencia validación:             {validation_ops:.2f} ops/día")

    if final_value > buy_hold and direction > 50:
        print("🟢 Hay evidencia de ventaja en el examen ciego.")
    elif direction > 50:
        print("🟡 La dirección supera 50%, pero aún no convierte la señal en beneficio.")
    else:
        print("🔴 No hay evidencia suficiente de generalización direccional.")

    print("\n--- ENTRENAMIENTO ---")
    print(f"Épocas realizadas:                 {len(losses)}")
    plt.figure(figsize=(9, 5))
    plt.plot(losses, label="Pérdida entrenamiento")
    plt.plot(val_losses, label="Pérdida validación")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.title("IA-Trade v8 | dirección multiactivo")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
