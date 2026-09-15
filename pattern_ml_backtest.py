"""IA de patrones secuenciales con regresión y validación walk-forward intradía.

La red recibe una secuencia de velas recientes y aprende a estimar el
rendimiento futuro del activo. La evaluación siempre se hace sobre periodos
posteriores que no ha visto durante su entrenamiento.
Todo el proceso es investigación histórica y simulación.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
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
ENTRY_RETURN = 0.0025
EXIT_RETURN = 0.0
MIN_TRAIN_FRACTION = 0.45
WALK_FORWARD_SPLITS = 4
TRAIN_VALIDATION_FRACTION = 0.15
TRAIN_EPOCHS = 120
TRAIN_PATIENCE = 20


def build_sequences(data):
    """Crea secuencias con estructura de vela, tendencia, momentum, volatilidad y volumen."""
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
        log_return,
        range_pct,
        body_pct,
        upper_wick_pct,
        lower_wick_pct,
        volume_change,
        distance_ema_8,
        distance_ema_21,
        ema_spread,
        momentum_4,
        momentum_8,
        volatility_8,
        volatility_16,
        volatility_32,
        relative_volume,
        position_16,
        position_32,
    ]

    frame = np.column_stack([series.to_numpy() for series in feature_series])

    future_return = close.shift(-INTRADAY_HORIZON_BARS) / close - 1
    future_values = future_return.to_numpy()

    sequences = []
    targets = []
    indices = []
    volatilities = []
    for end in range(LOOKBACK_BARS - 1, len(data) - INTRADAY_HORIZON_BARS):
        window = frame[end - LOOKBACK_BARS + 1:end + 1]
        target = future_values[end]
        volatility = volatility_32.iloc[end]
        if not np.isfinite(window).all() or not np.isfinite(target) or not np.isfinite(volatility):
            continue
        sequences.append(window)
        targets.append(float(target))
        indices.append(data.index[end])
        volatilities.append(float(volatility))

    if not sequences:
        raise ValueError("No hay suficientes secuencias limpias para entrenar la IA.")

    return (
        np.asarray(sequences, dtype=float),
        np.asarray(targets, dtype=float),
        indices,
        np.asarray(volatilities, dtype=float),
    )


def train_network(X_train, y_train):
    """Entrena por épocas usando validación temporal interna, sin mezclar futuro."""
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_flat = X_train.reshape(len(X_train), -1)
    split = int(len(X_flat) * (1 - TRAIN_VALIDATION_FRACTION))
    split = min(max(split, 1), len(X_flat) - 1)

    X_fit, X_val = X_flat[:split], X_flat[split:]
    y_fit, y_val = y_train[:split], y_train[split:]

    X_fit_scaled = scaler_x.fit_transform(X_fit)
    X_val_scaled = scaler_x.transform(X_val)
    y_fit_scaled = scaler_y.fit_transform(y_fit.reshape(-1, 1)).ravel()

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

    losses = []
    validation_mae = []
    best_score = np.inf
    best_state = None
    patience = 0

    print("Entrenando red neuronal por épocas con validación temporal...")
    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.fit(X_fit_scaled, y_fit_scaled)
        losses.append(model.loss_)

        val_predictions_scaled = model.predict(X_val_scaled)
        val_predictions = scaler_y.inverse_transform(val_predictions_scaled.reshape(-1, 1)).ravel()
        val_mae = mean_absolute_error(y_val, val_predictions)
        validation_mae.append(val_mae)

        print(
            f"Época {epoch:03d} | pérdida = {model.loss_:.5f} | "
            f"MAE validación = {val_mae:.4%}"
        )

        if val_mae < best_score - 0.00001:
            best_score = val_mae
            best_state = {
                "coefs_": [coef.copy() for coef in model.coefs_],
                "intercepts_": [bias.copy() for bias in model.intercepts_],
            }
            patience = 0
        else:
            patience += 1
            if patience >= TRAIN_PATIENCE:
                print("Parada temprana: el error de validación dejó de mejorar.")
                break

    if best_state is not None:
        model.coefs_ = best_state["coefs_"]
        model.intercepts_ = best_state["intercepts_"]

    model.loss_curve_ = losses
    model.n_iter_ = len(losses)
    model.loss_ = losses[-1]
    model.validation_mae_ = validation_mae
    return model, scaler_x, scaler_y


def predict_block(model, scaler_x, scaler_y, X_test):
    """Predice rendimiento futuro para un bloque totalmente fuera de muestra."""
    X_test_scaled = scaler_x.transform(X_test.reshape(len(X_test), -1))
    predicted_scaled = model.predict(X_test_scaled)
    return scaler_y.inverse_transform(predicted_scaled.reshape(-1, 1)).ravel()


def simulate_strategy(close, predicted_returns, volatility):
    """Simula entradas/salidas usando el rendimiento futuro estimado, sin broker."""
    cash = INITIAL_CASH
    position = 0.0
    equity = []
    trade_count = 0

    for price, prediction, vol in zip(close.to_numpy(), predicted_returns, volatility):
        entry_threshold = max(ENTRY_RETURN, float(vol))

        if position == 0 and prediction >= entry_threshold:
            position = (cash * (1 - COMMISSION)) / price
            cash = 0.0
            trade_count += 1
        elif position > 0 and prediction <= EXIT_RETURN:
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
    print("=== IA-Trade | IA DE PATRONES + REGRESIÓN + WALK-FORWARD ===")
    print(f"Activo: {TICKER}")
    print(f"Datos: {INTRADAY_PERIOD} | {INTRADAY_INTERVAL}")
    print(f"Patrón observado: últimas {LOOKBACK_BARS} velas")
    print("Información por vela: estructura + tendencia + momentum + volatilidad + volumen")
    print(f"Horizonte: {INTRADAY_HORIZON_BARS} velas")
    print("Objetivo: predecir el rendimiento futuro, no solo subir/bajar")
    print("Arquitectura: 128 → 64 neuronas")
    print("Validación: 4 bloques temporales fuera de muestra")
    print("Validación interna: temporal (sin mezclar futuro)")
    print("Modo: SIMULACIÓN / sin broker")
    print()

    data = download_market_data(
        TICKER,
        period=INTRADAY_PERIOD,
        interval=INTRADAY_INTERVAL,
    )
    X, y, indices, volatilities = build_sequences(data)
    splits = make_walk_forward_splits(len(X))

    print(f"Secuencias totales:      {len(X)}")
    print(f"Entrenamiento inicial:   {splits[0][1]}")
    print(f"Bloques OOS:             {len(splits)}")
    print(f"Entrada IA:              >= {ENTRY_RETURN:.2%} o volatilidad")
    print(f"Salida IA:               <= {EXIT_RETURN:.2%}")
    print()
    print("Entrenando y evaluando por bloques temporales...")

    all_predictions = []
    all_test_indices = []
    all_test_targets = []
    all_test_volatilities = []
    all_losses = []
    block_scores = []
    block_maes = []

    for split_id, train_end, test_start, test_end in splits:
        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]

        print()
        print(f"--- BLOQUE {split_id}/{len(splits)} ---")
        print(f"Entrena con:             {len(X_train)} secuencias")
        print(f"Prueba con:              {len(X_test)} secuencias nuevas")

        model, scaler_x, scaler_y = train_network(X_train, y_train)
        predictions = predict_block(model, scaler_x, scaler_y, X_test)

        actual_direction = (y_test > 0).astype(int)
        predicted_direction = (predictions > 0).astype(int)
        directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
        mae = mean_absolute_error(y_test, predictions) * 100

        block_scores.append(directional_accuracy)
        block_maes.append(mae)
        all_predictions.extend(predictions.tolist())
        all_test_indices.extend(indices[test_start:test_end])
        all_test_targets.extend(y_test.tolist())
        all_test_volatilities.extend(volatilities[test_start:test_end].tolist())
        all_losses.append(model.loss_curve_)

        print(f"Épocas realizadas:       {model.n_iter_}")
        print(f"Pérdida final:            {model.loss_:.5f}")
        print(f"MAE OOS:                  {mae:.3f}%")
        print(f"Acierto direccional OOS: {directional_accuracy:.2f}%")

    predictions_all = np.asarray(all_predictions, dtype=float)
    y_test_all = np.asarray(all_test_targets, dtype=float)
    volatilities_all = np.asarray(all_test_volatilities, dtype=float)

    directional_accuracy = np.mean((predictions_all > 0) == (y_test_all > 0)) * 100
    mae_all = mean_absolute_error(y_test_all, predictions_all) * 100

    close = data.loc[all_test_indices, "Close"].astype(float)
    final_value, trade_count, equity = simulate_strategy(
        close, predictions_all, volatilities_all
    )
    buy_hold = INITIAL_CASH * (close.iloc[-1] / close.iloc[0])

    print()
    print("=== RESULTADO FINAL WALK-FORWARD ===")
    print(f"Valor final IA:          {final_value:.2f} €")
    print(f"Rentabilidad IA:         {(final_value / INITIAL_CASH - 1) * 100:.2f} %")
    print(f"Buy & Hold final:        {buy_hold:.2f} €")
    print(f"Operaciones:             {trade_count}")
    print(f"Operaciones/día aprox.:  {trade_count / max(len(close) / 26, 1):.2f}")
    print(f"MAE OOS:                  {mae_all:.3f}%")
    print(f"Acierto direccional OOS: {directional_accuracy:.2f}%")
    print(f"OOS por bloque:          {', '.join(f'{s:.1f}%' for s in block_scores)}")
    print(f"MAE por bloque:          {', '.join(f'{m:.3f}%' for m in block_maes)}")

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
