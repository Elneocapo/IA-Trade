"""Backtest intradía de corto plazo con VectorBT.

Solo simulación: no conecta con brokers ni envía órdenes.
"""

from __future__ import annotations

import vectorbt as vbt

from config import (
    COMMISSION,
    INITIAL_CASH,
    INTRADAY_HORIZON_BARS,
    INTRADAY_INTERVAL,
    INTRADAY_PERIOD,
    INTRADAY_TEST_SIZE,
    INTRADAY_TRAIN_SIZE,
    TICKER,
)
from market import download_market_data
from walk_forward import ENTRY_THRESHOLD, EXIT_THRESHOLD, run_walk_forward


def main() -> None:
    print("=== IA-Trade | BACKTEST INTRADÍA DE CORTO PLAZO ===")
    print(f"Activo: {TICKER}")
    print(f"Datos: {INTRADAY_PERIOD} | {INTRADAY_INTERVAL}")
    print(f"Horizonte de la IA: {INTRADAY_HORIZON_BARS} velas ({INTRADAY_HORIZON_BARS * 15} min)")
    print("Modo: SIMULACIÓN / sin broker")
    print()

    data = download_market_data(
        TICKER,
        period=INTRADAY_PERIOD,
        interval=INTRADAY_INTERVAL,
    )

    prediction_frame, wf_stats = run_walk_forward(
        data,
        train_size=INTRADAY_TRAIN_SIZE,
        test_size=INTRADAY_TEST_SIZE,
        entry_threshold=ENTRY_THRESHOLD,
        exit_threshold=EXIT_THRESHOLD,
        horizon_bars=INTRADAY_HORIZON_BARS,
    )

    prices = data.loc[prediction_frame.index, "Close"].astype(float)
    probability_up = prediction_frame["probability_up"].astype(float)

    # La señal se conoce al cierre de una vela y se ejecuta como pronto en la siguiente.
    raw_entries = probability_up >= ENTRY_THRESHOLD
    raw_exits = probability_up < EXIT_THRESHOLD
    entries = raw_entries.shift(1, fill_value=False)
    exits = raw_exits.shift(1, fill_value=False)

    portfolio = vbt.Portfolio.from_signals(
        close=prices,
        entries=entries,
        exits=exits,
        init_cash=INITIAL_CASH,
        fees=COMMISSION,
        freq=INTRADAY_INTERVAL,
    )

    benchmark = vbt.Portfolio.from_holding(
        close=prices,
        init_cash=INITIAL_CASH,
        fees=COMMISSION,
        freq=INTRADAY_INTERVAL,
    )

    final_value = float(portfolio.value().iloc[-1])
    benchmark_value = float(benchmark.value().iloc[-1])
    return_pct = float(portfolio.total_return() * 100)
    benchmark_return_pct = float(benchmark.total_return() * 100)
    max_drawdown_pct = float(portfolio.max_drawdown() * 100)
    trades = int(portfolio.trades.count())

    print(f"Periodo real:            {data.index[0]} -> {data.index[-1]}")
    print(f"Velas descargadas:       {len(data)}")
    print(f"Señales OOS:             {len(prediction_frame)}")
    print(f"Ventana entrenamiento:   {INTRADAY_TRAIN_SIZE} velas")
    print(f"Ventana prueba:          {INTRADAY_TEST_SIZE} velas")
    print(f"Umbral entrada:          {ENTRY_THRESHOLD:.2f}")
    print(f"Umbral salida:           {EXIT_THRESHOLD:.2f}")
    print(f"Capital inicial:         {INITIAL_CASH:.2f} €")
    print()

    print("--- RESULTADO VECTORBT ---")
    print(f"Valor final IA:          {final_value:.2f} €")
    print(f"Rentabilidad IA:         {return_pct:.2f} %")
    print(f"Máximo drawdown IA:      {max_drawdown_pct:.2f} %")
    print(f"Operaciones:             {trades}")
    print()

    print("--- COMPARACIÓN ---")
    print(f"Buy & Hold final:        {benchmark_value:.2f} €")
    print(f"Buy & Hold rentabilidad: {benchmark_return_pct:.2f} %")
    advantage = final_value - benchmark_value
    print(f"Diferencia IA - B&H:     {advantage:+.2f} €")
    print()

    if trades == 0:
        print("🟡 La IA todavía no está generando suficientes entradas para evaluar una estrategia intradía.")
    elif advantage > 0:
        print("🟢 En esta muestra la estrategia IA supera a Buy & Hold.")
    else:
        print("🔴 En esta muestra la estrategia IA queda por debajo de Buy & Hold.")

    print()
    print("--- CALIDAD DE LA SEÑAL ---")
    print(f"Balanced accuracy OOS:  {wf_stats['balanced_accuracy_pct']:.2f}%")
    print("Las operaciones se generan únicamente con predicciones fuera de muestra.")
    print("Esto es investigación histórica, no una predicción de rentabilidad futura.")


if __name__ == "__main__":
    main()
