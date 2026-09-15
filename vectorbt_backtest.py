"""Backtest histórico con VectorBT sobre las señales out-of-sample de IA-Trade.

Solo simulación: no conecta con brokers ni envía órdenes.
"""

from __future__ import annotations

import numpy as np
import vectorbt as vbt

from config import COMMISSION, INITIAL_CASH, INTERVAL, PERIOD, TICKER
from market import download_market_data
from walk_forward import ENTRY_THRESHOLD, EXIT_THRESHOLD, run_walk_forward


def main() -> None:
    print("=== IA-Trade + VectorBT | BACKTEST HISTÓRICO ===")
    print(f"Activo: {TICKER}")
    print(f"Datos: {PERIOD} | {INTERVAL}")
    print("Modo: SIMULACIÓN / sin broker")
    print()

    data = download_market_data(TICKER, period=PERIOD, interval=INTERVAL)
    prediction_frame, wf_stats = run_walk_forward(data)

    prices = data.loc[prediction_frame.index, "Close"].astype(float)
    probability_up = prediction_frame["probability_up"].astype(float)

    # La señal se conoce al cierre y se ejecuta como pronto en la barra siguiente.
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
        freq=INTERVAL,
    )
    benchmark = vbt.Portfolio.from_holding(
        close=prices,
        init_cash=INITIAL_CASH,
        fees=COMMISSION,
        freq=INTERVAL,
    )

    final_value = float(portfolio.value().iloc[-1])
    benchmark_value = float(benchmark.value().iloc[-1])
    return_pct = float(portfolio.total_return() * 100)
    benchmark_return_pct = float(benchmark.total_return() * 100)
    max_drawdown_pct = float(portfolio.max_drawdown() * 100)
    trades = int(portfolio.trades.count())

    print(f"Señales OOS:             {len(prediction_frame)}")
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

    if advantage > 0:
        print("🟢 En este periodo la estrategia IA supera a Buy & Hold.")
    elif advantage < 0:
        print("🔴 En este periodo la estrategia IA queda por debajo de Buy & Hold.")
    else:
        print("🟡 En este periodo quedan prácticamente iguales.")

    print()
    print("Nota: VectorBT solo está evaluando las señales que el walk-forward generó fuera de muestra.")
    print(f"Balanced accuracy OOS del modelo: {wf_stats['balanced_accuracy_pct']:.2f}%")
    print("Esto es investigación histórica, no una predicción de rentabilidad futura.")


if __name__ == "__main__":
    main()
