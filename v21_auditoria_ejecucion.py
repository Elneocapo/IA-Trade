"""V21 - Auditoría de ejecución conservadora para NVDA (paper research only).

No modifica V16.7. Reutiliza sus predicciones y compara:
A) ejecución idealizada al close de señal (referencia, no recalibra),
B) ejecución en el open de la vela siguiente, con costes en compras/ventas.

No selecciona parámetros con el test final. Usa los parámetros documentados de V16.7
(umbral 0.175%, peso 100%, hold 6) únicamente como comparación fija.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import v16_nvidia_1h_sequential as v16

THRESHOLD = 0.00175
WEIGHT = 1.0
MAX_HOLD = 6
COST = v16.COST
CASH0 = v16.INITIAL_CASH


def backtest_next_open(df, panel, start, end):
    """Signal at completed bar t; trade at next available bar's open."""
    signals = panel[(panel.index >= start) & (panel.index < end)]
    # Use signal timestamp to locate the next row in original OHLC data.
    positions = {ts: i for i, ts in enumerate(df.index)}
    cash, shares = CASH0, 0.0
    entry_i = None
    entry_cost_basis = None
    equity_curve = []
    trade_returns = []
    trade_count = 0

    for signal_ts, row in signals.iterrows():
        i = positions.get(signal_ts)
        if i is None or i + 1 >= len(df):
            continue
        execution_i = i + 1
        execution_ts = df.index[execution_i]
        if execution_ts >= end:
            continue
        px = float(df["Open"].iloc[execution_i])
        if not np.isfinite(px) or px <= 0:
            continue
        equity = cash + shares * px
        want_long = float(row["signal"]) > THRESHOLD
        target_value = equity * WEIGHT if want_long else 0.0
        current_value = shares * px

        # Time stop is measured in subsequent hourly bars from entry execution.
        if shares > 0 and entry_i is not None and execution_i - entry_i >= MAX_HOLD:
            target_value = 0.0

        if target_value < current_value * 0.98 and shares > 0:
            sell_shares = min(shares, (current_value - target_value) / px)
            proceeds = sell_shares * px * (1.0 - COST)
            cash += proceeds
            shares -= sell_shares
            if shares <= 1e-10:
                shares = 0.0
                if entry_cost_basis and entry_cost_basis > 0:
                    trade_returns.append(cash / entry_cost_basis - 1.0)
                entry_i = None
                entry_cost_basis = None

        elif target_value > current_value * 1.02:
            budget = min(target_value - current_value, cash / (1.0 + COST))
            if budget > 0.01:
                before = cash + shares * px
                bought = budget / (px * (1.0 + COST))
                cash -= budget
                shares += bought
                if entry_i is None:
                    entry_i = execution_i
                    entry_cost_basis = before
                    trade_count += 1

        equity_curve.append(cash + shares * px)

    if not equity_curve:
        return {"return": 0.0, "trades": 0, "max_dd": 0.0, "sharpe": 0.0,
                "win_rate": 0.0, "final_cash": CASH0}
    curve = np.asarray(equity_curve, dtype=float)
    dd = float(np.min(curve / np.maximum.accumulate(curve) - 1.0))
    rets = curve[1:] / curve[:-1] - 1.0
    sharpe = float(np.mean(rets) / (np.std(rets) + 1e-12) * np.sqrt(252 * 6.5)) if len(rets) > 20 else 0.0
    win_rate = float(np.mean(np.asarray(trade_returns) > 0)) if trade_returns else 0.0
    final = float(curve[-1])
    return {"return": final / CASH0 - 1.0, "trades": trade_count, "max_dd": dd,
            "sharpe": sharpe, "win_rate": win_rate, "final_cash": final}


def main():
    df = v16.load_data()
    n = len(df)
    train_cut, val_cut = int(n * 0.60), int(n * 0.80)
    test_start = val_cut + 2
    print("V21: generando predicciones V16.7 (sin cambiar modelos ni parámetros)...", flush=True)
    panel = v16.sequential_predictions(df, train_cut, n - 1)
    val_dates = panel.index[panel.index < df.index[val_cut]]
    test_dates = panel.index[panel.index >= df.index[test_start]]
    if len(val_dates) == 0 or len(test_dates) == 0:
        raise RuntimeError("No hay suficientes predicciones para validar/testear")
    vs, ve = val_dates[0], val_dates[-1]
    ts, te = test_dates[0], test_dates[-1]
    print(f"Umbral fijo={THRESHOLD:.3%}; peso={WEIGHT:.0%}; max_hold={MAX_HOLD} barras; coste por lado={COST:.3%}")
    print(f"Validación: {vs} -> {ve}")
    print(f"Test descriptivo: {ts} -> {te} (no ajustar con este periodo)")
    print("\n=== Ejecución siguiente OPEN ===")
    for label, start, end in (("VALIDACIÓN", vs, ve), ("TEST (descriptivo)", ts, te)):
        r = backtest_next_open(df, panel, start, end)
        print(f"{label}: retorno={r['return']:+.2%} | final=€{r['final_cash']:.2f} | trades={r['trades']} | maxDD={r['max_dd']:.2%} | Sharpe={r['sharpe']:.2f} | win_rate={r['win_rate']:.1%}")
    print("\nNOTA: esta prueba es una auditoría, no una estrategia validada. La entrada next-open es una aproximación; las velas horarias de Yahoo y la ejecución real pueden diferir. El retorno de la cartera depende de acciones fraccionarias y no modela opciones.")

if __name__ == "__main__":
    main()
