"""IA-Trade V22 - Umbral seleccionado con ejecución next-open.

Paper research only. No broker or live orders.

Objetivo:
- Mantener el modelo/predicciones de V16.7 sin cambios.
- Mantener max_hold=6 y peso=100% para aislar el efecto del umbral.
- Seleccionar el umbral EXCLUSIVAMENTE en VALIDACIÓN.
- Ejecutar la señal en el OPEN de la vela siguiente.
- Contabilizar costes correctamente en compra y venta.
- Dejar el TEST completamente fuera de la selección.

Importante: esto NO prueba todavía que exista una ventaja robusta. El test sirve
solo como comprobación ciega del umbral elegido con validación.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
import v16_nvidia_1h_sequential as v16

# V16.7/V21 constants kept fixed to isolate threshold effect.
THRESHOLDS = (-0.0005, 0.0, 0.00025, 0.0005, 0.00075, 0.0010,
              0.00125, 0.0015, 0.00175, 0.0020, 0.0025, 0.0030, 0.0040)
MAX_HOLD = 6
WEIGHT = 1.0
COST = v16.COST
INITIAL_CASH = v16.INITIAL_CASH


def backtest_next_open(df, panel, start, end, threshold):
    """Signal at completed bar t; entry/exit at next hourly OPEN.

    Accounting is based on actual cash flows, including both sides' transaction costs.
    A trade is considered closed only when the share position reaches zero.
    """
    positions = {ts: i for i, ts in enumerate(df.index)}
    signals = panel[(panel.index >= start) & (panel.index < end)]

    cash = float(INITIAL_CASH)
    shares = 0.0
    entry_i = None
    entry_cash_outlay = None
    trade_returns = []
    curve = []
    entries = 0

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

        equity_before = cash + shares * px
        want_long = float(row["signal"]) > threshold
        target_value = equity_before * WEIGHT if want_long else 0.0
        current_value = shares * px

        # Time stop measured in hourly bars from actual entry execution.
        if shares > 0 and entry_i is not None and execution_i - entry_i >= MAX_HOLD:
            target_value = 0.0

        # Sell down to target. Cost is charged on proceeds.
        if target_value < current_value * 0.98 and shares > 0:
            sell_value = min(current_value, current_value - target_value)
            sell_shares = min(shares, sell_value / px)
            proceeds = sell_shares * px * (1.0 - COST)
            cash += proceeds
            shares -= sell_shares

            if shares <= 1e-12:
                shares = 0.0
                if entry_cash_outlay is not None and entry_cash_outlay > 0:
                    # Net closed-trade return: final cash received / cash committed at entry.
                    trade_returns.append(cash / entry_cash_outlay - 1.0)
                entry_i = None
                entry_cash_outlay = None

        # Buy up to target. Cost is paid on the gross purchase value.
        elif target_value > current_value * 1.02:
            desired_gross = min(target_value - current_value, cash / (1.0 + COST))
            if desired_gross > max(0.01, equity_before * 0.01):
                gross_cost = desired_gross
                buy_shares = gross_cost / (px * (1.0 + COST))
                total_cash_spent = gross_cost
                cash -= total_cash_spent
                shares += buy_shares

                if entry_i is None:
                    entry_i = execution_i
                    entry_cash_outlay = equity_before
                    entries += 1

        # Mark to market at this actual execution price.
        curve.append(cash + shares * px)

    if not curve:
        return {"return": 0.0, "final": INITIAL_CASH, "trades": 0,
                "max_dd": 0.0, "sharpe": 0.0, "win_rate": 0.0}

    curve = np.asarray(curve, dtype=float)
    peak = np.maximum.accumulate(curve)
    max_dd = float(np.min(curve / peak - 1.0))
    rets = curve[1:] / curve[:-1] - 1.0
    sharpe = float(np.mean(rets) / (np.std(rets) + 1e-12) * np.sqrt(252 * 6.5)) if len(rets) > 20 else 0.0
    win_rate = float(np.mean(np.asarray(trade_returns) > 0)) if trade_returns else 0.0
    final = float(curve[-1])

    return {
        "return": final / INITIAL_CASH - 1.0,
        "final": final,
        "trades": len(trade_returns),
        "entries": entries,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "win_rate": win_rate,
    }


def score_validation(r):
    """Conservative validation score: return first, risk second, frequency only lightly."""
    score = r["return"]
    score -= 0.30 * abs(min(r["max_dd"], 0.0))
    score += 0.015 * max(r["sharpe"], 0.0)
    # Penalize very sparse validation samples without rewarding brute-force frequency.
    if r["trades"] < 8:
        score -= 0.004 * (8 - r["trades"])
    return score


def buy_and_hold(df, start, end):
    p = df[(df.index >= start) & (df.index < end)]["Open"].astype(float)
    return float(p.iloc[-1] / p.iloc[0] - 1.0) if len(p) > 1 else 0.0


def main():
    overall = time.time()
    df = v16.load_data()
    n = len(df)
    train_cut = int(n * 0.60)
    val_cut = int(n * 0.80)
    test_start = val_cut + 2

    print("=== V22 | NVDA | 1h | threshold selection on VALIDATION only ===", flush=True)
    print("V22: generando predicciones V16.7 (modelo sin cambios)...", flush=True)
    panel = v16.sequential_predictions(df, train_cut, n - 1)
    if panel.empty:
        raise RuntimeError("No se pudieron generar predicciones")

    val_dates = panel.index[panel.index < df.index[val_cut]]
    test_dates = panel.index[panel.index >= df.index[test_start]]
    if len(val_dates) == 0 or len(test_dates) == 0:
        raise RuntimeError("No hay suficientes predicciones para validación/test")

    vs, ve = val_dates[0], val_dates[-1]
    ts, te = test_dates[0], test_dates[-1]

    print(f"Validación: {vs} -> {ve}")
    print(f"Test ciego/descriptivo: {ts} -> {te}")
    print(f"Coste por lado={COST:.3%} | hold={MAX_HOLD} | weight={WEIGHT:.0%}")
    print("\n=== SWEEP DE UMBRAL (SOLO VALIDACIÓN) ===")

    results = []
    for threshold in THRESHOLDS:
        r = backtest_next_open(df, panel, vs, ve, threshold)
        r["threshold"] = threshold
        r["score"] = score_validation(r)
        results.append(r)
        print(
            f"threshold={threshold:+.3%} | ret={r['return']:+.2%} | "
            f"trades={r['trades']} | DD={r['max_dd']:.2%} | "
            f"Sharpe={r['sharpe']:.2f} | WR={r['win_rate']:.1%} | score={r['score']:+.4f}"
        )

    best = max(results, key=lambda x: x["score"])
    threshold = best["threshold"]

    print("\n=== UMBRAL ELEGIDO ===")
    print(f"threshold={threshold:+.3%} (elegido únicamente con validación)")
    print(
        f"VALIDACIÓN: retorno={best['return']:+.2%} | final=€{best['final']:.2f} | "
        f"trades={best['trades']} | maxDD={best['max_dd']:.2%} | "
        f"Sharpe={best['sharpe']:.2f} | win_rate={best['win_rate']:.1%}"
    )

    print("\n=== TEST CIEGO | UMBRAL FIJO ===")
    test = backtest_next_open(df, panel, ts, te, threshold)
    bh = buy_and_hold(df, ts, te)
    print(
        f"TEST IA: retorno={test['return']:+.2%} | final=€{test['final']:.2f} | "
        f"trades={test['trades']} | maxDD={test['max_dd']:.2%} | "
        f"Sharpe={test['sharpe']:.2f} | win_rate={test['win_rate']:.1%}"
    )
    print(f"TEST B&H (open→open, descriptivo): {bh:+.2%} | final=€{INITIAL_CASH*(1+bh):.2f}")
    print("Comparación IA vs B&H:", "POSITIVA" if test["return"] > bh else "NEGATIVA")
    print(f"runtime={(time.time()-overall)/60:.1f} min")
    print("\nNO AJUSTAR NINGÚN PARÁMETRO DESPUÉS DE VER ESTE TEST. Si se cambia el umbral, hay que repetir el ciclo desde validación.")


if __name__ == "__main__":
    main()
