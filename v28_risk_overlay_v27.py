"""IA-Trade V28 - fase 2: reducir drawdown sobre V27 sin cambiar la señal.

Paper research only. No broker or live orders.

Base: V27. La señal de IA, las features y el entrenamiento NO cambian.
Solo se modifica el tamaño de la posicion segun el regimen de volatilidad
ya conocido en el cierre de la vela de señal. Los parametros del overlay se
seleccionan exclusivamente con VALIDACION; el TEST queda ciego.

Capital nuevo de investigacion: 500 EUR.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import v27_session_aware_v23_plus as v27

INITIAL_CASH = 500.0
COST = v27.COST
THRESHOLD = 0.00025  # benchmark V27 seleccionado con validacion; no se toca en esta fase
MAX_HOLD = v27.MAX_HOLD

# Risk overlay candidates. Same signal/trade timing; only exposure changes.
VOL_HIGH_CANDIDATES = (1.15, 1.30, 1.50, 1.75, 2.00)
MID_WEIGHT_CANDIDATES = (0.50, 0.65, 0.80)
HIGH_WEIGHT_CANDIDATES = (0.25, 0.40, 0.55)


def exposure_from_vol(vol_ratio: float, high_trigger: float, mid_weight: float, high_weight: float) -> float:
    if not np.isfinite(vol_ratio):
        return 1.0
    if vol_ratio >= high_trigger:
        return high_weight
    if vol_ratio >= 1.0:
        return mid_weight
    return 1.0


def backtest(df, panel, start, end, high_trigger, mid_weight, high_weight):
    positions = {ts: i for i, ts in enumerate(df.index)}
    signals = panel[(panel.index >= start) & (panel.index < end)]
    cash = float(INITIAL_CASH)
    shares = 0.0
    entry_i = None
    entry_outlay = None
    trade_returns = []
    curve = []

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
        long_signal = float(row["signal"]) > THRESHOLD
        exposure = exposure_from_vol(float(row["vol_ratio"]), high_trigger, mid_weight, high_weight) if long_signal else 0.0
        target_value = equity * exposure
        current_value = shares * px

        if shares > 0 and entry_i is not None and execution_i - entry_i >= MAX_HOLD:
            target_value = 0.0

        if target_value < current_value * 0.98 and shares > 0:
            sell_value = min(current_value, current_value - target_value)
            sell_shares = min(shares, sell_value / px)
            cash += sell_shares * px * (1.0 - COST)
            shares -= sell_shares
            if shares <= 1e-12:
                shares = 0.0
                if entry_outlay is not None and entry_outlay > 0:
                    trade_returns.append(cash / entry_outlay - 1.0)
                entry_i = None
                entry_outlay = None

        elif target_value > current_value * 1.02:
            desired = min(target_value - current_value, cash / (1.0 + COST))
            if desired > max(0.01, equity * 0.01):
                before = cash + shares * px
                shares += desired / (px * (1.0 + COST))
                cash -= desired
                if entry_i is None:
                    entry_i = execution_i
                    entry_outlay = before

        curve.append(cash + shares * px)

    if len(curve) < 2:
        return {"return": 0.0, "final": INITIAL_CASH, "trades": 0, "max_dd": 0.0, "sharpe": 0.0, "win_rate": 0.0}

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
        "max_dd": max_dd,
        "sharpe": sharpe,
        "win_rate": win_rate,
    }


def validation_score(r):
    # Prefer lower drawdown while preserving meaningful return and Sharpe.
    score = r["return"] - 0.65 * abs(min(r["max_dd"], 0.0)) + 0.025 * max(r["sharpe"], 0.0)
    if r["trades"] < 50:
        score -= 0.002 * (50 - r["trades"])
    return score


def main():
    t0 = time.time()
    # Reuse V27 exactly; only override simulation capital locally.
    v27.INITIAL_CASH = INITIAL_CASH
    df = v27.load_data()
    n = len(df)
    train_cut = int(n * 0.60)
    val_cut = int(n * 0.80)
    test_start = val_cut + 2

    print("=== V28 | V27 + risk overlay | NVDA 1h | capital=500 EUR ===", flush=True)
    print("Generando predicciones V27 sin cambios en el modelo/features...", flush=True)
    panel = v27.sequential_predictions(df, train_cut, n - 1)
    if panel.empty:
        raise RuntimeError("No predictions")

    val = panel.index[panel.index < df.index[val_cut]]
    test = panel.index[panel.index >= df.index[test_start]]
    vs, ve = val[0], val[-1]
    ts, te = test[0], test[-1]

    print(f"Validacion: {vs} -> {ve}")
    print(f"Test ciego: {ts} -> {te}")
    print(f"Threshold fijo V27={THRESHOLD:+.3%} | hold={MAX_HOLD} | coste/lado={COST:.3%}")

    results = []
    print("\n=== SWEEP RISK OVERLAY | SOLO VALIDACION ===")
    for hi in VOL_HIGH_CANDIDATES:
        for mid_w in MID_WEIGHT_CANDIDATES:
            for high_w in HIGH_WEIGHT_CANDIDATES:
                if high_w > mid_w:
                    continue
                r = backtest(df, panel, vs, ve, hi, mid_w, high_w)
                r.update(high_trigger=hi, mid_weight=mid_w, high_weight=high_w)
                r["score"] = validation_score(r)
                results.append(r)
                print(
                    f"vol_high={hi:.2f} | mid={mid_w:.2f} | high={high_w:.2f} | "
                    f"ret={r['return']:+.2%} | trades={r['trades']} | DD={r['max_dd']:.2%} | "
                    f"Sharpe={r['sharpe']:.2f} | WR={r['win_rate']:.1%} | score={r['score']:+.4f}"
                )

    best = max(results, key=lambda x: x["score"])
    print("\n=== OVERLAY ELEGIDO | VALIDACION ONLY ===")
    print(
        f"vol_high={best['high_trigger']:.2f} | mid_weight={best['mid_weight']:.2f} | "
        f"high_weight={best['high_weight']:.2f}"
    )
    print(
        f"VALIDACION: retorno={best['return']:+.2%} | final=€{best['final']:.2f} | "
        f"trades={best['trades']} | DD={best['max_dd']:.2%} | "
        f"Sharpe={best['sharpe']:.2f} | WR={best['win_rate']:.1%}"
    )

    print("\n=== TEST CIEGO | OVERLAY FIJO ===")
    test_r = backtest(df, panel, ts, te, best["high_trigger"], best["mid_weight"], best["high_weight"])
    bh_prices = df[(df.index >= ts) & (df.index < te)]["Open"].astype(float)
    bh = float(bh_prices.iloc[-1] / bh_prices.iloc[0] - 1.0) if len(bh_prices) > 1 else 0.0
    print(
        f"TEST IA: retorno={test_r['return']:+.2%} | final=€{test_r['final']:.2f} | "
        f"trades={test_r['trades']} | DD={test_r['max_dd']:.2%} | "
        f"Sharpe={test_r['sharpe']:.2f} | WR={test_r['win_rate']:.1%}"
    )
    print(f"TEST B&H: {bh:+.2%} | final=€{INITIAL_CASH*(1+bh):.2f}")
    print("Comparacion IA vs B&H:", "POSITIVA" if test_r["return"] > bh else "NEGATIVA")
    print(f"runtime={(time.time()-t0)/60:.1f} min")
    print("NO usar el resultado del test para reajustar el overlay. Si se cambia, repetir desde validacion.")


if __name__ == "__main__":
    main()
