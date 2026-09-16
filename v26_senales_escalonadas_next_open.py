"""IA-Trade V26 - Señales escalonadas con ejecución next-open.

Paper research only. No live orders.

Idea:
- No bajar simplemente el umbral hasta generar ruido.
- Convertir la intensidad de la señal en tamaño de posición:
  * señal >= ENTRY_STRONG: 100%
  * señal >= ENTRY_WEAK: 50%
  * señal < ENTRY_WEAK: 0%
- La salida es 0% cuando la señal deja de justificar la posición.
- Los umbrales de entrada se seleccionan SOLO con validación.
- El test final queda ciego.

Esto es un experimento separado; no modifica V16.7.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import v16_nvidia_1h_sequential as v16

WEAK_CANDIDATES = (0.0005, 0.00075, 0.0010, 0.00125, 0.0015)
STRONG_GAP_CANDIDATES = (0.0005, 0.00075, 0.0010, 0.00125, 0.0015)
MAX_HOLD = 6
COST = v16.COST
INITIAL_CASH = v16.INITIAL_CASH


def backtest(df, panel, start, end, weak, strong):
    positions = {ts: i for i, ts in enumerate(df.index)}
    signals = panel[(panel.index >= start) & (panel.index < end)]
    cash = INITIAL_CASH
    shares = 0.0
    entry_i = None
    curve = []
    trade_returns = []
    entry_cash = None
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

        equity = cash + shares * px
        sig = float(row["signal"])
        # Stepped exposure. Strong signals receive full size; moderate signals half.
        if sig >= strong:
            target_weight = 1.0
        elif sig >= weak:
            target_weight = 0.50
        else:
            target_weight = 0.0

        if shares > 0 and entry_i is not None and execution_i - entry_i >= MAX_HOLD:
            target_weight = 0.0

        target_value = target_weight * equity
        current_value = shares * px

        if target_value < current_value * 0.98 and shares > 0:
            sell_value = min(current_value, current_value - target_value)
            sell_shares = min(shares, sell_value / px)
            cash += sell_shares * px * (1.0 - COST)
            shares -= sell_shares
            if shares <= 1e-12:
                shares = 0.0
                if entry_cash is not None and entry_cash > 0:
                    trade_returns.append(cash / entry_cash - 1.0)
                entry_i = None
                entry_cash = None

        elif target_value > current_value * 1.02:
            desired_gross = min(target_value - current_value, cash / (1.0 + COST))
            if desired_gross > max(0.01, equity * 0.01):
                cash -= desired_gross
                shares += desired_gross / (px * (1.0 + COST))
                if entry_i is None:
                    entry_i = execution_i
                    entry_cash = equity
                    entries += 1

        curve.append((execution_ts, cash + shares * px))

    if not curve:
        return dict(return_=0, final=INITIAL_CASH, trades=0, max_dd=0, sharpe=0, win_rate=0, entries=0)
    eq = pd.Series(dict(curve)).sort_index().astype(float)
    arr = eq.to_numpy()
    peak = np.maximum.accumulate(arr)
    max_dd = float(np.min(arr / peak - 1.0))
    rets = arr[1:] / arr[:-1] - 1
    sharpe = float(np.mean(rets) / (np.std(rets) + 1e-12) * np.sqrt(252 * 6.5)) if len(rets) > 20 else 0.0
    wr = float(np.mean(np.asarray(trade_returns) > 0)) if trade_returns else 0.0
    final = float(arr[-1])
    return dict(return_=final / INITIAL_CASH - 1, final=final, trades=len(trade_returns), max_dd=max_dd, sharpe=sharpe, win_rate=wr, entries=entries)


def score(r):
    # Return/risk first; activity is only a tie-breaker, not a reward in itself.
    s = r["return_"] - 0.30 * abs(min(r["max_dd"], 0)) + 0.015 * max(r["sharpe"], 0)
    if r["trades"] < 15:
        s -= 0.004 * (15 - r["trades"])
    return s


def main():
    t0 = time.time()
    df = v16.load_data()
    n = len(df)
    train_cut = int(n * 0.60)
    val_cut = int(n * 0.80)
    test_start = val_cut + 2
    print("=== V26 | señales escalonadas | NVDA 1h | next-open ===", flush=True)
    print("Generando predicciones V16.7 sin cambiar el modelo...", flush=True)
    panel = v16.sequential_predictions(df, train_cut, n - 1)
    val = panel.index[panel.index < df.index[val_cut]]
    test = panel.index[panel.index >= df.index[test_start]]
    if len(val) == 0 or len(test) == 0:
        raise RuntimeError("No hay predicciones suficientes")
    vs, ve = val[0], val[-1]
    ts, te = test[0], test[-1]

    candidates = []
    for weak in WEAK_CANDIDATES:
        for gap in STRONG_GAP_CANDIDATES:
            strong = weak + gap
            r = backtest(df, panel, vs, ve, weak, strong)
            r.update(weak=weak, strong=strong, score=score(r))
            candidates.append(r)
            print(f"weak={weak:+.3%} | strong={strong:+.3%} | ret={r['return_']:+.2%} | trades={r['trades']} | DD={r['max_dd']:.2%} | Sharpe={r['sharpe']:.2f} | WR={r['win_rate']:.1%} | score={r['score']:+.4f}")

    best = max(candidates, key=lambda x: x["score"])
    print("\n=== CONFIGURACION ELEGIDA (VALIDACION SOLAMENTE) ===")
    print(f"weak={best['weak']:+.3%} | strong={best['strong']:+.3%}")
    print(f"VALIDACION: retorno={best['return_']:+.2%} | final=€{best['final']:.2f} | trades={best['trades']} | DD={best['max_dd']:.2%} | Sharpe={best['sharpe']:.2f} | WR={best['win_rate']:.1%}")

    test_r = backtest(df, panel, ts, te, best["weak"], best["strong"])
    bh_p = df[(df.index >= ts) & (df.index < te)]["Open"].astype(float)
    bh = float(bh_p.iloc[-1] / bh_p.iloc[0] - 1) if len(bh_p) > 1 else 0.0
    print("\n=== TEST CIEGO ===")
    print(f"TEST IA: retorno={test_r['return_']:+.2%} | final=€{test_r['final']:.2f} | trades={test_r['trades']} | DD={test_r['max_dd']:.2%} | Sharpe={test_r['sharpe']:.2f} | WR={test_r['win_rate']:.1%}")
    print(f"TEST B&H: {bh:+.2%} | final=€{INITIAL_CASH*(1+bh):.2f}")
    print(f"Comparacion IA vs B&H: {'POSITIVA' if test_r['return_'] > bh else 'NEGATIVA'}")
    print(f"runtime={(time.time()-t0)/60:.1f} min")
    print("NO reajustar los umbrales usando el test. Si cambiamos el objetivo, hay que repetir desde validacion.")


if __name__ == "__main__":
    main()
