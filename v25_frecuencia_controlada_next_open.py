"""IA-Trade V25 - Umbral con requisito de frecuencia, validacion-only.

Paper research only. No live orders.

Objetivo:
- Reutilizar exactamente las predicciones de V16.7.
- Ejecutar al next-open y contabilizar costes como V22.
- Elegir umbral SOLO con VALIDACION.
- Imponer un suelo explicito de actividad para no acabar con una estrategia
  que apenas opera.
- El TEST se muestra una sola vez como comprobacion ciega.

NOTA: El minimo de operaciones es una restriccion de investigacion, no una
afirmacion de que 30 operaciones sea el numero correcto para una estrategia.
Se podra cambiar en otra ronda, pero siempre antes de mirar el test de esa ronda.
"""
from __future__ import annotations

import time
import v22_umbral_next_open as v22

# Exploratory activity constraint: at least 30 closed trades in validation.
MIN_TRADES = 30

# Keep the same model, hold and weight as V16.7/V22.
MAX_HOLD = v22.MAX_HOLD
WEIGHT = v22.WEIGHT
COST = v22.COST
INITIAL_CASH = v22.INITIAL_CASH
THRESHOLDS = v22.THRESHOLDS


def main():
    overall = time.time()
    df = v22.v16.load_data()
    n = len(df)
    train_cut = int(n * 0.60)
    val_cut = int(n * 0.80)
    test_start = val_cut + 2

    print("=== V25 | NVDA | 1h | frecuencia minima en VALIDACION ===", flush=True)
    print("V25: generando predicciones V16.7 (modelo sin cambios)...", flush=True)
    panel = v22.v16.sequential_predictions(df, train_cut, n - 1)
    if panel.empty:
        raise RuntimeError("No se pudieron generar predicciones")

    val_dates = panel.index[panel.index < df.index[val_cut]]
    test_dates = panel.index[panel.index >= df.index[test_start]]
    if len(val_dates) == 0 or len(test_dates) == 0:
        raise RuntimeError("No hay suficientes predicciones para validacion/test")

    vs, ve = val_dates[0], val_dates[-1]
    ts, te = test_dates[0], test_dates[-1]

    print(f"Validacion: {vs} -> {ve}")
    print(f"Test ciego: {ts} -> {te}")
    print(f"Coste por lado={COST:.3%} | hold={MAX_HOLD} | weight={WEIGHT:.0%}")
    print(f"Suelo de actividad: >= {MIN_TRADES} trades cerrados en validacion")

    results = []
    print("\n=== CANDIDATOS DE UMBRAL | SOLO VALIDACION ===")
    for threshold in THRESHOLDS:
        r = v22.backtest_next_open(df, panel, vs, ve, threshold)
        r["threshold"] = threshold
        r["score"] = v22.score_validation(r)
        results.append(r)
        eligible = r["trades"] >= MIN_TRADES
        print(
            f"threshold={threshold:+.3%} | ret={r['return']:+.2%} | trades={r['trades']} | "
            f"DD={r['max_dd']:.2%} | Sharpe={r['sharpe']:.2f} | WR={r['win_rate']:.1%} | "
            f"eligible={'YES' if eligible else 'NO'} | score={r['score']:+.4f}"
        )

    eligible = [r for r in results if r["trades"] >= MIN_TRADES]
    if not eligible:
        raise RuntimeError("Ningun umbral alcanza el minimo de operaciones en validacion")

    best = max(eligible, key=lambda x: x["score"])
    threshold = best["threshold"]

    print("\n=== UMBRAL ELEGIDO V25 ===")
    print(f"threshold={threshold:+.3%} | elegido SOLO con validacion y >= {MIN_TRADES} trades")
    print(
        f"VALIDACION: retorno={best['return']:+.2%} | final=€{best['final']:.2f} | "
        f"trades={best['trades']} | maxDD={best['max_dd']:.2%} | "
        f"Sharpe={best['sharpe']:.2f} | win_rate={best['win_rate']:.1%}"
    )

    print("\n=== TEST CIEGO | UMBRAL FIJO ===")
    test = v22.backtest_next_open(df, panel, ts, te, threshold)
    bh = v22.buy_and_hold(df, ts, te)
    print(
        f"TEST IA: retorno={test['return']:+.2%} | final=€{test['final']:.2f} | "
        f"trades={test['trades']} | maxDD={test['max_dd']:.2%} | "
        f"Sharpe={test['sharpe']:.2f} | win_rate={test['win_rate']:.1%}"
    )
    print(f"TEST B&H: {bh:+.2%} | final=€{INITIAL_CASH*(1+bh):.2f}")
    print("Comparacion IA vs B&H:", "POSITIVA" if test["return"] > bh else "NEGATIVA")
    print(f"runtime={(time.time()-overall)/60:.1f} min")
    print("\nIMPORTANTE: este test queda ciego. No usarlo para cambiar MIN_TRADES ni threshold.")


if __name__ == "__main__":
    main()
