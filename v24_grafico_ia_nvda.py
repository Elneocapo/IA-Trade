"""V24 - Visualizador de velas + prediccion + operaciones de la IA.

Paper research only. No live orders.

Genera UNA SOLA VENTANA con:
1) Velas OHLC de la zona seleccionada.
2) Fondo por fase (validacion/test).
3) Senal continua de la IA y umbral usado.
4) Probabilidad alcista de la clasificacion.
5) Marcadores de ENTRADA y SALIDA ejecutados al siguiente OPEN.
6) Etiquetas con hora para inspeccionar especialmente la apertura.

No cambia ningun modelo ni ningun archivo anterior. Para la visualizacion usa
las predicciones originales de V16.7 y el umbral fijo actual 0.175%.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import v16_nvidia_1h_sequential as v16

THRESHOLD = 0.00175
MAX_HOLD = 6
WEIGHT = 1.0
COST = v16.COST
INITIAL_CASH = v16.INITIAL_CASH
PLOT_BARS = 350


def make_trades(df, panel, start, end):
    positions = {ts: i for i, ts in enumerate(df.index)}
    signals = panel[(panel.index >= start) & (panel.index < end)]
    cash = INITIAL_CASH
    shares = 0.0
    entry_i = None
    events = []
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
        want_long = float(row["signal"]) > THRESHOLD
        target_value = equity * WEIGHT if want_long else 0.0
        current_value = shares * px

        timed_exit = shares > 0 and entry_i is not None and execution_i - entry_i >= MAX_HOLD
        if timed_exit:
            target_value = 0.0

        if target_value < current_value * 0.98 and shares > 0:
            sell_value = min(current_value, current_value - target_value)
            sell_shares = min(shares, sell_value / px)
            proceeds = sell_shares * px * (1.0 - COST)
            cash += proceeds
            shares -= sell_shares
            if shares <= 1e-12:
                shares = 0.0
                events.append((execution_ts, "SELL", px, float(row["signal"]), float(row["prob_up"]), "time_stop" if timed_exit else "signal"))
                entry_i = None
        elif target_value > current_value * 1.02:
            desired_gross = min(target_value - current_value, cash / (1.0 + COST))
            if desired_gross > max(0.01, equity * 0.01):
                buy_shares = desired_gross / (px * (1.0 + COST))
                cash -= desired_gross
                shares += buy_shares
                if entry_i is None:
                    entry_i = execution_i
                    events.append((execution_ts, "BUY", px, float(row["signal"]), float(row["prob_up"]), "signal"))

        curve.append((execution_ts, cash + shares * px))

    return events, pd.Series(dict(curve)).sort_index()


def candle_ax(ax, data):
    x = mdates.date2num(data.index.to_pydatetime())
    # Infer width from median interval, avoiding zero width.
    if len(x) > 1:
        width = float(np.median(np.diff(x))) * 0.82
    else:
        width = 0.02

    for xi, (_, row) in zip(x, data.iterrows()):
        o, h, l, c = [float(row[k]) for k in ("Open", "High", "Low", "Close")]
        up = c >= o
        ax.vlines(xi, l, h, linewidth=0.8)
        bottom = min(o, c)
        height = max(abs(c - o), max(abs(c), 1.0) * 1e-5)
        rect = Rectangle((xi - width / 2, bottom), width, height,
                         fill=True, alpha=0.75,
                         linewidth=0.8,
                         edgecolor="black",
                         facecolor="white" if up else "black")
        ax.add_patch(rect)


def main():
    df = v16.load_data()
    n = len(df)
    train_cut = int(n * 0.60)
    val_cut = int(n * 0.80)

    print("Generando predicciones V16.7 para la visualizacion...", flush=True)
    panel = v16.sequential_predictions(df, train_cut, n - 1)
    if panel.empty:
        raise RuntimeError("No hay predicciones para visualizar")

    val_dates = panel.index[panel.index < df.index[val_cut]]
    test_dates = panel.index[panel.index >= df.index[val_cut + 2]]
    vs, ve = val_dates[0], val_dates[-1]
    ts, te = test_dates[0], test_dates[-1]

    # Mostramos las ultimas PLOT_BARS velas disponibles, incluyendo la zona de test.
    end = df.index[-1]
    start = df.index[max(0, len(df) - PLOT_BARS)]
    view = df.loc[start:end].copy()
    pv = panel.loc[(panel.index >= start) & (panel.index < end)].copy()
    if pv.empty:
        raise RuntimeError("La ventana elegida no contiene predicciones")

    events, equity = make_trades(df, panel, start, end)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(17, 10), sharex=True,
                                   gridspec_kw={"height_ratios": [3.6, 1.25]})
    fig.suptitle("IA-Trade V24 — NVDA 1H | velas + prediccion + operaciones", fontsize=15)

    candle_ax(ax1, view)
    ax1.set_ylabel("Precio NVDA")
    ax1.grid(alpha=0.18)

    # Mark opening window of every session subtly, useful to inspect the first 1H candle.
    for d in view.index.normalize().unique():
        if pd.isna(d):
            continue
        session_open = d + pd.Timedelta(hours=9, minutes=30)
        ax1.axvline(session_open, linewidth=0.6, alpha=0.18)

    # Entry/exit markers at the actual execution OPEN.
    for ts0, kind, px, sig, prob, reason in events:
        if kind == "BUY":
            ax1.scatter(ts0, px, marker="^", s=90, zorder=6, label="BUY" if "BUY" not in ax1.get_legend_handles_labels()[1] else "")
            ax1.annotate(f"BUY\n{ts0.strftime('%H:%M')}", (ts0, px), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=8)
        else:
            ax1.scatter(ts0, px, marker="v", s=90, zorder=6, label="SELL" if "SELL" not in ax1.get_legend_handles_labels()[1] else "")
            ax1.annotate(f"SELL\n{ts0.strftime('%H:%M')}", (ts0, px), xytext=(0, -28), textcoords="offset points", ha="center", fontsize=8)

    # Shade validation and test periods if visible.
    if ve >= start and vs <= end:
        ax1.axvspan(max(start, vs), min(end, ve), alpha=0.06, label="validacion")
    if te >= start and ts <= end:
        ax1.axvspan(max(start, ts), min(end, te), alpha=0.06, label="test")

    ax2.plot(pv.index, pv["signal"], linewidth=1.2, label="senal IA")
    ax2.axhline(THRESHOLD, linestyle="--", linewidth=1.0, label=f"umbral {THRESHOLD:.3%}")
    ax2.axhline(0, linewidth=0.7, alpha=0.5)
    ax2.set_ylabel("Señal")
    ax2.grid(alpha=0.18)

    ax2b = ax2.twinx()
    ax2b.plot(pv.index, pv["prob_up"], linewidth=0.9, alpha=0.55, label="prob_up")
    ax2b.axhline(0.5, linestyle=":", linewidth=0.8, alpha=0.6)
    ax2b.set_ylabel("Prob. alcista")
    ax2b.set_ylim(0, 1)

    # Opening markers in lower panel too, so time-of-day is obvious.
    for d in view.index.normalize().unique():
        session_open = d + pd.Timedelta(hours=9, minutes=30)
        if start <= session_open <= end:
            ax2.axvline(session_open, linewidth=0.6, alpha=0.18)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=8, maxticks=16))
    fig.autofmt_xdate(rotation=0)

    # Single combined legend from both axes.
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    h3, l3 = ax2b.get_legend_handles_labels()
    seen = set()
    hh, ll = [], []
    for h, l in list(zip(h1, l1)) + list(zip(h2, l2)) + list(zip(h3, l3)):
        if l and l not in seen:
            seen.add(l); hh.append(h); ll.append(l)
    ax1.legend(hh, ll, loc="upper left", ncol=3, fontsize=8)

    fig.text(0.01, 0.01,
             "▲ BUY / ▼ SELL = ejecucion al OPEN siguiente. Linea inferior = señal de IA; linea punteada = umbral. "
             "Lineas verticales = apertura de mercado 09:30 ET.", fontsize=9)
    plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    out = "v24_grafico_ia_nvda.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Grafico guardado en: {out}")
    print(f"Ventana: {start} -> {end} | velas={len(view)} | eventos={len(events)}")
    print("Se abre ahora la ventana de matplotlib. Cierra la ventana para terminar el programa.")
    plt.show()


if __name__ == "__main__":
    main()
