"""V24 - Visualizador de velas + prediccion + operaciones de la IA.

Paper research only. No live orders.

Genera UNA SOLA VENTANA con:
1) Velas OHLC de la zona seleccionada.
2) Separacion visual por dias/sesiones.
3) Senal continua de la IA y umbral usado.
4) Probabilidad alcista de la clasificacion.
5) Marcadores de ENTRADA y SALIDA ejecutados al siguiente OPEN.
6) Etiquetas con hora para inspeccionar especialmente la apertura.

No guarda imagen en disco. Al cerrar la ventana imprime primero los datos
completos de referencia V16.7 y despues el detalle de la ventana visualizada.
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
    if len(x) > 1:
        width = float(np.median(np.diff(x))) * 0.92
    else:
        width = 0.02

    day_keys = data.index.normalize()
    days = pd.Index(day_keys).unique()
    day_to_num = {day: idx for idx, day in enumerate(days)}

    for xi, (ts, row) in zip(x, data.iterrows()):
        o, h, l, c = [float(row[k]) for k in ("Open", "High", "Low", "Close")]
        up = c >= o
        day_idx = day_to_num[ts.normalize()]
        alpha = 0.78 if day_idx % 2 == 0 else 0.92
        ax.vlines(xi, l, h, linewidth=0.8, alpha=alpha)
        bottom = min(o, c)
        height = max(abs(c - o), max(abs(c), 1.0) * 1e-5)
        rect = Rectangle((xi - width / 2, bottom), width, height,
                         fill=True, alpha=alpha,
                         linewidth=0.65,
                         edgecolor="black",
                         facecolor="white" if up else "black")
        ax.add_patch(rect)


def print_reference_summary(df, panel, vs, ve, ts, te):
    """Print the same high-level figures shown by V16.7, but with fixed V24 params."""
    print("\n" + "=" * 68)
    print("DATOS DE REFERENCIA — MODELO V16.7 / CONFIGURACION V24")
    print("=" * 68)
    print(f"asset={v16.ASSET} | candles={v16.INTERVAL} | history={v16.PERIOD}")
    print(f"context={v16.LOOKBACK_BARS} bars (~10 trading days) | prediction=NEXT 1H candle")
    print(f"max_hold usado en V24={MAX_HOLD} barras (~{MAX_HOLD/6.5:.1f} trading days)")
    print(f"data={df.index[0]} -> {df.index[-1]} | total_bars={len(df)}")
    print(f"validation={vs} -> {ve}")
    print(f"test={ts} -> {te}")
    print(f"config V24: threshold={THRESHOLD:.2%} | max_weight={WEIGHT:.0%} | coste/lado={COST:.3%}")
    print("NOTA: V24 no vuelve a seleccionar parametros; usa el umbral fijo 0.175% para inspeccion visual.")
    print("=" * 68)


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

    unique_days = list(view.index.normalize().unique())
    for day_idx, d in enumerate(unique_days):
        day_data = view.loc[view.index.normalize() == d]
        if day_data.empty:
            continue
        day_start = day_data.index[0]
        day_end = day_data.index[-1]
        if day_idx % 2 == 1:
            ax1.axvspan(day_start, day_end + pd.Timedelta(minutes=30), alpha=0.055)
        session_open = d + pd.Timedelta(hours=9, minutes=30)
        if start <= session_open <= end:
            ax1.axvline(session_open, linewidth=0.7, alpha=0.22)

    for ts0, kind, px, sig, prob, reason in events:
        if kind == "BUY":
            ax1.scatter(ts0, px, marker="^", s=90, zorder=6, label="BUY" if "BUY" not in ax1.get_legend_handles_labels()[1] else "")
            ax1.annotate(f"BUY\n{ts0.strftime('%H:%M')}", (ts0, px), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=8)
        else:
            ax1.scatter(ts0, px, marker="v", s=90, zorder=6, label="SELL" if "SELL" not in ax1.get_legend_handles_labels()[1] else "")
            ax1.annotate(f"SELL\n{ts0.strftime('%H:%M')}", (ts0, px), xytext=(0, -28), textcoords="offset points", ha="center", fontsize=8)

    if ve >= start and vs <= end:
        ax1.axvspan(max(start, vs), min(end, ve), alpha=0.04, label="validacion")
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

    for d in unique_days:
        session_open = d + pd.Timedelta(hours=9, minutes=30)
        if start <= session_open <= end:
            ax2.axvline(session_open, linewidth=0.7, alpha=0.22)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=8, maxticks=16))
    fig.autofmt_xdate(rotation=0)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    h3, l3 = ax2b.get_legend_handles_labels()
    seen = set(); hh = []; ll = []
    for h, l in list(zip(h1, l1)) + list(zip(h2, l2)) + list(zip(h3, l3)):
        if l and l not in seen:
            seen.add(l); hh.append(h); ll.append(l)
    ax1.legend(hh, ll, loc="upper left", ncol=3, fontsize=8)

    fig.text(0.01, 0.01,
             "▲ BUY / ▼ SELL = ejecucion al OPEN siguiente. Linea inferior = señal IA. "
             "Fondo alterno = dia distinto. Lineas verticales = apertura 09:30 ET.", fontsize=9)
    plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    print(f"Ventana: {start} -> {end} | velas={len(view)} | eventos={len(events)}")
    print("Se abre ahora la ventana de matplotlib. Cierra la ventana para ver TODOS los datos en consola.")
    plt.show()

    # First print the full-model / experiment context, then the detailed visual window.
    print_reference_summary(df, panel, vs, ve, ts, te)

    print("\n" + "=" * 68)
    print("RESUMEN V24 — DATOS DE LA VENTANA")
    print("=" * 68)
    print(f"Periodo mostrado: {start} -> {end}")
    print(f"Velas mostradas: {len(view)}")
    print(f"Eventos BUY/SELL: {len(events)}")
    print(f"Umbral IA: {THRESHOLD:.3%}")
    print(f"Max hold: {MAX_HOLD} barras")
    print(f"Coste por lado: {COST:.3%}")
    if events:
        buys = sum(1 for e in events if e[1] == "BUY")
        sells = sum(1 for e in events if e[1] == "SELL")
        print(f"Compras: {buys} | Ventas: {sells}")
        print("\nEventos:")
        for ts0, kind, px, sig, prob, reason in events:
            print(f"  {ts0} | {kind:4s} | precio_open={px:.2f} | señal={sig:+.5f} | prob_up={prob:.1%} | motivo={reason}")
    else:
        print("No hubo operaciones en la ventana mostrada.")
    if len(equity) > 1:
        final_equity = float(equity.iloc[-1])
        ret = final_equity / INITIAL_CASH - 1.0
        peak = equity.cummax()
        dd = float((equity / peak - 1.0).min())
        print(f"\nCartera en ventana: €{INITIAL_CASH:.2f} -> €{final_equity:.2f} ({ret:+.2%})")
        print(f"Max drawdown de la ventana: {dd:.2%}")
    else:
        print("No hay suficiente curva de equity para calcular retorno/DD.")
    print("=" * 68)


if __name__ == "__main__":
    main()
