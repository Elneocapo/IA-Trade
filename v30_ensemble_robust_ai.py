"""IA-Trade V30 - ensemble robusto de la IA de NVDA 1H.

PAPER RESEARCH ONLY. NO BROKER. NO LIVE ORDERS.

V30 no reemplaza V27 con una mega-optimización. Añade diversidad controlada:
- dos familias de regresores HGB con regularización distinta;
- dos clasificadores que evalúan dirección a 1 y 3 barras;
- consenso entre horizontes 1/3/6 barras;
- validación walk-forward idéntica a V27;
- selección de threshold + consenso SOLO con VALIDACIÓN;
- TEST completamente ciego.

Objetivo: mejorar la robustez de la señal antes de convertirla en opciones.
Capital de investigación: 500 EUR.
"""
from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingRegressor

warnings.filterwarnings("ignore")

ASSET = "NVDA"
PERIOD = "2y"
INTERVAL = "1h"
LOOKBACK_BARS = 65
TRAIN_WINDOW = 1200
RETRAIN_EVERY = 24
MODEL_MAX_ITER = 120
INITIAL_CASH = 500.0
COST = 0.001
MAX_HOLD = 6
THRESHOLD_CANDIDATES = (0.0, 0.00025, 0.0005, 0.00075, 0.001, 0.00125, 0.0015, 0.00175, 0.002, 0.0025, 0.003)
CONSENSUS_CANDIDATES = (0.50, 0.60, 0.70, 0.80)


def load_data() -> pd.DataFrame:
    print(f"=== V30 | {ASSET} | {INTERVAL} | ensemble robusto ===", flush=True)
    df = yf.download(ASSET, period=PERIOD, interval=INTERVAL, auto_adjust=True, progress=False, prepost=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ASSET}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy().dropna()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert("America/New_York")
    return df.between_time("09:30", "16:00")


def features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"].astype(float)
    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    v = df["Volume"].astype(float)
    r = np.log(c).diff()
    z = pd.DataFrame(index=df.index)

    z["r1"] = r
    z["body"] = (c - o) / (o + 1e-9)
    z["range"] = (h - l) / (c + 1e-9)
    z["loc"] = (c - l) / (h - l + 1e-9)
    z["upper"] = (h - np.maximum(o, c)) / (c + 1e-9)
    z["lower"] = (np.minimum(o, c) - l) / (c + 1e-9)
    z["volchg"] = np.log1p(v).diff()

    for n in (2, 3, 6, 12, 24, 48, 65):
        z[f"mom{n}"] = c.pct_change(n)
        z[f"vol{n}"] = r.rolling(n).std()
        z[f"ema{n}"] = c / c.ewm(span=n, adjust=False).mean() - 1
        z[f"range{n}"] = z["range"].rolling(n).mean()
        z[f"vr{n}"] = v / (v.rolling(n).mean() + 1e-9)

    z["ema6_24"] = c.ewm(span=6, adjust=False).mean() / c.ewm(span=24, adjust=False).mean() - 1
    z["ema12_48"] = c.ewm(span=12, adjust=False).mean() / c.ewm(span=48, adjust=False).mean() - 1
    z["ema24_65"] = c.ewm(span=24, adjust=False).mean() / c.ewm(span=65, adjust=False).mean() - 1
    z["vol_ratio"] = z["vol6"] / (z["vol48"] + 1e-9)
    z["mom_ratio"] = z["mom6"] / (z["vol12"] + 1e-9)

    for lag in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48):
        z[f"lag_r{lag}"] = r.shift(lag)
        z[f"lag_body{lag}"] = z["body"].shift(lag)
        z[f"lag_loc{lag}"] = z["loc"].shift(lag)

    minutes = (df.index.hour * 60 + df.index.minute) - (9 * 60 + 30)
    progress = np.clip(minutes / 390.0, 0.0, 1.0)
    first30 = (minutes < 30).astype(float)
    first60 = (minutes < 60).astype(float)
    first90 = (minutes < 90).astype(float)
    midday = ((minutes >= 120) & (minutes < 270)).astype(float)
    last60 = (minutes >= 330).astype(float)
    last30 = (minutes >= 360).astype(float)

    z["hour_sin"] = np.sin(2 * np.pi * (df.index.hour + df.index.minute / 60 - 9.5) / 6.5)
    z["hour_cos"] = np.cos(2 * np.pi * (df.index.hour + df.index.minute / 60 - 9.5) / 6.5)
    z["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 5)
    z["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 5)
    z["session_progress"] = progress
    z["session_sin"] = np.sin(2 * np.pi * progress)
    z["session_cos"] = np.cos(2 * np.pi * progress)
    z["first_30m"] = first30
    z["first_60m"] = first60
    z["first_90m"] = first90
    z["midday"] = midday
    z["last_60m"] = last60
    z["last_30m"] = last30
    z["open_distance"] = progress
    z["open_distance_sq"] = progress ** 2
    z["close_distance"] = 1 - progress
    z["is_monday"] = (df.index.dayofweek == 0).astype(float)
    z["is_friday"] = (df.index.dayofweek == 4).astype(float)

    for name, flag in (("open30", first30), ("open60", first60), ("open90", first90), ("mid", midday), ("last60", last60), ("last30", last30)):
        z[f"{name}_body"] = z["body"] * flag
        z[f"{name}_range"] = z["range"] * flag
        z[f"{name}_r1"] = z["r1"] * flag
        z[f"{name}_volratio"] = z["vol_ratio"] * flag
        z[f"{name}_mom6"] = z["mom6"] * flag

    z["session_x_mom6"] = progress * z["mom6"]
    z["session_x_volratio"] = progress * z["vol_ratio"]
    z["session_x_range"] = progress * z["range"]
    z["early_strength"] = first90 * z["mom6"]
    z["late_strength"] = last60 * z["mom6"]

    # Micro-regime features: normalized momentum persistence and volatility shock.
    z["mom2_vs_mom6"] = z["mom2"] / (z["mom6"].abs() + 1e-9)
    z["range6_vs_48"] = z["range6"] / (z["range48"] + 1e-9)
    z["vol6_vs_24"] = z["vol6"] / (z["vol24"] + 1e-9)
    z["trend_agreement"] = np.sign(z["ema6_24"]) * np.sign(z["ema12_48"])
    z["trend_strength"] = 0.6 * z["ema6_24"] + 0.4 * z["ema12_48"]
    z["bar_pressure"] = (z["body"] + z["loc"] - 0.5) * z["range"]
    z["volume_pressure"] = z["bar_pressure"] * z["vr6"]

    prev_close = c.groupby(df.index.normalize()).transform("last").shift(1)
    z["gap_prev_close"] = o / prev_close - 1.0
    return z.replace([np.inf, -np.inf], np.nan)


def supervised(df: pd.DataFrame):
    f = features(df)
    c = df["Close"].astype(float)
    ret1 = c.shift(-1) / c - 1
    ret3 = c.shift(-3) / c - 1
    ret6 = c.shift(-6) / c - 1
    scale = f["vol6"].clip(lower=0.0005)
    y1 = ret1 / scale
    y3 = ret3 / scale
    y6 = ret6 / scale
    c1 = (ret1 > 0.0015).astype(int)
    c3 = (ret3 > 0.0015).astype(int)
    return f, ret1, y1, y3, y6, c1, c3


@dataclass
class Models:
    r1a: HistGradientBoostingRegressor
    r3a: HistGradientBoostingRegressor
    r6a: HistGradientBoostingRegressor
    r1b: HistGradientBoostingRegressor
    r3b: HistGradientBoostingRegressor
    r6b: HistGradientBoostingRegressor
    c1: ExtraTreesClassifier
    c3: ExtraTreesClassifier


def fit_models(X, y1, y3, y6, c1, c3) -> Models:
    common_a = dict(
        max_iter=MODEL_MAX_ITER,
        learning_rate=0.04,
        max_leaf_nodes=13,
        min_samples_leaf=20,
        l2_regularization=3.5,
        loss="absolute_error",
        random_state=42,
    )
    common_b = dict(
        max_iter=MODEL_MAX_ITER,
        learning_rate=0.035,
        max_leaf_nodes=19,
        min_samples_leaf=28,
        l2_regularization=6.5,
        loss="absolute_error",
        random_state=77,
    )
    return Models(
        HistGradientBoostingRegressor(**common_a).fit(X, y1),
        HistGradientBoostingRegressor(**common_a).fit(X, y3),
        HistGradientBoostingRegressor(**common_a).fit(X, y6),
        HistGradientBoostingRegressor(**common_b).fit(X, y1),
        HistGradientBoostingRegressor(**common_b).fit(X, y3),
        HistGradientBoostingRegressor(**common_b).fit(X, y6),
        ExtraTreesClassifier(n_estimators=220, max_depth=9, min_samples_leaf=10, max_features=0.60, class_weight="balanced", n_jobs=-1, random_state=59).fit(X, c1),
        ExtraTreesClassifier(n_estimators=220, max_depth=8, min_samples_leaf=12, max_features=0.65, class_weight="balanced", n_jobs=-1, random_state=83).fit(X, c3),
    )


def sequential_predictions(df: pd.DataFrame, split_start: int, split_end: int) -> pd.DataFrame:
    f, _, y1, y3, y6, c1, c3 = supervised(df)
    X = f.to_numpy(float)
    ys = [q.to_numpy(float) for q in (y1, y3, y6, c1, c3)]
    valid = [i for i in range(LOOKBACK_BARS - 1, len(f) - 6)
             if split_start <= i < split_end and all(np.isfinite(q[i]) for q in ys[:4]) and np.isfinite(X[i]).all()]
    out = []
    models: Models | None = None
    last_fit = -10**9
    started = time.time()
    for count, i in enumerate(valid, 1):
        if models is None or i - last_fit >= RETRAIN_EVERY:
            end = i
            start = max(LOOKBACK_BARS - 1, end - TRAIN_WINDOW)
            idx = np.arange(start, end)
            good = np.isfinite(X[idx]).all(axis=1)
            for q in ys:
                good &= np.isfinite(q[idx])
            idx = idx[good]
            if len(idx) >= 450:
                models = fit_models(X[idx], y1.to_numpy(float)[idx], y3.to_numpy(float)[idx], y6.to_numpy(float)[idx], c1.to_numpy(float)[idx], c3.to_numpy(float)[idx])
                last_fit = i
        if models is None:
            continue

        row = X[i].reshape(1, -1)
        r1 = 0.5 * (float(models.r1a.predict(row)[0]) + float(models.r1b.predict(row)[0]))
        r3 = 0.5 * (float(models.r3a.predict(row)[0]) + float(models.r3b.predict(row)[0]))
        r6 = 0.5 * (float(models.r6a.predict(row)[0]) + float(models.r6b.predict(row)[0]))
        p1 = float(models.c1.predict_proba(row)[0, 1])
        p3 = float(models.c3.predict_proba(row)[0, 1])

        scale = max(float(f["vol6"].iloc[i]), 0.0005)
        ret1 = r1 * scale
        ret3 = r3 / 3.0 * scale
        ret6 = r6 / 6.0 * scale
        # Directional consensus: all horizons contribute, but short horizon remains dominant.
        base = 0.56 * ret1 + 0.28 * ret3 + 0.16 * ret6
        prob_consensus = 0.58 * p1 + 0.42 * p3
        confidence = np.clip((prob_consensus - 0.5) * 2.0, -1.0, 1.0)
        trend_agreement = float(f["trend_agreement"].iloc[i])
        trend_boost = 1.08 if trend_agreement > 0 else (0.92 if trend_agreement < 0 else 1.0)
        signal = base * (0.70 + 0.80 * max(confidence, 0.0)) * trend_boost
        # Keep a direct consensus score so the backtester can reject isolated weak predictions.
        consensus = float(np.mean([ret1 > 0, ret3 > 0, ret6 > 0]))

        out.append((df.index[i], float(df["Close"].iloc[i]), ret1, ret3, ret6, p1, p3, prob_consensus, signal, consensus, trend_agreement))
        if count == 1 or count % 200 == 0 or count == len(valid):
            elapsed = time.time() - started
            rate = count / max(elapsed, 1e-9)
            eta = (len(valid) - count) / max(rate, 1e-9)
            print(f"prediction {count}/{len(valid)} | {rate:.1f} bars/s | ETA ~{eta/60:.1f} min", flush=True)

    return pd.DataFrame(out, columns=["date", "price", "pred1", "pred3", "pred6", "prob1", "prob3", "prob_consensus", "signal", "consensus", "trend_agreement"]).set_index("date")


def backtest_next_open(df, panel, start, end, threshold, min_consensus):
    pos = {ts: i for i, ts in enumerate(df.index)}
    signals = panel[(panel.index >= start) & (panel.index < end)]
    cash = INITIAL_CASH
    shares = 0.0
    entry_i = None
    entry_outlay = None
    trade_returns = []
    curve = []

    for ts, row in signals.iterrows():
        i = pos.get(ts)
        if i is None or i + 1 >= len(df):
            continue
        ex = i + 1
        if df.index[ex] >= end:
            continue
        px = float(df["Open"].iloc[ex])
        if not np.isfinite(px) or px <= 0:
            continue

        want = float(row["signal"]) > threshold and float(row["consensus"]) >= min_consensus and float(row["prob_consensus"]) > 0.50
        equity = cash + shares * px
        target = equity if want else 0.0
        current = shares * px
        if shares > 0 and entry_i is not None and ex - entry_i >= MAX_HOLD:
            target = 0.0

        if target < current * 0.98 and shares > 0:
            sell_value = min(current, current - target)
            sell_shares = min(shares, sell_value / px)
            proceeds = sell_shares * px * (1 - COST)
            cash += proceeds
            shares -= sell_shares
            if shares <= 1e-12:
                shares = 0.0
                if entry_outlay:
                    trade_returns.append(cash / entry_outlay - 1.0)
                entry_i = None
                entry_outlay = None
        elif target > current * 1.02:
            desired = min(target - current, cash / (1 + COST))
            if desired > max(0.01, equity * 0.01):
                before = cash + shares * px
                shares += desired / (px * (1 + COST))
                cash -= desired
                if entry_i is None:
                    entry_i = ex
                    entry_outlay = before

        curve.append(cash + shares * px)

    if len(curve) < 2:
        return dict(return_=0.0, final=INITIAL_CASH, trades=0, max_dd=0.0, sharpe=0.0, win_rate=0.0)
    arr = np.asarray(curve, dtype=float)
    peak = np.maximum.accumulate(arr)
    max_dd = float(np.min(arr / np.maximum(peak, 1e-9) - 1.0))
    rets = arr[1:] / np.maximum(arr[:-1], 1e-9) - 1.0
    sharpe = float(np.mean(rets) / (np.std(rets) + 1e-12) * math.sqrt(252 * 6.5)) if len(rets) > 20 else 0.0
    wr = float(np.mean(np.asarray(trade_returns) > 0)) if trade_returns else 0.0
    final = float(arr[-1])
    return dict(return_=final / INITIAL_CASH - 1.0, final=final, trades=len(trade_returns), max_dd=max_dd, sharpe=sharpe, win_rate=wr)


def selection_score(r):
    # Conservative validation objective: reward return/Sharpe, penalize DD and very low activity.
    s = r["return_"] - 0.38 * abs(min(r["max_dd"], 0.0)) + 0.020 * max(r["sharpe"], 0.0)
    if r["trades"] < 20:
        s -= 0.002 * (20 - r["trades"])
    return s


def buyhold(df, start, end):
    p = df[(df.index >= start) & (df.index < end)]["Open"].astype(float)
    return float(p.iloc[-1] / p.iloc[0] - 1.0) if len(p) > 1 else 0.0


def money_line(label: str, result: dict) -> str:
    gain = result["final"] - INITIAL_CASH
    return f"{label}: inicio=€{INITIAL_CASH:.2f} | final=€{result['final']:.2f} | ganado={gain:+.2f}€ | retorno={result['return_']:+.2%} | trades={result['trades']} | DD={result['max_dd']:.2%} | Sharpe={result['sharpe']:.2f} | WR={result['win_rate']:.1%}"


def main():
    t0 = time.time()
    df = load_data()
    n = len(df)
    train_cut = int(n * 0.60)
    val_cut = int(n * 0.80)
    test_start = val_cut + 2

    print(f"Capital inicial: €{INITIAL_CASH:.2f}", flush=True)
    print("Generando predicciones V30...", flush=True)
    panel = sequential_predictions(df, train_cut, n - 1)
    if panel.empty:
        raise RuntimeError("No predictions")

    val_idx = panel.index[panel.index < df.index[val_cut]]
    test_idx = panel.index[panel.index >= df.index[test_start]]
    vs, ve = val_idx[0], val_idx[-1]
    ts, te = test_idx[0], test_idx[-1]
    print(f"Validacion: {vs} -> {ve}")
    print(f"Test ciego: {ts} -> {te}")
    print(f"MAX_HOLD={MAX_HOLD} | coste/lado={COST:.3%}")

    best = None
    print("\n=== SWEEP V30 | VALIDACION ONLY ===")
    for th in THRESHOLD_CANDIDATES:
        for consensus in CONSENSUS_CANDIDATES:
            r = backtest_next_open(df, panel, vs, ve, th, consensus)
            r["threshold"] = th
            r["min_consensus"] = consensus
            r["score"] = selection_score(r)
            print(f"threshold={th:+.3%} | consensus>={consensus:.2f} | ret={r['return_']:+.2%} | trades={r['trades']} | DD={r['max_dd']:.2%} | Sharpe={r['sharpe']:.2f} | WR={r['win_rate']:.1%} | score={r['score']:+.4f}")
            if best is None or r["score"] > best["score"]:
                best = r

    assert best is not None
    test = backtest_next_open(df, panel, ts, te, best["threshold"], best["min_consensus"])
    bh = buyhold(df, ts, te)

    print("\n=== CONFIGURACION ELEGIDA | VALIDACION ONLY ===")
    print(f"threshold={best['threshold']:+.3%} | min_consensus={best['min_consensus']:.2f}")
    print(money_line("VALIDACION", best))

    print("\n=== TEST CIEGO ===")
    print(money_line("TEST IA", test))
    bh_final = INITIAL_CASH * (1 + bh)
    print(f"TEST B&H: inicio=€{INITIAL_CASH:.2f} | final=€{bh_final:.2f} | ganado={bh_final-INITIAL_CASH:+.2f}€ | retorno={bh:+.2%}")
    print(f"Diferencia IA vs B&H: {(test['return_'] - bh):+.2%}")
    print("No se usa el TEST para reajustar parametros.")

    out = panel.copy()
    out.to_csv("v30_predictions.csv")
    print("Archivo: v30_predictions.csv")
    print(f"runtime={(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
