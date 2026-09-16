"""IA-Trade V23 - NVDA 1H con conciencia explícita de sesión.

Paper trading / research only. No broker or live orders.

Objetivo:
- Mantener la arquitectura/modelos de V16.7.
- Añadir variables temporales de sesión más expresivas que hour_sin/hour_cos.
- Especial atención al comportamiento alrededor de la apertura de NYSE.
- Entrenamiento walk-forward: cada predicción usa solo datos disponibles hasta el cierre de esa vela.
- El test final NO se usa para seleccionar parámetros.

IMPORTANTE:
La apertura se representa con features de tiempo de mercado, no con información futura.
Las señales siguen siendo predicciones del siguiente movimiento horario.
"""
from __future__ import annotations

import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesClassifier

warnings.filterwarnings("ignore")

ASSET = "NVDA"
PERIOD = "2y"
INTERVAL = "1h"
LOOKBACK_BARS = 65
TRAIN_WINDOW = 1200
RETRAIN_EVERY = 24
MODEL_MAX_ITER = 140
INITIAL_CASH = 50.0
COST = 0.001

# Fixed candidate grid inherited from the current research cycle.
THRESHOLD_CANDIDATES = (
    0.0, 0.00025, 0.0005, 0.00075, 0.0010, 0.00125,
    0.0015, 0.00175, 0.0020, 0.0025, 0.0030, 0.0040
)
MAX_HOLD = 6
WEIGHT = 1.0


def load_data() -> pd.DataFrame:
    print(f"=== V23 | {ASSET} | {INTERVAL} | session-aware | context={LOOKBACK_BARS} ===", flush=True)
    df = yf.download(
        ASSET, period=PERIOD, interval=INTERVAL,
        auto_adjust=True, progress=False, prepost=False
    )
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

    # Price/volume structure inherited from V16.7.
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

    # Existing cyclical time encoding retained for apples-to-apples continuity.
    hour = df.index.hour + df.index.minute / 60.0
    z["hour_sin"] = np.sin(2 * np.pi * (hour - 9.5) / 6.5)
    z["hour_cos"] = np.cos(2 * np.pi * (hour - 9.5) / 6.5)
    z["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 5)
    z["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 5)

    # NEW: explicit NYSE session position (09:30 -> 16:00).
    minutes = (df.index.hour * 60 + df.index.minute) - (9 * 60 + 30)
    progress = np.clip(minutes / 390.0, 0.0, 1.0)
    z["session_progress"] = progress
    z["session_sin"] = np.sin(2 * np.pi * progress)
    z["session_cos"] = np.cos(2 * np.pi * progress)

    # Opening/closing regime flags. These are deterministic calendar features.
    z["first_30m"] = (minutes < 30).astype(float)
    z["first_60m"] = (minutes < 60).astype(float)
    z["first_90m"] = (minutes < 90).astype(float)
    z["midday"] = ((minutes >= 120) & (minutes < 270)).astype(float)
    z["last_60m"] = (minutes >= 330).astype(float)
    z["last_30m"] = (minutes >= 360).astype(float)

    # Relative location inside the trading day, centered around the opening burst.
    z["open_distance"] = progress
    z["open_distance_sq"] = progress ** 2
    z["close_distance"] = 1.0 - progress

    # Day-of-week interaction context.
    z["is_monday"] = (df.index.dayofweek == 0).astype(float)
    z["is_friday"] = (df.index.dayofweek == 4).astype(float)

    return z.replace([np.inf, -np.inf], np.nan)


def make_supervised(df: pd.DataFrame):
    f = features(df)
    c = df["Close"].astype(float)
    ret1 = c.shift(-1) / c - 1.0
    ret3 = c.shift(-3) / c - 1.0
    ret6 = c.shift(-6) / c - 1.0
    scale = f["vol6"].clip(lower=0.0005)
    return f, ret1, ret1 / scale, ret3 / scale, ret6 / scale, (ret1 > 0.0015).astype(int)


def fit_models(X, y1, y3, y6, yc):
    common = dict(
        max_iter=MODEL_MAX_ITER,
        learning_rate=0.04,
        max_leaf_nodes=13,
        min_samples_leaf=20,
        l2_regularization=3.5,
        loss="absolute_error",
        random_state=42,
    )
    return (
        HistGradientBoostingRegressor(**common).fit(X, y1),
        HistGradientBoostingRegressor(**common).fit(X, y3),
        HistGradientBoostingRegressor(**common).fit(X, y6),
        ExtraTreesClassifier(
            n_estimators=220, max_depth=9, min_samples_leaf=10,
            max_features=0.60, class_weight="balanced", n_jobs=-1,
            random_state=59
        ).fit(X, yc),
    )


def sequential_predictions(df: pd.DataFrame, split_start: int, split_end: int) -> pd.DataFrame:
    f, ret1, y1, y3, y6, yc = make_supervised(df)
    X = f.to_numpy(float)
    a, b, c, d = [q.to_numpy(float) for q in (y1, y3, y6, yc)]
    valid = [
        i for i in range(LOOKBACK_BARS - 1, len(f) - 6)
        if split_start <= i < split_end
        and np.isfinite(a[i]) and np.isfinite(b[i]) and np.isfinite(c[i])
        and np.isfinite(X[i]).all()
    ]

    out = []
    models = None
    last_fit = -10**9
    started = time.time()
    total = len(valid)

    for count, i in enumerate(valid, 1):
        if models is None or i - last_fit >= RETRAIN_EVERY:
            end = i
            start = max(LOOKBACK_BARS - 1, end - TRAIN_WINDOW)
            idx = np.arange(start, end)
            good = (
                np.isfinite(a[idx]) & np.isfinite(b[idx]) & np.isfinite(c[idx])
                & np.isfinite(d[idx]) & np.isfinite(X[idx]).all(axis=1)
            )
            idx = idx[good]
            if len(idx) >= 400:
                models = fit_models(X[idx], a[idx], b[idx], c[idx], d[idx])
                last_fit = i

        if models is None:
            continue

        m1, m3, m6, clf = models
        row = X[i].reshape(1, -1)
        p1 = float(m1.predict(row)[0])
        p3 = float(m3.predict(row)[0])
        p6 = float(m6.predict(row)[0])
        prob = float(clf.predict_proba(row)[0, 1])

        scale = max(float(f["vol6"].iloc[i]), 0.0005)
        r1 = p1 * scale
        r3 = p3 / 3.0 * scale
        r6 = p6 / 6.0 * scale
        base = 0.62 * r1 + 0.25 * r3 + 0.13 * r6
        confidence = np.clip((prob - 0.5) * 2, -1, 1)
        signal = base * (0.72 + 0.85 * max(confidence, 0))

        out.append((
            df.index[i], float(df["Close"].iloc[i]), r1, r3, r6, prob,
            signal, float(f["ema12_48"].iloc[i]), float(f["vol_ratio"].iloc[i]),
            float(f["session_progress"].iloc[i]), float(f["first_60m"].iloc[i]),
            float(f["midday"].iloc[i]), float(f["last_60m"].iloc[i]),
        ))

        if count == 1 or count % 200 == 0 or count == total:
            elapsed = time.time() - started
            rate = count / max(elapsed, 1e-9)
            eta = (total - count) / max(rate, 1e-9)
            print(
                f"prediction {count}/{total} | {rate:.1f} bars/s | ETA ~{eta/60:.1f} min",
                flush=True,
            )

    return pd.DataFrame(
        out,
        columns=[
            "date", "price", "pred1", "pred3", "pred6", "prob_up", "signal",
            "trend", "vol_ratio", "session_progress", "first_60m", "midday", "last_60m"
        ],
    ).set_index("date")


def backtest_next_open(df, panel, start, end, threshold):
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
        want_long = float(row["signal"]) > threshold
        target_value = equity * WEIGHT if want_long else 0.0
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
            desired_gross = min(target_value - current_value, cash / (1.0 + COST))
            if desired_gross > max(0.01, equity * 0.01):
                before = cash + shares * px
                buy_shares = desired_gross / (px * (1.0 + COST))
                cash -= desired_gross
                shares += buy_shares
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
    score = r["return"] - 0.30 * abs(min(r["max_dd"], 0.0)) + 0.015 * max(r["sharpe"], 0.0)
    if r["trades"] < 8:
        score -= 0.004 * (8 - r["trades"])
    return score


def buy_and_hold(df, start, end):
    p = df[(df.index >= start) & (df.index < end)]["Open"].astype(float)
    return float(p.iloc[-1] / p.iloc[0] - 1.0) if len(p) > 1 else 0.0


def main():
    overall = time.time()
    df = load_data()
    n = len(df)
    train_cut = int(n * 0.60)
    val_cut = int(n * 0.80)
    test_start = val_cut + 2

    print("V23: generando predicciones con features explícitas de sesión...", flush=True)
    panel = sequential_predictions(df, train_cut, n - 1)
    if panel.empty:
        raise RuntimeError("No se pudieron generar predicciones")

    val_dates = panel.index[panel.index < df.index[val_cut]]
    test_dates = panel.index[panel.index >= df.index[test_start]]
    if len(val_dates) == 0 or len(test_dates) == 0:
        raise RuntimeError("No hay suficientes predicciones")

    vs, ve = val_dates[0], val_dates[-1]
    ts, te = test_dates[0], test_dates[-1]

    print(f"Validación: {vs} -> {ve}")
    print(f"Test ciego: {ts} -> {te}")
    print(f"Coste por lado={COST:.3%} | hold={MAX_HOLD} | weight={WEIGHT:.0%}")
    print("\n=== SWEEP DE UMBRAL | SOLO VALIDACIÓN ===")

    results = []
    for threshold in THRESHOLD_CANDIDATES:
        r = backtest_next_open(df, panel, vs, ve, threshold)
        r["threshold"] = threshold
        r["score"] = validation_score(r)
        results.append(r)
        print(
            f"threshold={threshold:+.3%} | ret={r['return']:+.2%} | trades={r['trades']} | "
            f"DD={r['max_dd']:.2%} | Sharpe={r['sharpe']:.2f} | WR={r['win_rate']:.1%} | score={r['score']:+.4f}"
        )

    best = max(results, key=lambda x: x["score"])
    threshold = best["threshold"]
    test = backtest_next_open(df, panel, ts, te, threshold)
    bh = buy_and_hold(df, ts, te)

    print("\n=== UMBRAL ELEGIDO POR VALIDACIÓN ===")
    print(f"threshold={threshold:+.3%}")
    print(
        f"VALIDACIÓN: retorno={best['return']:+.2%} | final=€{best['final']:.2f} | "
        f"trades={best['trades']} | DD={best['max_dd']:.2%} | Sharpe={best['sharpe']:.2f} | WR={best['win_rate']:.1%}"
    )
    print("\n=== TEST CIEGO | UMBRAL FIJO ===")
    print(
        f"TEST IA: retorno={test['return']:+.2%} | final=€{test['final']:.2f} | "
        f"trades={test['trades']} | DD={test['max_dd']:.2%} | Sharpe={test['sharpe']:.2f} | WR={test['win_rate']:.1%}"
    )
    print(f"TEST B&H (open→open): {bh:+.2%} | final=€{INITIAL_CASH*(1+bh):.2f}")
    print(f"IA vs B&H en test: {'POSITIVA' if test['return'] > bh else 'NEGATIVA'}")
    print(f"runtime={(time.time()-overall)/60:.1f} min")
    print("\nNOTA: V23 es un experimento de investigación. No usar el resultado del test para volver a ajustar el umbral.")


if __name__ == "__main__":
    main()
