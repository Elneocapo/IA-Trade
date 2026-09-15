"""IA-Trade V16 - NVDA-only 1H short-horizon research.

Uses only NVDA regular-session 1H candles. The model sees the latest
10 trading days (120 hourly bars) and predicts normalized forward returns
at 1, 3, 6 and 12 trading-hour horizons. A trade can never stay open for
more than 12 market bars (~2 trading days).

Paper trading/backtesting only. To create another asset-specific AI later,
change ASSET below.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
warnings.filterwarnings("ignore")

ASSET = "NVDA"
PERIOD = "2y"                 # yfinance 1H history is limited; use all practical history
INTERVAL = "1h"
LOOKBACK = 120                 # ~10 regular trading days of 1H bars
MAX_HOLD_BARS = 12             # ~2 trading days
COST = 0.001
INITIAL_CASH = 50.0
HORIZONS = (1, 3, 6, 12)
SEEDS = (11, 23, 47, 71, 101)  # ensemble diversity


def load_data():
    print(f"=== TRAINING ASSET: {ASSET} | 1H | LOOKBACK: {LOOKBACK} BARS | MAX TRADE: {MAX_HOLD_BARS} BARS ===", flush=True)
    df = yf.download(ASSET, period=PERIOD, interval=INTERVAL, auto_adjust=True,
                     progress=False, prepost=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ASSET}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    # Keep regular US session only. yfinance timestamps are timezone-aware for intraday data.
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("America/New_York")
    else:
        idx = idx.tz_convert("America/New_York")
    df.index = idx
    df = df.between_time("09:30", "16:00")
    # Remove the 16:00 bar if the provider labels it as the session-close bucket;
    # this keeps bars aligned with the common 09:30-16:00 regular session.
    return df


def make_features(df):
    o, h, l, c, v = [df[k].astype(float) for k in ("Open", "High", "Low", "Close", "Volume")]
    x = pd.DataFrame(index=df.index)
    lr = np.log(c).diff()
    rng = (h - l) / c
    body = (c - o) / o
    x["ret"] = lr
    x["range"] = rng
    x["body"] = body
    x["upper_wick"] = (h - np.maximum(o, c)) / c
    x["lower_wick"] = (np.minimum(o, c) - l) / c
    x["close_location"] = (c - l) / (h - l + 1e-9)
    x["volume_change"] = np.log1p(v).diff()

    for n in (2, 3, 5, 8, 13, 21, 34, 55):
        x[f"mom{n}"] = c.pct_change(n)
        x[f"vol{n}"] = lr.rolling(n).std()
        x[f"range_mean{n}"] = rng.rolling(n).mean()
        x[f"volume_ratio{n}"] = v / (v.rolling(n).mean() + 1e-9)
        x[f"ema_gap{n}"] = c / c.ewm(span=n, adjust=False).mean() - 1

    x["ema8_21"] = c.ewm(span=8, adjust=False).mean() / c.ewm(span=21, adjust=False).mean() - 1
    x["ema21_55"] = c.ewm(span=21, adjust=False).mean() / c.ewm(span=55, adjust=False).mean() - 1
    x["vol_ratio"] = x["vol5"] / (x["vol34"] + 1e-9)
    x["range_pos20"] = (c - l.rolling(20).min()) / (h.rolling(20).max() - l.rolling(20).min() + 1e-9)

    # Intraday seasonality, learned only from NVDA itself.
    mins = df.index.hour * 60 + df.index.minute
    phase = (mins - 570) / 390 * 2 * np.pi
    x["tod_sin"] = np.sin(phase)
    x["tod_cos"] = np.cos(phase)
    x["day_sin"] = np.sin(df.index.dayofweek / 5 * 2 * np.pi)
    x["day_cos"] = np.cos(df.index.dayofweek / 5 * 2 * np.pi)
    return x.replace([np.inf, -np.inf], np.nan)


def build_dataset(df):
    feat = make_features(df)
    c = df["Close"].astype(float)
    vol = feat["vol21"].clip(lower=0.001)
    rows = []
    # Sequence ends at t; targets start after t, so there is no look-ahead.
    for i in range(LOOKBACK, len(df) - max(HORIZONS)):
        if not np.isfinite(feat.iloc[i - LOOKBACK:i].values).all():
            continue
        seq = feat.iloc[i - LOOKBACK:i].values.astype(np.float32)
        # Normalize each bar's return-like columns naturally through the scaler;
        # targets are volatility-normalized to make regimes more comparable.
        future = np.array([(c.iloc[i + h] / c.iloc[i] - 1) for h in HORIZONS], dtype=np.float32)
        scale = float(vol.iloc[i])
        target = future / (scale * np.sqrt(np.asarray(HORIZONS, dtype=np.float32)) + 1e-6)
        rows.append((df.index[i], seq, target, float(c.iloc[i]), scale))
    if not rows:
        raise RuntimeError("Not enough clean 1H observations for the selected lookback")
    return rows


def train_ensemble(X, Y):
    models = []
    for seed in SEEDS:
        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                solver="adam",
                alpha=0.0005,
                learning_rate_init=0.001,
                max_iter=250,
                early_stopping=True,
                validation_fraction=0.12,
                n_iter_no_change=18,
                random_state=seed,
                batch_size=64,
            ),
        )
        model.fit(X, Y)
        models.append(model)
    return models


def walk_forward(rows):
    dates = np.array([r[0] for r in rows])
    X = np.array([r[1].reshape(-1) for r in rows], dtype=np.float32)
    Y = np.array([r[2] for r in rows], dtype=np.float32)
    warm = max(700, int(len(rows) * 0.45))
    step = max(48, int(len(rows) * 0.025))
    out = []
    for end in range(warm, len(rows), step):
        stop = min(end + step, len(rows))
        models = train_ensemble(X[:end], Y[:end])
        pred = np.mean([m.predict(X[end:stop]) for m in models], axis=0)
        for j, i in enumerate(range(end, stop)):
            out.append((dates[i], *pred[j], rows[i][3], rows[i][4]))
        print(f"walk-forward {stop}/{len(rows)} | ensemble={len(SEEDS)}", flush=True)
    return pd.DataFrame(out, columns=["date", "s1", "s3", "s6", "s12", "price", "vol"]).set_index("date")


def backtest(panel, start, end, threshold, invest, stop_mult):
    dates = [d for d in panel.index if start <= d < end]
    cash, shares = INITIAL_CASH, 0.0
    entry_i = None
    entry_price = None
    curve, trades = [], 0
    for i, d in enumerate(dates):
        r = panel.loc[d]
        price = float(r.price)
        total = cash + shares * price
        # Multi-horizon score, emphasizing horizons that can actually fit inside 2 days.
        score = 0.15 * r.s1 + 0.25 * r.s3 + 0.30 * r.s6 + 0.30 * r.s12
        target_weight = invest if score > threshold else 0.0
        if shares > 0 and entry_i is not None:
            held = i - entry_i
            if held >= MAX_HOLD_BARS:
                target_weight = 0.0
            if entry_price is not None and price < entry_price * (1 - stop_mult):
                target_weight = 0.0
        target_value = target_weight * total
        current_value = shares * price
        if target_value < current_value * 0.98 and current_value > 0:
            sell_value = current_value - target_value
            cash += sell_value * (1 - COST)
            shares -= sell_value / price
            trades += 1
            if shares <= 1e-12:
                shares, entry_i, entry_price = 0.0, None, None
        if target_value > current_value * 1.02:
            buy_value = min(target_value - current_value, cash / (1 + COST))
            if buy_value > total * 0.01:
                shares += buy_value / (price * (1 + COST))
                cash -= buy_value * (1 + COST)
                trades += 1
                if entry_i is None:
                    entry_i, entry_price = i, price
        curve.append(cash + shares * price)
    if len(curve) < 2:
        return 0., 0, 0., 0., 0.
    curve = np.asarray(curve)
    peak = np.maximum.accumulate(curve)
    dd = float(np.min(curve / peak - 1))
    rets = curve[1:] / curve[:-1] - 1
    sharpe = float(np.mean(rets) / (np.std(rets) + 1e-12) * np.sqrt(252 * 6.5)) if len(rets) > 20 else 0.
    return float(curve[-1] / INITIAL_CASH - 1), trades, dd, sharpe, float(np.mean(rets > 0))


def buy_and_hold(panel, start, end):
    p = panel[(panel.index >= start) & (panel.index < end)].price
    return float(p.iloc[-1] / p.iloc[0] - 1) if len(p) > 1 else 0.


def main():
    df = load_data()
    rows = build_dataset(df)
    panel = walk_forward(rows)
    dates = sorted(panel.index)
    if len(dates) < 200:
        raise RuntimeError("Not enough out-of-sample 1H observations")
    cut = dates[len(dates) // 2]
    end = dates[-1]

    candidates = []
    for threshold in (0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
        for invest in (0.5, 0.75, 1.0):
            for stop_mult in (0.0, 0.015, 0.025, 0.04):
                r, t, dd, sh, hit = backtest(panel, dates[0], cut, threshold, invest, stop_mult)
                # Select using validation economics; final half remains untouched.
                objective = r + 0.15 * sh + 0.10 * dd - 0.00005 * t
                candidates.append((objective, r, threshold, invest, stop_mult, dd, sh, hit, t))
    best = max(candidates, key=lambda x: x[0])
    _, vr, threshold, invest, stop_mult, vdd, vsh, vhit, vt = best
    fr, trades, fdd, fsh, fhit = backtest(panel, cut, end, threshold, invest, stop_mult)
    bh = buy_and_hold(panel, cut, end)

    print("\n=== V16 NVDA 1H NEURAL ENSEMBLE SUMMARY ===")
    print(f"asset={ASSET} | source={PERIOD} {INTERVAL} | lookback={LOOKBACK} bars (~10d) | max_hold={MAX_HOLD_BARS} bars (~2d)")
    print(f"horizons={HORIZONS}h | ensemble={len(SEEDS)} models | regular_session_only=YES")
    print(f"selected threshold={threshold:.2f} | invest={invest:.0%} | stop={stop_mult:.1%}")
    print(f"validation IA={vr:+.2%} | DD={vdd:.2%} | Sharpe={vsh:.2f} | hit={vhit:.1%} | trades={vt}")
    print(f"FINAL IA={fr:+.2%} | {ASSET} B&H={bh:+.2%} | trades={trades} | maxDD={fdd:.2%} | Sharpe={fsh:.2f} | hit={fhit:.1%}")
    print(f"IA beats {ASSET} B&H: {'YES' if fr > bh else 'NO'}")


if __name__ == "__main__":
    main()
