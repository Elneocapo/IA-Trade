"""IA-Trade V16 - NVDA only, sequential 1H next-candle prediction.

Paper trading / research only. No broker or live orders.
At every hourly close the model uses only information already known, predicts
THE NEXT 1H CANDLE, and the simulator makes the next decision after each close.
The model context is the last ~10 regular US trading days (65 hourly bars).
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings("ignore")

ASSET = "NVDA"                 # Change this one variable for another asset AI later.
PERIOD = "2y"                  # 1H Yahoo history is much shorter than daily history.
INTERVAL = "1h"
LOOKBACK_BARS = 65             # ~10 regular US trading days x 6.5 hourly bars/day.
MAX_HOLD_BARS = 13             # ~2 regular trading days.
TRAIN_WINDOW = 1500            # Rolling recent history used for each retrain.
RETRAIN_EVERY = 12             # Refit every 12 bars; predictions still happen every bar.
COST = 0.001
INITIAL_CASH = 50.0


def load_data() -> pd.DataFrame:
    print(f"=== V16 | {ASSET} | {INTERVAL} | context={LOOKBACK_BARS} bars | max hold={MAX_HOLD_BARS} bars ===", flush=True)
    df = yf.download(ASSET, period=PERIOD, interval=INTERVAL, auto_adjust=True,
                     progress=False, prepost=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ASSET}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.dropna()
    # Keep regular-session bars only. Yahoo timestamps are exchange-local/UTC depending on version.
    # Filtering by hour after tz conversion makes the strategy independent of pre/post-market noise.
    if getattr(df.index, "tz", None) is not None:
        try:
            df.index = df.index.tz_convert("America/New_York")
        except Exception:
            pass
    df = df.between_time("09:30", "16:00")
    return df


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, o, v = [df[k].astype(float) for k in ("Close", "High", "Low", "Open", "Volume")]
    z = pd.DataFrame(index=df.index)
    lr = np.log(c).diff()
    z["ret1"] = lr
    z["range"] = (h - l) / c
    z["body"] = (c - o) / o
    z["upper_wick"] = (h - np.maximum(o, c)) / c
    z["lower_wick"] = (np.minimum(o, c) - l) / c
    z["close_location"] = (c - l) / (h - l + 1e-9)
    z["log_volume"] = np.log1p(v)
    z["vol_chg"] = z["log_volume"].diff()
    for n in (3, 6, 13, 21, 34):
        z[f"mom{n}"] = c.pct_change(n)
        z[f"ema{n}"] = c / c.ewm(span=n, adjust=False).mean() - 1
        z[f"vol{n}"] = lr.rolling(n).std()
        z[f"range_mean{n}"] = z["range"].rolling(n).mean()
        z[f"volume_ratio{n}"] = v / (v.rolling(n).mean() + 1e-9)
    z["ema_fast_slow"] = c.ewm(span=13, adjust=False).mean() / c.ewm(span=34, adjust=False).mean() - 1
    z["vol_ratio"] = z["vol6"] / (z["vol34"] + 1e-9)
    # Intraday seasonality is allowed because the hour is known before the next candle starts.
    if hasattr(df.index, "hour"):
        hour = df.index.hour + df.index.minute / 60.0
        z["hour_sin"] = np.sin(2 * np.pi * (hour - 9.5) / 6.5)
        z["hour_cos"] = np.cos(2 * np.pi * (hour - 9.5) / 6.5)
    if hasattr(df.index, "dayofweek"):
        dow = df.index.dayofweek
        z["dow_sin"] = np.sin(2 * np.pi * dow / 5)
        z["dow_cos"] = np.cos(2 * np.pi * dow / 5)
    return z.replace([np.inf, -np.inf], np.nan)


def make_supervised(df: pd.DataFrame):
    f = base_features(df)
    c = df["Close"].astype(float)
    # Predict the next hourly close-to-close return, normalized by recent volatility.
    future_ret = c.shift(-1) / c - 1
    scale = f["vol6"].clip(lower=0.0005)
    y = future_ret / scale
    d = f.copy()
    d["target"] = y
    return f, future_ret, d.dropna()


def flatten_context(x: pd.DataFrame, end_pos: int) -> np.ndarray:
    block = x.iloc[end_pos - LOOKBACK_BARS + 1:end_pos + 1]
    return block.to_numpy(dtype=float).reshape(-1)


def fit_model(X: np.ndarray, y: np.ndarray, seed: int = 42):
    return HistGradientBoostingRegressor(
        max_iter=700,
        learning_rate=0.035,
        max_leaf_nodes=31,
        min_samples_leaf=10,
        l2_regularization=1.5,
        loss="absolute_error",
        random_state=seed,
    ).fit(X, y)


def sequential_predictions(df: pd.DataFrame, split_start: int, split_end: int) -> pd.DataFrame:
    """Generate one prediction for every bar, strictly using data before that prediction.

    Retraining is batched for speed, but the signal is refreshed on every hourly close.
    """
    f, future_ret, supervised = make_supervised(df)
    # Need complete context and target availability.
    valid_positions = []
    for i in range(LOOKBACK_BARS - 1, len(f) - 1):
        if i < split_start or i >= split_end:
            continue
        block = f.iloc[i - LOOKBACK_BARS + 1:i + 1]
        if block.isna().any().any():
            continue
        valid_positions.append(i)
    if not valid_positions:
        return pd.DataFrame()

    # Precompute flattened examples for the rolling training window.
    X_all, y_all, pos_all = [], [], []
    for i in range(LOOKBACK_BARS - 1, len(f) - 1):
        block = f.iloc[i - LOOKBACK_BARS + 1:i + 1]
        target = future_ret.iloc[i]
        scale = f["vol6"].iloc[i]
        if block.isna().any().any() or not np.isfinite(target) or not np.isfinite(scale) or scale <= 0:
            continue
        X_all.append(block.to_numpy(dtype=float).reshape(-1))
        y_all.append(float(target / max(scale, 0.0005)))
        pos_all.append(i)
    X_all = np.asarray(X_all)
    y_all = np.asarray(y_all)
    pos_all = np.asarray(pos_all)

    outputs = []
    model = None
    last_fit_pos = -10**9
    for i in valid_positions:
        if model is None or i - last_fit_pos >= RETRAIN_EVERY:
            eligible = pos_all < i  # never train on the candle currently being predicted
            eligible_pos = pos_all[eligible]
            if len(eligible_pos) > TRAIN_WINDOW:
                eligible_idx = np.where(eligible)[0][-TRAIN_WINDOW:]
            else:
                eligible_idx = np.where(eligible)[0]
            if len(eligible_idx) < 300:
                continue
            model = fit_model(X_all[eligible_idx], y_all[eligible_idx], 42)
            last_fit_pos = i
        x = flatten_context(f, i).reshape(1, -1)
        pred_norm = float(model.predict(x)[0])
        pred_return = pred_norm * max(float(f["vol6"].iloc[i]), 0.0005)
        outputs.append((df.index[i], float(df["Close"].iloc[i]), pred_return, pred_norm))
    return pd.DataFrame(outputs, columns=["date", "price", "pred_return", "pred_norm"]).set_index("date")


def backtest(df: pd.DataFrame, panel: pd.DataFrame, start, end, threshold: float, max_weight: float):
    dates = [d for d in panel.index if start <= d < end]
    cash, shares = INITIAL_CASH, 0.0
    entry_bar = None
    curve, trade_returns = [], []
    last_total_before_entry = None
    for i, d in enumerate(dates):
        row = panel.loc[d]
        price = float(row.price)
        total = cash + shares * price
        signal = float(row.pred_return)
        target_weight = max_weight if signal > threshold else 0.0
        if shares > 0 and entry_bar is not None and i - entry_bar >= MAX_HOLD_BARS:
            target_weight = 0.0
        target_value = target_weight * total
        current_value = shares * price
        if target_value < current_value * 0.98 and current_value > 0:
            sell_value = current_value - target_value
            cash += sell_value * (1 - COST)
            shares -= sell_value / price
            if shares <= 1e-12:
                shares = 0.0
                if last_total_before_entry is not None:
                    trade_returns.append(total / last_total_before_entry - 1)
                entry_bar = None
                last_total_before_entry = None
        elif target_value > current_value * 1.02:
            buy_value = min(target_value - current_value, cash / (1 + COST))
            if buy_value > total * 0.01:
                before = total
                shares += buy_value / (price * (1 + COST))
                cash -= buy_value * (1 + COST)
                if entry_bar is None:
                    entry_bar = i
                    last_total_before_entry = before
        curve.append(cash + shares * price)
    if not curve:
        return 0.0, 0, 0.0, 0.0, 0.0
    curve = np.asarray(curve)
    peak = np.maximum.accumulate(curve)
    dd = float(np.min(curve / peak - 1))
    rets = curve[1:] / curve[:-1] - 1
    # Annualize using 6.5 hourly bars per session and ~252 sessions/year.
    ann = np.sqrt(252 * 6.5)
    sharpe = float(np.mean(rets) / (np.std(rets) + 1e-12) * ann) if len(rets) > 20 else 0.0
    win_rate = float(np.mean(np.asarray(trade_returns) > 0)) if trade_returns else 0.0
    return float(curve[-1] / INITIAL_CASH - 1), len(trade_returns), dd, sharpe, win_rate


def buy_and_hold(df: pd.DataFrame, start, end):
    p = df[(df.index >= start) & (df.index < end)]["Close"].astype(float)
    return float(p.iloc[-1] / p.iloc[0] - 1) if len(p) > 1 else 0.0


def main():
    df = load_data()
    n = len(df)
    # Chronological split. Validation is used only to choose the trading threshold/weight.
    train_cut = int(n * 0.60)
    val_cut = int(n * 0.80)
    # A purge gap prevents the one-step target around the split from crossing partitions.
    purge = 2
    test_start = val_cut + purge
    if n - test_start < 250:
        raise RuntimeError(f"Not enough final test data: {n-test_start} bars")

    # Build predictions only for validation + final test. Training samples always come from earlier bars.
    panel = sequential_predictions(df, train_cut, n - 1)
    if panel.empty:
        raise RuntimeError("Could not generate sequential predictions")
    val_dates = panel.index[panel.index < df.index[val_cut]]
    test_dates = panel.index[panel.index >= df.index[test_start]]
    if len(val_dates) < 100 or len(test_dates) < 100:
        raise RuntimeError("Not enough validation/test predictions")
    vstart, vend = val_dates[0], val_dates[-1]
    tstart, tend = test_dates[0], test_dates[-1]

    candidates = []
    # Threshold is on predicted next-hour return. Keep the grid modest to reduce overfitting.
    for threshold in (0.0005, 0.0010, 0.0015, 0.0020, 0.0030, 0.0040, 0.0050):
        for weight in (0.35, 0.50, 0.70, 0.90, 1.00):
            r, trades, dd, sh, wr = backtest(df, panel, vstart, vend, threshold, weight)
            score = r - 0.35 * abs(min(dd, 0)) + 0.01 * max(sh, 0)
            candidates.append((score, r, threshold, weight, trades, dd, sh, wr))
    best = max(candidates, key=lambda x: x[0])
    _, vr, threshold, weight, vt, vdd, vsh, vwr = best
    fr, ft, fdd, fsh, fwr = backtest(df, panel, tstart, tend, threshold, weight)
    bh = buy_and_hold(df, tstart, tend)

    print("\n=== V16 NVDA SEQUENTIAL 1H SUMMARY ===")
    print(f"asset={ASSET} | candles={INTERVAL} | history={PERIOD}")
    print(f"context={LOOKBACK_BARS} bars (~10 trading days) | prediction=NEXT 1H candle | max_hold={MAX_HOLD_BARS} bars")
    print(f"data={df.index[0]} -> {df.index[-1]} | total_bars={len(df)}")
    print(f"validation={vstart} -> {vend} | test={tstart} -> {tend}")
    print(f"selected threshold={threshold:.4%} | max_weight={weight:.0%}")
    print(f"validation IA={vr:+.2%} | DD={vdd:.2%} | Sharpe={vsh:.2f} | trades={vt} | win_rate={vwr:.1%}")
    print(f"FINAL IA={fr:+.2%} | {ASSET} B&H={bh:+.2%} | trades={ft} | maxDD={fdd:.2%} | Sharpe={fsh:.2f} | win_rate={fwr:.1%}")
    print(f"IA beats {ASSET} B&H: {'YES' if fr > bh else 'NO'}")


if __name__ == "__main__":
    main()
