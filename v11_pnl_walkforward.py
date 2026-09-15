"""IA-Trade v11 - PnL-aware walk-forward research.

Paper/simulation only. The goal is not directional accuracy; it is robust
out-of-sample PnL after costs. This version removes the old 09:57 rule and
uses time-ordered walk-forward evaluation, rank/return signals, costs and
next-bar execution to reduce look-ahead bias.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

warnings.filterwarnings("ignore")

TICKERS = ["AAPL", "MSFT", "SPY", "NVDA"]
PERIOD = "5y"
HORIZONS = (1, 3, 5, 10)
COST = 0.001
INITIAL_CASH = 50.0
LOOKBACK = 40


def features(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    close, high, low, op, vol = [df[c].astype(float) for c in ["Close", "High", "Low", "Open", "Volume"]]
    lr = np.log(close).diff()
    x["ret1"] = lr
    x["range"] = (high - low) / close
    x["body"] = (close - op) / op
    x["vol_chg"] = np.log1p(vol).diff()
    for n in (3, 5, 8, 13, 21, 40):
        x[f"mom{n}"] = close.pct_change(n)
        x[f"ema{n}"] = close / close.ewm(span=n, adjust=False).mean() - 1
        x[f"vol{n}"] = lr.rolling(n).std()
    x["range_pos20"] = (close - low.rolling(20).min()) / (high.rolling(20).max() - low.rolling(20).min() + 1e-9)
    x["volume_ratio20"] = vol / (vol.rolling(20).mean() + 1e-9)
    return x.replace([np.inf, -np.inf], np.nan)


def make_targets(close: pd.Series) -> pd.DataFrame:
    return pd.concat({f"y{h}": close.shift(-h) / close - 1 for h in HORIZONS}, axis=1)


def fit_predict(train_x, train_y, pred_x):
    # Squared-error regression predicts magnitude. Huber loss makes the model
    # less dominated by a few extreme market days.
    base = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.035, max_leaf_nodes=15,
        min_samples_leaf=20, l2_regularization=1.0, loss="huber", random_state=42
    )
    model = MultiOutputRegressor(base)
    model.fit(train_x, train_y)
    return model.predict(pred_x)


def walk_forward(df: pd.DataFrame):
    x = features(df)
    y = make_targets(df["Close"])
    data = x.join(y).dropna()
    X = data[x.columns].values
    Y = data[y.columns].values
    closes = df.loc[data.index, "Close"].values
    dates = data.index
    n = len(data)
    warm = max(260, int(n * 0.50))
    step = max(20, int(n * 0.05))
    preds, actual, px, pdts = [], [], [], []

    # Expanding-window walk-forward: each prediction is made using only past data.
    for end in range(warm, n, step):
        stop = min(end + step, n)
        pred = fit_predict(X[:end], Y[:end], X[end:stop])
        preds.append(pred)
        actual.append(Y[end:stop])
        px.extend(closes[end:stop])
        pdts.extend(dates[end:stop])
    return np.vstack(preds), np.vstack(actual), np.asarray(px), pdts


def signal_returns(pred):
    # Medium horizons carry more weight because one-day noise is high.
    w = np.array([0.10, 0.25, 0.40, 0.25])
    expected = pred @ w
    dispersion = np.std(pred, axis=1)
    confidence = np.abs(expected) / (dispersion + 1e-5)
    return expected, confidence


def backtest(prices, expected, confidence, threshold):
    cash = INITIAL_CASH
    shares = 0.0
    equity = []
    trades = 0
    for i in range(len(prices) - 1):
        p = prices[i]
        score = expected[i]
        # Enter/exit at the NEXT close. This avoids executing on the same price
        # that was used to form today's features.
        if shares == 0 and score > threshold and confidence[i] >= 0.25:
            risk_budget = min(cash * 0.35, cash)
            shares = risk_budget / (p * (1 + COST))
            cash -= shares * p * (1 + COST)
            trades += 1
        elif shares > 0 and score < -threshold * 0.35:
            cash += shares * p * (1 - COST)
            shares = 0.0
            trades += 1
        equity.append(cash + shares * p)
    final = cash + shares * prices[-1] * (1 - COST if shares else 1)
    equity.append(final)
    curve = np.asarray(equity)
    ret = final / INITIAL_CASH - 1
    peak = np.maximum.accumulate(curve)
    dd = np.min(curve / peak - 1)
    return final, ret, trades, dd


def main():
    all_results = []
    for ticker in TICKERS:
        print(f"\n=== {ticker} ===", flush=True)
        df = yf.download(ticker, period=PERIOD, interval="1d", auto_adjust=True, progress=False)
        if df.empty:
            print("Sin datos", flush=True)
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        pred, actual, prices, dates = walk_forward(df)
        expected, confidence = signal_returns(pred)
        # Tune threshold only on the first half of the walk-forward predictions.
        cut = len(prices) // 2
        candidates = np.linspace(0.001, 0.02, 20)
        scores = []
        for t in candidates:
            _, r, _, _ = backtest(prices[:cut], expected[:cut], confidence[:cut], t)
            scores.append(r)
        threshold = float(candidates[int(np.argmax(scores))])
        final, ret, trades, dd = backtest(prices[cut:], expected[cut:], confidence[cut:], threshold)
        actual_ret = np.average(actual[cut:], axis=0)
        pred_ret = np.average(pred[cut:], axis=0)
        direction = np.mean(np.sign(pred[cut:, 1]) == np.sign(actual[cut:, 1]))
        bh = INITIAL_CASH * (prices[-1] / prices[cut])
        bh_ret = bh / INITIAL_CASH - 1
        print(f"threshold={threshold:.4f} | pred5={pred_ret[2]:+.3%} | real5={actual_ret[2]:+.3%}")
        print(f"direction_3d={direction:.2%} | IA={ret:+.2%} | B&H={bh_ret:+.2%} | trades={trades} | maxDD={dd:.2%}")
        all_results.append((ret, bh_ret))

    if all_results:
        ia = np.mean([x[0] for x in all_results])
        bh = np.mean([x[1] for x in all_results])
        print("\n=== V11 SUMMARY ===")
        print(f"Mean IA return: {ia:+.2%}")
        print(f"Mean B&H return: {bh:+.2%}")
        print(f"IA beats B&H: {'YES' if ia > bh else 'NO'}")


if __name__ == "__main__":
    main()
