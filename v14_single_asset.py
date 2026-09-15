"""IA-Trade v14 - single-asset paper trading research.

The model learns ONLY from the configured ASSET and the paper strategy can
ONLY hold/trade that same asset. To switch the AI to another asset later,
change ASSET (for example, from "NVDA" to "AAPL").
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")

# ============================================================
# ONE LINE TO CHANGE THE ASSET IN THE FUTURE.
# The current AI is deliberately NVIDIA-only.
# ============================================================
ASSET = "NVDA"

PERIOD = "10y"
HORIZON = 5
COST = 0.001
INITIAL_CASH = 50.0


def features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    o = df["Open"].astype(float)
    v = df["Volume"].astype(float)

    z = pd.DataFrame(index=df.index)
    lr = np.log(c).diff()
    z["ret1"] = lr
    z["range"] = (h - l) / c
    z["body"] = (c - o) / o
    z["close_location"] = (c - l) / (h - l + 1e-9)
    z["volchg"] = np.log1p(v).diff()

    for n in (2, 3, 5, 8, 13, 21, 34, 55):
        z[f"mom{n}"] = c.pct_change(n)
        z[f"ema{n}"] = c / c.ewm(span=n, adjust=False).mean() - 1
        z[f"vol{n}"] = lr.rolling(n).std()

    for n in (10, 20, 50):
        lo = l.rolling(n).min()
        hi = h.rolling(n).max()
        z[f"range_pos{n}"] = (c - lo) / (hi - lo + 1e-9)
        z[f"volume_ratio{n}"] = v / (v.rolling(n).mean() + 1e-9)

    # Volatility regime and trend structure, all computed from NVDA only.
    z["vol_ratio_short_long"] = z["vol8"] / (z["vol34"] + 1e-9)
    z["ema_fast_slow"] = c.ewm(span=13, adjust=False).mean() / c.ewm(span=55, adjust=False).mean() - 1

    return z.replace([np.inf, -np.inf], np.nan)


def walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    x = features(df)
    c = df["Close"].astype(float)
    vol = x["vol21"]
    future = c.shift(-HORIZON) / c - 1

    # The model predicts NVDA's forward return normalized by NVDA's own
    # recent volatility. No other ticker enters the training set.
    y = future / (vol * np.sqrt(HORIZON) + 1e-6)
    d = x.assign(y=y, future=future).dropna()
    X = d[x.columns].values
    Y = d["y"].values

    n = len(d)
    warm = max(400, int(n * 0.50))
    step = max(20, int(n * 0.05))
    out = []

    for end in range(warm, n, step):
        stop = min(end + step, n)
        model = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.03,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2_regularization=2.0,
            loss="absolute_error",
            random_state=42,
        )
        model.fit(X[:end], Y[:end])
        pred = model.predict(X[end:stop])

        for dt, score in zip(d.index[end:stop], pred):
            vv = max(float(vol.loc[dt]), 0.003)
            out.append((dt, float(score), float(c.loc[dt]), vv))

    return pd.DataFrame(
        out, columns=["date", "score", "price", "vol"]
    ).set_index("date")


def load_data() -> pd.DataFrame:
    print(f"=== TRAINING ASSET: {ASSET} ===", flush=True)
    df = yf.download(
        ASSET,
        period=PERIOD,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise RuntimeError(f"No data returned for {ASSET}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def backtest(
    panel: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    score_min: float,
    invest: float,
) -> tuple[float, int, float, float]:
    dates = [d for d in panel.index if start <= d < end]
    cash = INITIAL_CASH
    shares = 0.0
    curve = []
    trades = 0

    for d in dates:
        row = panel.loc[d]
        price = float(row.price)
        total = cash + shares * price

        # One position only: ASSET itself. No cross-asset ranking is possible.
        target_weight = invest if float(row.score) > score_min else 0.0
        target_value = target_weight * total
        current_value = shares * price

        if target_value < current_value * 0.98:
            sell_value = current_value - target_value
            sell_value = min(sell_value, current_value)
            cash += sell_value * (1 - COST)
            shares -= sell_value / price
            trades += 1

        if target_value > current_value * 1.02:
            buy_value = min(target_value - current_value, cash / (1 + COST))
            if buy_value > total * 0.01:
                shares += buy_value / (price * (1 + COST))
                cash -= buy_value * (1 + COST)
                trades += 1

        curve.append(cash + shares * price)

    if not curve:
        return 0.0, 0, 0.0, 0.0

    curve = np.asarray(curve)
    peak = np.maximum.accumulate(curve)
    max_dd = float(np.min(curve / peak - 1))
    daily = curve[1:] / curve[:-1] - 1
    sharpe = (
        float(np.mean(daily) / (np.std(daily) + 1e-12) * np.sqrt(252))
        if len(daily) > 10
        else 0.0
    )
    return float(curve[-1] / INITIAL_CASH - 1), trades, max_dd, sharpe


def buy_and_hold(panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    p = panel[(panel.index >= start) & (panel.index < end)].price
    return float(p.iloc[-1] / p.iloc[0] - 1) if len(p) > 1 else 0.0


def main() -> None:
    df = load_data()
    panel = walk_forward(df)
    dates = sorted(panel.index)
    if len(dates) < 100:
        raise RuntimeError("Not enough out-of-sample observations")

    cut = dates[len(dates) // 2]
    end = dates[-1]

    # Policy selection happens ONLY on the first half.
    # The second half is never used to choose the threshold.
    candidates = []
    for score_min in (0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00):
        for invest in (0.50, 0.70, 0.90, 1.00):
            r, trades, dd, sharpe = backtest(
                panel, dates[0], cut, score_min, invest
            )
            utility = r - 0.30 * abs(min(dd, 0)) + 0.02 * max(sharpe, 0)
            candidates.append(
                (utility, r, score_min, invest, dd, sharpe, trades)
            )

    _, vr, score_min, invest, vdd, vsh, _ = max(
        candidates, key=lambda x: x[0]
    )
    final_return, trades, final_dd, final_sharpe = backtest(
        panel, cut, end, score_min, invest
    )
    bh = buy_and_hold(panel, cut, end)

    print("\n=== V14 NVDA-ONLY SUMMARY ===")
    print(f"asset={ASSET} | horizon={HORIZON}d | data={PERIOD}")
    print(f"selected: min_normalized_score={score_min:.2f} | invest={invest:.0%}")
    print(f"validation IA={vr:+.2%} | DD={vdd:.2%} | Sharpe={vsh:.2f}")
    print(
        f"FINAL IA={final_return:+.2%} | {ASSET} B&H={bh:+.2%} | "
        f"trades={trades} | maxDD={final_dd:.2%} | Sharpe={final_sharpe:.2f}"
    )
    print(f"IA beats {ASSET} B&H: {'YES' if final_return > bh else 'NO'}")
    print(f"\nFuture asset change: edit ASSET = \"{ASSET}\" at the top of this file.")


if __name__ == "__main__":
    main()
