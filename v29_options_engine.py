"""IA-Trade V29 - motor de opciones sobre la señal V27.

PAPER RESEARCH ONLY. NO BROKER. NO LIVE ORDERS.

Objetivo de V29:
- Mantener V27 intacta como cerebro de direccion para NVDA 1H.
- Convertir una señal alcista de V27 en una compra de CALL.
- Modelar prima, delta, theta, vencimiento, spread y multiplicador 100.
- Permitir comparar acciones vs opciones con el mismo punto de entrada.

IMPORTANTE:
Yahoo/yfinance da historico de OHLC de NVDA, pero no reconstruye una cadena
historica de opciones completa para cada vela. Por eso V29 tiene dos capas:
1) modo SYNTHETIC (por defecto): prima Black-Scholes + IV proxy, claramente
   etiquetada como simulacion; sirve para estudiar la arquitectura y sensibilidad.
2) modo CSV: preparado para consumir quotes historicas reales con columnas:
   timestamp,strike,expiration,type,bid,ask[,iv].

V29 NO selecciona parametros usando el TEST. Esta version es diagnostica y
no pretende demostrar rentabilidad de opciones con datos sinteticos.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import v27_session_aware_v23_plus as v27

INITIAL_CASH = 500.0
COST = v27.COST
THRESHOLD = 0.00025  # V27 benchmark: elegido previamente solo con validacion.
MAX_HOLD = v27.MAX_HOLD
CONTRACT_MULTIPLIER = 100.0
RISK_FREE = 0.0  # Deliberadamente neutral en la capa sintetica.
TRADING_DAYS_PER_YEAR = 252.0

# V29A: no se optimizan con TEST. Son configuraciones para diagnostico.
DTE_CANDIDATES = (7, 14, 21, 30)
TARGET_DELTA_CANDIDATES = (0.55, 0.60, 0.65, 0.70)
IV_MULT_CANDIDATES = (0.80, 1.00, 1.20)
SPREAD_CANDIDATES = (0.01, 0.02, 0.04)


def norm_cdf(x: float) -> float:
    return float(norm.cdf(x))


def bs_call(S: float, K: float, T: float, sigma: float, r: float = RISK_FREE) -> float:
    if not np.isfinite(S) or S <= 0 or not np.isfinite(K) or K <= 0:
        return np.nan
    if T <= 0:
        return max(S - K, 0.0)
    sigma = max(float(sigma), 1e-6)
    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def bs_call_delta(S: float, K: float, T: float, sigma: float, r: float = RISK_FREE) -> float:
    if T <= 0:
        return 1.0 if S > K else 0.0
    sigma = max(float(sigma), 1e-6)
    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    return norm_cdf(d1)


def strike_for_target_delta(S: float, T: float, sigma: float, target_delta: float, r: float = RISK_FREE) -> float:
    """Solve the Black-Scholes call strike that gives the requested delta.

    We use a binary search instead of an algebraic shortcut so the same
    function can later be replaced by a real option-chain selector.
    """
    lo = max(S * 0.20, 0.01)
    hi = S * 2.50
    target_delta = float(np.clip(target_delta, 0.05, 0.95))
    for _ in range(80):
        mid = (lo + hi) / 2.0
        d = bs_call_delta(S, mid, T, sigma, r)
        # Higher strike -> lower delta.
        if d > target_delta:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def realized_iv_proxy(df: pd.DataFrame, end_i: int, iv_mult: float) -> float:
    close = df["Close"].astype(float)
    ret = np.log(close).diff()
    window = ret.iloc[max(0, end_i - 48): end_i + 1].dropna()
    if len(window) < 12:
        base = 0.50
    else:
        hourly = float(window.std())
        base = hourly * math.sqrt(TRADING_DAYS_PER_YEAR * 6.5)
    return float(np.clip(base * iv_mult, 0.10, 2.50))


def synthetic_option_quote(df: pd.DataFrame, entry_i: int, now_i: int, dte: int, target_delta: float, iv_mult: float) -> dict:
    """Generate a synthetic quote for one rolling European call."""
    entry_ts = pd.Timestamp(df.index[entry_i])
    now_ts = pd.Timestamp(df.index[now_i])
    expiry_ts = entry_ts.normalize() + pd.Timedelta(days=dte)
    # If calendar expiry lands outside the stock series, remaining time can still be priced.
    T_days = max((expiry_ts - now_ts).total_seconds() / 86400.0, 0.0)
    T = T_days / 365.0
    S = float(df["Close"].iloc[now_i])
    sigma = realized_iv_proxy(df, now_i, iv_mult)
    K = strike_for_target_delta(S, max(T, 1.0 / 3650.0), sigma, target_delta)
    mid = bs_call(S, K, T, sigma)
    delta = bs_call_delta(S, K, T, sigma)
    return {
        "S": S,
        "K": K,
        "T": T,
        "sigma": sigma,
        "mid": max(float(mid), 0.0),
        "delta": float(delta),
        "expiry": expiry_ts,
    }


def load_option_csv(path: str | Path) -> pd.DataFrame:
    """Load real historical option quotes when available.

    Required columns: timestamp,strike,expiration,type,bid,ask
    Optional: iv
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    x = pd.read_csv(p)
    required = {"timestamp", "strike", "expiration", "type", "bid", "ask"}
    missing = required.difference(x.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True).dt.tz_convert("America/New_York")
    x["expiration"] = pd.to_datetime(x["expiration"], utc=True).dt.tz_convert("America/New_York")
    x["type"] = x["type"].str.lower()
    for c in ("strike", "bid", "ask"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    if "iv" in x.columns:
        x["iv"] = pd.to_numeric(x["iv"], errors="coerce")
    return x.dropna(subset=["timestamp", "expiration", "strike", "bid", "ask"])


def choose_real_call(chain: pd.DataFrame, ts: pd.Timestamp, dte: int, target_delta: float, stock_px: float) -> pd.Series | None:
    """Select the nearest real call quote to target DTE/delta.

    A real quote should ideally include IV. If IV is absent, a delta/strike proxy
    cannot be trusted enough for V29 selection, so we skip that row.
    """
    x = chain[(chain["timestamp"] == ts) & (chain["type"] == "call")].copy()
    if x.empty:
        return None
    x["days"] = (x["expiration"] - ts).dt.total_seconds() / 86400.0
    x = x[(x["days"] > 1) & (x["days"] <= dte * 2.0)]
    x = x[x["ask"] > 0]
    if x.empty:
        return None
    if "iv" not in x.columns:
        return None
    x = x[np.isfinite(x["iv"]) & (x["iv"] > 0)]
    if x.empty:
        return None
    # Approximate delta from BS only for selecting the nearest chain contract.
    t_years = np.maximum(x["days"].to_numpy(float) / 365.0, 1e-9)
    sigma = np.maximum(x["iv"].to_numpy(float), 0.05)
    strike = x["strike"].to_numpy(float)
    d1 = (np.log(stock_px / strike) + 0.5 * sigma * sigma * t_years) / (sigma * np.sqrt(t_years))
    x["delta_proxy"] = norm.cdf(d1)
    x["distance"] = (x["days"] - dte).abs() / max(dte, 1) + (x["delta_proxy"] - target_delta).abs()
    return x.sort_values("distance").iloc[0]


def backtest_synthetic(df: pd.DataFrame, panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                       dte: int, target_delta: float, iv_mult: float, spread: float) -> dict:
    """Backtest one long-call mapping using synthetic option prices.

    Fractional contract-equivalents are allowed intentionally so €500 can be
    studied even when one real 100-share option contract costs more than the account.
    This is NOT a statement that a real broker would accept fractional options.
    """
    positions = {ts: i for i, ts in enumerate(df.index)}
    signals = panel[(panel.index >= start) & (panel.index < end)]
    cash = float(INITIAL_CASH)
    contracts = 0.0
    entry_i = None
    entry_premium = None
    entry_cost = None
    entry_dte = dte
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

        bullish = float(row["signal"]) > THRESHOLD
        # Option mark at the next open is not available at the same close, so use next-bar stock open as S.
        S_exec = float(df["Open"].iloc[execution_i])
        if not np.isfinite(S_exec) or S_exec <= 0:
            continue

        # Build the option using the execution bar as entry anchor.
        dummy_df = df.copy()
        dummy_df.iloc[execution_i, dummy_df.columns.get_loc("Close")] = S_exec
        quote = synthetic_option_quote(dummy_df, execution_i, execution_i, entry_dte, target_delta, iv_mult)
        mid = quote["mid"]
        ask = mid * (1.0 + spread / 2.0)
        bid = mid * max(1.0 - spread / 2.0, 0.05)

        equity = cash + contracts * max(bid * CONTRACT_MULTIPLIER, 0.0)
        option_value = contracts * bid * CONTRACT_MULTIPLIER

        # Hard exit at MAX_HOLD bars or option expiry; no naked option selling.
        forced_exit = contracts > 0 and entry_i is not None and (
            execution_i - entry_i >= MAX_HOLD or quote["T"] <= 0
        )

        if contracts > 0 and (not bullish or forced_exit):
            proceeds = contracts * bid * CONTRACT_MULTIPLIER
            cash += proceeds * (1.0 - COST)
            if entry_cost and entry_cost > 0:
                trade_returns.append(cash / entry_cost - 1.0)
            contracts = 0.0
            entry_i = None
            entry_premium = None
            entry_cost = None

        elif contracts <= 0 and bullish and mid > 0:
            # Never spend more than 20% of equity on premium in this diagnostic layer.
            budget = equity * 0.20
            contract_cost = ask * CONTRACT_MULTIPLIER * (1.0 + COST)
            if contract_cost > 0 and budget > 0:
                new_contracts = budget / contract_cost
                cash -= new_contracts * contract_cost
                contracts = new_contracts
                entry_i = execution_i
                entry_premium = ask
                entry_cost = cash + contracts * entry_premium * CONTRACT_MULTIPLIER

        # Mark open options at the current bar's close.
        mark = bid
        curve.append(cash + contracts * mark * CONTRACT_MULTIPLIER)

    if len(curve) < 2:
        return {"return": 0.0, "final": INITIAL_CASH, "trades": 0, "max_dd": 0.0, "sharpe": 0.0, "win_rate": 0.0}

    curve = np.asarray(curve, dtype=float)
    peak = np.maximum.accumulate(curve)
    max_dd = float(np.min(curve / peak - 1.0))
    rets = curve[1:] / np.maximum(curve[:-1], 1e-9) - 1.0
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


def buyhold(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    p = df[(df.index >= start) & (df.index < end)]["Open"].astype(float)
    return float(p.iloc[-1] / p.iloc[0] - 1.0) if len(p) > 1 else 0.0


def main() -> None:
    t0 = time.time()
    v27.INITIAL_CASH = INITIAL_CASH
    df = v27.load_data()
    n = len(df)
    train_cut = int(n * 0.60)
    val_cut = int(n * 0.80)
    test_start = val_cut + 2

    print("=== V29 | OPTIONS ENGINE | NVDA 1H | capital=500 EUR ===", flush=True)
    print("Modo: SYNTHETIC Black-Scholes + IV proxy (NO son quotes historicas reales).", flush=True)
    print("V27 queda intacta como cerebro; V29 solo cambia el vehiculo de ejecucion.", flush=True)
    print("Generando panel V27...", flush=True)
    panel = v27.sequential_predictions(df, train_cut, n - 1)
    if panel.empty:
        raise RuntimeError("No predictions")

    val_idx = panel.index[panel.index < df.index[val_cut]]
    test_idx = panel.index[panel.index >= df.index[test_start]]
    vs, ve = val_idx[0], val_idx[-1]
    ts, te = test_idx[0], test_idx[-1]

    print(f"Validacion: {vs} -> {ve}")
    print(f"Test ciego: {ts} -> {te}")
    print(f"Threshold V27 fijo={THRESHOLD:+.3%} | MAX_HOLD={MAX_HOLD} | coste/lado={COST:.3%}")

    rows = []
    print("\n=== V29A DIAGNOSTICO | SOLO PARA COMPARAR ESTRUCTURAS ===")
    for dte in DTE_CANDIDATES:
        for delta in TARGET_DELTA_CANDIDATES:
            for ivm in IV_MULT_CANDIDATES:
                for spr in SPREAD_CANDIDATES:
                    val = backtest_synthetic(df, panel, vs, ve, dte, delta, ivm, spr)
                    test = backtest_synthetic(df, panel, ts, te, dte, delta, ivm, spr)
                    rows.append({"dte": dte, "delta": delta, "iv_mult": ivm, "spread": spr,
                                 **{f"val_{k}": v for k, v in val.items()},
                                 **{f"test_{k}": v for k, v in test.items()}})
                    print(
                        f"DTE={dte:2d} | delta={delta:.2f} | IVx={ivm:.2f} | spread={spr:.0%} | "
                        f"VAL={val['return']:+.2%} DD={val['max_dd']:.2%} trades={val['trades']:3d} | "
                        f"TEST={test['return']:+.2%} DD={test['max_dd']:.2%} trades={test['trades']:3d}"
                    )

    out = pd.DataFrame(rows)
    output_path = Path("v29_options_diagnostic.csv")
    out.to_csv(output_path, index=False)

    stock_val = v27.backtest_next_open(df, panel, vs, ve, THRESHOLD)
    stock_test = v27.backtest_next_open(df, panel, ts, te, THRESHOLD)
    bh_test = buyhold(df, ts, te)

    print("\n=== REFERENCIA | MISMA IA SOBRE ACCIONES ===")
    print(f"VAL stock: {stock_val['return']:+.2%} | DD={stock_val['max_dd']:.2%} | trades={stock_val['trades']}")
    print(f"TEST stock: {stock_test['return']:+.2%} | DD={stock_test['max_dd']:.2%} | trades={stock_test['trades']}")
    print(f"TEST B&H:   {bh_test:+.2%}")
    print("\n=== ARCHIVO ===")
    print(f"Guardado {output_path}")
    print("NO escoger configuracion ganadora con el TEST. La seleccion robusta vendra en V30 sobre VALIDACION.")
    print("runtime={:.1f} min".format((time.time() - t0) / 60.0))


if __name__ == "__main__":
    main()
