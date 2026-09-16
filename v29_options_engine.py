"""IA-Trade V29 - motor de opciones sobre la señal V27.

PAPER RESEARCH ONLY. NO BROKER. NO LIVE ORDERS.

Objetivo de V29:
- Mantener V27 intacta como cerebro de direccion para NVDA 1H.
- Convertir una señal alcista de V27 en una compra de CALL.
- Modelar prima, delta, vencimiento, spread, theta y multiplicador 100.
- Comparar acciones vs opciones usando exactamente la misma señal.

DATOS:
Yahoo/yfinance ofrece OHLC historico de NVDA, pero no reconstruye una cadena
historica completa de opciones para cada vela. Por eso V29 tiene un modo
SYNTHETIC (Black-Scholes + IV proxy) claramente identificado y deja preparada
una interfaz CSV para quotes historicas reales en una fase posterior.

Esta version NO usa el TEST para elegir una configuracion. Es diagnostica:
primero estudiamos si el vehiculo opcion mantiene la ventaja de la señal.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

import v27_session_aware_v23_plus as v27

INITIAL_CASH = 500.0
COST = v27.COST
THRESHOLD = 0.00025  # V27 benchmark; elegido previamente solo con validacion.
MAX_HOLD = v27.MAX_HOLD
CONTRACT_MULTIPLIER = 100.0
RISK_FREE = 0.0  # Neutro en la capa sintetica.
TRADING_DAYS_PER_YEAR = 252.0
HOURS_PER_TRADING_DAY = 6.5

# V29A: diagnostico; NO se selecciona un ganador usando el TEST.
DTE_CANDIDATES = (7, 14, 21, 30)
TARGET_DELTA_CANDIDATES = (0.55, 0.60, 0.65, 0.70)
IV_MULT_CANDIDATES = (0.80, 1.00, 1.20)
SPREAD_CANDIDATES = (0.01, 0.02, 0.04)
PREMIUM_BUDGET = 0.20  # Como maximo 20% del equity en la prima de una entrada.


def norm_cdf(x: float) -> float:
    """Normal CDF without adding another package dependency."""
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


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
    """Solve the strike producing the desired call delta at entry."""
    lo = max(S * 0.20, 0.01)
    hi = S * 2.50
    target_delta = float(np.clip(target_delta, 0.05, 0.95))
    for _ in range(80):
        mid = (lo + hi) / 2.0
        delta = bs_call_delta(S, mid, T, sigma, r)
        if delta > target_delta:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def session_expiry(entry_ts: pd.Timestamp, dte: int) -> pd.Timestamp:
    """Synthetic expiry at 16:00 NY time after DTE calendar days."""
    day = entry_ts.normalize() + pd.Timedelta(days=int(dte))
    return day + pd.Timedelta(hours=16)


def realized_iv_proxy(df: pd.DataFrame, end_i: int, iv_mult: float) -> float:
    """Annualized realized volatility proxy from the last ~48 hourly bars."""
    close = df["Close"].astype(float)
    ret = np.log(close).diff()
    window = ret.iloc[max(0, end_i - 48): end_i + 1].dropna()
    if len(window) < 12:
        base = 0.50
    else:
        hourly = float(window.std())
        base = hourly * math.sqrt(TRADING_DAYS_PER_YEAR * HOURS_PER_TRADING_DAY)
    return float(np.clip(base * iv_mult, 0.10, 2.50))


def option_mark(S: float, K: float, expiry_ts: pd.Timestamp, now_ts: pd.Timestamp, sigma: float) -> dict:
    remaining_days = max((expiry_ts - now_ts).total_seconds() / 86400.0, 0.0)
    T = remaining_days / 365.0
    mid = bs_call(S, K, T, sigma)
    delta = bs_call_delta(S, K, T, sigma)
    return {"mid": max(float(mid), 0.0), "delta": float(delta), "T": T, "sigma": sigma}


def load_option_csv(path: str | Path) -> pd.DataFrame:
    """Load real historical option quotes when a suitable data source is added.

    Required columns:
    timestamp,strike,expiration,type,bid,ask
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
    x["type"] = x["type"].astype(str).str.lower()
    for c in ("strike", "bid", "ask"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    if "iv" in x.columns:
        x["iv"] = pd.to_numeric(x["iv"], errors="coerce")
    return x.dropna(subset=["timestamp", "expiration", "strike", "bid", "ask"])


def backtest_synthetic(df: pd.DataFrame, panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                       dte: int, target_delta: float, iv_mult: float, spread: float) -> dict:
    """Backtest a long-call mapping using synthetic option prices.

    Fractional contract-equivalents are intentional for the €500 research account:
    a real listed stock option normally represents 100 shares and a broker does not
    have to accept fractional option contracts. This mode is therefore an economic
    sensitivity study, not a claim of live executability.
    """
    positions = {ts: i for i, ts in enumerate(df.index)}
    signals = panel[(panel.index >= start) & (panel.index < end)]

    cash = float(INITIAL_CASH)
    contracts = 0.0
    entry_i: int | None = None
    entry_cost = 0.0
    strike = np.nan
    expiry_ts: pd.Timestamp | None = None
    trade_returns: list[float] = []
    curve: list[float] = []

    for signal_ts, row in signals.iterrows():
        i = positions.get(signal_ts)
        if i is None or i + 1 >= len(df):
            continue
        execution_i = i + 1
        execution_ts = df.index[execution_i]
        if execution_ts >= end:
            continue

        S_exec = float(df["Open"].iloc[execution_i])
        if not np.isfinite(S_exec) or S_exec <= 0:
            continue

        bullish = float(row["signal"]) > THRESHOLD
        now_sigma = realized_iv_proxy(df, execution_i, iv_mult)

        # Mark existing position using the SAME strike and SAME expiry acquired at entry.
        if contracts > 0 and entry_i is not None and expiry_ts is not None:
            current = option_mark(S_exec, strike, expiry_ts, execution_ts, now_sigma)
            bid = current["mid"] * max(1.0 - spread / 2.0, 0.05)
            forced_exit = execution_i - entry_i >= MAX_HOLD or current["T"] <= 0
            if (not bullish) or forced_exit:
                proceeds = contracts * bid * CONTRACT_MULTIPLIER * (1.0 - COST)
                cash += proceeds
                trade_returns.append(proceeds / max(entry_cost, 1e-9) - 1.0)
                contracts = 0.0
                entry_i = None
                entry_cost = 0.0
                strike = np.nan
                expiry_ts = None

        # Open a new call only when flat and V27 says bullish.
        if contracts <= 0 and bullish:
            expiry = session_expiry(execution_ts, dte)
            T_entry = max((expiry - execution_ts).total_seconds() / 86400.0, 1.0) / 365.0
            K = strike_for_target_delta(S_exec, T_entry, now_sigma, target_delta)
            entry_mid = bs_call(S_exec, K, T_entry, now_sigma)
            ask = entry_mid * (1.0 + spread / 2.0)
            if np.isfinite(ask) and ask > 0:
                equity = cash
                budget = equity * PREMIUM_BUDGET
                contract_cost = ask * CONTRACT_MULTIPLIER * (1.0 + COST)
                qty = budget / contract_cost if contract_cost > 0 else 0.0
                if qty > 0:
                    total_cost = qty * contract_cost
                    cash -= total_cost
                    contracts = qty
                    entry_i = execution_i
                    entry_cost = total_cost
                    strike = K
                    expiry_ts = expiry

        # Mark the open option with the current bar's close.
        mark_value = 0.0
        if contracts > 0 and expiry_ts is not None:
            S_mark = float(df["Close"].iloc[execution_i])
            mark_sigma = realized_iv_proxy(df, execution_i, iv_mult)
            marked = option_mark(S_mark, strike, expiry_ts, execution_ts, mark_sigma)
            bid = marked["mid"] * max(1.0 - spread / 2.0, 0.05)
            mark_value = contracts * bid * CONTRACT_MULTIPLIER
        curve.append(cash + mark_value)

    # Conservative final liquidation at the last bar available before end.
    if contracts > 0 and expiry_ts is not None:
        last_candidates = df[(df.index >= start) & (df.index < end)]
        if not last_candidates.empty:
            last_ts = last_candidates.index[-1]
            last_i = positions[last_ts]
            S_last = float(df["Close"].iloc[last_i])
            sigma_last = realized_iv_proxy(df, last_i, iv_mult)
            marked = option_mark(S_last, strike, expiry_ts, last_ts, sigma_last)
            bid = marked["mid"] * max(1.0 - spread / 2.0, 0.05)
            proceeds = contracts * bid * CONTRACT_MULTIPLIER * (1.0 - COST)
            cash += proceeds
            trade_returns.append(proceeds / max(entry_cost, 1e-9) - 1.0)
            contracts = 0.0
            curve.append(cash)

    if len(curve) < 2:
        return {"return": 0.0, "final": INITIAL_CASH, "trades": 0,
                "max_dd": 0.0, "sharpe": 0.0, "win_rate": 0.0}

    curve_arr = np.asarray(curve, dtype=float)
    peak = np.maximum.accumulate(curve_arr)
    max_dd = float(np.min(curve_arr / np.maximum(peak, 1e-9) - 1.0))
    rets = curve_arr[1:] / np.maximum(curve_arr[:-1], 1e-9) - 1.0
    sharpe = float(np.mean(rets) / (np.std(rets) + 1e-12) * math.sqrt(252 * 6.5)) if len(rets) > 20 else 0.0
    win_rate = float(np.mean(np.asarray(trade_returns) > 0)) if trade_returns else 0.0
    final = float(curve_arr[-1])
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
    print("Modo: SYNTHETIC Black-Scholes + IV proxy.", flush=True)
    print("NO son precios historicos reales de opciones y NO hay broker/live orders.", flush=True)
    print("V27 queda intacta como cerebro; V29 cambia solo el vehiculo de ejecucion.", flush=True)
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
    print(f"Premium budget={PREMIUM_BUDGET:.0%} | multiplicador={CONTRACT_MULTIPLIER:.0f}")

    rows = []
    print("\n=== V29A DIAGNOSTICO | COMPARACION DE ESTRUCTURAS ===")
    for dte in DTE_CANDIDATES:
        for delta in TARGET_DELTA_CANDIDATES:
            for ivm in IV_MULT_CANDIDATES:
                for spr in SPREAD_CANDIDATES:
                    val = backtest_synthetic(df, panel, vs, ve, dte, delta, ivm, spr)
                    test = backtest_synthetic(df, panel, ts, te, dte, delta, ivm, spr)
                    rows.append({
                        "dte": dte,
                        "delta": delta,
                        "iv_mult": ivm,
                        "spread": spr,
                        **{f"val_{k}": v for k, v in val.items()},
                        **{f"test_{k}": v for k, v in test.items()},
                    })
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
    print("NO escoger configuracion con el TEST. La seleccion robusta vendra despues sobre VALIDACION.")
    print("runtime={:.1f} min".format((time.time() - t0) / 60.0))


if __name__ == "__main__":
    main()
