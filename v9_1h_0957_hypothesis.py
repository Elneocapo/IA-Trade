"""IA-Trade v9: prueba de la hipótesis 09:57 ET + vela 1H roja.

Objetivo: aislar primero la hipótesis de Cardona como anomalía estadística,
sin machine learning ni optimización sobre el test.

IMPORTANTE SOBRE DATOS:
Yahoo/yfinance limita las velas intradía de 5m a los últimos ~60 días.
Además, una vela de 5m que termina a las 10:00 no puede utilizarse para
representar exactamente el estado a las 09:57 sin introducir look-ahead.
Por eso esta versión usa dos proxies conservadores disponibles antes de
09:57:
  A) precio de apertura de la vela 09:55 (disponible a las 09:55)
  B) cierre de la vela 09:50 (disponible a las 09:55)

La vela 1H se define como 09:30-10:30 ET. "Roja" significa que el precio
observado por el proxy está por debajo del open de las 09:30.

No se simulan opciones ni se inventan primas. Se estudia primero el subyacente.
Modo exclusivamente SIMULACIÓN / INVESTIGACIÓN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "SPY", "QQQ"]
PERIOD = "60d"
INTERVAL = "5m"
INITIAL_CASH = 50.0
ROUND_TRIP_COST = 0.002  # 0.1% entrada + 0.1% salida, deliberadamente conservador
HORIZONS = (2, 3, 5)  # sesiones desde la señal


def download_5m(ticker: str) -> pd.DataFrame:
    data = yf.download(
        ticker, period=PERIOD, interval=INTERVAL,
        auto_adjust=True, progress=False, prepost=False,
    )
    if data.empty:
        raise RuntimeError(f"Sin datos para {ticker}")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    if data.index.tz is None:
        data.index = data.index.tz_localize("America/New_York")
    else:
        data.index = data.index.tz_convert("America/New_York")
    return data


def build_daily_signals(data: pd.DataFrame) -> pd.DataFrame:
    regular = data.between_time("09:30", "16:00")
    rows = []
    for date, day in regular.groupby(regular.index.date):
        day = day.sort_index()
        # Necesitamos explícitamente las barras 09:30, 09:50 y 09:55.
        def bar_at(hhmm: str):
            x = day[day.index.strftime("%H:%M") == hhmm]
            return x.iloc[0] if not x.empty else None

        b0930 = bar_at("09:30")
        b0950 = bar_at("09:50")
        b0955 = bar_at("09:55")
        if b0930 is None or b0950 is None or b0955 is None:
            continue

        open_1h = float(b0930["Open"])
        p_proxy_a = float(b0955["Open"])
        p_proxy_b = float(b0950["Close"])
        close_day = float(day.iloc[-1]["Close"])

        # Future daily closes. Nunca se usan para construir la señal.
        rows.append({
            "date": pd.Timestamp(date),
            "open_0930": open_1h,
            "price_0955_open": p_proxy_a,
            "price_0950_close": p_proxy_b,
            "red_a": p_proxy_a < open_1h,
            "red_b": p_proxy_b < open_1h,
            "close_day": close_day,
            "day_high": float(day["High"].max()),
            "day_low": float(day["Low"].min()),
        })

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return out


def add_forward_returns(signals: pd.DataFrame) -> pd.DataFrame:
    out = signals.copy()
    for h in HORIZONS:
        # Entrada a precio proxy y salida al cierre de la sesión h.
        future_close = out["close_day"].shift(-h)
        out[f"ret_a_{h}d"] = future_close / out["price_0955_open"] - 1
        out[f"ret_b_{h}d"] = future_close / out["price_0950_close"] - 1
        # MFE/MAE aproximados con máximos/mínimos diarios disponibles después
        # de la señal. Se calcula en una segunda función para no mezclar tiempos.
    return out


def summarize_asset(signals: pd.DataFrame, ticker: str) -> dict:
    # Solo filas con suficientes días futuros para el horizonte analizado.
    result = {"ticker": ticker, "days": len(signals)}
    for proxy in ("a", "b"):
        red = signals[f"red_{proxy}"]
        result[f"red_rate_{proxy}"] = float(red.mean())
        for h in HORIZONS:
            r = signals[f"ret_{proxy}_{h}d"].dropna()
            red_r = signals.loc[red, f"ret_{proxy}_{h}d"].dropna()
            green_r = signals.loc[~red, f"ret_{proxy}_{h}d"].dropna()
            result[f"all_{proxy}_{h}d"] = float(r.mean()) if len(r) else np.nan
            result[f"red_{proxy}_{h}d"] = float(red_r.mean()) if len(red_r) else np.nan
            result[f"green_{proxy}_{h}d"] = float(green_r.mean()) if len(green_r) else np.nan
            result[f"red_n_{proxy}_{h}d"] = int(len(red_r))
            # PUT-like economic direction: negative underlying return is good.
            result[f"put_edge_{proxy}_{h}d"] = float(-red_r.mean()) if len(red_r) else np.nan
    return result


def bootstrap_ci(values: np.ndarray, seed: int = 42, n: int = 5000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for i in range(n):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def pooled_analysis(all_signals: dict[str, pd.DataFrame]) -> None:
    print("\n=== POOLED ANALYSIS ===")
    print("La unidad estadística principal es el día-activo. No se seleccionan parámetros con estos resultados.")
    for proxy in ("a", "b"):
        print(f"\n--- Proxy {proxy.upper()} ---")
        for h in HORIZONS:
            red_values = []
            green_values = []
            all_values = []
            for df in all_signals.values():
                red_values.extend(df.loc[df[f"red_{proxy}"], f"ret_{proxy}_{h}d"].dropna().tolist())
                green_values.extend(df.loc[~df[f"red_{proxy}"], f"ret_{proxy}_{h}d"].dropna().tolist())
                all_values.extend(df[f"ret_{proxy}_{h}d"].dropna().tolist())
            red = np.asarray(red_values, dtype=float)
            green = np.asarray(green_values, dtype=float)
            all_r = np.asarray(all_values, dtype=float)
            red_mean = red.mean() if len(red) else np.nan
            green_mean = green.mean() if len(green) else np.nan
            all_mean = all_r.mean() if len(all_r) else np.nan
            ci_lo, ci_hi = bootstrap_ci(red - green[: len(red)] if len(green) == len(red) else red)
            # El CI anterior no es un contraste emparejado; se muestra solo como CI de la media roja.
            red_ci_lo, red_ci_hi = bootstrap_ci(red)
            print(
                f"{h}d | N roja={len(red):3d} | roja={red_mean*100:+.3f}% | "
                f"verde={green_mean*100:+.3f}% | total={all_mean*100:+.3f}% | "
                f"CI95 media roja=[{red_ci_lo*100:+.3f}%, {red_ci_hi*100:+.3f}%]"
            )


def economic_score(signals: pd.DataFrame, proxy: str, horizon: int) -> tuple[float, int, float]:
    """Backtest simple de PUT-like: solo entra cuando la vela es roja.

    Se usa el retorno del subyacente invertido como proxy de dirección PUT.
    NO es un precio de opción y no debe interpretarse como rentabilidad de una opción.
    """
    r = signals.loc[signals[f"red_{proxy}"], f"ret_{proxy}_{horizon}d"].dropna()
    if r.empty:
        return np.nan, 0, np.nan
    # Coste redondo conservador sobre el subyacente.
    net = -r - ROUND_TRIP_COST
    return float(net.mean()), int(len(net)), float(np.mean(net > 0))


def main() -> None:
    print("=== IA-Trade v9 | HIPÓTESIS 09:57 ET + 1H ROJA ===")
    print("Objetivo: comprobar primero si existe anomalía estadística antes de entrenar otra IA.")
    print("Datos: Yahoo Finance 5m, ventana máxima reciente (~60 días).")
    print("Proxy A: precio de apertura 09:55; Proxy B: cierre 09:50.")
    print("Ambos proxies están disponibles antes de 09:57 y evitan usar el cierre 10:00.")
    print("IMPORTANTE: no es todavía un backtest de opciones.\n")

    summaries = []
    all_signals = {}
    for ticker in TICKERS:
        print(f"Descargando {ticker}...")
        data = download_5m(ticker)
        signals = add_forward_returns(build_daily_signals(data))
        all_signals[ticker] = signals
        s = summarize_asset(signals, ticker)
        summaries.append(s)
        print(
            f"  {ticker}: {s['days']} sesiones | roja A={s['red_rate_a']*100:.1f}% | "
            f"roja B={s['red_rate_b']*100:.1f}%"
        )
        for proxy in ("a", "b"):
            for h in HORIZONS:
                print(
                    f"    {proxy.upper()} {h}d: roja={s[f'red_{proxy}_{h}d']*100:+.3f}% "
                    f"(n={s[f'red_n_{proxy}_{h}d']}) | "
                    f"verde={s[f'green_{proxy}_{h}d']*100:+.3f}%"
                )

    pooled_analysis(all_signals)

    print("\n=== ECONOMIC SCREEN (SUBYACENTE, NO OPCIONES) ===")
    for proxy in ("a", "b"):
        for h in HORIZONS:
            values = []
            wins = []
            for df in all_signals.values():
                score, n, win = economic_score(df, proxy, h)
                if np.isfinite(score):
                    values.extend((-df.loc[df[f"red_{proxy}"], f"ret_{proxy}_{h}d"] - ROUND_TRIP_COST).dropna().tolist())
            if values:
                arr = np.asarray(values)
                print(
                    f"{proxy.upper()} {h}d | media PUT-like={arr.mean()*100:+.3f}% | "
                    f"mediana={np.median(arr)*100:+.3f}% | tasa positiva={np.mean(arr>0)*100:.1f}% | N={len(arr)}"
                )

    print("\n=== DIAGNÓSTICO ===")
    print("🟡 V9 es una prueba de hipótesis, no una IA y no una estrategia de opciones.")
    print("🟡 La muestra intradía está limitada por la disponibilidad histórica de Yahoo.")
    print("🔴 No se debe declarar válida la regla 09:57 ET hasta replicarla con una fuente histórica de mayor profundidad.")
    print("\nSiguiente decisión experimental: si ambos proxies muestran la misma anomalía, diseñar una v10 que añada contexto 1H (tendencia/volatilidad) sin tocar un test ciego reservado.")


if __name__ == "__main__":
    main()
