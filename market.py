"""Descarga y prepara datos de mercado."""

import pandas as pd
import yfinance as yf


def download_market_data(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Devuelve OHLCV limpio para un ticker."""
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)

    if data.empty:
        raise RuntimeError(f"No se pudieron descargar datos para {ticker}.")

    # yfinance puede devolver columnas MultiIndex incluso para un solo ticker.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise RuntimeError(f"Faltan columnas de mercado: {missing}")

    return data[required].dropna().copy()
