"""Punto de entrada de IA-Trade.

Primera fase: investigación y backtesting. No conecta con ningún broker.
"""

from config import INTERVAL, PERIOD, TICKER
from market import download_market_data
from backtest import run_backtest, summarize


def main() -> None:
    print(f"Descargando {TICKER} ({PERIOD}, {INTERVAL})...")
    data = download_market_data(TICKER, PERIOD, INTERVAL)
    results = run_backtest(data)
    stats = summarize(results)

    print("\n=== IA-Trade | Backtest ===")
    print(f"Activo:             {TICKER}")
    print(f"Capital inicial:    {stats['initial_value']:.2f} €")
    print(f"Valor final:        {stats['final_value']:.2f} €")
    print(f"Rentabilidad:       {stats['return_pct']:.2f} %")
    print(f"Máximo drawdown:    {stats['max_drawdown_pct']:.2f} %")
    print("\nModo: SIMULACIÓN. No se ha enviado ninguna orden real.")


if __name__ == "__main__":
    main()
