"""Punto de entrada de IA-Trade.

Segunda fase: modelo de machine learning + backtesting fuera de muestra.
No conecta con ningún broker.
"""

from config import INTERVAL, PERIOD, TICKER
from market import download_market_data
from ml_backtest import run_ml_portfolio_backtest


def main() -> None:
    print(f"Descargando {TICKER} ({PERIOD}, {INTERVAL})...")
    data = download_market_data(TICKER, PERIOD, INTERVAL)
    _, stats = run_ml_portfolio_backtest(data)

    print("\n=== IA-Trade | Machine Learning Backtest ===")
    print(f"Activo:             {TICKER}")
    print(f"Periodo de prueba:  {stats['test_start'].date()} ({stats['test_days']} días)")
    print(f"Capital inicial:    {stats['initial_value']:.2f} €")
    print(f"Valor final:        {stats['final_value']:.2f} €")
    print(f"Rentabilidad:       {stats['return_pct']:.2f} %")
    print(f"Máximo drawdown:    {stats['max_drawdown_pct']:.2f} %")
    print(f"Acierto dirección:  {stats['accuracy_pct']:.2f} %")
    print("\nModo: SIMULACIÓN. El modelo no ha enviado ninguna orden real.")


if __name__ == "__main__":
    main()
