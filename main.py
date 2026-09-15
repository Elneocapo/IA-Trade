"""Punto de entrada de IA-Trade.

Evaluación walk-forward del modelo de machine learning.
No conecta con ningún broker ni envía órdenes reales.
"""

from config import INTERVAL, PERIOD, TICKER
from evaluation import buy_and_hold_curve, summarize_values
from market import download_market_data
from walk_forward import run_walk_forward


def main() -> None:
    print(f"Descargando {TICKER} ({PERIOD}, {INTERVAL})...")
    data = download_market_data(TICKER, PERIOD, INTERVAL)

    results, wf_stats = run_walk_forward(data)
    ia_stats = summarize_values(results["portfolio_value"])
    hold_stats = summarize_values(buy_and_hold_curve(data.loc[results.index[0] : results.index[-1]]))

    print("\n=== IA-Trade | Walk-Forward ML ===")
    print(f"Activo:                 {TICKER}")
    print(
        f"Periodo de prueba:      {wf_stats['test_start'].date()} -> "
        f"{wf_stats['test_end'].date()} ({wf_stats['test_days']} días)"
    )
    print(f"Ventanas walk-forward:  {wf_stats['windows']}")
    print(f"Capital inicial:        {ia_stats['initial_value']:.2f} €")
    print(f"Valor final IA:         {ia_stats['final_value']:.2f} €")
    print(f"Rentabilidad IA:        {ia_stats['return_pct']:.2f} %")
    print(f"Máximo drawdown IA:     {ia_stats['max_drawdown_pct']:.2f} %")
    print(f"Acierto dirección:      {wf_stats['accuracy_pct']:.2f} %")
    print(f"Cambios de posición:    {wf_stats['trades']}")
    print(f"Valor final Buy&Hold:   {hold_stats['final_value']:.2f} €")
    print(f"Rentabilidad Buy&Hold:  {hold_stats['return_pct']:.2f} %")
    print(f"Drawdown Buy&Hold:      {hold_stats['max_drawdown_pct']:.2f} %")

    advantage = ia_stats["final_value"] - hold_stats["final_value"]
    if advantage > 0:
        print("Comparación:             IA supera a Buy&Hold en este periodo.")
    elif advantage < 0:
        print("Comparación:             Buy&Hold supera a la IA en este periodo.")
    else:
        print("Comparación:             Empate en este periodo.")

    print("\nModo: SIMULACIÓN. No se ha enviado ninguna orden real.")


if __name__ == "__main__":
    main()
