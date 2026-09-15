"""Punto de entrada de IA-Trade.

Evalúa el modelo con walk-forward diario y abre una segunda fase de
investigación intradía. Todo sigue siendo simulación: no conecta con brokers.
"""

from config import (
    INTERVAL,
    INTRADAY_HORIZON_BARS,
    INTRADAY_INTERVAL,
    INTRADAY_PERIOD,
    INTRADAY_TEST_SIZE,
    INTRADAY_TRAIN_SIZE,
    PERIOD,
    TICKER,
)
from evaluation import buy_and_hold_curve, summarize_values
from market import download_market_data
from walk_forward import run_walk_forward


def print_report(label: str, data, results, wf_stats) -> None:
    ia_stats = summarize_values(results["portfolio_value"])
    hold_data = data.loc[results.index[0] : results.index[-1]]
    hold_stats = summarize_values(buy_and_hold_curve(hold_data))

    print(f"\n=== IA-Trade | {label} ===")
    print(f"Activo:                 {TICKER}")
    print(
        f"Periodo de prueba:      {wf_stats['test_start']} -> "
        f"{wf_stats['test_end']} ({wf_stats['test_days']} barras)"
    )
    print(f"Ventanas walk-forward:  {wf_stats['windows']}")
    print(f"Horizonte objetivo:     {wf_stats['horizon_bars']} barras")
    print(f"Entrada por P(subida):  >= {wf_stats['entry_threshold']:.2f}")
    if wf_stats["horizon_bars"] == 1:
        print(f"Salida por P(subida):   <  {wf_stats['exit_threshold']:.2f}")
    else:
        print("Salida:                 horizonte fijo")
    print("Objetivo ML:            SUBE / NEUTRO / BAJA")
    print(f"Capital inicial:        {ia_stats['initial_value']:.2f} €")
    print(f"Valor final IA:         {ia_stats['final_value']:.2f} €")
    print(f"Rentabilidad IA:        {ia_stats['return_pct']:.2f} %")
    print(f"Máximo drawdown IA:     {ia_stats['max_drawdown_pct']:.2f} %")
    print(f"Acierto objetivo:       {wf_stats['accuracy_pct']:.2f} %")
    print(f"Balanced accuracy:      {wf_stats['balanced_accuracy_pct']:.2f} %")
    print(f"Objetivos SUBE:         {wf_stats['target_up_rate_pct']:.2f} %")
    print(f"Objetivos NEUTRO:       {wf_stats['target_neutral_rate_pct']:.2f} %")
    print(f"Objetivos BAJA:         {wf_stats['target_down_rate_pct']:.2f} %")
    print(f"Brier multiclass:       {wf_stats['brier_score']:.4f}")
    print(f"Log-loss multiclass:    {wf_stats['multiclass_log_loss']:.4f}")
    print(f"P(subida) media:        {wf_stats['average_confidence_pct']:.2f} %")
    print(f"Días/barras invertido:  {wf_stats['days_in_market_pct']:.2f} %")
    print(f"Cambios de posición:    {wf_stats['trades']}")
    print(f"Valor final Buy&Hold:   {hold_stats['final_value']:.2f} €")
    print(f"Rentabilidad Buy&Hold:  {hold_stats['return_pct']:.2f} %")
    print(f"Drawdown Buy&Hold:      {hold_stats['max_drawdown_pct']:.2f} %")

    print("=== Señales que más usa el modelo ===")
    for feature, importance in wf_stats["top_features"].items():
        print(f"{feature:<28} {importance:.3f}")

    advantage = ia_stats["final_value"] - hold_stats["final_value"]
    if advantage > 0:
        print("Comparación:             IA supera a Buy&Hold en este periodo.")
    elif advantage < 0:
        print("Comparación:             Buy&Hold supera a la IA en este periodo.")
    else:
        print("Comparación:             Empate en este periodo.")


def main() -> None:
    print(f"Descargando {TICKER} ({PERIOD}, {INTERVAL})...")
    daily_data = download_market_data(TICKER, PERIOD, INTERVAL)
    daily_results, daily_stats = run_walk_forward(daily_data)
    print_report("Walk-Forward ML | Diario", daily_data, daily_results, daily_stats)

    print(f"\nDescargando {TICKER} ({INTRADAY_PERIOD}, {INTRADAY_INTERVAL})...")
    intraday_data = download_market_data(TICKER, INTRADAY_PERIOD, INTRADAY_INTERVAL)

    try:
        intraday_results, intraday_stats = run_walk_forward(
            intraday_data,
            train_size=INTRADAY_TRAIN_SIZE,
            test_size=INTRADAY_TEST_SIZE,
            horizon_bars=INTRADAY_HORIZON_BARS,
        )
        print_report(
            "Walk-Forward ML | Intradía 15m",
            intraday_data,
            intraday_results,
            intraday_stats,
        )
    except ValueError as error:
        print(f"\nNo se pudo evaluar la fase intradía: {error}")

    print("\nModo: SIMULACIÓN. No se ha enviado ninguna orden real.")
    print("Objetivo de esta fase: encontrar una ventaja estadística reproducible, no asumir rentabilidad.")


if __name__ == "__main__":
    main()
