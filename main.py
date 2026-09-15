"""Punto de entrada de IA-Trade.

Evalúa el modelo con entrenamiento explícito, validación, test y walk-forward.
Todo sigue siendo simulación: no conecta con brokers.
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
    VALIDATION_TICKERS,
)
from baseline_experiment import run_baseline_experiment
from evaluation import buy_and_hold_curve, summarize_values
from market import download_market_data
from walk_forward import (
    run_directional_walk_forward,
    run_expanding_temporal_experiment,
    run_training_horizon_experiment,
    run_walk_forward,
)


def print_horizon_experiment(ticker: str, stats_list: list[dict]) -> None:
    print("=== EXPERIMENTO DE ENTRENAMIENTO | HORIZONTES ===")
    print(f"Activo: {ticker}")
    print("División: 60% entrenamiento / 20% validación / 20% test")
    print("Importante: el test NO participa en el entrenamiento.")
    print()
    print("Horizonte | Train acc | Val acc | Test acc | Test balanced | Test Brier | Test log-loss")
    for stats in stats_list:
        train = stats["train"]
        validation = stats["validation"]
        test = stats["test"]
        print(
            f"{stats['horizon']:>9} | "
            f"{train['accuracy_pct']:>9.2f}% | "
            f"{validation['accuracy_pct']:>7.2f}% | "
            f"{test['accuracy_pct']:>8.2f}% | "
            f"{test['balanced_accuracy_pct']:>13.2f}% | "
            f"{test['brier_score']:>10.4f} | "
            f"{test['log_loss']:>12.4f}"
        )
    print()
    print("Lectura: buscamos que validación y test conserven señal, no simplemente que Train sea alto.")


def print_expanding_temporal_experiment(ticker: str, stats_list: list[dict]) -> None:
    print("=== EXPERIMENTO TEMPORAL EXPANSIVO | APRENDER Y AVANZAR ===")
    print(f"Activo: {ticker}")
    print("Entrenamiento inicial: 50% del historial")
    print("Validación: bloques consecutivos del 10%, ampliando el entrenamiento después de cada bloque")
    print("Test final: último 10%, reservado y no usado para decidir el modelo")
    print()
    print("Horizonte | Folds | Val acc media | Val balanced media | Final train | Final test | Test balanced")
    for stats in stats_list:
        print(
            f"{stats['horizon']:>9} | "
            f"{len(stats['folds']):>5} | "
            f"{stats['validation_mean_accuracy_pct']:>12.2f}% | "
            f"{stats['validation_mean_balanced_accuracy_pct']:>18.2f}% | "
            f"{stats['final_train']['accuracy_pct']:>11.2f}% | "
            f"{stats['final_test']['accuracy_pct']:>10.2f}% | "
            f"{stats['final_test']['balanced_accuracy_pct']:>13.2f}%"
        )
        for fold in stats["folds"]:
            validation = fold["validation"]
            print(
                f"    Fold {fold['fold']}: "
                f"train {fold['train_start'].date()} -> {fold['train_end'].date()} | "
                f"val {fold['validation_start'].date()} -> {fold['validation_end'].date()} | "
                f"val acc {validation['accuracy_pct']:.2f}% | "
                f"val balanced {validation['balanced_accuracy_pct']:.2f}%"
            )
        print(
            f"    TEST FINAL: {stats['final_test_start'].date()} -> {stats['final_test_end'].date()} | "
            f"acc {stats['final_test']['accuracy_pct']:.2f}% | "
            f"balanced {stats['final_test']['balanced_accuracy_pct']:.2f}% | "
            f"Brier {stats['final_test']['brier_score']:.4f} | "
            f"log-loss {stats['final_test']['log_loss']:.4f}"
        )
        print()
    print("Lectura: queremos ver rendimiento de validación relativamente estable entre épocas, y que el test final no se derrumbe.")


def print_baseline_experiment(ticker: str, stats: dict) -> None:
    print("=== EXPERIMENTO DE BASELINES | ¿LA IA APORTA ALGO? ===")
    print(f"Activo:                 {ticker}")
    print(f"Horizonte:              {stats['horizon']} barra")
    print("Clases:                 SUBE vs BAJA (NEUTRO excluido)")
    print("Mismo periodo y mismas muestras para todos los métodos.")
    print("Validación:             4 bloques temporales consecutivos")
    print("Test final:             último bloque, reservado")
    print()
    print("Método      | Val acc media | Val balanced | Test acc | Test balanced")
    for method, label in (
        ("random", "Azar"),
        ("majority", "Mayoritaria"),
        ("momentum", "Momentum"),
        ("model", "Random Forest"),
    ):
        validation = stats["validation_means"][method]
        final = stats["final_test"][method]
        print(
            f"{label:<12} | "
            f"{validation['accuracy_pct']:>12.2f}% | "
            f"{validation['balanced_accuracy_pct']:>12.2f}% | "
            f"{final['accuracy_pct']:>8.2f}% | "
            f"{final['balanced_accuracy_pct']:>13.2f}%"
        )

    print()
    print("Detalle de validación por época:")
    print("Fold | Azar bal. | Mayoritaria bal. | Momentum bal. | Random Forest bal.")
    for fold in stats["validation_folds"]:
        print(
            f"{fold['fold']:>4} | "
            f"{fold['random']['balanced_accuracy_pct']:>9.2f}% | "
            f"{fold['majority']['balanced_accuracy_pct']:>16.2f}% | "
            f"{fold['momentum']['balanced_accuracy_pct']:>13.2f}% | "
            f"{fold['model']['balanced_accuracy_pct']:>18.2f}%"
        )

    print()
    print(
        f"Test final: {stats['final_test_start'].date()} -> {stats['final_test_end'].date()} | "
        f"{stats['final_test']['samples']} muestras"
    )
    print(
        "Lectura: si Random Forest no supera de forma consistente a momentum y a la "
        "mayoritaria, no tiene sentido complicar la IA todavía."
    )


def print_report(label: str, ticker: str, data, results, wf_stats) -> None:
    ia_stats = summarize_values(results["portfolio_value"])
    hold_data = data.loc[results.index[0] : results.index[-1]]
    hold_stats = summarize_values(buy_and_hold_curve(hold_data))

    print(f"\n=== IA-Trade | {label} ===")
    print(f"Activo:                 {ticker}")
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

    print("=== ¿La confianza contiene señal? ===")
    print("Bin       Muestras  P(subida)  Acierto al alza  Retorno medio  Retorno mediano")
    for row in wf_stats["confidence_analysis"]:
        print(
            f"{row['bin']:<8} {row['samples']:>8}  "
            f"{row['mean_probability_pct']:>9.2f}%  "
            f"{row['up_rate_pct']:>14.2f}%  "
            f"{row['mean_future_return_pct']:>13.3f}%  "
            f"{row['median_future_return_pct']:>15.3f}%"
        )

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


def print_directional_diagnostic(ticker: str, stats: dict) -> None:
    print("=== DIAGNÓSTICO DIRECCIONAL | SOLO MOVIMIENTOS ACCIONABLES ===")
    print(f"Activo:                 {ticker}")
    print(f"Muestras evaluadas:     {stats['samples']}")
    print(f"Ventanas walk-forward:  {stats['windows']}")
    print("Clases:                 SUBE vs BAJA (NEUTRO excluido)")
    print(f"Acierto binario:        {stats['accuracy_pct']:.2f} %")
    print(f"Balanced accuracy:      {stats['balanced_accuracy_pct']:.2f} %")
    print(f"Baseline mayoritaria:   {stats['baseline_accuracy_pct']:.2f} %")
    print(f"Objetivos SUBE:         {stats['target_up_rate_pct']:.2f} %")
    print(f"Brier binario:          {stats['brier_score']:.4f}")
    print(f"Log-loss binario:       {stats['log_loss']:.4f}")
    print(f"P(subida) media:        {stats['mean_probability_up_pct']:.2f} %")


def main() -> None:
    print("=== IA-TRADE | LABORATORIO DE APRENDIZAJE Y VALIDACIÓN ===")

    print(f"\nDescargando {TICKER} ({PERIOD}, {INTERVAL}) para el experimento de aprendizaje...")
    horizon_data = download_market_data(TICKER, PERIOD, INTERVAL)
    try:
        horizon_stats = run_training_horizon_experiment(horizon_data)
        print_horizon_experiment(TICKER, horizon_stats)
    except ValueError as error:
        print(f"\nNo se pudo ejecutar el experimento de horizontes: {error}")

    try:
        temporal_stats = run_expanding_temporal_experiment(horizon_data)
        print_expanding_temporal_experiment(TICKER, temporal_stats)
    except ValueError as error:
        print(f"\nNo se pudo ejecutar el experimento temporal expansivo: {error}")

    # Antes de cambiar de modelo o añadir complejidad, comprobamos si el
    # Random Forest aporta información frente a reglas extremadamente simples.
    try:
        baseline_stats = run_baseline_experiment(horizon_data, horizon_bars=1)
        print_baseline_experiment(TICKER, baseline_stats)
    except ValueError as error:
        print(f"\nNo se pudo ejecutar el experimento de baselines: {error}")

    print("\n=== VALIDACIÓN MULTIACTIVO | MISMA IA, SIN AJUSTAR UMBRALES ===")
    for ticker in VALIDATION_TICKERS:
        print(f"\nDescargando {ticker} ({PERIOD}, {INTERVAL})...")
        daily_data = download_market_data(ticker, PERIOD, INTERVAL)
        daily_results, daily_stats = run_walk_forward(daily_data)
        print_report(
            "Walk-Forward ML | Diario",
            ticker,
            daily_data,
            daily_results,
            daily_stats,
        )

        try:
            directional_stats = run_directional_walk_forward(daily_data)
            print_directional_diagnostic(ticker, directional_stats)
        except ValueError as error:
            print(f"\nNo se pudo evaluar el diagnóstico direccional: {error}")

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
            TICKER,
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
