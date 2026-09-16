"""V20 diagnóstico-only: does not change or optimize V16.7.
Run: python v20_diagnostico_v167.py
Uses the exact sequential prediction function from v16_nvidia_1h_sequential.py.
Test statistics are descriptive only and must never be used to select parameters.
"""
import numpy as np
import pandas as pd
import v16_nvidia_1h_sequential as v16

THRESHOLD_V167 = 0.00175  # selected in V16.7 validation; report-only here


def summarize(name, panel, prices):
    print(f"\n=== {name}: señal/frecuencia ===")
    if panel.empty:
        print("Sin predicciones")
        return
    s = panel.signal.astype(float)
    print(f"Barras evaluadas: {len(panel)}")
    print(f"Señales > 0: {(s > 0).sum()}")
    print(f"Señales > umbral V16.7 ({THRESHOLD_V167:.3%}): {(s > THRESHOLD_V167).sum()}")
    print("Cuantiles de señal:")
    print(s.quantile([0, .1, .25, .5, .75, .9, .95, .99, 1]).to_string(float_format=lambda x: f"{x:.5%}"))
    bins = [-np.inf, 0, .0005, .001, .00175, .0025, .005, np.inf]
    labels = ["<=0", "0–0.05%", "0.05–0.10%", "0.10–0.175%", "0.175–0.25%", "0.25–0.50%", ">0.50%"]
    groups = pd.cut(s, bins=bins, labels=labels)
    print("Conteo por intensidad (no es una medida de rentabilidad):")
    print(groups.value_counts(sort=False).to_string())
    fut = pd.DataFrame(index=panel.index)
    close = prices.Close.astype(float)
    for h in (1, 3, 6):
        fut[f"ret{h}"] = close.shift(-h).reindex(panel.index) / close.reindex(panel.index) - 1
    fut["grupo"] = groups
    print("Retorno futuro bruto por grupo (diagnóstico, no señal):")
    print(fut.groupby("grupo", observed=False).agg({f"ret{h}":["count","mean","median"] for h in (1,3,6)}).to_string(float_format=lambda x: f"{x:.4%}"))


def main():
    df = v16.load_data()
    n = len(df)
    cut_train, cut_val = int(n * .60), int(n * .80)
    print("V20: generando predicciones con función V16.7; no se cambian parámetros.", flush=True)
    panel = v16.sequential_predictions(df, cut_train, n - 1)
    if panel.empty:
        raise RuntimeError("No se generaron predicciones.")
    val = panel[(panel.index >= df.index[cut_train]) & (panel.index < df.index[cut_val])]
    test = panel[panel.index >= df.index[min(cut_val + 2, n - 1)]]
    print(f"Validación: {val.index.min()} -> {val.index.max()}")
    print(f"Test (solo descriptivo): {test.index.min()} -> {test.index.max()}")
    summarize("VALIDACIÓN", val, df)
    summarize("TEST FINAL — NO USAR PARA AJUSTAR", test, df)
    print("\nAUDITORÍA pendiente: V16.7 calcula señal usando información de la barra y simula ejecución al mismo close.")
    print("Si el close completo aún no era conocido al ejecutar, ese supuesto puede ser optimista.")
    print("V20 no corrige ese supuesto ni calcula un nuevo profit; solo diagnostica señales.")

if __name__ == "__main__":
    main()
