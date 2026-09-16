# IA-Trade — bitácora de investigación

> **Objetivo:** investigar si existe una ventaja económica reproducible en NVDA, no maximizar accuracy ni fabricar un backtest bonito. Solo investigación/backtesting y paper trading; no órdenes reales.
>
> **Regla de trabajo:** ENTRENAR → PROBAR → ANALIZAR → MEJORAR → VOLVER A PROBAR. Prioridad: ROBUSTEZ > RESULTADO BONITO; VENTAJA ECONÓMICA > ACCURACY.

## Arquitectura acordada

- **Un modelo por activo.** El modelo actual es exclusivamente NVDA. No mezclar NVDA con AAPL/MSFT ni convertirlo en un modelo multi-activo.
- Marco actual: velas de 1 hora de Yahoo Finance (`yfinance`), mercado regular de EE. UU.
- El test final es ciego: se usa para medir, nunca para elegir umbrales, filtros o parámetros.
- Comparación principal: comprar y mantener NVDA durante exactamente el mismo periodo.
- Una versión con más operaciones no es automáticamente mejor. Puede ser una señal de sobreajuste.

## Resultados que hay que conservar como referencia

### V16.6 / V16.7 — benchmark de referencia

Resultado comunicado por la ejecución local del usuario:

- Periodo de datos: 2024-09-16 a 2026-09-15; 3.487 barras horarias.
- Validación: 2025-11-25 a 2026-04-23.
- Test final: 2026-04-23 a 2026-09-15, unos 145 días.
- Test: IA +6,23%; NVDA buy-and-hold +5,90%; 17 operaciones; drawdown máximo −2,16%; Sharpe 1,41.
- Capital simulado: 50 € → 53,11 € para IA; 52,95 € para buy-and-hold.
- Margen sobre buy-and-hold: solo +0,33 puntos porcentuales. Muestra pequeña; **no prueba una ventaja robusta**.
- V16.7 permitió holds de 4/6/8/10/13 barras y un abanico más amplio de umbrales. Seleccionó 6 barras y threshold 0,18% en la ejecución registrada.

**No modificar el benchmark congelado** `v16_nvidia_1h_sequential.py` ni su copia `v16_6_backup.py` salvo petición expresa. La V16.7 es el punto de comparación, no una prueba definitiva.

### V17 / V18 — experimento de frecuencia descartado

Se probaron entradas long+short, umbrales menores y retenciones cortas para forzar actividad. En el test final resultó −0,47%, 0 operaciones y NVDA buy-and-hold +5,90%. **Descartado**: aumentar frecuencia alterando varias cosas no conservó la señal. No usarlo como benchmark ni restaurar esos cambios sobre V16.7.

### V19 — refinamiento de entrada

- Se comparó entrada fija con umbral frente a percentil móvil de señales pasadas.
- Seleccionó `fixed`, threshold 0,175%, 100% de exposición y hold 6 barras.
- Repitió exactamente los resultados reportados de V16.7: test +6,23%, B&H +5,90%, 17 operaciones, DD −2,16%, Sharpe 1,41; ejecución 6,8 min.
- Conclusión: la regla adaptativa no añadió frecuencia ni mejora. 🟡 Señal de que simplemente retocar el umbral/forma de entrada no resuelve el cuello de botella.

## Qué significan los parámetros de V16.7

Estos números son decisiones de diseño del experimento, no constantes universales ni “valores óptimos” demostrados:

- `ASSET="NVDA"`: decisión explícita de entrenar un modelo independiente por empresa.
- `PERIOD="2y"`: ventana histórica solicitada a Yahoo; limita la muestra disponible y no equivale a décadas de regímenes.
- `INTERVAL="1h"`: marco horario de investigación actual; no permite reconstruir con precisión el estado de una vela a las 09:57 ET.
- `LOOKBACK_BARS=65`: contexto de 65 velas (~10 sesiones regulares de 6,5 horas). Incluye escalas de indicadores hasta 65 barras; es una hipótesis de contexto, no una duración garantizada de memoria.
- `TRAIN_WINDOW=1200`: máximo aproximado de 1.200 barras recientes para cada ajuste; busca adaptarse a cambios de régimen sin usar toda la historia. No significa 1.200 días.
- `RETRAIN_EVERY=24`: reajuste cada 24 barras horarias (~3,7 sesiones regulares), compromiso computacional/adaptativo elegido para el walk-forward.
- `MODEL_MAX_ITER=140`: límite de iteraciones de HistGradientBoosting; hiperparámetro de complejidad/tiempo, no número de épocas de una red neuronal.
- `COST=0.001`: coste simulado del 0,1% por lado de compra/venta. Es una aproximación conservadora simplificada; no modela spreads, slippage ni costes de opciones.
- `INITIAL_CASH=50`: capital ficticio para hacer los resultados comprensibles; no afecta al porcentaje si el modelo es fraccional y no hay mínimos/comisiones fijas.
- `HOLD_CANDIDATES=(4,6,8,10,13)`: horizontes máximos en barras horarias; permite explorar salidas cortas a unas dos sesiones. Se selecciona solo con validación.
- `THRESHOLD_CANDIDATES`: rejilla de cortes sobre retorno/señal predicha. Cada valor es candidato, no una probabilidad. Elegir entre muchos valores aumenta riesgo de sobreajuste a validación.
- Pesos `(0.25,...,1.0)`: fracción objetivo del capital expuesta; 1.0 equivale a 100% de la cartera simulada, no apalancamiento.
- `trend_filter` y `vol_filter`: interruptores booleanos para evaluar filtros simples; no deben interpretarse como verdades de mercado.
- `0.0015` en el clasificador: etiqueta “sube” si el retorno de la próxima vela supera 0,15%; es una definición de objetivo, no un límite de rentabilidad garantizada.
- `0.62/0.25/0.13`: pesos que combinan estimaciones de retorno a 1, 3 y 6 barras. Son una receta fija heredada; necesitan análisis de sensibilidad, no asumir que son óptimos.
- `0.72` y `0.85`: modificadores de intensidad por confianza del clasificador. No convierten la confianza en una probabilidad calibrada.
- `0.32` y `0.018` en la puntuación de validación: penalización de drawdown y premio por Sharpe usados para ordenar candidatos. La escala de estos términos influye en qué candidato gana; no son métricas económicas naturales.
- Penalizaciones por menos de 10 operaciones y win rate bajo: reglas de selección para evitar estrategias demasiado escasas o de aciertos muy bajos. Pueden sesgar la elección; no son prueba estadística.
- `60%/80%` de cortes temporales: reserva tramo inicial para entrenar, siguiente tramo para validar y último tramo para test ciego. La separación temporal evita mezclar futuro con pasado, aunque el tamaño de test sigue siendo limitado.

## Hipótesis 09:57 ET — separada del benchmark horario

Existe una idea recordada pero **no verificada**: observar a las 09:57 ET una vela horaria aún en formación y tomar sesgo bajista si está roja; se mencionó una expiración mínima de dos días. No tratarla como hecho ni incorporarla a la estrategia horaria actual sin datos intrahorarios. Para evaluarla hacen falta velas de 1m/5m con timestamps y reglas de ejecución realistas. Una vela 1H cerrada no permite reconstruir exactamente qué se veía a las 09:57.

## Próximos pasos acordados

1. Mantener V16.7 como referencia inmutable y no volver a forzar operaciones por sí mismas.
2. Antes de cambiar el modelo, diagnosticar en validación (sin mirar para ajustar el test): distribución de señales, entradas omitidas, duración real de posiciones, retornos por señal y sensibilidad de resultados a costes.
3. Cambiar **una sola dimensión por versión**, registrar hipótesis antes de ejecutar y reportar validación y test por separado.
4. Si el test final se consulta para evaluar una versión, no volver a seleccionar parámetros con él. Para nuevos ajustes, reservar una futura ventana realmente no vista o usar un protocolo walk-forward predefinido.
5. “No se encuentra una ventaja robusta” es un resultado válido.

## Cómo continuar en otro chat

Pega este archivo o indica que leas `BITACORA_INVESTIGACION.md`. Estado: NVDA-only, V16.7 benchmark; V17/V18 frecuencia descartadas; V19 no mejoró frente a V16.7. Priorizar diagnóstico y documentación antes de otra modificación de estrategia.
