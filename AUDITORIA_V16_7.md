# Auditoría técnica de V16.7 (NVDA, 1H)

## Estado
Revisión estática del código; no es un backtest nuevo y no modifica V16.7. No usar como señal de inversión.

## Hallazgos

### 1. Entrada al mismo cierre
`features()` usa OHLCV de la barra `i`, incluida su `Close`, `High`, `Low` y `Volume`. `sequential_predictions()` genera una señal para esa barra. `backtest()` valora y compra/vende usando `panel.price`, que es el cierre de esa misma barra.

Aunque la etiqueta predice el movimiento de `Close[i]` a `Close[i+1]`, para operar al cierre `i` habría que conocer el cierre definitivo y ejecutar a ese precio simultáneamente. En la práctica puede ser optimista, especialmente si la señal solo se conoce tras cerrar la vela. Una comparación más conservadora debería ejecutar en `Open[i+1]` (y contabilizar el gap), o definir una ejecución intrabar con datos de menor intervalo.

### 2. Registro de retorno de operación no incluye bien el precio neto de salida
En `backtest()`, `total=cash+shares*price` se calcula antes de vender y se añade `total/entry_value-1` a `trs` después de vender. Por tanto, el retorno registrado para win rate se basa en el valor previo a la venta, no en el efectivo neto recibido tras aplicar `COST`. Esto puede sesgar el win rate, aunque la curva de capital sí incorpora el coste de venta.

### 3. No modela opciones
La estrategia compra/vende acciones de NVDA como proxy. No calcula calls/puts, prima, volatilidad implícita, strike, vencimiento, spread ni pérdida por theta. Por ello, sus retornos no se pueden interpretar como retornos de opciones.

### 4. Datos y barras horarias
La descarga usa Yahoo Finance 1h y filtra timestamps entre 09:30 y 16:00 ET. Antes de usar señales en vivo hay que verificar exactamente qué intervalo representa cada timestamp y cuándo queda disponible el OHLCV de esa barra. El archivo por sí solo no demuestra que el timestamp equivalga al momento de cierre ejecutable.

## Próximo experimento recomendado
Crear una versión separada (sin tocar V16.7) que:
1. Mantenga idénticas las predicciones, umbral y reglas de señal.
2. Ejecute las decisiones en la siguiente apertura disponible, no en el cierre que generó la señal.
3. Aplique costes tanto en compras como ventas y registre el retorno de operación desde el efectivo realmente invertido hasta el efectivo neto de salida.
4. Compare contra buy-and-hold con el mismo intervalo de entrada/salida.
5. Reporte cantidad de operaciones, retorno, drawdown, win rate y diferencia frente al benchmark.
6. Mantenga el test final descriptivo, sin seleccionar parámetros con él.

## Conclusión
El resultado histórico de V16.7 (+6,23% reportado frente a +5,90% buy-and-hold) es una pista, no evidencia robusta de ventaja: el diferencial reportado es pequeño, hay pocas operaciones y queda pendiente validar ejecución y contabilidad de trades.