# Ruta NVDA 1H -> opciones

## Estado actual

**V27** sigue siendo el cerebro de direccion de NVDA 1H.
La señal se genera walk-forward y la ejecucion de acciones usa next-open.

**V28** trabaja el control de riesgo sobre esa señal.

**V29** introduce el vehiculo de opciones en modo paper/sintetico.
No envia ordenes a ningun broker.

## Fases

### V29 — Motor de opciones

Convierte una señal alcista de V27 en una compra de CALL y modela:

- prima con Black-Scholes en modo sintetico;
- strike objetivo por delta;
- vencimiento fijo desde la entrada;
- IV proxy;
- spread bid/ask;
- coste de operacion;
- multiplicador de 100;
- presupuesto maximo de prima;
- salida por senal o MAX_HOLD;
- comparacion contra la misma IA usando acciones.

El modo sintetico permite estudiar sensibilidad, pero no demuestra que las
opciones hubieran producido ese resultado en mercado real.

### V30 — Seleccion robusta

Solo despues de V29A:

1. seleccionar configuracion exclusivamente en VALIDACION;
2. repetir la seleccion en varios subperiodos de validacion;
3. medir estabilidad frente a DTE, delta, IV y spread;
4. mantener un TEST final completamente ciego;
5. exigir que la ventaja no dependa de una sola configuracion.

### V31 — Datos historicos reales de opciones

Sustituir gradualmente el motor sintetico por cadenas historicas reales con:

- bid/ask reales;
- strike y vencimiento reales;
- IV real cuando exista;
- liquidez / volumen / open interest;
- slippage realista;
- reglas de ejecucion por siguiente cotizacion disponible.

V29 ya incluye un adaptador CSV para este paso.

### V32 — Direccion alcista y bajista

V27 actualmente se usa como predictor alcista. Antes de mapear una señal
negativa a PUTs, entrenar y validar explicitamente un modelo de downside.
No asumir que `signal < 0` tiene la misma calidad predictiva que `signal > 0`.

### V33 — Paper trading en tiempo casi real

El sistema leeria datos nuevos, ejecutaria la misma logica sobre una cuenta
virtual y guardaria cada decision:

`timestamp -> señal IA -> contrato elegido -> precio teorico/quote -> entrada -> salida -> P&L`

Sin dinero real.

### Fase final — Integracion de broker

Solo tras pasar las fases anteriores se estudiaria una integracion real.
La capa de broker debe quedar separada del modelo para que el modelo nunca
pueda enviar una orden por accidente durante investigacion.

La integracion real depende ademas de las reglas del broker, permisos para
opciones, disponibilidad de producto, capital y requisitos legales aplicables.

## Regla del proyecto

No usar el TEST para elegir parametros. Si una modificacion mejora el TEST
pero empeora la validacion o la robustez entre subperiodos, se descarta.
