# IA-Trade

Proyecto de investigación para construir un sistema de trading algorítmico con datos reales, backtesting, gestión de riesgo y modelos de machine learning.

> **Importante:** esta versión funciona únicamente en simulación/backtesting. No conecta con brokers, no coloca órdenes reales y no promete beneficios.

## Estado actual

- [x] Estructura inicial
- [x] Descarga de datos históricos con `yfinance`
- [x] Estrategia base de medias móviles
- [x] Cartera virtual con capital inicial de 50 €
- [x] Comisiones simuladas
- [x] Primer modelo de machine learning (Random Forest)
- [x] Separación temporal entre entrenamiento y prueba
- [x] Evaluación walk-forward
- [x] Comparación contra buy & hold
- [ ] Tests automáticos
- [ ] Gestión de riesgo avanzada
- [ ] Paper trading en tiempo real
- [ ] Dashboard

## Instalación en Windows

```cmd
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
```

## Ejecutar la evaluación

```cmd
python main.py
```

Por defecto usa AAPL, 2 años de datos diarios y una cartera virtual de 50 €.

## Cómo funciona ahora

El modelo utiliza información disponible en cada día (rendimientos, medias móviles, volatilidad, volumen y RSI) para intentar predecir si el precio de cierre del día siguiente será superior al actual.

En lugar de entrenar una sola vez y probar una sola vez, la validación **walk-forward** divide el historial en varios bloques. Para cada bloque, el modelo aprende únicamente con datos anteriores y después se evalúa sobre datos posteriores que no había visto.

La simulación también retrasa la posición un día respecto a la predicción para evitar operar usando información futura. Se compara el resultado de la IA con una estrategia sencilla de **buy & hold** sobre el mismo periodo.

Un resultado positivo en backtesting no demuestra que el modelo vaya a ser rentable en el futuro. Con solo dos años de un activo todavía necesitamos más historial, varios activos, tests y controles de riesgo antes de sacar conclusiones.

## Filosofía del proyecto

No vamos a meter dinero real porque un modelo de IA diga que una operación parece buena. Primero se valida con datos históricos, después con paper trading y solo mucho más adelante se estudiaría una integración con un broker bajo una cuenta de un adulto y con controles de riesgo.
