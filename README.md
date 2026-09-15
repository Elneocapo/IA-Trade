# IA-Trade

Proyecto de investigación para construir un sistema de trading algorítmico con datos reales, backtesting, gestión de riesgo y modelos de machine learning.

> **Importante:** esta versión funciona únicamente en simulación/backtesting. No conecta con brokers, no coloca órdenes reales y no promete beneficios.

## Estado actual

- [x] Estructura inicial
- [x] Descarga de datos históricos con `yfinance`
- [x] Estrategia base de medias móviles
- [x] Cartera virtual con capital inicial de 50 €
- [x] Comisiones simuladas
- [x] Backtest y cálculo de rentabilidad/drawdown
- [ ] Tests automáticos
- [ ] Gestión de riesgo avanzada
- [ ] Comparación contra buy & hold
- [ ] Features para machine learning
- [ ] Modelo ML y validación walk-forward
- [ ] Paper trading en tiempo real
- [ ] Dashboard

## Instalación en Windows

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecutar el backtest

```cmd
python main.py
```

Por defecto usa AAPL, 2 años de datos diarios y una cartera virtual de 50 €.

## Filosofía del proyecto

No vamos a meter dinero real porque un modelo "de IA" diga que una operación parece buena. Primero se valida con datos históricos, después con paper trading y solo mucho más adelante se estudiaría una integración con un broker bajo una cuenta de un adulto y con controles de riesgo.
