"""Configuración del proyecto IA-Trade.

El proyecto arranca en modo simulación: nunca envía órdenes reales.
"""

TICKER = "AAPL"
# Más historial para que el walk-forward diario tenga muchas ventanas de prueba.
PERIOD = "5y"
INTERVAL = "1d"

# Segunda vía de investigación: intradía para buscar oportunidades de mayor
# frecuencia. Yahoo Finance limita el histórico disponible de velas intradía,
# por eso esta fase usa 15 minutos y una ventana reciente.
INTRADAY_PERIOD = "60d"
INTRADAY_INTERVAL = "15m"
INTRADAY_TRAIN_SIZE = 500
INTRADAY_TEST_SIZE = 100
# 8 velas de 15m = 2 horas. Probamos un horizonte más largo porque el horizonte
# de 1 hora no convirtió la señal predictiva en una estrategia rentable.
INTRADAY_HORIZON_BARS = 8

INITIAL_CASH = 50.0
COMMISSION = 0.001  # 0.1% simulada por operación

# Seguridad: esta primera versión no tiene conexión con brokers.
PAPER_TRADING_ONLY = True
