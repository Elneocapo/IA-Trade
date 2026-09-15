"""Configuración del proyecto IA-Trade.

El proyecto arranca en modo simulación: nunca envía órdenes reales.
"""

TICKER = "AAPL"
PERIOD = "2y"
INTERVAL = "1d"
INITIAL_CASH = 50.0
COMMISSION = 0.001  # 0.1% simulada por operación

# Seguridad: esta primera versión no tiene conexión con brokers.
PAPER_TRADING_ONLY = True
