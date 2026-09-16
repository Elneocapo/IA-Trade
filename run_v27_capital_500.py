"""Run V27 with 500 EUR starting capital.

Historical V27 remains unchanged for reproducibility. This wrapper overrides
its simulation capital at runtime, so all reported absolute EUR values start
from 500 EUR while percentages/trade logic remain unchanged.
"""
import v27_session_aware_v23_plus as v27

v27.INITIAL_CASH = 500.0

if __name__ == "__main__":
    v27.main()
