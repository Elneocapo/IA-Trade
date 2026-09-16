"""Launcher for V31 that suppresses warning spam and applies the V31 options-hold experiment.

The directional V31 model/backtest remains unchanged. Only the synthetic CALL
visualization layer is patched at runtime so CALLs do not exit on a small,
temporary signal dip.
"""
from __future__ import annotations
import os
import warnings

# Also inherited by joblib/loky worker processes.
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

# Controlled options-management experiment:
# - minimum 8 bars before a signal-based CALL exit;
# - signal must actually deteriorate to <= 0.0 to trigger that exit;
# - hard maximum hold is 36 bars (~5.5 trading days on 1h regular-session data);
# - option budget stays at 20%, so we change holding behavior, not exposure.
source = open("v31_robust_options_ai.py", "r", encoding="utf-8").read()
source = source.replace(
    'OPTION_DTE=14; OPTION_DELTA=0.60; OPTION_IV_MULT=1.00; OPTION_SPREAD=0.01; OPTION_BUDGET=0.20',
    'OPTION_DTE=14; OPTION_DELTA=0.60; OPTION_IV_MULT=1.00; OPTION_SPREAD=0.01; OPTION_BUDGET=0.20\nOPTION_MIN_HOLD=8; OPTION_MAX_HOLD=36; OPTION_EXIT_SIGNAL=0.0'
)
source = source.replace(
    'forced=ex-entry_i>=MAX_HOLD or T<=0',
    'forced=ex-entry_i>=OPTION_MAX_HOLD or T<=0'
)
source = source.replace(
    'if (not bullish) or forced:',
    'if ((not bullish) and ex-entry_i>=OPTION_MIN_HOLD and float(row["signal"])<=OPTION_EXIT_SIGNAL) or forced:'
)

exec(compile(source, "v31_robust_options_ai.py", "exec"), globals(), globals())
