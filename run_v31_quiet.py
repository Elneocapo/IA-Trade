"""Launcher for V31 that suppresses sklearn/joblib warning spam.

It does not modify V31's model, parameters, data or results.
"""
from __future__ import annotations
import os
import runpy
import warnings

# Also inherited by joblib/loky worker processes.
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

runpy.run_path("v31_robust_options_ai.py", run_name="__main__")
