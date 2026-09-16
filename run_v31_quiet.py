"""V31 launcher for Windows.

Runs the real V31 on main. Warnings are suppressed before sklearn/joblib workers
are created. The strategy remains paper research only.
"""
from __future__ import annotations
import os
os.environ["PYTHONWARNINGS"]="ignore:.*sklearn.utils.parallel.delayed.*:UserWarning"
import warnings
warnings.filterwarnings("ignore", message=r".*sklearn\.utils\.parallel\.delayed.*", category=UserWarning)
warnings.filterwarnings("ignore")

print("[V31] Ejecutando run_v31_quiet.py | rama main | warnings sklearn filtrados", flush=True)
from v31_robust_options_ai import main

if __name__ == "__main__":
    main()
