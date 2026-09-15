# v11 research notes

Goal: optimize robust out-of-sample paper-trading PnL, not classification accuracy.

Changes vs earlier versions:
- no 09:57 rule
- continuous multi-horizon return prediction (1/3/5/10 days)
- HistGradientBoosting with Huber loss instead of MLP classifier/regressor
- expanding-window walk-forward predictions
- threshold tuned only on the first half of walk-forward predictions
- final evaluation on the later half
- next-bar execution assumption and transaction costs
- long-only paper simulation

This is research only; no live broker orders.
