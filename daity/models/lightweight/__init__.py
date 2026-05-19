"""Lightweight FT-Transformer + Set Transformer cohort model.

Designed as a smaller (~300K param) alternative to the cohort transformer.
Per-stock FT-Transformer processes engineered tabular features; cross-stock
Set Transformer enables relative-value / portfolio-level interactions.

Designed to support:
  - Multi-task supervised pretraining (regression + classification heads)
  - STE top-K portfolio head fine-tuning with Sharpe loss (Flavor B)
"""
