"""Projet 4 — Retail ML Forecasting (MIT 2026).

Pipeline : ventes (Kafka/S3) -> features -> LightGBM (SageMaker) -> agent LLM
-> dashboard. Le package est volontairement découpé par étape du pipeline :
`ingest`, `features`, `train`, `evaluate`, `inference`, `agent`.
"""

__version__ = "0.1.0"
