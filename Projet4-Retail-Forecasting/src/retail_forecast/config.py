"""Configuration centrale du projet.

Un seul endroit pour tout ce qui change entre la démo locale et l'AWS Learner
Lab (bucket, backend d'ingestion, provider LLM, seuils). Le reste du code ne
fait jamais de ``if os.environ[...]`` éparpillé.

Usage::

    from retail_forecast.config import SETTINGS, RAW_DIR

    print(SETTINGS.s3_uri("raw", "sales"))
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CURATED_DIR = DATA_DIR / "curated"
FORECAST_DIR = DATA_DIR / "forecast"
DOCS_DIR = ROOT / "docs"


def _load_dotenv() -> None:
    """Charge ``.env`` si python-dotenv est installé (optionnel en CI).

    Les variables AWS laissées vides dans ``.env`` sont supprimées : sinon boto3
    reçoit ``AWS_PROFILE=""`` et lève un ``ProfileNotFound``.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dépendance optionnelle
        return
    load_dotenv(ROOT / ".env")
    for key in ("AWS_PROFILE", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        if os.environ.get(key) == "":
            os.environ.pop(key)


@dataclass(frozen=True)
class Settings:
    """Paramètres figés, construits depuis l'environnement."""

    region: str
    role_arn: str
    bucket: str
    profile: str
    s3_endpoint_url: str
    s3_access_key: str
    s3_secret_key: str
    ingest_backend: str
    kafka_bootstrap: str
    kafka_topic: str
    forecast_horizon: int
    model_name: str
    wape_threshold: float
    llm_provider: str
    llm_endpoint: str
    ollama_model: str
    ollama_base_url: str
    bedrock_model_id: str

    @classmethod
    def load(cls, env: dict[str, str] | None = None) -> "Settings":
        _load_dotenv()
        e = os.environ if env is None else env
        return cls(
            region=e.get("AWS_REGION", "us-east-1"),
            role_arn=e.get("SAGEMAKER_ROLE_ARN", ""),
            bucket=e.get("S3_BUCKET", ""),
            profile=e.get("AWS_PROFILE", ""),
            s3_endpoint_url=e.get("S3_ENDPOINT_URL", ""),
            s3_access_key=e.get("AWS_ACCESS_KEY_ID", ""),
            s3_secret_key=e.get("AWS_SECRET_ACCESS_KEY", ""),
            ingest_backend=e.get("INGEST_BACKEND", "kafka").lower(),
            kafka_bootstrap=e.get("KAFKA_BOOTSTRAP", "localhost:19092"),
            kafka_topic=e.get("KAFKA_TOPIC", "sales.events"),
            forecast_horizon=int(e.get("FORECAST_HORIZON", "28")),
            model_name=e.get("MODEL_NAME", "lgbm-global"),
            wape_threshold=float(e.get("WAPE_THRESHOLD", "0.30")),
            llm_provider=e.get("LLM_PROVIDER", "ollama").lower(),
            llm_endpoint=e.get("LLM_ENDPOINT", "retail-llm-hf"),
            ollama_model=e.get("OLLAMA_MODEL", "llama3.1:8b"),
            ollama_base_url=e.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            bedrock_model_id=e.get(
                "BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"
            ),
        )

    # ---- helpers S3 -----------------------------------------------------
    @property
    def is_aws_configured(self) -> bool:
        """Vrai si un bucket est renseigné : bascule les I/O vers S3."""
        return bool(self.bucket)

    def s3_uri(self, *parts: str) -> str:
        """``SETTINGS.s3_uri("raw", "sales")`` -> ``s3://bucket/raw/sales``."""
        if not self.bucket:
            raise ValueError("S3_BUCKET non configuré (voir .env)")
        tail = "/".join(p.strip("/") for p in parts if p)
        return f"s3://{self.bucket}/{tail}" if tail else f"s3://{self.bucket}"


SETTINGS = Settings.load()
