"""Enregistre le modèle entraîné dans le SageMaker Model Registry.

Deux objets : un *model package group* (le conteneur logique, créé une fois) et
un *model package* (une version, avec ses métriques et son statut d'approbation).

C'est la boîte « Model Registry / Nouvelle version » du diagramme et ce qui
permet au pipeline d'inférence de ne charger qu'un modèle **approuvé**.

Usage::

    python -m infra.sagemaker.register                       # dernier job réussi
    python -m infra.sagemaker.register --job retail-lgbm-...
"""

from __future__ import annotations

import argparse
import json

from infra.sagemaker.launch import IMAGE, read_metrics, _sm_client
from retail_forecast.config import SETTINGS
from retail_forecast.ingest.storage import s3_client

GROUP = "retail-forecast-lgbm"


def ensure_group(sm) -> None:
    """Crée le model package group s'il n'existe pas."""
    try:
        sm.describe_model_package_group(ModelPackageGroupName=GROUP)
        print(f"[registry] groupe existant : {GROUP}")
    except sm.exceptions.ClientError:
        sm.create_model_package_group(
            ModelPackageGroupName=GROUP,
            ModelPackageGroupDescription="Prévision de la demande J+1 (LightGBM global).",
        )
        print(f"[registry] groupe créé : {GROUP}")


def latest_completed_job(sm) -> str:
    """Nom du Training Job réussi le plus récent."""
    jobs = sm.list_training_jobs(
        StatusEquals="Completed", SortBy="CreationTime", SortOrder="Descending", MaxResults=5
    )["TrainingJobSummaries"]
    if not jobs:
        raise SystemExit("aucun Training Job terminé : lancer --batch A d'abord")
    return jobs[0]["TrainingJobName"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Registry : enregistre le modèle.")
    parser.add_argument("--job", default="", help="Training Job à enregistrer")
    parser.add_argument("--threshold", type=float, default=SETTINGS.wape_threshold)
    args = parser.parse_args(argv)

    sm = _sm_client()
    job = args.job or latest_completed_job(sm)
    metrics = read_metrics(s3_client(), SETTINGS.bucket, job)

    approved = metrics["test_wape"] <= args.threshold
    status = "Approved" if approved else "PendingManualApproval"
    model_url = f"s3://{SETTINGS.bucket}/models/{job}/output/model.tar.gz"

    ensure_group(sm)
    response = sm.create_model_package(
        ModelPackageGroupName=GROUP,
        ModelPackageDescription=f"LightGBM global — job {job}",
        ModelApprovalStatus=status,
        InferenceSpecification={
            "Containers": [{"Image": IMAGE, "ModelDataUrl": model_url}],
            "SupportedTransformInstanceTypes": ["ml.m5.large"],
            "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.large"],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
        },
        CustomerMetadataProperties={
            "training_job": job,
            "test_wape": f"{metrics['test_wape']:.4f}",
            "baseline_wape": f"{metrics['baseline_wape']:.4f}",
            "improvement": f"{metrics['improvement_vs_baseline']:.4f}",
            "test_bias": f"{metrics['test_bias']:+.4f}",
            "horizon": str(metrics["horizon"]),
        },
    )
    arn = response["ModelPackageArn"]
    print(f"[registry] version enregistrée ({status}) : {arn}")
    print(f"[registry] WAPE {metrics['test_wape']:.4f} <= seuil {args.threshold} -> "
          f"{'approuvée' if approved else 'approbation manuelle requise'}")
    print("[registry] " + json.dumps({"model_data": model_url}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
