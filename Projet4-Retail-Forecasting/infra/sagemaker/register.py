"""Enregistre le modèle entraîné dans le SageMaker Model Registry.

Deux objets : un *model package group* (le conteneur logique, créé une fois) et
un *model package* (une version, avec ses métriques et son statut d'approbation).

Ce parcours autonome conserve le statut PendingManualApproval. Seul le
pipeline avec évaluation indépendante peut approuver automatiquement ; le
manifeste actif est ensuite publié après validation de l'inférence.

Usage::

    python -m infra.sagemaker.register                       # dernier job réussi
    python -m infra.sagemaker.register --job retail-lgbm-...
"""

from __future__ import annotations

import argparse
import json
import math

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


def gate_decision(metrics: dict, threshold: float) -> tuple[float, bool, str]:
    """Décide de l'approbation : WAPE moyen des horizons comparé au seuil.

    Même grandeur que la condition ``GateWAPE`` du pipeline SageMaker. Les
    artefacts antérieurs à l'ajout de ``mean_wape`` retombent sur le WAPE de
    l'horizon principal.
    """
    wape_gate = float(metrics["mean_wape"] if "mean_wape" in metrics else metrics["test_wape"])
    approuve = math.isfinite(wape_gate) and 0 <= wape_gate <= threshold
    return wape_gate, approuve, "Approved" if approuve else "PendingManualApproval"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Registry : enregistre le modèle.")
    parser.add_argument("--job", default="", help="Training Job à enregistrer")
    parser.add_argument("--threshold", type=float, default=SETTINGS.wape_threshold)
    args = parser.parse_args(argv)

    sm = _sm_client()
    job = args.job or latest_completed_job(sm)
    metrics = read_metrics(s3_client(), SETTINGS.bucket, job)

    # Même grandeur que le gate du pipeline SageMaker : le WAPE moyen des
    # horizons s'il est présent (nouveaux artefacts), sinon celui de l'horizon
    # principal (artefacts d'avant l'ajout de mean_wape).
    gate_wape, approved, status = gate_decision(metrics, args.threshold)
    # Le chemin autonome ne dispose pas de preuve indépendante.
    approved, status = False, "PendingManualApproval"
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
            "mean_wape": f"{gate_wape:.4f}",
            "baseline_wape": f"{metrics['baseline_wape']:.4f}",
            "improvement": f"{metrics['improvement_vs_baseline']:.4f}",
            "test_bias": f"{metrics['test_bias']:+.4f}",
            "horizon": str(metrics["horizon"]),
        },
    )
    arn = response["ModelPackageArn"]
    print(f"[registry] version enregistrée ({status}) : {arn}")
    print(f"[registry] WAPE moyen {gate_wape:.4f} <= seuil {args.threshold} -> "
          f"{'approuvée' if approved else 'approbation manuelle requise'}")
    print("[registry] " + json.dumps({"model_data": model_url}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
