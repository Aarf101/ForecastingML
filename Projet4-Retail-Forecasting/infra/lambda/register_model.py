"""Handler Lambda — gate WAPE + enregistrement au Model Registry.

Utilisé comme état ``GateWAPE`` de la state machine Step Functions : il relit les
métriques du Training Job directement dans son artefact S3, compare ``test_wape``
au seuil, puis enregistre (ou bloque) la version du modèle.

Aucune dépendance hors ``boto3`` : l'archive ``model.tar.gz`` est lue avec
``tarfile`` (stdlib), donc pas de couche Lambda à maintenir.

Entrée attendue (event) :
  bucket        bucket du lac
  training_job  nom du Training Job qui vient de finir
  threshold     seuil de WAPE (défaut WAPE_THRESHOLD ou 0.45)
  group         groupe de packages (défaut ``retail-forecast-lgbm``)
  image         image d'inférence (défaut : conteneur SKLearn managé)

Sortie : ``{"approved": bool, "wape": float, "baseline_wape": float, ...}``
"""

import io
import json
import os
import tarfile

import boto3

DEFAULT_IMAGE = (
    "683313688378.dkr.ecr.us-east-1.amazonaws.com/"
    "sagemaker-scikit-learn:1.4-2-py312-cpu-py3"
)

_s3 = boto3.client("s3")
_sm = boto3.client("sagemaker")


def _metrics(bucket: str, job: str) -> dict:
    """Extrait ``metrics.json`` de ``model.tar.gz`` sans le décompresser sur disque."""
    key = f"models/{job}/output/model.tar.gz"
    body = _s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("metrics.json"))
        return json.loads(tar.extractfile(member).read())


def lambda_handler(event, context=None):
    bucket = event["bucket"]
    job = event["training_job"]
    threshold = float(event.get("threshold") or os.environ.get("WAPE_THRESHOLD", 0.45))
    group = event.get("group", "retail-forecast-lgbm")
    image = event.get("image", DEFAULT_IMAGE)

    metrics = _metrics(bucket, job)
    wape = float(metrics["test_wape"])
    baseline = float(metrics.get("baseline_wape", 0.0))
    approved = wape <= threshold
    status = "Approved" if approved else "PendingManualApproval"

    print(
        f"[gate] {job} : WAPE {wape:.4f} vs seuil {threshold:.2f} "
        f"-> {status} (baseline {baseline:.4f})"
    )

    try:
        _sm.describe_model_package_group(ModelPackageGroupName=group)
    except _sm.exceptions.ClientError:
        _sm.create_model_package_group(
            ModelPackageGroupName=group,
            ModelPackageGroupDescription="Retail demand forecasting (LightGBM)",
        )

    response = _sm.create_model_package(
        ModelPackageGroupName=group,
        ModelPackageDescription=f"LightGBM {job} — WAPE {wape:.4f}",
        ModelApprovalStatus=status,
        InferenceSpecification={
            "Containers": [
                {
                    "Image": image,
                    "ModelDataUrl": f"s3://{bucket}/models/{job}/output/model.tar.gz",
                    "Environment": {
                        "SAGEMAKER_PROGRAM": "inference.py",
                        "SAGEMAKER_SUBMIT_DIRECTORY": f"s3://{bucket}/code/sourcedir.tar.gz",
                        "SAGEMAKER_REQUIREMENTS": "/opt/ml/model/requirements.txt",
                    },
                }
            ],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
            "SupportedTransformInstanceTypes": ["ml.m5.large"],
            "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.large"],
        },
        CustomerMetadataProperties={
            "test_wape": f"{wape:.6f}",
            "baseline_wape": f"{baseline:.6f}",
            "training_job": job,
            "threshold": f"{threshold:.2f}",
        },
    )

    arn = response["ModelPackageArn"]
    print(f"[gate] version enregistrée : {arn} ({status})")
    return {
        "approved": approved,
        "status": status,
        "wape": wape,
        "baseline_wape": baseline,
        "threshold": threshold,
        "model_package_arn": arn,
    }
