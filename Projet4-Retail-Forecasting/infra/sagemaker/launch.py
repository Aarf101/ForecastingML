"""Lance un Training Job SageMaker (LightGBM, conteneur SKLearn managé).

Pas de SDK ``sagemaker`` : uniquement boto3, pour rester léger et lisible.
Le code (ce script d'entrée + le paquet ``retail_forecast``) est empaqueté en
``sourcedir.tar.gz`` et déposé dans S3 ; SageMaker le télécharge dans le
conteneur et exécute ``train_entry.py`` avec les hyperparamètres en arguments.

Usage::

    python -m infra.sagemaker.launch --wait
    python -m infra.sagemaker.launch --dry-run      # affiche la requête, ne lance rien
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

from retail_forecast.config import ROOT, SETTINGS
from retail_forecast.ingest.storage import s3_client

# Image managée SageMaker scikit-learn (us-east-1), publiée par AWS.
IMAGE = (
    "683313688378.dkr.ecr.us-east-1.amazonaws.com/"
    "sagemaker-scikit-learn:1.4-2-py312-cpu-py3"
)

PROGRAM = "train_entry.py"
SOURCEDIR_KEY = "code/sourcedir.tar.gz"
INSTANCE = "ml.m5.large"  # 2 vCPU / 8 Go — le moins cher qui suffit
MAX_RUNTIME = 1200  # 20 min : garde-fou anti-dépense


def _sm_client():
    """Client boto3 SageMaker (S3 et SageMaker ne partagent pas le même client)."""
    import boto3

    kwargs: dict = {"region_name": SETTINGS.region}
    if SETTINGS.s3_access_key:  # uniquement en mode S3 local
        kwargs["aws_access_key_id"] = SETTINGS.s3_access_key
        kwargs["aws_secret_access_key"] = SETTINGS.s3_secret_key
    return boto3.client("sagemaker", **kwargs)


def build_tarball() -> bytes:
    """Empaquette le paquet retail_forecast + le script d'entrée + requirements."""
    src = ROOT / "src" / "retail_forecast"
    entry = Path(__file__).with_name("train_entry.py")
    handler = Path(__file__).with_name("inference.py")
    reqs = Path(__file__).with_name("requirements.txt")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in sorted(src.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            tar.add(path, arcname=str(Path("retail_forecast") / path.relative_to(src)))
        tar.add(entry, arcname=PROGRAM)
        tar.add(handler, arcname="inference.py")
        tar.add(reqs, arcname="requirements.txt")
    return buf.getvalue()


def upload_code(client, bucket: str) -> str:
    """Dépose sourcedir.tar.gz dans S3 et renvoie son URI."""
    body = build_tarball()
    client.put_object(
        Bucket=bucket, Key=SOURCEDIR_KEY, Body=body, ContentType="application/gzip"
    )
    uri = f"s3://{bucket}/{SOURCEDIR_KEY}"
    print(f"[launch] code empaqueté : {len(body) / 1024:.0f} Ko -> {uri}")
    return uri


def job_request(bucket: str, job_name: str, code_uri: str, args) -> dict:
    """Construit la requête CreateTrainingJob."""
    return {
        "TrainingJobName": job_name,
        "AlgorithmSpecification": {
            "TrainingImage": IMAGE,
            "TrainingInputMode": "File",
        },
        "RoleArn": SETTINGS.role_arn,
        "InputDataConfig": [
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{bucket}/curated/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "CompressionType": "None",
                "ContentType": "application/x-parquet",
            }
        ],
        "OutputDataConfig": {"S3OutputPath": f"s3://{bucket}/models/"},
        "ResourceConfig": {
            "InstanceType": INSTANCE,
            "InstanceCount": 1,
            "VolumeSizeInGB": 30,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": MAX_RUNTIME},
        "HyperParameters": {
            "sagemaker_program": PROGRAM,
            "sagemaker_submit_directory": code_uri,
            "sagemaker_container_log_level": "20",
            "sagemaker_region": SETTINGS.region,
            "horizons": args.horizons,
            "n-estimators": str(args.estimators),
            "early-stopping-rounds": str(args.early_stopping),
        },
        "Tags": [
            {"Key": "project", "Value": "retail-forecast"},
            {"Key": "phase", "Value": "training"},
        ],
    }


def read_metrics(client, bucket: str, job_name: str) -> dict:
    """Extrait metrics.json du model.tar.gz produit par le job."""
    key = f"models/{job_name}/output/model.tar.gz"
    body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("metrics.json"))
        return json.loads(tar.extractfile(member).read())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Training Job SageMaker (LightGBM).")
    parser.add_argument("--horizons", default="1,7,14,28", help="ex. 1,7,14,28")
    parser.add_argument("--estimators", type=int, default=300)
    parser.add_argument("--early-stopping", type=int, default=40)
    parser.add_argument("--wait", action="store_true", help="attendre la fin du job")
    parser.add_argument("--dry-run", action="store_true", help="ne rien envoyer à AWS")
    parser.add_argument("--name", default="", help="nom du job (défaut horodaté)")
    args = parser.parse_args(argv)

    if not SETTINGS.bucket or not SETTINGS.role_arn:
        raise SystemExit("S3_BUCKET et SAGEMAKER_ROLE_ARN doivent être renseignés (.env)")

    client = s3_client()
    sm = _sm_client()
    job_name = args.name or f"retail-lgbm-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    code_uri = f"s3://{SETTINGS.bucket}/{SOURCEDIR_KEY}"
    request = job_request(SETTINGS.bucket, job_name, code_uri, args)

    if args.dry_run:
        print(json.dumps(request, indent=2))
        return 0

    upload_code(client, SETTINGS.bucket)
    print(f"[launch] job {job_name} sur {INSTANCE} (max {MAX_RUNTIME // 60} min)")
    sm.create_training_job(**request)

    if not args.wait:
        print("[launch] lancé — suivi : AWS_PROFILE=lab aws sagemaker describe-training-job "
              f"--training-job-name {job_name} --query TrainingJobStatus")
        return 0

    waiter = sm.get_waiter("training_job_completed_or_stopped")
    started = time.time()
    waiter.wait(TrainingJobName=job_name, WaiterConfig={"Delay": 20, "MaxAttempts": 60})
    minutes = (time.time() - started) / 60
    job = sm.describe_training_job(TrainingJobName=job_name)
    status = job["TrainingJobStatus"]
    print(f"[launch] statut {status} en {minutes:.1f} min (~{minutes / 60 * 0.115:.3f} €)")

    if status != "Completed":
        print(f"[launch] échec : {job.get('FailureReason', 'inconnue')}")
        return 1

    metrics = read_metrics(client, SETTINGS.bucket, job_name)
    print("[launch] métriques :")
    for key in (
        "test_wape",
        "baseline_wape",
        "improvement_vs_baseline",
        "test_rmse",
        "test_bias",
        "best_iteration",
    ):
        print(f"  {key:<26} {metrics.get(key)}")
    print(f"[launch] artefact : s3://{SETTINGS.bucket}/models/{job_name}/output/model.tar.gz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
