"""Batch Transform : applique le modèle approuvé et écrit ``forecast/``.

Le job lit le **modèle approuvé du Model Registry** (pas un chemin en dur),
fabrique les features d'inférence à partir de ``curated/``, envoie une ligne par
``(série, horizon)`` au conteneur, puis reconstruit un Parquet propre dans
``forecast/`` avec la version du modèle et la date d'exécution — c'est ce qui
permet de savoir, plus tard, qui a produit quelle prévision.

Usage::

    python -m infra.sagemaker.batch_transform
    python -m infra.sagemaker.batch_transform --horizons 1,7,14,28 --dry-run
"""

from __future__ import annotations

import argparse
import io
import json
import math
import tarfile
from datetime import datetime, timezone

import pandas as pd
from botocore.exceptions import ClientError, WaiterError

from infra.sagemaker.launch import _sm_client
from retail_forecast.config import SETTINGS
from retail_forecast.features.build import active_catalog, add_target, build_features
from retail_forecast.ingest.storage import s3_client

GROUP = "retail-forecast-lgbm"
INPUT_KEY = "inference/input/features.jsonl"
OUTPUT_PREFIX = "forecast/raw"
FORECAST_KEY = "forecast/table/forecast.parquet"
INSTANCE = "ml.m5.large"


def read_curated(s3) -> pd.DataFrame:
    """Lit **tout** ``curated/sales/`` (le job Glue écrit des ``part-*.parquet``).

    Lire une clé unique ``curated/sales/sales.parquet`` échouait dès que Glue
    avait repris la main sur le préfixe : on liste donc le préfixe et on
    concatène, comme le fait le canal d'entraînement SageMaker (``S3Prefix``).
    """
    paginator = s3.get_paginator("list_objects_v2")
    frames: list[pd.DataFrame] = []
    for page in paginator.paginate(Bucket=SETTINGS.bucket, Prefix="curated/sales/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith((".parquet", ".pq")):
                continue
            body = s3.get_object(Bucket=SETTINGS.bucket, Key=key)["Body"].read()
            frames.append(pd.read_parquet(io.BytesIO(body)))
            print(f"[transform] lu {key} ({obj['Size'] / 1024:.0f} Ko)")
    if not frames:
        raise FileNotFoundError("aucun Parquet sous curated/sales/")
    return pd.concat(frames, ignore_index=True)


def approved_model(sm) -> tuple[str, str, int]:
    """Renvoie le modèle actif validé, jamais le dernier Approved par défaut."""
    from infra.sagemaker.lifecycle import read_active

    active, _etag = read_active(s3_client())
    detail = sm.describe_model_package(ModelPackageName=active["model_package_arn"])
    if detail["ModelApprovalStatus"] != "Approved":
        raise RuntimeError("le modèle actif n'est plus approuvé")
    container = detail["InferenceSpecification"]["Containers"][0]
    return container["ModelDataUrl"], container["Image"], detail["ModelPackageVersion"]


def model_features(s3, model_url: str) -> list[str]:
    """Liste des features attendues, lue dans metadata.json de l'artefact."""
    return model_metadata(s3, model_url)["features"]


def model_metadata(s3, model_url: str) -> dict:
    """Charge le contrat de features et les vocabulaires figés à l'entraînement."""
    bucket, key = model_url.replace("s3://", "").split("/", 1)
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("metadata.json"))
        return json.loads(tar.extractfile(member).read())


def build_inference_rows(
    sales: pd.DataFrame, features: list[str], horizons: list[int], *, levels=None
):
    """Dernière origine connue par série, répliquée pour chaque horizon."""
    if not horizons or any(h < 1 for h in horizons) or len(set(horizons)) != len(horizons):
        raise ValueError("horizons requis, positifs et distincts")
    feats = build_features(active_catalog(sales), horizon=1, dropna=False, levels=levels)
    last = feats["date"].max()
    origin = feats[(feats["date"] == last) & feats[features].notna().all(axis=1)]
    if origin.empty:
        raise ValueError("aucune série avec un historique suffisant à la dernière date")
    rows = []
    for horizon in horizons:
        block = add_target(origin, horizon=horizon)
        block = block[["store_id", "item_id", "date", *features]]
        block["horizon"] = horizon
        block["target_date"] = last + pd.Timedelta(days=horizon)
        rows.append(block)
    out = pd.concat(rows, ignore_index=True)
    print(f"[transform] origine {last.date()} · {len(origin):,} séries "
          f"× {len(horizons)} horizons = {len(out):,} lignes")
    return out


def _role() -> str:
    """Rôle d'exécution du modèle.

    ``SAGEMAKER_ROLE_ARN`` n'est pas toujours défini (ex. exécution depuis une
    instance EC2 où seul ``LabRole`` est disponible) : on retombe sur le rôle du
    compte, sinon ``ExecutionRoleArn`` part vide et l'API répond
    « Invalid length for parameter ExecutionRoleArn ».
    """
    if SETTINGS.role_arn:
        return SETTINGS.role_arn
    import boto3 as _boto3

    account = _boto3.client("sts", region_name=SETTINGS.region).get_caller_identity()[
        "Account"
    ]
    return f"arn:aws:iam::{account}:role/LabRole"


def ensure_model(sm, name: str, image: str, model_url: str) -> str:
    """Crée la ressource Model (conteneur + artefact + handler d'inférence)."""
    try:
        existing = sm.describe_model(ModelName=name)['PrimaryContainer']
        if (existing.get('ModelDataUrl') != model_url or existing.get('Image') != image
                or existing.get('Environment', {}).get('SAGEMAKER_SUBMIT_DIRECTORY') != '/opt/ml/model/code'
                or existing.get('Environment', {}).get('SAGEMAKER_PROGRAM') != 'inference.py'):
            raise ValueError('modèle existant incompatible : recréer le déploiement')
        print(f"[transform] modèle existant : {name}")
    except ClientError as exc:
        if exc.response['Error']['Code'] != 'ValidationException':
            raise
        sm.create_model(
            ModelName=name,
            PrimaryContainer={
                "Image": image,
                "ModelDataUrl": model_url,
                "Environment": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                    # SKLearn exige ce répertoire explicite pour copier le
                    # handler vers /opt/ml/code avant son installation.
                    # LightGBM n'est pas dans l'image SKLearn : le conteneur
                    # installe requirements.txt au démarrage du serving.
                    "SAGEMAKER_REQUIREMENTS": "/opt/ml/model/requirements.txt",
                    # L'image python3.12 est « externally managed » (PEP 668) :
                    # sans cette variable, `pip install` échoue au démarrage et
                    # l'endpoint passe en Failed ("model process exited").
                    "PIP_BREAK_SYSTEM_PACKAGES": "1",
                },
            },
            ExecutionRoleArn=_role(),
        )
        print(f"[transform] modèle créé : {name}")
    return name


def upload_rows(s3, rows: pd.DataFrame, features: list[str], *, input_key: str = INPUT_KEY) -> str:
    """Écrit une ligne JSON par (série, horizon) dans S3."""
    cols = ["store_id", "item_id", "date", "horizon", "target_date", *features]
    payload = rows[cols].copy()
    payload["date"] = payload["date"].astype(str)
    payload["target_date"] = payload["target_date"].astype(str)
    body = "\n".join(
        json.dumps(rec, default=float) for rec in payload.to_dict("records")
    ).encode("utf-8")
    s3.put_object(Bucket=SETTINGS.bucket, Key=input_key, Body=body)
    print(
        f"[transform] entrée {len(rows):,} lignes ({len(body) / 1e6:.2f} Mo) -> "
        f"s3://{SETTINGS.bucket}/{input_key}"
    )
    return f"s3://{SETTINGS.bucket}/{input_key}"


def run_transform(sm, job_name: str, model_name: str, input_uri: str) -> None:
    """Crée et attend le Transform Job."""
    sm.create_transform_job(
        TransformJobName=job_name,
        ModelName=model_name,
        BatchStrategy="SingleRecord",
        MaxConcurrentTransforms=1,
        MaxPayloadInMB=2,
        TransformInput={
            "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": input_uri}},
            "ContentType": "application/json",
            "SplitType": "Line",
        },
        TransformOutput={
            "S3OutputPath": f"s3://{SETTINGS.bucket}/{OUTPUT_PREFIX}/{job_name}/",
            "Accept": "application/json",
            "AssembleWith": "Line",
        },
        TransformResources={"InstanceType": INSTANCE, "InstanceCount": 1},
    )
    print(f"[transform] job {job_name} sur {INSTANCE} — attente…")
    try:
        sm.get_waiter("transform_job_completed_or_stopped").wait(
            TransformJobName=job_name, WaiterConfig={"Delay": 15, "MaxAttempts": 60}
        )
    except WaiterError:
        if sm.describe_transform_job(TransformJobName=job_name)["TransformJobStatus"] == "InProgress":
            sm.stop_transform_job(TransformJobName=job_name)
        raise
    status = sm.describe_transform_job(TransformJobName=job_name)["TransformJobStatus"]
    print(f"[transform] statut {status}")
    if status != "Completed":
        raise SystemExit(f"transform job {status} : voir CloudWatch")


def _prediction(line: str) -> float:
    """Extrait le scalaire d'une ligne de sortie (formats conteneur variables)."""
    value = json.loads(line)
    while isinstance(value, list):
        if len(value) != 1:
            raise ValueError("une unique prédiction requise par ligne")
        value = value[0]
    if isinstance(value, dict):
        if len(value) != 1:
            raise ValueError("une unique prédiction requise par ligne")
        return _prediction(json.dumps(next(iter(value.values()))))
    return float(value)


def collect_output(s3, rows: pd.DataFrame, version: int, job_name: str, *, forecast_key: str = FORECAST_KEY) -> pd.DataFrame:
    """Télécharge la sortie du transform, la recolle aux lignes d'entrée."""
    # Un seul fichier d'entrée : la sortie attendue porte exactement son nom.
    # Lire tout forecast/raw/ mélangeait les anciens jobs avec le job courant.
    keys = [f"{OUTPUT_PREFIX}/{job_name}/features.jsonl.out"]
    lines: list[str] = []
    for key in keys:
        body = s3.get_object(Bucket=SETTINGS.bucket, Key=key)["Body"].read().decode()
        lines.extend(l for l in body.splitlines() if l.strip())
    if len(lines) != len(rows):
        raise SystemExit(f"{len(lines)} prédictions pour {len(rows)} lignes d'entrée")

    forecast = rows[["store_id", "item_id", "date", "target_date", "horizon"]].copy()
    forecast = forecast.rename(columns={"date": "origin_date"})
    forecast["forecast_units"] = [_prediction(l) for l in lines]
    if not forecast["forecast_units"].map(lambda value: math.isfinite(value) and value >= 0).all():
        raise ValueError("prévisions invalides : publication du Parquet bloquée")
    forecast["model_version"] = f"v{version}"
    forecast["model_job"] = job_name
    forecast["run_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    buf = io.BytesIO()
    forecast.to_parquet(buf, index=False, compression="snappy")
    s3.put_object(
        Bucket=SETTINGS.bucket,
        Key=forecast_key,
        Body=buf.getvalue(),
        ContentType="application/parquet",
    )
    print(f"[transform] écrit s3://{SETTINGS.bucket}/{forecast_key} "
          f"({len(forecast):,} lignes)")
    return forecast


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch Transform -> forecast/")
    parser.add_argument("--horizons", default=",".join(map(str, range(1, 17))))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="dépose seulement l'entrée du job (parcours console)",
    )
    args = parser.parse_args(argv)
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]

    sm, s3 = _sm_client(), s3_client()
    model_url, image, version = approved_model(sm)
    print(f"[transform] modèle approuvé v{version} : {model_url}")

    metadata = model_metadata(s3, model_url)
    features = metadata["features"]
    if not set(horizons).issubset(metadata["horizons"]):
        raise SystemExit("horizons demandés absents du modèle approuvé")
    sales = read_curated(s3)
    sales["date"] = pd.to_datetime(sales["date"])
    rows = build_inference_rows(
        sales, features, horizons, levels=metadata.get("categorical_levels")
    )

    if args.dry_run:
        print(rows[["store_id", "item_id", "horizon", "target_date"]].head(6).to_string())
        return 0

    if args.prepare_only:
        # Sert au parcours 100 % console : on dépose seulement l'entrée du job,
        # puis on crée le modèle et le Batch Transform à la main dans la console.
        uri = upload_rows(s3, rows, features)
        print(f"[transform] entrée prête : {uri} ({len(rows):,} lignes)")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    model_name = f"retail-lgbm-v{version}-{stamp}"
    ensure_model(sm, model_name, image, model_url)
    job_name = f"retail-transform-{stamp}"
    input_uri = upload_rows(
        s3, rows, features, input_key=f"inference/input/{job_name}/features.jsonl"
    )
    run_transform(sm, job_name, model_name, input_uri)
    forecast = collect_output(s3, rows, version, job_name)

    preview = forecast.head(5)[
        ["store_id", "item_id", "target_date", "horizon", "forecast_units"]
    ]
    print(preview.to_string(index=False))
    print(
        f"[transform] total {forecast['forecast_units'].sum():,.0f} unités prévues "
        f"sur {forecast['target_date'].nunique()} dates cibles"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
