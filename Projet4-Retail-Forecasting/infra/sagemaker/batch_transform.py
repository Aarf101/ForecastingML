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
import tarfile
from datetime import datetime, timezone

import pandas as pd

from infra.sagemaker.launch import _sm_client
from retail_forecast.config import SETTINGS
from retail_forecast.features.build import build_features
from retail_forecast.ingest.storage import s3_client

GROUP = "retail-forecast-lgbm"
INPUT_KEY = "inference/input/features.jsonl"
OUTPUT_PREFIX = "forecast/raw"
FORECAST_KEY = "forecast/table/forecast.parquet"
INSTANCE = "ml.m5.large"


def approved_model(sm) -> tuple[str, str, int]:
    """Renvoie (ModelDataUrl, Image, version) du dernier modèle approuvé."""
    packages = sm.list_model_packages(
        ModelPackageGroupName=GROUP,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )["ModelPackageSummaryList"]
    if not packages:
        raise SystemExit(f"aucun modèle approuvé dans {GROUP}")
    version = packages[0]["ModelPackageVersion"]
    detail = sm.describe_model_package(ModelPackageName=packages[0]["ModelPackageArn"])
    container = detail["InferenceSpecification"]["Containers"][0]
    return container["ModelDataUrl"], container["Image"], version


def model_features(s3, model_url: str) -> list[str]:
    """Liste des features attendues, lue dans metadata.json de l'artefact."""
    bucket, key = model_url.replace("s3://", "").split("/", 1)
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("metadata.json"))
        return json.loads(tar.extractfile(member).read())["features"]


def build_inference_rows(sales: pd.DataFrame, features: list[str], horizons: list[int]):
    """Dernière origine connue par série, répliquée pour chaque horizon."""
    feats = build_features(sales, horizon=1, dropna=False)
    last = feats["date"].max()
    origin = feats[(feats["date"] == last) & feats[features].notna().all(axis=1)]
    origin = origin[["store_id", "item_id", "date", *features]]
    rows = []
    for horizon in horizons:
        block = origin.copy()
        block["horizon"] = horizon
        block["target_date"] = last + pd.Timedelta(days=horizon)
        rows.append(block)
    out = pd.concat(rows, ignore_index=True)
    print(f"[transform] origine {last.date()} · {len(origin):,} séries "
          f"× {len(horizons)} horizons = {len(out):,} lignes")
    return out


def ensure_model(sm, name: str, image: str, model_url: str) -> str:
    """Crée la ressource Model (conteneur + artefact + handler d'inférence)."""
    try:
        sm.describe_model(ModelName=name)
        print(f"[transform] modèle existant : {name}")
    except sm.exceptions.ClientError:
        sm.create_model(
            ModelName=name,
            PrimaryContainer={
                "Image": image,
                "ModelDataUrl": model_url,
                "Environment": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model",
                    # LightGBM n'est pas dans l'image SKLearn : le conteneur
                    # installe requirements.txt au démarrage du serving.
                    "SAGEMAKER_REQUIREMENTS": "/opt/ml/model/requirements.txt",
                },
            },
            ExecutionRoleArn=SETTINGS.role_arn,
        )
        print(f"[transform] modèle créé : {name}")
    return name


def upload_rows(s3, rows: pd.DataFrame, features: list[str]) -> str:
    """Écrit une ligne JSON par (série, horizon) dans S3."""
    cols = ["store_id", "item_id", "date", "horizon", "target_date", *features]
    payload = rows[cols].copy()
    payload["date"] = payload["date"].astype(str)
    payload["target_date"] = payload["target_date"].astype(str)
    body = "\n".join(
        json.dumps(rec, default=float) for rec in payload.to_dict("records")
    ).encode("utf-8")
    s3.put_object(Bucket=SETTINGS.bucket, Key=INPUT_KEY, Body=body)
    print(
        f"[transform] entrée {len(rows):,} lignes ({len(body) / 1e6:.2f} Mo) -> "
        f"s3://{SETTINGS.bucket}/{INPUT_KEY}"
    )
    return f"s3://{SETTINGS.bucket}/{INPUT_KEY}"


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
            "S3OutputPath": f"s3://{SETTINGS.bucket}/{OUTPUT_PREFIX}/",
            "Accept": "application/json",
            "AssembleWith": "Line",
        },
        TransformResources={"InstanceType": INSTANCE, "InstanceCount": 1},
    )
    print(f"[transform] job {job_name} sur {INSTANCE} — attente…")
    sm.get_waiter("transform_job_completed_or_stopped").wait(
        TransformJobName=job_name, WaiterConfig={"Delay": 15, "MaxAttempts": 40}
    )
    status = sm.describe_transform_job(TransformJobName=job_name)["TransformJobStatus"]
    print(f"[transform] statut {status}")
    if status != "Completed":
        raise SystemExit(f"transform job {status} : voir CloudWatch")


def _prediction(line: str) -> float:
    """Extrait le scalaire d'une ligne de sortie (formats conteneur variables)."""
    value = json.loads(line)
    while isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, dict):
        return _prediction(json.dumps(next(iter(value.values()))))
    return float(value)


def collect_output(s3, rows: pd.DataFrame, version: int, job_name: str) -> pd.DataFrame:
    """Télécharge la sortie du transform, la recolle aux lignes d'entrée."""
    listed = s3.list_objects_v2(Bucket=SETTINGS.bucket, Prefix=OUTPUT_PREFIX + "/")
    keys = [o["Key"] for o in listed.get("Contents", []) if not o["Key"].endswith("/")]
    if not keys:
        raise SystemExit("aucune sortie dans forecast/raw/ — vérifier les logs du job")
    lines: list[str] = []
    for key in keys:
        body = s3.get_object(Bucket=SETTINGS.bucket, Key=key)["Body"].read().decode()
        lines.extend(l for l in body.splitlines() if l.strip())
    if len(lines) != len(rows):
        raise SystemExit(f"{len(lines)} prédictions pour {len(rows)} lignes d'entrée")

    forecast = rows[["store_id", "item_id", "date", "target_date", "horizon"]].copy()
    forecast = forecast.rename(columns={"date": "origin_date"})
    forecast["forecast_units"] = [_prediction(l) for l in lines]
    forecast["model_version"] = f"v{version}"
    forecast["model_job"] = job_name
    forecast["run_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    buf = io.BytesIO()
    forecast.to_parquet(buf, index=False, compression="snappy")
    s3.put_object(
        Bucket=SETTINGS.bucket,
        Key=FORECAST_KEY,
        Body=buf.getvalue(),
        ContentType="application/parquet",
    )
    print(f"[transform] écrit s3://{SETTINGS.bucket}/{FORECAST_KEY} "
          f"({len(forecast):,} lignes)")
    return forecast


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch Transform -> forecast/")
    parser.add_argument("--horizons", default="1,7,14,28")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]

    sm, s3 = _sm_client(), s3_client()
    model_url, image, version = approved_model(sm)
    print(f"[transform] modèle approuvé v{version} : {model_url}")

    features = model_features(s3, model_url)
    sales = pd.read_parquet(
        io.BytesIO(
            s3.get_object(Bucket=SETTINGS.bucket, Key="curated/sales/sales.parquet")[
                "Body"
            ].read()
        )
    )
    sales["date"] = pd.to_datetime(sales["date"])
    rows = build_inference_rows(sales, features, horizons)

    if args.dry_run:
        print(rows[["store_id", "item_id", "horizon", "target_date"]].head(6).to_string())
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    model_name = f"retail-lgbm-v{version}-{stamp}"
    ensure_model(sm, model_name, image, model_url)
    input_uri = upload_rows(s3, rows, features)

    job_name = f"retail-transform-{stamp}"
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
