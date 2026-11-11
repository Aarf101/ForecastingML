"""Mise en place complète du projet **depuis l'intérieur d'AWS**.

Point d'entrée « zéro vers tout » : à exécuter dans le terminal d'une instance
SageMaker ou dans CloudShell, sans poste de travail et sans ``make``. Chaque
étape est idempotente — relancer le script ne casse rien et ne recrée que ce qui
manque.

Étapes :
  1. bucket S3 (chiffrement + blocage public)
  2. dataset : génération puis dépôt dans ``raw/historique/`` s'il est absent
  3. code d'entraînement : ``code/sourcedir.tar.gz``
  4. job Glue ``retail-etl-raw-to-curated`` (script poussé dans S3)
  5. ingestion : Kinesis + Firehose + Lambda producteur
  6. orchestration : Lambda de gate + state machine + règle EventBridge
  7. endpoint LLM Hugging Face (optionnel, ``--with-llm``)

Usage :
    python -m infra.aws.setup                  # tout sauf le LLM
    python -m infra.aws.setup --with-llm       # + endpoint serverless
    python -m infra.aws.setup --inventory      # seulement l'état des ressources
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import boto3

from infra.aws import kinesis_ingest as ingest
from infra.aws import orchestrate as orch
from retail_forecast.config import SETTINGS

REGION = SETTINGS.region
HISTORY_KEY = "raw/historique/sales.parquet"


def _s3():
    return boto3.client("s3", region_name=REGION)


def ensure_bucket() -> None:
    """Crée le bucket s'il manque, avec chiffrement et blocage de l'accès public."""
    s3 = _s3()
    try:
        s3.head_bucket(Bucket=SETTINGS.bucket)
        print(f"[s3] {SETTINGS.bucket} existe déjà")
        return
    except Exception:  # noqa: BLE001 - on tente la création
        pass

    s3.create_bucket(Bucket=SETTINGS.bucket)
    s3.put_bucket_encryption(
        Bucket=SETTINGS.bucket,
        ServerSideEncryptionConfiguration={
            "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
        },
    )
    s3.put_public_access_block(
        Bucket=SETTINGS.bucket,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    print(f"[s3] {SETTINGS.bucket} créé (chiffrement AES256, public bloqué)")


def ensure_dataset(regenerate: bool = False) -> None:
    """Dépose ``raw/historique/sales.parquet`` ; le génère s'il n'existe pas."""
    s3 = _s3()
    if not regenerate:
        try:
            s3.head_object(Bucket=SETTINGS.bucket, Key=HISTORY_KEY)
            print(f"[data] s3://{SETTINGS.bucket}/{HISTORY_KEY} présent")
            return
        except Exception:  # noqa: BLE001
            pass

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "sales.parquet"
        print("[data] génération du jeu synthétique (≈1 min)…")
        subprocess.check_call(
            [sys.executable, "-m", "retail_forecast.data.simulate", "--out", str(out)]
        )
        size = out.stat().st_size
        s3.upload_file(str(out), SETTINGS.bucket, HISTORY_KEY)
        print(f"[data] déposé ({size / 1e6:.2f} Mo)")


def ensure_training_code() -> None:
    """Empaquette et dépose ``code/sourcedir.tar.gz``."""
    from infra.sagemaker.launch import upload_code

    uri = upload_code(_s3(), SETTINGS.bucket)
    print(f"[code] {uri}")


def ensure_llm() -> None:
    """Crée l'endpoint serverless Hugging Face (s'il n'existe pas)."""
    code = subprocess.call(
        [sys.executable, "-m", "infra.sagemaker.llm_endpoint", "--create", "--wait"]
    )
    if code:
        print("[llm] création impossible — la suite du pipeline n'en dépend pas")


def inventory() -> None:
    """Inventaire lisible de ce qui existe dans le compte."""
    ingest.status(ingest._clients())
    orch.status(orch._clients())
    s3 = _s3()
    paginator = s3.get_paginator("list_objects_v2")
    total = size = 0
    try:
        for page in paginator.paginate(Bucket=SETTINGS.bucket):
            for obj in page.get("Contents", []):
                total += 1
                size += obj["Size"]
        print(f"s3       : {total} objet(s) · {size / 1e6:.2f} Mo")
    except Exception as exc:  # noqa: BLE001
        print(f"s3       : illisible ({str(exc).splitlines()[0][:70]})")
    try:
        sm = boto3.client("sagemaker", region_name=REGION)
        groups = sm.list_model_package_groups().get("ModelPackageGroupSummaryList", [])
        print(f"registry : {len(groups)} groupe(s)")
    except Exception:  # noqa: BLE001
        print("registry : ?")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-llm", action="store_true", help="créer l'endpoint LLM")
    parser.add_argument("--no-schedule", action="store_true", help="pas de règle EventBridge")
    parser.add_argument("--schedule", default="rate(1 day)", help="expression EventBridge")
    parser.add_argument("--regenerate-data", action="store_true", help="refaire le dataset")
    parser.add_argument("--inventory", action="store_true", help="afficher seulement l'état")
    args = parser.parse_args(argv)

    if not SETTINGS.bucket:
        raise SystemExit(
            "S3_BUCKET vide : définir S3_BUCKET (ex. retail-forecast-<compte>-<nom>)"
        )

    print(f"[init] region={REGION} bucket={SETTINGS.bucket}")
    if args.inventory:
        inventory()
        return 0

    ensure_bucket()
    ensure_dataset(args.regenerate_data)
    ensure_training_code()

    ic = ingest._clients()
    role = ingest._role_arn(ic)
    print(f"[iam] rôle d'exécution : {role}")

    ingest.create_stream(ic)
    ingest.create_delivery_stream(ic, role)
    ingest.deploy_lambda(ic, role, SETTINGS.bucket)

    oc = orch._clients()
    orch.deploy_glue_job(oc, role)
    gate_arn = orch.deploy_lambda(oc, role)
    machine_arn = orch.deploy_machine(oc, role, gate_arn)
    if args.no_schedule:
        print("[events] pas de planification")
    else:
        orch.deploy_rule(oc, machine_arn, role, args.schedule)

    if args.with_llm:
        ensure_llm()

    print("\n== inventaire ==")
    inventory()
    print("\nMise en place terminée. Étape suivante : python -m infra.aws.full_run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

