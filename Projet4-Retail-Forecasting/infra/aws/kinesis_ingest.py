"""Ingestion 100 % AWS : Kinesis Data Streams -> Firehose -> S3 ``raw/sales/``.

Remplace la chaîne locale ``Redpanda + to_s3.py`` déployée jusqu'ici. Trois objets
sont créés dans le compte :

  * **Kinesis Data Streams** ``retail-sales-stream`` (1 shard, mode PROVISIONED) :
    le transport d'événements, équivalent managé de Kafka ;
  * **Firehose** ``retail-sales-to-s3`` : consomme le stream et dépose des objets
    gzip dans ``s3://<bucket>/raw/sales/ingest_date=…/`` (buffer 60 s / 5 Mo) ;
  * **Lambda** ``retail-replay-to-kinesis`` : le producteur, exécuté dans AWS. Il
    relit le dataset depuis S3 et publie dans le stream.

Coût : 0,015 $/h par shard Kinesis + 0,029 $/Go livré par Firehose + Lambda au
free tier. Un shard supprimé ne coûte plus rien.

Usage :
    python -m infra.aws.kinesis_ingest up          # crée stream + firehose + lambda
    python -m infra.aws.kinesis_ingest run         # invoque le Lambda (rejeu)
    python -m infra.aws.kinesis_ingest status      # état des trois objets + objets S3
    python -m infra.aws.kinesis_ingest down        # supprime tout (dans le bon ordre)
"""

from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import boto3

from retail_forecast.config import SETTINGS

REGION = SETTINGS.region
STREAM = "retail-sales-stream"
DELIVERY = "retail-sales-to-s3"
FUNCTION = "retail-replay-to-kinesis"
HANDLER_FILE = Path(__file__).resolve().parents[1] / "lambda" / "replay_to_kinesis.py"


def _clients() -> dict:
    """Un client par service (boto3 n'a pas de client « multi-services »)."""
    return {
        "kinesis": boto3.client("kinesis", region_name=REGION),
        "firehose": boto3.client("firehose", region_name=REGION),
        "lambda": boto3.client("lambda", region_name=REGION),
        "s3": boto3.client("s3", region_name=REGION),
        "sts": boto3.client("sts", region_name=REGION),
    }


def _role_arn(c: dict) -> str:
    """Rôle d'exécution : celui du lab (``LabRole``), jamais un rôle créé."""
    if SETTINGS.role_arn:
        return SETTINGS.role_arn
    account = c["sts"].get_caller_identity()["Account"]
    return f"arn:aws:iam::{account}:role/LabRole"


def _stream_arn(c: dict) -> str:
    return c["kinesis"].describe_stream(StreamName=STREAM)["StreamDescription"][
        "StreamARN"
    ]


def create_stream(c: dict) -> None:
    """Crée le stream s'il n'existe pas, puis attend qu'il soit ACTIVE."""
    try:
        c["kinesis"].describe_stream(StreamName=STREAM)
        print(f"[kinesis] {STREAM} existe déjà")
    except c["kinesis"].exceptions.ResourceNotFoundException:
        c["kinesis"].create_stream(
            StreamName=STREAM,
            ShardCount=1,
            StreamModeDetails={"StreamMode": "PROVISIONED"},
        )
        print(f"[kinesis] création de {STREAM} (1 shard)")

    for _ in range(60):
        status = c["kinesis"].describe_stream(StreamName=STREAM)["StreamDescription"][
            "StreamStatus"
        ]
        if status == "ACTIVE":
            print(f"[kinesis] {STREAM} ACTIVE")
            return
        time.sleep(2)
    raise RuntimeError(f"{STREAM} n'est pas passé ACTIVE à temps")



def create_delivery_stream(c: dict, role: str) -> None:
    """Firehose : source = le stream, destination = S3 partitionné par date."""
    try:
        state = c["firehose"].describe_delivery_stream(DeliveryStreamName=DELIVERY)[
            "DeliveryStreamDescription"
        ]["DeliveryStreamStatus"]
        if state == "ACTIVE":
            print(f"[firehose] {DELIVERY} existe déjà (ACTIVE)")
            return
        raise RuntimeError(f"{DELIVERY} existe en état {state} : lancer 'down' d'abord")
    except c["firehose"].exceptions.ResourceNotFoundException:
        pass

    bucket = SETTINGS.bucket
    c["firehose"].create_delivery_stream(
        DeliveryStreamName=DELIVERY,
        DeliveryStreamType="KinesisStreamAsSource",
        KinesisStreamSourceConfiguration={
            "KinesisStreamARN": _stream_arn(c),
            "RoleARN": role,
        },
        ExtendedS3DestinationConfiguration={
            "BucketARN": f"arn:aws:s3:::{bucket}",
            "RoleARN": role,
            # Partitionnement dynamique : un « dossier » par jour de livraison,
            # même convention que la clé brute produite localement auparavant.
            "Prefix": "raw/sales/ingest_date=!{timestamp:yyyy-MM-dd}/",
            "ErrorOutputPrefix": "raw/_errors/!{firehose:error-output-type}/",
            "BufferingHints": {"SizeInMBs": 5, "IntervalInSeconds": 60},
            "CompressionFormat": "GZIP",
            "DynamicPartitioningConfiguration": {"Enabled": True},
        },
    )
    print(f"[firehose] création de {DELIVERY} -> s3://{bucket}/raw/sales/ingest_date=…/")

    for _ in range(60):
        state = c["firehose"].describe_delivery_stream(DeliveryStreamName=DELIVERY)[
            "DeliveryStreamDescription"
        ]["DeliveryStreamStatus"]
        if state == "ACTIVE":
            print(f"[firehose] {DELIVERY} ACTIVE")
            return
        time.sleep(3)
    raise RuntimeError(f"{DELIVERY} n'est pas passé ACTIVE à temps")


def deploy_lambda(c: dict, role: str, bucket: str) -> None:
    """Empaquette le handler et crée (ou met à jour) la fonction."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(HANDLER_FILE, arcname="replay_to_kinesis.py")
    code = buf.getvalue()

    env = {
        "Variables": {
            "BUCKET": bucket,
            "STREAM_NAME": STREAM,
            "REPLAY_PREFIX": "replay/sales",
            "CHECKPOINT_KEY": "replay/_state/checkpoint.json",
        }
    }

    try:
        c["lambda"].get_function(FunctionName=FUNCTION)
        c["lambda"].update_function_code(FunctionName=FUNCTION, ZipFile=code)
        c["lambda"].get_waiter("function_updated_v2").wait(FunctionName=FUNCTION)
        c["lambda"].update_function_configuration(
            FunctionName=FUNCTION,
            Runtime="python3.12",
            Handler="replay_to_kinesis.lambda_handler",
            Role=role,
            Timeout=900,
            MemorySize=512,
            Environment=env,
        )
        c["lambda"].get_waiter("function_updated_v2").wait(FunctionName=FUNCTION)
        print(f"[lambda] {FUNCTION} mise à jour ({len(code) / 1024:.1f} Ko)")
    except c["lambda"].exceptions.ResourceNotFoundException:
        c["lambda"].create_function(
            FunctionName=FUNCTION,
            Runtime="python3.12",
            Role=role,
            Handler="replay_to_kinesis.lambda_handler",
            Code={"ZipFile": code},
            Timeout=900,
            MemorySize=512,
            Environment=env,
            Description="Rejoue le dataset de ventes dans Kinesis (ingestion AWS)",
        )
        c["lambda"].get_waiter("function_active_v2").wait(FunctionName=FUNCTION)
        print(f"[lambda] {FUNCTION} créée ({len(code) / 1024:.1f} Ko)")


# --------------------------------------------------------------------------- #
# exécution / inspection / suppression
# --------------------------------------------------------------------------- #
def run(c: dict, max_seconds: int) -> dict:
    """Invoque le Lambda et affiche la fin de ses logs."""
    import base64

    resp = c["lambda"].invoke(
        FunctionName=FUNCTION,
        Payload=json.dumps({"max_seconds": max_seconds}).encode("utf-8"),
        LogType="Tail",
    )
    payload = json.loads(resp["Payload"].read() or b"{}")
    if "FunctionError" in resp:
        raise RuntimeError(f"Lambda en erreur : {payload}")
    if resp.get("LogResult"):
        print(base64.b64decode(resp["LogResult"]).decode("utf-8", "replace").strip())
    return payload


def status(c: dict) -> None:
    """État des trois objets + nombre d'objets livrés dans ``raw/sales/``."""
    try:
        desc = c["kinesis"].describe_stream(StreamName=STREAM)["StreamDescription"]
        print(f"kinesis  : {STREAM} {desc['StreamStatus']} · {len(desc['Shards'])} shard(s)")
    except Exception as exc:  # noqa: BLE001 - affichage seulement
        print(f"kinesis  : absent ({type(exc).__name__})")

    try:
        state = c["firehose"].describe_delivery_stream(DeliveryStreamName=DELIVERY)[
            "DeliveryStreamDescription"
        ]["DeliveryStreamStatus"]
        print(f"firehose : {DELIVERY} {state}")
    except Exception as exc:  # noqa: BLE001
        print(f"firehose : absent ({type(exc).__name__})")

    try:
        state = c["lambda"].get_function(FunctionName=FUNCTION)["Configuration"]["State"]
        print(f"lambda   : {FUNCTION} {state}")
    except Exception as exc:  # noqa: BLE001
        print(f"lambda   : absent ({type(exc).__name__})")

    paginator = c["s3"].get_paginator("list_objects_v2")
    count = size = 0
    try:
        for page in paginator.paginate(Bucket=SETTINGS.bucket, Prefix="raw/sales/"):
            for obj in page.get("Contents", []):
                count += 1
                size += obj["Size"]
        print(f"raw/sales/: {count} objet(s) · {size / 1024:.1f} Ko")
    except Exception as exc:  # noqa: BLE001
        print(f"raw/sales/: illisible ({str(exc).splitlines()[0][:60]})")


def down(c: dict) -> None:
    """Supprime Firehose, puis le Lambda, puis le stream (ordre imposé par AWS)."""
    try:
        c["firehose"].delete_delivery_stream(
            DeliveryStreamName=DELIVERY, AllowForceDelete=True
        )
        print(f"[firehose] {DELIVERY} supprimé")
    except c["firehose"].exceptions.ResourceNotFoundException:
        print(f"[firehose] {DELIVERY} déjà absent")

    try:
        c["lambda"].delete_function(FunctionName=FUNCTION)
        print(f"[lambda] {FUNCTION} supprimée")
    except c["lambda"].exceptions.ResourceNotFoundException:
        print(f"[lambda] {FUNCTION} déjà absente")

    try:
        c["kinesis"].delete_stream(StreamName=STREAM, EnforceConsumerDeletion=True)
        print(f"[kinesis] {STREAM} supprimé")
    except c["kinesis"].exceptions.ResourceNotFoundException:
        print(f"[kinesis] {STREAM} déjà absent")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["up", "run", "status", "down"], help="action à exécuter"
    )
    parser.add_argument(
        "--max-seconds", type=int, default=600, help="budget d'exécution du Lambda"
    )
    args = parser.parse_args(argv)

    c = _clients()
    if not SETTINGS.bucket:
        raise SystemExit("S3_BUCKET vide dans .env : lancer d'abord 'make aws-setup'")

    if args.action == "up":
        role = _role_arn(c)
        print(f"[init] bucket={SETTINGS.bucket} region={REGION} role={role}")
        create_stream(c)
        create_delivery_stream(c, role)
        deploy_lambda(c, role, SETTINGS.bucket)
    elif args.action == "run":
        print(json.dumps(run(c, args.max_seconds), indent=2))
    elif args.action == "status":
        status(c)
    else:
        down(c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
