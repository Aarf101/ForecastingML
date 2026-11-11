"""Firehose DirectPut : la livraison Kafka -> S3 de la diapositive d'architecture.

Firehose n'accepte que quatre familles de sources : DirectPut, Kinesis, MSK et
Splunk. Un Kafka auto-hébergé n'en fait pas partie, et MSK est refusé par ce
compte (``kafka:ListClustersV2`` -> AccessDenied). Le pont
``retail_forecast.ingest.kafka_to_firehose`` pousse donc les lots en
**DirectPut** ; Firehose assure la livraison : tamponnage 60 s / 5 Mo,
compression gzip, rangement sous ``raw/sales/AAAA/MM/JJ/HH/`` et préfixe
d'erreur ``raw/_errors/``.

    python -m infra.aws.firehose_direct up
    python -m infra.aws.firehose_direct status
    python -m infra.aws.firehose_direct down
"""

from __future__ import annotations

import argparse
import time

from retail_forecast.config import SETTINGS

STREAM = "retail-sales-direct"
PREFIX = "raw/sales/"


def _client(region: str | None = None):
    import boto3

    return boto3.client("firehose", region_name=region or SETTINGS.region)


def _role() -> str:
    """Rôle d'exécution : SAGEMAKER_ROLE_ARN, sinon ``LabRole`` du compte courant.

    Le lab interdit ``iam:CreateRole`` : on réutilise donc le rôle fourni.
    """
    if SETTINGS.role_arn:
        return SETTINGS.role_arn
    import boto3

    account = boto3.client("sts", region_name=SETTINGS.region).get_caller_identity()[
        "Account"
    ]
    return f"arn:aws:iam::{account}:role/LabRole"


def _status(client, name: str = STREAM) -> str | None:
    """Statut du stream, ou ``None`` s'il n'existe pas."""
    try:
        desc = client.describe_delivery_stream(DeliveryStreamName=name)[
            "DeliveryStreamDescription"
        ]
        return desc["DeliveryStreamStatus"]
    except client.exceptions.ResourceNotFoundException:
        return None


def wait_active(client, name: str = STREAM, tries: int = 40) -> str:
    """Attend le passage à ACTIVE (création : ~30-60 s)."""
    for _ in range(tries):
        state = _status(client, name)
        if state == "ACTIVE":
            return state
        time.sleep(3)
    raise RuntimeError(f"{name} n'est pas passé ACTIVE à temps")


def ensure(client=None, name: str = STREAM, prefix: str = PREFIX) -> str:
    """Crée le delivery stream DirectPut s'il manque ; renvoie son statut."""
    client = client or _client()
    state = _status(client, name)
    if state == "ACTIVE":
        print(f"[firehose] {name} existe déjà (ACTIVE)")
        return state
    if state in ("CREATING", "UPDATING"):
        print(f"[firehose] {name} en cours ({state}) : attente…")
        return wait_active(client, name)
    if state is not None:
        raise RuntimeError(f"{name} existe en état {state} : lancer 'down' d'abord")

    bucket = SETTINGS.bucket
    if not bucket:
        raise SystemExit("S3_BUCKET vide : renseigner .env ou exporter la variable")

    client.create_delivery_stream(
        DeliveryStreamName=name,
        DeliveryStreamType="DirectPut",
        ExtendedS3DestinationConfiguration={
            "BucketARN": f"arn:aws:s3:::{bucket}",
            "RoleARN": _role(),
            "Prefix": prefix,
            "ErrorOutputPrefix": "raw/_errors/!{firehose:error-output-type}/",
            "BufferingHints": {"SizeInMBs": 5, "IntervalInSeconds": 60},
            "CompressionFormat": "GZIP",
        },
    )
    print(f"[firehose] création de {name} (DirectPut) -> s3://{bucket}/{prefix}")
    return wait_active(client, name)


def down(client=None, name: str = STREAM) -> None:
    """Supprime le delivery stream (les objets déjà livrés restent dans S3)."""
    client = client or _client()
    state = _status(client, name)
    if state is None:
        print(f"[firehose] {name} absent")
        return
    client.delete_delivery_stream(DeliveryStreamName=name)
    print(f"[firehose] suppression de {name} demandée (état {state})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Firehose DirectPut -> S3 raw/sales/")
    parser.add_argument("action", choices=["up", "status", "down"])
    parser.add_argument("--stream", default=STREAM)
    parser.add_argument("--region", default=SETTINGS.region)
    args = parser.parse_args(argv)

    client = _client(args.region)

    if args.action == "up":
        ensure(client, args.stream)
        print(
            "\nÉtape suivante, sur la même machine (le pont Kafka) :\n"
            f"  python -m retail_forecast.ingest.kafka_to_firehose "
            f"--stream {args.stream} --max-messages 20000"
        )
    elif args.action == "status":
        state = _status(client, args.stream)
        print(f"[firehose] {args.stream} : {state or 'absent'}")
        print(f"  destination : s3://{SETTINGS.bucket}/{PREFIX}")
        print("  tamponnage  : 5 Mo ou 60 s · compression GZIP")
    else:
        down(client, args.stream)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
