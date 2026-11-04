"""Accès stockage objet compatible S3 (S3 local en local, S3 sur AWS).

Un seul point d'entrée : ``s3_client()``. Si ``S3_ENDPOINT_URL`` est renseigné
(S3 local), boto3 est pointé dessus ; sinon il utilise S3 et la chaîne de
credentials standard (``~/.aws/credentials`` du Learner Lab, rôle IAM, etc.).
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator
from typing import Any

from retail_forecast.config import SETTINGS, Settings


def s3_client(settings: Settings = SETTINGS):
    """Client boto3 S3 (S3 local ou AWS). Import paresseux pour rester testable."""
    import boto3

    kwargs: dict[str, Any] = {"region_name": settings.region}
    if settings.s3_endpoint_url:
        kwargs["endpoint_url"] = settings.s3_endpoint_url
    if settings.s3_access_key:
        kwargs["aws_access_key_id"] = settings.s3_access_key
        kwargs["aws_secret_access_key"] = settings.s3_secret_key
    return boto3.client("s3", **kwargs)


def ensure_bucket(client, bucket: str) -> None:
    """Crée le bucket s'il n'existe pas (S3 local local)."""
    try:
        client.head_bucket(Bucket=bucket)
    except Exception:  # botocore.exceptions.ClientError / NoCredentialsError
        client.create_bucket(Bucket=bucket)


def raw_key(
    ingest_date: str, batch_id: str, prefix: str = "raw/sales", ext: str = "json.gz"
) -> str:
    """Clé S3 partitionnée : ``raw/sales/ingest_date=YYYY-MM-DD/batch-...json.gz``."""
    return f"{prefix}/ingest_date={ingest_date}/{batch_id}.{ext}"


def to_gzip_jsonl(records: Iterable[dict]) -> bytes:
    """Sérialise des enregistrements en JSON Lines compressé (1 objet = 1 ligne)."""
    buf = bytearray()
    for rec in records:
        buf += json.dumps(rec, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        buf += b"\n"
    return gzip.compress(bytes(buf), compresslevel=6)


def chunks(items: Iterable, size: int) -> Iterator[list]:
    """Découpe un itérable en listes de ``size`` (dernier lot incomplet conservé)."""
    if size < 1:
        raise ValueError("size doit être >= 1")
    batch: list = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def put_batch(
    client, bucket: str, key: str, records: list[dict]
) -> tuple[int, int]:
    """Écrit un lot gzip dans S3 ; renvoie (nb_enregistrements, nb_octets)."""
    body = to_gzip_jsonl(records)
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/gzip",
        ContentEncoding="gzip",
    )
    return len(records), len(body)
