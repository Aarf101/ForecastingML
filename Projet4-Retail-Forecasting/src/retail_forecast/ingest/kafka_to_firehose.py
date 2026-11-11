"""Consumer Kafka -> Firehose (DirectPut) -> S3 ``raw/sales/``.

C'est la chaîne de la diapositive d'architecture : **Kafka transporte, Firehose
livre**. Firehose ne sait pas consommer un Kafka générique — ses sources sont
DirectPut, Kinesis, MSK et Splunk. MSK étant refusé par ce compte
(``kafka:ListClustersV2`` -> AccessDenied), ce pont fait le ``PutRecordBatch``
et Firehose assure le reste : tamponnage 60 s / 5 Mo, compression gzip,
rangement sous ``raw/sales/AAAA/MM/JJ/HH/`` et préfixe d'erreur.

    python -m retail_forecast.ingest.kafka_to_firehose --max-messages 20000

Différence avec ``to_s3.py`` : ce module n'écrit **aucun** objet S3 lui-même.
Le tamponnage est délégué au service, ce qui est le comportement du schéma
(et ce que ``to_s3.py`` imitait en local).

Sémantique : *at-least-once*. Le commit Kafka n'a lieu qu'après l'accusé de
Firehose, donc un crash peut dupliquer un lot — le job Glue déduplique sur
``(store_id, item_id, date)``.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from retail_forecast.config import SETTINGS
from retail_forecast.ingest.storage import encode_line

# Limites dures de l'API PutRecordBatch.
MAX_RECORDS = 500
MAX_REQUEST_BYTES = 4 * 1024 * 1024
MAX_RECORD_BYTES = 1000 * 1024


@dataclass
class Stats:
    """Compteurs d'une session de pont."""

    messages: int = 0
    sent: int = 0
    requests: int = 0
    retries: int = 0


def to_records(values: list) -> list[dict]:
    """Convertit des messages Kafka en ``{"Data": bytes}`` pour Firehose.

    Chaque enregistrement se termine par ``\\n`` (JSON Lines) : Firehose
    concatène sans délimiteur, donc l'objet S3 serait illisible autrement.
    """
    out = []
    for value in values:
        line = encode_line(value) if isinstance(value, dict) else bytes(value)
        if not line.endswith(b"\n"):
            line += b"\n"
        if len(line) > MAX_RECORD_BYTES:
            raise ValueError(
                f"enregistrement de {len(line):,} octets > limite Firehose "
                f"({MAX_RECORD_BYTES:,})"
            )
        out.append({"Data": line})
    return out


def put_batch(firehose, stream: str, values: list) -> tuple[int, list, str | None]:
    """Envoie un lot ; renvoie (nb_acceptés, enregistrements à réessayer, erreur)."""
    payload = to_records(values)
    if len(payload) > MAX_RECORDS:
        raise ValueError(f"{len(payload)} enregistrements > limite Firehose ({MAX_RECORDS})")
    total = sum(len(r["Data"]) for r in payload)
    if total > MAX_REQUEST_BYTES:
        raise ValueError(f"lot de {total:,} octets > limite ({MAX_REQUEST_BYTES:,})")

    resp = firehose.put_record_batch(DeliveryStreamName=stream, Records=payload)

    failed_count = int(resp.get("FailedPutCount", 0) or 0)
    if failed_count <= 0:
        return len(payload), [], None

    # Firehose renvoie un statut par enregistrement, dans l'ordre d'envoi.
    responses = resp.get("RequestResponses") or []
    failed, error = [], None
    for i, value in enumerate(values):
        status = responses[i] if i < len(responses) else {}
        if status.get("ErrorCode"):
            failed.append(value)
            error = error or status.get("ErrorMessage") or status.get("ErrorCode")
    if not failed:  # FailedPutCount incohérent : on rejoue tout, prudemment
        failed = list(values)
    return len(payload) - len(failed), failed, error


def run(
    consumer,
    firehose,
    stream: str,
    batch_size: int = MAX_RECORDS,
    flush_seconds: float = 1.0,
    max_messages: int | None = None,
    max_retries: int = 5,
) -> Stats:
    """Boucle de pont : accumule, pousse vers Firehose, commit après accusé."""
    stats = Stats()
    buffer: list = []
    last_flush = time.monotonic()

    def flush() -> None:
        nonlocal buffer, last_flush
        if not buffer:
            return
        pending, buffer = buffer, []

        for attempt in range(1, max_retries + 1):
            ok, failed, error = put_batch(firehose, stream, pending)
            stats.requests += 1
            stats.sent += ok
            if not failed:
                break
            if attempt == max_retries:
                raise RuntimeError(
                    f"{len(failed)} enregistrement(s) refusé(s) par Firehose "
                    f"après {max_retries} tentatives : {error}"
                )
            stats.retries += 1
            time.sleep(min(0.2 * 2**attempt, 5.0))
            pending = failed

        # Le commit vient APRÈS l'accusé Firehose : au pire un doublon, jamais une
        # perte. Glue déduplique.
        if hasattr(consumer, "commit"):
            consumer.commit()
        last_flush = time.monotonic()

    try:
        for msg in consumer:
            buffer.append(msg.value)
            stats.messages += 1
            if (
                len(buffer) >= batch_size
                or (time.monotonic() - last_flush) >= flush_seconds
                or (max_messages and stats.messages >= max_messages)
            ):
                flush()
            if max_messages and stats.messages >= max_messages:
                break
    finally:
        flush()

    return stats


def build_consumer(bootstrap: str, topic: str, group: str):
    """Consumer Kafka, désérialisation JSON, commit manuel."""
    from kafka import KafkaConsumer

    from retail_forecast.ingest.to_s3 import JsonValueDeserializer

    return KafkaConsumer(
        topic,
        bootstrap_servers=[b.strip() for b in bootstrap.split(",") if b.strip()],
        group_id=group,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        value_deserializer=JsonValueDeserializer(),
        consumer_timeout_ms=15000,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pont Kafka -> Firehose (DirectPut) -> S3 raw/sales/"
    )
    parser.add_argument("--bootstrap", default=SETTINGS.kafka_bootstrap)
    parser.add_argument("--topic", default=SETTINGS.kafka_topic)
    parser.add_argument("--group", default="retail-kafka-to-firehose")
    parser.add_argument(
        "--stream",
        default="retail-sales-direct",
        help="delivery stream Firehose (voir infra.aws.firehose_direct)",
    )
    parser.add_argument("--batch-size", type=int, default=MAX_RECORDS)
    parser.add_argument(
        "--flush-seconds",
        type=float,
        default=1.0,
        help="fréquence d'envoi ; Firehose tamponne ensuite 60 s",
    )
    parser.add_argument("--max-messages", type=int, default=None)
    parser.add_argument("--region", default=SETTINGS.region)
    args = parser.parse_args(argv)

    import boto3

    firehose = boto3.client("firehose", region_name=args.region)
    consumer = build_consumer(args.bootstrap, args.topic, args.group)

    print(
        f"[bridge] {args.topic} ({args.bootstrap}) -> Firehose {args.stream} "
        f"-> s3://{SETTINGS.bucket}/raw/sales/"
    )
    try:
        stats = run(
            consumer,
            firehose,
            args.stream,
            batch_size=args.batch_size,
            flush_seconds=args.flush_seconds,
            max_messages=args.max_messages,
        )
    finally:
        consumer.close()

    print(
        f"[bridge] terminé : {stats.sent:,}/{stats.messages:,} enregistrements "
        f"acceptés en {stats.requests} requête(s) "
        f"({stats.retries} reprise(s)) — Firehose livre sous ~60 s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
