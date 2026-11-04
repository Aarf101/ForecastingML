"""Consumer Kafka -> S3 local : micro-lots gzip partitionnés par date d'ingestion.

Reproduit le comportement de **Firehose** (buffer + flush toutes les N secondes)
pour que le reste du pipeline (S3 raw -> Glue -> features) soit identique en
local et sur AWS.

    python -m retail_forecast.ingest.to_s3 --max-messages 5000 --batch-size 500
    python -m retail_forecast.ingest.to_s3 --local-dir data/raw   # sans Docker

Sémantique : *at-least-once*. Le commit Kafka n'a lieu qu'après l'écriture de
l'objet, donc un crash peut dupliquer un lot — le job Glue déduplique sur
(store_id, item_id, date).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from retail_forecast.config import SETTINGS
from retail_forecast.ingest.producer import decode
from retail_forecast.ingest.storage import (
    ensure_bucket,
    put_batch,
    raw_key,
    s3_client,
    to_gzip_jsonl,
)


@dataclass
class Stats:
    """Compteurs d'une session de consommation."""

    messages: int = 0
    batches: int = 0
    bytes: int = 0
    keys: list[str] = field(default_factory=list)


try:  # kafka-python >= 3 : protocole explicite (évite un DeprecationWarning)
    from kafka.serializer import Deserializer as _Deserializer
except ImportError:  # pragma: no cover - kafka-python 2.x
    _Deserializer = object  # type: ignore[assignment,misc]


class JsonValueDeserializer(_Deserializer):  # type: ignore[misc]
    """Désérialiseur JSON compatible kafka-python 2.x (appel) et 3.x (protocole)."""

    def deserialize(self, topic, headers, data):  # kafka-python >= 3
        return decode(data)

    def __call__(self, data):  # kafka-python 2.x
        return decode(data)

    def close(self) -> None:
        return None


class S3Sink:
    """Écrit les lots dans S3 (AWS) ou S3 local (endpoint custom)."""

    def __init__(self, bucket: str, client=None):
        self.bucket = bucket
        self.client = client or s3_client()
        ensure_bucket(self.client, bucket)

    def write(self, key: str, records: list[dict]) -> tuple[int, int]:
        return put_batch(self.client, self.bucket, key, records)


class LocalSink:
    """Écrit les mêmes objets sur disque — tests et démo sans Docker."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def write(self, key: str, records: list[dict]) -> tuple[int, int]:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        body = to_gzip_jsonl(records)
        path.write_bytes(body)
        return len(records), len(body)


def batch_key(now: datetime, seq: int, prefix: str = "raw/sales") -> tuple[str, str]:
    """Construit la partition + la clé S3 d'un lot."""
    ingest_date = now.strftime("%Y-%m-%d")
    batch_id = f"batch-{now.strftime('%Y%m%dT%H%M%S')}-{seq:04d}"
    return ingest_date, raw_key(ingest_date, batch_id, prefix=prefix)


def run(
    consumer,
    sink,
    prefix: str = "raw/sales",
    batch_size: int = 500,
    flush_seconds: float = 5.0,
    max_messages: int | None = None,
    log=print,
    now_fn=lambda: datetime.now(timezone.utc),
) -> Stats:
    """Consomme, agrège et écrit des lots ; renvoie les compteurs."""
    stats = Stats()
    buffer: list[dict] = []
    seq = 0
    last_flush = time.monotonic()

    def flush() -> None:
        nonlocal buffer, last_flush, seq
        if not buffer:
            return
        seq += 1
        _, key = batch_key(now_fn(), seq, prefix)
        count, nbytes = sink.write(key, buffer)
        stats.batches += 1
        stats.bytes += nbytes
        stats.keys.append(key)
        log(f"[consumer] {key}  {count} msgs  {nbytes / 1024:.1f} Ko")
        buffer = []
        last_flush = time.monotonic()
        if hasattr(consumer, "commit"):
            consumer.commit()

    try:
        for msg in consumer:
            buffer.append(msg.value)
            stats.messages += 1
            if (
                len(buffer) >= batch_size
                or (time.monotonic() - last_flush) >= flush_seconds
                or (max_messages is not None and stats.messages >= max_messages)
            ):
                flush()
            if max_messages is not None and stats.messages >= max_messages:
                break
    finally:
        flush()
    return stats


def build_consumer(
    bootstrap: str,
    topic: str,
    group_id: str = "retail-to-s3",
    idle_timeout_ms: int = 15000,
):
    """KafkaConsumer prêt à l'emploi (import paresseux)."""
    from kafka import KafkaConsumer

    return KafkaConsumer(
        topic,
        bootstrap_servers=[b.strip() for b in bootstrap.split(",")],
        group_id=group_id,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        value_deserializer=JsonValueDeserializer(),
        consumer_timeout_ms=idle_timeout_ms,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kafka -> S3 (micro-lots gzip).")
    parser.add_argument("--bootstrap", default=SETTINGS.kafka_bootstrap)
    parser.add_argument("--topic", default=SETTINGS.kafka_topic)
    parser.add_argument("--group", default="retail-to-s3")
    parser.add_argument("--bucket", default=SETTINGS.bucket or "retail-forecast-local")
    parser.add_argument("--prefix", default="raw/sales")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--flush-seconds", type=float, default=5.0)
    parser.add_argument("--max-messages", type=int, default=None)
    parser.add_argument("--local-dir", default=None, help="écrire sur disque au lieu de S3")
    args = parser.parse_args(argv)

    sink = LocalSink(args.local_dir) if args.local_dir else S3Sink(args.bucket)
    consumer = build_consumer(args.bootstrap, args.topic, args.group)
    try:
        stats = run(
            consumer,
            sink,
            prefix=args.prefix,
            batch_size=args.batch_size,
            flush_seconds=args.flush_seconds,
            max_messages=args.max_messages,
        )
    finally:
        consumer.close()

    target = args.local_dir or f"s3://{args.bucket}"
    print(
        f"[consumer] terminé : {stats.messages:,} msgs -> {stats.batches} objets "
        f"({stats.bytes / 1024:.1f} Ko) dans {target}/{args.prefix}/"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
