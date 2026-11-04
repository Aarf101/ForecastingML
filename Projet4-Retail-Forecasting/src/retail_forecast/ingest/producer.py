"""Producteur Kafka : rejoue l'historique de ventes dans ``sales.events``.

Simule le flux d'un système de caisse (POS) : chaque ligne du Parquet devient un
événement JSON. Le même code fonctionne contre Redpanda (local) et MSK (AWS) —
seul ``KAFKA_BOOTSTRAP`` change.

    python -m retail_forecast.ingest.producer --limit 5000 --rate 1000
    python -m retail_forecast.ingest.producer --loop          # flux continu (démo)
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from retail_forecast.config import SETTINGS

RECORD_FIELDS = (
    "date",
    "store_id",
    "item_id",
    "category",
    "unit_price",
    "promo",
    "is_holiday",
    "lead_time_days",
    "stock_on_hand",
    "stockout",
    "units",
)
INT_FIELDS = {"promo", "is_holiday", "lead_time_days", "stock_on_hand", "stockout", "units"}
FLOAT_FIELDS = {"unit_price"}


def record_from_row(row: dict, event_time: str | None = None) -> dict:
    """Transforme une ligne (dict) en événement JSON typé, sans NaN."""
    rec: dict = {}
    for field in RECORD_FIELDS:
        value = row.get(field)
        missing = value is None or (not isinstance(value, str) and pd.isna(value))
        if field == "date":
            rec[field] = None if missing else pd.Timestamp(value).strftime("%Y-%m-%d")
        elif field in INT_FIELDS:
            rec[field] = 0 if missing else int(value)
        elif field in FLOAT_FIELDS:
            rec[field] = None if missing else round(float(value), 2)
        else:
            rec[field] = None if missing else str(value)
    rec["event_time"] = event_time or pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    return rec


def encode(record: dict) -> bytes:
    """Sérialisation utilisée comme ``value_serializer`` Kafka."""
    return json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def decode(raw: bytes) -> dict:
    """Désérialisation utilisée comme ``value_deserializer`` Kafka."""
    return json.loads(raw.decode("utf-8"))


def records_from_frame(df: pd.DataFrame, limit: int | None = None) -> Iterator[dict]:
    """Itère les événements d'un DataFrame (``limit`` lignes max)."""
    rows = df if limit is None else df.head(limit)
    for row in rows.to_dict(orient="records"):
        yield record_from_row(row)


try:  # kafka-python >= 3 : protocole explicite (évite un DeprecationWarning)
    from kafka.serializer import Serializer as _Serializer
except ImportError:  # pragma: no cover - kafka-python 2.x
    _Serializer = object  # type: ignore[assignment,misc]


class JsonValueSerializer(_Serializer):  # type: ignore[misc]
    """Sérialiseur JSON compatible kafka-python 2.x (appel) et 3.x (protocole)."""

    def serialize(self, topic, headers, data):  # kafka-python >= 3
        return encode(data)

    def __call__(self, data):  # kafka-python 2.x
        return encode(data)

    def close(self) -> None:
        return None


def build_producer(bootstrap: str, client_id: str = "retail-producer"):
    """Instancie un KafkaProducer (import paresseux : les tests n'en ont pas besoin)."""
    from kafka import KafkaProducer

    return KafkaProducer(
        bootstrap_servers=[b.strip() for b in bootstrap.split(",")],
        client_id=client_id,
        value_serializer=JsonValueSerializer(),
        acks=1,
        linger_ms=50,
        retries=3,
    )


def produce(
    df: pd.DataFrame,
    topic: str,
    bootstrap: str,
    rate: float | None = None,
    limit: int | None = None,
    loop: bool = False,
    progress_every: int = 1000,
) -> int:
    """Envoie les événements ; renvoie le nombre de messages produits."""
    producer = build_producer(bootstrap)
    sent = 0
    started = time.monotonic()
    try:
        while True:
            for rec in records_from_frame(df, limit=limit):
                producer.send(topic, rec)
                sent += 1
                if rate:
                    time.sleep(1.0 / rate)
                if progress_every and sent % progress_every == 0:
                    elapsed = max(time.monotonic() - started, 1e-9)
                    print(f"[producer] {sent:,} messages  ({sent / elapsed:,.0f}/s)", flush=True)
            if not loop:
                break
    finally:
        producer.flush()
        producer.close()
    elapsed = max(time.monotonic() - started, 1e-9)
    print(f"[producer] terminé : {sent:,} messages vers '{topic}' en {elapsed:.1f}s")
    return sent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rejoue les ventes dans Kafka.")
    parser.add_argument("--data", default="data/raw/sales.parquet")
    parser.add_argument("--topic", default=SETTINGS.kafka_topic)
    parser.add_argument("--bootstrap", default=SETTINGS.kafka_bootstrap)
    parser.add_argument("--limit", type=int, default=None, help="nb max de lignes")
    parser.add_argument("--rate", type=float, default=None, help="messages/seconde")
    parser.add_argument("--loop", action="store_true", help="rejouer en boucle")
    args = parser.parse_args(argv)

    path = Path(args.data)
    if not path.exists():
        raise SystemExit(
            f"{path} introuvable — lancez : python -m retail_forecast.data.simulate"
        )
    df = pd.read_parquet(path) if path.suffix in {".parquet", ".pq"} else pd.read_csv(path)
    produce(
        df,
        topic=args.topic,
        bootstrap=args.bootstrap,
        rate=args.rate,
        limit=args.limit,
        loop=args.loop,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
