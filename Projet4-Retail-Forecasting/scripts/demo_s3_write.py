#!/usr/bin/env python3
"""Montre que le code d'ingestion écrit réellement dans le S3 du compte AWS.

Remplace Kafka par un faux consumer (3 messages) pour ne dépendre que de S3 :
c'est le même chemin de code que `to_s3.py`, jusqu'à `put_object`.

    python scripts/demo_s3_write.py                 # écrit puis vérifie
    python scripts/demo_s3_write.py --delete        # supprime l'objet de démo
    python scripts/demo_s3_write.py --objects 3     # trois lots au lieu d'un
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retail_forecast.config import SETTINGS  # noqa: E402
from retail_forecast.ingest.producer import record_from_row  # noqa: E402
from retail_forecast.ingest.storage import s3_client  # noqa: E402
from retail_forecast.ingest.to_s3 import S3Sink, run  # noqa: E402

ROW = {
    "date": "2024-12-30",
    "store_id": "S01",
    "item_id": "I001",
    "category": "grocery",
    "unit_price": 4.5,
    "promo": 0,
    "is_holiday": 0,
    "lead_time_days": 3,
    "stock_on_hand": 20,
    "stockout": 0,
    "units": 7,
}


class FakeMsg:
    def __init__(self, value):
        self.value = value


class FakeConsumer:
    """Tient lieu de Kafka : itère des messages et compte les commits."""

    def __init__(self, records):
        self._records = records
        self.commits = 0

    def __iter__(self):
        return iter(FakeMsg(r) for r in self._records)

    def commit(self):
        self.commits += 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Écrit un lot de démo dans S3.")
    parser.add_argument("--prefix", default="raw/_demo")
    parser.add_argument("--messages", type=int, default=3)
    parser.add_argument("--delete", action="store_true", help="supprime l'objet de démo")
    args = parser.parse_args()

    if not SETTINGS.bucket:
        raise SystemExit("S3_BUCKET non configuré (.env)")

    print(f"bucket   : {SETTINGS.bucket}")
    print(f"endpoint : {SETTINGS.s3_endpoint_url or 'AWS (defaut boto3)'}")
    print(f"profil   : {SETTINGS.profile or '(defaut)'} · region {SETTINGS.region}")

    client = s3_client()
    print(f"boto3 pointe sur : {client.meta.endpoint_url}\n")

    if args.delete:
        listing = client.list_objects_v2(Bucket=SETTINGS.bucket, Prefix=args.prefix)
        for obj in listing.get("Contents", []):
            client.delete_object(Bucket=SETTINGS.bucket, Key=obj["Key"])
            print(f"supprime : {obj['Key']}")
        return 0

    records = [record_from_row(ROW) for _ in range(args.messages)]
    consumer = FakeConsumer(records)
    stats = run(
        consumer,
        S3Sink(SETTINGS.bucket),
        prefix=args.prefix,
        batch_size=len(records),
        flush_seconds=3600,
    )
    print(f"\ncommits Kafka : {consumer.commits} (apres l'ecriture)")
    print(f"objets        : {stats.batches} · {stats.bytes} octets")

    for key in stats.keys:
        head = client.head_object(Bucket=SETTINGS.bucket, Key=key)
        print(f"verifie       : s3://{SETTINGS.bucket}/{key} ({head['ContentLength']} o)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
