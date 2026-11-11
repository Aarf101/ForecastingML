"""Prépare les morceaux JSONL gzip que le Lambda rejouera dans Kinesis.

Une seule fois : le parquet historique (876k lignes) est découpé en morceaux de
``--rows`` événements, chacun compressé en JSON Lines et déposé sous
``s3://<bucket>/replay/sales/``. Le Lambda n'a ainsi ni pandas ni pyarrow à
charger — donc aucune couche (layer) à installer — et chaque invocation travaille
sur un objet de quelques mégaoctets.

Le format produit est **identique** à celui du consumer ``to_s3.py`` (une vente
par ligne, mêmes champs), ce qui permet au job Glue de lire indifféremment les
deux origines.

Usage :
    python -m scripts.prepare_replay_chunks --rows 20000
    python -m scripts.prepare_replay_chunks --rows 20000 --delete   # purge d'abord
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from retail_forecast.config import RAW_DIR, SETTINGS
from retail_forecast.ingest.producer import record_from_row
from retail_forecast.ingest.storage import s3_client, to_gzip_jsonl

PREFIX = "replay/sales"


def _clean_prefix(client, bucket: str) -> int:
    """Supprime les morceaux existants (évite les doublons au rejeu)."""
    paginator = client.get_paginator("list_objects_v2")
    removed = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=PREFIX):
        keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
        if keys:
            client.delete_objects(Bucket=bucket, Delete={"Objects": keys})
            removed += len(keys)
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20_000, help="lignes par morceau")
    parser.add_argument("--limit", type=int, default=0, help="0 = tout le dataset")
    parser.add_argument("--delete", action="store_true", help="purge le préfixe d'abord")
    args = parser.parse_args(argv)

    if not SETTINGS.bucket:
        raise SystemExit("S3_BUCKET vide dans .env : lancer d'abord 'make aws-setup'")

    frame = pd.read_parquet(Path(RAW_DIR) / "sales.parquet")
    if args.limit:
        frame = frame.head(args.limit)
    print(f"[replay] {len(frame):,} lignes à découper ({args.rows:,} par objet)")

    client = s3_client()
    if args.delete:
        print(f"[replay] purge : {_clean_prefix(client, SETTINGS.bucket)} objet(s) supprimé(s)")

    total = 0
    chunks = 0
    for start in range(0, len(frame), args.rows):
        chunk = frame.iloc[start:start + args.rows]
        records = [record_from_row(row) for row in chunk.to_dict("records")]
        body = to_gzip_jsonl(records)
        chunks += 1
        key = f"{PREFIX}/chunk-{chunks:05d}.json.gz"
        client.put_object(
            Bucket=SETTINGS.bucket,
            Key=key,
            Body=body,
            ContentType="application/gzip",
            ContentEncoding="gzip",
        )
        total += len(body)
        print(f"[replay] {key}  {len(records):,} événements  {len(body) / 1024:.1f} Ko")

    print(
        f"[replay] terminé : {chunks} objet(s) · {total / 1e6:.2f} Mo sous "
        f"s3://{SETTINGS.bucket}/{PREFIX}/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
